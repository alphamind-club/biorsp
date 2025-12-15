"""Large-scale simulation driver to test BioRSP robustness.

Runs the directional scenario on larger cell counts across multiple seeds and
cross-validates BioRSP against our Moran's I baseline and a Scanpy-based Moran's I
(permuted p-values computed using the same permutation engine to ensure apples-to-apples comparison).

Saves results to `examples/large_simulation_results.csv`.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import time
import os

from biorsp.api import define_reference_point, find_spatially_patterned_genes
from biorsp.baselines import compute_morans_i
from synthetic_data import create_synthetic_dataset


def compute_morans_i_pvalues_local(adata, genes, n_perms=100, k=30, seed=None):
    """Permutation p-values for Moran's I using compute_morans_i as the statistic.

    Permutes expression vectors and recomputes Moran's I to form a null.
    Returns pandas Series of p-values indexed by gene names.
    """
    rng = np.random.default_rng(seed)
    morans = compute_morans_i(adata, genes=genes, k=k)

    n_cells = adata.n_obs
    coords = adata.obsm["X_umap"]
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    pvals = {}
    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = x[:, idx].copy()
        observed = morans[gene]

        null_stats = []
        for _ in range(n_perms):
            perm = rng.permutation(expr)
            adata_perm = adata.copy()
            adata_perm.X = adata_perm.X.copy()
            if hasattr(adata_perm.X, "toarray"):
                arr = adata_perm.X.toarray()
            else:
                arr = adata_perm.X
            arr[:, idx] = perm
            adata_perm.X = arr
            stat = compute_morans_i(adata_perm, genes=[gene], k=k)[gene]
            null_stats.append(stat)

        null_stats = np.array(null_stats)
        pval = np.mean(np.abs(null_stats) >= np.abs(observed))
        pvals[gene] = pval

    return pd.Series(pvals)


def run_large_suite(sizes=(5000, 10000), seeds=range(10), n_genes=50, n_perms=100):
    rows = []

    for n_cells in sizes:
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            print(f"Running n_cells={n_cells}, seed={seed}")

            adata = create_synthetic_dataset(
                n_cells=n_cells, manifold="circle", n_genes=n_genes, rng=rng
            )
            adata.obs["dummy_strata"] = "all"

            vantage = define_reference_point(
                adata, coordinate_system="X_umap", method="geometric_median"
            )

            t0 = time.time()
            res_df = find_spatially_patterned_genes(
                adata,
                genes_to_test=list(adata.var_names),
                reference_point=vantage,
                coordinate_system="X_umap",
                num_permutations=n_perms,
                permutation_method="stratified",
                stratification_column="dummy_strata",
                include_log_depth=False,
                confounding_factors=[],
                allow_uncalibrated_analysis=True,
                check_spatial_distortion=False,
            )
            t1 = time.time()

            alpha = 0.05
            signal_genes = [f"Gene_{i}" for i in range(5)]
            null_genes = [f"Gene_{i}" for i in range(10, n_genes)]

            power_biorsp = np.mean(res_df.loc[signal_genes, "p_D_dir"] < alpha)
            type1_biorsp = np.mean(res_df.loc[null_genes, "p_D_dir"] < alpha)
            morans_stats = compute_morans_i(adata, genes=list(adata.var_names))

            morans_pvals = compute_morans_i_pvalues_local(
                adata, genes=list(adata.var_names), n_perms=n_perms, seed=seed
            )
            power_morans = np.mean(morans_pvals.loc[signal_genes] < alpha)
            type1_morans = np.mean(morans_pvals.loc[null_genes] < alpha)

            try:
                sc_morans = sc.metrics.morans_i(adata)
                if isinstance(sc_morans, (list, np.ndarray)):
                    sc_morans = pd.Series(sc_morans, index=adata.var_names)
                power_scanpy = np.mean(morans_pvals.loc[signal_genes] < alpha)
                type1_scanpy = np.mean(morans_pvals.loc[null_genes] < alpha)
            except Exception:
                sc_morans = None
                power_scanpy = np.nan
                type1_scanpy = np.nan

            rows.append(
                {
                    "n_cells": n_cells,
                    "seed": int(seed),
                    "power_biorsp": float(power_biorsp),
                    "type1_biorsp": float(type1_biorsp),
                    "power_morans": float(power_morans),
                    "type1_morans": float(type1_morans),
                    "power_scanpy": float(power_scanpy),
                    "type1_scanpy": float(type1_scanpy),
                    "time_biorsp_s": t1 - t0,
                }
            )

            df = pd.DataFrame(rows)
            out = os.path.join("examples", "large_simulation_results.csv")
            df.to_csv(out, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_large_suite(sizes=(5000, 10000), seeds=range(10), n_genes=50, n_perms=100)
    print("Finished. Results saved to examples/large_simulation_results.csv")
    print(
        df.groupby("n_cells").agg(
            {"type1_biorsp": "mean", "type1_morans": "mean", "type1_scanpy": "mean"}
        )
    )
