import numpy as np
import pandas as pd
import scanpy as sc

import biorsp
from biorsp.baselines import run_baselines
from examples.synthetic_data import create_synthetic_dataset

ALPHA = 0.05


def main():
    rng = np.random.default_rng(42)
    adata = create_synthetic_dataset(
        n_cells=2000, manifold="circle", n_genes=50, rng=rng
    )
    sc.pp.log1p(adata)

    vantage = biorsp.define_reference_point(adata, method="geometric_median")

    genes_to_scan = ["Gene_0", "Gene_5", "Gene_10"]

    results = biorsp.find_spatially_patterned_genes(
        adata,
        genes_to_test=genes_to_scan,
        reference_point=vantage,
        num_permutations=500,
        confounding_factors=["batch", "clusters"],
        neighbors_for_matching=30,
        check_spatial_distortion=False,
        random_seed=42,
        allow_uncalibrated_analysis=True,
    )

    adata.uns["biorsp"]["params"] = {
        "vantage_point": vantage,
        "embedding_key": "X_umap",
    }

    target_gene = "Gene_0"

    from biorsp.interpretation import (
        generate_interpretation_report,
        identify_peak_sectors,
    )

    identify_peak_sectors(adata, target_gene, threshold_percentile=75)

    generate_interpretation_report(
        adata,
        [target_gene],
        obs_keys=["clusters", "batch"],
    )

    baselines_df = run_baselines(adata, genes_to_scan)

    pd.concat(
        [results[["ARIA", "p_CRA"]], baselines_df],
        axis=1,
    )

    from biorsp.stability import test_consistency_across_embeddings

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=2)

    test_consistency_across_embeddings(
        adata,
        genes_to_test=[target_gene],
        coordinate_systems=["X_umap", "X_pca"],
        reference_method="geometric_median",
        analysis_settings={
            "allow_uncalibrated_analysis": True,
            "check_spatial_distortion": False,
            "confounding_factors": ["batch", "clusters"],
        },
    )

    adata_null = adata.copy()
    adata_null.X = rng.permutation(adata.X)

    results_null = biorsp.find_spatially_patterned_genes(
        adata_null,
        genes_to_test=genes_to_scan,
        reference_point=vantage,
        num_permutations=200,
        confounding_factors=["batch"],
        check_spatial_distortion=False,
        random_seed=123,
        allow_uncalibrated_analysis=True,
    )

    (results_null["p_CRA"] < ALPHA).mean()

    results[results["p_CRA"] < ALPHA].sort_values("ARIA", ascending=False)


if __name__ == "__main__":
    main()
