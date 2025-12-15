import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from biorsp.api import define_reference_point, find_spatially_patterned_genes
from biorsp.baselines import compute_morans_i
from synthetic_data import add_gene_expression, create_synthetic_dataset


def compute_morans_i_pvalues(adata, genes):
    """Compute Moran's I statistics and convert to p-values using permutation test."""
    morans = compute_morans_i(adata, genes=genes)

    n_perms = 100
    n_cells = adata.n_obs
    coords = adata.obsm["X_umap"]
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    pvals = {}
    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = x[:, idx]
        observed_i = morans[gene]

        null_stats = []
        for _ in range(n_perms):
            perm_expr = np.random.permutation(expr)
            expr_centered = perm_expr - perm_expr.mean()

            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=10).fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            w = np.zeros((n_cells, n_cells))
            for i in range(n_cells):
                for j in range(1, len(indices[i])):
                    neighbor = indices[i, j]
                    w[i, neighbor] = 1.0 / (distances[i, j] + 1e-9)

            w = w / (w.sum(axis=1, keepdims=True) + 1e-9)

            null_i = (
                n_cells
                * (expr_centered @ w @ expr_centered)
                / (w.sum() * (expr_centered**2).sum())
            )
            null_stats.append(null_i)

        null_stats = np.array(null_stats)
        pval = np.mean(np.abs(null_stats) >= np.abs(observed_i))
        pvals[gene] = pval

    return pd.Series(pvals)


def run_simulation_suite():
    results = []
    rng = np.random.default_rng(42)
    alpha = 0.05
    n_cells = 500
    n_genes = 50

    print("Running Scenario 1: Directional Gradient")
    adata = create_synthetic_dataset(
        n_cells=n_cells, manifold="circle", n_genes=n_genes, rng=rng
    )

    adata.obs["dummy_strata"] = "all"

    vantage = define_reference_point(
        adata, coordinate_system="X_umap", method="geometric_median"
    )
    res_df = find_spatially_patterned_genes(
        adata,
        genes_to_test=list(adata.var_names),
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="stratified",
        stratification_column="dummy_strata",
        include_log_depth=False,
        expression_method="log_normalized",
        confounding_factors=[],
        allow_uncalibrated_analysis=True,
    )

    power_biorsp = np.mean(
        res_df.loc[[f"Gene_{i}" for i in range(5)], "p_D_dir"] < alpha
    )

    null_genes = [f"Gene_{i}" for i in range(10, 50)]
    type1_biorsp = np.mean(res_df.loc[null_genes, "p_D_dir"] < alpha)

    results.append(
        {
            "Scenario": "Directional",
            "Method": "BioRSP",
            "Power": power_biorsp,
            "TypeI": type1_biorsp,
        }
    )

    print("Computing Moran's I for Scenario 1...")
    morans_pvals = compute_morans_i_pvalues(adata, genes=list(adata.var_names))

    power_morans = np.mean(morans_pvals.loc[[f"Gene_{i}" for i in range(5)]] < alpha)
    type1_morans = np.mean(morans_pvals.loc[null_genes] < alpha)

    results.append(
        {
            "Scenario": "Directional",
            "Method": "Moran's I",
            "Power": power_morans,
            "TypeI": type1_morans,
        }
    )

    print("Running Scenario 2: No-signal Null")
    adata_null = create_synthetic_dataset(
        n_cells=n_cells, manifold="circle", n_genes=n_genes, rng=rng
    )
    covariates = rng.normal(0, 1, (500, 2))
    adata_null.obsm["covariates"] = covariates

    for i in range(50):
        add_gene_expression(
            adata_null,
            adata_null.obsm["X_umap"],
            adata_null.obs["latent_t"],
            gene_type="no_signal",
            gene_name=f"Gene_{i}",
            covariates=covariates,
            rng=rng,
        )

    vantage_null = define_reference_point(
        adata_null, coordinate_system="X_umap", method="geometric_median"
    )
    res_df_null = find_spatially_patterned_genes(
        adata_null,
        genes_to_test=list(adata_null.var_names),
        reference_point=vantage_null,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="knn",
        confounding_factors=[],
        allow_uncalibrated_analysis=True,
    )
    type1_null = np.mean(res_df_null["p_D_dir"] < alpha)

    results.append(
        {
            "Scenario": "No-Signal",
            "Method": "BioRSP",
            "Power": np.nan,
            "TypeI": type1_null,
        }
    )

    print("Computing Moran's I for Scenario 2...")
    morans_pvals_null = compute_morans_i_pvalues(
        adata_null, genes=list(adata_null.var_names)
    )
    type1_morans_null = np.mean(morans_pvals_null < alpha)

    results.append(
        {
            "Scenario": "No-Signal",
            "Method": "Moran's I",
            "Power": np.nan,
            "TypeI": type1_morans_null,
        }
    )

    print("Running Scenario 3: Batch Confounding")
    adata_batch = create_synthetic_dataset(
        n_cells=n_cells,
        manifold="circle",
        n_genes=n_genes,
        rng=rng,
    )
    batch_labels = rng.choice(["B1", "B2"], 500)
    adata_batch.obs["batch"] = batch_labels

    for i in range(5):
        add_gene_expression(
            adata_batch,
            adata_batch.obsm["X_umap"],
            adata_batch.obs["latent_t"],
            gene_type="batch_confounded",
            gene_name=f"Gene_{i}",
            batch_labels=batch_labels,
            rng=rng,
        )

    vantage_batch = define_reference_point(
        adata_batch, coordinate_system="X_umap", method="geometric_median"
    )
    res_df_batch = find_spatially_patterned_genes(
        adata_batch,
        genes_to_test=list(adata_batch.var_names),
        reference_point=vantage_batch,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="stratified",
        stratification_column="batch",
        confounding_factors=["batch"],
        allow_uncalibrated_analysis=True,
    )

    type1_batch = np.mean(
        res_df_batch.loc[[f"Gene_{i}" for i in range(5)], "p_D_dir"] < alpha
    )
    results.append(
        {
            "Scenario": "Batch Confounding",
            "Method": "BioRSP (Stratified)",
            "Power": np.nan,
            "TypeI": type1_batch,
        }
    )

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON: BioRSP vs Moran's I")
    print("=" * 80)
    print("\nDetailed Results Table:")
    print(df_results.to_string(index=False))

    pivot_power = df_results[df_results["Power"].notna()].pivot_table(
        index="Scenario", columns="Method", values="Power", aggfunc="first"
    )
    pivot_type1 = df_results.pivot_table(
        index="Scenario", columns="Method", values="TypeI", aggfunc="first"
    )

    print("\n" + "-" * 80)
    print("POWER Comparison (higher is better):")
    print("-" * 80)
    print(pivot_power.to_string())

    print("\n" + "-" * 80)
    print("TYPE I ERROR RATE Comparison (target: α=0.05):")
    print("-" * 80)
    print(pivot_type1.to_string())

    print("\n" + "-" * 80)
    print("SUMMARY:")
    print("-" * 80)
    dir_biorsp = df_results[
        (df_results["Scenario"] == "Directional") & (df_results["Method"] == "BioRSP")
    ].iloc[0]
    dir_morans = df_results[
        (df_results["Scenario"] == "Directional")
        & (df_results["Method"] == "Moran's I")
    ].iloc[0]
    null_biorsp = df_results[
        (df_results["Scenario"] == "No-Signal") & (df_results["Method"] == "BioRSP")
    ].iloc[0]
    null_morans = df_results[
        (df_results["Scenario"] == "No-Signal") & (df_results["Method"] == "Moran's I")
    ].iloc[0]

    print(f"\nDirectional Gradient (Signal Detection):")
    print(
        f"  BioRSP  Power={dir_biorsp['Power']:.3f}, Type I={dir_biorsp['TypeI']:.3f}"
    )
    print(
        f"  Moran's I Power={dir_morans['Power']:.3f}, Type I={dir_morans['TypeI']:.3f}"
    )
    print(
        f"  Power Advantage (BioRSP): {dir_biorsp['Power'] - dir_morans['Power']:.3f}"
    )
    print(
        f"  Type I Control (BioRSP vs Target): {abs(dir_biorsp['TypeI'] - 0.05):.3f} vs {abs(dir_morans['TypeI'] - 0.05):.3f}"
    )

    print(f"\nNo-Signal Null (Type I Error):")
    print(f"  BioRSP  Type I={null_biorsp['TypeI']:.3f}")
    print(f"  Moran's I Type I={null_morans['TypeI']:.3f}")
    print(
        f"  Type I Inflation (BioRSP vs Target): {abs(null_biorsp['TypeI'] - 0.05):.3f}"
    )
    print(
        f"  Type I Inflation (Moran's I vs Target): {abs(null_morans['TypeI'] - 0.05):.3f}"
    )
    print("=" * 80 + "\n")

    df_results.to_csv("simulation_results.csv", index=False)


if __name__ == "__main__":
    run_simulation_suite()
