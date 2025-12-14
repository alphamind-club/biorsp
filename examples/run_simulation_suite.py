import numpy as np
import pandas as pd

from biorsp.api import define_reference_point, find_spatially_patterned_genes
from examples.synthetic_data import add_gene_expression, create_synthetic_dataset


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

    vantage = define_reference_point(
        adata, coordinate_system="X_umap", method="geometric_median"
    )
    res_df = find_spatially_patterned_genes(
        adata,
        genes_to_test=list(adata.var_names),
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="knn",
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
    print(df_results)
    df_results.to_csv("simulation_results.csv")


if __name__ == "__main__":
    run_simulation_suite()
