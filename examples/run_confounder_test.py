import biorsp
from examples.synthetic_data import create_synthetic_dataset


def run_confounder_test():
    adata = create_synthetic_dataset(n_cells=1000, manifold="blobs", n_genes=20)
    vantage = biorsp.define_reference_point(adata, method="geometric_median")

    target_gene = "Gene_5"

    res_global = biorsp.find_spatially_patterned_genes(
        adata,
        genes_to_test=[target_gene],
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        stratification_column=None,
        allow_uncalibrated_analysis=True,
        reference_point_stability_tests=1,
        permutation_method="knn",
    )
    p_global = res_global.loc[target_gene, "p_CRA"]

    res_strat = biorsp.find_spatially_patterned_genes(
        adata,
        genes_to_test=[target_gene],
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        stratification_column="clusters",
        allow_uncalibrated_analysis=True,
        reference_point_stability_tests=1,
    )
    p_strat = res_strat.loc[target_gene, "p_CRA"]

    print(f"Global p-value: {p_global}")
    print(f"Stratified p-value: {p_strat}")


if __name__ == "__main__":
    run_confounder_test()
