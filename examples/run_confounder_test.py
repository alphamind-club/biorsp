import numpy as np
import pandas as pd
import scanpy as sc
from biorsp.api import define_reference_point, find_spatially_patterned_genes
from synthetic_data import create_synthetic_dataset, add_gene_expression


def run_confounder_test():
    print("Running Confounder Test: Batch Effects")

    n_cells = 1000
    n_genes = 50
    rng = np.random.default_rng(42)

    adata = create_synthetic_dataset(
        n_cells=n_cells,
        manifold="circle",
        n_genes=n_genes,
        rng=rng,
    )

    coords = adata.obsm["X_umap"]
    center = np.mean(coords, axis=0)
    radii = np.linalg.norm(coords - center, axis=1)
    median_r = np.median(radii)

    batch_labels = np.where(radii < median_r, "B1", "B2")
    adata.obs["batch"] = batch_labels

    print("Generating 50 batch-confounded genes...")
    for i in range(n_genes):
        add_gene_expression(
            adata,
            adata.obsm["X_umap"],
            adata.obs["latent_t"],
            gene_type="batch_confounded",
            gene_name=f"Gene_{i}",
            batch_labels=batch_labels,
            effect_size=2.0,
            rng=rng,
        )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    vantage = define_reference_point(
        adata, coordinate_system="X_umap", method="geometric_median"
    )

    alpha = 0.05
    print("\nRunning Unstratified Analysis (Naive)...")
    adata.obs["dummy_global"] = "all"

    res_unstratified = find_spatially_patterned_genes(
        adata,
        genes_to_test=list(adata.var_names),
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="stratified",
        stratification_column="dummy_global",
        confounding_factors=[],
        allow_uncalibrated_analysis=True,
        check_spatial_distortion=False,
    )

    type1_unstratified = np.mean(res_unstratified["p_D_dir"] < alpha)
    print(f"Unstratified Type I Error (FPR): {type1_unstratified:.3f}")

    print("\nRunning Stratified Analysis (Correct)...")
    res_stratified = find_spatially_patterned_genes(
        adata,
        genes_to_test=list(adata.var_names),
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="stratified",
        stratification_column="batch",
        confounding_factors=["batch"],
        allow_uncalibrated_analysis=True,
        check_spatial_distortion=False,
    )

    type1_stratified = np.mean(res_stratified["p_D_dir"] < alpha)
    print(f"Stratified Type I Error (FPR): {type1_stratified:.3f}")

    print("\n" + "=" * 60)
    print("CONFOUNDER TEST RESULTS")
    print("=" * 60)
    print(f"Target Alpha: {alpha}")
    print(f"Unstratified FPR: {type1_unstratified:.3f} (Expected: High/Fail)")
    print(f"Stratified FPR:   {type1_stratified:.3f} (Expected: ~0.05/Pass)")

    if type1_unstratified > 0.1 and type1_stratified <= 0.1:
        print("\nSUCCESS: Stratification restored Type I error control.")
    else:
        print(
            "\nWARNING: Results do not clearly demonstrate the benefit of stratification."
        )


if __name__ == "__main__":
    run_confounder_test()
