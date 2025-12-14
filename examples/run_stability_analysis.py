import numpy as np

from examples.synthetic_data import create_synthetic_dataset


def run_stability_analysis():
    adata = create_synthetic_dataset(n_cells=1000, manifold="circle")

    coords = adata.obsm["X_umap"]
    theta = np.radians(45)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    adata.obsm["X_pca"] = coords @ rot.T

    genes = ["Gene_0", "Gene_1", "Gene_10"]

    from biorsp.stability import test_consistency_across_embeddings

    test_consistency_across_embeddings(
        adata,
        genes_to_test=genes,
        coordinate_systems=["X_umap", "X_pca"],
        reference_method="geometric_median",
        analysis_settings={"allow_uncalibrated_analysis": True, "check_spatial_distortion": False, "confounding_factors": ["batch", "clusters"]},
    )


if __name__ == "__main__":
    run_stability_analysis()
