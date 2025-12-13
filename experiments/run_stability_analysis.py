import numpy as np
import biorsp
from experiments.synthetic_data import create_synthetic_dataset


def run_stability_analysis():
    print("Running Stability Analysis...")

    adata = create_synthetic_dataset(n_cells=1000, manifold="circle")

    coords = adata.obsm["X_umap"]
    theta = np.radians(45)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    adata.obsm["X_pca"] = coords @ rot.T

    genes = ["Gene_0", "Gene_1", "Gene_10"]  # Directional, Directional, Random

    print(f"Checking stability across UMAP and PCA for genes: {genes}")



    biorsp.check_embedding_stability(
        adata,
        genes,
        embeddings=["X_umap", "X_pca"],
        vantage_mode="geometric_median",
        n_perm=0,
    )

    print("\nStability Analysis Complete.")


if __name__ == "__main__":
    run_stability_analysis()
