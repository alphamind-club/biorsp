import pandas as pd
import numpy as np
from biorsp.preprocessing.dimensionality_reduction import compute_tsne, run_umap


def test_dimensionality_reduction():
    """
    Test the dimensionality reduction functions for DGE data.
    - Loads the data and applies t-SNE and UMAP.
    - Verifies the shape of the results and checks for NaNs.
    """
    dge_file_path = "data/dge/GSM4647185_FetalHeart_1_dge.txt"
    try:
        dge_matrix = pd.read_csv(dge_file_path, delimiter="\t", index_col=0)
        print(f"Loaded DGE matrix with shape: {dge_matrix.shape}")
    except FileNotFoundError:
        print(f"File not found: {dge_file_path}")
        return

    random_state = 42
    tsne_perplexity = 30
    umap_neighbors = 15
    umap_min_dist = 0.1

    tsne_results = compute_tsne(
        dge_matrix, random_state=random_state, perplexity=tsne_perplexity
    )
    print(f"t-SNE results shape: {tsne_results.shape}")

    assert tsne_results.shape == (
        dge_matrix.shape[1],
        2,
    ), "t-SNE results should have shape (num_samples, 2)."
    assert not np.isnan(tsne_results).any(), "t-SNE results contain NaN values."

    umap_results = run_umap(
        dge_matrix,
        random_state=random_state,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
    )
    print(f"UMAP results shape: {umap_results.shape}")

    assert umap_results.shape == (
        dge_matrix.shape[1],
        2,
    ), "UMAP results should have shape (num_samples, 2)."
    assert not np.isnan(umap_results).any(), "UMAP results contain NaN values."

    print("All dimensionality reduction tests passed successfully.")


if __name__ == "__main__":
    test_dimensionality_reduction()
