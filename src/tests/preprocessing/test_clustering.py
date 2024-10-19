import pandas as pd
import numpy as np
from biorsp.preprocessing.clustering import compute_dbscan


def test_clustering():
    """
    Test the clustering functions for DGE data.
    - Loads the data and t-SNE results.
    - Applies DBSCAN clustering.
    - Verifies the shape of the results and checks for NaNs.
    - Checks that clusters contain valid labels.
    """
    dge_file_path = "data/dge/GSM4647185_FetalHeart_1_dge.txt"
    try:
        dge_matrix = pd.read_csv(dge_file_path, delimiter="\t", index_col=0)
        print(f"Loaded DGE matrix with shape: {dge_matrix.shape}")
    except FileNotFoundError:
        print(f"File not found: {dge_file_path}")
        return

    tsne_file_path = "data/tsne_results.csv"
    try:
        tsne_results = pd.read_csv(tsne_file_path).to_numpy()
        print(f"Loaded t-SNE results with shape: {tsne_results.shape}")
    except FileNotFoundError:
        print(f"File not found: {tsne_file_path}")
        return

    dbscan_eps = 0.5
    dbscan_min_samples = 10

    dbscan_labels = compute_dbscan(
        tsne_results, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    print(f"DBSCAN labels shape: {dbscan_labels.shape}")

    assert dbscan_labels.shape == (
        tsne_results.shape[0],
    ), "DBSCAN labels should have shape (num_samples,)."
    assert not np.isnan(dbscan_labels).any(), "DBSCAN labels contain NaN values."
    assert -1 in dbscan_labels, "DBSCAN should label some points as noise (-1)."

    print("All clustering tests passed successfully.")


if __name__ == "__main__":
    test_clustering()
