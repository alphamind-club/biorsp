import pandas as pd
from sklearn.cluster import DBSCAN


def run_dbscan(tsne_results, eps=4, min_samples=50, save_path=None):
    """
    Run DBSCAN on the t-SNE results.

    Parameters:
    - tsne_results: A 2D numpy array with the t-SNE (or UMAP) coordinates for each cell.
    - eps: The epsilon parameter for DBSCAN.
    - min_samples: The minimum number of samples for DBSCAN.
    - save_path: Path to save the DBSCAN results.

    Returns:
    - dbscan_labels: A 1D numpy array with the DBSCAN cluster labels for each cell.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(tsne_results)
    dbscan_labels += 1

    if save_path:
        dbscan_results_df = pd.DataFrame(dbscan_labels, columns=["Cluster"])
        dbscan_results_df.to_csv(save_path, index=False)

    return dbscan_labels
