import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP


def run_tsne(dge_matrix_filtered, random_state=42, save_path=None):
    """
    Run t-SNE on the filtered data.

    Parameters:
    - dge_matrix_filtered: A dataframe containing the filtered data.
    - random_state: The random state for reproducibility.
    - save_path: Path to save the t-SNE results.

    Returns:
    - tsne_results: A 2D numpy array with the t-SNE (or UMAP) coordinates for each cell.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=1000)
    tsne_results = tsne.fit_transform(dge_matrix_filtered.T)

    if save_path:
        tsne_results_df = pd.DataFrame(tsne_results, columns=["x", "y"])
        tsne_results_df.to_csv(save_path, index=False)

    return tsne_results


def run_umap(dge_matrix_filtered, random_state=42, save_path=None):
    """
    Run UMAP on the filtered data.

    Parameters:
    - dge_matrix_filtered: A dataframe containing the filtered data.
    - random_state: The random state for reproducibility.
    - save_path: Path to save the UMAP results.

    Returns:
    - umap_results: A 2D numpy array with the UMAP coordinates for each cell.
    """
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    umap_results = umap_reducer.fit_transform(dge_matrix_filtered.T)

    if save_path:
        umap_results_df = pd.DataFrame(umap_results, columns=["x", "y"])
        umap_results_df.to_csv(save_path, index=False)

    return umap_results
