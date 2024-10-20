import numpy as np


def find_foreground_background_points(
    gene_name, dge_matrix, tsne_results, dbscan_df, threshold=1, selected_clusters=None
):
    """
    Find foreground and background points based on gene expression levels in bioRSP.

    Parameters:
    - gene_name: The gene of interest.
    - dge_matrix: DataFrame containing gene expression data (rows=genes, columns=cells).
    - tsne_results: 2D numpy array with t-SNE (or UMAP) coordinates for each cell.
    - dbscan_df: DataFrame with DBSCAN cluster labels for each cell.
    - threshold: Expression level threshold for foreground points (default=1).
    - selected_clusters: List of cluster labels to focus on (optional).

    Returns:
    - foreground_points: Numpy array of (x, y) coordinates for cells with expression above the threshold.
    - background_points: Numpy array of (x, y) coordinates for all cells within selected clusters.
    """
    dbscan_clusters = dbscan_df["cluster"].values

    if len(dbscan_clusters) != tsne_results.shape[0]:
        raise ValueError(
            "DBSCAN cluster labels do not match the number of t-SNE results."
        )

    if gene_name not in dge_matrix.index:
        raise ValueError(f"Gene '{gene_name}' not found in the dataset.")

    gene_expression = dge_matrix.loc[gene_name]
    cell_barcodes = dge_matrix.columns

    # Filter cells by selected clusters if specified
    if selected_clusters is not None:
        selected_indices = [
            idx
            for idx, cluster in enumerate(dbscan_clusters)
            if cluster in selected_clusters
        ]
        # Filter the t-SNE results and gene expression data based on selected clusters
        tsne_results = tsne_results[selected_indices]
        selected_barcodes = [cell_barcodes[idx] for idx in selected_indices]
        gene_expression = gene_expression[selected_barcodes]
        cell_index_map = {barcode: idx for idx, barcode in enumerate(selected_barcodes)}
    else:
        selected_indices = range(len(dbscan_clusters))
        cell_index_map = {barcode: idx for idx, barcode in enumerate(cell_barcodes)}

    # Foreground points: Cells with gene expression above the threshold
    foreground_indices = gene_expression[gene_expression > threshold].index
    foreground_points = np.array(
        [tsne_results[cell_index_map[barcode]] for barcode in foreground_indices]
    )

    # Background points are all cells within the selected clusters
    background_points = tsne_results

    return foreground_points, np.array(background_points)
