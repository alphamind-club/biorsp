import numpy as np


def find_foreground_background_points(
    gene_name, dge_matrix, tsne_results, dbscan_df, threshold=1, selected_clusters=None
):
    """
    Function to find foreground and background points based on gene expression levels in BioRSP,
    with DBSCAN results integrated to allow for cluster-based analysis. Both foreground and
    background points are filtered by clusters if specified.

    Parameters:
    - gene_name: The gene of interest.
    - dge_matrix: A dataframe containing the gene expression data (rows = genes, columns = cells).
    - tsne_results: A 2D numpy array with the t-SNE (or UMAP) coordinates for each cell.
    - dbscan_df: DataFrame with DBSCAN cluster labels for each cell.
    - threshold: The expression level threshold for the foreground points. Default is 1.
    - selected_clusters: A list of cluster labels to focus on, or None to include all cells.

    Returns:
    - foreground_points: A numpy array of (x, y) coordinates for cells with gene expression
    above the threshold.
    - background_points: A numpy array of (x, y) coordinates for all cells (or cells
    in selected clusters).
    """

    dbscan_clusters = dbscan_df["Cluster"].values

    if len(dbscan_clusters) != tsne_results.shape[0]:
        raise ValueError(
            "DBSCAN cluster labels do not match the number of t-SNE results."
        )

    if gene_name not in dge_matrix.index:
        raise ValueError(f"Gene {gene_name} not found in the dataset.")

    gene_expression = dge_matrix.loc[gene_name]
    cell_barcodes = dge_matrix.columns
    cell_index_map = {barcode: idx for idx, barcode in enumerate(cell_barcodes)}

    # Foreground points: cells where gene expression is above the threshold
    foreground_indices = gene_expression[gene_expression > threshold].index
    foreground_points = [
        tsne_results[cell_index_map[barcode]] for barcode in foreground_indices
    ]
    foreground_points = np.array(foreground_points)

    if selected_clusters is not None:
        foreground_indices_filtered = [
            i
            for i in foreground_indices
            if dbscan_clusters[cell_index_map[i]] in selected_clusters
        ]
        foreground_points = np.array(
            [
                tsne_results[cell_index_map[barcode]]
                for barcode in foreground_indices_filtered
            ]
        )

        # Filter cells that are part of the selected clusters for background points
        selected_indices = [
            idx
            for idx, cluster in enumerate(dbscan_clusters)
            if cluster in selected_clusters
        ]
        background_points = tsne_results[selected_indices]
    else:
        background_points = tsne_results

    return foreground_points, np.array(background_points)
