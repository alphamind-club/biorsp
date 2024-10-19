def filter_cells_by_umi(dge_matrix, threshold_umi, plot=False, save_path=None):
    """
    Filter cells by UMI count.

    Parameters:
    - dge_matrix: A dataframe containing the gene expression data (rows = genes, columns = cells).
    - threshold_umi: The minimum UMI count threshold.
    - plot: If True, plot the UMI count histogram.
    - save_path: Path to save the filtered data.

    Returns:
    - dge_matrix_filtered: A dataframe containing the filtered cells.
    """
    umi_counts_per_cell = dge_matrix.sum(axis=0)

    if plot:
        import matplotlib.pyplot as plt

        plt.hist(umi_counts_per_cell, bins=50)
        plt.xlabel("UMI counts")
        plt.ylabel("Number of cells")
        plt.show()

    filtered_cells = umi_counts_per_cell[umi_counts_per_cell > threshold_umi].index
    dge_matrix_filtered = dge_matrix[filtered_cells]

    if save_path:
        dge_matrix_filtered.to_csv(save_path, sep="\t")

    return dge_matrix_filtered


def filter_genes_by_expression(dge_matrix_filtered, threshold_gene, save_path=None):
    gene_counts_per_cell = (dge_matrix_filtered > 0).sum(axis=1)
    filtered_genes = gene_counts_per_cell[gene_counts_per_cell > threshold_gene].index
    dge_matrix_filtered = dge_matrix_filtered.loc[filtered_genes]

    if save_path:
        dge_matrix_filtered.to_csv(save_path, sep="\t")

    return dge_matrix_filtered


def filter_dge_matrix(dge_matrix, umi_threshold, gene_threshold, save_path=None):
    """
    Filter the DGE matrix by UMI and gene thresholds.

    Parameters:
    - dge_matrix: A dataframe containing the gene expression data (rows = genes, columns = cells).
    - umi_threshold: The minimum UMI count threshold.
    - gene_threshold: The minimum gene expression count threshold.
    - save_path: Path to save the filtered data.

    Returns:
    - dge_matrix_filtered: A dataframe containing the filtered cells and genes.
    """
    dge_matrix_filtered = filter_cells_by_umi(
        dge_matrix, umi_threshold, plot=True, save_path=save_path
    )
    dge_matrix_filtered = filter_genes_by_expression(
        dge_matrix_filtered, gene_threshold, save_path=save_path
    )

    return dge_matrix_filtered
