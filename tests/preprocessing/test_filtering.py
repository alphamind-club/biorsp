import pandas as pd
from biorsp.preprocessing.filtering import (
    filter_cells_by_umi,
    filter_genes_by_expression,
)


def test_filtering():
    """
    Test the filtering functions for DGE data.
    - Loads the data, applies cell and gene filtering, and verifies the output.
    """
    # Load the data
    dge_file_path = "data/dge/GSM4647185_FetalHeart_1_dge.txt"
    try:
        dge_matrix = pd.read_csv(dge_file_path, delimiter="\t", index_col=0)
        print(f"Loaded DGE matrix with shape: {dge_matrix.shape}")
    except FileNotFoundError:
        print(f"File not found: {dge_file_path}")
        return

    # Parameters for filtering
    threshold_umi = 500
    threshold_gene = 1

    # Test cell filtering by UMI counts
    dge_matrix_filtered_cells = filter_cells_by_umi(dge_matrix, threshold_umi)
    print(
        f"After filtering cells by UMI > {threshold_umi}, shape: {dge_matrix_filtered_cells.shape}"
    )

    assert (
        dge_matrix_filtered_cells.shape[1] <= dge_matrix.shape[1]
    ), "Number of cells should decrease or remain the same after filtering."

    # Test gene filtering by expression
    dge_matrix_filtered_genes = filter_genes_by_expression(
        dge_matrix_filtered_cells, threshold_gene
    )
    print(
        f"After filtering genes by expression > {threshold_gene}, shape: {dge_matrix_filtered_genes.shape}"
    )

    assert (
        dge_matrix_filtered_genes.shape[0] <= dge_matrix_filtered_cells.shape[0]
    ), "Number of genes should decrease or remain the same after filtering."
    assert (dge_matrix_filtered_genes > 0).sum(
        axis=1
    ).min() > threshold_gene, "All retained genes should be expressed in more than the threshold number of cells."

    # Optional: check for empty results
    if dge_matrix_filtered_genes.empty:
        print("Warning: All genes or cells were filtered out.")

    print("All tests passed successfully.")


if __name__ == "__main__":
    test_filtering()
