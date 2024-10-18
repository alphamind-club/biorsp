import matplotlib.pyplot as plt


def plot_umap(umap_results, save_path=None, show_plot=False):
    """
    Plot the UMAP results.

    Parameters:
    - umap_results: A 2D numpy array with the UMAP coordinates for each cell.
    - save_path: Optional. Path to save the plot as an image file.
    - show_plot: If True, display the plot.
    """
    plt.scatter(umap_results[:, 0], umap_results[:, 1], s=1)
    plt.xlabel("x")
    plt.ylabel("y")

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.clf()  # Clear figure after saving or showing
