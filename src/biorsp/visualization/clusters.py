import matplotlib.pyplot as plt


def plot_clusters(tsne_results, dbscan_labels, save_path=None, show_plot=False):
    """
    Plot the clusters.

    Parameters:
    - tsne_results: A 2D numpy array with the t-SNE (or UMAP) coordinates for each cell.
    - dbscan_labels: A 1D numpy array with the DBSCAN cluster labels for each cell.
    - save_path: Optional. Path to save the plot as an image file.
    - show_plot: If True, display the plot.
    """
    plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=dbscan_labels, s=1, cmap="tab20"
    )
    plt.xlabel("x")
    plt.ylabel("y")

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.clf()  # Clear figure after saving or showing
