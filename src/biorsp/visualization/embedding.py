import matplotlib.pyplot as plt


def plot_embedding(
    embedding_results,
    labels=None,
    method="t-SNE",
    save_path=None,
    show_plot=False,
    point_size=1,
    colormap="tab20",
):
    """
    Plot embedding results (e.g., t-SNE, UMAP) with optional cluster labels.

    Parameters:
    - embedding_results: A 2D numpy array with the embedding coordinates for each point.
    - labels: Optional. A 1D numpy array with cluster labels for each point (e.g., from DBSCAN).
    - method: A string specifying the embedding method, used for plot titles ("t-SNE", "UMAP", etc.).
    - save_path: Optional. Path to save the plot as an image file.
    - show_plot: If True, display the plot.
    - point_size: Size of points in the scatter plot.
    - colormap: Colormap to use for the scatter plot (only applies when labels are provided).
    """
    if labels is not None:
        plt.scatter(
            embedding_results[:, 0],
            embedding_results[:, 1],
            c=labels,
            s=point_size,
            cmap=colormap,
        )
    else:
        plt.scatter(embedding_results[:, 0], embedding_results[:, 1], s=point_size)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method} Embedding" + (" with Clusters" if labels is not None else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.clf()  # Clear figure after saving or showing to free up memory.
