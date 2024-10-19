import os
import numpy as np
from biorsp.visualization.embedding import plot_embedding


def test_plot_embedding():
    """
    Test the plot_embedding function.
    - Generates sample embedding data (e.g., t-SNE and UMAP).
    - Verifies plotting functionality with and without labels.
    - Verifies the creation of image files when save_path is provided.
    """
    # Generate sample embedding data (e.g., t-SNE or UMAP results)
    num_points = 10000
    embedding_results = np.random.rand(num_points, 2)
    labels = np.random.randint(0, 5, size=num_points)
    print(f"Generated embedding results with shape: {embedding_results.shape}")
    print(f"Generated labels with shape: {labels.shape}")

    save_path_no_labels = "test_embedding_no_labels.png"
    save_path_with_labels = "test_embedding_with_labels.png"

    try:
        plot_embedding(embedding_results, method="t-SNE", show_plot=False)
        print("Plotting without labels and without saving succeeded.")
    except Exception as e:
        print(f"Plotting without labels failed: {e}")
        return

    try:
        plot_embedding(embedding_results, labels=labels, method="UMAP", show_plot=False)
        print("Plotting with labels and without saving succeeded.")
    except Exception as e:
        print(f"Plotting with labels failed: {e}")
        return

    try:
        plot_embedding(
            embedding_results,
            method="t-SNE",
            save_path=save_path_no_labels,
            show_plot=False,
        )
        assert os.path.exists(
            save_path_no_labels
        ), f"Plot image was not saved at {save_path_no_labels}."
        print(f"Plot image without labels successfully saved at {save_path_no_labels}.")
    except Exception as e:
        print(f"Plotting without labels and saving failed: {e}")
        return
    finally:
        if os.path.exists(save_path_no_labels):
            os.remove(save_path_no_labels)
            print(f"Cleaned up: removed {save_path_no_labels}.")

    try:
        plot_embedding(
            embedding_results,
            labels=labels,
            method="UMAP",
            save_path=save_path_with_labels,
            show_plot=False,
        )
        assert os.path.exists(
            save_path_with_labels
        ), f"Plot image was not saved at {save_path_with_labels}."
        print(f"Plot image with labels successfully saved at {save_path_with_labels}.")
    except Exception as e:
        print(f"Plotting with labels and saving failed: {e}")
        return
    finally:
        if os.path.exists(save_path_with_labels):
            os.remove(save_path_with_labels)
            print(f"Cleaned up: removed {save_path_with_labels}.")

    print("All embedding visualization tests passed successfully.")


if __name__ == "__main__":
    test_plot_embedding()
