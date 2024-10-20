import os
import numpy as np
import pandas as pd

from biorsp.analysis.find_points import find_foreground_background_points
from biorsp.analysis.rsp_calculations import calculate_rsp_area, calculate_differences
from biorsp.analysis.polar_conversion import convert_to_polar
from biorsp.preprocessing.filtering import filter_dge_matrix
from biorsp.preprocessing.dimensionality_reduction import compute_tsne
from biorsp.preprocessing.clustering import compute_dbscan
from biorsp.visualization.embedding import plot_embedding
from biorsp.visualization.rsp import (
    plot_foreground_background,
    plot_rsp_polar,
    plot_rsp_comparison,
)


def test_analysis_workflow():
    """
    Test the complete workflow for RSP analysis.
    - Reads DGE data, computes t-SNE and DBSCAN, isolates a gene, and performs RSP analysis.
    - Verifies the functionality of each step and the creation of plots.
    """
    dge_file_path = "data/dge/GSM4647185_FetalHeart_1_dge.txt"
    try:
        dge_matrix = pd.read_csv(dge_file_path, delimiter="\t", index_col=0)
        print(f"Loaded DGE matrix with shape: {dge_matrix.shape}")
    except FileNotFoundError:
        print(f"File not found: {dge_file_path}")
        return

    gene_name = "Tnnt2"  # Replace with a gene of interest
    threshold = 1
    selected_clusters = [1]
    random_state = 42

    save_path_tsne = "test_tsne.png"
    save_path_foreground_background = "test_foreground_background.png"
    save_path_polar = "test_rsp_polar.png"
    save_path_comparison = "test_rsp_comparison.png"

    try:
        dge_matrix_filtered = filter_dge_matrix(
            dge_matrix, umi_threshold=500, gene_threshold=1
        )
        print(f"Filtered DGE matrix with shape: {dge_matrix_filtered.shape}")
    except ValueError as e:
        print(f"ValueError during DGE matrix filtering: {e}")
        return

    try:
        tsne_results = compute_tsne(
            dge_matrix_filtered,
            n_components=2,
            perplexity=30,
            random_state=random_state,
        )
        print(f"Computed t-SNE results with shape: {tsne_results.shape}")
    except ValueError as e:
        print(f"ValueError during t-SNE computation: {e}")
        return

    try:
        dbscan_labels = compute_dbscan(tsne_results, eps=4, min_samples=10)
        dbscan_results = pd.DataFrame(dbscan_labels, columns=["cluster"])
        unique_clusters = set(dbscan_labels)
        print(f"Computed DBSCAN results with unique clusters: {unique_clusters}")
    except ValueError as e:
        print(f"ValueError during DBSCAN computation: {e}")
        return

    try:
        plot_embedding(
            tsne_results,
            labels=dbscan_labels,
            method="t-SNE",
            save_path=save_path_tsne,
            show_plot=True,
        )
        assert os.path.exists(
            save_path_tsne
        ), f"t-SNE plot image was not saved at {save_path_polar}."
        print(f"t-SNE plot image successfully saved at {save_path_polar}.")
    except ValueError as e:
        print(f"ValueError during t-SNE plot: {e}")
        return
    finally:
        if os.path.exists(save_path_tsne):
            try:
                os.remove(save_path_tsne)
                print(f"Removed t-SNE plot image at {save_path_tsne}.")
            except OSError as e:
                print(f"Error removing t-SNE plot image at {save_path_tsne}: {e}")

    try:
        foreground_points, background_points = find_foreground_background_points(
            gene_name=gene_name,
            dge_matrix=dge_matrix_filtered,
            tsne_results=tsne_results,
            threshold=threshold,
            dbscan_df=dbscan_results,
            selected_clusters=selected_clusters,
        )
        print(f"Foreground points shape: {foreground_points.shape}")
        print(f"Background points shape: {background_points.shape}")
    except ValueError as e:
        print(f"ValueError during foreground/background point calculation: {e}")
        return

    try:
        plot_foreground_background(
            foreground_points=foreground_points,
            background_points=background_points,
            save_path=save_path_foreground_background,
            show_plot=True,
        )
    except ValueError as e:
        print(f"ValueError during foreground/background point plotting: {e}")
        return
    finally:
        if os.path.exists(save_path_foreground_background):
            try:
                os.remove(save_path_foreground_background)
                print(
                    f"Removed foreground/background point plot image at {save_path_foreground_background}."
                )
            except OSError as e:
                print(
                    f"Error removing foreground/background point plot image at {save_path_foreground_background}: {e}"
                )

    try:
        vantage_point = np.mean(background_points, axis=0)
        _, fg_theta = convert_to_polar(foreground_points, vantage_point)
        _, bg_theta = convert_to_polar(background_points, vantage_point)
        print(f"Converted foreground points to polar coordinates: {fg_theta.shape}")
        print(f"Converted background points to polar coordinates: {bg_theta.shape}")
    except ValueError as e:
        print(f"ValueError during conversion to polar coordinates: {e}")
        return

    try:
        differences = calculate_differences(
            foreground_points=foreground_points,
            background_points=background_points,
            scanning_window=np.pi / 2,
            resolution=1000,
            vantage_point=vantage_point,
            angle_range=[0, 2 * np.pi],
            mode="absolute",
        )
        rsp_area = calculate_rsp_area(
            differences=differences, angle_range=[0, 2 * np.pi], resolution=1000
        )
        print(f"Calculated RSP area: {rsp_area}")
    except ValueError as e:
        print(f"ValueError during RSP area calculation: {e}")
        return

    try:
        plot_rsp_polar(differences, save_path=save_path_polar, show_plot=True)
        assert os.path.exists(
            save_path_polar
        ), f"Polar plot not saved at {save_path_polar}."
        print(f"Polar plot saved at {save_path_polar}.")
    except ValueError as e:
        print(f"ValueError during polar plot: {e}")
        return
    except OSError as e:
        print(f"OSError during saving polar plot: {e}")
        return
    finally:
        if os.path.exists(save_path_polar):
            try:
                os.remove(save_path_polar)
                print(f"Cleaned up: removed {save_path_polar}.")
            except OSError as e:
                print(f"OSError during cleanup of {save_path_polar}: {e}")

    try:
        plot_rsp_comparison(
            rsp_area=rsp_area,
            differences=differences,
            save_path=save_path_comparison,
            show_plot=True,
        )
        assert os.path.exists(
            save_path_comparison
        ), f"Comparison plot not saved at {save_path_comparison}."
        print(f"Comparison plot saved at {save_path_comparison}.")
    except ValueError as e:
        print(f"ValueError during comparison plot: {e}")
        return
    except OSError as e:
        print(f"OSError during saving comparison plot: {e}")
        return
    finally:
        if os.path.exists(save_path_comparison):
            try:
                os.remove(save_path_comparison)
                print(f"Cleaned up: removed {save_path_comparison}.")
            except OSError as e:
                print(f"OSError during cleanup of {save_path_comparison}: {e}")

    print("All RSP analysis workflow tests passed successfully.")


if __name__ == "__main__":
    test_analysis_workflow()
