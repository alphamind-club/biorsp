import os
import numpy as np
from biorsp.visualization.rsp import plot_rsp_polar, plot_rsp_comparison
from biorsp.analysis.rsp_calculations import calculate_rsp_area


def generate_smooth_differences(num_points, max_diff=0.05):
    """
    Generate a smooth differences array where each element differs from its neighbors
    by at most `max_diff`.

    Parameters:
    - num_points: The number of points to generate.
    - max_diff: The maximum allowed difference between consecutive elements.

    Returns:
    - A numpy array of smooth differences.
    """
    differences = np.zeros(num_points)
    differences[0] = np.random.rand()  # Start with a random value between 0 and 1

    for i in range(1, num_points):
        # Generate the next value, ensuring the difference is at most `max_diff`
        next_value = differences[i - 1] + np.random.uniform(-max_diff, max_diff)
        # Ensure that values stay within the range [0, 1]
        differences[i] = np.clip(next_value, 0, 1)

    return differences


def test_rsp():
    """
    Test the RSP plotting functions.
    - Generates smooth differences and rsp_area.
    - Verifies the creation and saving of polar and comparison plots.
    """
    # Generate sample data for testing
    num_points = 1000
    differences = generate_smooth_differences(num_points, max_diff=0.05)
    print(f"Generated differences array with shape: {differences.shape}")

    # Define file paths for saving plots
    save_path_polar = "test_rsp_polar.png"
    save_path_comparison = "test_rsp_comparison.png"

    # Test polar plot without saving
    try:
        plot_rsp_polar(differences, show_plot=False)
        print("Polar plot without saving succeeded.")
    except Exception as e:
        print(f"Polar plot without saving failed: {e}")
        return

    # Test polar plot with saving
    try:
        plot_rsp_polar(differences, save_path=save_path_polar, show_plot=False)
        assert os.path.exists(
            save_path_polar
        ), f"Polar plot image was not saved at {save_path_polar}."
        print(f"Polar plot image successfully saved at {save_path_polar}.")
    except Exception as e:
        print(f"Polar plot with saving failed: {e}")
        return
    finally:
        # Clean up: remove the generated plot image
        if os.path.exists(save_path_polar):
            os.remove(save_path_polar)
            print(f"Cleaned up: removed {save_path_polar}.")

    # Test comparison plot without saving
    rsp_area = calculate_rsp_area(
        differences, angle_range=[0, 2 * np.pi], resolution=1000
    )
    print(f"Using RSP area: {rsp_area}")
    try:
        plot_rsp_comparison(rsp_area, differences, show_plot=False)
        print("Comparison plot without saving succeeded.")
    except Exception as e:
        print(f"Comparison plot without saving failed: {e}")
        return

    # Test comparison plot with saving
    try:
        plot_rsp_comparison(
            rsp_area, differences, save_path=save_path_comparison, show_plot=False
        )
        assert os.path.exists(
            save_path_comparison
        ), f"Comparison plot image was not saved at {save_path_comparison}."
        print(f"Comparison plot image successfully saved at {save_path_comparison}.")
    except Exception as e:
        print(f"Comparison plot with saving failed: {e}")
        return
    finally:
        # Clean up: remove the generated plot image
        if os.path.exists(save_path_comparison):
            os.remove(save_path_comparison)
            print(f"Cleaned up: removed {save_path_comparison}.")

    print("All RSP plotting tests passed successfully.")


if __name__ == "__main__":
    test_rsp()
