import numpy as np
import matplotlib.pyplot as plt


def plot_rsp_polar(differences, save_path=None, show_plot=True):
    """
    Plot the RSP in polar coordinates.

    Parameters:
    - differences: The differences array calculated during RSP analysis.
    - save_path: Optional. If provided, saves the plot to the specified path.
    - show_plot: If True, displays the plot on screen.
    """
    plt.figure()
    plt.subplot(polar=True)
    plt.plot(np.linspace(0, 2 * np.pi, 1000), differences, color="red")
    plt.ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    plt.clf()  # Clear the figure after showing or saving


def plot_rsp_comparison(rsp_area, differences, save_path=None, show_plot=True):
    """
    Plot the RSP comparison between the uniform radius and the RSP differences.

    Parameters:
    - rsp_area: The RSP area calculated.
    - differences: The differences array calculated during RSP analysis.
    - save_path: Optional. If provided, saves the plot to the specified path.
    - show_plot: If True, displays the plot on screen.
    """
    radius = np.sqrt(rsp_area / np.pi)

    plt.figure()
    plt.subplot(polar=True)

    # Plot uniform radius
    plt.plot(np.linspace(0, 2 * np.pi, 1000), np.ones(1000) * radius, color="black")

    # Plot differences
    plt.plot(np.linspace(0, 2 * np.pi, 1000), differences, color="red")

    # Fill overlap area
    overlap = np.minimum(radius, differences)
    plt.fill(np.linspace(0, 2 * np.pi, 1000), overlap, color="gray", alpha=0.5)

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    plt.clf()  # Clear the figure after showing or saving
