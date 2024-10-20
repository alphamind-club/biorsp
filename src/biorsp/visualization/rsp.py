import numpy as np
import matplotlib.pyplot as plt


def plot_foreground_background(
    foreground_points,
    background_points,
    save_path=None,
    show_plot=True,
    foreground_color="red",
    background_color="grey",
    point_size=1,
    title=None,
    xlabel="x",
    ylabel="y",
):
    """
    Plot foreground and background points.

    Parameters:
    - foreground_points: Numpy array of (x, y) coordinates for the foreground points.
    - background_points: Numpy array of (x, y) coordinates for the background points.
    - save_path: Optional. Path to save the plot as an image file (e.g., 'plot.png').
    - show_plot: If True, display the plot. Default is True.
    - foreground_color: Color for foreground points (default='red').
    - background_color: Color for background points (default='grey').
    - point_size: Size of the points in the scatter plot (default=1).
    - title: Optional. Title of the plot.
    - xlabel: Label for the x-axis (default='x').
    - ylabel: Label for the y-axis (default='y').

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        background_points[:, 0],
        background_points[:, 1],
        color=background_color,
        s=point_size,
        label="Background",
        alpha=0.5,
    )
    plt.scatter(
        foreground_points[:, 0],
        foreground_points[:, 1],
        color=foreground_color,
        s=point_size,
        label="Foreground",
        alpha=0.8,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")

    if show_plot:
        plt.show()

    plt.clf()


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

    plt.clf()


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

    plt.plot(np.linspace(0, 2 * np.pi, 1000), np.ones(1000) * radius, color="black")
    plt.plot(np.linspace(0, 2 * np.pi, 1000), differences, color="red")

    overlap = np.minimum(radius, differences)
    plt.fill(np.linspace(0, 2 * np.pi, 1000), overlap, color="gray", alpha=0.5)

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    plt.clf()
