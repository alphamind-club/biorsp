import numpy as np
from biorsp.analysis.polar_conversion import convert_to_polar, in_scanning_range
from biorsp.analysis.histogram import compute_histogram, compute_cdf


def compute_cdfs(
    fg_projection, bg_projection, angle, scanning_window, resolution, mode
):
    """
    Compute the CDFs of the foreground and background projections.

    Parameters:
    - fg_projection: Numpy array of foreground angles in radians.
    - bg_projection: Numpy array of background angles in radians.
    - angle: Angle of the scanning window in radians.
    - scanning_window: Size of the scanning window in radians.
    - resolution: Number of bins for the histogram.
    - mode: Mode for scaling CDFs.

    Returns:
    - fg_cdf: Numpy array of foreground CDF.
    - bg_cdf: Numpy array of background CDF.
    """
    fg_histogram = compute_histogram(fg_projection, resolution, angle, scanning_window)
    bg_histogram = compute_histogram(bg_projection, resolution, angle, scanning_window)

    fg_cdf = compute_cdf(fg_histogram)
    bg_cdf = compute_cdf(bg_histogram)

    if mode == "absolute":
        bg_total = np.sum(bg_histogram)
        fg_total = np.sum(fg_histogram)
        if bg_total > 0:
            scaling_factor = fg_total / bg_total
            fg_cdf *= scaling_factor

    return fg_cdf, bg_cdf


def compute_area(fg_cdf, bg_cdf, window):
    """
    Compute the area under the absolute difference between foreground and background CDFs.

    Parameters:
    - fg_cdf: Numpy array of foreground CDF.
    - bg_cdf: Numpy array of background CDF.
    - window: Size of the scanning window in radians.

    Returns:
    - The area under the absolute difference between the CDFs.
    """
    dx = window / fg_cdf.shape[0]
    return np.trapz(np.abs(bg_cdf - fg_cdf), dx=dx)


def calculate_differences(
    foreground_points,
    background_points,
    scanning_window,
    resolution,
    vantage_point,
    angle_range,
    mode,
):
    """
    Calculate the differences between foreground and background CDFs.

    Parameters:
    - foreground_points: Numpy array of (x, y) coordinates for foreground cells.
    - background_points: Numpy array of (x, y) coordinates for background cells.
    - scanning_window: Scanning window size in radians.
    - resolution: Number of bins for the histogram.
    - vantage_point: 2D numpy array for the vantage point.
    - angle_range: Angular range for CDF computation.
    - mode: Mode for scaling the foreground and background CDFs.

    Returns:
    - differences: Numpy array of differences between the foreground and background CDFs.
    """
    _, fg_theta = convert_to_polar(foreground_points, vantage_point)
    _, bg_theta = convert_to_polar(background_points, vantage_point)

    differences = np.empty(resolution)
    angles = np.linspace(angle_range[0], angle_range[1], resolution, endpoint=False)

    for i, angle in enumerate(angles):
        fg_in_range = in_scanning_range(fg_theta, angle, scanning_window)
        bg_in_range = in_scanning_range(bg_theta, angle, scanning_window)

        fg_projection = fg_theta[fg_in_range]
        bg_projection = bg_theta[bg_in_range]

        fg_cdf, bg_cdf = compute_cdfs(
            fg_projection, bg_projection, angle, scanning_window, resolution, mode
        )
        differences[i] = compute_area(fg_cdf, bg_cdf, scanning_window)

    return differences
