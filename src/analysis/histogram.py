import numpy as np


def compute_histogram(projection, resolution, angle, window):
    """
    Compute the histogram of a projection.

    Parameters:
    - projection: A numpy array of angles in radians.
    - resolution: The number of bins in the histogram.
    - angle: The angle of the scanning window in radians.
    - window: The scanning window size in radians.

    Returns:
    - A numpy array of the histogram.
    """
    start_angle = (angle - window / 2) % (2 * np.pi)
    end_angle = (angle + window / 2) % (2 * np.pi)
    bin_edges = np.linspace(0, window, resolution + 1)

    if start_angle > end_angle:
        # Scanning window crosses the 2*pi boundary
        # Adjust angles by subtracting start_angle and wrapping around
        adjusted_projection = (projection - start_angle) % (2 * np.pi)
        # Keep only the points within the window (0 to window)
        adjusted_projection = adjusted_projection[adjusted_projection <= window]
    else:
        # Scanning window does not cross boundary
        # Keep only points within the scanning window
        adjusted_projection = projection[
            (projection >= start_angle) & (projection <= end_angle)
        ]
        # Adjust angles to start from 0
        adjusted_projection = adjusted_projection - start_angle

    histogram, _ = np.histogram(adjusted_projection, bins=bin_edges)
    return histogram


def compute_cdf(histogram):
    """
    Compute the cumulative distribution function (CDF) of a histogram.

    Parameters:
    - histogram: A numpy array of the histogram.

    Returns:
    - A numpy array of the CDF.
    """
    total = np.sum(histogram)
    if total > 0:
        cdf = np.cumsum(histogram).astype(np.float64) / total
    else:
        cdf = np.zeros_like(histogram, dtype=np.float64)
    return cdf
