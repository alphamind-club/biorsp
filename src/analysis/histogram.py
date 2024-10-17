import numpy as np

def compute_histogram(projection, resolution, angle, window):
    """
    Compute a histogram of a projection based on a scanning window.

    Parameters:
    - projection: Numpy array of angles in radians.
    - resolution: Number of bins in the histogram.
    - angle: The angle of the scanning window in radians.
    - window: The scanning window size in radians.

    Returns:
    - A numpy array representing the histogram of the projection.
    """
    start_angle = (angle - window / 2) % (2 * np.pi)
    end_angle = (angle + window / 2) % (2 * np.pi)
    bin_edges = np.linspace(0, window, resolution + 1)

    if start_angle > end_angle:
        # Handle case where scanning window crosses 2*pi boundary
        adjusted_projection = (projection - start_angle) % (2 * np.pi)
        adjusted_projection = adjusted_projection[adjusted_projection <= window]
    else:
        # Adjust for projections within the scanning window
        adjusted_projection = projection[(projection >= start_angle) & (projection <= end_angle)]
        adjusted_projection = adjusted_projection - start_angle

    histogram, _ = np.histogram(adjusted_projection, bins=bin_edges)
    return histogram


def compute_cdf(histogram):
    """
    Compute the cumulative distribution function (CDF) of a histogram.

    Parameters:
    - histogram: Numpy array of histogram values.

    Returns:
    - Numpy array representing the CDF.
    """
    total = np.sum(histogram)
    if total > 0:
        cdf = np.cumsum(histogram).astype(np.float64) / total
    else:
        cdf = np.zeros_like(histogram, dtype=np.float64)
    return cdf
