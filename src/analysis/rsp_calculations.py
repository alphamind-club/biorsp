import numpy as np
from src.analysis.polar_conversion import convert_to_polar, in_scanning_range
from src.analysis.histogram import compute_histogram, compute_cdf


def compute_cdfs(
    fg_projection, bg_projection, angle, scanning_window, resolution, mode
):
    """
    Compute the CDFs of the foreground and background projections.

    Parameters:
    - fg_projection: A numpy array of angles in radians.
    - bg_projection: A numpy array of angles in radians.
    - angle: The angle of the scanning window in radians.
    - scanning_window: The scanning window size in radians.
    - resolution: The number of bins in the histogram.
    - mode: The mode to use for scaling the foreground and background CDFs.

    Returns:
    - fg_cdf: A numpy array of the foreground CDF.
    - bg_cdf: A numpy array of the background CDF.
    """
    fg_histogram = compute_histogram(fg_projection, resolution, angle, scanning_window)
    bg_histogram = compute_histogram(bg_projection, resolution, angle, scanning_window)

    fg_cdf = compute_cdf(fg_histogram)
    bg_cdf = compute_cdf(bg_histogram)

    if mode == "absolute":
        # Avoid division by zero
        bg_total = np.sum(bg_histogram)
        fg_total = np.sum(fg_histogram)
        if bg_total > 0:
            scaling_factor = fg_total / bg_total
            fg_cdf *= scaling_factor
        else:
            # No background points; cannot scale
            pass  # fg_cdf remains as is

    return fg_cdf, bg_cdf


def compute_area(fg_cdf, bg_cdf, window):
    """
    Compute the area under the absolute difference between the foreground and background CDFs.

    Parameters:
    - fg_cdf: A numpy array of the foreground CDF.
    - bg_cdf: A numpy array of the background CDF.
    - window: The scanning window size in radians.

    Returns:
    - The area under the absolute difference between the foreground and background CDFs.
    """
    dx = window / fg_cdf.shape[0]
    area_diff = np.trapz(np.abs(bg_cdf - fg_cdf), dx=dx)
    # area_diff = np.abs(area_diff)  # Ensure area_diff is non-negative
    return area_diff


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
    Calculate the differences between the foreground and background projections.

    Parameters:
    - foreground_points: A numpy array of (x, y) coordinates for cells with gene expression
    above the threshold.
    - background_points: A numpy array of (x, y) coordinates for all cells (or cells
    in selected clusters).
    - scanning_window: The scanning window size in radians.
    - resolution: The number of bins in the histogram.
    - vantage_point: A 2D numpy array with the vantage point.
    - angle_range: The range of angles to consider for the CDF computation.
    - mode: The mode to use for scaling the foreground and background CDFs.

    Returns:
    - differences: A numpy array of the differences between the foreground and background CDFs.
    """
    # Convert to polar coordinates and sort
    _, fg_theta = convert_to_polar(foreground_points, vantage_point)
    _, bg_theta = convert_to_polar(background_points, vantage_point)

    # Calculate CDFs for each angle
    differences = np.empty(resolution)
    angles = np.linspace(angle_range[0], angle_range[1], resolution, endpoint=False)
    # delta_theta = (angle_range[1] - angle_range[0]) / resolution

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


def calculate_rsp_area(
    foreground_points,
    background_points,
    vantage_point,
    scanning_window=np.pi,
    resolution=1000,
    angle_range=np.array([0, 2 * np.pi]),
    mode="absolute",
):
    """
    Calculate the RSP area and RMSD.

    Parameters:
    - foreground_points: A numpy array of (x, y) coordinates for cells with gene expression
    above the threshold.
    - background_points: A numpy array of (x, y) coordinates for all cells (or cells
    in selected clusters).
    - vantage_point: A 2D numpy array with the vantage point.
    - scanning_window: The scanning window size in radians.
    - resolution: The number of bins in the histogram.
    - angle_range: The range of angles to consider for the CDF computation.
    - mode: The mode to use for scaling the foreground and background CDFs.

    Returns:
    - rsp_area: The RSP area.
    - differences: A numpy array of the differences between the foreground and background CDFs.
    - rmsd: The RMSD.
    """
    differences = calculate_differences(
        foreground_points,
        background_points,
        scanning_window,
        resolution,
        vantage_point,
        angle_range,
        mode,
    )

    delta_theta = (angle_range[1] - angle_range[0]) / resolution
    segment_areas = 0.5 * delta_theta * np.power(differences, 2)
    rsp_area = np.sum(segment_areas)

    # RMSD calculation
    rmsd = np.sqrt(np.mean(np.square(differences)))

    return rsp_area, differences, rmsd


def calculate_deviation_score(rsp_area, differences, resolution, angle_range):
    """
    Calculate the deviation score.

    Parameters:
    - rsp_area (float): Calculated RSP area.
    - differences (ndarray): Differences calculated during the process.
    - resolution (int): Resolution for the calculation.
    - angle_range (array): The angular range over which the radar scans.

    Returns:
    - deviation_score (float): Deviation score.
    """
    # Equivalent circular radius
    radius = np.sqrt(rsp_area / np.pi)

    delta_theta = (angle_range[1] - angle_range[0]) / resolution

    # Calculate the overlap area
    intersection_area = np.sum(np.minimum(differences, radius)) * delta_theta

    # Deviation score
    if rsp_area != 0:
        deviation_score = intersection_area / rsp_area
    else:
        deviation_score = 0  # or handle as undefined

    return deviation_score
