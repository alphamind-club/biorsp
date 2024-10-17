import numpy as np
from src.analysis.polar_conversion import convert_to_polar, in_scanning_range
from src.analysis.cdf_calculations import compute_cdfs, compute_area

def calculate_differences(
    foreground_points, 
    background_points, 
    scanning_window, 
    resolution, 
    vantage_point, 
    angle_range, 
    mode
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
    - mode: Mode for scaling foreground and background CDFs.

    Returns:
    - differences: Numpy array of differences between foreground and background CDFs.
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

def calculate_rsp_area(differences, angle_range, resolution):
    """
    Calculate the RSP area from the differences.

    Parameters:
    - differences: Numpy array of differences between foreground and background CDFs.
    - angle_range: The range of angles over which the differences are calculated.
    - resolution: The number of bins in the histogram.

    Returns:
    - rsp_area: The calculated RSP area.
    """
    delta_theta = (angle_range[1] - angle_range[0]) / resolution
    segment_areas = 0.5 * delta_theta * np.power(differences, 2)
    rsp_area = np.sum(segment_areas)
    
    return rsp_area

def calculate_rmsd(differences):
    """
    Calculate the Root Mean Square Deviation (RMSD) from the differences.

    Parameters:
    - differences: Numpy array of differences between foreground and background CDFs.

    Returns:
    - rmsd: The calculated RMSD.
    """
    rmsd = np.sqrt(np.mean(np.square(differences)))
    
    return rmsd
