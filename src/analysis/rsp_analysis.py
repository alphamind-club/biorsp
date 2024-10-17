import numpy as np
from src.analysis.rsp_calculations import calculate_deviation_score, calculate_rsp_area


def rsp(
    foreground_points, background_points, vantage_point, 
    scanning_window=np.pi, resolution=1000, angle_range=np.array([0, 2 * np.pi]), mode="absolute"
):
    """
    Perform full RSP analysis including RSP area, RMSD, and deviation score.

    Parameters:
    - foreground_points: Numpy array of (x, y) coordinates for foreground cells.
    - background_points: Numpy array of (x, y) coordinates for background cells.
    - vantage_point: 2D numpy array of the vantage point.
    - scanning_window: The scanning window size in radians.
    - resolution: Number of bins for the histogram.
    - angle_range: Range of angles for CDF computation.
    - mode: Scaling mode for foreground and background CDFs (default="absolute").

    Returns:
    - rsp_area: Calculated RSP area.
    - rmsd: Root Mean Square Deviation.
    - deviation_score: Deviation score.
    - differences: Numpy array of differences between foreground and background CDFs.
    """
    rsp_area, differences, rmsd = calculate_rsp_area(
        foreground_points, background_points, vantage_point, 
        scanning_window, resolution, angle_range, mode
    )

    deviation_score = calculate_deviation_score(rsp_area, differences, resolution, angle_range)
    return rsp_area, rmsd, deviation_score, differences
