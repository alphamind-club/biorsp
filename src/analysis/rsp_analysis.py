import numpy as np
from src.analysis.rsp_calculations import calculate_deviation_score, calculate_rsp_area


def rsp(
    foreground_points,
    background_points,
    vantage_point,
    scanning_window=np.pi,
    resolution=1000,
    angle_range=np.array([0, 2 * np.pi]),
    mode="absolute",
):
    """
    Perform the full RSP analysis including RSP area, RMSD, and deviation score.

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
    - rsp_area (float): Calculated RSP area.
    - rmsd (float): Root Mean Square Deviation.
    - deviation_score (float): Deviation score.
    - differences (ndarray): Differences calculated during the process.
    """
    rsp_area, differences, rmsd = calculate_rsp_area(
        foreground_points,
        background_points,
        vantage_point,
        scanning_window,
        resolution,
        angle_range,
        mode,
    )

    deviation_score = calculate_deviation_score(
        rsp_area, differences, resolution, angle_range
    )

    return rsp_area, rmsd, deviation_score, differences
