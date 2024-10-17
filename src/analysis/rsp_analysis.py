import numpy as np
from src.analysis.rsp_calculations import (
    calculate_deviation_score,
    calculate_rsp_area,
    calculate_differences,
    calculate_rmsd,
)


def perform_rsp_analysis(
    foreground_points,
    background_points,
    vantage_point,
    scanning_window=np.pi,
    resolution=1000,
    angle_range=np.array([0, 2 * np.pi]),
    mode="absolute",
):
    """
    Perform full RSP analysis including RSP area, RMSD, and deviation score.

    Parameters:
    - foreground_points: Numpy array of (x, y) coordinates for foreground cells.
    - background_points: Numpy array of (x, y) coordinates for background cells.
    - vantage_point: 2D numpy array for the vantage point.
    - scanning_window: Scanning window size in radians.
    - resolution: Number of bins for the histogram.
    - angle_range: Angular range for CDF computation.
    - mode: Mode for scaling foreground and background CDFs (default="absolute").

    Returns:
    - rsp_area: Calculated RSP area.
    - rmsd: Root Mean Square Deviation.
    - deviation_score: Deviation score.
    - differences: Numpy array of differences between foreground and background CDFs.
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

    rsp_area = calculate_rsp_area(differences, angle_range, resolution)
    rmsd = calculate_rmsd(differences)
    deviation_score = calculate_deviation_score(
        rsp_area, differences, resolution, angle_range
    )

    return rsp_area, rmsd, deviation_score, differences
