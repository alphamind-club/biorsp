import numpy as np


def convert_to_polar(coords, vantage_point):
    """
    Convert 2D coordinates to polar coordinates.

    Parameters:
    - coords: A 2D numpy array with the coordinates to be converted.
    - vantage_point: A 2D numpy array with the vantage point.

    Returns:
    - sorted_r: A numpy array of sorted radial coordinates.
    - sorted_theta: A numpy array of sorted angular coordinates.
    """
    if coords.shape[0] == 0:
        return np.array([]), np.array([])
    translated_coords = coords - vantage_point

    r = np.sqrt(translated_coords[:, 0] ** 2 + translated_coords[:, 1] ** 2)
    theta = np.arctan2(translated_coords[:, 1], translated_coords[:, 0])
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    sorted_indices = np.argsort(theta)
    sorted_r = r[sorted_indices]
    sorted_theta = theta[sorted_indices]

    return sorted_r, sorted_theta


def in_scanning_range(pt_theta, angle, window):
    """
    Check if a point is in the scanning window.

    Parameters:
    - pt_theta: The angle of the point in radians.
    - angle: The angle of the scanning window in radians.
    - window: The scanning window size in radians.

    Returns:
    - True if the point is in the scanning window, False otherwise.
    """
    angular_difference = np.abs((pt_theta - angle + np.pi) % (2 * np.pi) - np.pi)
    return angular_difference <= window / 2
