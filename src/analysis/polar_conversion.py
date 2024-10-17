import numpy as np

def convert_to_polar(coords, vantage_point):
    """
    Convert 2D coordinates to polar coordinates.

    Parameters:
    - coords: 2D numpy array of coordinates.
    - vantage_point: 2D numpy array representing the reference point for polar conversion.

    Returns:
    - sorted_r: Numpy array of radial coordinates, sorted by angular coordinates.
    - sorted_theta: Numpy array of angular coordinates, sorted.
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


def in_scanning_range(point_theta, angle, window):
    """
    Check if an angular coordinate is within a scanning range.

    Parameters:
    - point_theta: The angle of the point in radians.
    - angle: The angle of the scanning window in radians.
    - window: The size of the scanning window in radians.

    Returns:
    - True if the point is within the scanning range, False otherwise.
    """
    angular_difference = np.abs((point_theta - angle + np.pi) % (2 * np.pi) - np.pi)
    return angular_difference <= window / 2
