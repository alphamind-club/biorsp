import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr


def cartesian_to_polar(coords, vantage_point):
    """
    Convert Cartesian coordinates to polar coordinates relative to a vantage point.
    """
    coords = np.asarray(coords)
    vantage_point = np.asarray(vantage_point)

    centered = coords - vantage_point
    r = np.sqrt(np.sum(centered**2, axis=1))
    theta = np.arctan2(centered[:, 1], centered[:, 0])

    return r, theta


def compute_geodesic_distances(adjacency, vantage_indices):
    """
    Compute geodesic distances on the graph from vantage point(s).

    Parameters
    ----------
    adjacency : scipy.sparse.spmatrix
        Adjacency matrix of the kNN graph.
    vantage_indices : int or list of int
        Index of the vantage cell(s).

    Returns
    -------
    r_geo : np.ndarray
        Geodesic distances.
    """
    if isinstance(vantage_indices, (int, np.integer)):
        indices = [vantage_indices]
    else:
        indices = vantage_indices

    dist_matrix = dijkstra(adjacency, directed=False, indices=indices)

    if len(indices) > 1:
        r_geo = np.min(dist_matrix, axis=0)
    else:
        r_geo = dist_matrix.flatten()

    return r_geo


def define_angular_grid(delta_phi_deg=5):
    """
    Define angular grid points.
    """
    num_points = int(np.ceil(360 / delta_phi_deg))
    grid_points = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    return grid_points


def bin_cells_angular(theta, weights, grid_points):
    """
    Bin cells into angular grid and sum their weights.

    Parameters
    ----------
    theta : np.ndarray
        Cell angles.
    weights : np.ndarray
        Cell weights (e.g. expression or 1 for background).
    grid_points : np.ndarray
        Grid centers.

    Returns
    -------
    hist : np.ndarray
        Weighted histogram.
    """


    hist, _ = np.histogram(
        theta, bins=len(grid_points), range=(-np.pi, np.pi), weights=weights
    )
    return hist


def convolve_histogram(hist, window_width_bins):
    """
    Apply circular convolution to smooth the histogram (sliding window).
    """
    kernel = np.ones(window_width_bins)

    pad_width = window_width_bins // 2
    hist_padded = np.concatenate([hist[-pad_width:], hist, hist[:pad_width]])

    convolved = np.convolve(hist_padded, kernel, mode="valid")


    if len(convolved) > len(hist):
        start = (len(convolved) - len(hist)) // 2
        convolved = convolved[start : start + len(hist)]
    elif len(convolved) < len(hist):
        pass

    return convolved


def compute_sector_counts_convolved(
    theta, r, weights, grid_points, window_width_deg=30, r_min=None, r_max=None
):
    """
    Compute weighted counts in angular sectors using convolution.
    """
    valid_r = np.ones(len(r), dtype=bool)
    if r_min is not None:
        valid_r &= r >= r_min
    if r_max is not None:
        valid_r &= r < r_max

    active_theta = theta[valid_r]
    active_weights = weights[valid_r]

    hist = bin_cells_angular(active_theta, active_weights, grid_points)

    delta_phi_deg = np.rad2deg(grid_points[1] - grid_points[0])
    window_width_bins = int(np.round(window_width_deg / delta_phi_deg))
    window_width_bins = max(1, window_width_bins)

    counts = convolve_histogram(hist, window_width_bins)

    return counts


def compute_local_distortion(r_geo, r_eucl, r_min=None, r_max=None):
    """
    Compute local distortion score: correlation of high-dim geodesic vs. 2D Euclidean distance.

    Parameters
    ----------
    r_geo : np.ndarray
        Geodesic distances from vantage point.
    r_eucl : np.ndarray
        Euclidean distances from vantage point in embedding.
    r_min : float, optional
        Minimum radius for annulus.
    r_max : float, optional
        Maximum radius for annulus.

    Returns
    -------
    distortion_score : float
        Pearson correlation coefficient.
    """
    mask = np.ones(len(r_geo), dtype=bool)
    if r_min is not None:
        mask &= r_eucl >= r_min
    if r_max is not None:
        mask &= r_eucl < r_max

    if np.sum(mask) < 10:
        return np.nan

    corr, _ = pearsonr(r_geo[mask], r_eucl[mask])
    return corr


def bin_cells_sparse(theta, grid_points, r=None, r_min=None, r_max=None):
    """
    Vectorized binning of cells into angular sectors using sparse matrix.

    Parameters
    ----------
    theta : np.ndarray
        Cell angles in [-pi, pi].
    grid_points : np.ndarray
        Grid centers in [-pi, pi].
    r : np.ndarray, optional
        Cell radii.
    r_min : float, optional
        Minimum radius filter.
    r_max : float, optional
        Maximum radius filter.

    Returns
    -------
    B_map : scipy.sparse.csr_matrix
        Binary matrix (N_cells x K_bins) where B_ij = 1 if cell i is in bin j.
    """
    n_cells = len(theta)
    n_bins = len(grid_points)

    mask = np.ones(n_cells, dtype=bool)
    if r is not None:
        if r_min is not None:
            mask &= r >= r_min
        if r_max is not None:
            mask &= r < r_max

    delta = grid_points[1] - grid_points[0]
    edges = np.concatenate([grid_points - delta / 2, [grid_points[-1] + delta / 2]])

    theta = np.arctan2(np.sin(theta), np.cos(theta))

    bin_indices = np.digitize(theta, edges) - 1

    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    valid_indices = np.where(mask)[0]
    valid_bins = bin_indices[mask]

    data = np.ones(len(valid_indices), dtype=np.float32)
    B_map = csr_matrix((data, (valid_indices, valid_bins)), shape=(n_cells, n_bins))

    return B_map
