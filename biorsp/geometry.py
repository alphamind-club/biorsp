"""Geometry-related functions for BioRSP."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix                
from scipy.sparse.csgraph import dijkstra                
from scipy.stats import pearsonr                

MIN_CELL_COUNT_FOR_DISTORTION = 10


def geometric_median(
    points: Sequence[np.ndarray],
    eps: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute a geometric median for a set of points.

    The geometric median is robust to outliers and useful as a vantage candidate."""
    pts = np.asarray(points)
    median = np.mean(pts, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(pts - median, axis=1)
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1 / distances
        new_median = np.sum(weights[:, np.newaxis] * pts, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - median) < eps:
            break
        median = new_median
    return median


def bootstrap_vantage(coords, n_boot=25, frac=0.8, seed=0):
    """Bootstrap the vantage to quantify stability (VSS)."""
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords)
    n_cells = len(coords)

    medians = []
    for _ in range(n_boot):
        idx = rng.choice(n_cells, size=max(10, int(frac * n_cells)), replace=False)
        medians.append(geometric_median(coords[idx]))

    medians = np.vstack(medians)
    mean_med = np.mean(medians, axis=0)
    pairwise = np.linalg.norm(medians[:, None, :] - medians[None, :, :], axis=-1)
    vss = np.median(pairwise)
    return mean_med, vss, medians


def cartesian_to_polar(
    coords: np.ndarray,
    vantage_point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates relative to a vantage point.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_cells, 2).
    vantage_point : np.ndarray
        Coordinates used as origin for the polar transform.

    Returns
    -------
    r, theta : Tuple[np.ndarray, np.ndarray]
        Radii and angles in radians for each cell."""
    coords = np.asarray(coords)
    vantage_point = np.asarray(vantage_point)

    centered = coords - vantage_point
    r = np.sqrt(np.sum(centered**2, axis=1))
    theta = np.arctan2(centered[:, 1], centered[:, 0])

    return r, theta


def compute_geodesic_distances(
    adjacency: csr_matrix,
    vantage_indices: Optional[Union[int, Sequence[int]]],
) -> np.ndarray:
    """Compute geodesic distances on the graph from vantage point(s).

    Parameters
    ----------
    adjacency : scipy.sparse.spmatrix
        Adjacency matrix of the kNN graph.
    vantage_indices : int or list of int
        Index of the vantage cell(s).

    Returns
    -------
    r_geo : np.ndarray
        Geodesic distances."""
    if vantage_indices is None:
        raise ValueError("vantage_indices must be provided")

    if isinstance(vantage_indices, (int, np.integer)):
        indices = [int(vantage_indices)]
    else:
        indices = list(vantage_indices)

    dist_matrix = dijkstra(adjacency, directed=False, indices=indices)

    if len(indices) > 1:
        r_geo = np.min(dist_matrix, axis=0)
    else:
        r_geo = dist_matrix.flatten()

    return r_geo


def define_angular_grid(delta_phi_deg=5):
    """Define angular grid points."""
    num_points = int(np.ceil(360 / delta_phi_deg))
    grid_points = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    return grid_points


def bin_cells_angular(theta, weights, grid_points):
    """Bin cells into angular grid and sum their weights.

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
        Weighted histogram."""
    hist, _ = np.histogram(
        theta,
        bins=len(grid_points),
        range=(-np.pi, np.pi),
        weights=weights,
    )
    return hist


def convolve_histogram(hist, window_width_bins):
    """Apply circular convolution to smooth the histogram (sliding window)."""
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
    theta,
    r,
    weights,
    grid_points,
    window_width_deg=30,
    r_min=None,
    r_max=None,
):
    """Compute weighted counts in angular sectors using convolution."""
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
    """Compute local distortion score.

    Computes the correlation between high-dimensional geodesic distances and
    2D Euclidean distances in the embedding.

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
        Pearson correlation coefficient."""
    mask = np.ones(len(r_geo), dtype=bool)
    if r_min is not None:
        mask &= r_eucl >= r_min
    if r_max is not None:
        mask &= r_eucl < r_max

    if np.sum(mask) < MIN_CELL_COUNT_FOR_DISTORTION:
        return np.nan

    corr, _ = pearsonr(r_geo[mask], r_eucl[mask])
    return corr


def bin_cells_sparse(theta, grid_points, r=None, r_min=None, r_max=None):
    """Vectorized binning of cells into angular sectors using sparse matrix.

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
    b_map : scipy.sparse.csr_matrix
        Binary matrix (N_cells x K_bins) where b_ij = 1 if cell i is in bin j."""
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
    b_map = csr_matrix((data, (valid_indices, valid_bins)), shape=(n_cells, n_bins))

    return b_map


def sector_counts_from_map(bin_map, weights):
    """Fast weighted sector counts using a precomputed bin map."""
    result = bin_map.T @ weights
    if hasattr(result, "A"):
        result = result.A
    if weights.ndim == 1:
        return np.asarray(result).squeeze()
    return np.asarray(result)
