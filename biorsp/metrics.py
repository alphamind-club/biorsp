"""Metrics for BioRSP analysis."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .geometry import compute_sector_counts_convolved, sector_counts_from_map

EPSILON = 1e-9
MIN_GROUP_SIZE = 2


def compute_rsp_curve(
    theta: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray,
    bg_weights: np.ndarray,
    grid_points: np.ndarray,
    window_width_deg: float = 30.0,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    epsilon: float = 1e-6,
    bin_map: Optional[Any] = None,
) -> Dict[str, Any]:
    if bin_map is None:
        n_f_arr = compute_sector_counts_convolved(
            theta,
            r,
            weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
        )
        n_b_arr = compute_sector_counts_convolved(
            theta,
            r,
            bg_weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
        )
    else:
        hist_f = sector_counts_from_map(bin_map, weights)
        hist_b = sector_counts_from_map(bin_map, bg_weights)
        delta_phi_deg = np.rad2deg(grid_points[1] - grid_points[0])
        window_width_bins = int(np.round(window_width_deg / delta_phi_deg))
        window_width_bins = max(1, window_width_bins)
        n_f_arr = np.convolve(
            np.concatenate(
                [
                    hist_f[-window_width_bins // 2 :],
                    hist_f,
                    hist_f[: window_width_bins // 2],
                ],
            ),
            np.ones(window_width_bins),
            mode="valid",
        )[: len(hist_f)]
        n_b_arr = np.convolve(
            np.concatenate(
                [
                    hist_b[-window_width_bins // 2 :],
                    hist_b,
                    hist_b[: window_width_bins // 2],
                ],
            ),
            np.ones(window_width_bins),
            mode="valid",
        )[: len(hist_b)]

    valid_r = np.ones(len(r), dtype=bool)
    if r_min is not None:
        valid_r &= r >= r_min
    if r_max is not None:
        valid_r &= r < r_max

    n_f_annulus = max(float(np.sum(weights[valid_r])), EPSILON)
    n_b_annulus = max(float(np.sum(bg_weights[valid_r])), EPSILON)

    total_n_f: float = float(np.sum(n_f_arr))
    total_n_b: float = float(np.sum(n_b_arr))

    p_f = n_f_arr / (total_n_f + epsilon)
    p_b = n_b_arr / (total_n_b + epsilon)

    q_f = n_f_arr / (n_b_arr + epsilon)
    q_f_global: float = n_f_annulus / n_b_annulus

    rsp = np.log(q_f + epsilon) - np.log(q_f_global + epsilon)

    return {
        "n_F": n_f_arr,
        "n_B": n_b_arr,
        "p_F": p_f,
        "p_B": p_b,
        "q_F": q_f,
        "rsp": rsp,
        "N_F": total_n_f,
        "N_B": total_n_b,
    }


def compute_wasserstein_circular(
    p: np.ndarray,
    q: np.ndarray,
    grid_points: np.ndarray,
) -> float:
    diff = p - q
    cumsum = np.cumsum(diff)
    center = np.median(cumsum)
    delta = (2 * np.pi) / len(grid_points)
    return delta * np.sum(np.abs(cumsum - center))


def compute_cra(
    p_F: np.ndarray,
    p_B: np.ndarray,
    grid_points: np.ndarray,
) -> Tuple[float, float, float]:
    w1 = compute_wasserstein_circular(p_F, p_B, grid_points)
    d_dir, theta_dir = compute_directional_deviance(p_F, p_B, grid_points)
    return w1, d_dir, theta_dir


def compute_directional_deviance(
    p_F: np.ndarray,
    p_B: np.ndarray,
    grid_points: np.ndarray,
) -> Tuple[float, float]:
    diff = p_F - p_B
    r = np.sum(diff * np.exp(1j * grid_points))
    d_dir = np.abs(r)
    theta_dir = np.angle(r)
    return d_dir, theta_dir


def _build_permutation_indices_stratified(
    strata: np.ndarray,
    n_perm: int,
    min_stratum_size: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_cells = len(strata)

    unique_strata, counts = np.unique(strata, return_counts=True)
    small_mask = counts < min_stratum_size

    if np.any(small_mask):
        small_strata_set = set(unique_strata[small_mask])
        strata_merged = np.array(
            ["__small__" if s in small_strata_set else s for s in strata],
        )
    else:
        strata_merged = strata.copy()
        small_strata_set = set()

    perm_indices = np.zeros((n_perm, n_cells), dtype=np.int64)

    for b in range(n_perm):
        idx_map = np.arange(n_cells)

        for stratum in np.unique(strata_merged):
            mask = strata_merged == stratum
            stratum_idx = np.where(mask)[0]

            if len(stratum_idx) >= MIN_GROUP_SIZE:
                shuffled = rng.permutation(stratum_idx)
                idx_map[stratum_idx] = shuffled

        perm_indices[b, :] = idx_map

    stratum_info = {
        "n_strata": len(np.unique(strata_merged)),
        "stratum_sizes": {
            s: np.sum(strata_merged == s) for s in np.unique(strata_merged)
        },
        "merged_small_strata": list(small_strata_set),
        "min_stratum_size_used": min_stratum_size,
    }

    return perm_indices, stratum_info


def _build_permutation_indices_knn(
    covariates: np.ndarray,
    n_perm: int,
    k: int = 30,
    stratify_by: Optional[np.ndarray] = None,
    min_stratum_size: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_cells = covariates.shape[0]
    covariates = np.asarray(covariates)

    perm_indices = np.zeros((n_perm, n_cells), dtype=np.int64)

    if stratify_by is None:
        k_actual = min(k, n_cells)
        nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm="auto").fit(covariates)
        neigh_idx = nbrs.kneighbors(return_distance=False)

        for b in range(n_perm):
            choice = rng.integers(0, k_actual, size=n_cells)
            perm_indices[b, :] = neigh_idx[np.arange(n_cells), choice]

        knn_info = {
            "k_used": k_actual,
            "stratified": False,
            "n_strata": 1,
        }
    else:
        unique_strata, counts = np.unique(stratify_by, return_counts=True)
        small_mask = counts < min_stratum_size

        if np.any(small_mask):
            small_strata_set = set(unique_strata[small_mask])
            strata_merged = np.array(
                ["__small__" if s in small_strata_set else s for s in stratify_by],
            )
        else:
            strata_merged = stratify_by.copy()
            small_strata_set = set()

        knn_info = {
            "stratified": True,
            "n_strata": len(np.unique(strata_merged)),
            "stratum_sizes": {
                s: np.sum(strata_merged == s) for s in np.unique(strata_merged)
            },
            "merged_small_strata": list(small_strata_set),
        }

        for b in range(n_perm):
            idx_map = np.arange(n_cells)

            for stratum in np.unique(strata_merged):
                mask = strata_merged == stratum
                stratum_idx = np.where(mask)[0]

                if len(stratum_idx) < MIN_GROUP_SIZE:
                    continue

                sub_cov = covariates[stratum_idx]
                k_stratum = min(k, len(stratum_idx))
                nbrs = NearestNeighbors(n_neighbors=k_stratum, algorithm="auto").fit(
                    sub_cov,
                )
                neigh_idx = nbrs.kneighbors(return_distance=False)

                choice = rng.integers(0, k_stratum, size=len(stratum_idx))
                local_donors = neigh_idx[np.arange(len(stratum_idx)), choice]

                idx_map[stratum_idx] = stratum_idx[local_donors]

            perm_indices[b, :] = idx_map

    return perm_indices, knn_info


def run_conditional_permutation(
    theta: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray,
    bg_weights: np.ndarray,
    grid_points: np.ndarray,
    window_width_deg: float,
    r_min: Optional[float],
    r_max: Optional[float],
    perm_indices: np.ndarray,
    bin_map: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run conditional permutation test and return summary statistics.

    This function computes observed and null CRA/W1 and directional deviance
    statistics by applying precomputed permutation indices to the foreground
    weights.
    """
    res = compute_rsp_curve(
        theta,
        r,
        weights,
        bg_weights,
        grid_points,
        window_width_deg,
        r_min,
        r_max,
        bin_map=bin_map,
    )
    w1_obs, d_dir_obs, theta_dir_obs = compute_cra(res["p_F"], res["p_B"], grid_points)

    n_perm = perm_indices.shape[0]
    null_w1 = np.zeros(n_perm)
    null_d_dir = np.zeros(n_perm)

    for b in range(n_perm):
        perm_w = weights[perm_indices[b, :]]

        perm_res = compute_rsp_curve(
            theta,
            r,
            perm_w,
            bg_weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
            bin_map=bin_map,
        )
        w1, d_dir, _ = compute_cra(perm_res["p_F"], perm_res["p_B"], grid_points)
        null_w1[b] = w1
        null_d_dir[b] = d_dir

    p_w1 = (1 + np.sum(null_w1 >= w1_obs)) / (1 + n_perm)
    p_d = (1 + np.sum(null_d_dir >= d_dir_obs)) / (1 + n_perm)

    null_mean_w1 = np.mean(null_w1)
    null_std_w1 = np.std(null_w1)
    z_w1 = (w1_obs - null_mean_w1) / (null_std_w1 + EPSILON)

    return {
        "p_w1": p_w1,
        "p_d_dir": p_d,
        "w1_obs": w1_obs,
        "d_dir_obs": d_dir_obs,
        "theta_dir_obs": theta_dir_obs,
        "null_w1": null_w1,
        "null_d_dir": null_d_dir,
        "Z_w1": z_w1,
        "null_mean_w1": null_mean_w1,
        "null_std_w1": null_std_w1,
        "rsp_obs": res["rsp"],
    }


def compute_approximate_null_z_score(
    theta,
    weights,
    bg_weights,
    grid_points,
    window_width_deg,
    strata=None,
):
    """Compute Z-score for D_dir using CLT approximation.

    This approximation assumes stratified permutation of labels."""
    n_grid = len(grid_points)
    delta = 2 * np.pi / n_grid

    bin_indices = np.round((theta - grid_points[0]) / delta).astype(int) % n_grid

    window_width_bins = int(np.round(window_width_deg / np.rad2deg(delta)))
    window_width_bins = max(1, window_width_bins)

    u_bins = np.zeros(n_grid, dtype=np.complex128)
    grid_complex = np.exp(1j * grid_points)

    for j in range(n_grid):
        indices = (
            np.arange(
                j - window_width_bins // 2,
                j - window_width_bins // 2 + window_width_bins,
            )
            % n_grid
        )
        u_bins[j] = np.sum(grid_complex[indices])

    u_cells = u_bins[bin_indices]

    n_f = np.sum(weights)
    n_b = np.sum(bg_weights)

    x_obs = np.sum(weights * u_cells) / n_f
    y_obs = np.sum(bg_weights * u_cells) / n_b

    d_obs = np.abs(x_obs - y_obs)

    if strata is None:
        strata = np.zeros(len(weights), dtype=int)

    unique_strata = np.unique(strata)

    e_x = 0j
    var_x_real = 0.0
    var_x_imag = 0.0
    cov_x_ri = 0.0

    for s in unique_strata:
        mask = strata == s
        w_s = weights[mask]
        u_s = u_cells[mask]
        n_s = len(w_s)

        if n_s < MIN_GROUP_SIZE:
            e_x += np.sum(w_s * u_s)
            continue

        mean_w = np.mean(w_s)
        mean_u = np.mean(u_s)

        e_x += n_s * mean_w * mean_u

        var_w_sum = np.sum((w_s - mean_w) ** 2)

        a_s = u_s.real
        b_s = u_s.imag
        mean_a = np.mean(a_s)
        mean_b = np.mean(b_s)

        var_a_sum = np.sum((a_s - mean_a) ** 2)
        var_b_sum = np.sum((b_s - mean_b) ** 2)
        cov_ab_sum = np.sum((a_s - mean_a) * (b_s - mean_b))

        factor = var_w_sum / (n_s - 1)

        var_x_real += factor * var_a_sum
        var_x_imag += factor * var_b_sum
        cov_x_ri += factor * cov_ab_sum

    e_x /= n_f
    var_x_real /= n_f**2
    var_x_imag /= n_f**2
    cov_x_ri /= n_f**2

    z_mean = e_x - y_obs

    if d_obs < EPSILON:
        return 0.0

    theta_obs = np.angle(x_obs - y_obs)
    proj_vec = np.array([np.cos(theta_obs), np.sin(theta_obs)])

    var_proj = (
        proj_vec[0] ** 2 * var_x_real
        + proj_vec[1] ** 2 * var_x_imag
        + 2 * proj_vec[0] * proj_vec[1] * cov_x_ri
    )

    mean_proj = z_mean.real * proj_vec[0] + z_mean.imag * proj_vec[1]

    return (d_obs - mean_proj) / (np.sqrt(var_proj) + EPSILON)
