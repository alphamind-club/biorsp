import numpy as np

from .geometry import compute_sector_counts_convolved


def compute_rsp_curve(
    theta,
    r,
    weights,
    bg_weights,
    grid_points,
    window_width_deg=30,
    r_min=None,
    r_max=None,
    epsilon=1e-6,
):
    """
    Compute the RSP curve and densities using convolution.

    Parameters
    ----------
    weights : np.ndarray
        Continuous weights for foreground (e.g. expression).
    bg_weights : np.ndarray
        Weights for background (usually all 1s).
    """
    n_F = compute_sector_counts_convolved(
        theta, r, weights, grid_points, window_width_deg, r_min, r_max
    )
    n_B = compute_sector_counts_convolved(
        theta, r, bg_weights, grid_points, window_width_deg, r_min, r_max
    )

    valid_r: np.ndarray[tuple[int], np.dtype[np.Any]] = np.ones(len(r), dtype=bool)
    if r_min is not None:
        valid_r &= r >= r_min
    if r_max is not None:
        valid_r &= r < r_max

    N_F = np.sum(weights[valid_r])
    N_B = np.sum(bg_weights[valid_r])

    N_F: float = max(N_F, 1e-9)
    N_B: float = max(N_B, 1e-9)




    sum_n_F: np.floating[np.Any] = np.sum(n_F)
    sum_n_B: np.floating[np.Any] = np.sum(n_B)

    p_F = n_F / (sum_n_F + epsilon)
    p_B = n_B / (sum_n_B + epsilon)

    q_F = n_F / (n_B + epsilon)
    q_F_global: float = N_F / N_B

    rsp = np.log(q_F + epsilon) - np.log(q_F_global + epsilon)

    return {
        "n_F": n_F,
        "n_B": n_B,
        "p_F": p_F,
        "p_B": p_B,
        "q_F": q_F,
        "rsp": rsp,
        "N_F": N_F,
        "N_B": N_B,
    }


def compute_a1(p_F, p_B):
    """
    Compute A1: Coverage bias (Total Variation Distance).
    A1 = 0.5 * sum |p_F - p_B|
    Assumes p_F and p_B sum to 1.
    """
    return 0.5 * np.sum(np.abs(p_F - p_B))


def compute_a2(p_F, grid_points):
    """
    Compute A2: Angular skew (Mean Resultant Length).
    A2 = | sum p_F * e^(i*phi) |
    Assumes p_F sums to 1.
    """
    R1 = np.sum(p_F * np.exp(1j * grid_points))
    a2 = np.abs(R1)
    theta_hat = np.angle(R1)

    R2 = np.sum(p_F * np.exp(2j * grid_points))
    a2_2nd = np.abs(R2)

    return a2, theta_hat, a2_2nd


def compute_directional_deviance(p_F, p_B, grid_points):
    """
    Compute Directional Deviance (D_dir).
    D_dir = | sum (p_F - p_B) * e^(i*phi) |

    Parameters
    ----------
    p_F : np.ndarray
        Foreground probability density (sums to 1).
    p_B : np.ndarray
        Background probability density (sums to 1).
    grid_points : np.ndarray
        Angular grid points (radians).

    Returns
    -------
    d_dir : float
        Magnitude of the directional deviance vector.
    theta_dir : float
        Angle of the directional deviance vector.
    """
    diff = p_F - p_B
    R = np.sum(diff * np.exp(1j * grid_points))
    d_dir = np.abs(R)
    theta_dir = np.angle(R)
    return d_dir, theta_dir


def run_stratified_permutation(
    theta,
    r,
    weights,
    bg_weights,
    grid_points,
    window_width_deg,
    r_min,
    r_max,
    stratify_by=None,
    n_perm=1000,
):
    """
    Run stratified permutation test for Directional Deviance.

    Parameters
    ----------
    theta : np.ndarray
        Cell angles.
    r : np.ndarray
        Cell radii.
    weights : np.ndarray
        Cell weights (expression).
    bg_weights : np.ndarray
        Background weights.
    grid_points : np.ndarray
        Angular grid.
    window_width_deg : float
        Smoothing window width.
    r_min : float
        Min radius.
    r_max : float
        Max radius.
    stratify_by : np.ndarray, optional
        Labels for stratified permutation (e.g. batch or cluster).
    n_perm : int
        Number of permutations.

    Returns
    -------
    dict
        Test results including p-value, observed D_dir, and null distribution.
    """
    res = compute_rsp_curve(
        theta, r, weights, bg_weights, grid_points, window_width_deg, r_min, r_max
    )
    obs_d_dir, obs_theta = compute_directional_deviance(
        res["p_F"], res["p_B"], grid_points
    )

    null_d_dir: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_perm)

    valid_mask: np.ndarray[tuple[int], np.dtype[np.Any]] = np.ones(
        len(weights), dtype=bool
    )
    if r_min is not None:
        valid_mask &= r >= r_min
    if r_max is not None:
        valid_mask &= r < r_max

    indices = np.where(valid_mask)[0]

    perm_groups = []
    if stratify_by is None:
        perm_groups.append(indices)
    else:
        unique_strata = np.unique(stratify_by[indices])
        for s in unique_strata:
            s_mask = (stratify_by == s) & valid_mask
            s_indices = np.where(s_mask)[0]
            if len(s_indices) > 1:
                perm_groups.append(s_indices)


    null_rsp_list = []

    for i in range(n_perm):
        perm_weights = weights.copy()

        for grp in perm_groups:
            vals = weights[grp]
            np.random.shuffle(vals)
            perm_weights[grp] = vals

        perm_res = compute_rsp_curve(
            theta,
            r,
            perm_weights,
            bg_weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
        )
        d_dir, _ = compute_directional_deviance(
            perm_res["p_F"], perm_res["p_B"], grid_points
        )
        null_d_dir[i] = d_dir
        null_rsp_list.append(perm_res["rsp"])

    p_val = (1 + np.sum(null_d_dir >= obs_d_dir)) / (1 + n_perm)

    return {
        "p_value": p_val,
        "obs_d_dir": obs_d_dir,
        "obs_theta": obs_theta,
        "null_d_dir": null_d_dir,
        "null_rsp": np.array(null_rsp_list),
    }


def run_permutation_test(
    theta,
    r,
    weights,
    bg_weights,
    grid_points,
    window_width_deg,
    r_min,
    r_max,
    stratify_by=None,
    n_perm=100,
):
    """
    Run stratified permutation test.
    """
    res = compute_rsp_curve(
        theta, r, weights, bg_weights, grid_points, window_width_deg, r_min, r_max
    )
    obs_a1 = compute_a1(res["p_F"], res["p_B"])
    obs_a2, _, _ = compute_a2(res["p_F"], grid_points)

    a1_null = []
    a2_null = []

    valid_mask: np.ndarray[tuple[int], np.dtype[np.Any]] = np.ones(
        len(weights), dtype=bool
    )
    if r_min is not None:
        valid_mask &= r >= r_min
    if r_max is not None:
        valid_mask &= r < r_max

    indices = np.where(valid_mask)[0]

    for _ in range(n_perm):
        perm_weights = weights.copy()

        if stratify_by is None:
            perm_vals = weights[indices]
            np.random.shuffle(perm_vals)
            perm_weights[indices] = perm_vals
        else:
            unique_strata = np.unique(stratify_by[indices])
            for s in unique_strata:
                s_mask = (stratify_by == s) & valid_mask
                s_indices = np.where(s_mask)[0]
                if len(s_indices) > 1:
                    s_vals = weights[s_indices]
                    np.random.shuffle(s_vals)
                    perm_weights[s_indices] = s_vals

        perm_res = compute_rsp_curve(
            theta,
            r,
            perm_weights,
            bg_weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
        )
        perm_a1 = compute_a1(perm_res["p_F"], perm_res["p_B"])
        perm_a2, _, _ = compute_a2(perm_res["p_F"], grid_points)

        a1_null.append(perm_a1)
        a2_null.append(perm_a2)

    a1_null = np.array(a1_null)
    a2_null = np.array(a2_null)

    p_a1 = (1 + np.sum(a1_null >= obs_a1)) / (1 + n_perm)
    p_a2 = (1 + np.sum(a2_null >= obs_a2)) / (1 + n_perm)

    a1_mean: np.floating[np.Any] = np.mean(a1_null)
    a1_std: np.floating[np.Any] = np.std(a1_null)
    z_a1 = (obs_a1 - a1_mean) / (a1_std + 1e-9)

    a2_mean: np.floating[np.Any] = np.mean(a2_null)
    a2_std: np.floating[np.Any] = np.std(a2_null)
    z_a2 = (obs_a2 - a2_mean) / (a2_std + 1e-9)

    return {
        "p_a1": p_a1,
        "p_a2": p_a2,
        "z_a1": z_a1,
        "z_a2": z_a2,
        "a1_null": a1_null,
        "a2_null": a2_null,
    }


def run_bootstrap_test(
    theta,
    r,
    weights,
    bg_weights,
    grid_points,
    window_width_deg,
    r_min,
    r_max,
    n_boot=100,
):
    """
    Run bootstrap to estimate uncertainty of theta_hat.
    """
    valid_mask: np.ndarray[tuple[int], np.dtype[np.Any]] = np.ones(
        len(weights), dtype=bool
    )
    if r_min is not None:
        valid_mask &= r >= r_min
    if r_max is not None:
        valid_mask &= r < r_max

    indices = np.where(valid_mask)[0]
    n_samples: int = len(indices)

    thetas = []

    for _ in range(n_boot):
        boot_indices = np.random.choice(indices, n_samples, replace=True)


        boot_theta = theta[boot_indices]
        boot_r = r[boot_indices]
        boot_weights = weights[boot_indices]
        boot_bg_weights = bg_weights[boot_indices]

        res = compute_rsp_curve(
            boot_theta,
            boot_r,
            boot_weights,
            boot_bg_weights,
            grid_points,
            window_width_deg,
            r_min,
            r_max,
        )
        _, theta_hat, _ = compute_a2(res["p_F"], grid_points)
        thetas.append(theta_hat)

    thetas = np.array(thetas)

    return thetas
