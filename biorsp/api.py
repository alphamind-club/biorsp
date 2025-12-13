import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any

from .geometry import (
    cartesian_to_polar,
    compute_geodesic_distances,
    compute_local_distortion,
    define_angular_grid,
)
from .metrics import (
    compute_a1,
    compute_a2,
    compute_directional_deviance,
    compute_rsp_curve,
    run_stratified_permutation,
)
from .preprocessing import compute_gene_weights


def set_vantage(
    adata,
    key="X_umap",
    mode="coordinates",
    x=None,
    y=None,
    cluster_key=None,
    cluster_id=None,
    pseudotime_key=None,
    density_key=None,
):
    """
    Define vantage point coordinates or index.

    Returns
    -------
    vantage : np.ndarray or int
        Coordinates (if mode='coordinates' or 'cluster_center' or 'density_peak')
        or Index (if mode='graph_node' or 'pseudotime_root')
    """
    if mode == "coordinates":
        if x is None or y is None:
            raise ValueError("x and y must be provided for mode='coordinates'")
        return np.array([x, y])

    elif mode == "cluster_center":
        if cluster_key is None or cluster_id is None:
            raise ValueError(
                "cluster_key and cluster_id must be provided for mode='cluster_center'"
            )

        if cluster_key not in adata.obs:
            raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")

        mask = adata.obs[cluster_key] == cluster_id
        if np.sum(mask) == 0:
            raise ValueError(f"Cluster {cluster_id} not found in {cluster_key}")

        coords = adata.obsm[key][mask]
        centroid = np.mean(coords, axis=0)
        return centroid

    elif mode == "density_peak":
        coords = adata.obsm[key]
        H, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=50)
        idx = np.unravel_index(np.argmax(H), H.shape)
        x_peak = (xedges[idx[0]] + xedges[idx[0] + 1]) / 2
        y_peak = (yedges[idx[1]] + yedges[idx[1] + 1]) / 2
        return np.array([x_peak, y_peak])

    elif mode == "geometric_median":
        coords = adata.obsm[key]

        def geometric_median(points, eps=1e-5, max_iter=100):
            points = np.array(points)
            median = np.mean(points, axis=0)
            for _ in range(max_iter):
                distances = np.linalg.norm(points - median, axis=1)
                distances = np.where(distances == 0, 1e-10, distances)
                weights = 1 / distances
                new_median = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(
                    weights
                )
                if np.linalg.norm(new_median - median) < eps:
                    break
                median = new_median
            return median

        return geometric_median(coords)

    elif mode == "pseudotime_root":
        if pseudotime_key is None:
            raise ValueError(
                "pseudotime_key must be provided for mode='pseudotime_root'"
            )

        pt = adata.obs[pseudotime_key].values
        root_idx = np.argmin(pt)
        return adata.obsm[key][root_idx]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def scan_genes(
    adata,
    genes,
    vantage_point,
    embedding_key="X_umap",
    use_graph=False,
    graph_key="connectivities",
    method="log_normalized",
    q=0.9,
    layer=None,
    delta_phi_deg=None,
    window_width_deg=None,
    r_min=None,
    r_max=None,
    stratify_key=None,
    n_perm=100,
    check_distortion=True,
) -> pd.DataFrame:
    """
    Batch scan multiple genes with confounder-robust inference and distortion checks.
    """
    if delta_phi_deg is None:
        delta_phi_deg = 5
    if window_width_deg is None:
        window_width_deg = 30

    coords = adata.obsm[embedding_key]

    vantage_idx = None
    vantage_coords = None

    if isinstance(vantage_point, (int, np.integer)):
        vantage_idx = vantage_point
        vantage_coords = coords[vantage_idx]
    else:
        vantage_coords = np.asarray(vantage_point)
        if use_graph or check_distortion:
            from sklearn.neighbors import NearestNeighbors

            nbrs: NearestNeighbors = NearestNeighbors(n_neighbors=1).fit(coords)
            _, indices = nbrs.kneighbors([vantage_coords])
            vantage_idx = indices[0][0]

    r_eucl, theta = cartesian_to_polar(coords, vantage_coords)

    r_geo = None
    if use_graph or check_distortion:
        if graph_key not in adata.obsp:
            print(f"Warning: {graph_key} not found. Skipping graph-based operations.")
            use_graph = False
            check_distortion = False
        else:
            adj = adata.obsp[graph_key]
            r_geo = compute_geodesic_distances(adj, vantage_idx)

    r = r_geo if use_graph else r_eucl

    if check_distortion and r_geo is not None:
        dist_score = compute_local_distortion(r_geo, r_eucl, r_min, r_max)
        if np.isnan(dist_score):
            print("Warning: Insufficient cells for distortion check.")
        elif dist_score < 0.5:
            print(
                f"CRITICAL WARNING: Local distortion score is low ({dist_score:.2f})."
            )
            print(
                "The embedding may not preserve angular relationships around this vantage point."
            )
            print("Results should be interpreted with extreme caution.")

    grid_points: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = (
        define_angular_grid(delta_phi_deg)
    )
    bg_weights: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(len(r))

    stratify_by = None
    if stratify_key is not None:
        if stratify_key not in adata.obs:
            raise ValueError(f"Stratification key {stratify_key} not found.")
        stratify_by = adata.obs[stratify_key].values

    results_list = []
    rsp_curves = {}


    for gene in tqdm(genes, desc="Scanning genes"):
        try:
            weights = compute_gene_weights(adata, gene, method=method, q=q, layer=layer)

            res = compute_rsp_curve(
                theta,
                r,
                weights,
                bg_weights,
                grid_points,
                window_width_deg,
                r_min,
                r_max,
            )

            d_dir, theta_dir = compute_directional_deviance(
                res["p_F"], res["p_B"], grid_points
            )

            a1 = compute_a1(res["p_F"], res["p_B"])
            a2, theta_hat, _ = compute_a2(res["p_F"], grid_points)

            row = {
                "gene": gene,
                "D_dir": d_dir,
                "theta_dir": theta_dir,
                "A1": a1,
                "A2": a2,
                "theta_hat": theta_hat,
                "N_F": res["N_F"],
            }

            if n_perm > 0:
                perm_res = run_stratified_permutation(
                    theta,
                    r,
                    weights,
                    bg_weights,
                    grid_points,
                    window_width_deg,
                    r_min,
                    r_max,
                    stratify_by=stratify_by,
                    n_perm=n_perm,
                )
                row["p_value"] = perm_res["p_value"]
                row["D_dir_null_mean"] = np.mean(perm_res["null_d_dir"])

            results_list.append(row)
            rsp_curves[gene] = res["rsp"]

        except Exception as e:
            print(f"Error processing {gene}: {e}")

    df_results = pd.DataFrame(results_list).set_index("gene")

    new_cols = df_results.columns
    for col in new_cols:
        adata.var[col] = df_results[col]

    if "biorsp" not in adata.uns:
        adata.uns["biorsp"] = {}

    adata.uns["biorsp"]["rsp_curves"] = pd.DataFrame(rsp_curves, index=grid_points)
    adata.uns["biorsp"]["params"] = {
        "vantage_point": vantage_point,
        "embedding_key": embedding_key,
        "use_graph": use_graph,
        "window_width_deg": window_width_deg,
        "delta_phi_deg": delta_phi_deg,
        "stratify_key": stratify_key,
    }

    return df_results


def compute_rsp(adata, gene, vantage_point, **kwargs):
    """
    Wrapper for single gene.
    """
    res: pd.DataFrame = scan_genes(adata, [gene], vantage_point, **kwargs)
    return res.loc[gene]
