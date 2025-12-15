"""BioRSP API module for radial spatial pattern analysis."""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .geometry import (
    bin_cells_sparse,
    bootstrap_vantage,
    cartesian_to_polar,
    compute_geodesic_distances,
    compute_local_distortion,
    define_angular_grid,
)
from .metrics import (
    _build_permutation_indices_knn,
    _build_permutation_indices_stratified,
    run_conditional_permutation,
)
from .preprocessing import build_covariate_matrix, compute_gene_weights
from .validation import (
    BioRSPValidationError,
    compute_fdr_bh,
    validate_annulus_cells,
    validate_covariates,
    validate_effective_mass,
    validate_embedding_distortion,
    validate_minimum_cells,
    validate_strata_sizes,
    validate_vantage_stability,
)

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

__version__ = "3.0.0-dev"


def get_code_version() -> str:
    """Get a unique identifier for the current BioRSP code version for reproducibility.

    This function creates a short hash of the source code to help track which
    version of the analysis was used for your results.

    Returns
    -------
    str
        Short hash string representing the current code version.

    """
    source = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256(source.encode()).hexdigest()[:8]


def define_reference_point(
    spatial_data: AnnData,
    coordinate_system: str = "X_umap",
    method: str = "coordinates",
    x_coordinate: float | None = None,
    y_coordinate: float | None = None,
    cluster_column: str | None = None,
    cluster_name: str | None = None,
    trajectory_column: str | None = None,
) -> np.ndarray:
    """Choose a reference point (center) for radial spatial analysis.

    This function determines where to place the "center" of your analysis. All
    spatial patterns will be measured relative to this point.

    Args:
        spatial_data: AnnData object containing spatial coordinates and gene expression.
        coordinate_system: Name of the coordinate system in spatial_data.obsm to use.
        method: How to choose the reference point. Options are:
            - 'coordinates': Use specific x,y coordinates you provide
            - 'cluster_center': Center of a specific cell cluster
            - 'density_peak': Location with highest cell density
            - 'geometric_median': Balanced center of all cells
            - 'trajectory_start': Cell with earliest developmental time
        x_coordinate: X position when method='coordinates'.
        y_coordinate: Y position when method='coordinates'.
        cluster_column: Column name for cluster labels when method='cluster_center'.
        cluster_name: Specific cluster name when method='cluster_center'.
        trajectory_column: Column name for developmental time when
            method='trajectory_start'.

    Returns:
        Reference point coordinates as a numpy array.

    Raises:
        ValueError: If required parameters for the chosen method are not provided.

    Examples:
        center = define_reference_point(
            data,
            method="coordinates",
            x_coordinate=5.0,
            y_coordinate=-2.0,
        )

        center = define_reference_point(
            data,
            method="cluster_center",
            cluster_column="cell_type",
            cluster_name="stem_cells",
        )

    """
    if method == "coordinates":
        if x_coordinate is None or y_coordinate is None:
            msg = (
                "x_coordinate and y_coordinate must be provided for "
                "method='coordinates'"
            )
            raise ValueError(msg)
        return np.array([x_coordinate, y_coordinate])

    if method == "cluster_center":
        if cluster_column is None or cluster_name is None:
            msg = (
                "cluster_column and cluster_name must be provided for "
                "method='cluster_center'"
            )
            raise ValueError(msg)

        if cluster_column not in spatial_data.obs:
            msg = f"Cluster column {cluster_column} not found in spatial_data.obs"
            raise ValueError(msg)

        mask = spatial_data.obs[cluster_column] == cluster_name
        if np.sum(mask) == 0:
            msg = f"Cluster {cluster_name} not found in {cluster_column}"
            raise ValueError(msg)

        coords = spatial_data.obsm[coordinate_system][mask]
        return np.mean(coords, axis=0)

    if method == "density_peak":
        coords = spatial_data.obsm[coordinate_system]
        h, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=50)
        idx = np.unravel_index(np.argmax(h), h.shape)
        x_peak = (xedges[idx[0]] + xedges[idx[0] + 1]) / 2
        y_peak = (yedges[idx[1]] + yedges[idx[1] + 1]) / 2
        return np.array([x_peak, y_peak])

    if method == "geometric_median":
        coords = spatial_data.obsm[coordinate_system]

        def geometric_median(
            points: np.ndarray,
            eps: float = 1e-5,
            max_iter: int = 100,
        ) -> np.ndarray:
            points = np.array(points)
            median = np.mean(points, axis=0)
            for _ in range(max_iter):
                distances = np.linalg.norm(points - median, axis=1)
                distances = np.where(distances == 0, 1e-10, distances)
                weights = 1 / distances
                new_median = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(
                    weights,
                )
                if np.linalg.norm(new_median - median) < eps:
                    break
                median = new_median
            return median

        return geometric_median(coords)

    if method == "trajectory_start":
        if trajectory_column is None:
            msg = "trajectory_column must be provided for method='trajectory_start'"
            raise ValueError(
                msg,
            )

        trajectory_values = spatial_data.obs[trajectory_column].to_numpy()
        earliest_idx = np.argmin(trajectory_values)
        return spatial_data.obsm[coordinate_system][earliest_idx]

    msg = (
        f"Unknown method: {method}. Choose from: coordinates, cluster_center, "
        "density_peak, geometric_median, trajectory_start"
    )
    raise ValueError(
        msg,
    )


def find_spatially_patterned_genes(
    spatial_data: AnnData,
    genes_to_test: list[str],
    reference_point: int | Sequence[float] | np.ndarray,
    coordinate_system: str = "X_umap",
    condition_column: str | None = None,
    condition_value: str | None = None,
    inner_radius_percentile: float = 0.1,
    outer_radius_percentile: float = 0.9,
    *,
    use_graph_distances: bool = False,
    graph_distances_key: str = "connectivities",
    expression_method: str = "log_normalized",
    expression_threshold_percentile: float = 0.9,
    expression_layer: str | None = None,
    angle_resolution_degrees: float = 5.0,
    smoothing_window_degrees: float = 30.0,
    confounding_factors: list[str] | None = None,
    require_sample_info: bool = True,
    require_batch_info: bool = True,
    require_depth_info: bool = True,
    allow_uncalibrated_analysis: bool = False,
    num_permutations: int = 500,
    permutation_method: str = "stratified",
    stratification_column: str | None = None,
    include_log_depth: bool = True,
    neighbors_for_matching: int = 30,
    min_group_size: int = 20,
    min_cells_required: int = 200,
    min_expression_mass: float = 10.0,
    min_cells_in_analysis_region: int = 100,
    allow_exploratory_mode: bool = False,
    check_spatial_distortion: bool = True,
    min_distortion_correlation: float = 0.5,
    reference_point_stability_tests: int = 25,
    stability_threshold_fraction: float = 0.05,
    significance_threshold: float = 0.05,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Find genes that show spatial patterns radiating from a center point.

    This is the main function for discovering genes with radial spatial patterns.
    It uses advanced statistical methods to account for technical biases and
    provides reliable results for scientific publications.

    Args:
        spatial_data: AnnData object containing spatial coordinates and
            gene expression data.
        genes_to_test: List of gene names to analyze for spatial patterns.
        reference_point: Center point for analysis. Either coordinates (array)
            or a global cell index (integer).
        coordinate_system: Name of the 2D coordinate system in spatial_data.obsm.
        condition_column: Optional column in spatial_data.obs used to select cells.
        condition_value: Value in condition_column that selects cells for analysis.

        inner_radius_percentile: Inner boundary as percentile of distances from center
            (0.1 = 10th percentile).
        outer_radius_percentile: Outer boundary as percentile of distances from center
            (0.9 = 90th percentile).
        use_graph_distances: Use graph-based distances instead of
            straight-line distances.
        graph_distances_key: Name of distance matrix in spatial_data.obsp.

        expression_method: How to process gene expression.
            ('log_normalized' recommended)
        expression_threshold_percentile: Expression threshold percentile for analysis.
        expression_layer: Optional data layer containing expression values.

        angle_resolution_degrees: How finely to divide the circle.
            Smaller values give finer resolution.
        smoothing_window_degrees: Width of smoothing window for detection.

        confounding_factors: List of column names for technical factors.
            Examples: sample, batch, sequencing depth columns.
        require_sample_info: Require sample information for confounder control.
        require_batch_info: Require batch information for confounder control.
        require_depth_info: Require sequencing depth information (e.g., n_counts).
        allow_uncalibrated_analysis: If True, run without full confounder checks.

        num_permutations: Number of random permutations for statistical testing.
            Higher values increase accuracy.
        permutation_method: Method for creating matched random samples.
            Options: 'stratified' or 'knn'.
        stratification_column: Column name for grouping similar cells during testing.
        neighbors_for_matching: Number of similar cells to match for 'knn'.
        min_group_size: Minimum cells per group required for statistics.

        min_cells_required: Minimum total cells needed for analysis.
        min_expression_mass: Minimum gene expression level required.
        min_cells_in_analysis_region: Minimum cells in the analysis distance range.
        allow_exploratory_mode: Allow analysis with relaxed quality controls.

        check_spatial_distortion: Verify that spatial coordinates are reliable.
        min_distortion_correlation: Minimum correlation allowed for spatial coordinates.
        reference_point_stability_tests: Number of tests for center point stability.
        stability_threshold_fraction: Threshold for acceptable center point variation.

        significance_threshold: Statistical significance level (e.g., 0.05 for 5%).

        random_seed: Random number seed for reproducible results.

    Returns:
        pd.DataFrame: Analysis results with gene names as index and columns including:
            - ARIA: Pattern strength statistic
            - p_CRA: Statistical significance (uncorrected)
            - q_CRA: Statistical significance (corrected for multiple testing)
            - discovery: Whether gene shows significant spatial pattern

    Examples:
        results = find_spatially_patterned_genes(
            spatial_data=my_data,
            genes_to_test=['Gene1', 'Gene2', 'Gene3'],
            reference_point=center_coords,
            confounding_factors=['sample', 'batch']
        )

        results = find_spatially_patterned_genes(
            spatial_data=my_data,
            genes_to_test=all_genes,
            reference_point=center_coords,
            coordinate_system='X_pca',
            inner_radius_percentile=0.2,
            outer_radius_percentile=0.8,
            num_permutations=1000,
            confounding_factors=['sample', 'batch', 'sequencing_depth']
        )

    """
    analysis_settings = {
        "version": __version__,
        "version_hash": get_code_version(),
        "coordinate_system": coordinate_system,
        "condition_column": condition_column,
        "condition_value": condition_value,
        "permutation_method": permutation_method,
        "num_permutations": num_permutations,
        "random_seed": random_seed,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if condition_column is not None:
        if condition_column not in spatial_data.obs:
            msg = f"Condition column '{condition_column}' not found in spatial_data.obs"
            raise ValueError(
                msg,
            )

        condition_mask = spatial_data.obs[condition_column] == condition_value
        subset_data = spatial_data[condition_mask].copy()
        analysis_settings["cells_in_condition"] = condition_mask.sum()

        if subset_data.n_obs == 0:
            msg = f"No cells found with {condition_column} = {condition_value}"
            raise ValueError(
                msg,
            )
    else:
        subset_data = spatial_data
        analysis_settings["total_cells"] = spatial_data.n_obs

    validate_minimum_cells(
        subset_data.n_obs,
        min_cells=min_cells_required,
        exploratory_min=50,
        allow_exploratory=allow_exploratory_mode,
    )

    validate_covariates(
        subset_data,
        confounding_factors,
        require_sample=require_sample_info,
        require_batch=require_batch_info,
        require_depth=require_depth_info,
        allow_uncalibrated=allow_uncalibrated_analysis,
    )

    if coordinate_system not in subset_data.obsm:
        msg = f"Coordinate system '{coordinate_system}' not found in subset_data.obsm"
        raise ValueError(
            msg,
        )

    spatial_coords = subset_data.obsm[coordinate_system][
        :,
        :2,
    ]

    if isinstance(reference_point, (int, np.integer)):
        reference_index = int(reference_point)
        if reference_index >= len(spatial_coords):
            msg = f"Reference index {reference_index} out of bounds"
            raise ValueError(msg)
        reference_coords = spatial_coords[reference_index]
    else:
        reference_coords = np.asarray(reference_point)

        coord_matcher = NearestNeighbors(n_neighbors=1).fit(spatial_coords)
        _, indices = coord_matcher.kneighbors([reference_coords])
        reference_index = int(indices[0][0])

    analysis_settings["reference_coords"] = reference_coords.tolist()
    analysis_settings["reference_index"] = reference_index

    provenance = analysis_settings.copy()

    _, vss, _ = bootstrap_vantage(
        spatial_coords,
        n_boot=reference_point_stability_tests,
        frac=0.8,
        seed=random_seed,
    )
    emb_diam = np.max(
        np.linalg.norm(spatial_coords - spatial_coords.mean(axis=0), axis=1),
    )

    validate_vantage_stability(
        vss,
        emb_diam,
        threshold_frac=stability_threshold_fraction,
    )

    provenance["vss"] = vss
    provenance["embedding_diameter"] = emb_diam

    r_eucl, theta = cartesian_to_polar(spatial_coords, reference_coords)

    r_geo = None
    if use_graph_distances or check_spatial_distortion:
        if graph_distances_key not in subset_data.obsp:
            if use_graph_distances:
                msg = (
                    f"Graph '{graph_distances_key}' not found "
                    "but use_graph_distances is True"
                )
                raise ValueError(
                    msg,
                )
            warnings.warn(
                f"Graph '{graph_distances_key}' not found; skipping distortion check",
                stacklevel=2,
            )
            check_spatial_distortion = False
        else:
            adj = subset_data.obsp[graph_distances_key]
            r_geo = compute_geodesic_distances(adj, reference_index)

    r = cast("np.ndarray", r_geo if use_graph_distances else r_eucl)

    if check_spatial_distortion and r_geo is not None:
        r_min_dist = np.quantile(r_eucl, 0.1)
        r_max_dist = np.quantile(r_eucl, 0.9)

        dist_score = compute_local_distortion(r_geo, r_eucl, r_min_dist, r_max_dist)
        validate_embedding_distortion(
            dist_score,
            min_correlation=min_distortion_correlation,
        )

        provenance["distortion_score"] = dist_score

    r_min = np.quantile(r, inner_radius_percentile)
    r_max = np.quantile(r, outer_radius_percentile)

    annulus_mask = (r >= r_min) & (r < r_max)
    n_annulus = annulus_mask.sum()

    validate_annulus_cells(n_annulus, min_annulus=min_cells_in_analysis_region)

    provenance["r_min"] = r_min
    provenance["r_max"] = r_max
    provenance["r_min_quantile"] = inner_radius_percentile
    provenance["r_max_quantile"] = outer_radius_percentile
    provenance["n_annulus"] = n_annulus

    grid_points = define_angular_grid(angle_resolution_degrees)
    bin_map = bin_cells_sparse(theta, grid_points, r=r, r_min=r_min, r_max=r_max)
    bg_weights = np.ones(len(r))

    provenance["n_angular_bins"] = len(grid_points)
    provenance["delta_phi_deg"] = angle_resolution_degrees
    provenance["window_width_deg"] = smoothing_window_degrees

    covariates = build_covariate_matrix(
        subset_data,
        keys=confounding_factors,
        include_log_depth=include_log_depth,
    )

    provenance["covariate_keys"] = confounding_factors
    provenance["covariate_dim"] = covariates.shape[1] if covariates is not None else 0

    logger.info(
        "Building %s gene-independent permutation indices using '%s' engine...",
        num_permutations,
        permutation_method,
    )

    if permutation_method == "stratified":
        if stratification_column is not None:
            if stratification_column not in subset_data.obs:
                msg = f"Stratify key '{stratification_column}' not in subset_data.obs"
                raise ValueError(
                    msg,
                )
            strata = subset_data.obs[stratification_column].astype(str).to_numpy()
        else:
            strata_cols = [
                subset_data.obs[key].astype(str)
                for key in confounding_factors or []
                if (
                    key in subset_data.obs
                    and subset_data.obs[key].dtype.name in ["category", "object"]
                )
            ]

            if len(strata_cols) == 0:
                msg = (
                    "No discrete covariates found for stratified CRT. "
                    "Provide stratification_column or use permutation_method='knn'."
                )
                raise ValueError(
                    msg,
                )

            strata = pd.DataFrame(strata_cols).T.agg("_".join, axis=1).to_numpy()

        validate_strata_sizes(strata, min_stratum_size=min_group_size)

        perm_indices, strata_info = _build_permutation_indices_stratified(
            strata,
            num_permutations,
            min_stratum_size=min_group_size,
            seed=random_seed,
        )

        provenance["crt_strata_info"] = strata_info

    elif permutation_method == "knn":
        if covariates is None:
            msg = "kNN CRT requires covariates"
            raise ValueError(msg)

        perm_indices, knn_info = _build_permutation_indices_knn(
            covariates,
            num_permutations,
            k=neighbors_for_matching,
            stratify_by=(
                subset_data.obs[stratification_column].to_numpy()
                if stratification_column
                else None
            ),
            min_stratum_size=min_group_size,
            seed=random_seed,
        )

        provenance["crt_knn_info"] = knn_info

    else:
        msg = (
            f"Unknown permutation_method: {permutation_method}. "
            "Use 'stratified' or 'knn'."
        )
        raise ValueError(
            msg,
        )

    logger.info("Permutation indices built: shape %s", perm_indices.shape)

    def _process_single_gene(
        gene: str,
    ) -> tuple[dict[str, Any] | None, np.ndarray | None]:
        try:
            weights = compute_gene_weights(
                subset_data,
                gene,
                method=expression_method,
                q=expression_threshold_percentile,
                layer=expression_layer,
            )

            validate_effective_mass(
                weights,
                min_effective_mass=min_expression_mass,
                gene_name=gene,
            )

            perm_res = run_conditional_permutation(
                theta,
                r,
                weights,
                bg_weights,
                grid_points,
                smoothing_window_degrees,
                r_min,
                r_max,
                perm_indices=perm_indices,
                bin_map=bin_map,
            )

            row = {
                "gene": gene,
                "ARIA": perm_res["w1_obs"],
                "Z_W1": perm_res["Z_w1"],
                "D_dir": perm_res["d_dir_obs"],
                "theta_dir": perm_res["theta_dir_obs"],
                "p_CRA": perm_res["p_w1"],
                "p_D_dir": perm_res["p_d_dir"],
                "N_F": np.sum(weights[annulus_mask]),
                "null_mean_W1": perm_res["null_mean_w1"],
                "null_std_W1": perm_res["null_std_w1"],
            }

            return row, perm_res["rsp_obs"]
        except BioRSPValidationError as e:
            warnings.warn(f"Gene {gene} failed validation: {e}", stacklevel=2)
            return None, None
        except (ValueError, KeyError) as exc:
            logger.exception("Error processing gene %s", gene)
            warnings.warn(f"Error processing gene {gene}: {exc}", stacklevel=2)
            return None, None

    results_list = []
    rsp_curves_dict = {}

    for gene in tqdm(genes_to_test, desc="Scanning genes"):
        row, rsp_curve = _process_single_gene(gene)
        if row is not None:
            results_list.append(row)
            rsp_curves_dict[gene] = rsp_curve

    if len(results_list) == 0:
        msg = "No genes passed validation. Check expression levels and filters."
        raise ValueError(
            msg,
        )

    df_results = pd.DataFrame(results_list).set_index("gene")

    _, qvalues = compute_fdr_bh(
        df_results["p_CRA"].to_numpy(),
        alpha=significance_threshold,
    )
    df_results["q_CRA"] = qvalues
    df_results["discovery"] = qvalues < significance_threshold

    provenance["n_genes_tested"] = len(df_results)
    provenance["n_discoveries"] = df_results["discovery"].sum()
    provenance["fdr_alpha"] = significance_threshold

    for col in df_results.columns:
        spatial_data.var.loc[df_results.index, f"biorsp_{col}"] = df_results[col]

    if "biorsp" not in spatial_data.uns:
        spatial_data.uns["biorsp"] = {}

    spatial_data.uns["biorsp"]["results"] = df_results
    spatial_data.uns["biorsp"]["provenance"] = provenance
    spatial_data.uns["biorsp"]["perm_indices_shape"] = perm_indices.shape

    if rsp_curves_dict:
        spatial_data.uns["biorsp"]["rsp_curves"] = pd.DataFrame(
            rsp_curves_dict,
            index=grid_points,
        )

    logger.info(
        "BioRSP complete: Tested genes=%s; Discoveries (q<%s)=%s",
        len(df_results),
        significance_threshold,
        df_results["discovery"].sum(),
    )
    logger.info(
        "Results stored in spatial_data.var['biorsp_*'] and spatial_data.uns['biorsp']",
    )

    return df_results


def analyze_single_gene(
    spatial_data: AnnData,
    gene_name: str,
    reference_point: int | Sequence[float] | np.ndarray,
    **analysis_settings: object,
) -> pd.Series:
    """Analyze spatial pattern for one gene.

    This is a convenience function for testing individual genes. For analyzing
    multiple genes, use find_spatially_patterned_genes instead.

    Args:
        spatial_data: AnnData object containing spatial coordinates and gene expression.
        gene_name: Name of the gene to analyze.
        reference_point: Center point coordinates (array) or cell index (integer).
        **analysis_settings: Additional settings passed to
            find_spatially_patterned_genes.

    Returns:
        Analysis results for the single gene as a pandas Series.

    Raises:
        ValueError: If the gene is not found or analysis fails.

    Examples:
        result = analyze_single_gene(data, "Gene_X", center_coords)
        logger.info("Pattern strength: %.3f", result['ARIA'])
        logger.info("Significant: %s", result['discovery'])

    """
    results: pd.DataFrame = find_spatially_patterned_genes(
        spatial_data,
        [gene_name],
        reference_point,
        **analysis_settings,
    )
    return results.loc[gene_name]
