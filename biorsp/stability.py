"""Stability and parameter robustness utilities for BioRSP.

These functions test whether spatially patterned genes are reproducible
across different coordinate systems and analysis parameters.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

from scipy.stats import circstd, spearmanr

from .api import define_reference_point, find_spatially_patterned_genes
from .validation import BioRSPValidationError

logger = logging.getLogger(__name__)

MIN_SUCCESS_CONFIGS = 2
MIN_VALID_SYSTEMS = 2
MIN_SAMPLES_FOR_STATS = 2
EPSILON = 1e-9


def measure_angular_variation(angles_radians: np.ndarray) -> float:
    """Compute a circular standard deviation for angles in radians.

    Returns
    -------
    float
        Circular standard deviation (radians) or ``np.nan`` when insufficient data.

    """
    if len(angles_radians) < MIN_SAMPLES_FOR_STATS:
        return np.nan

    return circstd(angles_radians)


def consistency_across_embeddings(
    spatial_data: AnnData,
    genes_to_test: list[str],
    coordinate_systems: list[str],
    reference_method: str = "geometric_median",
    min_consistency_score: float = 0.7,
    max_variation_threshold: float = 0.3,
    max_directional_spread: float = np.pi / 4,
    analysis_settings: dict | None = None,
) -> pd.DataFrame:
    """Check if gene patterns are consistent across different coordinate systems.

    This function tests the same genes using different ways of representing spatial
    relationships (like UMAP, t-SNE, PCA coordinates). Genes that show similar
    patterns in multiple coordinate systems are more likely to represent real
    biological signals rather than artifacts of the visualization method.

    A gene passes the stability test if:
    1. Pattern strength is similar across coordinate systems.
    2. Pattern strength doesn't vary too much across systems.
    3. Pattern direction is focused and not randomly spread.

    Args:
        spatial_data: AnnData object with gene expression and spatial coordinates.
        genes_to_test: List of gene names to evaluate for stability.
        coordinate_systems: List of coordinate system names in spatial_data.obsm.
            Need at least 2 coordinate systems.
        reference_method: Method for choosing center point.
            Same across all coordinate systems.
        min_consistency_score: Minimum similarity score required.
            (0.7 = 70% similar)
        max_variation_threshold: Maximum allowed variation in pattern strength.
        max_directional_spread: Maximum spread in pattern direction (π/4 = 45°).
        analysis_settings: Additional settings passed to find_spatially_patterned_genes.

    Returns:
        pd.DataFrame: Stability results for each gene with columns:
            - pattern_strength_mean
            - pattern_strength_std
            - pattern_strength_cv (variation)
            - directional_spread (focus of pattern direction)
            - consistency_score (similarity across coordinate systems)
            - passes_stability_test (overall result)

    """
    if len(coordinate_systems) < MIN_VALID_SYSTEMS:
        msg = (
            f"Multi-embedding stability requires at least {MIN_VALID_SYSTEMS} "
            "coordinate systems"
        )
        raise ValueError(msg)

    for coord_sys in coordinate_systems:
        if coord_sys not in spatial_data.obsm:
            msg = f"Coordinate system '{coord_sys}' not found in spatial_data.obsm"
            raise ValueError(msg)

    results_by_system = {}

    for coord_sys in coordinate_systems:
        logger.info("Testing coordinate system: %s", coord_sys)

        try:
            reference_point = define_reference_point(
                spatial_data,
                coordinate_system=coord_sys,
                method=reference_method,
            )

            res = find_spatially_patterned_genes(
                spatial_data,
                genes_to_test,
                reference_point=reference_point,
                coordinate_system=coord_sys,
                num_permutations=0,
                **(analysis_settings or {}),
            )

            results_by_system[coord_sys] = res

        except (BioRSPValidationError, ValueError, KeyError):
            logger.exception(
                "Validation/processing error for coordinate system %s",
                coord_sys,
            )
            results_by_system[coord_sys] = None

    strength_cols = {
        k: v["ARIA"] for k, v in results_by_system.items() if v is not None
    }
    strength_df = pd.DataFrame(strength_cols)

    dir_cols = {
        k: v["theta_dir"] for k, v in results_by_system.items() if v is not None
    }
    dir_df = pd.DataFrame(dir_cols)

    rows = []
    for gene in strength_df.index:
        s_vals = strength_df.loc[gene].dropna().to_numpy()
        d_vals = dir_df.loc[gene].dropna().to_numpy()

        strength_mean = np.mean(s_vals)
        strength_std = np.std(s_vals)
        strength_cv = strength_std / (abs(strength_mean) + EPSILON)

        directional_spread = circstd(d_vals) if len(d_vals) > 1 else np.nan

        consistency_score = 1.0 - strength_cv

        passes = (
            (strength_cv < max_variation_threshold)
            and (
                (directional_spread < max_directional_spread)
                if not np.isnan(directional_spread)
                else True
            )
            and (consistency_score >= min_consistency_score)
        )

        rows.append(
            {
                "gene": gene,
                "pattern_strength_mean": strength_mean,
                "pattern_strength_std": strength_std,
                "pattern_strength_cv": strength_cv,
                "directional_spread": directional_spread,
                "consistency_score": consistency_score,
                "passes_stability_test": passes,
            },
        )

    return pd.DataFrame(rows).set_index("gene")


def parameter_robustness(
    spatial_data: AnnData,
    genes_to_test: list[str],
    reference_point: int | Sequence[float] | np.ndarray,
    coordinate_system: str,
    angular_resolution_grid: list[float] | None = None,
    smoothing_window_grid: list[float] | None = None,
    radius_range_grid: list[tuple] | None = None,
    neighborhood_size_grid: list[int] | None = None,
    max_variation_threshold: float = 0.3,
    analysis_settings: dict | None = None,
) -> pd.DataFrame:
    """Test how sensitive gene pattern detection is to different parameter choices.

    This function helps ensure that discovered patterns are robust and not just
    artifacts of specific parameter settings. It tests the same genes with different
    combinations of analysis parameters and checks if the results are consistent.

    A gene passes the robustness test if its pattern strength doesn't vary too much
    when we change the analysis settings (like smoothing amount or neighborhood size).

    Args:
        spatial_data: AnnData object with gene expression and spatial coordinates.
        genes_to_test: List of gene names to evaluate for parameter sensitivity.
        reference_point: Reference point coordinates or index for radial analysis.
        coordinate_system: Name of coordinate system in spatial_data.obsm.
        angular_resolution_grid: Angular bin widths in degrees to test.
            Defaults to [3, 5, 10].
        smoothing_window_grid: Smoothing window sizes to test.
            Defaults to [20, 30, 45].
        radius_range_grid: Annulus radius ranges as quantiles.
            Defaults: [(0.1, 0.9), (0.2, 0.8)].
        neighborhood_size_grid: Number of neighbors for conditional testing.
            Defaults to e.g. [20, 30, 50].
        max_variation_threshold: Maximum allowed variation (0.3 = 30%).
        analysis_settings: Additional fixed settings passed to the API.

    Returns:
        pd.DataFrame: Parameter sensitivity results for each gene with columns:
            - pattern_strength_mean
            - pattern_strength_std
            - pattern_strength_cv (variation)
            - passes_robustness_test (overall result)

    """
    if angular_resolution_grid is None:
        angular_resolution_grid = [3.0, 5.0, 10.0]
    if smoothing_window_grid is None:
        smoothing_window_grid = [20.0, 30.0, 45.0]
    if radius_range_grid is None:
        radius_range_grid = [(0.1, 0.9), (0.2, 0.8)]
    if neighborhood_size_grid is None:
        neighborhood_size_grid = [20, 30, 50]

    parameter_combinations = [
        {
            "angular_resolution_deg": ar,
            "smoothing_window_deg": sw,
            "min_radius_quantile": r[0],
            "max_radius_quantile": r[1],
            "neighborhood_size": nn,
        }
        for ar, sw, r, nn in itertools.product(
            angular_resolution_grid,
            smoothing_window_grid,
            radius_range_grid,
            neighborhood_size_grid,
        )
    ]

    logger.info("Testing %d parameter combinations...", len(parameter_combinations))

    results_by_config = []

    for i, config in enumerate(parameter_combinations):
        logger.info("Config %d/%d: %s", i + 1, len(parameter_combinations), config)

        try:
            res = find_spatially_patterned_genes(
                spatial_data,
                genes_to_test,
                reference_point=reference_point,
                coordinate_system=coordinate_system,
                num_permutations=0,
                **config,
                **(analysis_settings or {}),
            )
            results_by_config.append(res)
        except BioRSPValidationError as exc:
            msg = f"Parameter config failed for gene set: {exc}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        except ValueError:
            logger.exception("ValueError running parameter config")
            continue
        except (RuntimeError, KeyError, TypeError):
            logger.exception("Unexpected error in parameter grid")
            continue

    if len(results_by_config) < MIN_SUCCESS_CONFIGS:
        msg = "Parameter robustness check failed: too few configurations succeeded"
        raise ValueError(msg)

    sensitivity_rows = []

    for gene in genes_to_test:
        pattern_strength_vals = []
        direction_vals = []

        for res in results_by_config:
            if gene in res.index:
                pattern_strength_vals.append(res.loc[gene, "ARIA"])
                direction_vals.append(res.loc[gene, "theta_dir"])

        if len(pattern_strength_vals) < MIN_SAMPLES_FOR_STATS:
            continue

        strength_mean = np.mean(pattern_strength_vals)
        strength_std = np.std(pattern_strength_vals)
        strength_cv = strength_std / (abs(strength_mean) + EPSILON)

        direction_spread = measure_angular_variation(np.array(direction_vals))

        robust = strength_cv < max_variation_threshold

        sensitivity_rows.append(
            {
                "gene": gene,
                "pattern_strength_mean": strength_mean,
                "pattern_strength_cv": strength_cv,
                "directional_spread": direction_spread,
                "n_configs": len(pattern_strength_vals),
                "passes_robustness_test": robust,
            },
        )

    df_sensitivity = pd.DataFrame(sensitivity_rows).set_index("gene")

    logger.info(
        "Parameter robustness results: Robust=%d; Sensitive=%d",
        df_sensitivity["passes_robustness_test"].sum(),
        (~df_sensitivity["passes_robustness_test"]).sum(),
    )

    return df_sensitivity


def comprehensive_stability_check(
    spatial_data: AnnData,
    genes_to_test: list[str],
    coordinate_systems: list[str],
    reference_method: str = "geometric_median",
    *,
    run_parameter_sensitivity: bool = False,
    analysis_settings: dict | None = None,
) -> pd.DataFrame:
    """Complete stability analysis combining coordinate systems and.

    This function performs a thorough evaluation of gene pattern stability by
    testing consistency across different spatial representations and
    parameter settings.

    Args:
        spatial_data: AnnData object with gene expression and spatial coordinates.
        genes_to_test: List of gene names to evaluate for stability.
        coordinate_systems: List of coordinate system names in spatial_data.obsm.
            Need at least 2 coordinate systems.
        reference_method: Method for center point.
        run_parameter_sensitivity: If True, test sensitivity to parameters.
        analysis_settings: Additional settings passed to the API function.

    Returns:
        pd.DataFrame: Complete stability report. The final column
        'passes_all_stability_tests' contains a boolean per gene.

    """
    coord_stability = consistency_across_embeddings(
        spatial_data,
        genes_to_test,
        coordinate_systems,
        reference_method=reference_method,
        **(analysis_settings or {}),
    )

    if run_parameter_sensitivity:
        logger.info("Running parameter sensitivity analysis...")

        coord_sys = coordinate_systems[0]
        reference_point = define_reference_point(
            spatial_data,
            coordinate_system=coord_sys,
            method=reference_method,
        )

        param_stability = parameter_robustness(
            spatial_data,
            genes_to_test,
            reference_point,
            coord_sys,
            **(analysis_settings or {}),
        )

        combined = coord_stability.join(
            param_stability[["pattern_strength_cv", "passes_robustness_test"]],
            how="left",
            rsuffix="_parameter",
        )

        combined["passes_all_stability_tests"] = combined[
            "passes_stability_test"
        ] & combined["passes_robustness_test"].fillna(value=True)

    else:
        combined = coord_stability.copy()
        passes = combined["passes_stability_test"]
        combined["passes_all_stability_tests"] = passes

    total = len(combined)
    p_all = combined["passes_all_stability_tests"]
    high_conf = int(p_all.sum())
    low_conf = int((~p_all).sum())

    logger.info(
        "FINAL STABILITY ANALYSIS RESULTS: "
        "Total=%d; HighConfidence=%d; LowConfidence=%d",
        total,
        high_conf,
        low_conf,
    )

    return combined


def validate_spatial_representations(
    spatial_data: AnnData,
    genes_to_test: list[str],
    coordinate_systems: list[str] | None = None,
    reference_method: str = "geometric_median",
    num_permutations: int = 0,
) -> pd.DataFrame:
    """Check stability of results across multiple coordinate systems (legacy function).

    This is a legacy wrapper around
    `test_consistency_across_embeddings` for backward compatibility.

    Args:
        spatial_data: AnnData object with gene expression and spatial coordinates.
        genes_to_test: List of gene names to evaluate.
        coordinate_systems: List of coordinate system names in spatial_data.obsm.
        reference_method: Method for choosing center point.
        num_permutations: Number of permutations (use 0 for effect sizes only).

    Returns:
        pd.DataFrame: Stability metrics for each gene.

    """
    results_by_system = {}

    if coordinate_systems is None:
        coordinate_systems = ["X_umap", "X_pca"]

    for coord_sys in coordinate_systems:
        if coord_sys not in spatial_data.obsm:
            continue

        try:
            reference_point = define_reference_point(
                spatial_data,
                coordinate_system=coord_sys,
                method=reference_method,
            )
        except BioRSPValidationError as exc:
            msg = f"Coordinate system {coord_sys} skipped: {exc}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        except (ValueError, KeyError):
            logger.exception("Skipping coordinate system %s due to error", coord_sys)
            continue

        res: pd.DataFrame = find_spatially_patterned_genes(
            spatial_data,
            genes_to_test,
            reference_point=reference_point,
            coordinate_system=coord_sys,
            num_permutations=num_permutations,
        )
        results_by_system[coord_sys] = res

    if len(results_by_system) < MIN_VALID_SYSTEMS:
        msg = "Need at least 2 valid coordinate systems to check stability."
        raise ValueError(msg)

    stability_rows = []

    system_keys = list(results_by_system.keys())
    pattern_corrs = []

    for i in range(len(system_keys)):
        for j in range(i + 1, len(system_keys)):
            k1, k2 = system_keys[i], system_keys[j]
            df1, df2 = results_by_system[k1], results_by_system[k2]

            common_genes = df1.index.intersection(df2.index)
            if len(common_genes) < MIN_SAMPLES_FOR_STATS:
                continue

            corr, _ = spearmanr(
                df1.loc[common_genes, "pattern_strength"],
                df2.loc[common_genes, "pattern_strength"],
            )
            pattern_corrs.append(corr)

    for gene in genes_to_test:
        pattern_vals = [
            results_by_system[k].loc[gene, "pattern_strength"]
            for k in system_keys
            if gene in results_by_system[k].index
        ]

        if len(pattern_vals) < MIN_SAMPLES_FOR_STATS:
            continue

        pattern_mean: float = np.mean(pattern_vals)
        pattern_std: float = np.std(pattern_vals)
        pattern_cv: float = pattern_std / (pattern_mean + 1e-6)

        stability_rows.append(
            {
                "gene": gene,
                "pattern_strength_cv": pattern_cv,
                "mean_pattern_strength": pattern_mean,
            },
        )

    return pd.DataFrame(stability_rows).set_index("gene")
