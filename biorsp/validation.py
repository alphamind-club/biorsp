"""Validation helpers and checks for BioRSP analysis.

This module exposes small validators and utility functions that enforce
analysis preconditions and report helpful messages for users.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


MIN_VALID_STRATA = 1


class BioRSPValidationError(Exception):
    """Exception raised when a BioRSP validation check fails."""


def validate_minimum_cells(
    n_cells: int,
    min_cells: int = 200,
    exploratory_min: int = 50,
    *,
    allow_exploratory: bool = False,
) -> None:
    """Enforce a minimum number of cells for valid analysis.

    This function raises a BioRSPValidationError if the cell count is too low
    to perform reliable conditional inference. It may also warn when
    the count is below publication thresholds but exploratory mode is allowed.
    """
    if n_cells < exploratory_min:
        msg = (
            f"Too few cells ({n_cells}). Minimum required: {exploratory_min} "
            f"(exploratory) or {min_cells} (publication-grade). "
            f"BioRSP requires sufficient cells for valid conditional inference."
        )
        raise BioRSPValidationError(msg)

    if n_cells < min_cells:
        if allow_exploratory:
            cell_msg = (
                f"Cell count ({n_cells}) below publication threshold "
                f"({min_cells}). "
            )
            warnings.warn(
                (
                    cell_msg + "Results are EXPLORATORY and may not be calibrated. "
                    "Use with caution and report as uncalibrated."
                ),
                UserWarning,
                stacklevel=2,
            )
        else:
            msg = (
                f"Cell count ({n_cells}) below threshold ({min_cells}). "
                f"Set allow_exploratory=True to proceed with uncalibrated results."
            )
            raise BioRSPValidationError(msg)


def validate_effective_mass(
    weights: np.ndarray,
    min_effective_mass: float = 10.0,
    gene_name: str = "unknown",
) -> None:
    """Ensure a gene has sufficient aggregate expression mass for analysis.

    Parameters
    ----------
    weights : np.ndarray
        Continuous expression weights in [0, 1].
    min_effective_mass : float
        Minimum sum of weights required for reliable inference.
    gene_name : str
        Gene identifier used in error messages.

    Raises
    ------
    BioRSPValidationError
        If the effective mass is below the specified threshold.

    """
    eff_mass = np.sum(weights)

    if eff_mass < min_effective_mass:
        msg = (
            f"Gene {gene_name} has insufficient effective mass ({eff_mass:.2f} < "
            f"{min_effective_mass}). "
            "Expression too sparse for reliable directional inference. "
            "Consider filtering low-expressed genes before analysis."
        )
        raise BioRSPValidationError(msg)


def validate_annulus_cells(n_annulus: int, min_annulus: int = 100) -> None:
    """Ensure the analysis annulus contains sufficient cells for angular statistics.

    Parameters
    ----------
    n_annulus : int
        Number of cells in the analysis annulus.
    min_annulus : int
        Minimum accepted number of cells.

    Raises
    ------
    BioRSPValidationError
        If annulus size is below the specified threshold.

    """
    if n_annulus < min_annulus:
        msg = (
            f"Annulus contains too few cells ({n_annulus} < {min_annulus}). "
            f"Adjust annulus boundaries (r_min/r_max) or use a different vantage. "
            f"Angular statistics require sufficient cells for calibration."
        )
        raise BioRSPValidationError(msg)


def validate_strata_sizes(
    strata: np.ndarray,
    min_stratum_size: int = 20,
) -> tuple[bool, list[str]]:
    """Validate that strata sizes meet a minimum threshold and merge small strata.

    Returns a tuple ``(valid, small_strata)`` where ``valid`` indicates whether
    all strata meet the minimum size and ``small_strata`` lists the labels
    deemed too small.
    """
    unique, counts = np.unique(strata, return_counts=True)
    small_mask = counts < min_stratum_size
    small_strata = list(unique[small_mask])

    if len(small_strata) > 0:
        n_small_cells = np.sum(counts[small_mask])
        warnings.warn(
            f"{len(small_strata)} strata have < {min_stratum_size} cells "
            f"({n_small_cells} cells total). "
            "These will be merged into '__small__' stratum. "
            "Consider coarser stratification if this affects many cells.",
            UserWarning,
            stacklevel=2,
        )

    n_valid_strata = np.sum(~small_mask) + (1 if len(small_strata) > 0 else 0)
    if n_valid_strata < MIN_VALID_STRATA:
        msg = (
            f"After merging small strata, only {n_valid_strata} stratum remains. "
            f"Conditional inference requires at least 2 strata. "
            f"Use coarser stratification or switch to kNN CRT."
        )
        raise BioRSPValidationError(msg)

    return len(small_strata) == 0, small_strata


def validate_covariates(
    adata: AnnData,
    covariate_keys: list[str] | None,
    *,
    require_sample: bool = True,
    require_batch: bool = True,
    require_depth: bool = True,
    allow_uncalibrated: bool = False,
) -> None:
    """Enforce mandatory covariate set for confounder-robust inference.

    At minimum: sample, batch, log-depth should be controlled.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    covariate_keys : list or None
        User-provided covariate keys.
    require_sample : bool
        Require 'sample' or 'sample_id' in covariates.
    require_batch : bool
        Require 'batch' in covariates.
    require_depth : bool
        Require 'n_counts' or 'total_counts' for depth control.
    allow_uncalibrated : bool
        If True, allow analysis without required covariates (with strong warning).

    Raises
    ------
    BioRSPValidationError
        If required covariates missing and allow_uncalibrated=False.

    """
    if covariate_keys is None:
        covariate_keys = []

    missing = []

    if require_sample:
        sample_keys = [
            "sample",
            "sample_id",
            "donor",
            "donor_id",
            "patient",
            "library_id",
        ]
        if not any(k in adata.obs for k in sample_keys) and not any(
            k in covariate_keys for k in sample_keys
        ):
            missing.append("sample (or sample_id/donor/patient)")

    if require_batch:
        batch_keys = ["batch", "batch_id", "library", "library_id", "experiment_id"]
        if not any(k in adata.obs for k in batch_keys) and not any(
            k in covariate_keys for k in batch_keys
        ):
            missing.append("batch")

    if require_depth:
        depth_keys = ["n_counts", "total_counts", "nCount_RNA"]
        if not any(k in adata.obs for k in depth_keys):
            missing.append("n_counts (or total_counts)")

    if len(missing) > 0:
        msg = (
            f"Missing required covariates: {', '.join(missing)}. "
            f"Confounder-robust inference requires controlling for: "
            f"sample, batch, and sequencing depth. "
            f"Without these, results are EXPLORATORY and may be driven by technical "
            f"artifacts. "
        )

        if allow_uncalibrated:
            warnings.warn(
                msg + "Proceeding in UNCALIBRATED mode. DO NOT use for "
                "publication claims.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise BioRSPValidationError(
                msg + "Set allow_uncalibrated=True to proceed (not recommended).",
            )


def validate_vantage_stability(
    vss: float,
    embedding_diameter: float,
    threshold_frac: float = 0.05,
) -> None:
    """Enforce vantage stability via bootstrap VSS.

    Parameters
    ----------
    vss : float
        Vantage stability score (median pairwise distance of bootstrap vantages).
    embedding_diameter : float
        Diameter of embedding (max distance from center).
    threshold_frac : float
        VSS must be < threshold_frac * diameter (default 0.05 = 5%).

    Raises
    ------
    BioRSPValidationError
        If vantage is unstable.

    """
    threshold = threshold_frac * embedding_diameter

    if vss > threshold:
        msg = (
            f"Vantage instability detected: VSS = {vss:.4f} > {threshold:.4f} "
            f"({threshold_frac*100:.1f}% of embedding diameter). "
            f"The vantage point moves substantially under bootstrap resampling, "
            f"indicating unreliable focal geometry. "
            f"Solutions: (1) choose a different vantage, (2) subset to a more "
            f"homogeneous cell population, or (3) use a different embedding."
        )
        raise BioRSPValidationError(msg)


def validate_embedding_distortion(
    distortion_score: float,
    min_correlation: float = 0.5,
) -> None:
    """Enforce embedding quality via graph-vs-embedding distance correlation.

    Parameters
    ----------
    distortion_score : float
        Pearson correlation between graph geodesic and embedding Euclidean distances.
    min_correlation : float
        Minimum acceptable correlation (default 0.5).

    Raises
    ------
    BioRSPValidationError
        If embedding is too distorted.

    """
    if np.isnan(distortion_score):
        warnings.warn(
            (
                "Could not compute distortion score (insufficient cells in annulus). "
                "Proceeding without distortion check."
            ),
            UserWarning,
            stacklevel=2,
        )
        return

    if distortion_score < min_correlation:
        msg = (
            f"Embedding distortion detected: correlation = {distortion_score:.3f} < "
            f"{min_correlation}. "
            "The embedding does not preserve local graph structure around the vantage. "
            "Angular statistics may reflect embedding artifacts, not biology. "
            "Solutions: (1) use a different embedding (e.g., PCA instead of UMAP), "
            "(2) choose a vantage in a less distorted region, or "
            "(3) use graph-geodesic distances (use_graph=True)."
        )
        raise BioRSPValidationError(msg)


def compute_fdr_bh(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Parameters
    ----------
    pvalues : np.ndarray
        P-values for multiple tests.
    alpha : float
        Target FDR level.

    Returns
    -------
    reject : np.ndarray (bool)
        True for rejected hypotheses (discoveries).
    qvalues : np.ndarray
        Adjusted q-values.

    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sort_idx = np.argsort(pvalues)
    sorted_p = pvalues[sort_idx]

    thresholds = alpha * np.arange(1, n + 1) / n
    reject_sorted = sorted_p <= thresholds

    if np.any(reject_sorted):
        max_idx = np.where(reject_sorted)[0][-1]
        reject_sorted[: max_idx + 1] = True

    reject = np.zeros(n, dtype=bool)
    reject[sort_idx] = reject_sorted

    qvalues = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    qvalues = np.minimum(qvalues, 1.0)

    qvalues_unsorted = np.zeros(n)
    qvalues_unsorted[sort_idx] = qvalues

    return reject, qvalues_unsorted


def compute_fdr_by(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Yekutieli FDR control (safe under dependence).

    More conservative than BH; use when testing across multiple conditions/vantages.

    Parameters
    ----------
    pvalues : np.ndarray
        P-values.
    alpha : float
        Target FDR.

    Returns
    -------
    reject : np.ndarray (bool)
        Discoveries.
    qvalues : np.ndarray
        Adjusted q-values.

    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    c_n = np.sum(1.0 / np.arange(1, n + 1))

    alpha_by = alpha / c_n

    return compute_fdr_bh(pvalues, alpha=alpha_by)
