"""BioRSP: Radial Spatial Patterning Analysis for Single-Cell Data.

This package implements robust, reproducible directional expression gradient
analysis for single-cell embeddings. It includes methods for inference with
confounder control, stability checks across embeddings, and reproducibility
tracking.
"""

from .api import (
    analyze_single_gene,
    define_reference_point,
    find_spatially_patterned_genes,
    get_code_version,
)
from .interpretation import generate_interpretation_report, identify_peak_sectors
from .stability import (
    comprehensive_stability_check,
    consistency_across_embeddings,
    measure_angular_variation,
    parameter_robustness,
    validate_spatial_representations,
)
from .validation import (
    BioRSPValidationError,
    compute_fdr_bh,
    validate_covariates,
    validate_effective_mass,
    validate_minimum_cells,
)

__version__ = "3.0.0-dev"

__all__ = [
    "BioRSPValidationError",
    "analyze_single_gene",
    "comprehensive_stability_check",
    "compute_fdr_bh",
    "consistency_across_embeddings",
    "define_reference_point",
    "find_spatially_patterned_genes",
    "generate_interpretation_report",
    "get_code_version",
    "identify_peak_sectors",
    "measure_angular_variation",
    "parameter_robustness",
    "validate_covariates",
    "validate_effective_mass",
    "validate_minimum_cells",
    "validate_spatial_representations",
]
