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
    measure_angular_variation,
    test_consistency_across_embeddings,
    test_parameter_robustness,
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
    "find_spatially_patterned_genes",
    "define_reference_point",
    "analyze_single_gene",
    "get_code_version",
    "test_consistency_across_embeddings",
    "test_parameter_robustness",
    "comprehensive_stability_check",
    "measure_angular_variation",
    "validate_spatial_representations",
    "BioRSPValidationError",
    "compute_fdr_bh",
    "validate_minimum_cells",
    "validate_effective_mass",
    "validate_covariates",
    "generate_interpretation_report",
    "identify_peak_sectors",
]
