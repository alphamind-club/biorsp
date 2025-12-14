"""Utilities and datasets for experiments used in tests and analysis."""

from .synthetic_data import (
    add_gene_expression,
    create_synthetic_dataset,
    generate_manifold,
)

__all__ = [
    "add_gene_expression",
    "create_synthetic_dataset",
    "generate_manifold",
]
