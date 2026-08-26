"""Evaluation utilities for Sesipy."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .localization import sample_surface, scoring_surface
    from .signals import (
        compare_power_distributions,
        neighborhood_adjusted_correlation,
        normalize_metrics,
        rank_power_distributions,
    )

__all__ = [
    "scoring_surface",
    "sample_surface",
    "neighborhood_adjusted_correlation",
    "normalize_metrics",
    "compare_power_distributions",
    "rank_power_distributions",
]

_MODULES = {
    "scoring_surface": ".localization",
    "sample_surface": ".localization",
    "neighborhood_adjusted_correlation": ".signals",
    "normalize_metrics": ".signals",
    "compare_power_distributions": ".signals",
    "rank_power_distributions": ".signals",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
