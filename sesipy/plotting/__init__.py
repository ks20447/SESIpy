"""Plotting module for Sesipy visualizations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plot2D import (
        Plot2D,
        mpl_font_size,
        mpl_use_cmap,
        mpl_use_latex,
        mpl_use_seaborn,
    )
    from .plot3D import Plot3D

__all__ = [
    "Plot2D",
    "Plot3D",
    "mpl_use_latex",
    "mpl_font_size",
    "mpl_use_seaborn",
    "mpl_use_cmap",
]

_MODULES = {
    "Plot2D": ".plot2D",
    "Plot3D": ".plot3D",
    "mpl_use_latex": ".plot2D",
    "mpl_font_size": ".plot2D",
    "mpl_use_seaborn": ".plot2D",
    "mpl_use_cmap": ".plot2D",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
