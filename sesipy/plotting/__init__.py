"""Plotting module for Sesipy visualizations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plot2D import Plot2D
    from .plot3D import Plot3D

__all__ = [
    "Plot3D",
    "Plot2D",
]


def __getattr__(name):
    if name == "Plot2D":
        from .plot2D import Plot2D

        return Plot2D

    if name == "Plot3D":
        from .plot3D import Plot3D

        return Plot3D

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
