"""Plotting module for Sesipy visualizations."""

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
