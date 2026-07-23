"""Simulation module for Sesipy."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worlds import (
        Indoor,
        Obstacle,
        Outdoor,
        Rectangle,
        Wall,
        World,
        WorldBuilder,
        WorldDescriptor,
    )

__all__ = [
    "World",
    "WorldBuilder",
    "WorldDescriptor",
    "Indoor",
    "Outdoor",
    "Obstacle",
    "Wall",
    "Rectangle",
]


def __getattr__(name):
    if name in {
        "World",
        "Indoor",
        "Outdoor",
        "Obstacle",
        "WorldBuilder",
        "WorldDescriptor",
        "Wall",
        "Rectangle",
    }:
        from .worlds import (
            World,
            Indoor,
            Outdoor,
            Obstacle,
            WorldBuilder,
            WorldDescriptor,
            Wall,
            Rectangle,
        )

        return {
            "World": World,
            "Indoor": Indoor,
            "Outdoor": Outdoor,
            "Obstacle": Obstacle,
            "WorldBuilder": WorldBuilder,
            "WorldDescriptor": WorldDescriptor,
            "Wall": Wall,
            "Rectangle": Rectangle,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
