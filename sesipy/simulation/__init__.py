"""Simulation module for Sesipy."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worlds import World, Indoor, Outdoor, Obstacle, WorldBuilder, WorldDescriptor

__all__ = [
    "World",
    "Indoor",
    "Outdoor",
    "Obstacle",
]


def __getattr__(name):
    if name in {"World", "Indoor", "Outdoor", "Obstacle", "WorldBuilder", "WorldDescriptor"}:
        from .worlds import World, Indoor, Outdoor, Obstacle, WorldBuilder, WorldDescriptor

        return {
            "World": World,
            "Indoor": Indoor,
            "Outdoor": Outdoor,
            "Obstacle": Obstacle,
            "WorldBuilder": WorldBuilder,
            "WorldDescriptor": WorldDescriptor,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
