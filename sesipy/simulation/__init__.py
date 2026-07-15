"""Simulation module for Sesipy."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worlds import World, Indoor, Outdoor, Obstacle

__all__ = [
    "World",
    "Indoor",
    "Outdoor",
    "Obstacle",
]


def __getattr__(name):
    if name in {"World", "Indoor", "Outdoor", "Obstacle"}:
        from .worlds import World, Indoor, Outdoor, Obstacle

        return {
            "World": World,
            "Indoor": Indoor,
            "Outdoor": Outdoor,
            "Obstacle": Obstacle,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
