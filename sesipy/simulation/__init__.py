"""Simulation module for Sesipy."""

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
