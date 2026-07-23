"""World building module for Sesipy simulations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import (
        Obstacle,
        create_box_mesh,
        create_grid_box_mesh,
        create_grid_plane_mesh,
        create_plane_mesh,
        plane_parameters,
        rotation_matrix_z,
    )
    from .world_builder import Rectangle, Wall, World, WorldBuilder, WorldDescriptor
    from .worlds import Indoor, Outdoor

__all__ = [
    "World",
    "WorldBuilder",
    "WorldDescriptor",
    "Wall",
    "Rectangle",
    "Indoor",
    "Outdoor",
    "Obstacle",
    "rotation_matrix_z",
    "create_plane_mesh",
    "plane_parameters",
    "create_grid_plane_mesh",
    "create_box_mesh",
    "create_grid_box_mesh",
]

_MODULES = {
    "World": ".world_builder",
    "WorldBuilder": ".world_builder",
    "WorldDescriptor": ".world_builder",
    "Wall": ".world_builder",
    "Rectangle": ".world_builder",
    "Indoor": ".worlds",
    "Outdoor": ".worlds",
    "Obstacle": ".utils",
    "rotation_matrix_z": ".utils",
    "create_plane_mesh": ".utils",
    "plane_parameters": ".utils",
    "create_grid_plane_mesh": ".utils",
    "create_box_mesh": ".utils",
    "create_grid_box_mesh": ".utils",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
