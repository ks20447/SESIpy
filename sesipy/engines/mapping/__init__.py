"""Mapping module for spatial engines."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import Environment, Sampler2D, Sampler3D
    from .utils import (
        cluster_pointcloud,
        extract_lidar_metadata,
        map_yaml_to_polygon,
        mesh_error,
        remove_boundary_points,
        remove_small_holes,
        simulate_lidar,
    )

__all__ = [
    "Environment",
    "Sampler2D",
    "Sampler3D",
    "map_yaml_to_polygon",
    "simulate_lidar",
    "cluster_pointcloud",
    "remove_small_holes",
    "remove_boundary_points",
    "extract_lidar_metadata",
    "mesh_error",
]

_MODULES = {
    "Environment": ".environment",
    "Sampler2D": ".environment",
    "Sampler3D": ".environment",
    "map_yaml_to_polygon": ".utils",
    "simulate_lidar": ".utils",
    "cluster_pointcloud": ".utils",
    "remove_small_holes": ".utils",
    "remove_boundary_points": ".utils",
    "extract_lidar_metadata": ".utils",
    "mesh_error": ".utils",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
