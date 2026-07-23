"""Mapping module for spatial engines."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import Environment, Sampler2D, Sampler3D
    from .utils import (
        clean_lidar,
        extract_lidar_metadata,
        map_yaml_to_polygon,
        mesh_error,
        simulate_lidar,
    )

__all__ = [
    "Environment",
    "Sampler2D",
    "Sampler3D",
    "map_yaml_to_polygon",
    "simulate_lidar",
    "clean_lidar",
    "extract_lidar_metadata",
    "mesh_error",
]

_MODULES = {
    "Environment": ".environment",
    "Sampler2D": ".environment",
    "Sampler3D": ".environment",
    "map_yaml_to_polygon": ".utils",
    "simulate_lidar": ".utils",
    "clean_lidar": ".utils",
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
