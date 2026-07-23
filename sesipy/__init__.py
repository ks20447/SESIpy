"""Sesipy - Spatial Electromagnetic Simulation in Python."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import ArrayFactory, NormalFactory, Symbols, suppress_c_output
    from .data_storage import DatabaseReader, Database, DatabaseAoA, DatabasePS, decode, encode
    from .simulation import (
        World,
        Indoor,
        Outdoor,
        Obstacle,
        WorldBuilder,
        WorldDescriptor,
        Wall,
        Rectangle,
    )
    from .plotting import (
        Plot2D,
        Plot3D,
        mpl_font_size,
        mpl_use_cmap,
        mpl_use_latex,
        mpl_use_seaborn,
    )
    from .engines import (
        ReceiverArray,
        TransmitterArray,
        PointSource,
        IsotropicReceiver,
        Scene,
        scattering_power,
        to_dBm,
        smooth_point_data,
        threshold_point_data,
        create_mesh_copies,
        Environment,
        Sampler2D,
        Sampler3D,
        AoA,
        extract_aoa,
        aoa_projection_2D,
        multi_aoa_projection_2D,
        map_yaml_to_polygon,
        simulate_lidar,
        clean_lidar,
        extract_lidar_metadata,
        mesh_error,
    )

__all__ = [
    # Utils
    "ArrayFactory",
    "NormalFactory",
    "Symbols",
    "suppress_c_output",
    # Data storage
    "DatabaseReader",
    "Database",
    "DatabasePS",
    "DatabaseAoA",
    "encode",
    "decode",
    # Simulation
    "World",
    "Indoor",
    "Outdoor",
    "Obstacle",
    "WorldBuilder",
    "WorldDescriptor",
    "Wall",
    "Rectangle",
    # Plotting
    "Plot2D",
    "Plot3D",
    "mpl_use_latex",
    "mpl_font_size",
    "mpl_use_seaborn",
    "mpl_use_cmap",
    # Engines
    "ReceiverArray",
    "TransmitterArray",
    "PointSource",
    "IsotropicReceiver",
    "Scene",
    "scattering_power",
    "to_dBm",
    "smooth_point_data",
    "threshold_point_data",
    "create_mesh_copies",
    "Environment",
    "Sampler2D",
    "Sampler3D",
    "AoA",
    "extract_aoa",
    "aoa_projection_2D",
    "multi_aoa_projection_2D",
    "map_yaml_to_polygon",
    "simulate_lidar",
    "clean_lidar",
    "extract_lidar_metadata",
    "mesh_error",
]


def __getattr__(name):
    if name in {"ArrayFactory", "NormalFactory", "Symbols", "suppress_c_output"}:
        module = import_module(".utils", __name__)
        return getattr(module, name)

    if name in {"DatabaseReader", "Database", "DatabasePS", "DatabaseAoA", "encode", "decode"}:
        module = import_module(".data_storage", __name__)
        return getattr(module, name)

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
        module = import_module(".simulation", __name__)
        return getattr(module, name)

    if name in {
        "Plot2D",
        "Plot3D",
        "mpl_use_latex",
        "mpl_font_size",
        "mpl_use_seaborn",
        "mpl_use_cmap",
    }:
        module = import_module(".plotting", __name__)
        return getattr(module, name)

    if name in {
        "ReceiverArray",
        "TransmitterArray",
        "PointSource",
        "IsotropicReceiver",
        "Scene",
        "scattering_power",
        "to_dBm",
        "smooth_point_data",
        "threshold_point_data",
        "create_mesh_copies",
        "Environment",
        "Sampler2D",
        "Sampler3D",
        "AoA",
        "extract_aoa",
        "aoa_projection_2D",
        "multi_aoa_projection_2D",
        "map_yaml_to_polygon",
        "simulate_lidar",
        "clean_lidar",
        "extract_lidar_metadata",
        "mesh_error",
    }:
        module = import_module(".engines", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
