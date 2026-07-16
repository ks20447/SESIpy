"""Sesipy - Spatial Electromagnetic Simulation in Python."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import ArrayFactory, NormalFactory
    from .simulation import World, Indoor, Outdoor, Obstacle
    from .plotting import Plot2D, Plot3D
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
    )

__all__ = [
    # Utils
    "ArrayFactory",
    "NormalFactory",
    # Simulation
    "World",
    "Indoor",
    "Outdoor",
    "Obstacle",
    # Plotting
    "Plot2D",
    "Plot3D",
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
]


def __getattr__(name):
    if name in {"ArrayFactory", "NormalFactory"}:
        module = import_module(".utils", __name__)
        return getattr(module, name)

    if name in {"World", "Indoor", "Outdoor", "Obstacle"}:
        module = import_module(".simulation", __name__)
        return getattr(module, name)

    if name in {"Plot2D", "Plot3D"}:
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
    }:
        module = import_module(".engines", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
