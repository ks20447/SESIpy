"""Sesipy - Spatial Electromagnetic Simulation in Python."""

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
]
