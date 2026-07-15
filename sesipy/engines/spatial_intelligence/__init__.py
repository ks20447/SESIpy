"""Spatial intelligence engine for electromagnetic simulations."""

from .antennas import PointSource, IsotropicReceiver, TransmitterArray, ReceiverArray
from .scene import Scene
from .angle_of_arrival import extract_aoa, aoa_projection_2D
from .utils import (
    scattering_power,
    to_dBm,
    smooth_point_data,
    threshold_point_data,
    create_mesh_copies,
)

__all__ = [
    "PointSource",
    "IsotropicReceiver",
    "TransmitterArray",
    "ReceiverArray",
    "Scene",
    "scattering_power",
    "to_dBm",
    "smooth_point_data",
    "threshold_point_data",
    "create_mesh_copies",
    "extract_aoa",
    "aoa_projection_2D",
]
