"""Spatial intelligence engine for electromagnetic simulations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .angle_of_arrival import (
        AoA,
        aoa_projection_2D,
        extract_aoa,
        multi_aoa_projection_2D,
    )
    from .antennas import (
        IsotropicReceiver,
        PointSource,
        ReceiverArray,
        TransmitterArray,
    )
    from .scene import Scene
    from .utils import (
        create_mesh_copies,
        scattering_power,
        smooth_point_data,
        threshold_point_data,
        to_dBm,
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
    "AoA",
    "extract_aoa",
    "aoa_projection_2D",
    "multi_aoa_projection_2D",
]

_MODULES = {
    "PointSource": ".antennas",
    "IsotropicReceiver": ".antennas",
    "TransmitterArray": ".antennas",
    "ReceiverArray": ".antennas",
    "Scene": ".scene",
    "AoA": ".angle_of_arrival",
    "extract_aoa": ".angle_of_arrival",
    "aoa_projection_2D": ".angle_of_arrival",
    "multi_aoa_projection_2D": ".angle_of_arrival",
    "scattering_power": ".utils",
    "to_dBm": ".utils",
    "smooth_point_data": ".utils",
    "threshold_point_data": ".utils",
    "create_mesh_copies": ".utils",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
