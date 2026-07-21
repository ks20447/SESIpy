"""Engines module for Sesipy."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spatial_intelligence import (
        PointSource,
        IsotropicReceiver,
        TransmitterArray,
        ReceiverArray,
        Scene,
        scattering_power,
        to_dBm,
        smooth_point_data,
        threshold_point_data,
        create_mesh_copies,
    )
    from .mapping import Environment, Sampler2D, Sampler3D

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
    "Environment",
    "Sampler2D",
    "Sampler3D",
]


def __getattr__(name):
    if name in {
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
    }:
        from .spatial_intelligence import (
            PointSource,
            IsotropicReceiver,
            TransmitterArray,
            ReceiverArray,
            Scene,
            scattering_power,
            to_dBm,
            smooth_point_data,
            threshold_point_data,
            create_mesh_copies,
        )

        return {
            "PointSource": PointSource,
            "IsotropicReceiver": IsotropicReceiver,
            "TransmitterArray": TransmitterArray,
            "ReceiverArray": ReceiverArray,
            "Scene": Scene,
            "scattering_power": scattering_power,
            "to_dBm": to_dBm,
            "smooth_point_data": smooth_point_data,
            "threshold_point_data": threshold_point_data,
            "create_mesh_copies": create_mesh_copies,
        }[name]

    if name in {"Environment", "Sampler2D", "Sampler3D"}:
        from .mapping import Environment, Sampler2D, Sampler3D

        return {
            "Environment": Environment,
            "Sampler2D": Sampler2D,
            "Sampler3D": Sampler3D,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
