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
        AoA,
        extract_aoa,
        aoa_projection_2D,
        multi_aoa_projection_2D,
    )
    from .mapping import (
        Environment,
        Sampler2D,
        Sampler3D,
        cluster_pointcloud,
        extract_lidar_metadata,
        map_yaml_to_polygon,
        mesh_error,
        remove_boundary_points,
        remove_small_holes,
        simulate_lidar,
    )
    from .evaluation import (
        compare_power_distributions,
        neighborhood_adjusted_correlation,
        normalize_metrics,
        rank_power_distributions,
        sample_surface,
        scoring_surface,
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
    "scoring_surface",
    "sample_surface",
    "neighborhood_adjusted_correlation",
    "normalize_metrics",
    "compare_power_distributions",
    "rank_power_distributions",
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
        "AoA",
        "extract_aoa",
        "aoa_projection_2D",
        "multi_aoa_projection_2D",
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
            AoA,
            extract_aoa,
            aoa_projection_2D,
            multi_aoa_projection_2D,
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
            "AoA": AoA,
            "extract_aoa": extract_aoa,
            "aoa_projection_2D": aoa_projection_2D,
            "multi_aoa_projection_2D": multi_aoa_projection_2D,
        }[name]

    if name in {
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
    }:
        from .mapping import (
            Environment,
            Sampler2D,
            Sampler3D,
            cluster_pointcloud,
            extract_lidar_metadata,
            map_yaml_to_polygon,
            mesh_error,
            remove_boundary_points,
            remove_small_holes,
            simulate_lidar,
        )

        return {
            "Environment": Environment,
            "Sampler2D": Sampler2D,
            "Sampler3D": Sampler3D,
            "map_yaml_to_polygon": map_yaml_to_polygon,
            "simulate_lidar": simulate_lidar,
            "cluster_pointcloud": cluster_pointcloud,
            "remove_small_holes": remove_small_holes,
            "remove_boundary_points": remove_boundary_points,
            "extract_lidar_metadata": extract_lidar_metadata,
            "mesh_error": mesh_error,
        }[name]

    if name in {
        "scoring_surface",
        "sample_surface",
        "neighborhood_adjusted_correlation",
        "normalize_metrics",
        "compare_power_distributions",
        "rank_power_distributions",
    }:
        from .evaluation import (
            compare_power_distributions,
            neighborhood_adjusted_correlation,
            normalize_metrics,
            rank_power_distributions,
            sample_surface,
            scoring_surface,
        )

        return {
            "scoring_surface": scoring_surface,
            "sample_surface": sample_surface,
            "neighborhood_adjusted_correlation": neighborhood_adjusted_correlation,
            "normalize_metrics": normalize_metrics,
            "compare_power_distributions": compare_power_distributions,
            "rank_power_distributions": rank_power_distributions,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
