import yaml
import json
import meshio
import numpy as np
import shapely as sp
from sesipy.simulation import World, Indoor, Outdoor, WorldBuilder, WorldDescriptor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.utils import ArrayFactory
# from sesipy.engines.spatial_intelligence import extract_aoa, aoa_projection_2D
# from sesipy.engines import (
#     PointSource,
#     IsotropicReceiver,
#     Scene,
#     scattering_power,
#     to_dBm,
#     Environment,
# )


def main():

    descriptor = WorldDescriptor(floor=True, roof=False, walls=False, boundary_z=2.0)

    world_poly = sp.Polygon(
        [(-2, -2), (2, -2), (2, 2), (0, 4), (-2, 2)],
        holes=[[(-1, -1), (0, -1), (0, 0), (-1, 0)]],
    )

    descriptor.build_from_polygon(world_poly, obstacle_heights=[1.0])
    descriptor.write_to_yaml("test.yaml")
    descriptor.write_to_json("test.json")
        
    with open("test.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    world = WorldBuilder(data, scatter_resolution=0.2)

    plotter = Plot3D()
    plotter.plot_scatterers(world.scatterers)
    plotter.show()
    
    with open("test.json", "r") as f:
        data = json.load(f)

    world = WorldBuilder(data, scatter_resolution=0.2)

    plotter = Plot3D()
    plotter.plot_scatterers(world.scatterers)
    plotter.show()


if __name__ == "__main__":
    main()
