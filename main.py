from sesipy.engines.mapping import map_yaml_to_polygon
from sesipy.simulation.worlds.world_builder import WorldDescriptor, WorldBuilder
from sesipy.plotting import Plot2D, Plot3D


def main():

    map_polygon = map_yaml_to_polygon("square_world.yaml")

    world_desc = WorldDescriptor(floor=True, roof=False, walls=False, boundary_z=5)
    world_desc.build_from_polygon(map_polygon.simplify(0.1), obstacle_heights=[])

    square_world = WorldBuilder(params=world_desc.get_data())

    plotter = Plot2D()
    plotter.plot_polygon(square_world.floor_plan)
    plotter.show()

    plotter = Plot3D()
    plotter.plot_scatterers(square_world.blockers)
    plotter.plot_scatterers(square_world.scatterers)
    plotter.show()


if __name__ == "__main__":
    main()
