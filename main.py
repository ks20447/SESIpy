import numpy as np
from shapely import contains_xy
from sklearn.cluster import KMeans
from sesipy.utils import ArrayFactory
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines.mapping import Environment, Sampler2D
from sesipy.simulation.worlds import Indoor
from sesipy.engines.spatial_intelligence import (
    IsotropicReceiver,
    PointSource,
    TransmitterArray,
    Scene,
)
from sesipy.engines.spatial_intelligence.angle_of_arrival import (
    extract_aoa,
    aoa_projection_2D,
    Measurement,
    spectrum_spatial_distribution_2D
)
from sesipy.engines.evaluation.signals import neighborhood_adjusted_correlation
from sesipy.engines.evaluation.localization import scoring_surface, sample_surface

FREQ = 2.4e9
POWER = 0.1


def initialise_receiver():

    receiver = IsotropicReceiver()

    receiver.target_freq = FREQ
    receiver.steering_points = ArrayFactory.circle(200, 0.5)
    receiver.beamform_array = ArrayFactory.circle(4, receiver.target_wavelength / 4)

    return receiver


def initialise_transmitter():

    transmitter = PointSource(FREQ, POWER)

    return transmitter


def sample_transmitter(points, normals):

    transmitter = TransmitterArray(
        FREQ,
        POWER,
        polarization=np.array([0.0, 0.0, 1.0], dtype=np.complex64),
    )
    transmitter.points = points
    transmitter.point_normals = normals

    transmitter.point_area = np.array([1.0] * len(transmitter.points_mesh.points))

    return transmitter


def initialise_scene(world, transmitter, receiver):

    scene = Scene(scatter=True, cuda=True)

    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world.blocker_mesh])
    scene.add_scatterers(world.scatterers)

    return scene


def main():

    world = Indoor(scatter_resolution=0.2)
    env = Environment(world.floor_plan, world.scatter_mesh)

    transmitter = initialise_transmitter()
    receiver = initialise_receiver()
    scene = initialise_scene(world, transmitter, receiver)

    t_loc = env.env2D.random_sample_2D(1, buffer=-0.2, z=0.5)[0]
    transmitter.translate_to(t_loc)

    path = np.load("path.npy")
    ori = np.zeros(len(path))

    measurements = {
        f"{i}": []
        for i in range(len(path))
    }

    for i, (loc, rot) in enumerate(zip(path, ori)):
        array_rot = np.array([0.0, 0.0, rot])

        array_points = receiver.beamform_array + loc
        array_points = ArrayFactory.rotate(array_points, array_rot, loc)

        scatter, _ = scene.sample_receiver_scattering(
            array_points, [array_rot] * len(array_points)
        )
        mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[
            0
        ]

        steering_mesh = receiver.wave_front_steering(
            array_points, mean_scatter
        )

        aoa = extract_aoa(steering_mesh, drop_dB=0.4)

        steering_mesh_abs = receiver.wave_front_steering(
            array_points, mean_scatter, relative=False
        )

        measurements[f"{i}"].append(Measurement(
            position=np.asarray(loc),
            orientation=array_rot,
            aoa=aoa,
            steering_mesh=steering_mesh_abs
        ))
        
    mapx_min, mapy_min, mapx_max, mapy_max = env.env2D.polygon.bounds
    x_map = np.linspace(mapx_min, mapx_max, 200)
    y_map = np.linspace(mapy_min, mapy_max, 200)
    X_map, Y_map = np.meshgrid(x_map, y_map)
    map_mask = contains_xy(env.env2D.polygon, X_map, Y_map)
        
    for measure in measurements.values():

        steer_mesh = measure[0].steering_mesh
        pos = measure[0].position

        theta_abs = steer_mesh.point_data["Theta"]
        r_abs = steer_mesh.point_data["Power"]

        spat_dist = spectrum_spatial_distribution_2D(X_map, Y_map, pos[0:2], theta_abs, r_abs)
        spat_dist = np.where(map_mask, spat_dist, 0.0)

        gamma = 2.5
        spat_dist **= gamma

        dist_sum = spat_dist.sum()
        if dist_sum > 0:
            spat_dist /= dist_sum

        measure[0].projection = spat_dist
        
    probs = [m[0].projection for m in measurements.values()]
        
    joint_spat_dist = np.prod(probs, axis=0)
    joint_sum = joint_spat_dist.sum()
    if joint_sum > 0:
        joint_spat_dist /= joint_sum
        
    flat_prob = joint_spat_dist.ravel()
    num_grid_points = flat_prob.size

    n_samples = 200
    sampled_indices = np.random.choice(
        num_grid_points, 
        size=n_samples, 
        p=flat_prob
    )

    jy, jx = np.unravel_index(sampled_indices, X_map.shape)

    sampled_x = X_map[jy, jx]
    sampled_y = Y_map[jy, jx]

    dx = (mapx_max - mapx_min) / (X_map.shape[1] - 1)
    dy = (mapy_max - mapy_min) / (X_map.shape[0] - 1)

    sampled_x += np.random.uniform(-dx / 2, dx / 2, size=n_samples)
    sampled_y += np.random.uniform(-dy / 2, dy / 2, size=n_samples)

    dist_sample = np.column_stack((sampled_x, sampled_y))

    n_candidates = 20
    kmeans_map = KMeans(
        n_clusters=n_candidates,
        n_init="auto",
        random_state=42,
    )

    kmeans_map.fit(dist_sample)
    candidates = kmeans_map.cluster_centers_
    
    plot_swarm = Plot2D(2, 2)

    plot_swarm.set_ax(0, 0)
    plot_swarm.plot_polygon(env.env2D.polygon)
    plot_swarm.plot_meshgrid_surface(X_map, Y_map, probs[-1])

    plot_swarm.set_ax(0, 1)
    plot_swarm.plot_polygon(env.env2D.polygon)
    plot_swarm.plot_meshgrid_surface(X_map, Y_map, probs[-6])

    plot_swarm.set_ax(1, 0)
    plot_swarm.plot_polygon(env.env2D.polygon)
    plot_swarm.plot_meshgrid_surface(X_map, Y_map, joint_spat_dist)
    plot_swarm.plot_scatter(path[:, 0:2], c="green", marker="o", s=100)
    plot_swarm.plot_scatter(np.array([t_loc[0:2]]), c="red", s=100)

    plot_swarm.set_ax(1, 1)
    plot_swarm.plot_polygon(env.env2D.polygon)
    plot_swarm.plot_scatter(dist_sample, s=10, marker="o", c="orange")
    plot_swarm.plot_scatter(candidates, s=50, marker="^", c="red")

    plot_swarm.show()

if __name__ == "__main__":
    main()
