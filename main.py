import numpy as np
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

    world = Indoor(scatter_resolution=0.4)
    env = Environment(world.floor_plan, world.scatter_mesh)

    transmitter = initialise_transmitter()
    receiver = initialise_receiver()
    scene = initialise_scene(world, transmitter, receiver)

    np.random.seed(10)
    t_loc = env.env2D.random_sample_2D(1, buffer=-0.5, z=0.5)[0]
    transmitter.translate_to(t_loc)

    path, ori = env.env2D.linear_path_2D((0, 0), (1, 1), 1, buffer=0.0, z=0.5)

    idx = 0
    loc, rot = path[idx], ori[idx]

    array_rot = np.array([0.0, 0.0, rot])

    array_points = receiver.beamform_array + loc
    array_points = ArrayFactory.rotate(array_points, array_rot, loc)

    scatter, _ = scene.sample_receiver_scattering(
        array_points, [array_rot] * len(array_points)
    )
    mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[0]

    n_edge = 9
    n_rand = 10
    n_neighbors = 10

    steering_mesh = receiver.wave_front_steering(array_points, mean_scatter)

    aoa = extract_aoa(steering_mesh, drop_dB=0.4)

    steering_mesh_abs = receiver.wave_front_steering(
        array_points, mean_scatter, relative=False
    )

    fov = aoa_projection_2D(loc[0:2], aoa, length=100)
    intersect_fov = env.env2D.polygon_intersect_2D(fov)

    fov_sampler = Sampler2D(intersect_fov, centroid_height=0.5)
    edge_samples = fov_sampler.edge_sample_2D(n_edge, buffer=-0.5, z=0.5)
    random_samples = fov_sampler.random_sample_2D(n_rand, buffer=-0.2, z=0.5)
    polygon_samples = fov_sampler.join_samples(
        edge_samples, fov_sampler.centroid, random_samples
    )

    candidate_spectrums = []
    for sample in polygon_samples:
        transmitter.translate_to(sample)

        candidate_scatter, _ = scene.sample_receiver_scattering(
            array_points, [array_rot] * len(array_points)
        )
        mean_candidate_scatter = np.array(
            [np.mean(scat, axis=2) for scat in candidate_scatter]
        ).T[0]
        candidate_mesh = receiver.wave_front_steering(
            array_points, mean_candidate_scatter, relative=False
        )
        candidate_spectrums.append(candidate_mesh)

    ref = steering_mesh_abs.point_data["Power"].ravel()

    candidate_powers = [c.point_data["Power"] for c in candidate_spectrums]
    adjusted_scores = neighborhood_adjusted_correlation(
        ref, candidate_powers, polygon_samples[:, :2], n_neighbors
    )

    X, Y, Z = scoring_surface(polygon_samples, adjusted_scores, boundary=intersect_fov)
    sample_points, prob_surface = sample_surface(X, Y, Z, 500)
    
    n_candidates = 20
    kmeans = KMeans(
        n_clusters=n_candidates,
        n_init="auto",
        random_state=42,
    )
    kmeans.fit(sample_points)
    candidate_locations = kmeans.cluster_centers_

    plot_samples = Plot2D()
    
    plot_samples.plot_fov(loc[0:2], intersect_fov)
    plot_samples.plot_meshgrid_surface(X, Y, prob_surface)

    plot_samples.plot_polygon(env.env2D.polygon)
    plot_samples.plot_scatter(sample_points, c="orange", s=10, marker="o")
    plot_samples.plot_scatter(candidate_locations, c="red", s=100, marker="^")

    plot_samples.auto_equal_aspect()
    plot_samples.show()

if __name__ == "__main__":
    main()
