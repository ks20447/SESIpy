import numpy as np
import pandas as pd
from sesipy.utils import ArrayFactory
from sesipy.plotting import Plot2D
from sesipy.engines.mapping import Environment
from sesipy.simulation.worlds import Outdoor
from sesipy.engines.spatial_intelligence.utils import to_dBm, scattering_power
from sesipy.data_storage import DatabaseReader, DatabasePS, DatabaseAoA, decode
from sesipy.engines.spatial_intelligence import IsotropicReceiver, PointSource, Scene
from sesipy.engines.spatial_intelligence.angle_of_arrival import (
    extract_aoa,
    aoa_projection_2D,
    AoA,
)

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


def initialise_scene(world, transmitter, receiver):

    scene = Scene(scatter=True, cuda=True)

    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world.blocker_mesh])
    scene.add_scatterers([world.scatterers[-1]])

    return scene


def visualize_results(env):

    df_t = DatabaseReader("point_sources.csv")

    df = DatabaseReader("angle_of_arrivals.csv")
    x, y = df.mean("X"), df.mean("Y")
    theta = df["AoATheta"].apply(decode)
    amp = df["AoAAmplitude"].apply(decode)

    aoa = [
        AoA(peak=a, left=l, right=r, theta_peaks=t, power_peaks=p)
        for a, l, r, t, p in zip(
            df["AoAPeak"],
            df["AoALeft"],
            df["AoARight"],
            df["AoAThetaPeaks"],
            df["AoAPowerPeaks"],
        )
    ]

    fovs = [aoa_projection_2D((px, py), a, length=100) for px, py, a in zip(x, y, aoa)]
    intersect_fovs = [env.env2D.polygon_intersect_2D(fov) for fov in fovs]

    plotter1 = Plot2D(1, 1)
    plotter1.plot_polygon(env.env2D.polygon, fill_holes=True)

    for fov in intersect_fovs:
        plotter1.plot_polygon(fov, fill_outline=True, c="r", opacity=0.1)

    plotter1.plot_scatter(np.array([[x, y]]), marker="o")
    plotter1.plot_scatter(np.array([[df_t["X"], df_t["Y"]]]), s=30)

    plotter2 = Plot2D(2, 4)

    for t, r, ax, a in zip(theta, amp, plotter2.axes.flatten(), aoa):
        plotter2.use_ax(ax)
        plotter2.plot_aoa(t, r, aoa=a)

    plotter1.axes_set()
    plotter2.show()


def main(new_sim=True):

    world = Outdoor(scatter_resolution=0.2, seed=2)
    env = Environment(world.floor_plan, world.scatter_mesh)

    if new_sim:
        transmitter = initialise_transmitter()
        receiver = initialise_receiver()
        scene = initialise_scene(world, transmitter, receiver)

        transmitter_data = DatabasePS(transmitter, reset=True)
        transmitter_data.assign_world(world)

        receiver_data = DatabaseAoA(reset=True)
        receiver_data.assign_world(world)

        np.random.seed(10)
        transmitter.translate_to(env.env2D.random_sample_2D(1, buffer=-0.5, z=0.5)[0])
        transmitter_data.update()
        transmitter_data.record()

        path, theta = env.env2D.linear_path_2D(
            (-19, 14), (19, -14), 10, buffer=-1.0, z=0.5
        )

        for loc, rot in zip(path, theta):

            array_rot = np.array([0.0, 0.0, rot])

            array_points = receiver.beamform_array + loc
            array_points = ArrayFactory.rotate(array_points, array_rot, loc)

            scatter, _ = scene.sample_receiver_scattering(
                array_points, [array_rot] * len(array_points)
            )
            mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[0]
            steering_mesh = receiver.wave_front_steering(array_points, mean_scatter)

            aoa = extract_aoa(steering_mesh, drop_dB=0.5)

            receiver_data.update(
                TransmitterID=transmitter.id,
                X=array_points[:, 0],
                Y=array_points[:, 1],
                Z=array_points[:, 2],
                RSS=to_dBm(scattering_power(mean_scatter)),
                AoAPeak=aoa.peak,
                AoALeft=aoa.left,
                AoARight=aoa.right,
                AoAThetaPeaks=aoa.theta_peaks,
                AoAPowerPeaks=aoa.power_peaks,
                AoATheta=steering_mesh.point_data["Theta"],
                AoAAmplitude=steering_mesh.point_data["Power"],
            )
            receiver_data.record()

    visualize_results(env)


if __name__ == "__main__":
    main(new_sim=True)
