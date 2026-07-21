import numpy as np
from sesipy.simulation import Indoor
from sesipy.utils import ArrayFactory
from sesipy.engines.spatial_intelligence import (
    to_dBm,
    scattering_power,
    extract_aoa,
)
from sesipy.engines import (
    PointSource,
    IsotropicReceiver,
    Scene,
)
from sesipy.data_storage import DatabasePS, DatabaseAoA


def main():

    world_indoor = Indoor(scatter_resolution=0.2)

    transmitter = PointSource(2.4e9, 1.0)

    receiver = IsotropicReceiver()
    receiver.target_freq = transmitter.freq
    receiver.steering_points = ArrayFactory.circle(200, 0.5)
    receiver.beamform_array = ArrayFactory.circle(4, transmitter.wavelength / 4)

    database_ps = DatabasePS(transmitter)
    database_ps.assign_world(world_indoor)
    
    database_aoa = DatabaseAoA(receiver)
    database_aoa.assign_world(world_indoor)

    r_locs = [np.array([-5.0, -3.0, 0.5]), np.array([5.0, 3.0, 0.5])]
    r_rots = [np.radians([0.0, 0.0, 45.0]), np.radians([0.0, 0.0, 0.0])]

    scene = Scene(scatter=True, cuda=True)

    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world_indoor.blocker_mesh])
    scene.add_scatterers([world_indoor.scatter_mesh])
    
    t_locs = [np.array([0.0, 0.0, 0.5]), np.array([1.0, 1.0, 0.5])]
    
    for trans_loc in t_locs:
    
        transmitter.translate_to(trans_loc)
        database_ps.update()
        database_ps.record()

        for array_loc, array_rot in r_locs, r_rots:

            array_points = receiver.beamform_array + array_loc
            array_points = ArrayFactory.rotate(
                array_points, rotation=array_rot, center=array_loc
            )

            scatter, _ = scene.sample_receiver_scattering(
                array_points, [array_rot] * len(array_points)
            )
            mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[0]

            steering_mesh = receiver.wave_front_steering(array_points, mean_scatter)

            aoa = extract_aoa(steering_mesh, drop_dB=1.0)

            database_aoa.update(
                TransmitterID=transmitter.id,
                X=array_points[:, 0],
                Y=array_points[:, 1],
                Z=array_points[:, 2],
                RSS=to_dBm(scattering_power(mean_scatter)),
                AoAPeak=aoa.peak,
                AoALeft=aoa.left,
                AoARight=aoa.right,
                AoAPeaks=aoa.theta_peaks,
                AoATheta=steering_mesh.point_data["Theta"],
                AoAAmplitude=steering_mesh.point_data["Power"],
            )
            database_aoa.record()
            


if __name__ == "__main__":
    main()
