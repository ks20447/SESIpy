import meshio
import numpy as np
import pyvista as pv
from scipy.constants import speed_of_light
from .mesh_handlers import AntennaWrapper
from .utils import scattering_power, to_dBm
from ...utils import ArrayFactory
from lyceanem.base_classes import (
    points,
    structures,
    antenna_structures,
)
from lyceanem.electromagnetics.beamforming import WavefrontWeights


class TransmitterArray(AntennaWrapper):

    def __init__(self, freq, power, polarization):

        super().__init__()

        self.freq = freq
        self.power = power
        self.wavelength = speed_of_light / self.freq
        self.polarization = polarization


class ReceiverArray(AntennaWrapper):

    def __init__(self, gain=1.0):

        super().__init__()

        self.gain = gain
        self._aperture_gain = None
        self._target_freq = None
        self._target_wavelength = None
        self._steering_points = None
        self._beamform_array = None
        self._aoa_threshold = None

    @property
    def target_freq(self):
        return self._target_freq

    @target_freq.setter
    def target_freq(self, freq):
        self._target_freq = freq
        self._target_wavelength = speed_of_light / self.target_freq
        self.aperture_gain = (self.target_wavelength**2) / (4 * np.pi)

    @property
    def target_wavelength(self):
        return self._target_wavelength
    
    @property
    def aperture_gain(self):
        return self._aperture_gain
    
    @aperture_gain.setter
    def aperture_gain(self, val):
        self._aperture_gain = val * self.gain

    @property
    def steering_points(self):
        return self._steering_points

    @steering_points.setter
    def steering_points(self, points):
        self._steering_points = points

    @property
    def beamform_array(self):
        return self._beamform_array

    @beamform_array.setter
    def beamform_array(self, points):
        self._beamform_array = points

    @property
    def aoa_threshold(self):
        return self._aoa_threshold

    @aoa_threshold.setter
    def aoa_threshold(self, value):
        self._aoa_threshold = value

    def create_aperture(self):

        if self.points_mesh is None:
            raise ValueError("No antenna points have been assigned.")

        burner = meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0]]),
            cells=[],
        )
        burner.point_data["Normals"] = np.array([[0.0, 0.0, 1.0]])
        burner.point_data["Area"] = np.array([[1.0]])

        aperture_points = points([burner, self.points_mesh.meshio_mesh])

        aperture_structure = structures(
            [self.structure_mesh.meshio_mesh] if self.structure_mesh is not None else []
        )

        self._aperture = antenna_structures(
            aperture_structure,
            aperture_points,
        )
        
    def wave_front_steering(self, array_points, scatter):

        location = np.mean(array_points, axis=0)

        steering_mesh = pv.PolyData(self.steering_points + location)
        steering_power = np.empty(steering_mesh.number_of_points)

        for i, point in enumerate(steering_mesh.points):

            steering_vec = point - location
            steering_vec /= np.linalg.norm(steering_vec)

            weights = WavefrontWeights(
                array_points,
                steering_vec,
                self.target_wavelength,
            )

            steering_power[i] = to_dBm(
                scattering_power(scatter, weights=weights)
            )

        steering_mesh.point_data["Power"] = steering_power - np.max(steering_power)

        return pv.to_meshio(steering_mesh)
        
        
class PointSource(TransmitterArray):
    
    def __init__(self, freq, power):
        
        super().__init__(freq, power, np.array([0.0, 0.0, 1.0], dtype=np.complex64))
            
        self.points = np.array([[0.0, 0.0, 0.0]])
        self.structure = None
        self.point_normals = np.array([[0.0, 0.0, 1.0]])
        self.point_area = np.array([1.0])
        
        
class IsotropicReceiver(ReceiverArray):
    
    def __init__(self):
        
        super().__init__(gain=1.0)
        
        self.points = np.vstack(ArrayFactory.spherical(12, 7, 0.1))
        self.structure = None
        
        self.normal_factory.apply("outward")
        self.point_area = np.array([1.0] * len(self.points_mesh.points))