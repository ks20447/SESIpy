import gc
import threading
import time
import statistics

import numpy as np
import pandas as pd
import psutil
import shapely as sp
import pynvml

from scipy.constants import speed_of_light as C

from sesipy.simulation.worlds import WorldDescriptor, WorldBuilder
from sesipy.engines.spatial_intelligence import (
    IsotropicReceiver,
    PointSource,
    Scene,
)

FREQ = 2.4e9
WAVELENGTH = C / FREQ
POWER = 0.1
GPU_ID = 0
RUNS_PER_SCALE = 5


def initialise_test_world(scale):
    world_desc = WorldDescriptor(True, False, True, boundary_z=1.0)
    world_poly = sp.box(-scale, -scale, scale, scale)
    world_desc.build_from_polygon(world_poly, [])
    return WorldBuilder(world_desc.get_data(), scatter_resolution=WAVELENGTH)


def initialise_receiver():
    receiver = IsotropicReceiver()
    receiver.target_freq = FREQ
    return receiver


def initialise_transmitter():
    return PointSource(FREQ, POWER)


def initialise_scene(world, transmitter, receiver):
    scene = Scene(scatter=True, cuda=True)
    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world.blocker_mesh])
    scene.add_scatterers(world.scatterers)
    return scene


class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.005):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        self.running = False
        self.thread = None
        self.peak_memory = 0
        self.peak_utilisation = 0

    def _monitor(self):
        while self.running:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilisation = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.peak_memory = max(self.peak_memory, memory.used)
            self.peak_utilisation = max(self.peak_utilisation, utilisation.gpu)
            time.sleep(self.interval)

    def start(self):
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.peak_memory = memory.used
        self.peak_utilisation = 0
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def get_memory_mb(self):
        return self.peak_memory / 1024**2

    def get_utilisation_percent(self):
        return self.peak_utilisation


def get_gpu_memory():
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return memory.used / 1024**2


def synchronise_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    try:
        import cupy

        cupy.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass


def warmup_scattering(scene):
    synchronise_gpu()
    scene.calculate_receiver_scattering()
    synchronise_gpu()


def profile_scattering(scene, world, runs=RUNS_PER_SCALE):
    process = psutil.Process()

    # warmup_scattering(scene)

    gc.collect()
    clear_cuda_cache()
    synchronise_gpu()

    cpu_memory_before = process.memory_info().rss
    gpu_memory_before = get_gpu_memory()

    gpu_monitor = GPUMonitor(GPU_ID)
    gpu_monitor.start()

    execution_times = []

    for _ in range(runs):
        synchronise_gpu()
        start = time.perf_counter()

        scene.calculate_receiver_scattering()

        synchronise_gpu()
        execution_times.append(time.perf_counter() - start)

    gpu_monitor.stop()

    cpu_memory_after = process.memory_info().rss
    gpu_memory_after = get_gpu_memory()

    median_time = statistics.median(execution_times)

    return {
        "calculation_time_s": median_time,
        "cpu_memory_before_mb": cpu_memory_before / 1024**2,
        "cpu_memory_after_mb": cpu_memory_after / 1024**2,
        "cpu_memory_delta_mb": (cpu_memory_after - cpu_memory_before) / 1024**2,
        "gpu_memory_before_mb": gpu_memory_before,
        "gpu_memory_after_mb": gpu_memory_after,
        "gpu_memory_delta_mb": (gpu_memory_after - gpu_memory_before),
        "gpu_peak_memory_mb": gpu_monitor.get_memory_mb(),
        "gpu_peak_utilisation_percent": gpu_monitor.get_utilisation_percent(),
    }


def main():
    
    pynvml.nvmlInit()
    scales = np.arange(0.0, 51.0, 10.0)
    scales[0] = 1.0

    results = []

    for scale in scales:
        print(f"\nScale: {scale:.3f}")

        world = initialise_test_world(scale)
        transmitter = initialise_transmitter()
        receiver = initialise_receiver()

        scene = initialise_scene(world, transmitter, receiver)
        transmitter.translate_to(np.array([0.0, 0.0, 0.5]))
        receiver.translate_to(np.array([0.5, 0.5, 0.5]))

        n_points = np.sum(world.scatter_metadata["points"])

        print(f"Profiling over {RUNS_PER_SCALE} runs...")
        result = profile_scattering(scene, world)

        result["scale"] = scale
        result["n_points"] = n_points
        results.append(result)

        print(
            f"Points:          {n_points:,}\n"
            f"Median Time:     {result['calculation_time_s']:.4f} s\n"
            f"CPU mem delta:   {result['cpu_memory_delta_mb']:.1f} MB\n"
            f"GPU mem delta:   {result['gpu_memory_delta_mb']:.1f} MB\n"
            f"GPU peak mem:    {result['gpu_peak_memory_mb']:.1f} MB\n"
            f"GPU utilisation: {result['gpu_peak_utilisation_percent']:.1f}%"
        )

        del scene
        del receiver
        del transmitter
        del world

        gc.collect()
        clear_cuda_cache()

    df = pd.DataFrame(results)
    df = df[
        [
            "scale",
            "n_points",
            "calculation_time_s",
            "cpu_memory_before_mb",
            "cpu_memory_after_mb",
            "cpu_memory_delta_mb",
            "gpu_memory_before_mb",
            "gpu_memory_after_mb",
            "gpu_memory_delta_mb",
            "gpu_peak_memory_mb",
            "gpu_peak_utilisation_percent",
        ]
    ]

    df.to_csv("sesipy_scattering_profile.csv", index=False)
    print("\n")
    print(df.to_string(index=False))

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
