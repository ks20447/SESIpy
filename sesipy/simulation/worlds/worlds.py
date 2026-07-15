import time
import yaml
import numpy as np
from pathlib import Path
from .world_builder import WorldBuilder


class Indoor(WorldBuilder):

    def __init__(self, scatter_resolution=1.0):
        config_path = Path(__file__).resolve().parent / "config" / "indoor_params.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        super().__init__(params=params, scatter_resolution=scatter_resolution)
        
        self.create_pgm()


class Outdoor(WorldBuilder):

    def __init__(self, scatter_resolution=1.0, seed=int(time.time())):
        config_path = Path(__file__).resolve().parent / "config" / "outdoor_params.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        np.random.seed(seed)

        super().__init__(params=params, scatter_resolution=scatter_resolution)
        
        self.create_pgm()
