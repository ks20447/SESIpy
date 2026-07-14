import meshio
import numpy as np
from typing import Literal
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

NormalType = Literal[
    "x",
    "-x",
    "y",
    "-y",
    "z",
    "-z",
    "forward",
    "backward",
    "target",
    "outward",
    "inward",
]


@dataclass
class NormalFactory:
    mesh: meshio.Mesh

    def calculate(
        self,
        mode: NormalType | None,
        *,
        target: meshio.Mesh | np.ndarray | None = None,
        forward: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ) -> np.ndarray:


        self.mode = mode
        
        if mode is None:
            self.mode = None
            return None

        if mode in ("outward", "inward"):
            normals = self._outward()
            return -normals if mode == "inward" else normals

        if mode in ("target",):
            if target is None:
                raise ValueError("target must be provided")
            self.target = target
            return self._target(target)

        if mode in ("forward", "backward"):
            if forward is None:
                raise ValueError("forward must be provided")
            
            self.forward = forward

            direction = np.asarray(forward, dtype=float)
            direction /= np.linalg.norm(direction)

            if rotation is not None:
                R = Rotation.from_euler("xyz", rotation).as_matrix()
                direction = R @ direction

            normals = self._target(self.mesh.points + direction)
            return -normals if mode == "backward" else normals

        directions = {
            "x": np.array([1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }

        if mode in directions:
            return self._target(self.mesh.points + directions[mode])

        raise ValueError(f"Unknown normal mode '{mode}'")

    def apply(self, *args, **kwargs) -> np.ndarray:
        normals = self.calculate(*args, **kwargs)
        self.mesh.point_data["Normals"] = normals
        return normals

    def _target(self, target: meshio.Mesh | np.ndarray) -> np.ndarray:
        if isinstance(target, meshio.Mesh):
            target_points = target.points
        else:
            target_points = np.asarray(target)

        vectors = target_points - self.mesh.points
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _outward(self) -> np.ndarray:
        center = self.mesh.points.mean(axis=0)

        normals = self.mesh.points - center
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms
    
    def get_args(self):
        return {
            "mode" : self.mode,
            "target" : self.target if hasattr(self, "target") else None,
            "forward": self.forward if hasattr(self, "forward") else None,
        }


