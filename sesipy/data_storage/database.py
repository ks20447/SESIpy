import csv
import zlib
import base64
from pathlib import Path


def encode(arr):
    return base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")


def decode(s, dtype):
    data = zlib.decompress(base64.b64decode(s))
    return np.frombuffer(data, dtype=dtype)


class Database:

    def __init__(self, filename, headers):
        self.filename = Path(filename)
        self.headers = ["id", *headers]

        self.state = {header: None for header in headers}

        if self.filename.exists():
            with self.filename.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                self._next_id = int(rows[-1]["id"]) + 1
            else:
                self._next_id = 0
        else:
            self._next_id = 0
            with self.filename.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state:
                raise KeyError(f"Unknown state field '{key}'")
            self.state[key] = value

    def __setitem__(self, key, value):
        if key not in self.state:
            raise KeyError(f"Unknown state field '{key}'")
        self.state[key] = value

    def __getitem__(self, key):
        return self.state[key]

    def record(self):
        row = {"id": self._next_id}
        row.update(self.state)

        with self.filename.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)

        self._next_id += 1


class DatabasePS(Database):

    def __init__(self, point_source, filename="point_sources.csv"):

        self.source = point_source

        headers = [
            "TransmitterID",
            "World",
            "X",
            "Y",
            "Z",
            "Freq",
            "Power",
            "Polar_X",
            "Polar_Y",
            "Polar_Z",
        ]

        super().__init__(
            filename, headers
        )

    def assign_world(self, world):
        super().update(
            World=world.name if world.name != "" else id(world),
        )

    def update(self):

        x, y, z = self.source.points_mesh.center
        freq = self.source.freq
        power = self.source.power
        px, py, pz = self.source.polarization

        super().update(
            TransmitterID=self.source.id,
            X=x,
            Y=y,
            Z=z,
            Freq=freq,
            Power=power,
            Polar_X=px,
            Polar_Y=py,
            Polar_Z=pz,
        )


class DatabaseAoA(Database):
    
    def __init__(self, array, filename="angle_of_arrivals.csv"): 
        
        self.array = array
        
        headers = [
            "World",
            "TransmitterID",
            "X",
            "Y",
            "Z",
            "RSS",
            "AoAPeak",
            "AoALeft",
            "AoARight",
            "AoAPeaks",
            "AoATheta",
            "AoAAmplitude",
        ]
        
        super().__init__(filename, headers)
        
        
    def assign_world(self, world):
        super().update(
            World=world.name if world.name != "" else id(world),
        )
        
        
    def update(self, **kwargs):
        
        theta = encode(kwargs.get("AoATheta"))
        kwargs.pop("AoATheta")
        
        amplitude = encode(kwargs.get("AoAAmplitude"))
        kwargs.pop("AoAAmplitude")
        
        super().update(
            **kwargs,
            AoATheta=theta,
            AoAAmplitude=amplitude
        )