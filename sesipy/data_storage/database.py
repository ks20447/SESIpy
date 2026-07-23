import csv
import json
import zlib
import base64
import numpy as np
import pandas as pd
from pathlib import Path


def encode(arr):
    return base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")


def decode(s, dtype=np.float32):
    data = zlib.decompress(base64.b64decode(s))
    return np.frombuffer(data, dtype=dtype)

class DatabaseReader:

    def __init__(self, filename):
        self.filename = Path(filename)
        self.reload()

    @staticmethod
    def _decode(value):
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def reload(self):
        self.df = pd.read_csv(self.filename)
        self.df = self.df.apply(lambda col: col.map(self._decode))

    @property
    def columns(self):
        return list(self.df.columns)

    @property
    def size(self):
        return len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df[key]

    def row(self, idx):
        return self.df.iloc[idx]

    def rows(self, *idx):
        return self.df.iloc[list(idx)]

    def head(self, n=5):
        return self.df.head(n)

    def tail(self, n=5):
        return self.df.tail(n)

    def ids(self):
        return self.df["id"].to_numpy()

    def values(self, column):
        return self.df[column].tolist()

    def array(self, column):
        return self.df[column].to_numpy()

    def filter(self, **kwargs):
        mask = pd.Series(True, index=self.df.index)
        for key, value in kwargs.items():
            mask &= self.df[key] == value
        return self.df[mask]

    def select(self, *columns):
        return self.df.loc[:, columns]

    def unique(self, column):
        return self.df[column].unique()

    def first(self):
        return self.df.iloc[0]

    def last(self):
        return self.df.iloc[-1]

    def iterrows(self):
        yield from self.df.iterrows()

    def to_dataframe(self):
        return self.df.copy()

    def to_dict(self):
        return self.df.to_dict("records")

    def describe(self):
        return self.df.describe(include="all")

    def info(self):
        return self.df.info()
    
    def mean(self, column):
        return np.array([
            np.mean(np.asarray(value))
            for value in self.df[column]
        ])

class Database:

    def __init__(self, filename, headers, reset=False):
        self.filename = Path(filename)
        self.headers = ["id", *headers]

        self.state = {header: None for header in headers}

        if reset:
            self._next_id = 0
            with self.filename.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

        elif self.filename.exists():
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

        for key, value in self.state.items():
            row[key] = self._serialize(value)

        with self.filename.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)

        self._next_id += 1
        
    @staticmethod
    def _serialize(value):
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())

        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)

        if isinstance(value, np.generic):
            return value.item()

        return value


class DatabasePS(Database):

    def __init__(self, point_source, filename="point_sources.csv", reset=False):

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

        super().__init__(filename, headers, reset=reset)

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

    def __init__(self, filename="angle_of_arrivals.csv", reset=False):

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
            "AoAThetaPeaks",
            "AoAPowerPeaks",
            "AoATheta",
            "AoAAmplitude",
        ]

        super().__init__(filename, headers, reset=reset)

    def assign_world(self, world):
        super().update(
            World=world.name if world.name != "" else id(world),
        )

    def update(self, **kwargs):

        theta = encode(kwargs.get("AoATheta"))
        kwargs.pop("AoATheta")

        amplitude = encode(kwargs.get("AoAAmplitude"))
        kwargs.pop("AoAAmplitude")

        super().update(**kwargs, AoATheta=theta, AoAAmplitude=amplitude)
