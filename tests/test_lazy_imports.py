"""Run with: python -m unittest discover -s tests -p 'test_lazy_imports.py'."""

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LazyImportTests(unittest.TestCase):
    def run_python(self, source):
        result = subprocess.run(
            [sys.executable, '-B', '-c', source],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lightweight_exports_do_not_load_unrelated_dependencies(self):
        cases = [
            ('to_dBm', 'engines', 'spatial_intelligence.utils',
             ['lyceanem', 'open3d', 'pyvista', 'sesipy.engines.spatial_intelligence.scene']),
            ('Environment', 'engines', 'mapping.environment',
             ['lyceanem', 'open3d', 'vtk', 'sesipy.engines.mapping.utils']),
            ('normalize_metrics', 'engines', 'evaluation.signals',
             ['lyceanem', 'open3d', 'shapely', 'sesipy.engines.evaluation.localization']),
            ('Obstacle', 'simulation', 'worlds.utils',
             ['cv2', 'pyvista', 'sesipy.simulation.worlds.world_builder',
              'sesipy.simulation.worlds.worlds']),
        ]
        for name, parent, leaf, forbidden in cases:
            with self.subTest(name=name):
                self.run_python(f'''
import importlib
import sys
import sesipy
value = getattr(sesipy, {name!r})
parent = importlib.import_module('sesipy.' + {parent!r})
assert sesipy.__dict__[{name!r}] is value, 'Root export was not cached'
assert getattr(parent, {name!r}) is value
assert parent.__dict__[{name!r}] is value, 'Parent export was not cached'
for prefix in {forbidden!r}:
    assert not any(m == prefix or m.startswith(prefix + '.') for m in sys.modules), prefix
leaf = importlib.import_module('sesipy.' + {parent!r} + '.' + {leaf!r})
assert value is getattr(leaf, {name!r})
''')

    def test_every_public_export_resolves(self):
        self.run_python('''
import importlib
from pathlib import Path
for path in sorted(Path('sesipy').rglob('__init__.py')):
    name = '.'.join(path.parent.parts)
    package = importlib.import_module(name)
    exports = getattr(package, '__all__', [])
    assert len(exports) == len(set(exports)), name
    for export in exports:
        getattr(package, export)
    namespace = {}
    exec('from ' + name + ' import *', namespace)
    for export in exports:
        assert namespace[export] is getattr(package, export), (name, export)
''')

    def test_world_builder_uses_explicit_dependencies(self):
        self.run_python('''
from sesipy import WorldBuilder, WorldDescriptor
from shapely.geometry import box

descriptor = WorldDescriptor(False, False, False, 2)
descriptor.build_from_polygon(box(0, 0, 2, 2), [])
assert descriptor.get_data()['bound']['xmax'] == 2.0
world = WorldBuilder({
    'bound': {'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 2},
    'floor': True, 'roof': False, 'walls': False, 'boundary_z': 2,
    'generator': {'type': 'fixed', 'fixed_obstacles': []},
})
assert len(world.scatter_mesh.points) > 0
assert len(world.blocker_mesh.points) > 0
''')

    def test_unknown_attributes_and_discovery(self):
        self.run_python('''
import sesipy.engines as engines
import sesipy.simulation as simulation
for package in (engines, simulation):
    assert set(package.__all__) <= set(dir(package))
    try:
        getattr(package, '_nonexistent_export')
    except AttributeError:
        pass
    else:
        raise AssertionError('Unknown attribute did not raise AttributeError')
''')


if __name__ == '__main__':
    unittest.main()
