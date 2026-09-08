"""Dependency-free tests for export generation and stale-file detection."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/generate_inits.py'
spec = importlib.util.spec_from_file_location('generate_inits', SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class ExportGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('sesipy/__init__.py', '"""Example package."""\n')
        self.write('api_exports.json', '{"packages": {"sesipy": ["Alpha"]}}\n')
        self.write('sesipy/group/alpha.py', '__all__ = ["Alpha"]\nclass Alpha: pass\n')

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(SCRIPT), '--root', str(self.root), *args],
                              capture_output=True, text=True, timeout=10)

    def test_new_module_and_function_propagate_without_importing_source(self):
        self.write('sesipy/group/new.py',
                   '__all__ = ["new_function"]\nraise RuntimeError("must not execute")\n'
                   'def new_function(): pass\n')
        output = generator.generate(self.root)
        group = output[self.root / 'sesipy/group/__init__.py']
        root = output[self.root / 'sesipy/__init__.py']
        self.assertIn("'new_function': '.new'", group)
        self.assertNotIn('new_function', root)
        self.write('api_exports.json', '{"packages": {"sesipy": ["Alpha", "new_function"]}}')
        self.assertIn("'new_function': '.group.new'",
                      generator.generate(self.root)[self.root / 'sesipy/__init__.py'])

    def test_check_does_not_write_and_generation_is_idempotent(self):
        original = (self.root / 'sesipy/__init__.py').read_bytes()
        self.assertEqual(self.run_cli('--check').returncode, 1)
        self.assertEqual((self.root / 'sesipy/__init__.py').read_bytes(), original)
        self.assertFalse((self.root / 'sesipy/group/__init__.py').exists())
        self.assertEqual(self.run_cli().returncode, 0)
        times = {p: p.stat().st_mtime_ns for p in self.root.rglob('__init__.py')}
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual({p: p.stat().st_mtime_ns for p in times}, times)
        self.assertEqual(self.run_cli('--check').returncode, 0)
        self.write('sesipy/group/new.py', '__all__ = ["new_function"]\ndef new_function(): pass\n')
        self.assertEqual(self.run_cli('--check').returncode, 1)

    def test_renamed_export_requires_selection_update(self):
        self.write('sesipy/group/alpha.py', '__all__ = ["Beta"]\nclass Beta: pass\n')
        with self.assertRaisesRegex(generator.ExportError, 'no longer exported'):
            generator.generate(self.root)

    def test_duplicate_export_is_rejected_before_writes(self):
        self.write('sesipy/group/other.py', '__all__ = ["Alpha"]\nclass Alpha: pass\n')
        original = (self.root / 'sesipy/__init__.py').read_bytes()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('duplicate export', result.stderr)
        self.assertEqual((self.root / 'sesipy/__init__.py').read_bytes(), original)

    def test_invalid_module_declarations(self):
        cases = [
            ('class Alpha: pass\n', 'exactly one'),
            ('__all__ = ["Missing"]\n', 'not defined'),
            ('__all__ = list()\n', 'literal'),
            ('__all__ = []\n__all__.append("Alpha")\nclass Alpha: pass\n', 'dynamically'),
            ('__all__ = ["Alpha", "Alpha"]\nclass Alpha: pass\n', 'duplicate'),
        ]
        for source, message in cases:
            with self.subTest(message=message):
                self.write('sesipy/group/alpha.py', source)
                with self.assertRaisesRegex(generator.ExportError, message):
                    generator.generate(self.root)

    def test_internal_module_is_not_exported(self):
        self.write('sesipy/group/internal.py', '__all__ = []\nclass Helper: pass\n')
        self.assertNotIn('Helper', generator.generate(self.root)[self.root / 'sesipy/group/__init__.py'])

    def test_submodule_name_collision_is_rejected(self):
        self.write('sesipy/group/alpha.py', '__all__ = ["alpha"]\ndef alpha(): pass\n')
        with self.assertRaisesRegex(generator.ExportError, 'collide'):
            generator.generate(self.root)

    def test_generated_import_is_lazy_and_caches_only_requested_symbol(self):
        self.write('sesipy/group/broken.py', '__all__ = ["Broken"]\nraise RuntimeError("eager import")\nclass Broken: pass\n')
        self.assertEqual(self.run_cli().returncode, 0)
        result = subprocess.run([sys.executable, '-B', '-c', '''
import sys
import sesipy
assert 'sesipy.group.alpha' not in sys.modules
from sesipy import Alpha
assert sesipy.__dict__['Alpha'] is Alpha
assert 'sesipy.group.broken' not in sys.modules
'''], cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repository_generated_files_are_current(self):
        for path, expected in generator.generate(ROOT).items():
            self.assertEqual(path.read_text(encoding='utf-8'), expected, str(path))


if __name__ == '__main__':
    unittest.main()
