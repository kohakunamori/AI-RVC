"""Official packages must not hide the frozen app's dependency search path."""
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from lib.upstream_imports import activate_upstream_packages


class UpstreamImportIsolationTests(unittest.TestCase):
    def test_official_packages_replace_application_packages_and_keep_dependencies(self):
        for regular in (False, True):
            with self.subTest(regular=regular), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app, upstream = root / "application", root / "upstream"
                for name in ("configs", "i18n", "infer", "tools"):
                    (app / name).mkdir(parents=True)
                    (app / name / "__init__.py").write_text("ORIGIN = 'application'", encoding="utf-8")
                    (app / name / "probe.py").write_text("ORIGIN = 'application'", encoding="utf-8")
                    (upstream / name).mkdir(parents=True)
                    if regular:
                        (upstream / name / "__init__.py").write_text("ORIGIN = 'upstream'", encoding="utf-8")
                    (upstream / name / "probe.py").write_text("ORIGIN = 'upstream'", encoding="utf-8")
                archive = root / "bundled-dependencies.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr("audit_dependency.py", "ORIGIN = 'bundled'\n")
                script = f"""
import importlib, sys
from pathlib import Path
from lib.upstream_imports import activate_upstream_packages
sys.path[:0] = [{str(app)!r}, {str(archive)!r}]
import infer.probe
assert infer.probe.ORIGIN == 'application'
before = list(sys.path)
activate_upstream_packages(Path({str(upstream)!r}))
assert all(entry in sys.path for entry in before)
for name in ('configs', 'i18n', 'infer', 'tools'):
    module = importlib.import_module(name + '.probe')
    assert module.ORIGIN == 'upstream', module.__file__
import audit_dependency
assert audit_dependency.ORIGIN == 'bundled'
"""
                result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_replacement_invalidates_only_replaced_inspect_file_caches(self):
        for regular in (False, True):
            with self.subTest(regular=regular), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app, upstream = root / "application", root / "upstream"
                for name in ("configs", "i18n", "infer", "tools"):
                    (app / name).mkdir(parents=True)
                    (app / name / "__init__.py").write_text("", encoding="utf-8")
                    (app / name / "probe.py").write_text("def locate(): pass\n", encoding="utf-8")
                    (upstream / name).mkdir(parents=True)
                    if regular:
                        (upstream / name / "__init__.py").write_text("", encoding="utf-8")
                    (upstream / name / "probe.py").write_text("def locate(): pass\n", encoding="utf-8")
                (app / "audit_dependency.py").write_text("def locate(): pass\n", encoding="utf-8")
                script = f"""
import importlib, inspect, sys
from pathlib import Path
from lib.upstream_imports import activate_upstream_packages
sys.path.insert(0, {str(app)!r})
names = ('configs', 'i18n', 'infer', 'tools')
old_modules = [importlib.import_module(name + '.probe') for name in names]
import audit_dependency
# Populate inspect's real caches as stack inspection in lazy_loader does.
assert inspect.getmodule(compile('pass', '_uncached_before_activation.py', 'exec')) is None
assert inspect._filesbymodname['infer'] == sys.modules['infer'].__file__
dependency_file = inspect.getabsfile(audit_dependency)
assert inspect.modulesbyfile[dependency_file] == 'audit_dependency'
activate_upstream_packages(Path({str(upstream)!r}))
assert inspect.modulesbyfile[dependency_file] == 'audit_dependency'
assert inspect._filesbymodname['audit_dependency'] == audit_dependency.__file__
# A namespace package has no source file. Stale application paths must not
# make inspect try getfile(namespace), or resolve old code to the new module.
assert inspect.getmodule(compile('pass', '_uncached_after_activation.py', 'exec')) is None
assert inspect.stack()
for name, old_module in zip(names, old_modules):
    module = importlib.import_module(name + '.probe')
    assert inspect.getmodule(module.locate.__code__) is module
    assert inspect.getmodule(old_module.locate.__code__) is None
    # Activation resolves macOS temp symlinks and Windows short directory names.
    assert Path(inspect.getfile(module.locate)).resolve().is_relative_to(Path({str(upstream)!r}).resolve())
assert inspect.getmodule(audit_dependency.locate.__code__) is audit_dependency
"""
                result = subprocess.run(
                    [sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_upstream_package_does_not_activate_application_code(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(sys, "path", list(sys.path)):
            before = list(sys.path)
            with self.assertRaises(FileNotFoundError):
                activate_upstream_packages(Path(temporary))
            self.assertEqual(sys.path, before)


if __name__ == "__main__":
    unittest.main()
