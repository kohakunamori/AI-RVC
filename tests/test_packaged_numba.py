"""Reproduce archive-only cache failures and exercise source-backed native audio."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class PackagedNumbaTests(unittest.TestCase):
    def test_archived_code_fails_but_source_import_compiles_and_reuses_cache(self):
        probe = r'''
import importlib.util
import json
import sys
import types
from pathlib import Path
import numpy as np
from numba import njit

root, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
source = root / 'lib' / 'mixer.py'
# Match PyInstaller's PYZ code, despite a separate .py file existing as data.
archived = types.ModuleType('archive_probe')
archived.__file__ = str(source)
sys.modules[archived.__name__] = archived
try:
    exec(compile(source.read_bytes(), str(Path('lib') / 'mixer.py'), 'exec'), archived.__dict__)
except RuntimeError as error:
    if 'no locator available' not in str(error):
        raise
else:
    raise AssertionError('The archived-source regression was not reproduced')

from tools.packaged_smoke import check_audio
print(json.dumps(check_audio(output)))
'''
        with tempfile.TemporaryDirectory(prefix="ai-rvc-numba-") as directory:
            cache = Path(directory) / "cache"
            env = dict(os.environ, NUMBA_CACHE_DIR=str(cache),
                       NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator", PYTHONUTF8="1")
            reports = []
            for phase in ("cold", "warm"):
                result = subprocess.run(
                    [sys.executable, "-c", probe, str(ROOT), str(Path(directory) / phase)],
                    cwd=directory, env=env, text=True, encoding="utf-8", capture_output=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                import json
                reports.append(json.loads(result.stdout.strip().splitlines()[-1]))
            self.assertEqual(reports[0]["cache_hits"], 0)
            self.assertGreaterEqual(reports[1]["cache_hits"], 1)
            self.assertTrue(Path(reports[0]["cache"]).is_relative_to(cache))
            self.assertTrue(list(cache.rglob("*.nbc")))

    def test_both_build_routes_run_the_real_frozen_verifier(self):
        spec = (ROOT / "AI-RVC.spec").read_text(encoding="utf-8-sig")
        workflow = (ROOT / ".github/workflows/build-executables.yml").read_text(encoding="utf-8")
        self.assertIn("verify_packaged_runtime(Path(DISTPATH)", spec)
        self.assertIn('python tools/verify_packaged_runtime.py "${EXE}"', workflow)
        self.assertIn("tools/pyinstaller_hooks", workflow)
        self.assertIn("'pyinstaller_hooks'", spec)
        self.assertIn('--add-data "ui/*.py${SEP}ui"', workflow)
        self.assertIn('--add-data "ui/multitrack/dist/player.html${SEP}ui/multitrack/dist"', workflow)
        self.assertNotIn('--add-data "ui${SEP}ui"', workflow)
        self.assertIn('python tools/package_runtime.py .', workflow)
        self.assertIn('python tools/package_runtime.py "dist/${NAME}/_internal" --packaged', workflow)

    def test_package_layout_requires_built_player_and_rejects_development_dependencies(self):
        from tools.package_runtime import validate_runtime_layout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "player is missing"):
                validate_runtime_layout(root)
            player = root / "ui/multitrack/dist/player.html"
            player.parent.mkdir(parents=True)
            player.write_text("<!doctype html><html>bundled player</html>", encoding="utf-8")
            validate_runtime_layout(root)
            (root / "ui/multitrack/node_modules").mkdir()
            with self.assertRaisesRegex(RuntimeError, "node_modules"):
                validate_runtime_layout(root)

    def test_source_preparation_accepts_build_dependencies_but_packaged_cli_rejects_them(self):
        from tools import package_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            player = root / "ui/multitrack/dist/player.html"
            player.parent.mkdir(parents=True)
            player.write_text("<!doctype html><html>bundled player</html>", encoding="utf-8")
            (root / "ui/multitrack/node_modules").mkdir()
            with patch.object(package_runtime, "validate_upstream_sources", return_value=495) as sources:
                with patch.object(sys, "argv", ["package_runtime.py", str(root)]):
                    package_runtime.main()
                sources.assert_called_once_with(root.resolve())
                with patch.object(sys, "argv", ["package_runtime.py", str(root), "--packaged"]):
                    with self.assertRaisesRegex(RuntimeError, "node_modules"):
                        package_runtime.main()


if __name__ == "__main__":
    unittest.main()
