"""Reproduce archive-only cache failures and exercise source-backed native audio."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
