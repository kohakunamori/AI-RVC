"""Native frozen GUI + bundled CUDA contract tests (no GPU required)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativeDesktopReleaseTests(unittest.TestCase):
    def test_native_studio_is_frozen_in_both_build_routes(self):
        workflow = (ROOT / ".github/workflows/build-executables.yml").read_text(encoding="utf-8")
        spec = (ROOT / "AI-RVC.spec").read_text(encoding="utf-8")
        for token in (
            'PySide6-Essentials>=6.8,<7', 'sounddevice>=0.5,<1',
            '--hidden-import=desktop.gui', '--hidden-import=sounddevice',
            '--add-data "desktop/*.py${SEP}desktop"',
            '--collect-all _sounddevice_data',
            'python tools/verify_packaged_runtime.py "${EXE}"',
            'python tools/package_runtime.py "dist/${NAME}/_internal" --packaged --cuda 12.8',
            'WINDOWED_FLAGS+=(--windowed)',
        ):
            self.assertIn(token, workflow)
        self.assertIn("'desktop.gui'", spec)
        self.assertIn("'sounddevice'", spec)
        self.assertIn("'PySide6.QtWidgets'", spec)
        self.assertIn("console=sys.platform != 'win32'", spec)
        self.assertNotIn("--collect-all gradio", workflow)
        self.assertNotIn("'gradio',", spec)
        self.assertIn("torch==2.11.0", workflow)
        self.assertIn("https://download.pytorch.org/whl/cu128", workflow)

    def test_frozen_smoke_uses_real_qt_and_does_not_start_web(self):
        smoke = (ROOT / "tools/packaged_smoke.py").read_text(encoding="utf-8")
        verify = (ROOT / "tools/verify_packaged_runtime.py").read_text(encoding="utf-8")
        self.assertIn("def check_desktop()", smoke)
        self.assertIn('from desktop.gui import DesktopWindow', smoke)
        self.assertIn('report["desktop_tabs"] = check_desktop()', smoke)
        self.assertIn('QT_QPA_PLATFORM="offscreen"', verify)
        self.assertNotIn("blocks.launch", smoke)
        ast.parse(smoke)
        ast.parse(verify)

    def test_version_and_release_notes_are_consistent(self):
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(version, "2.0.0")
        notes = (ROOT / "docs/releases" / f"v{version}.md").read_text(encoding="utf-8")
        for item in ("## English", "CUDA 12.8", "Windows", "Linux", "SHA256SUMS", "NVIDIA"):
            self.assertIn(item, notes)


if __name__ == "__main__":
    unittest.main()
