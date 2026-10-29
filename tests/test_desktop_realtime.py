"""Realtime audio tests requiring no soundcard, GPU, or actual RVC model."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from desktop.realtime import RollingAudio, RealtimeRVC, RealtimeSettings, validate_settings

ROOT = Path(__file__).resolve().parents[1]


class RollingRealtimeTests(unittest.TestCase):
    def test_48k_input_blocks_are_exact_16k_audio_context(self):
        history = RollingAudio(48000, 250, 1200, 40)
        self.assertEqual(history.block_frames, 12000)
        self.assertEqual(history.hop, 480)
        self.assertEqual(history.fade_frames, 1920)
        expected = history.total_frames * 16000 // 48000
        self.assertEqual(history.input16.size, expected)
        samples = np.zeros(12000, dtype=np.float32)
        samples[200:700] = 0.4
        wave = history.ingest(samples)
        self.assertEqual(wave.size, expected)
        self.assertTrue(np.isfinite(wave).all())
        self.assertGreater(float(np.max(np.abs(wave))), 0.01)

    def test_sola_has_exact_block_size_and_keeps_finite_output(self):
        history = RollingAudio(48000, 250, 1200, 40)
        total = history.block_frames + history.fade_frames + history.search_frames
        t = np.arange(total, dtype=np.float32) / 48000
        audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        a = history.align(audio, 48000)
        b = history.align(audio, 48000)
        self.assertEqual(a.shape, (12000,))
        self.assertEqual(b.shape, (12000,))
        self.assertTrue(np.isfinite(b).all())
        self.assertFalse(history.first)

    def test_mismatched_rvc_output_rate_is_resampled_to_device(self):
        history = RollingAudio(48000, 200, 1000, 40)
        expected48 = history.block_frames + history.fade_frames + history.search_frames
        data40 = np.zeros(int(expected48 * 40000 / 48000), dtype=np.float32)
        # SciPy resample_poly lengths can be rounded to nearest sample.
        result = history.align(np.pad(data40, (0, 12)), 40000)
        self.assertEqual(len(result), history.block_frames)

    def test_rejects_malformed_blocks_and_inference_outputs(self):
        history = RollingAudio(48000, 250, 1200, 40)
        with self.assertRaises(ValueError):
            history.ingest(np.ones(2000, dtype=np.float32))
        with self.assertRaises(RuntimeError):
            history.align(np.zeros(500), 48000)


class RealtimeCallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pth = Path(self.tmp.name) / "test.pth"
        self.pth.write_bytes(b"dummy")

    def test_audio_callback_does_not_import_torch_or_block(self):
        rvc = RealtimeRVC(RealtimeSettings(model_path=str(self.pth)))
        frames = 12000
        microphone = np.ones((frames, 1), dtype=np.float32) * 0.1
        playback = np.empty((frames, 2), dtype=np.float32)
        rvc._callback(microphone, playback, frames, None, None)
        self.assertTrue(np.all(playback == 0))
        self.assertEqual(rvc._incoming.get_nowait().shape, (frames,))
        self.assertEqual(rvc.underruns, 1)
        rvc._outgoing.put_nowait(np.ones(frames, np.float32) * 0.2)
        rvc._callback(microphone, playback, frames, None, None)
        np.testing.assert_allclose(playback[:, 0], 0.2)
        np.testing.assert_allclose(playback[:, 1], 0.2)

    def test_validation_fails_closed_without_real_model(self):
        with self.assertRaises(ValueError):
            validate_settings(RealtimeSettings(model_path="missing.pth"))
        with self.assertRaises(ValueError):
            validate_settings(RealtimeSettings(model_path=str(self.pth), index_rate=0.5))
        with self.assertRaises(ValueError):
            validate_settings(RealtimeSettings(model_path=str(self.pth), f0_method="swiftf0"))


class NativeEntrypointTests(unittest.TestCase):
    def test_runpy_defaults_to_native_and_web_is_explicit(self):
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('parser.add_argument("--web"', source)
        self.assertIn("if not args.web:", source)
        self.assertIn("from desktop.gui import launch", source)
        gui = (ROOT / "desktop/gui.py").read_text(encoding="utf-8")
        ast.parse(gui)
        self.assertIn("class DesktopWindow(QMainWindow):", gui)
        self.assertIn("RealtimeRVC(settings)", gui)
        self.assertNotIn("QWebEngineView", gui)
        self.assertNotIn("import gradio", gui)
        reqs = (ROOT / "requirements_desktop.txt").read_text(encoding="utf-8")
        self.assertIn("PySide6", reqs)
        self.assertIn("sounddevice", reqs)
        self.assertNotIn("\ngr adio", reqs.replace("gradio", "gr adio"))
        self.assertNotIn("\nfastapi", reqs)


if __name__ == "__main__":
    unittest.main()
