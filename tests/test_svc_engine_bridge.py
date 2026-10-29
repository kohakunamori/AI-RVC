"""Contract tests: simulate local engines; never download heavyweight checkpoints."""
from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
# The main infer/__init__.py imports torch; these tests intentionally avoid
# installing the full GPU stack by importing the standalone bridge file.
import importlib.util
module_name = "_svc_bridge_under_test"
bridge_path = Path(__file__).resolve().parents[1] / "infer" / "svc_engine_bridge.py"
spec = importlib.util.spec_from_file_location(module_name, bridge_path)
bridge_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = bridge_module
spec.loader.exec_module(bridge_module)
SVCJob, convert_vocals = bridge_module.SVCJob, bridge_module.convert_vocals

class TestLocalSVCBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "src.wav"
        self.target = self.root / "ref.wav"
        x = np.sin(2*np.pi*330*np.arange(4410)/44100).astype("float32")
        sf.write(self.source, x, 44100, subtype="FLOAT")
        sf.write(self.target, x, 44100, subtype="FLOAT")
        self.engine_repo = self.root/"engine"
        self.engine_repo.mkdir()
        (self.engine_repo/"inference.py").write_text("# mock")
        self.python = Path(sys.executable).resolve()

    def job(self):
        return SVCJob(engine="seed_vc", source_vocals=self.source, target_reference=self.target,
                      output_wav=self.root/"converted.wav", engine_repository=self.engine_repo,
                      engine_python=self.python, pitch_shift=0, fp16=False)

    def fake_run(self, args, **kwargs):
        self.assertEqual(Path(kwargs["cwd"]).resolve(), self.engine_repo.resolve())
        self.assertNotIn("shell", kwargs)
        target_dir = Path(args[args.index("--output")+1])
        (target_dir/"converted").mkdir(exist_ok=True)
        sf.write(target_dir/"converted"/"vocal.wav",
                 np.ones(4410, dtype="float32")*0.4, 44100, subtype="PCM_16")
        return subprocess.CompletedProcess(args, 0, "", "")

    def test_seed_preserves_pitch_and_emits_float(self):
        with patch("_svc_bridge_under_test.subprocess.run", side_effect=self.fake_run) as mock:
            output = convert_vocals(self.job())
        self.assertEqual(sf.info(output).subtype, "FLOAT")
        self.assertEqual(sf.info(output).samplerate, 44100)
        args = mock.call_args.args[0]
        self.assertEqual(args[args.index("--f0-condition")+1], "True")
        self.assertEqual(args[args.index("--auto-f0-adjust")+1], "False")
        self.assertEqual(args[args.index("--semi-tone-shift")+1], "0")

    def test_yingmusic_uses_pitch_preserving_worker(self):
        (self.engine_repo/"my_inference.py").write_text("# mock")
        checkpoint = self.root/"full.pt"
        config = self.root/"config.yml"
        checkpoint.write_bytes(b"test")
        config.write_text("dummy")
        with self.assertRaisesRegex(ValueError, "requires a locally installed"):
            convert_vocals(replace(self.job(), engine="yingmusic_svc"))
        job = replace(self.job(), engine="yingmusic_svc", checkpoint=checkpoint,
                      model_config=config, pitch_shift=2)
        with patch("_svc_bridge_under_test.subprocess.run", side_effect=self.fake_run) as mock:
            output = convert_vocals(job)
        self.assertTrue(output.is_file())
        args = mock.call_args.args[0]
        self.assertTrue(any("yingmusic_svc_worker.py" in token for token in args))
        self.assertEqual(args[args.index("--pitch-shift")+1], "2")

    def test_soulx_uses_own_f0_worker_and_single_generated_wav(self):
        (self.engine_repo/"cli").mkdir()
        (self.engine_repo/"cli"/"inference_svc.py").write_text("# mock")
        checkpoint, cfg = self.root/"soulx.pt", self.root/"soulx.yml"
        checkpoint.write_bytes(b"test")
        cfg.write_text("model: test")
        job = replace(self.job(), engine="soulx_singer",
                      checkpoint=checkpoint, model_config=cfg)
        def fake_soulx(args, **kwargs):
            output = Path(args[args.index("--output")+1])
            (output/"generated").mkdir()
            (output/"preprocess_prompt").mkdir()
            sf.write(output/"generated"/"generated.wav",
                     np.ones(4410, dtype="float32")*0.2, 44100, subtype="FLOAT")
            sf.write(output/"preprocess_prompt"/"vocal.wav",
                     np.ones(4410, dtype="float32")*0.3, 44100, subtype="FLOAT")
            return subprocess.CompletedProcess(args, 0, "", "")
        with patch("_svc_bridge_under_test.subprocess.run", side_effect=fake_soulx) as mock:
            output = convert_vocals(job)
        self.assertEqual(sf.info(output).subtype, "FLOAT")
        args = mock.call_args.args[0]
        self.assertTrue(any("soulx_svc_worker.py" in arg for arg in args))
        self.assertEqual(args[args.index("--pitch-shift")+1], "0")

    def test_does_not_silently_fallback_after_failure(self):
        fail = subprocess.CompletedProcess([], 2, "failed", "")
        with patch("_svc_bridge_under_test.subprocess.run", return_value=fail):
            with self.assertRaisesRegex(RuntimeError, "exited 2"):
                convert_vocals(self.job())
        self.assertFalse((self.root/"converted.wav").exists())

    def test_protects_existing_audio_and_missing_reference(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            convert_vocals(replace(self.job(), output_wav=self.source))
        with self.assertRaises(FileNotFoundError):
            convert_vocals(replace(self.job(), target_reference=self.root/"missing.wav"))

    def test_rejects_nan_audio(self):
        def broken(args, **kwargs):
            out = Path(args[args.index("--output")+1])
            sf.write(out/"broken.wav", np.array([np.nan]), 44100, subtype="FLOAT")
            return subprocess.CompletedProcess(args, 0, "", "")
        with patch("_svc_bridge_under_test.subprocess.run", side_effect=broken):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                convert_vocals(self.job())

if __name__ == "__main__":
    unittest.main()
