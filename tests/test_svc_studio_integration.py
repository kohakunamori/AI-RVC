"""Integration contracts for the WebUI SVC selector and local model registry.

This test deliberately runs without torch/gradio/model weights and mocks only
the expensive audio components. Real GPU inference remains a separate gate.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_direct(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = load_direct("_svc_model_registry_test", "tools/svc_model_registry.py")
studio = load_direct("_svc_studio_test", "infer/svc_cover_workflow.py")


class TestSVCRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.registry_file = self.dir / "engines.json"
        self.repo = self.dir / "seed-vc"
        self.repo.mkdir()
        (self.repo / "inference.py").write_text("x", encoding="utf-8")
        self.python = self.dir / "python"
        self.python.write_text("x", encoding="utf-8")

    def test_paths_are_persisted_and_status_is_explanatory(self):
        previous = registry.load_registry(self.registry_file)
        self.assertIn("yingmusic_svc", previous)
        self.assertFalse(registry.engine_health("seed_vc", previous)[0])
        data = registry.save_engine_config("seed_vc", {
            "engine_repository": str(self.repo),
            "engine_python": str(self.python),
        }, self.registry_file)
        self.assertTrue(registry.engine_health("seed_vc", data)[0])
        self.assertEqual(registry.load_registry(self.registry_file), data)
        self.assertTrue(json.loads(self.registry_file.read_text())["seed_vc"]["engine_python"])

    def test_soulx_is_registered_and_requires_own_checkpoint(self):
        data = registry.load_registry(self.registry_file)
        self.assertIn("soulx_singer", data)
        self.assertFalse(registry.engine_health("soulx_singer", data)[0])
        self.assertIn("soulx_singer", [value for _, value in registry.engine_choices()])

    def test_yingmusic_requires_weight_and_config(self):
        data = registry.save_engine_config("yingmusic_svc", {
            "engine_repository": str(self.repo),
            "engine_python": str(self.python),
        }, self.registry_file)
        (self.repo / "my_inference.py").write_text("# worker", encoding="utf-8")
        ok, reason = registry.engine_health("yingmusic_svc", data)
        self.assertFalse(ok)
        self.assertIn("checkpoint", reason)
        ckpt, yaml_path = self.dir / "model.pt", self.dir / "model.yml"
        ckpt.write_text("x")
        yaml_path.write_text("x")
        data = registry.save_engine_config("yingmusic_svc", {
            "checkpoint": str(ckpt), "model_config": str(yaml_path),
        }, self.registry_file)
        self.assertTrue(registry.engine_health("yingmusic_svc", data)[0])

    def test_refuses_unknown_models_and_unsafe_registry_keys(self):
        with self.assertRaises(ValueError):
            registry.save_engine_config("hq_svc", {"engine_python": "whatever"}, self.registry_file)
        with self.assertRaises(ValueError):
            registry.save_engine_config("seed_vc", {"command": "rm -rf"}, self.registry_file)
        with self.assertRaises(ValueError):
            registry.load_registry(self._write_bad_json())

    def _write_bad_json(self):
        f = self.dir / "invalid.json"
        f.write_text('{"run_arbitrary_command":{"command":"bad"}}', encoding="utf-8")
        return f


class TestSVCStudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = self.dir/"song.wav"
        self.ref = self.dir/"singer.wav"
        self.src.write_bytes(b"source")
        self.ref.write_bytes(b"reference")
        self.old = {name: sys.modules.get(name) for name in (
            "infer.svc_engine_bridge", "infer.separator", "lib.mixer",
        )}
        bridge = types.ModuleType("infer.svc_engine_bridge")
        class SVCJob:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        def convert(job):
            self.assertEqual(job.pitch_shift, 0)
            self.assertTrue(job.source_vocals.exists())
            job.output_wav.write_bytes(b"mock converted audio")
        bridge.SVCJob = SVCJob
        bridge.convert_vocals = convert
        sys.modules["infer.svc_engine_bridge"] = bridge
        separator_mod = types.ModuleType("infer.separator")
        class Separator:
            def __init__(self, model_filename, device):
                self.model_filename, self.device = model_filename, device
            def separate(self, audio, directory):
                output = Path(directory)
                output.mkdir(parents=True, exist_ok=True)
                lead, backing = output/"lead.wav", output/"backing.wav"
                lead.write_bytes(b"lead")
                backing.write_bytes(b"backing")
                return str(lead), str(backing)
            def unload_model(self):
                return None
        separator_mod.KaraokeSeparator = Separator
        separator_mod.RoformerSeparator = Separator
        sys.modules["infer.separator"] = separator_mod
        mixer = types.ModuleType("lib.mixer")
        def mix(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"mix")
        mixer.mix_vocals_and_accompaniment = mix
        sys.modules["lib.mixer"] = mixer
        self.addCleanup(self.restore_modules)

    def restore_modules(self):
        for name, original in self.old.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def fake_options(self, engine):
        return {
            "engine_repository": self.dir,
            "engine_python": self.dir/"python",
            "checkpoint": None,
            "model_config": None,
        }

    def test_complete_song_with_karaoke_returns_cover_and_stems(self):
        with patch.object(studio, "get_engine_job_options", self.fake_options):
            result = studio.run_svc_cover(
                str(self.src), str(self.ref), "seed_vc", "song",
                str(self.dir/"outputs"), karaoke=True, pitch_shift=0,
            )
        self.assertTrue(Path(result["cover"]).is_file())
        self.assertTrue(Path(result["lead_vocals"]).is_file())
        self.assertTrue(Path(result["converted_vocals"]).is_file())
        self.assertTrue(Path(result["accompaniment"]).is_file())

    def test_vocals_mode_does_not_require_rvc_or_separator(self):
        with patch.object(studio, "get_engine_job_options", self.fake_options):
            result = studio.run_svc_cover(
                str(self.src), str(self.ref), "seed_vc", "vocals",
                str(self.dir/"outputs"),
            )
        self.assertIsNone(result["cover"])
        self.assertTrue(Path(result["converted_vocals"]).is_file())

    def test_bad_input_never_launches_engine(self):
        with patch.object(studio, "get_engine_job_options", self.fake_options):
            with self.assertRaises(FileNotFoundError):
                studio.run_svc_cover(str(self.src), str(self.dir/"nonexistent.wav"),
                    "seed_vc", "song", str(self.dir/"outputs"))


class TestUIWiring(unittest.TestCase):
    def test_main_cover_button_has_engine_and_reference_inputs(self):
        source = (ROOT / "ui/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn('cover_svc_engine = gr.Dropdown(', source)
        self.assertIn('cover_svc_reference = gr.Audio(', source)
        self.assertIn('with gr.Tab("🎛️ SVC 模型管理 / Models"):', source)
        self.assertIn("svc_save_btn.click(", source)
        self.assertIn("process_zero_shot_cover(", source)
        self.assertIn("cover_svc_engine,", source[source.index("cover_event = cover_btn.click"):])
        functions = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
        self.assertIn("process_cover_with_downloads", functions)
        self.assertIn("process_zero_shot_cover", functions)


if __name__ == "__main__":
    unittest.main()
