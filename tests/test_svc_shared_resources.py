"""Shared-venv, shared-weight-cache and private reference-feature contracts.

Runs with stdlib, numpy and soundfile; never imports or downloads PyTorch.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

def direct(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

registry = direct("_svc_registry_shared_test", "tools/svc_model_registry.py")
cache = direct("_svc_reference_cache_test", "tools/svc_reference_cache.py")


class SharedEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.file = self.base / "engines.json"
        self.repo = self.base / "seed"
        self.repo.mkdir()
        (self.repo / "inference.py").write_text("# mock")
        self.python = self.base / "python"
        self.python.write_bytes(b"stub")
        self.cache = self.base / "weights"

    def test_common_python_used_by_default_with_per_engine_override(self):
        shared = registry.save_shared_config({
            "engine_python": str(self.python),
            "cache_directory": str(self.cache),
        }, self.file)
        self.assertEqual(registry.shared_cache_dir(shared), self.cache.resolve())
        saved = registry.save_engine_config("seed_vc", {
            "engine_repository": str(self.repo),
        }, self.file)
        self.assertTrue(registry.engine_health("seed_vc", saved)[0])
        self.assertEqual(registry.shared_python(saved, "seed_vc"), str(self.python))
        override = self.base / "override"
        override.write_bytes(b"stub")
        saved = registry.save_engine_config("seed_vc", {
            "engine_python": str(override),
        }, self.file)
        self.assertEqual(registry.shared_python(saved, "seed_vc"), str(override))
        self.assertEqual(registry.get_engine_job_options("seed_vc", saved)["engine_python"], override)

    def test_legacy_json_remains_compatible_and_rejects_arbitrary_fields(self):
        self.file.write_text(json.dumps({"seed_vc": {"engine_repository": "/tmp"}}))
        data = registry.load_registry(self.file)
        self.assertEqual(data["_shared"], {"engine_python": "", "cache_directory": ""})
        with self.assertRaises(ValueError):
            registry.save_shared_config({"shell_command": "rm -rf"}, self.file)
        with self.assertRaises(ValueError):
            registry.save_shared_config({"engine_python": []}, self.file)


class ReferenceF0CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.voice = self.base / "reference.wav"
        self.config = self.base / "model.yml"
        self.impl = self.base / "pipeline.py"
        sf.write(self.voice, np.zeros(2048, np.float32), 44100)
        self.config.write_text("model: current")
        self.impl.write_text("preprocess-v1")
        self.calls = 0

    def prep(self, directory):
        self.calls += 1
        directory.mkdir(parents=True, exist_ok=True)
        vocal = directory / "vocal.wav"
        f0 = directory / "vocal_f0.npy"
        sf.write(vocal, np.zeros(2048, np.float32), 44100)
        np.save(f0, np.array([120, 130], dtype=np.float32))
        return vocal, f0

    def render(self):
        return cache.prepare_soulx_reference(
            self.voice, self.config, self.impl,
            self.base / "fallback", self.prep,
        )

    def test_repeated_target_uses_same_reference_F0_and_audio(self):
        with patch.dict(os.environ, {"AI_RVC_SVC_CACHE_DIR": str(self.base / "shared"),
                                     "AI_RVC_REFERENCE_CACHE": "1"}):
            a = self.render()
            b = self.render()
        self.assertEqual(a, b)
        self.assertEqual(self.calls, 1)
        self.assertTrue(a[0].is_file())
        self.assertTrue(a[1].is_file())

    def test_invalidates_when_audio_or_preprocess_changes(self):
        with patch.dict(os.environ, {"AI_RVC_SVC_CACHE_DIR": str(self.base / "shared"),
                                     "AI_RVC_REFERENCE_CACHE": "1"}):
            first = self.render()
            self.impl.write_text("preprocess-v2")
            second = self.render()
            self.config.write_text("model: revised")
            third = self.render()
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(self.calls, 3)

    def test_cache_can_be_disabled_without_persisting_voice(self):
        with patch.dict(os.environ, {"AI_RVC_SVC_CACHE_DIR": str(self.base / "shared"),
                                     "AI_RVC_REFERENCE_CACHE": "0"}):
            a = self.render()
            b = self.render()
        self.assertEqual(a, b)
        self.assertEqual(self.calls, 2)
        self.assertFalse((self.base / "shared").exists())


if __name__ == "__main__":
    unittest.main()
