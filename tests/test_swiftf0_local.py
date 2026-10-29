"""Tests for the optional, fully local SwiftF0 -> RVC pitch-grid bridge."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from configs.schema import validate_config
from infer.f0_extractor import SwiftF0Extractor, get_f0_extractor


class TestSwiftF0Local(unittest.TestCase):
    @staticmethod
    def fake_module(times, pitch, confidence):
        fake = types.ModuleType("swift_f0")

        class SwiftF0:
            def detect(self, audio, sample_rate, fmin, fmax):
                assert sample_rate == 16000
                assert fmin == 50
                assert fmax == 1100
                assert audio.dtype == np.float32
                return types.SimpleNamespace(
                    timestamps=np.array(times), pitch_hz=np.array(pitch),
                    confidence=np.array(confidence),
                )

        fake.SwiftF0 = SwiftF0
        return fake

    def test_10ms_interpolation_preserves_unvoiced_regions(self):
        stub = self.fake_module(
            [0.0, 0.016, 0.032, 0.048],
            [100.0, 110.0, 120.0, 130.0],
            [1.0, 1.0, 0.0, 1.0],
        )
        with patch.dict(sys.modules, {"swift_f0": stub}):
            f0 = get_f0_extractor("swiftf0", device="cpu").extract(np.zeros(800, dtype=np.float32))
        self.assertEqual(f0.dtype, np.dtype("float32"))
        np.testing.assert_allclose(f0[:2], [100, 106.25], atol=1e-3)
        np.testing.assert_array_equal(f0[2:], np.zeros(4, dtype=np.float32))

    def test_silence_and_missing_dependency(self):
        stub = self.fake_module([0, 0.016], [100, 100], [1, 1])
        with patch.dict(sys.modules, {"swift_f0": stub}):
            self.assertEqual(SwiftF0Extractor().extract(np.array([], dtype=np.float32)).tolist(), [0])
        with patch.dict(sys.modules, {"swift_f0": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements_sota.txt"):
                SwiftF0Extractor()

    def test_strict_rvc_mode_routing(self):
        self.assertEqual(
            validate_config({"cover": {"use_official": False, "f0_method": "swiftf0"}})["cover"]["f0_method"],
            "swiftf0",
        )
        with self.assertRaises(ValueError):
            validate_config({"cover": {"use_official": True, "f0_method": "swiftf0"}})
        self.assertEqual(validate_config({"cover": {"dereverb_mode": "off"}})["cover"]["dereverb_mode"], "off")
        with self.assertRaises(ValueError):
            validate_config({"cover": {"dereverb_mode": "automatic"}})


if __name__ == "__main__":
    unittest.main()
