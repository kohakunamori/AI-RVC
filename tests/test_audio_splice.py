import importlib.util
from pathlib import Path
import sys

import numpy as np
import unittest
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rvc_audio_splice", Path(__file__).resolve().parents[1] / "infer/audio_splice.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
plan_splice, splice_contexts = module.plan_splice, module.splice_contexts
trim_render_context = module.trim_render_context


class TelkNetAudioSpliceTests(unittest.TestCase):
    def test_three_chunk_source_timeline_survives_hubert_tail_rounding(self):
        for rounding_frames in [0, 400, 800]:
            with self.subTest(rounding_frames=rounding_frames):
                sr, pad, context = 40000, 120000, 6000
                # Reproduce the Far far away first-half cut points, including fractional
                # final source frame. Missing *padding* must not shorten any source chunk.
                edges = [0, 2292400, 4496000, 4939673]
                source = np.linspace(-.4, .4, edges[-1], dtype=np.float64)
                padded = np.pad(source, (pad, pad), mode='reflect')
                chunks = []
                for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
                    raw = padded[start:end + 2 * pad - rounding_frames]
                    chunks.append(trim_render_context(
                        raw, content_frames=end-start, padding_frames=pad,
                        left_context_frames=context if index else 0,
                        right_context_frames=context if index < 2 else 0))
                merged = chunks[0]
                for chunk in chunks[1:]:
                    plan = plan_splice(merged[-2*context:], chunk[:2*context], sr)
                    merged = splice_contexts(merged, chunk, 2*context, plan)
                np.testing.assert_array_equal(merged, source)

    def test_trim_does_not_hide_missing_source_samples_with_padding(self):
        with self.assertRaisesRegex(ValueError, "requested source interval"):
            trim_render_context(np.zeros(500), content_frames=400, padding_frames=200)

    def test_identical_stereo_context_retains_samples_and_timeline(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                rng = np.random.default_rng(17)
                original = rng.normal(0, 0.1, (24000, 2)).astype(dtype)
                left, right = original[:16000], original[8000:]
                plan = plan_splice(left[-8000:], right[:8000], 8000)
                result = splice_contexts(left, right, 8000, plan)
                assert result.dtype == dtype
                np.testing.assert_array_equal(result, original)

    def test_quiet_fade_does_not_extend_into_next_word(self):
        sr = 16000
        t = np.arange(6 * sr) / sr
        left = 0.2 * np.sin(2 * np.pi * 223 * t)
        right = -left.copy()
        left[2 * sr:3 * sr] = right[2 * sr:3 * sr] = 0
        plan = plan_splice(left, right, sr)
        assert plan.quiet
        assert 2 * sr <= plan.start < plan.start + plan.frames <= 3 * sr
        out = splice_contexts(left, right, len(left), plan)
        np.testing.assert_array_equal(out[3 * sr:], right[3 * sr:])
        np.testing.assert_array_equal(out[:2 * sr], left[:2 * sr])

    def test_continuous_opposite_phase_does_not_cancel_a_syllable(self):
        sr = 48000
        t = np.arange(6 * sr) / sr
        left = 0.2 * np.sin(2 * np.pi * 230 * t)
        right = -left
        plan = plan_splice(left, right, sr)
        out = splice_contexts(left, right, len(left), plan)
        assert not plan.quiet
        assert plan.frames <= sr * 0.002
        centre = plan.start + plan.frames // 2
        # A 100 ms syllable must keep its energy, without any gain compensation.
        window = slice(centre - sr // 20, centre + sr // 20)
        ratio = np.sum(out[window] ** 2) / np.sum(left[window] ** 2)
        assert 10 * np.log10(ratio) > -0.2
        assert np.max(np.abs(out)) <= 0.2 + 1e-12
        np.testing.assert_array_equal(out[:plan.start], left[:plan.start])
        np.testing.assert_array_equal(out[plan.start + plan.frames:], right[plan.start + plan.frames:])

    def test_same_plan_keeps_cover_and_stems_aligned(self):
        sr = 8000
        t = np.arange(sr * 2) / sr
        vocal_a = np.sin(2 * np.pi * 127 * t) * 0.1
        vocal_b = np.cos(2 * np.pi * 127 * t) * 0.1
        backing = np.sin(2 * np.pi * 271 * t) * 0.1
        plan = plan_splice(vocal_a[-sr:], vocal_b[:sr], sr)
        vocal = splice_contexts(vocal_a, vocal_b, sr, plan)
        music = splice_contexts(backing, backing, sr, plan)
        cover = splice_contexts(vocal_a + backing, vocal_b + backing, sr, plan)
        np.testing.assert_allclose(cover, vocal + music, atol=1e-15)

    def test_short_context_and_silence_have_exact_length(self):
        for count in [1, 2, 8, 31]:
            with self.subTest(count=count):
                left, right = np.zeros(count), np.zeros(count)
                plan = plan_splice(left, right, 48000)
                assert len(splice_contexts(left, right, count, plan)) == count

    def test_stereo_phase_cancellation_is_not_mistaken_for_silence(self):
        t = np.arange(32000) / 16000
        tone = 0.2 * np.sin(2 * np.pi * 219 * t)
        left = np.column_stack((tone, -tone))
        plan = plan_splice(left, -left, 16000)
        assert not plan.quiet

    def test_nonfinite_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            plan_splice(np.array([np.nan]), np.ones(1), 8000)
