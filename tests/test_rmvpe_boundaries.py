"""Boundary verification must distinguish predictor anticipation from real notes."""
from pathlib import Path
import importlib.util

import numpy as np
import unittest
import tempfile
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "infer" / "rmvpe_boundaries.py"
spec = importlib.util.spec_from_file_location("rmvpe_boundaries_under_test", MODULE)
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)

SR = 16000
TIME = np.arange(2 * SR) / SR
FRAME_TIME = np.arange(201) * 0.01


def render(frequency, harmonics=(1.0, 0.5, 0.3, 0.2)):
    phase = 2 * np.pi * np.cumsum(frequency) / SR
    return sum(a * np.sin((i + 1) * phase + 0.13 * i) for i, a in enumerate(harmonics)) * 0.2


def transition_with_gap(kind):
    audio = render(np.where(TIME < 0.8, 525.0, 350.0))
    f0 = np.where(FRAME_TIME < 0.76, 525.0, 350.0)
    gap = (FRAME_TIME >= 0.53) & (FRAME_TIME < 0.76)
    f0[gap] = 0
    gap_samples = (TIME >= 0.53) & (TIME < 0.76)
    if kind == "silence":
        audio[gap_samples] = 0
    elif kind == "noise":
        audio[gap_samples] = np.random.default_rng(613).normal(0, 0.2, gap_samples.sum())
    elif kind == "consonant":
        consonant = (TIME >= 0.64) & (TIME < 0.65)
        audio[consonant] = np.random.default_rng(614).normal(0, 0.2, consonant.sum())
    elif kind == "different_note":
        audio = render(np.where(TIME < 0.53, 525.0, np.where(TIME < 0.76, 440.0, np.where(TIME < 0.8, 525.0, 350.0))))
    return audio, f0, gap


def legacy_decode(salience, mapping, threshold):
    center = np.argmax(salience, axis=1) + 4
    padded = np.pad(salience, ((0, 0), (4, 4)))
    local = np.array([padded[i, c - 4:c + 5] for i, c in enumerate(center)])
    cents = np.array([mapping[c - 4:c + 5] for c in center])
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sum(local * cents, axis=1) / np.sum(local, axis=1)
    result[np.max(padded, axis=1) <= threshold] = 0
    return result


class TelkNetRmvpeBoundariesTests(unittest.TestCase):
    def test_real_note_changes_are_not_delayed(self):
        for old in [130.0, 220.0, 350.0, 440.0, 525.0, 880.0]:
            for ratio in [0.5, 2 / 3, 1.5, 2.0]:
                for noise in [0.0, 0.01]:
                    with self.subTest(old=old, ratio=ratio, noise=noise):
                        new = old * ratio
                        audio = render(np.where(TIME < 0.8, old, new))
                        audio += np.random.default_rng(61011).normal(0, noise, len(audio))
                        f0 = np.where(FRAME_TIME < 0.8, old, new)
                        result, changes = boundaries.correct_premature_boundaries(audio, f0)
                        assert np.array_equal(result, f0)
                        assert changes.size == 0

    def test_steady_notes_and_continuous_glides_are_unchanged(self):
        for frequency in [130.0, 220.0, 350.0, 440.0, 525.0, 880.0]:
            for glide in [False, True]:
                with self.subTest(frequency=frequency, glide=glide):
                    hz = frequency * 2 ** ((TIME - 1) * 0.5) if glide else np.full(len(TIME), frequency)
                    f0 = frequency * 2 ** ((FRAME_TIME - 1) * 0.5) if glide else np.full(len(FRAME_TIME), frequency)
                    result, changes = boundaries.correct_premature_boundaries(render(hz), f0)
                    assert np.array_equal(result, f0)
                    assert changes.size == 0

    def test_note_is_not_extended_into_unvoiced_content(self):
        for kind in ['silence', 'white_noise', 'highband_noise']:
            with self.subTest(kind=kind):
                audio = render(np.full(len(TIME), 525.0))
                mask = TIME >= 0.8
                rng = np.random.default_rng(611011)
                if kind == "silence":
                    audio[mask] = 0
                elif kind == "white_noise":
                    audio[mask] = rng.normal(0, 0.2, mask.sum())
                else:
                    audio[mask] = np.sin(2 * np.pi * 7100 * TIME[mask]) * rng.normal(0, 0.1, mask.sum())
                f0 = np.where(FRAME_TIME < 0.8, 525.0, 350.0)
                result, changes = boundaries.correct_premature_boundaries(audio, f0)
                assert np.array_equal(result, f0)
                assert changes.size == 0

    def test_premature_prediction_is_corrected_only_before_the_actual_note(self):
        for wrong_frequency in [315.0, 350.0]:
            with self.subTest(wrong_frequency=wrong_frequency):
                audio = render(np.where(TIME < 0.8, 525.0, 350.0))
                f0 = np.where(FRAME_TIME < 0.76, 525.0, 350.0)
                f0[(FRAME_TIME >= 0.76) & (FRAME_TIME < 0.8)] = wrong_frequency
                result, changes = boundaries.correct_premature_boundaries(audio, f0)
                assert changes.size > 0
                assert np.all((FRAME_TIME[changes] >= 0.76) & (FRAME_TIME[changes] < 0.8))
                assert np.all(result[changes] == 525.0)
                assert np.array_equal(result[FRAME_TIME >= 0.8], f0[FRAME_TIME >= 0.8])

    def test_octave_drop_with_a_weak_fundamental_is_retained(self):
        audio = render(np.where(TIME < 0.8, 525.0, 262.5), (0.05, 1.0, 0.02, 0.7))
        f0 = np.where(FRAME_TIME < 0.8, 525.0, 262.5)
        result, changes = boundaries.correct_premature_boundaries(audio, f0)
        assert np.array_equal(result, f0)
        assert changes.size == 0

    def test_rapid_alternating_notes_do_not_reuse_an_earlier_anchor(self):
        for step_seconds in [0.03, 0.05, 0.08]:
            with self.subTest(step_seconds=step_seconds):
                frequencies = np.array([220.0, 440.0, 330.0, 525.0])
                sample_notes = frequencies[np.floor(TIME / step_seconds).astype(int) % len(frequencies)]
                frame_notes = frequencies[np.floor(FRAME_TIME / step_seconds).astype(int) % len(frequencies)]
                result, changes = boundaries.correct_premature_boundaries(render(sample_notes), frame_notes)
                assert np.array_equal(result, frame_notes)
                assert changes.size == 0

    def test_existing_small_pitch_modulation_and_distortion_are_not_flattened(self):
        sample_notes = 350 * 2 ** (0.12 * np.sin(2 * np.pi * 6 * TIME))
        frame_notes = 350 * 2 ** (0.12 * np.sin(2 * np.pi * 6 * FRAME_TIME))
        distorted = np.tanh(5 * render(sample_notes))
        result, changes = boundaries.correct_premature_boundaries(distorted, frame_notes)
        assert np.array_equal(result, frame_notes)
        assert changes.size == 0

    def test_unvoiced_frames_are_never_filled(self):
        for gap in [0.05, 0.23, 0.5]:
            with self.subTest(gap=gap):
                audio = render(np.where(TIME < 0.8, 525.0, 350.0))
                f0 = np.where(FRAME_TIME < 0.76, 525.0, 350.0)
                f0[(FRAME_TIME >= 0.76 - gap) & (FRAME_TIME < 0.76)] = 0
                result, changes = boundaries.correct_premature_boundaries(audio, f0)
                assert np.array_equal(result == 0, f0 == 0)
                if gap > 0.32:
                    assert changes.size == 0

    def test_continuously_supported_gap_and_premature_boundary_are_corrected_together(self):
        audio, f0, gap = transition_with_gap("periodic")
        result, changes, restored = boundaries.correct_rmvpe_transitions(audio, f0)
        assert changes.size > 0
        assert np.array_equal(restored, np.flatnonzero(gap))
        assert np.all(result[gap] == 525.0)
        assert np.array_equal(result[FRAME_TIME >= 0.8], f0[FRAME_TIME >= 0.8])

    def test_unsupported_frame_anywhere_in_a_gap_blocks_recovery(self):
        for kind in ['silence', 'noise', 'consonant', 'different_note']:
            with self.subTest(kind=kind):
                audio, f0, gap = transition_with_gap(kind)
                result, _, restored = boundaries.correct_rmvpe_transitions(audio, f0)
                assert restored.size == 0
                assert np.array_equal(result[gap], f0[gap])

    def test_gap_without_a_verified_premature_boundary_is_untouched(self):
        audio, f0, gap = transition_with_gap("periodic")
        # Here the new note actually begins where the predictor says it does.
        audio = render(np.where(TIME < 0.76, 525.0, 350.0))
        result, changes, restored = boundaries.correct_rmvpe_transitions(audio, f0)
        assert changes.size == 0 and restored.size == 0
        assert np.array_equal(result, f0)

    def test_vector_decoder_preserves_legacy_values_including_edge_bins(self):
        for dtype in [np.float32, np.float64]:
            for threshold in [0.0, 0.02, 0.05]:
                with self.subTest(dtype=dtype, threshold=threshold):
                    rng = np.random.default_rng(61011)
                    salience = rng.random((70, 360)).astype(dtype) * 0.1
                    salience[:3] = 0
                    salience[3] = threshold
                    salience[4, 0] = 1
                    salience[5, -1] = 1
                    mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
                    actual = boundaries.local_average_cents(salience, mapping, threshold)
                    expected = legacy_decode(salience, mapping, threshold)
                    assert np.array_equal(actual, expected)

    def test_invalid_boundary_inputs_are_rejected(self):
        for which in ['shape', 'frame_count', 'audio_nan', 'pitch_nan']:
            with self.subTest(which=which):
                audio = np.zeros(32000)
                f0 = np.zeros(201)
                if which == "shape":
                    audio = audio[:, None]
                elif which == "frame_count":
                    f0 = f0[:-1]
                elif which == "audio_nan":
                    audio[0] = np.nan
                else:
                    f0[0] = np.nan
                with self.assertRaises(ValueError):
                    boundaries.correct_premature_boundaries(audio, f0)
