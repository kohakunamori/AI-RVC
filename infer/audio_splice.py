"""Join independent renders on their original sample timeline.

Inference context is not a crossfade duration. Select a quiet, well-matched
boundary within that context and blend only the boundary. No time stretching,
sample offsets, gain normalization or additional model inference is involved.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplicePlan:
    start: int
    frames: int
    quiet: bool


def trim_render_context(raw: np.ndarray, *, content_frames: int, padding_frames: int,
                        left_context_frames: int = 0, right_context_frames: int = 0) -> np.ndarray:
    """Crop the source interval from rendered context, not from a rounded tail.

    HuBERT's feature stride can shorten the *padded* render by one or two
    frames. The desired source samples still exist inside its right padding.
    Counting backwards from that shortened tail discards real source samples
    at every chunk and shifts all following words. Absolute bounds retain them.
    """
    start = padding_frames - left_context_frames
    end = padding_frames + content_frames + right_context_frames
    if (min(content_frames, padding_frames, left_context_frames, right_context_frames) < 0
            or start < 0 or end > len(raw)):
        raise ValueError("Rendered context does not cover the requested source interval")
    return raw[start:end]


def _window_mean(values: np.ndarray, frames: int) -> np.ndarray:
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (sums[frames:] - sums[:-frames]) / frames


def plan_splice(left: np.ndarray, right: np.ndarray, sample_rate: int) -> SplicePlan:
    """Plan once from the two aligned vocal contexts, then reuse for all stems.

    Quiet boundaries use up to 20 ms. With continuous singing, use the least
    different 2 ms boundary, avoiding long cancellation of independent phases.
    Compare channel energies, not mono sums (stereo can itself be anti-phase).
    Keep distance from context edges; prefer the centre only to break ties.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    if left.shape != right.shape or not len(left) or sample_rate <= 0:
        raise ValueError("Splice contexts must have the same non-empty shape and sample rate")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Splice contexts contain non-finite samples")
    size = len(left)
    energy = np.maximum(np.mean(left.astype(np.float64) ** 2, axis=1),
                        np.mean(right.astype(np.float64) ** 2, axis=1))
    wide = min(size, max(2, round(sample_rate * 0.020)))
    means = _window_mean(energy, wide)
    margin = min(round(sample_rate * 0.5), max(0, (size - wide) // 4))
    candidates = np.arange(margin, len(means) - margin)
    reference = float(np.percentile(means, 95))
    quiet_limit = max(1e-12, reference * 1e-4)  # -40 dB, both renders
    guard = round(sample_rate * 0.010)
    quiet = candidates[(means[candidates] <= quiet_limit)
                       & (means[np.maximum(0, candidates - guard)] <= quiet_limit)
                       & (means[np.minimum(len(means) - 1, candidates + guard)] <= quiet_limit)]
    if quiet.size:
        # The quiet run must contain the whole fade; no fade into a word onset.
        start = int(quiet[np.argmin(np.abs(quiet + wide / 2 - size / 2))])
        return SplicePlan(start, wide, True)

    narrow = min(size, max(2, round(sample_rate * 0.002)))
    difference = np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2, axis=1)
    score = _window_mean(difference, narrow)
    power = _window_mean(energy, narrow)
    margin = min(round(sample_rate * 0.5), max(0, (size - narrow) // 4))
    candidates = np.arange(margin, len(score) - margin)
    # Small energy preference chooses softer material when mismatch is similar.
    # The tie-breaker cannot outweigh an actual waveform mismatch.
    cost = score[candidates] + 0.01 * power[candidates]
    cost += max(reference, 1e-24) * 1e-9 * np.abs(candidates + narrow / 2 - size / 2) / size
    return SplicePlan(int(candidates[np.argmin(cost)]), narrow, False)


def splice_contexts(left: np.ndarray, right: np.ndarray, overlap: int,
                    plan: SplicePlan) -> np.ndarray:
    """Discard redundant context while preserving every original timeline sample."""
    left, right = np.asarray(left), np.asarray(right)
    if not (np.issubdtype(left.dtype, np.floating) and np.issubdtype(right.dtype, np.floating)):
        raise ValueError("Splicing requires floating-point audio")
    if overlap < 1 or overlap > min(len(left), len(right)):
        raise ValueError("Invalid shared audio context")
    end = plan.start + plan.frames
    if plan.start < 0 or plan.frames < 1 or end > overlap:
        raise ValueError("Splice plan is outside the shared context")
    dtype = np.result_type(left.dtype, right.dtype, np.float32)
    base = len(left) - overlap
    # Complementary weights preserve identical renders without a +3 dB bump.
    fade = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, plan.frames, dtype=dtype)))
    if left.ndim > 1:
        fade = fade[:, None]
    a = left[base + plan.start:base + end].astype(dtype, copy=False)
    b = right[plan.start:end].astype(dtype, copy=False)
    mixed = a + (b - a) * fade
    return np.concatenate((left[:base + plan.start], mixed, right[end:])).astype(dtype, copy=False)
