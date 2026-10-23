"""Waveform checks for premature RMVPE note changes in offline conversion.

The original neural track supplies every anchor. Corrected frames cannot
extend a hold or become a new pitch hypothesis. A preceding prediction gap is
restored only when the same original note is verified continuously across it.
"""
from __future__ import annotations

import numpy as np


def local_average_cents(salience: np.ndarray, cents_mapping: np.ndarray, threshold: float) -> np.ndarray:
    """Compute the original nine-bin decoder without copying the full matrix."""
    centers = np.argmax(salience, axis=1)
    bins = centers[:, None] + np.arange(-4, 5)
    local = np.take_along_axis(salience, np.clip(bins, 0, salience.shape[1] - 1), axis=1)
    local[(bins < 0) | (bins >= salience.shape[1])] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = np.sum(local * cents_mapping[bins + 4], axis=1) / np.sum(local, axis=1)
    # The original decoder includes zero padding in the peak calculation.
    peak = np.maximum(salience[np.arange(len(salience)), centers], 0)
    cents[peak <= threshold] = 0
    return cents


def _next_period_support(frames: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Check whether the new period is already present after the frame center."""
    size = frames.shape[1]
    coordinates = np.arange(size)[None, :]
    best = np.full(len(frames), -1.0)
    for offset in (-1, 0, 1):
        lag = np.maximum(1, periods + offset)[:, None]
        length = size - lag
        valid = (coordinates < length) & (length > 1)
        window = (0.5 + 0.5 * np.cos(np.pi * (2 * coordinates / np.maximum(length - 1, 1) - 1))) * valid
        mass = np.maximum(window.sum(axis=1, keepdims=True), 1e-20)
        right = np.take_along_axis(frames, np.minimum(coordinates + lag, size - 1), axis=1)
        left = frames - np.sum(frames * window, axis=1, keepdims=True) / mass
        right = right - np.sum(right * window, axis=1, keepdims=True) / mass
        denominator = np.sqrt(np.sum(left * left * window, axis=1) * np.sum(right * right * window, axis=1))
        correlation = np.divide(np.sum(left * right * window, axis=1), denominator,
                                out=np.zeros_like(denominator), where=denominator > 1e-20)
        best = np.maximum(best, correlation)
    return best


def correct_premature_boundaries(audio: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject a short note change contradicted by a locally dominant old note.

    Input is the same padded 16 kHz mono waveform used by RMVPE, with centered
    10 ms pitch frames. A stable ORIGINAL note must precede a >=3 semitone jump
    by at most 32 missing frames. Examine at most the next 80 ms, and require
    >=90% local harmonic support for that note versus <70% for the new pitch.
    Actual new-note periodicity after the center vetoes a correction, including
    ordinary octave changes. Unvoiced frames and unsupported cases are retained.

    This does not synthesize effects, remove noise, or alter the output gain.
    It is deliberately not used by training or streaming inference.
    """
    samples = np.asarray(audio)
    original = np.asarray(f0)
    if samples.ndim != 1 or original.ndim != 1:
        raise ValueError("RMVPE boundary verification requires mono audio and a one-dimensional pitch track")
    if len(samples) < 2 or len(original) != len(samples) // 160 + 1:
        raise ValueError("RMVPE boundary verification audio and pitch frame counts differ")
    if not np.isfinite(samples).all() or not np.isfinite(original).all():
        raise ValueError("RMVPE boundary verification requires finite audio and pitch")
    corrected = original.copy()
    accepted = np.flatnonzero(original > 0)
    empty = np.empty(0, dtype=np.intp)
    if len(accepted) < 4:
        return corrected, empty

    starts, anchors = accepted[3:], accepted[2:-1]
    stable = np.maximum(
        np.abs(12 * np.log2(original[accepted[1:-2]] / original[anchors])),
        np.abs(12 * np.log2(original[accepted[:-3]] / original[anchors])),
    ) <= 0.3
    stable &= (anchors - accepted[:-3]) <= 4
    possible = (stable & (starts - anchors <= 33)
                & (np.abs(12 * np.log2(original[starts] / original[anchors])) >= 3)
                # A 32 ms window must contain at least four old-note cycles.
                & (original[anchors] >= 125))
    starts, anchors = starts[possible], anchors[possible]
    if not len(starts):
        return corrected, empty
    rows = (starts[:, None] + np.arange(8)).ravel()
    anchors = np.repeat(anchors, 8)
    valid = rows < len(original)
    rows, anchors = rows[valid], anchors[valid]
    valid = original[rows] > 0
    rows, anchors = rows[valid], anchors[valid]
    valid = np.abs(12 * np.log2(original[rows] / original[anchors])) >= 3
    rows, anchors = rows[valid], anchors[valid]
    rows, first = np.unique(rows, return_index=True)
    anchors = anchors[first]

    coordinates = np.arange(-256, 256)
    window = np.hanning(512)
    frequencies = np.fft.rfftfreq(2048, 1 / 16000)
    band = (frequencies >= 125) & (frequencies <= 4000)
    frequency = frequencies[band]
    half_bandwidth = 1.5 * 16000 / 512

    def frames_at(selected: np.ndarray, offsets: np.ndarray = coordinates) -> np.ndarray:
        indices = selected[:, None] * 160 + offsets
        end = len(samples) - 1
        # Materialize selected frames only, not a padded copy of the full song.
        reflected = np.abs((indices + end) % (2 * end) - end)
        return samples[reflected].astype(np.float64, copy=False)

    changed = []
    for start in range(0, len(rows), 128):
        selected, reference = rows[start:start + 128], anchors[start:start + 128]
        frames = frames_at(selected)
        anchor_frames = frames_at(reference)
        centered = frames - np.sum(frames * window, axis=1, keepdims=True) / window.sum()
        spectrum = np.abs(np.fft.rfft(centered * window, n=2048, axis=1)) ** 2
        spectrum = spectrum[:, band]

        def harmonic_support(pitch: np.ndarray) -> np.ndarray:
            harmonic = np.rint(frequency[None, :] / pitch[:, None])
            distance = np.abs(frequency[None, :] - harmonic * pitch[:, None])
            mask = (harmonic >= 1) & (distance <= half_bandwidth)
            return np.sum(spectrum * mask, axis=1) / np.maximum(spectrum.sum(axis=1), 1e-20)

        old_support = harmonic_support(original[reference])
        new_support = harmonic_support(original[selected])
        center_power = np.mean(frames[:, 176:336].reshape(-1, 2, 80) ** 2, axis=2).min(axis=1)
        anchor_power = np.mean(anchor_frames ** 2, axis=1)
        use = ((old_support >= 0.90) & (new_support < 0.70)
               & (center_power > 1e-6) & (center_power >= 0.25 * anchor_power))
        if not np.any(use):
            continue
        candidates = selected[use]
        following = frames_at(candidates, np.arange(512))
        new_period = np.rint(16000 / original[candidates]).astype(np.intp)
        retains_new_note = _next_period_support(following, new_period) >= 0.75
        chosen = candidates[~retains_new_note]
        corrected[chosen] = original[reference[use][~retains_new_note]]
        changed.extend(chosen.tolist())
    return corrected, np.asarray(changed, dtype=np.intp)


def correct_rmvpe_transitions(audio: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct a premature boundary without leaving a verified note fragmented.

    Gap recovery requires BOTH an original stable note and its waveform-verified
    continuation at a premature boundary. Every intervening frame must support
    that same note. Any breath, silence or unsupported frame rejects the entire
    gap, so a corrected value cannot start an iterative voicing extension.
    """
    corrected, boundaries = correct_premature_boundaries(audio, f0)
    original = np.asarray(f0)
    empty = np.empty(0, dtype=np.intp)
    starts = boundaries[(boundaries > 0) & (original[np.maximum(boundaries - 1, 0)] == 0)]
    if not len(starts):
        return corrected, boundaries, empty
    voiced = np.flatnonzero(original > 0)
    restored = []
    coordinates = np.arange(-256, 256)
    window = np.hanning(512)
    frequencies = np.fft.rfftfreq(2048, 1 / 16000)
    band = (frequencies >= 125) & (frequencies <= 4000)
    frequencies = frequencies[band]
    samples = np.asarray(audio)
    end = len(samples) - 1
    for start in starts:
        anchor = voiced[np.searchsorted(voiced, start) - 1]
        count = start - anchor - 1
        if not 1 <= count <= 32 or corrected[start] != original[anchor]:
            continue
        rows = np.arange(anchor + 1, start)
        indices = rows[:, None] * 160 + coordinates
        reflected = np.abs((indices + end) % (2 * end) - end)
        frames = samples[reflected].astype(np.float64, copy=False)
        anchor_indices = anchor * 160 + coordinates
        anchor_frame = samples[np.abs((anchor_indices + end) % (2 * end) - end)]
        center_power = np.mean(frames[:, 176:336].reshape(-1, 2, 80) ** 2, axis=2).min(axis=1)
        anchor_power = np.mean(anchor_frame.astype(np.float64) ** 2)
        if not np.all((center_power > 1e-6) & (center_power >= 0.25 * anchor_power)):
            continue
        centered = frames - np.sum(frames * window, axis=1, keepdims=True) / window.sum()
        spectrum = np.abs(np.fft.rfft(centered * window, n=2048, axis=1)) ** 2
        spectrum = spectrum[:, band]
        pitch = original[anchor]
        harmonic = np.rint(frequencies / pitch)
        mask = (harmonic >= 1) & (np.abs(frequencies - harmonic * pitch) <= 46.875)
        support = np.sum(spectrum[:, mask], axis=1) / np.maximum(spectrum.sum(axis=1), 1e-20)
        if not np.all(support >= 0.95):
            continue
        periods = np.full(count, round(16000 / pitch), dtype=np.intp)
        if not np.all(_next_period_support(frames, periods) >= 0.92):
            continue
        # Require both halves to be periodic too, not just a tonal neighbour
        # dominating the 32 ms window around a brief consonant or interruption.
        if not all(np.all(_next_period_support(half, periods) >= 0.88)
                   for half in (frames[:, :256], frames[:, 256:])):
            continue
        corrected[rows] = pitch
        restored.extend(rows.tolist())
    return corrected, boundaries, np.asarray(restored, dtype=np.intp)
