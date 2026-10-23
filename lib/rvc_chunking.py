"""Render and join RVC context on the original audio sample timeline."""
from __future__ import annotations

import numpy as np

from infer.audio_splice import plan_splice, splice_contexts, trim_render_context
from infer.quality_policy import compute_chunk_crossfade_samples


def render_aligned_segments(pipeline, model, net_g, sid, audio, audio_pad, pitch,
                            pitchf, times, index, vectors, index_rate, version,
                            protect, target_sr, cut_points):
    boundaries = sorted(set(int(t) // pipeline.window * pipeline.window for t in cut_points))
    if any(t <= 0 or t >= len(audio) for t in boundaries):
        raise ValueError("RVC chunk boundaries must be inside the source timeline")
    context = min(pipeline.t_pad_tgt, compute_chunk_crossfade_samples(
        tgt_sr=target_sr, t_pad_tgt=pipeline.t_pad_tgt,
        segment_count=len(boundaries) + 1,
    ))
    positions = [0, *boundaries, len(audio)]
    lengths = [round(end * target_sr / 16000) - round(start * target_sr / 16000)
               for start, end in zip(positions, positions[1:])]
    contexts = [min(context, left, right) for left, right in zip(lengths, lengths[1:])]
    start = 0
    result = None
    for position, end in enumerate([*boundaries, len(audio)]):
        last = position == len(boundaries)
        left_context = contexts[position - 1] if position else 0
        right_context = contexts[position] if not last else 0
        stop = None if last else end + pipeline.t_pad2 + pipeline.window
        f0_stop = None if last else (end + pipeline.t_pad2) // pipeline.window
        frame_start = start // pipeline.window
        raw = pipeline.vc(
            model, net_g, sid, audio_pad[start:stop],
            pitch[:, frame_start:f0_stop] if pitch is not None else None,
            pitchf[:, frame_start:f0_stop] if pitchf is not None else None,
            times, index, vectors, index_rate, version, protect,
        )
        chunk = trim_render_context(
            raw, content_frames=round(end * target_sr / 16000) - round(start * target_sr / 16000),
            padding_frames=pipeline.t_pad_tgt,
            left_context_frames=left_context,
            right_context_frames=right_context,
        )
        if result is None:
            result = chunk
        elif left_context:
            overlap = 2 * left_context
            plan = plan_splice(result[-overlap:], chunk[:overlap], target_sr)
            result = splice_contexts(result, chunk, overlap, plan)
        else:
            result = np.concatenate((result, chunk))
        start = end
    expected = round(len(audio) * target_sr / 16000)
    if len(result) != expected or not np.isfinite(result).all():
        raise RuntimeError(f"RVC output does not preserve the source timeline: {len(result)} != {expected}")
    return result
