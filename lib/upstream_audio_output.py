"""Checked floating-output and full-vector retrieval adapters for pinned RVC.

The output adapter keeps network, normalization, and feature dimensions unchanged.
Its encoding changes use unit gain instead of 32768 and float32 instead of int16.
When retrieval is enabled, use the shared exact FP32-vector search, including
zero-distance handling and explicit failures instead of swallowed index errors.
No source checkout or model weights are modified. A changed API fails closed.
The separate F0 adapter restores native unvoiced zeros so protection is effective.
"""
from __future__ import annotations

import inspect
import textwrap
from types import MethodType
from types import SimpleNamespace

from infer.rmvpe_boundaries import correct_rmvpe_transitions
from lib.rvc_chunking import render_aligned_segments


def preserve_unvoiced_f0(pipeline, *, verify_boundaries=False) -> None:
    """Keep extractor voicing decisions, as in the standard RVC protection path."""
    original = type(pipeline).get_f0
    source = textwrap.dedent(inspect.getsource(original))
    interpolation = '''    try:
        uv = f0 == 0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    except Exception:
        traceback.print_exc()'''
    if source.count(interpolation) != 1:
        raise RuntimeError("官方 RVC F0 接口发生变化；已停止转换")
    validation = '''    if not np.isfinite(f0).all() or np.any(f0 < 0):
        raise ValueError("F0 输出必须是有限的非负频率")'''
    source = source.replace(interpolation, validation, 1)
    shift = "    f0 *= pow(2, f0_up_key / 12)"
    if source.count(shift) != 1:
        raise RuntimeError("官方 RVC 音高移调接口发生变化；已停止转换")
    if verify_boundaries:
        source = source.replace(shift, '    if f0_method == "rmvpe":\n        f0 = _correct_rmvpe(x, f0)[0]\n' + shift, 1)
    namespace = dict(original.__globals__)
    namespace['_correct_rmvpe'] = correct_rmvpe_transitions
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    pipeline.get_f0 = MethodType(namespace[original.__name__], pipeline)


def _align_vc_context(source, namespace):
    start = source.index('    for t in opt_ts:')
    end = source.index('    audio_opt = np.concatenate(audio_opt)', start)
    end += len('    audio_opt = np.concatenate(audio_opt)')
    source = source[:start] + '''    audio_opt = _render_aligned_segments(
        self, model, net_g, sid, audio, audio_pad, pitch, pitchf, times,
        index, index_vectors, index_rate, version, protect, tgt_sr, opt_ts,
    )''' + source[end:]
    namespace['_render_aligned_segments'] = render_aligned_segments
    source = source.replace('pitchf = pitchf.astype(np.float32)',
                            'pitchf = pitchf.astype(np.float64 if getattr(self, "rvc_dtype", None) == torch.float64 else np.float32)')
    source = source.replace('torch.tensor(pitchf, device=self.device).unsqueeze(0).float()',
                            'torch.tensor(pitchf, device=self.device).unsqueeze(0)')
    return source


def preserve_float_vc_output(pipeline, *, index_loader=None, index_retriever=None, align_context=False) -> None:
    original = type(pipeline).pipeline
    source = textwrap.dedent(inspect.getsource(original))
    changes = {
        "max_int16 = 32768": "max_int16 = 1.0",
        "audio_opt = (audio_opt * max_int16).astype(np.int16)":
            "audio_opt = (audio_opt * max_int16).astype(np.float64 if audio_opt.dtype == np.float64 else np.float32)",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError("官方 RVC 音频编码接口发生变化，无法保证浮点输出；已停止转换")
        source = source.replace(before, after, 1)
    namespace = dict(original.__globals__)
    if align_context:
        source = _align_vc_context(source, namespace)
    if index_loader is not None:
        before = '''    if (
        file_index != ""
        and os.path.exists(file_index)
        and index_rate != 0
    ):
        try:
            index = faiss.read_index(file_index)
            index_vectors = index.reconstruct_n(0, index.ntotal)
        except:
            traceback.print_exc()
            index = index_vectors = None'''
        if source.count(before) != 1 or index_retriever is None:
            raise RuntimeError("官方 RVC 索引接口发生变化，无法保证完整精度检索；已停止转换")
        source = source.replace(before, '    if index_rate != 0:\n        index, index_vectors = _rvc_load_index(file_index)', 1)
        namespace['_rvc_load_index'] = index_loader
    if index_loader is None and not align_context:
        exec(compile(source, inspect.getfile(original), "exec"), namespace)
        pipeline.pipeline = MethodType(namespace[original.__name__], pipeline)
        return
    vc_original = type(pipeline).vc
    vc_source = textwrap.dedent(inspect.getsource(vc_original))
    vc_namespace = dict(vc_original.__globals__)
    if index_loader is not None:
        before_retrieval = '''        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)'''
        if vc_source.count(before_retrieval) != 1:
            raise RuntimeError("官方 RVC 特征检索接口发生变化；已停止转换")
        vc_source = vc_source.replace(before_retrieval, '        npy = _rvc_retrieve(index, index_vectors, npy, 8)', 1)
        vc_namespace['_rvc_retrieve'] = index_retriever
    synthesis = '    with torch.no_grad():\n        hasp = pitch is not None and pitchf is not None'
    if vc_source.count(synthesis) != 1:
        raise RuntimeError("官方 RVC 合成入口发生变化；已停止转换")
    vc_source = vc_source.replace(synthesis, '''    feats = feats.to(dtype=next(net_g.parameters()).dtype)
    if pitchf is not None:
        pitchf = pitchf.to(dtype=torch.float64 if feats.dtype == torch.float64 else torch.float32)
''' + synthesis, 1)
    before_output = 'synthesized[0, 0].data.cpu().float().numpy()'
    if vc_source.count(before_output) != 1:
        raise RuntimeError("官方 RVC 合成输出接口发生变化；已停止转换")
    vc_source = vc_source.replace(before_output, 'synthesized[0, 0].detach().cpu().numpy()', 1)
    exec(compile(vc_source, inspect.getfile(vc_original), "exec"), vc_namespace)
    pipeline.vc = MethodType(vc_namespace[vc_original.__name__], pipeline)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    pipeline.pipeline = MethodType(namespace[original.__name__], pipeline)


def initialize_uvr_spectrum(spec_utils_module) -> None:
    """Initialize cropped-out bins, matching the upstream UVR buffer fix."""
    original = spec_utils_module.cmb_spectrogram_to_wave
    source = textwrap.dedent(inspect.getsource(original))
    before = "spec_s = np.ndarray("
    if source.count(before) != 1:
        raise RuntimeError("官方 UVR5 频谱接口发生变化；已停止分离")
    source = source.replace(before, "spec_s = np.zeros(", 1)
    namespace = dict(original.__globals__)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    spec_utils_module.cmb_spectrogram_to_wave = namespace[original.__name__]


def preserve_float_uvr_output(vr_module) -> None:
    """Initialize legacy spectrum buffers and replace checked integer encoders."""
    import numpy as np

    initialize_uvr_spectrum(vr_module.spec_utils)
    writer = vr_module.sf.write

    def write_float_wav(path, data, sample_rate, **kwargs):
        if str(path).lower().endswith(".wav"):
            kwargs["subtype"] = "FLOAT"
        return writer(path, data, sample_rate, **kwargs)

    for class_name in ("AudioPre", "AudioPreDeEcho"):
        model_class = getattr(vr_module, class_name)
        original = model_class._path_audio_
        source = textwrap.dedent(inspect.getsource(original))
        for stem in ("wav_instrument", "wav_vocals"):
            before = f'(np.array({stem}) * 32768).astype("int16")'
            if source.count(before) != 2:
                raise RuntimeError("官方 UVR5 音频编码接口发生变化，无法保证浮点输出；已停止分离")
            source = source.replace(before, f"np.asarray({stem}, dtype=np.float32)")
        namespace = dict(original.__globals__)
        namespace["sf"] = SimpleNamespace(write=write_float_wav)
        exec(compile(source, inspect.getfile(original), "exec"), namespace)
        model_class._path_audio_ = namespace[original.__name__]
