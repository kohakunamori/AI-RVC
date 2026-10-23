# -*- coding: utf-8 -*-
"""
混音模块 - 人声与伴奏混合
"""
import numpy as np
import librosa
import soundfile as sf
from numba import njit
from pathlib import Path
from typing import Optional

from lib.audio import soft_clip_array

try:
    from lib.logger import log
except ImportError:
    log = None

try:
    from pedalboard import Pedalboard, Reverb, Compressor, Gain
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


_JUCE_COMB_TUNINGS_44100 = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_JUCE_ALLPASS_TUNINGS_44100 = (556, 441, 341, 225)
_JUCE_STEREO_SPREAD_44100 = 23


def _delay_samples(tuning: int, sr: int, channel: int) -> int:
    """Return the exact integer delay used by JUCE's FreeVerb implementation."""
    spread = _JUCE_STEREO_SPREAD_44100 if channel == 1 else 0
    return (int(sr) * (int(tuning) + spread)) // 44100


@njit(cache=True, nogil=True)
def _juce_reverb_kernel_float64(
    source: np.ndarray,
    comb_delays: np.ndarray,
    allpass_delays: np.ndarray,
    damping: float,
    feedback: float,
    wet_gain: float,
    dry_gain: float,
) -> np.ndarray:
    """Compiled sample loop matching JUCE Reverb's comb/all-pass topology."""
    channel_count, sample_count = source.shape
    processed = np.empty_like(source)
    max_comb_delay = int(np.max(comb_delays))
    max_allpass_delay = int(np.max(allpass_delays))

    for channel in range(channel_count):
        comb_buffers = np.zeros((8, max_comb_delay), dtype=np.float64)
        comb_indices = np.zeros(8, dtype=np.int64)
        comb_last = np.zeros(8, dtype=np.float64)
        allpass_buffers = np.zeros((4, max_allpass_delay), dtype=np.float64)
        allpass_indices = np.zeros(4, dtype=np.int64)

        for sample_index in range(sample_count):
            if channel_count == 1:
                filter_input = source[0, sample_index] * 0.015
            else:
                filter_input = (
                    source[0, sample_index] + source[1, sample_index]
                ) * 0.015

            wet_sample = 0.0
            for filter_index in range(8):
                buffer_index = comb_indices[filter_index]
                delayed = comb_buffers[filter_index, buffer_index]
                state = (
                    delayed * (1.0 - damping)
                    + comb_last[filter_index] * damping
                )
                comb_last[filter_index] = state
                comb_buffers[filter_index, buffer_index] = (
                    filter_input + state * feedback
                )
                buffer_index += 1
                if buffer_index >= comb_delays[channel, filter_index]:
                    buffer_index = 0
                comb_indices[filter_index] = buffer_index
                wet_sample += delayed

            for filter_index in range(4):
                buffer_index = allpass_indices[filter_index]
                delayed = allpass_buffers[filter_index, buffer_index]
                state = wet_sample + delayed * 0.5
                allpass_buffers[filter_index, buffer_index] = state
                buffer_index += 1
                if buffer_index >= allpass_delays[channel, filter_index]:
                    buffer_index = 0
                allpass_indices[filter_index] = buffer_index
                wet_sample = delayed - wet_sample

            processed[channel, sample_index] = (
                wet_sample * wet_gain + source[channel, sample_index] * dry_gain
            )

    return processed


def _apply_juce_reverb_float64(
    audio: np.ndarray,
    sr: int,
    room_size: float,
    wet_level: float,
) -> np.ndarray:
    """Run Pedalboard's JUCE/FreeVerb algorithm natively in FLOAT64.

    Pedalboard's built-in effects use JUCE float buffers and therefore always
    return FLOAT32.  This implementation keeps the same delay tunings,
    feedback/damping law, gain staging, stereo spread, width, and output length
    while evaluating the same sample loop in a compiled FLOAT64 Numba kernel.
    """
    source = np.asarray(audio)
    if source.dtype != np.dtype("float64"):
        raise TypeError("FLOAT64 reverb requires a FLOAT64 source array")
    if int(sr) <= 0:
        raise ValueError("reverb sample rate must be positive")
    if not 0.0 <= float(room_size) <= 1.0:
        raise ValueError("room_size must be between 0 and 1")
    if not 0.0 <= float(wet_level) <= 1.0:
        raise ValueError("wet_level must be between 0 and 1")

    if source.ndim == 1:
        source = source.reshape(1, -1)
    if source.ndim != 2 or source.shape[0] not in (1, 2):
        raise ValueError("reverb audio must contain one or two channels")

    damping = 0.5 * 0.4
    feedback = float(room_size) * 0.28 + 0.7
    wet_gain = float(wet_level) * 3.0
    dry_gain = (1.0 - float(wet_level)) * 2.0
    comb_delays = np.asarray(
        [
            [_delay_samples(tuning, sr, channel) for tuning in _JUCE_COMB_TUNINGS_44100]
            for channel in range(source.shape[0])
        ],
        dtype=np.int64,
    )
    allpass_delays = np.asarray(
        [
            [_delay_samples(tuning, sr, channel) for tuning in _JUCE_ALLPASS_TUNINGS_44100]
            for channel in range(source.shape[0])
        ],
        dtype=np.int64,
    )
    return _juce_reverb_kernel_float64(
        np.ascontiguousarray(source),
        comb_delays,
        allpass_delays,
        damping,
        feedback,
        wet_gain,
        dry_gain,
    )


def _probe_sample_rate(path: str) -> int:
    """Probe sample rate from file metadata and expose failures explicitly."""
    try:
        sample_rate = int(sf.info(path).samplerate)
    except Exception as exc:
        raise RuntimeError(f"无法读取混音输入的采样率: {path}") from exc
    if sample_rate <= 0:
        raise RuntimeError(f"混音输入的采样率无效: {path}, sample_rate={sample_rate}")
    return sample_rate


def load_audio_for_mix(path: str, target_sr: Optional[int] = None) -> tuple:
    """
    加载音频用于混音。

    Args:
        path: 音频路径
        target_sr: 目标采样率；为 None 时保持原始采样率

    Returns:
        tuple: (audio_data, sample_rate)
    """
    if log:
        log.detail(f"加载音频: {Path(path).name}")

    try:
        info = sf.info(path)
    except Exception:
        # Formats outside libsndfile (for example some browser MP3 inputs) are
        # decoded through librosa.  Once libsndfile has positively identified a
        # container, failures must propagate instead of silently narrowing a
        # damaged DOUBLE/PCM32 file to librosa's float32 representation.
        audio, sr = librosa.load(path, sr=target_sr, mono=False)
        audio = np.asarray(audio, dtype=np.float32)
    else:
        dtype = "float64" if str(info.subtype).upper() in {"DOUBLE", "PCM_32"} else "float32"
        decoded, sr = sf.read(path, dtype=dtype, always_2d=True)
        audio = decoded.T
        if target_sr is not None and int(target_sr) > 0 and sr != int(target_sr):
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=int(target_sr),
                axis=-1,
            ).astype(dtype, copy=False)
            sr = int(target_sr)

    if audio.ndim == 1 or audio.shape[0] == 1:
        mono = audio.reshape(-1)
        audio = np.stack([mono, mono])
        if log:
            log.detail("单声道已扩展为双声道")

    if log:
        log.detail(f"音频形状: {audio.shape}, 采样率: {sr}Hz")

    return audio, sr


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.3,
    wet_level: float = 0.2,
) -> np.ndarray:
    """对人声应用混响效果。"""
    if log:
        log.detail(f"应用混响: room_size={room_size}, wet_level={wet_level}")

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    if audio.dtype == np.dtype("float64"):
        processed = _apply_juce_reverb_float64(audio, sr, room_size, wet_level)
    else:
        if not PEDALBOARD_AVAILABLE:
            raise RuntimeError("Pedalboard 不可用，无法执行 FLOAT32 混响处理")

        board = Pedalboard([
            Reverb(room_size=room_size, wet_level=wet_level, dry_level=1.0 - wet_level)
        ])
        processed = board(audio, sr)

    if log:
        log.detail("混响处理完成")

    return processed


def adjust_audio_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """将音频裁切/补零到目标长度。"""
    current_length = audio.shape[-1]

    if current_length == target_length:
        return audio
    if current_length > target_length:
        return audio[..., :target_length]

    pad_amount = target_length - current_length
    if audio.ndim == 1:
        return np.pad(audio, (0, pad_amount))
    return np.pad(audio, ((0, 0), (0, pad_amount)))


def mix_vocals_and_accompaniment(
    vocals_path: str,
    accompaniment_path: str,
    output_path: str,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
    reverb_amount: float = 0.0,
    target_sr: Optional[int] = None,
) -> str:
    """
    混合人声和伴奏。

    Args:
        vocals_path: 人声音频路径
        accompaniment_path: 伴奏音频路径
        output_path: 输出路径
        vocals_volume: 人声音量 (0-2)
        accompaniment_volume: 伴奏音量 (0-2)
        reverb_amount: 人声混响量 (0-1)
        target_sr: 目标采样率；None 时自动采用两轨中更高采样率

    Returns:
        str: 输出文件路径
    """
    from infer.contracts import number, integer
    vocals_volume = number(vocals_volume, "vocals_volume", 0, 2)
    accompaniment_volume = number(accompaniment_volume, "accompaniment_volume", 0, 2)
    reverb_amount = number(reverb_amount, "reverb_amount", 0, 1)
    if target_sr is None:
        vocals_sr = _probe_sample_rate(vocals_path)
        accompaniment_sr = _probe_sample_rate(accompaniment_path)
        target_sr = max(vocals_sr, accompaniment_sr)
    else:
        target_sr = integer(target_sr, "target_sr", 8000, 192000)

    if log:
        log.progress("开始混音处理...")
        log.audio(f"人声文件: {Path(vocals_path).name}")
        log.audio(f"伴奏文件: {Path(accompaniment_path).name}")
        log.config(f"人声音量: {vocals_volume}, 伴奏音量: {accompaniment_volume}")
        log.config(f"混响量: {reverb_amount}, 目标采样率: {target_sr}Hz")

    if log:
        log.detail("加载人声音频...")
    vocals, sr = load_audio_for_mix(vocals_path, target_sr)

    if log:
        log.detail("加载伴奏音频...")
    accompaniment, _ = load_audio_for_mix(accompaniment_path, target_sr)

    if reverb_amount > 0:
        if log:
            log.progress("应用人声混响...")
        vocals = apply_reverb(vocals, sr, room_size=0.4, wet_level=reverb_amount)

    vocals = soft_clip_array(vocals * vocals_volume, threshold=0.85, ceiling=0.95)
    accompaniment = soft_clip_array(
        accompaniment * accompaniment_volume,
        threshold=0.85,
        ceiling=0.95,
    )

    vocals_len = vocals.shape[-1]
    accompaniment_len = accompaniment.shape[-1]
    target_len = max(vocals_len, accompaniment_len)

    if target_len <= 0:
        raise ValueError("混音失败：音频长度为 0")

    if log:
        log.detail(f"人声长度: {vocals_len}, 伴奏长度: {accompaniment_len}")
        if vocals_len != accompaniment_len:
            log.detail(f"长度不一致，已补齐到最长长度: {target_len}")

    vocals = adjust_audio_length(vocals, target_len)
    accompaniment = adjust_audio_length(accompaniment, target_len)

    if log:
        log.progress("混合音轨...")
    mixed = vocals + accompaniment

    max_val = float(np.max(np.abs(mixed)))
    if log:
        log.detail(f"混合后峰值: {max_val:.4f}")

    mixed = soft_clip_array(mixed, threshold=0.90, ceiling=0.98)
    if log:
        final_peak = float(np.max(np.abs(mixed)))
        log.detail(f"软削波后峰值: {final_peak:.4f}")

    if mixed.ndim == 2:
        mixed = mixed.T

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log.progress(f"保存混音文件: {output_path}")

    output_dtype = np.float64 if mixed.dtype == np.dtype("float64") else np.float32
    sf.write(
        output_path,
        mixed.astype(output_dtype, copy=False),
        sr,
        subtype="DOUBLE" if output_dtype == np.float64 else "FLOAT",
    )

    output_size = Path(output_path).stat().st_size
    duration = target_len / sr

    if log:
        log.success("混音完成")
        log.audio(f"输出时长: {duration:.2f}秒")
        log.audio(f"输出大小: {output_size / 1024 / 1024:.2f} MB")

    return output_path


def check_pedalboard_available() -> bool:
    """检查 pedalboard 是否可用。"""
    return PEDALBOARD_AVAILABLE
