"""RVC realtime microphone conversion with bounded audio queues.

PortAudio callback never performs Torch/FAISS/ML inference. A worker preloads
the bundled RVC streaming model and processes audio context asynchronously.
This is a *buffered* realtime path (not a sub-50ms claim). The selected stream
must support simultaneous capture/playback at the chosen common sample rate.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class RealtimeSettings:
    model_path: str
    index_path: str = ""
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 48000
    block_ms: int = 250
    context_ms: int = 1200
    crossfade_ms: int = 40
    pitch_shift: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.0
    device: str = "cuda"
    input_gain: float = 1.0
    output_gain: float = 0.8


def validate_settings(settings: RealtimeSettings) -> RealtimeSettings:
    if not Path(settings.model_path).is_file() or Path(settings.model_path).suffix.lower() != ".pth":
        raise ValueError("请选择已经存在的 RVC .pth 模型")
    if settings.index_rate and not Path(settings.index_path).is_file():
        raise ValueError("索引比例不为 0 时必须提供有效 .index 文件")
    if settings.sample_rate not in (32000, 40000, 44100, 48000):
        raise ValueError("实时音频设备采样率只能是 32k/40k/44.1k/48k")
    if settings.block_ms not in (200, 250, 300, 400):
        raise ValueError("音频块应使用已验证的 200/250/300/400 ms 设置")
    if not 500 <= settings.context_ms <= 3000:
        raise ValueError("上下文长度必须在 500–3000 ms 内")
    if not 10 <= settings.crossfade_ms <= 80:
        raise ValueError("交叉淡入淡出应在 10–80 ms 内")
    if settings.f0_method not in ("rmvpe", "fcpe", "pm"):
        raise ValueError("实时 RVC 支持 RMVPE、FCPE 或 PM")
    if not -12 <= settings.pitch_shift <= 12:
        raise ValueError("音高必须位于 -12 至 +12 半音")
    if not 0 <= settings.index_rate <= 1 or not 0 <= settings.input_gain <= 3 or not 0 <= settings.output_gain <= 3:
        raise ValueError("索引与增益参数越界")
    for index in (settings.input_device, settings.output_device):
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("无效的音频设备索引")
    return settings


def audio_devices() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    import sounddevice as sd
    sources, sinks = [], []
    for idx, device in enumerate(sd.query_devices()):
        hostapi = sd.query_hostapis(device["hostapi"])["name"]
        label = f'{idx}: {device["name"]} ({hostapi})'
        if device["max_input_channels"]:
            sources.append((label, idx))
        if device["max_output_channels"]:
            sinks.append((label, idx))
    return sources, sinks


class RollingAudio:
    """Exact-length input history; output takes one aligned block per callback."""
    def __init__(self, sample_rate: int, block_ms: int, context_ms: int,
                 crossfade_ms: int):
        if sample_rate % 100:
            raise ValueError("实时音频采样率必须可精确划分为 10 ms 帧")
        self.rate = sample_rate
        self.hop = sample_rate // 100
        self.block_frames = (sample_rate * block_ms // 1000 // self.hop) * self.hop
        self.context_frames = (sample_rate * context_ms // 1000 // self.hop) * self.hop
        self.fade_frames = (sample_rate * crossfade_ms // 1000 // self.hop) * self.hop
        self.search_frames = self.hop
        self.total_frames = self.context_frames + self.block_frames + self.fade_frames + self.search_frames
        self.input16 = np.zeros(self.total_frames * 16000 // sample_rate, dtype=np.float32)
        self.previous_tail = np.zeros(self.fade_frames, dtype=np.float32)
        self.first = True

    def ingest(self, chunk: np.ndarray) -> np.ndarray:
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if len(mono) != self.block_frames:
            raise ValueError(f"声卡音频帧数变化: {len(mono)} != {self.block_frames}")
        down = resample_poly(mono, 16000, self.rate).astype(np.float32)
        count = self.block_frames * 16000 // self.rate
        if len(down) != count:
            raise ValueError("输入音频 16 kHz 重采样长度异常")
        self.input16[:-count] = self.input16[count:]
        self.input16[-count:] = down
        return self.input16

    def align(self, generated: np.ndarray, model_rate: int) -> np.ndarray:
        audio = np.asarray(generated, dtype=np.float32).ravel()
        if model_rate != self.rate:
            audio = resample_poly(audio, self.rate, model_rate).astype(np.float32)
        minimum = self.block_frames + self.fade_frames + self.search_frames
        if len(audio) < minimum:
            raise RuntimeError(f"RVC 输出长度不足，至少需要 {minimum} 帧，实际 {len(audio)}")
        audio = audio[-minimum:]
        max_shift = self.search_frames
        shift = 0
        if not self.first:
            # SOLA: choose best small positive offset by normalized correlation.
            reference = self.previous_tail
            best = -float("inf")
            for n in range(0, max_shift + 1, max(1, self.hop // 4)):
                candidate = audio[n:n + self.fade_frames]
                denom = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
                score = float(np.dot(candidate, reference) / denom) if denom > 1e-8 else 0.0
                if score > best:
                    best, shift = score, n
        output = audio[shift:shift + self.block_frames].copy()
        if not self.first and self.fade_frames:
            fade = np.linspace(0, 1, self.fade_frames, dtype=np.float32)
            output[:self.fade_frames] = self.previous_tail * (1 - fade) + output[:self.fade_frames] * fade
        # Only the future tail is consumed at the next block boundary.
        self.previous_tail = audio[shift + self.block_frames:shift + self.block_frames + self.fade_frames].copy()
        self.first = False
        return output


class RealtimeRVC:
    """Threaded engine, with nonblocking PortAudio callbacks and bounded queues."""
    def __init__(self, settings: RealtimeSettings,
                 notify: Callable[[str], None] | None = None):
        self.settings = validate_settings(settings)
        self.notify = notify or (lambda message: None)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._stream = None
        self._incoming: queue.Queue = queue.Queue(maxsize=3)
        self._outgoing: queue.Queue = queue.Queue(maxsize=3)
        self._lock = threading.Lock()
        self.last_error = ""
        self.last_runtime_ms = 0.0
        self.underruns = 0
        self.overruns = 0
        self.state = "stopped"

    @property
    def running(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def start(self) -> None:
        if self.running:
            raise RuntimeError("实时变声已运行")
        self._stop.clear()
        self.last_error = ""
        self.underruns = self.overruns = 0
        self._incoming = queue.Queue(maxsize=3)
        self._outgoing = queue.Queue(maxsize=3)
        self.state = "loading"
        self._worker = threading.Thread(target=self._run, name="rvc-realtime", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> bool:
        if self._worker:
            self._worker.join(timeout=timeout)
        return not self.running

    def _callback(self, indata, outdata, frames, timing, status) -> None:
        # Only tiny copies and queue ops; never block PortAudio on Torch/GUI I/O.
        outdata.fill(0)
        if self._stop.is_set():
            return
        if status:
            self.overruns += 1
        try:
            self._incoming.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            self.overruns += 1  # Drop stale input rather than accumulating latency.
        try:
            block = self._outgoing.get_nowait()
            if block.shape[0] == frames:
                outdata[:, :2] = block[:, None]
            else:
                self.underruns += 1
        except queue.Empty:
            self.underruns += 1

    def _run(self) -> None:
        import sounddevice as sd
        import torch
        model = None
        stream = None
        try:
            from types import SimpleNamespace
            from infer.lib.rtrvc import RVC
            cfg = SimpleNamespace(device=torch.device(self.settings.device),
                                  is_half=False, use_jit=False, dml=False)
            model = RVC(
                key=self.settings.pitch_shift, formant=0.0,
                pth_path=self.settings.model_path,
                index_path=self.settings.index_path,
                index_rate=self.settings.index_rate,
                n_cpu=1, inp_q=None, opt_q=None, config=cfg,
            )
            if not getattr(model, "net_g", None) or not getattr(model, "model", None):
                raise RuntimeError("RVC 模型加载失败；请检查 HuBERT/RMVPE 资源和模型兼容性")
            audio = RollingAudio(
                self.settings.sample_rate, self.settings.block_ms,
                self.settings.context_ms, self.settings.crossfade_ms,
            )
            block = audio.block_frames
            sd.check_input_settings(device=self.settings.input_device, channels=1,
                                    samplerate=self.settings.sample_rate, dtype="float32")
            sd.check_output_settings(device=self.settings.output_device, channels=2,
                                     samplerate=self.settings.sample_rate, dtype="float32")
            stream = sd.Stream(
                device=(self.settings.input_device, self.settings.output_device),
                samplerate=self.settings.sample_rate,
                blocksize=block, channels=(1, 2), dtype="float32",
                callback=self._callback,
            )
            stream.start()
            self.state = "active"
            self.notify("麦克风实时变声已启动")
            while not self._stop.is_set():
                try:
                    chunk = self._incoming.get(timeout=0.1)
                except queue.Empty:
                    continue
                start = time.perf_counter()
                data16 = audio.ingest(chunk * self.settings.input_gain)
                tensor = torch.from_numpy(data16.copy()).to(cfg.device)
                count16 = block * 16000 // self.settings.sample_rate
                result = model.infer(
                    tensor, count16, audio.context_frames // audio.hop,
                    (block + audio.fade_frames + audio.search_frames) // audio.hop,
                    self.settings.f0_method,
                )
                if hasattr(result, "detach"):
                    result = result.detach().float().cpu().numpy()
                rendered = audio.align(result, int(model.tgt_sr))
                rendered = np.clip(rendered * self.settings.output_gain, -0.99, 0.99)
                self.last_runtime_ms = (time.perf_counter() - start) * 1000
                if self.last_runtime_ms > self.settings.block_ms:
                    self.overruns += 1
                try:
                    self._outgoing.put_nowait(rendered)
                except queue.Full:
                    # Drop stale output rather than pushing latency ever higher.
                    self.overruns += 1
                    try:
                        self._outgoing.get_nowait()
                    except queue.Empty:
                        pass
                    self._outgoing.put_nowait(rendered)
        except Exception as exc:
            self.last_error = str(exc)
            self.state = "error"
            self.notify(f"实时变声错误：{exc}")
        finally:
            if stream:
                try:
                    stream.stop()
                finally:
                    stream.close()
            self._stream = None
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.state != "error":
                self.state = "stopped"
            self._stop.set()
