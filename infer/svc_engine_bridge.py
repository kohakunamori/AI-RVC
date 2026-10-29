"""Optional isolated singing voice conversion backends.

Keep incompatible training/inference dependencies outside AI-RVC's environment.
This bridge consumes a prepared *lead vocal*, never the full song.
No direct internet/API calls or silent model fallback are performed here.
"""
from __future__ import annotations

import os
import subprocess
import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

EngineName = Literal["seed_vc", "yingmusic_svc", "soulx_singer"]


@dataclass(frozen=True)
class SVCJob:
    engine: EngineName
    source_vocals: Path
    target_reference: Path
    output_wav: Path
    engine_repository: Path
    engine_python: Path
    checkpoint: Path | None = None
    model_config: Path | None = None
    diffusion_steps: int = 35
    pitch_shift: int = 0
    fp16: bool = True
    cuda_index: int = 0
    timeout_seconds: int = 3600


def _required_file(value: Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validate(job: SVCJob) -> tuple[Path, Path, Path, Path, Path, Path | None, Path | None]:
    if job.engine not in ("seed_vc", "yingmusic_svc", "soulx_singer"):
        raise ValueError(f"Unsupported singing backend: {job.engine!r}")
    if type(job.diffusion_steps) is not int or not 1 <= job.diffusion_steps <= 150:
        raise ValueError("diffusion_steps must be an integer in [1, 150]")
    if type(job.pitch_shift) is not int or not -24 <= job.pitch_shift <= 24:
        raise ValueError("pitch_shift must be an integer in [-24, 24]")
    if type(job.cuda_index) is not int or job.cuda_index < 0:
        raise ValueError("cuda_index must be a nonnegative integer")
    if type(job.timeout_seconds) is not int or job.timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if type(job.fp16) is not bool:
        raise ValueError("fp16 must be a boolean")

    source = _required_file(job.source_vocals, "Source vocal")
    target = _required_file(job.target_reference, "Target reference")
    python = _required_file(job.engine_python, "Backend Python interpreter")
    folder = Path(job.engine_repository).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Backend repository missing: {folder}")
    script = _required_file(folder / ({"seed_vc": "inference.py", "yingmusic_svc": "my_inference.py", "soulx_singer": "cli/inference_svc.py"}[job.engine]), "Backend entrypoint")
    output = Path(job.output_wav).expanduser().resolve()
    if output.suffix.lower() != ".wav":
        raise ValueError("Output path must be .wav")
    if output in (source, target):
        raise ValueError("Refusing to overwrite source or reference audio")
    if (job.checkpoint is None) != (job.model_config is None):
        raise ValueError("Checkpoint and model config must be supplied together")
    if job.engine in ("yingmusic_svc", "soulx_singer") and job.checkpoint is None:
        raise ValueError("This SVC engine requires a locally installed checkpoint and YAML config")
    checkpoint = _required_file(job.checkpoint, "Checkpoint") if job.checkpoint else None
    config = _required_file(job.model_config, "Model config") if job.model_config else None
    return source, target, output, folder, python, checkpoint, config


def convert_vocals(job: SVCJob) -> Path:
    """Run a local model in its own environment, emit verified float32 WAV.

    A stable RVC .pth model is NOT a valid zero-shot SVC reference; target_reference
    must be a real, authorized audio sample. Each invocation is isolated.
    """
    source, target, output, root, python, checkpoint, config = _validate(job)
    output.parent.mkdir(parents=True, exist_ok=True)
    from tools.svc_model_registry import shared_cache_dir
    cache = shared_cache_dir()
    # Explicit, common weight locations; don't create three independent
    # Hugging Face / Torch caches just because the engines use separate venvs.
    cache_hf = cache / "huggingface"
    cache_torch = cache / "torch"
    cache_hf.mkdir(parents=True, exist_ok=True)
    cache_torch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svc_job_", dir=output.parent) as tmp:
        scratch = Path(tmp)
        args = [str(python)]
        if job.engine == "seed_vc":
            args += [
                str(root / "inference.py"),
                "--source", str(source),
                "--target", str(target),
                "--output", str(scratch),
                "--f0-condition", "True",
                "--auto-f0-adjust", "False",
                "--length-adjust", "1.0",
                "--semi-tone-shift", str(job.pitch_shift),
                "--diffusion-steps", str(job.diffusion_steps),
                "--inference-cfg-rate", "0.7",
                "--fp16", str(job.fp16),
            ]
            if checkpoint:
                args += ["--checkpoint", str(checkpoint), "--config", str(config)]
        else:
            worker_name = "yingmusic_svc_worker.py" if job.engine == "yingmusic_svc" else "soulx_svc_worker.py"
            worker = Path(__file__).resolve().parents[1] / "tools" / worker_name
            worker = _required_file(worker, "AI-RVC YingMusic worker")
            args += [
                str(worker),
                "--source", str(source),
                "--target", str(target),
                "--output", str(scratch),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--diffusion-steps", str(job.diffusion_steps),
                "--pitch-shift", str(job.pitch_shift),
                "--cuda-index", str(job.cuda_index),
                "--fp16", str(job.fp16),
            ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = str(cache_hf)
        env["HF_HUB_CACHE"] = str(cache_hf / "hub")
        env["TORCH_HOME"] = str(cache_torch)
        env["AI_RVC_SVC_CACHE_DIR"] = str(cache)
        # Never capture huge CLI debug logs into memory for multi-minute songs.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
            try:
                result = subprocess.run(
                    args, cwd=str(root), env=env,
                    stdout=log_file, stderr=subprocess.STDOUT,
                    check=False, timeout=job.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{job.engine} exceeded the {job.timeout_seconds}s inference limit") from exc
            if result.returncode != 0:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - 5000))
                output_log = log_file.read()[-4000:]
                raise RuntimeError(f"{job.engine} exited {result.returncode}:\n{output_log}")
        # SoulX preprocessing also emits temporary vocal WAVs; the output
        # contract is only the generated stem, not those intermediates.
        rendered = ([scratch / "generated" / "generated.wav"]
                    if job.engine == "soulx_singer" else sorted(scratch.rglob("*.wav")))
        rendered = [p for p in rendered if p.is_file()]
        if len(rendered) != 1:
            raise RuntimeError(f"{job.engine} must create exactly one converted vocal WAV, found {len(rendered)}")
        # Stream even hours-long audio in bounded-size blocks; don't hold a
        # second full song in RAM merely to convert PCM16 -> Float32.
        temporary_output = scratch / "validated_float32.wav"
        with sf.SoundFile(str(rendered[0]), mode="r") as inp:
            if inp.frames <= 0 or inp.samplerate <= 0:
                raise ValueError(f"{job.engine} produced empty audio")
            with sf.SoundFile(
                str(temporary_output), mode="w", samplerate=inp.samplerate,
                channels=inp.channels, format="WAV", subtype="FLOAT",
            ) as dst:
                for block in inp.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    if not np.isfinite(block).all():
                        raise ValueError(f"{job.engine} produced non-finite audio")
                    dst.write(block)
        os.replace(temporary_output, output)
    return output
