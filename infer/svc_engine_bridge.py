"""Optional isolated singing voice conversion backends.

Keep incompatible training/inference dependencies outside AI-RVC's environment.
This bridge consumes a prepared *lead vocal*, never the full song.
No direct internet/API calls or silent model fallback are performed here.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

EngineName = Literal["seed_vc", "yingmusic_svc"]


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
    if job.engine not in ("seed_vc", "yingmusic_svc"):
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
    script = _required_file(folder / ("inference.py" if job.engine == "seed_vc" else "my_inference.py"), "Backend entrypoint")
    output = Path(job.output_wav).expanduser().resolve()
    if output.suffix.lower() != ".wav":
        raise ValueError("Output path must be .wav")
    if output in (source, target):
        raise ValueError("Refusing to overwrite source or reference audio")
    if (job.checkpoint is None) != (job.model_config is None):
        raise ValueError("Checkpoint and model config must be supplied together")
    if job.engine == "yingmusic_svc" and job.checkpoint is None:
        raise ValueError("YingMusic-SVC requires a locally installed checkpoint and YAML config")
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
            worker = Path(__file__).resolve().parents[1] / "tools" / "yingmusic_svc_worker.py"
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
        try:
            result = subprocess.run(
                args, cwd=str(root), env=env, capture_output=True, text=True,
                check=False, timeout=job.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{job.engine} exceeded the {job.timeout_seconds}s inference limit") from exc
        if result.returncode != 0:
            output_log = (result.stderr + "\n" + result.stdout)[-4000:]
            raise RuntimeError(f"{job.engine} exited {result.returncode}:\n{output_log}")
        rendered = sorted(scratch.rglob("*.wav"))
        if len(rendered) != 1:
            raise RuntimeError(f"{job.engine} must create exactly one converted vocal WAV, found {len(rendered)}")
        converted, rate = sf.read(str(rendered[0]), dtype="float32", always_2d=True)
        if not rate or converted.size == 0 or not np.isfinite(converted).all():
            raise ValueError(f"{job.engine} produced empty or non-finite audio")
        # Engines synthesize mono vocals: preserve channels but keep float precision.
        sf.write(str(output), converted, rate, subtype="FLOAT")
    return output
