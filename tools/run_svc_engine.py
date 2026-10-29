"""Standalone AI-RVC launcher for locally installed zero-shot singing engines.

Example:
python tools/run_svc_engine.py --engine seed_vc --engine-repo /opt/seed-vc \\
    --engine-python /opt/seed-vc/.venv/bin/python --source vocals.wav \\
    --reference singer.wav --output converted.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer.svc_engine_bridge import SVCJob, convert_vocals


def main() -> None:
    p = argparse.ArgumentParser(description="Local Seed-VC / YingMusic-SVC singing conversion")
    p.add_argument("--engine", required=True, choices=["seed_vc", "yingmusic_svc"])
    p.add_argument("--engine-repo", type=Path, help="Defaults to model manager setting")
    p.add_argument("--engine-python", type=Path, help="Defaults to shared/engine Python in registry")
    p.add_argument("--source", required=True, type=Path, help="Clean lead stem, not full accompaniment")
    p.add_argument("--reference", required=True, type=Path, help="Consented target singer audio")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--pitch-shift", type=int, default=0)
    p.add_argument("--cuda-index", type=int, default=0)
    p.add_argument("--fp16", choices=["true", "false"], default="true")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    # Allow zero-configuration calls after configuring the WebUI model manager.
    requires_registry = (args.engine_repo is None or args.engine_python is None
                         or (args.engine in ("yingmusic_svc", "soulx_singer")
                             and (args.checkpoint is None or args.config is None)))
    if requires_registry:
        from tools.svc_model_registry import get_engine_job_options
        defaults = get_engine_job_options(args.engine)
    else:
        defaults = {}
    job = SVCJob(
        engine=args.engine,
        source_vocals=args.source,
        target_reference=args.reference,
        output_wav=args.output,
        engine_repository=args.engine_repo or defaults.get("engine_repository"),
        engine_python=args.engine_python or defaults.get("engine_python"),
        checkpoint=args.checkpoint or defaults.get("checkpoint"),
        model_config=args.config or defaults.get("model_config"),
        diffusion_steps=args.steps,
        pitch_shift=args.pitch_shift,
        cuda_index=args.cuda_index,
        fp16=args.fp16 == "true",
        timeout_seconds=args.timeout,
    )
    output = convert_vocals(job)
    print(f"Float32 WAV: {output}")


if __name__ == "__main__":
    main()
