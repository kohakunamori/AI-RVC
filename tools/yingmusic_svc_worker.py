"""Invoke the public YingMusic-SVC Python API with *explicit* pitch preservation.

Run only through the external YingMusic Python environment and working directory.
The upstream CLI sets semi_tone_shift=None and automatically changes song key.
That is undesirable when remixing against the original instrumental track.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=35)
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--fp16", choices=["True", "False"], default="True")
    argv = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("YingMusic-SVC worker currently requires CUDA; CPU inference is unverified")
    if argv.cuda_index < 0 or argv.cuda_index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device unavailable")
    # This worker file lives in AI-RVC; upstream my_inference.py is in the\n    # separately installed engine repository (our subprocess working directory).\n    sys.path.insert(0, str(Path.cwd().resolve()))\n    from my_inference import load_models_api, run_inference

    device = torch.device(f"cuda:{argv.cuda_index}")
    settings = argparse.Namespace(
        source=str(Path(argv.source).resolve()),
        target=str(Path(argv.target).resolve()),
        output=str(Path(argv.output).resolve()),
        expname="converted",
        checkpoint=str(Path(argv.checkpoint).resolve()),
        config=str(Path(argv.config).resolve()),
        diffusion_steps=argv.diffusion_steps,
        # The upstream default (None) automatically shifts the singer's key.
        # Explicit zero keeps the original song in the instrumental's key.
        semi_tone_shift=argv.pitch_shift,
        cuda=device,
        fp16=argv.fp16 == "True",
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=True,
        accompany=None,
    )
    models = load_models_api(settings, device=device)
    rendered = run_inference(settings, models, device=device)
    if not Path(rendered).is_file():
        raise FileNotFoundError(f"YingMusic-SVC did not return a WAV: {rendered}")
    print(f"CONVERTED_VOCALS={rendered}")


if __name__ == "__main__":
    main()
