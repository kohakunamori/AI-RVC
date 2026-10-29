"""Headless SoulX-Singer-SVC worker for already separated lead vocals.

Uses the official PreprocessPipeline for F0 extraction without running another
vocal separator, then calls the public cli.inference_svc API. All packages
come from the external SoulX installation, not AI-RVC's Python environment.
"""
from __future__ import annotations
import argparse
import gc
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--fp16", choices=["True", "False"], default="True")
    args = parser.parse_args()
    if not 1 <= args.diffusion_steps <= 150:
        raise ValueError("Invalid diffusion step count")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SoulX worker currently requires CUDA")
    device = f"cuda:{args.cuda_index}"
    if args.cuda_index < 0 or args.cuda_index >= torch.cuda.device_count():
        raise ValueError("CUDA index is unavailable")
    # The worker is launched with cwd=the external SoulX repository.
    sys.path.insert(0, str(Path.cwd().resolve()))
    from preprocess.pipeline import PreprocessPipeline
    from soulxsinger.utils.file_utils import load_config
    from cli.inference_svc import build_model, process

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from svc_reference_cache import prepare_soulx_reference

    def preprocess(filepath: str, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        pipeline = PreprocessPipeline(
            device=device, language="Mandarin",
            save_dir=str(folder), vocal_sep=False,
            max_merge_duration=60000, midi_transcribe=False,
        )
        try:
            pipeline.run(
                audio_path=str(Path(filepath).resolve()), vocal_sep=False,
                max_merge_duration=60000, language="Mandarin",
            )
            vocal = folder / "vocal.wav"
            f0 = folder / "vocal_f0.npy"
            if not vocal.is_file() or not f0.is_file():
                raise FileNotFoundError(f"SoulX preprocessing failed: {filepath}")
            return vocal, f0
        finally:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prompt_wav, prompt_f0 = prepare_soulx_reference(
        reference_file=Path(args.target).resolve(),
        model_config=Path(args.config).resolve(),
        implementation_file=Path.cwd().resolve() / "preprocess" / "pipeline.py",
        fallback_directory=out / "preprocess_prompt",
        run_preprocess=lambda folder: preprocess(args.target, folder),
    )
    target_wav, target_f0 = preprocess(args.source, out / "preprocess_target")
    prepared = [(prompt_wav, prompt_f0), (target_wav, target_f0)]

    model_config = load_config(str(Path(args.config).resolve()))
    model = build_model(
        model_path=str(Path(args.checkpoint).resolve()),
        config=model_config, device=device, use_fp16=args.fp16 == "True",
    )
    # SoulX uses prompt for the target timbre, and target for the source melody.
    settings = argparse.Namespace(
        prompt_wav_path=str(prepared[0][0]),
        prompt_f0_path=str(prepared[0][1]),
        target_wav_path=str(prepared[1][0]),
        target_f0_path=str(prepared[1][1]),
        device=device,
        save_dir=str(out / "generated"),
        auto_shift=False,
        pitch_shift=args.pitch_shift,
        n_steps=args.diffusion_steps,
        cfg=2.0,
        use_fp16=args.fp16 == "True",
    )
    process(settings, model_config, model)
    generated = out / "generated" / "generated.wav"
    if not generated.is_file():
        raise FileNotFoundError("SoulX did not produce generated.wav")
    # The bridge expects one WAV; remove only generated staging WAVs.
    # Preprocess creates two other WAVs: placing final result in a separate
    # task-selected directory avoids output auto-discovery ambiguity.
    print(f"SOULX_CONVERTED={generated}")


if __name__ == "__main__":
    main()
