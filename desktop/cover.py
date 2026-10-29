"""Desktop offline cover orchestration, with no Gradio import.

Reuses the project's official/quality RVC pipeline and locally configured
Seed-VC, YingMusic and SoulX backends. Invoked from a GUI worker thread.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
ENGINE_IDS = ("rvc", "seed_vc", "yingmusic_svc", "soulx_singer")


@dataclass(frozen=True)
class CoverRequest:
    audio_file: str
    engine: str = "rvc"
    input_mode: str = "song"
    rvc_model: str = ""
    rvc_index: str = ""
    reference_audio: str = ""
    pitch_shift: int = 0
    karaoke: bool = True
    dereverb: bool = False
    diffusion_steps: int = 35
    output_dir: str = ""


def run_cover(request: CoverRequest, progress: Callable[[str], None] | None = None) -> dict:
    if request.engine not in ENGINE_IDS:
        raise ValueError("未知歌声转换引擎")
    if request.input_mode not in ("song", "vocals"):
        raise ValueError("请选择完整歌曲或干声输入")
    if not Path(request.audio_file).is_file():
        raise FileNotFoundError("输入音频不存在")
    if not -12 <= request.pitch_shift <= 12:
        raise ValueError("音高偏移超出允许范围")
    if not 1 <= request.diffusion_steps <= 150:
        raise ValueError("扩散步数超出允许范围")

    output = str(Path(request.output_dir).expanduser().resolve()
                 if request.output_dir else ROOT / "outputs")
    config_file = ROOT / "configs" / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    cover = config.get("cover", {})
    device = config.get("device", "cuda")
    say = progress or (lambda _: None)

    if request.engine != "rvc":
        if not request.reference_audio or not Path(request.reference_audio).is_file():
            raise FileNotFoundError("零样本 SVC 必须提供有效目标声线参考音频")
        from infer.svc_cover_workflow import run_svc_cover
        return run_svc_cover(
            audio_path=request.audio_file,
            reference_path=request.reference_audio,
            engine=request.engine, input_mode=request.input_mode,
            output_directory=output, device=device,
            karaoke=request.karaoke,
            separator_model=cover.get("roformer_model", "hybrid:leap_xe90_vocals+leap62_instrumental"),
            karaoke_model=cover.get("karaoke_model", "ensemble:mvsep_9205_avg"),
            dereverb=request.dereverb,
            pitch_shift=request.pitch_shift,
            diffusion_steps=request.diffusion_steps,
            progress=lambda message, stage, total: say(f"[{stage}/{total}] {message}"),
        )

    if not request.rvc_model or not Path(request.rvc_model).is_file():
        raise FileNotFoundError("RVC 模式需要本地 .pth 模型")
    use_index = bool(request.rvc_index and Path(request.rvc_index).is_file())
    from infer.cover_pipeline import get_cover_pipeline
    pipe = get_cover_pipeline(device)
    # Use the same official/quality routes and separator models as the existing
    # project, but without importing gradio or starting an HTTP service.
    result = pipe.process(
        input_audio=request.audio_file, input_mode=request.input_mode,
        model_path=request.rvc_model,
        index_path=request.rvc_index if use_index else None,
        pitch_shift=request.pitch_shift,
        index_ratio=float(cover.get("index_rate", 0.5)) if use_index else 0.0,
        rms_mix_rate=float(cover.get("rms_mix_rate", 0.0)),
        protect=float(cover.get("protect", 0.33)),
        speaker_id=int(cover.get("speaker_id", 0)),
        f0_method=cover.get("f0_method", "rmvpe"),
        demucs_model=cover.get("demucs_model", "htdemucs_ft"),
        demucs_shifts=int(cover.get("demucs_shifts", 10)),
        demucs_overlap=float(cover.get("demucs_overlap", 0.5)),
        demucs_split=bool(cover.get("demucs_split", True)),
        roformer_model=cover.get("roformer_model", "hybrid:leap_xe90_vocals+leap62_instrumental"),
        separator=cover.get("separator", "roformer"),
        uvr5_model=cover.get("uvr5_model", "HP2_all_vocals"),
        uvr5_agg=int(cover.get("uvr5_agg", 10)),
        uvr5_format=cover.get("uvr5_format", "wav"),
        use_official=bool(cover.get("use_official", True)),
        silence_gate=bool(cover.get("silence_gate", False)),
        silence_threshold_db=float(cover.get("silence_threshold_db", -50)),
        silence_smoothing_ms=float(cover.get("silence_smoothing_ms", 50)),
        silence_min_duration_ms=int(cover.get("silence_min_duration_ms", 200)),
        karaoke_separation=request.karaoke,
        karaoke_model=cover.get("karaoke_model", "ensemble:mvsep_9205_avg"),
        karaoke_merge_backing_into_accompaniment=bool(cover.get("karaoke_merge_backing_into_accompaniment", True)),
        source_constraint_mode=cover.get("source_constraint_mode", "auto"),
        vc_pipeline_mode=cover.get("vc_pipeline_mode", "current"),
        dereverb_mode="force" if request.dereverb else "off",
        output_dir=output,
        model_display_name=Path(request.rvc_model).stem,
        progress_callback=lambda message, step, total: say(f"[{step}/{total}] {message}"),
    )
    return result
