"""One-click local singing cover using the same separators and mixer as AI-RVC.

This does not import or require an RVC character model. External zero-shot
models use the target reference recording provided by the user.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable

from tools.svc_model_registry import get_engine_job_options


def run_svc_cover(
    audio_path: str,
    reference_path: str,
    engine: str,
    input_mode: str,
    output_directory: str,
    *,
    device: str = "cuda",
    karaoke: bool = True,
    separator_model: str = "hybrid:leap_xe90_vocals+leap62_instrumental",
    karaoke_model: str = "ensemble:mvsep_9205_avg",
    dereverb: bool = False,
    pitch_shift: int = 0,
    diffusion_steps: int = 35,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
    reverb_amount: float = 0.0,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    from infer.svc_engine_bridge import SVCJob, convert_vocals

    if input_mode not in ("song", "vocals"):
        raise ValueError(f"Unsupported input mode: {input_mode}")
    if engine not in ("seed_vc", "yingmusic_svc"):
        raise ValueError(f"Unsupported external SVC engine: {engine}")
    source = Path(audio_path).resolve()
    reference = Path(reference_path).resolve()
    if not source.is_file() or not reference.is_file():
        raise FileNotFoundError("原曲和目标声线参考音频均必须存在")
    options = get_engine_job_options(engine)
    output_root = Path(output_directory).expanduser().resolve()
    # A dedicated per-job directory prevents collisions across concurrent UI requests.
    task_dir = output_root / f"svc_{engine}_{uuid.uuid4().hex[:12]}"
    task_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "cover": None, "converted_vocals": None,
        "vocals": str(source), "lead_vocals": None,
        "backing_vocals": None, "accompaniment": None,
        "accompaniment_without_harmony": None,
        "all_files_dir": str(task_dir),
    }

    def report(message: str, stage: int) -> None:
        if progress:
            progress(message, stage, 4)

    lead = str(source)
    accompaniment = None
    if input_mode == "song":
        if karaoke:
            # Karaoke's second stem contains backing harmonies + instrumental.
            from infer.separator import KaraokeSeparator
            report("正在提取主唱及带和声伴奏", 1)
            separator = KaraokeSeparator(model_filename=karaoke_model, device=device)
        else:
            from infer.separator import RoformerSeparator
            report("正在分离人声及伴奏", 1)
            separator = RoformerSeparator(model_filename=separator_model, device=device)
        try:
            lead, accompaniment = separator.separate(str(source), str(task_dir / "stems"))
        finally:
            if hasattr(separator, "unload_model"):
                separator.unload_model()
        result["vocals"] = lead
        result["lead_vocals"] = lead
        result["accompaniment"] = accompaniment
        if not karaoke:
            result["accompaniment_without_harmony"] = accompaniment
    else:
        report("使用已准备的干声", 1)

    if dereverb:
        from infer.separator import RoformerDereverbSeparator
        report("正在可选去混响", 2)
        processor = RoformerDereverbSeparator(device=device)
        try:
            lead = processor.separate_dry(lead, str(task_dir / "dereverb"))
        finally:
            processor.unload_model()

    report("正在执行本地声线转换", 3)
    converted = task_dir / "converted_vocals.wav"
    job = SVCJob(
        engine=engine, source_vocals=Path(lead),
        target_reference=reference, output_wav=converted,
        pitch_shift=pitch_shift, diffusion_steps=diffusion_steps,
        **options,
    )
    convert_vocals(job)
    result["converted_vocals"] = str(converted)
    if input_mode == "vocals":
        result["vocals"] = str(source)
    if accompaniment:
        report("正在与伴奏混音", 4)
        from lib.mixer import mix_vocals_and_accompaniment
        mixed = task_dir / "cover.wav"
        mix_vocals_and_accompaniment(
            vocals_path=str(converted), accompaniment_path=accompaniment,
            output_path=str(mixed), vocals_volume=vocals_volume,
            accompaniment_volume=accompaniment_volume, reverb_amount=reverb_amount,
        )
        result["cover"] = str(mixed)
    else:
        report("干声转换完成", 4)
    return result
