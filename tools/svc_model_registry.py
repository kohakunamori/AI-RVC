"""Local SVC model registry: editable paths, no cloud inference or hidden downloads.

User-provided install locations live in configs/svc_engines.json. The app
never executes an arbitrary configured command: only known engine workers.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "configs" / "svc_engines.json"
KNOWN_ENGINES = {
    "rvc": {
        "name": "RVC v1/v2", "homepage": "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
        "checkpoint_required": False, "supported": True,
    },
    "seed_vc": {
        "name": "Seed-VC Singing 44k",
        "homepage": "https://github.com/Plachtaa/seed-vc",
        "checkpoint_required": False, "supported": True,
    },
    "yingmusic_svc": {
        "name": "YingMusic-SVC Full",
        "homepage": "https://github.com/GiantAILab/YingMusic-SVC",
        "checkpoint_required": True, "supported": True,
    },
    "soulx_singer": {
        "name": "SoulX-Singer-SVC",
        "homepage": "https://github.com/Soul-AILab/SoulX-Singer",
        "checkpoint_required": True, "supported": False,
    },
    "hq_svc": {
        "name": "HQ-SVC",
        "homepage": "https://github.com/ShawnPi233/HQ-SVC",
        "checkpoint_required": True, "supported": False,
    },
}
PATH_FIELDS = ("engine_repository", "engine_python", "checkpoint", "model_config")
DEFAULTS = {key: {field: "" for field in PATH_FIELDS}
            for key in KNOWN_ENGINES if key != "rvc"}


def load_registry(path: Path = REGISTRY_FILE) -> dict:
    if not Path(path).exists():
        return {k: dict(v) for k, v in DEFAULTS.items()}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - set(DEFAULTS):
        raise ValueError("Unexpected SVC registry engine keys")
    result = {k: dict(v) for k, v in DEFAULTS.items()}
    for engine, values in data.items():
        if not isinstance(values, dict) or set(values) - set(PATH_FIELDS):
            raise ValueError(f"Invalid registry fields for {engine}")
        for key, value in values.items():
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"Invalid {engine}.{key} path")
            result[engine][key] = value.strip()
    return result


def save_engine_config(engine: str, paths: dict, registry_file: Path = REGISTRY_FILE) -> dict:
    if engine not in DEFAULTS or not KNOWN_ENGINES[engine]["supported"]:
        raise ValueError(f"Engine is not configurable: {engine}")
    if set(paths) - set(PATH_FIELDS):
        raise ValueError("Unknown engine configuration field")
    new_data = load_registry(registry_file)
    for key, value in paths.items():
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError(f"Invalid {key} path")
        new_data[engine][key] = value.strip()
    target = Path(registry_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace; prevent partially written model registry.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, suffix=".json",
        prefix=".svc-", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(new_data, stream, indent=2, ensure_ascii=False)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return new_data


def engine_health(engine: str, registry: dict | None = None) -> tuple[bool, str]:
    if engine not in KNOWN_ENGINES:
        return False, "未知 SVC 引擎"
    if not KNOWN_ENGINES[engine]["supported"]:
        return False, "尚未接入推理后端（仅在规划中）"
    if engine == "rvc":
        return True, "使用现有 RVC .pth/.index 角色管理"
    values = (load_registry() if registry is None else registry)[engine]
    repo = Path(values.get("engine_repository") or ".").expanduser()
    python = Path(values.get("engine_python") or ".").expanduser()
    if not values.get("engine_repository") or not repo.is_dir():
        return False, "请先配置有效的引擎仓库目录"
    entry = repo / ("inference.py" if engine == "seed_vc" else "my_inference.py")
    if not entry.is_file():
        return False, f"缺少引擎入口：{entry.name}"
    if not values.get("engine_python") or not python.is_file():
        return False, "请配置已安装依赖的独立 Python 解释器路径"
    required = ("checkpoint", "model_config") if KNOWN_ENGINES[engine]["checkpoint_required"] else ()
    for field in required:
        if not values.get(field) or not Path(values[field]).expanduser().is_file():
            return False, f"缺少 {field} 权重/配置文件"
    if bool(values.get("checkpoint")) != bool(values.get("model_config")):
        return False, "checkpoint 与 model_config 必须同时设置"
    if values.get("checkpoint") and not Path(values["checkpoint"]).expanduser().is_file():
        return False, "权重路径不存在"
    if values.get("model_config") and not Path(values["model_config"]).expanduser().is_file():
        return False, "模型 YAML 路径不存在"
    return True, "本地文件齐备（推理能力仍需 GPU 测试）"


def engine_choices() -> list[tuple[str, str]]:
    """Expose installed-capable engines only; prospective models are catalog entries."""
    return [(meta["name"], name) for name, meta in KNOWN_ENGINES.items()
            if meta["supported"]]


def status_markdown() -> str:
    entries = load_registry()
    rows = ["| 引擎 | 状态 | 来源 |", "|---|---|---|"]
    for key, meta in KNOWN_ENGINES.items():
        ok, info = engine_health(key, entries)
        rows.append(f'| {meta["name"]} | {"✅" if ok else "⚠️"} {info} | [仓库]({meta["homepage"]}) |')
    return "\n".join(rows)


def get_engine_job_options(engine: str, registry: dict | None = None) -> dict:
    if engine not in ("seed_vc", "yingmusic_svc"):
        raise ValueError("Engine is not a supported external SVC backend")
    values = load_registry() if registry is None else registry
    healthy, reason = engine_health(engine, values)
    if not healthy:
        raise RuntimeError(f"{engine} 未就绪: {reason}")
    options = values[engine]
    return dict(engine_repository=Path(options["engine_repository"]).expanduser(),
                engine_python=Path(options["engine_python"]).expanduser(),
                checkpoint=Path(options["checkpoint"]).expanduser() if options["checkpoint"] else None,
                model_config=Path(options["model_config"]).expanduser() if options["model_config"] else None)
