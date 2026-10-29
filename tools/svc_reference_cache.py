"""Private local cache for SoulX target-reference F0 preprocessing.

The source song is never cached here. Only a *reference singer* whose file
contents and preprocessing implementation are unchanged is reused. No GPU
objects are persisted. Disable via AI_RVC_REFERENCE_CACHE=0.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

CACHE_FORMAT = "soulx-reference-preprocessing-v1"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ready(root: Path, expected_key: str) -> bool:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return (manifest.get("key") == expected_key
                and (root / "vocal.wav").is_file()
                and (root / "vocal.wav").stat().st_size > 44
                and (root / "vocal_f0.npy").is_file()
                and (root / "vocal_f0.npy").stat().st_size > 20)
    except (OSError, ValueError, TypeError):
        return False


def prepare_soulx_reference(
    reference_file: Path,
    model_config: Path,
    implementation_file: Path,
    fallback_directory: Path,
    run_preprocess: Callable[[Path], tuple[Path, Path]],
) -> tuple[Path, Path]:
    """Reuse prepared reference vocal+F0 across songs with identical inputs.

    Cache keys include audio bytes, model configuration and preprocessing
    source so a new upstream version cannot silently reuse stale features.
    """
    if os.environ.get("AI_RVC_REFERENCE_CACHE", "1").lower() in ("0", "false", "off"):
        return run_preprocess(fallback_directory)
    base = os.environ.get("AI_RVC_SVC_CACHE_DIR", "").strip()
    if not base:
        return run_preprocess(fallback_directory)
    digest = hashlib.sha256(CACHE_FORMAT.encode("ascii"))
    for source in (reference_file, model_config, implementation_file):
        digest.update(_digest_file(Path(source)).encode("ascii"))
    key = digest.hexdigest()
    root = Path(base).expanduser().resolve() / "preprocess" / "soulx_reference"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / key
    if _ready(dest, key):
        return dest / "vocal.wav", dest / "vocal_f0.npy"
    if dest.exists():
        raise RuntimeError(f"Stale or partial SoulX reference cache: {dest}; remove it before retrying")
    staging = Path(tempfile.mkdtemp(prefix=".reference-", dir=root))
    try:
        audio, f0 = run_preprocess(staging)
        audio, f0 = Path(audio), Path(f0)
        if not audio.is_file() or not f0.is_file():
            raise FileNotFoundError("Preprocessed reference vocal or pitch is missing")
        for item, name in ((audio, "vocal.wav"), (f0, "vocal_f0.npy")):
            dest_file = staging / name
            if item.resolve() != dest_file.resolve():
                shutil.copy2(item, dest_file)
        (staging / "manifest.json").write_text(json.dumps({
            "format": CACHE_FORMAT, "key": key,
        }), encoding="utf-8")
        if not _ready(staging, key):
            raise RuntimeError("SoulX reference preprocess output invalid")
        try:
            staging.rename(dest)
        except OSError:
            if not _ready(dest, key):
                raise
        return dest / "vocal.wav", dest / "vocal_f0.npy"
    finally:
        shutil.rmtree(staging, ignore_errors=True)
