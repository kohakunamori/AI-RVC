"""Run cold/warm frozen checks outside the build tree with a dedicated cache."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def verify(executable: Path, *, audio_only: bool = False) -> list[dict]:
    executable = executable.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="ai-rvc-package-中文-") as directory:
        root = Path(directory)
        cache = root / "numba-cache"
        env = dict(os.environ, PYTHONPATH="", PYTHONHOME="", PYTHONUTF8="1",
                   AI_RVC_DEVICE="cpu", NUMBA_CACHE_DIR=str(cache),
                   NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator",
                   NUMBA_DEBUG_CACHE="1", QT_QPA_PLATFORM="offscreen",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        reports = []
        for phase in ("cold", "warm"):
            output = root / phase
            arguments = [str(executable), "--internal-worker", "package-smoke", "--output", str(output)]
            if audio_only or phase == "warm":
                arguments.append("--audio-only")
            subprocess.run(arguments, cwd=root, env=env, check=True, timeout=300)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            if not report["frozen"] or not Path(report["audio"]["cache"]).is_relative_to(cache):
                raise RuntimeError("The smoke test did not use the frozen app and isolated cache")
            if phase == "warm" and report["audio"]["cache_hits"] < 1:
                raise RuntimeError("The second process did not reuse the compiled reverb cache")
            reports.append(report)
        if not list(cache.rglob("*.nbc")):
            raise RuntimeError("The frozen runtime produced no native Numba cache")
        return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.executable, audio_only=args.audio_only), indent=2))


if __name__ == "__main__":
    main()
