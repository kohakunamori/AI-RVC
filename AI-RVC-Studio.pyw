# -*- coding: utf-8 -*-
"""Double-click on Windows to open the native AI-RVC Studio without a console."""
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from lib.ffmpeg_runtime import configure_ffmpeg_runtime
    configure_ffmpeg_runtime()
    from desktop.gui import launch
    raise SystemExit(launch())
