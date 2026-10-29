"""Exercise frozen native desktop Qt GUI and audio backends without hardware."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_audio(output: Path) -> dict:
    import librosa
    import numpy as np
    import soundfile as sf
    from lib.mixer import _juce_reverb_kernel_float64, apply_reverb, mix_vocals_and_accompaniment

    kernel = _juce_reverb_kernel_float64
    source = Path(kernel.py_func.__code__.co_filename)
    if not source.is_absolute() or not source.is_file():
        raise RuntimeError(f"Numba kernel source is not relocatable: {source}")
    signal = np.zeros((2, 48000), dtype=np.float64)
    signal[:, 0] = [0.200000000000001, 0.100000000000001]
    double = apply_reverb(signal, 48000, room_size=0.3, wet_level=0.2)
    single = apply_reverb(signal.astype(np.float32), 48000, room_size=0.3, wet_level=0.2)
    if double.dtype != np.float64 or double.shape != signal.shape or not np.isfinite(double).all():
        raise RuntimeError("FLOAT64 reverb lost precision, shape or finite samples")
    if not np.any(double[:, 1000:]) or not kernel.nopython_signatures:
        raise RuntimeError("Reverb did not produce a tail using native JIT compilation")
    np.testing.assert_allclose(double, single, rtol=2e-4, atol=1e-7)
    resampled = librosa.resample(double, orig_sr=48000, target_sr=44100, res_type="kaiser_fast")
    if resampled.shape != (2, 44100) or not np.isfinite(resampled).all():
        raise RuntimeError("Bundled resampler produced invalid audio")
    # Exercise another cached librosa kernel, not just its import or soxr path.
    spectrum = np.abs(librosa.stft(resampled[0]))
    librosa.amplitude_to_db(spectrum)
    librosa.util.normalize(resampled)
    output.mkdir(parents=True, exist_ok=True)
    vocals, accompaniment, mixed = (output / name for name in ("vocals.wav", "accompaniment.wav", "mix.wav"))
    sf.write(vocals, signal.T, 48000, subtype="DOUBLE")
    sf.write(accompaniment, signal.T * 0.1, 48000, subtype="DOUBLE")
    mix_vocals_and_accompaniment(str(vocals), str(accompaniment), str(mixed), reverb_amount=0.2)
    audio, rate = sf.read(mixed, dtype="float64", always_2d=True)
    if sf.info(mixed).subtype != "DOUBLE" or rate != 48000 or audio.shape != (48000, 2) or not np.isfinite(audio).all():
        raise RuntimeError("Bundled mixer failed the DOUBLE WAV round trip")
    return {"source": str(source), "cache": kernel._cache.cache_path,
            "cache_hits": sum(kernel.stats.cache_hits.values()), "samples": len(audio)}


def check_desktop() -> list[str]:
    """Exercise the actual frozen Qt application without HTTP or microphones."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    from desktop.gui import DesktopWindow

    app = QApplication.instance() or QApplication([])
    window = DesktopWindow()
    try:
        assert window.windowTitle().startswith("AI-RVC Studio")
        tabs = window.findChild(QTabWidget)
        if tabs is None or tabs.count() != 3:
            raise RuntimeError("Desktop Studio is missing expected native pages")
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        if not all(labels) or not any("实时" in label for label in labels):
            raise RuntimeError("Realtime voice conversion tab is missing")
        return labels
    finally:
        # Qt owns native audio/graphics resources and a timer tied to the window.
        # Stop callbacks, explicitly destroy QObjects while QApplication exists,
        # and flush deferred deletes before Python unloads bundled Qt libraries.
        from PySide6.QtCore import QCoreApplication, QEvent
        window.timer.stop()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        app.quit()




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()
    report = {"frozen": bool(getattr(sys, "frozen", False)), "audio": check_audio(args.output)}
    if not args.audio_only:
        report["desktop_tabs"] = check_desktop()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
