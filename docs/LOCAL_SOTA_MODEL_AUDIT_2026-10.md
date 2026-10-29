# AI-RVC: verifiable local/free model audit (2026-10-29)

This is an **availability/compatibility audit**, not a claim that every default checkpoint is globally SOTA. Rankings are task- and validation-set-specific; benchmark scores are not interchangeable.

## Eligible local model routes

| Stage | Current stable default | Free/local candidate | Decision |
| --- | --- | --- | --- |
| Vocals | pcunwa Leap XE 90 (SHA-pinned in `infer/separator.py`) | Keep pinned Leap XE + Leap instrumental 62 | No verified freely downloadable *better* checkpoint with reproducible end-to-end results on our target set; keep reliable baseline |
| Lead/back karaoke | MVSep 9205 average (three SHA-pinned checkpoints) | Retain 9205 until a reproducible, downloadable higher-scoring replacement is verified | No invented checkpoint URL or unverified MVSep 9570 weight |
| Dereverb | anvuew Stereo BS-RoFormer 22.5050 (pinned) | Keep existing checkpoint; add configurable `force` / `off` | MVSep 2026.07 beats it on benchmark, but weight availability/local compatibility was not independently verified |
| F0 | RMVPE | **SwiftF0 0.3.0 (MIT)** via optional local install | New custom-VC route, verified public PyPI package/API; should be A/B compared with RMVPE on singing |
| Encoder / RVC voice conversion | HuBERT + existing RVC v1/v2 | No drop-in replacement | Existing `.pth`/`.index` content-feature contracts must be preserved |

## Published results (not local measurements)

- MVSep **BS-RoFormer 124 bands 2026.07**: Multisong vocal SDR **12.33**. MVSep says this model is temporarily **paid only**; it is **not eligible** for a free API/local default without verified public weights.
- MVSep **Mel-RoFormer Karaoke / Duet #9570**: lead **11.0848** and back **7.1379** SDR. We could not independently verify a free downloadable checkpoint/config, therefore do not add a fake model entry.
- Existing MVSep **9205**: lead **10.4296** SDR. Its three local checkpoints are already pinned by hash.
- MVSep **2026.07 dereverb**: vocal SDR **9.81**, while existing anvuew Stereo model scores **8.80**. No verified freely downloadable replacement checkpoint.
- **SwiftF0 0.3.0**: optional MIT-licensed CPU/ONNX local package; public pitch benchmark reports its pitch-F1 statistically tied with RMVPE. It is NOT proven better for every Japanese singing style.

Sources:
- [MVSep 2026 updates](https://www.mvsep.com/en/news)
- [MVSep lead/back leaderboard](https://www.mvsep.com/quality_checker/leaderboard/lead_back_vocals)
- [MVSep BS RoFormer models](https://mvsep.com/algorithms/34)
- [SwiftF0 upstream](https://github.com/lars76/swift-f0)
- [SwiftF0 PyPI 0.3.0](https://pypi.org/project/swift-f0/)

## Usage

Stable mode: no action required. Original default models continue working.

Opt in to **local SwiftF0** in `configs/config.json`:
```json
{
  "cover": {
    "use_official": false,
    "vc_pipeline_mode": "current",
    "f0_method": "swiftf0"
  }
}
```
Install the optional dependency in the same Python environment:

```bash
python -m pip install -r requirements_sota.txt
```

Do NOT set `f0_method=swiftf0` with the official RVC pipeline. The upstream official adapter does not implement this method; config validation intentionally rejects it. SwiftF0's native 16 ms pitch output is mapped to RVC's 10 ms grid without filling voiced gaps or extrapolating beyond the source.

For already-clean lead stems, `cover.dereverb_mode = "off"` skips the forced RoFormer dereverb stage in **song input mode**. Default `"force"` preserves existing behavior. `input_mode="vocals"` already skips separation and dereverb.

## Required before switching stable defaults

1. Locate a **legitimately redistributable** checkpoint, config, and stable URL; verify SHA-256 and actual local inference.
2. Run fixed-song A/B tests: Multisong SDR is not sufficient. Evaluate singing F0, sibilants, high notes, bleed, final RVC speaker similarity, naturalness, and latency.
3. Check Windows CUDA/CPU, DML, packaging, RAM/VRAM and precision; no silent downgrades or unsupported paid endpoints.
4. Only then revise model defaults, pin weight revisions, update the UI/settings, and add release integration checks.

Nothing in this branch downloads or uploads user audio to a paid service. No API key is required.
