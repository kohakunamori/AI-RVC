# AI-RVC multi-engine SVC roadmap: Naturalness, timbre and song fidelity
Updated 2026-10-29. Status: design + two isolated CLI bridges committed; **no real-GPU end-to-end acceptance yet**.

## Product direction

- Retain RVC v1/v2 and existing character checkpoints as a legacy/high-quality trained baseline.
- Add zero-shot singing backends with *authorized target reference recordings*. RVC pth/index are **not** reference audio or compatible weights for different architectures.
- Process: song -> vocals/instrumental stems -> optional lead/back split -> adjustable DeReverb -> SVC engine -> non-destructive mixing -> blinded quality comparison.
- Require free local code/weights. No silent API, no premium inference, no secretly substituted model.

## Model shortlist (no universal cross-dataset SOTA claim)

| Model | Access | Strength and limitation | Status |
|---|---|---|---|
| RVC v2 | existing local pth/index | Strong trained-character identity baseline | Existing project unchanged |
| Seed-VC singing 44k | [code](https://github.com/Plachtaa/seed-vc), [checkpoints](https://huggingface.co/Plachta/Seed-VC), GPL-3 | Local F0-conditioned, direct M4Singer vs RVC comparison; official repo archived | New isolated bridge implemented; inference untested |
| YingMusic-SVC | [MIT project](https://github.com/GiantAILab/YingMusic-SVC), [full checkpoint](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt) | Singing-specific Flow training; robust to harmony and F0 errors; CUDA external env | New isolated bridge implemented; inference untested |
| HQ-SVC | [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40249), [code](https://github.com/ShawnPi233/HQ-SVC), [weights](https://huggingface.co/shawnpi/HQ-SVC) | High-fidelity zero-shot and super-resolution; Linux/WSL CUDA 11.8+ | Planned headless adapter; inspect code and weight licenses before redistribution |
| SoulX-Singer-SVC | [Apache-2.0 source & weights](https://github.com/Soul-AILab/SoulX-Singer), [pretrained models](https://huggingface.co/Soul-AILab/SoulX-Singer) | 2026 zero-shot audio-to-audio SVC with pitch control; prompt + source F0 preprocessing required; Japanese fidelity needs independent evaluation | High-priority next adapter; local code/weights verified but not implemented here |

Published Seed-VC M4Singer experiment: RVC v2 F0CORR 0.9404 / RMSE 30.43 / SECS 0.7264 / OVRL 3.12; Seed-VC 0.9375 / 33.35 / 0.7405 / 3.06. None dominates on every metric. Other research results use different evaluation datasets; don't mix benchmark numbers.

An independent **same-protocol comparison from the YingMusic-SVC paper (Mix Vocal condition, source lead + harmony)**: Seed-VC CER=17.30%, log-F0 correlation=84.02%, naturalness MOS=2.93, singer-similarity MOS=2.78; YingMusic-Full CER=15.90%, log-F0=86.47%, naturalness=3.31, similarity=3.12. These improvements are specific to that benchmark, not validated Japanese-song results. Source: https://arxiv.org/abs/2512.04793 (Table 2).

## Implemented now: local, separately installed engine subprocesses

Code: infer/svc_engine_bridge.py, tools/run_svc_engine.py, tools/yingmusic_svc_worker.py.

Both backends consume separated, dry lead singing audio and a target speaker reference. They must run inside their own installed Python environment. Outputs are validated and normalized to Float32 WAV. Existing one-click UI and default RVC inference remain unchanged.

Seed-VC example (bash, adjust paths for Windows):

    python tools/run_svc_engine.py --engine seed_vc \
      --engine-repo "/path/seed-vc" --engine-python "/path/seed-vc/venv/bin/python" \
      --source "/path/lead.wav" --reference "/path/target_singer.wav" \
      --output "/path/converted_seed.wav" --steps 35 --pitch-shift 0

For Seed the singing path uses f0-condition True and auto-f0-adjust False to preserve key, with a 44.1kHz pretrained checkpoint downloaded from Hugging Face on first run (no paid API). For fully offline runs, prefetch model weights and dependencies.

YingMusic-SVC example:

    python tools/run_svc_engine.py --engine yingmusic_svc \
      --engine-repo "/path/YingMusic-SVC" --engine-python "/path/ymsvc/bin/python" \
      --source "/path/lead.wav" --reference "/path/target_singer.wav" \
      --checkpoint "/path/YingMusic-SVC-full.pt" \
      --config "/path/YingMusic-SVC/configs/YingMusic-SVC.yml" \
      --output "/path/converted_ying.wav" --steps 35 --pitch-shift 0

**YingMusic upstream CLI uses automatic pitch adjustment when semi_tone_shift is None.** The custom worker explicitly passes the requested semitone value; default zero avoids unnecessary transposition against the original accompaniment. No reverb or accompaniment mixing happens inside our SVC worker. CUDA is required by this current worker.

Use the upstream model's separate environment installation instructions. Do not install both engines in the host AI-RVC requirements or vendor GPL dependencies without respecting the license.

## WebUI integration (implemented, not GPU-validated)

The existing **AI Cover** tab now has a **Singing Engine** dropdown:
- **RVC v1/v2**: remains selected by default; select a character checkpoint/index exactly as before.
- **Seed-VC / YingMusic-SVC**: upload a singer reference WAV or MP3 (requires permission to use that voice), choose diffusion steps and optional dereverb. Existing character RVC .pth/index are not required.
- For a full song, choose `song`; the selected Karaoke model produces lead vocals + backing/instrumental, the external SVC engine converts only the lead, and AI-RVC mixes the converted vocal with the accompaniment using the existing Float32 WAV export.
- For an already-isolated vocal, choose `vocals`; the interface outputs the converted vocal and deliberately does not fabricate an accompaniment or mixed song.

The **SVC 模型管理 / Models** tab centrally lists RVC, Seed-VC, YingMusic-SVC and planned HQ-SVC / SoulX-Singer-SVC. For installed external engines configure:
`engine_repository` (cloned source), `engine_python` (absolute Python executable within isolated env), plus `checkpoint` and `model_config` (required together for YingMusic, optional for Seed). These are saved locally in `configs/svc_engines.json`; path validation does **not** prove that dependencies, CUDA or inference have passed. No automatic pip install and no paid API.

**When switching to Seed-VC or YingMusic-SVC, the character dropdown becomes inactive**; when returning to RVC, the original character selection and conversion pathway are retained.

Known limitations: GPUs, official model checkpoints and Windows 50-series execution are untested in this PR. New SVC engines may still require model downloads using their own upstream installation instructions; the UI does not automatically install code, Python or checkpoints. Karaoke default is MVSep 9205 and uses existing weights. New engines bypass existing RVC-specific source constraints, protecting unprocessed high frequency detail.

## Development schedule (gated, no implied completion)

### P0 — Accept actual model inference
1. Run real 1–3 min singing stems on Windows 50-series GPU / WSL and NVIDIA CUDA where available. Record Torch/CUDA versions, model file versions, peak VRAM, failure logs, song duration.
2. Validate that no key change occurs: compare source and output F0 under same instrumentation; compare with original backing harmony and note alignment.
3. Validate long-song 44.1kHz WAV transitions (clicks, missing syllables, drift), proper EOF, reference audio effect, 16-bit vs Float32 exports.
4. Test low/high voice and falsetto, breathy consonants, fast Japanese phrasing and long vibrato.
5. Only then mark the engines as runtime validated.

### P1 — Integrate true one-click engine selection
1. Introduce config property svc_engine = rvc, seed_vc, yingmusic_svc, hq_svc.
2. Switch UI between a trained RVC model picker and a zero-shot reference audio upload. Never demand RVC .pth when zero-shot engine selected.
3. Reuse separation + karaoke + dereverb once. Set engine-specific sample rate and pitch knobs. Preserve backing phase and original song key.
4. Gate extra source-constrained DSP and clipping with explicit quality A/B; don't apply RVC-specific heuristics indiscriminately to newer generators.
5. Add model registry with checkpoint URLs/revisions/sha256 and separate weight license conditions. Show free vs merely public, install size, VRAM, compute.

### P1 — Optimize singing naturalness and character identity
- Capture dry voice references over multiple registers and vowel types; reference sample cleanliness matters more than raw duration.
- Avoid pitch auto-adjust on original-key covers. Preserve vibrato depth/rate, glides, head/chest register transitions, breaths, sibilants and vowel tails.
- Test changes in naturalness separately from similarity: high target timbre score can mask robotic enunciation and phoneme loss.
- Preserve converted raw vocals; apply postprocessing per track with toggles, not forced serial gates. Reintroduce calibrated room reverb only after reconstruction, never overprocess source.
- Separate identity leakage to original singer from resemblance to reference, rather than just using one speaker embedding score.

### P2 — SoulX-Singer-SVC / HQ-SVC, reference and timbre enhancements
- Prioritize SoulX-Singer-SVC (Apache-2.0, public SVC weights, local F0 pipeline) for a third engine adapter; check whether its audio-to-audio preprocess preserves Japanese lyrics and whether high notes are stable.\n- Adapt official HQ-SVC headlessly in Linux/WSL CUDA, only after verified reference inference and license checks. Add on-device engine readiness tests.
- Investigate multi-reference conditioning, safe vibrato preservation, source/destination loudness normalization, phase-consistent backing mixing.
- Test fine-tuning per singer where a zero-shot engine underperforms a high-quality trained RVC checkpoint. Fine-tuning is an alternative, not a free win.

## Benchmark spec

At least 20 authorized short clips across female/male, high note, low note, vibrato, fast Japanese lyrics, falsetto, dense harmonies, long-note tails and noisy sources. Use exactly the *same stems, target samples, backing and loudness normalization* across engines; A/B/X shuffle identities for listeners.

| Score axis | Quantitative metrics | Listening review |
|---|---|---|
| Pitch accuracy | F0CORR, F0RMSE, octave error, voiced recall, vibrato rate/depth | Glides and long notes sound natural |
| Timbre identity | cosine speaker embedding with controls across register; cross-speaker confusion | Reference resemblance, source identity leakage |
| Vocal/lyric fidelity | Japanese ASR CER as supplementary; phoneme alignment | Japanese consonants, vowels, phrasing |
| Naturalness | adapted MOS/SingMOS when calibrated, not as ground truth | metallic buzz, breath, false high harmonics, formant drift |
| Artifacts | spectral clipping, clicks, discontinuities, duration drift | reverb tails, transitions, noise in silence |
| Cost | RTF, peak VRAM, RAM, disk, platform failure rate | latency-quality tradeoff |

Promote a default engine only if it is available for genuinely free local use, passes end-to-end integration with reversible fallback, and wins *on the target test suite without major regressions*. No universal paper-only ranking.
