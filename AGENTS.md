# Codex Project Guidelines — SesaML

This file contains repository architecture rules, environment constraints, and development guidelines for Codex AI agents working on **SesaML** (Akan/Twi Speech Recognition).

> **Note**: This file is excluded from git via `.gitignore` to prevent tracking local AI assistant instructions.

---

## 1. Project Overview & Architecture

SesaML is an end-to-end Automatic Speech Recognition (ASR) system for Akan (Twi) built with PyTorch and HuggingFace.

**Current state:** a trained Conformer-CTC at **WER 0.3917 / CER 0.1264** (`conformer/v004` decoded with beam search + the character n-gram LM; 0.4705 greedy), plus speaker diarization. See [`docs/asr/results.md`](docs/asr/results.md) for every measured number.

### Repository layout
Every tree is split by **model domain**, so the translation model of phase two lands beside speech recognition rather than through it. `asr/` is what exists today; `translation/` is the sibling it gets.

```
src/      config.py, main.py, utils/   shared     |  src/asr/{models,data,decoding,training,inference,diarization}
scripts/  common.sh, setup/, datasets/ shared     |  scripts/asr/{train,evaluate,transcribe,export_model,serve_app}
outputs/  logs/, models/               shared     |  outputs/asr/{registry,checkpoints,runs,exports,lm}
docs/     info.md, roadmap_next_steps  shared     |  docs/asr/{conformer,deepspeech,whisper,results,...}
```

`PathConfig.domain_dir` (default `outputs/asr`) is what run directories, checkpoints and the registry hang off; `logs/` stays shared so one directory still answers "what happened in run X". A translation run sets `domain`/`domain_dir` and cannot collide with an ASR checkpoint or a registry version of the same name.

### Core Modules (`src/`)
- `src/config.py`: Configuration tunables and automatic device detection (`mps` on Apple Silicon, `cuda`, or `cpu`). Audio front-end is **80 log-mel bands, n_fft 400, hop 160**; `AudioConfig.n_mels` is the single source of truth for feature width — `ModelConfig` deliberately has no `n_feats`.
- `src/asr/models/`:
  - `deepspeech.py`: DeepSpeech2 CNN + BiGRU CTC baseline (~23M params, 2× subsampling). The GRUs are length-aware via `pack_padded_sequence`.
  - `conformer.py`: Conformer CTC (~8.4M small / ~27M medium, 4× subsampling with dynamic attention masking).
  - `whisper_model.py`: Optional HuggingFace Whisper wrapper, used as a comparison baseline only. No default repository — it runs solely when `--whisper-repo` or `MODEL_REPO_ID` names one.
  - `__init__.py`: Architecture registry (`build_architecture`, `subsampling_factor`).
- `src/asr/data/`:
  - `text_transform.py`: Twi vocabulary (`a-z`, `0-9`, `ɛ`, `ɔ`, `'`, space, blank) plus typographic folding (`’` → `'`).
  - `audio_transforms.py`: **log-mel → per-utterance CMVN → SpecAugment**, in that order.
  - `dataset.py` & `hf_dataset.py`: Collator, corpus probes, HuggingFace loader. Both fail loudly on missing columns or missing audio.
  - `bucketing.py`: Length-bucketed batching (20% → 97% padding efficiency).
  - `feasibility.py`: Which samples CTC can align at all. 18.6% of the corpus could not, and returned no gradient.
  - `alignment.py`: CTC forced alignment (own Viterbi; torchaudio's is deprecated). Validated against that reference.
  - `segmenting.py`: Cuts long clips into utterances at aligned pauses, splitting audio and text together, on word boundaries.
- `src/asr/diarization/`: `turns.py` (turn algebra), `segmentation.py` (energy VAD), `backends.py` (pyannote / ECAPA / spectral), `pipeline.py` (`DiarizedTranscriber`).
- `src/asr/training/` & `src/asr/inference/`: Length-aware forwarding, AMP, gradient clipping, resume-from-checkpoint, `Transcriber` caching, TorchScript/ExecuTorch export.
- `src/asr/decoding/`: `language_model.py` (character n-gram, stupid backoff), `beam_search.py` (CTC prefix beam search with LM shallow fusion), `decoder.py` (`build_decoder`, shared by inference and evaluation so both decode identically).
- `src/utils/`: `audio_io.py` (soundfile-backed decoding), `model_registry.py` (versioned exports), `run_logger.py` (run directories, checkpoint resolution).

---

## 2. Environment & Python Version Constraints

- **Python Version**: **`≥ 3.10, < 3.14`** (Python **3.11** or **3.12** recommended).
  - *Critical constraint*: Python 3.14+ is unsupported by `numba`/`llvmlite` C-extensions.
- **Auto-Environment Installers**:
  - Linux / macOS: `scripts/setup/ensure_python.sh` (auto-installs 3.11 via `uv` or `apt`).
  - Windows: `scripts/setup/ensure_python.ps1` (auto-installs 3.11 via `uv` or `winget`).
- **Dependencies**: `requirements.txt` (requires `portaudio` via `brew install portaudio` on macOS for `PyAudio`).

### Known environment traps
- **`torchaudio.load` needs `torchcodec`**, which needs a matching FFmpeg. Always decode through `src/utils/audio_io.py`, which falls back to `soundfile`. Never call `torchaudio.load` directly.
- **`datasets` 4.x decodes audio through torchcodec too**, and indexing *any* column decodes the whole row. `hf_dataset.py` casts to `Audio(decode=False)` and decodes the bytes itself.
- **pyannote models are gated** — three separate repos must be accepted. The `ecapa` backend needs no licence and is the default.
- **Windows consoles are cp1252**: logging a Twi `ɛ`/`ɔ` will crash a run unless stdout is reconfigured (handled in `run_logger.py`).
- On Windows, `DataLoader(num_workers>0)` requires an `if __name__ == "__main__":` guard in the entry point.
- **Detached Windows runs die when their console closes**, in two separate ways. tqdm's write to a dead stdout blocks then raises (handled in `src/utils/progress.py`), and *below Python* the Intel Fortran runtime inside MKL aborts on `CTRL_CLOSE_EVENT` with `forrtl: error (200): program aborting due to window-CLOSE event`. Long runs must set **`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`**; one lost 26 of 40 epochs to it.

---

## 3. Key Developer Workflows & Commands

### Script layout
Scripts are grouped by stage of work, so the **translation** phase of the project gets its own sibling directory instead of being mixed into the ASR wrappers:

```
scripts/
├── common.sh    ├── setup/       interpreter + venv bootstrap
├── run_tests.sh ├── datasets/    corpus acquisition (shared across models)
                 ├── asr/         train, evaluate, transcribe, export_model, serve_app
                 ├── lm/          build_lm, build_translation_worklist
                 └── maintenance/ clean_outputs, summarize_runs
```

`common.sh` finds the repo root by walking up to `requirements.txt` rather than counting `..` levels, and exports `SCRIPTS_DIR` for wrappers that invoke a sibling — a hard-coded `..` breaks the moment a script is nested. Python scripts under `scripts/<group>/` resolve the root with `parents[2]`.

```bash
# 1. Environment Setup (auto-detects python3.11 / python3.12)
scripts/setup/setup_env.sh

# 2. Download all 3 Akan speech datasets at a go
python scripts/datasets/download_all_datasets.py --num-samples 100

# 3. Model Training (--architecture deepspeech / conformer / conformer-medium)
scripts/asr/train.sh --architecture conformer --epochs 80 --batch-size 32 --lr 8e-4

# 4. Evaluation & Metrics (WER / CER)
scripts/asr/evaluate.sh --architecture conformer

# 5. Transcription & Inference
scripts/asr/transcribe.sh data/sample.wav --architecture conformer --noise-reduction

# 6. Speaker-attributed transcript (who spoke when + what was said)
python -m src.main diarize --audio recording.wav --backend auto --num-speakers 2

# 7. Model versions — list, promote, roll back
python -m src.main models list
python -m src.main models rollback --architecture conformer

# 8. Demo app (Transcribe + Speaker Diarization tabs)
scripts/asr/serve_app.sh

# 9. Prune outputs/ (dry run by default; never touches the registry)
python scripts/maintenance/clean_outputs.py
python scripts/maintenance/clean_outputs.py --apply

# 10. Language model for decoding (17.2% relative WER gain, no retraining)
python scripts/lm/build_lm.py
python -m src.main evaluate --architecture conformer --decoder beam

# 11. Run Test Suite (187 tests)
bash scripts/run_tests.sh
```

### Publishing metrics
Checkpoints are selected on **WER**, and the registry publishes the metrics captured when that checkpoint was written. These were once decoupled - the checkpoint was chosen on val_loss while the published metrics came from the best-WER epoch - which promoted a model worse than the one it replaced under a better-looking number. **Verify a published WER with an `evaluate` run before trusting it in docs or slides.**

### Training notes
- A data source is **required** (`--hf-dataset` or `--csv-path`); there is no default corpus. The old default was a text-only translation CSV that trained to a fake loss of 0.0.
- A validation set is required for multi-epoch runs unless `--allow-no-validation`.
- Long runs: `--resume outputs/asr/checkpoints/<run_id>/last_model.pt`.
- Batches are length-bucketed by default; `--no-bucket-batches` disables it.
- Every run reports **CTC feasibility**; `--min-headroom 1.0` drops samples whose target is longer than their audio can align. A sample that cannot align returns no gradient but still occupies a batch slot.
- Every finished run is **published to the model registry automatically**, but promoted only if it beats the served version's WER.

### Decoding
Greedy decoding argmaxes each frame independently, so it cannot prefer a spelling that exists in Twi. Beam search with a character n-gram LM fused in measured **0.4705 → 0.3917 WER**, no retraining. `DecodingConfig` holds the swept weights (α 0.5, β 0.5, beam 25); past α ≈ 0.8 the LM overrules the acoustics and WER climbs again.

Inference and evaluation both go through `build_decoder`, so a reported WER describes the system the app serves. **Training-time validation stays greedy** — beam search is ~250x slower per utterance and runs on every clip every epoch. The LM lives at `outputs/asr/lm/twi_char.json`, is not committed, and is rebuilt with `scripts/lm/build_lm.py`.

Not KenLM: `kenlm`/`pyctcdecode` pin `numpy<2` against this project's numpy 2.3.3 under torch 2.9.

### Twi dialect
Every audio corpus, **including the split WER is measured on**, is Asante Twi (98.9% on doubled-vowel markers). `data/corpus/verified_data.csv` is Akuapem-leaning (77.8%) and is **excluded from the language model by default** — an ablation showed dropping its 25,420 lines leaves WER unchanged and improves CER. `Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` scores the same 77.6% and made both metrics worse, so it is not used.

The measured lesson: for the LM, **matching dialect and domain beats volume** — 17,544 in-domain transcripts beat 49,703 lines of mixed text. New Twi text is only worth adding if it is Asante **and lexically rich**; `scripts/lm/build_translation_worklist.py` produces the English worklist for that.

Two large candidates were measured and rejected: `Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` (worse on both metrics) and `michsethowusu/twi-english-parallel-synthetic-50m` (47.9M rows that are a few hundred words recombined into 2-word pairs — 634 distinct types in 400k rows, and **worse than no LM at all** when used alone). **Judge a text corpus by distinct word types, not row count**: `crowdsourced_data.csv` holds 1,124 types in 695 lines, nearly twice what 400,000 synthetic rows contain.

### Serving models
The project serves **its own trained models**, resolved from the registry. There is no default Whisper repository: `MODEL_REPO_ID` is optional and empty, and the app hides the `whisper` option unless it is set. Anything that picks a model must go through the registry — never `latest_checkpoint()` alone, which returns `None` once superseded run directories are pruned and would silently fall back to a third-party model.

### Model registry (`outputs/asr/registry/`)
The app, CLI and evaluation serve the **promoted** version, never the newest file. `ModelRegistry.publish()` archives; promotion is conditional on improvement, so a collapsed run cannot replace a working demo. `models rollback` returns to the last version that was actually served. Each version records the front-end settings it was trained on — loading 80-mel weights under a 128-mel config degrades silently rather than raising, so the registry compares and warns.

In git: `registry.json` and each `metadata.json` are tracked; `.pt` weights are not.

---

## 4. Documentation Index (`docs/`)

- [`docs/asr/results.md`](docs/asr/results.md): **Measured results** — WER/CER, learning curves, published exports, throughput, the bugs that made training work, and the 500h negative result.
- [`docs/asr/deepspeech.md`](docs/asr/deepspeech.md): DeepSpeech2 architecture & baseline rationale.
- [`docs/asr/conformer.md`](docs/asr/conformer.md): Conformer CTC architecture & 4× subsampling efficiency.
- [`docs/asr/whisper.md`](docs/asr/whisper.md): Whisper fine-tuning & transfer learning.
- [`docs/asr/speaker_diarization.md`](docs/asr/speaker_diarization.md): Diarization pipeline — three backends (pyannote / ECAPA / spectral), turn post-processing, Gradio & CLI.
- [`docs/asr/noise_robustness.md`](docs/asr/noise_robustness.md): Training noise augmentation vs production Silero VAD & spectral gating.
- [`docs/roadmap_next_steps.md`](docs/roadmap_next_steps.md): What is done, and what is next (KenLM rescoring, making the 500h corpus usable).

---

## 5. Working Norms

- **Measure, don't assume.** Every performance claim in `docs/` has a number behind it produced by this repo. If something is unverified, say so in the text.
- **Fail loudly on bad data.** The most expensive bug in this project's history was a pipeline that silently substituted noise for missing audio and trained happily for 50 minutes. Prefer a hard error at construction over a fallback that produces plausible garbage.
- **A negative result is a result.** `run-big` (113h → WER 1.0) is documented as prominently as the successful run.
