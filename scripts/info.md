# Scripts

Entry points for the common SesaML workflows. Every script can be run from any
directory — they resolve the repository root themselves, load `.env`, and
activate `.venv` when it exists.

```bash
chmod +x scripts/*.sh scripts/*/*.sh      # once, after cloning
```

## Layout

Scripts are grouped by the stage of work they belong to. Speech recognition
lives under `asr/`; the second phase of the project is **translation**, which
gets its own sibling directory rather than being mixed in with these. Anything
shared by both — environment bootstrap, corpus downloads, run maintenance —
stays outside the per-model directories.

```
scripts/
├── common.sh              shared bash helpers, sourced by every wrapper
├── run_tests.sh           the test suite
├── setup/                 interpreter + virtualenv bootstrap
├── datasets/              corpus acquisition (shared across models)
├── asr/                   speech recognition: train, evaluate, transcribe, export, serve
├── lm/                    decoding language model + the translation worklist
└── maintenance/           outputs/ pruning and run summaries
```

`common.sh` locates the repository root by walking up to `requirements.txt`
rather than counting `..` levels, so a script keeps working wherever it is
moved, and exports `SCRIPTS_DIR` for wrappers that need to invoke a sibling.

| Script | What it does |
| --- | --- |
| `setup/setup_env.sh` | Creates `.venv`, installs `requirements.txt`, seeds `.env` and the `outputs/` tree. |
| `setup/ensure_python.sh` | Checks for compatible Python (>=3.10, <3.14) on Linux/macOS; auto-installs Python 3.11 via `uv` or `apt` if missing. |
| `setup/ensure_python.ps1` | Checks for compatible Python on Windows; auto-installs Python 3.11 via `uv` or `winget` if missing. |
| `datasets/download_dataset.sh` | Downloads a HuggingFace speech dataset into `data/` and writes a manifest to `outputs/`. |
| `datasets/download_all_datasets.sh` | Downloads all 3 Akan speech datasets into `data/` at a go (`ghanaopendata`, `Lagyamfi`, `ghananlpcommunity`). |
| `asr/train.sh` | Starts a training run; logs, metrics and checkpoints go to `outputs/`. |
| `asr/evaluate.sh` | Scores a checkpoint (loss/WER/CER) and saves per-sample predictions. |
| `asr/transcribe.sh` | Transcribes one audio file and stores the transcript. |
| `asr/export_model.sh` | Exports a checkpoint to `.pt` / `.pth` / `.pte` plus a tracked manifest. |
| `asr/serve_app.sh` | Serves the Gradio web app (transcription + speaker diarization). Reports which model version it is about to serve, and warns if none is published. |
| `lm/build_lm.py` | Builds the Twi character n-gram LM used for beam-search decoding. |
| `lm/build_translation_worklist.py` | Writes the English sentences awaiting Asante Twi translation to `data/corpus/english/`. |
| `maintenance/clean_outputs.py` | Prunes `outputs/` to the artifacts still in use. Dry run by default. |
| `maintenance/summarize_runs.py` | Prints a table of all recorded runs from `outputs/asr/runs/index.jsonl`. |
| `run_tests.sh` | Runs the test suite (pytest, falling back to `tests/run_all_tests.py`). |
| `common.sh` | Shared bash helpers; sourced by the others, not run directly. |

## Typical flow

```bash
# Standard setup (uses Python >= 3.10, < 3.14):
scripts/setup/setup_env.sh

# Or specify a specific interpreter from a Python manager (pyenv, brew, asdf, etc.):
PYTHON=python3.11 scripts/setup/setup_env.sh

scripts/datasets/download_dataset.sh --num-samples 100        # small slice first
scripts/asr/train.sh --epochs 5 --batch-size 8
scripts/asr/train.sh --architecture conformer --device mps # train Conformer CTC on Apple Silicon
scripts/asr/evaluate.sh                                  # uses the newest checkpoint
scripts/asr/export_model.sh --format torchscript
scripts/asr/serve_app.sh                                 # try it in the browser
python scripts/maintenance/summarize_runs.py
```

`evaluate.sh`, `transcribe.sh` and `export_model.sh` default to the newest
`outputs/asr/checkpoints/<run-id>/best_model.pt`. Pass `--model-path` to pin a
specific checkpoint.

## Command samples for all model types & devices

SesaML supports both custom end-to-end CTC models (`deepspeech`, `conformer`, `conformer-medium`) and pre-trained/fine-tuned HuggingFace `whisper` models, accelerating inference and training on Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu`.

### 1. Baseline DeepSpeech2 CTC (`deepspeech`)
Residual CNN stem + Bidirectional GRU with 2× time downsampling.

```bash
# Train on Apple Silicon GPU
scripts/asr/train.sh --architecture deepspeech --epochs 10 --batch-size 10 --device mps

# Evaluate trained checkpoint
scripts/asr/evaluate.sh --architecture deepspeech --device mps

# Transcribe audio clip
scripts/asr/transcribe.sh data/sample.wav --model-type deepspeech --architecture deepspeech --device mps

# Export model (TorchScript .pt, State Dict .pth, or ExecuTorch .pte)
scripts/asr/export_model.sh --architecture deepspeech --format torchscript
scripts/asr/export_model.sh --architecture deepspeech --format executorch
```

### 2. Conformer-Small CTC (`conformer`)
Convolution-augmented Transformer encoder (~10M parameters) with 4× time subsampling.

```bash
# Train Conformer-S on Apple Silicon GPU
scripts/asr/train.sh --architecture conformer --epochs 15 --batch-size 8 --device mps

# Train on CUDA GPU
scripts/asr/train.sh --architecture conformer --epochs 15 --batch-size 16 --device cuda

# Evaluate Conformer checkpoint
scripts/asr/evaluate.sh --architecture conformer --device mps

# Transcribe audio with noise reduction pre-processing
scripts/asr/transcribe.sh data/sample.wav --model-type deepspeech --architecture conformer --device mps --noise-reduction

# Export Conformer model
scripts/asr/export_model.sh --architecture conformer --format torchscript
```

### 3. Conformer-Medium CTC (`conformer-medium`)
Conformer-M encoder (~30M parameters) for larger multi-corpus training runs.

```bash
# Train Conformer-M with environment variable overrides
ARCHITECTURE=conformer-medium BATCH_SIZE=4 EPOCHS=30 scripts/asr/train.sh --device mps

# Evaluate Conformer-M
scripts/asr/evaluate.sh --architecture conformer-medium --device mps

# Export Conformer-M
scripts/asr/export_model.sh --architecture conformer-medium --format torchscript
```

### 4. Fine-Tuned Whisper (`whisper`)
HuggingFace sequence-to-sequence Transformer model. No default - pass `--whisper-repo` or set `MODEL_REPO_ID`; otherwise the repo's own trained models are used.

```bash
# Transcribe audio using default HuggingFace Whisper model
scripts/asr/transcribe.sh data/sample.wav --model-type whisper

# Transcribe audio using custom HuggingFace Whisper repository
scripts/asr/transcribe.sh data/sample.wav --model-type whisper --whisper-repo user/akan-whisper-model

# Serve Gradio web app with custom Whisper repository
MODEL_REPO_ID="openai/whisper-small" scripts/asr/serve_app.sh   # optional baseline
```

## Datasets

Three Akan corpora are wired up. Training combines the base splits of the first
two by default; the health corpus is opt-in (see [data/info.md](../data/info.md#the-health-corpus)).

| corpus | clips | audio | text column | access |
| --- | --- | --- | --- | --- |
| `ghanaopendata/twi-speech-text-multispeaker-16k` | 15,560 (`train`) | ~19 h | `text` | public |
| `Lagyamfi/akan_audio_processed` | 2,187 (`train`) + 259 (`test`) | ~3 h | `sentence` | gated (`auto`) |
| `ghananlpcommunity/twi-health-asr-gemini-500hrs` | 59,291 (`train`) | ~494 h | `transcription` | public, CC-BY-NC |

All are 16 kHz Twi with the same orthography, so they concatenate cleanly, and
the transcription column is auto-detected whatever it is called.

`Lagyamfi/akan_audio_processed` advertises 26,906 rows across 22 splits, but that
is 2,446 unique clips duplicated 11× — the base `train`/`test` plus ten augmented
copies (`train_Noise_Aug`, `test_Pitch_Aug`, …). The `*_Aug` splits are skipped by
default: they would over-weight those clips 11:1 against the larger corpus, and
the training pipeline already applies SpecAugment on the fly. Pass
`--include-augmented` to `download_dataset.sh` if you want them anyway.

Being gated means one-time setup: accept the terms on the dataset page while
logged in, then put a read token in `.env` as `HF_TOKEN`.

```bash
scripts/datasets/download_dataset.sh --dataset Lagyamfi/akan_audio_processed
scripts/asr/train.sh                                   # both corpora, validates on Lagyamfi:test
scripts/asr/train.sh --architecture conformer --device mps # Conformer CTC on Apple Silicon MPS
ARCHITECTURE=conformer-medium scripts/asr/train.sh     # Conformer-M model
HF_DATASETS="ghanaopendata/twi-speech-text-multispeaker-16k:train" scripts/asr/train.sh

# add the health corpus - note the smaller batch size for its 30s clips
HF_DATASETS="ghanaopendata/twi-speech-text-multispeaker-16k:train ghananlpcommunity/twi-health-asr-gemini-500hrs:train" \
  BATCH_SIZE=2 scripts/asr/train.sh
```

Every run probes each corpus before training and logs clip duration, estimated
hours and vocabulary coverage, warning when clips are long enough to threaten
GPU memory or when transcripts contain characters the CTC vocabulary drops.

## Configuration

`.env` (copied from `env.template`) is loaded automatically. Useful variables:

- `HF_TOKEN` — HuggingFace access token for gated datasets/models
- `MODEL_REPO_ID` — optional Whisper repository id. Empty by default; the app serves the trained models from `outputs/asr/registry/`
- `ARCHITECTURE` — Default CTC architecture (`deepspeech`, `conformer`, `conformer-medium`)

Scripts also honour a few overrides so you don't have to repeat flags:

```bash
ARCHITECTURE=conformer EPOCHS=30 BATCH_SIZE=16 scripts/asr/train.sh
HF_DATASET=some/other-corpus scripts/datasets/download_dataset.sh
MODEL_PATH=outputs/asr/checkpoints/train-20260809-101500/best_model.pt scripts/asr/evaluate.sh
```

Anything not consumed by a script is forwarded to the underlying CLI, so
`python -m src.main train --help` documents the full flag set.
