# Scripts

Entry points for the common SesaML workflows. Every script can be run from any
directory — they resolve the repository root themselves, load `.env`, and
activate `.venv` when it exists.

```bash
chmod +x scripts/*.sh      # once, after cloning
```

| Script | What it does |
| --- | --- |
| `setup_env.sh` | Creates `.venv`, installs `requirements.txt`, seeds `.env` and the `outputs/` tree. |
| `download_dataset.sh` | Downloads a HuggingFace speech dataset into `data/` and writes a manifest to `outputs/`. |
| `train.sh` | Starts a training run; logs, metrics and checkpoints go to `outputs/`. |
| `evaluate.sh` | Scores a checkpoint (loss/WER/CER) and saves per-sample predictions. |
| `transcribe.sh` | Transcribes one audio file and stores the transcript. |
| `export_model.sh` | Exports a checkpoint to `.pt` / `.pth` / `.pte` plus a tracked manifest. |
| `serve_app.sh` | Serves the Gradio web app from `app/app.py`. |
| `run_tests.sh` | Runs the test suite (pytest, falling back to `tests/run_all_tests.py`). |
| `summarize_runs.py` | Prints a table of all recorded runs from `outputs/runs/index.jsonl`. |
| `common.sh` | Shared bash helpers; sourced by the others, not run directly. |

## Typical flow

```bash
# Standard setup (uses Python >= 3.10, < 3.14):
scripts/setup_env.sh

# Or specify a specific interpreter from a Python manager (pyenv, brew, asdf, etc.):
PYTHON=python3.11 scripts/setup_env.sh

scripts/download_dataset.sh --num-samples 100        # small slice first
scripts/train.sh --epochs 5 --batch-size 8
scripts/train.sh --architecture conformer --device mps # train Conformer CTC on Apple Silicon
scripts/evaluate.sh                                  # uses the newest checkpoint
scripts/export_model.sh --format torchscript
scripts/serve_app.sh                                 # try it in the browser
python scripts/summarize_runs.py
```

`evaluate.sh`, `transcribe.sh` and `export_model.sh` default to the newest
`outputs/checkpoints/<run-id>/best_model.pt`. Pass `--model-path` to pin a
specific checkpoint.

## Command samples for all model types & devices

SesaML supports both custom end-to-end CTC models (`deepspeech`, `conformer`, `conformer-medium`) and pre-trained/fine-tuned HuggingFace `whisper` models, accelerating inference and training on Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu`.

### 1. Baseline DeepSpeech2 CTC (`deepspeech`)
Residual CNN stem + Bidirectional GRU with 2× time downsampling.

```bash
# Train on Apple Silicon GPU
scripts/train.sh --architecture deepspeech --epochs 10 --batch-size 10 --device mps

# Evaluate trained checkpoint
scripts/evaluate.sh --architecture deepspeech --device mps

# Transcribe audio clip
scripts/transcribe.sh data/sample.wav --model-type deepspeech --architecture deepspeech --device mps

# Export model (TorchScript .pt, State Dict .pth, or ExecuTorch .pte)
scripts/export_model.sh --architecture deepspeech --format torchscript
scripts/export_model.sh --architecture deepspeech --format executorch
```

### 2. Conformer-Small CTC (`conformer`)
Convolution-augmented Transformer encoder (~10M parameters) with 4× time subsampling.

```bash
# Train Conformer-S on Apple Silicon GPU
scripts/train.sh --architecture conformer --epochs 15 --batch-size 8 --device mps

# Train on CUDA GPU
scripts/train.sh --architecture conformer --epochs 15 --batch-size 16 --device cuda

# Evaluate Conformer checkpoint
scripts/evaluate.sh --architecture conformer --device mps

# Transcribe audio with noise reduction pre-processing
scripts/transcribe.sh data/sample.wav --model-type deepspeech --architecture conformer --device mps --noise-reduction

# Export Conformer model
scripts/export_model.sh --architecture conformer --format torchscript
```

### 3. Conformer-Medium CTC (`conformer-medium`)
Conformer-M encoder (~30M parameters) for larger multi-corpus training runs.

```bash
# Train Conformer-M with environment variable overrides
ARCHITECTURE=conformer-medium BATCH_SIZE=4 EPOCHS=30 scripts/train.sh --device mps

# Evaluate Conformer-M
scripts/evaluate.sh --architecture conformer-medium --device mps

# Export Conformer-M
scripts/export_model.sh --architecture conformer-medium --format torchscript
```

### 4. Fine-Tuned Whisper (`whisper`)
HuggingFace sequence-to-sequence Transformer model (default: `CiBeDL/twi_trained_whisper`).

```bash
# Transcribe audio using default HuggingFace Whisper model
scripts/transcribe.sh data/sample.wav --model-type whisper

# Transcribe audio using custom HuggingFace Whisper repository
scripts/transcribe.sh data/sample.wav --model-type whisper --whisper-repo user/akan-whisper-model

# Serve Gradio web app with custom Whisper repository
MODEL_REPO_ID="CiBeDL/twi_trained_whisper" scripts/serve_app.sh
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
scripts/download_dataset.sh --dataset Lagyamfi/akan_audio_processed
scripts/train.sh                                   # both corpora, validates on Lagyamfi:test
scripts/train.sh --architecture conformer --device mps # Conformer CTC on Apple Silicon MPS
ARCHITECTURE=conformer-medium scripts/train.sh     # Conformer-M model
HF_DATASETS="ghanaopendata/twi-speech-text-multispeaker-16k:train" scripts/train.sh

# add the health corpus - note the smaller batch size for its 30s clips
HF_DATASETS="ghanaopendata/twi-speech-text-multispeaker-16k:train ghananlpcommunity/twi-health-asr-gemini-500hrs:train" \
  BATCH_SIZE=2 scripts/train.sh
```

Every run probes each corpus before training and logs clip duration, estimated
hours and vocabulary coverage, warning when clips are long enough to threaten
GPU memory or when transcripts contain characters the CTC vocabulary drops.

## Configuration

`.env` (copied from `env.template`) is loaded automatically. Useful variables:

- `HF_TOKEN` — HuggingFace access token for gated datasets/models
- `MODEL_REPO_ID` — Whisper repository id used by the Gradio app in `app/app.py`
- `ARCHITECTURE` — Default CTC architecture (`deepspeech`, `conformer`, `conformer-medium`)

Scripts also honour a few overrides so you don't have to repeat flags:

```bash
ARCHITECTURE=conformer EPOCHS=30 BATCH_SIZE=16 scripts/train.sh
HF_DATASET=some/other-corpus scripts/download_dataset.sh
MODEL_PATH=outputs/checkpoints/train-20260809-101500/best_model.pt scripts/evaluate.sh
```

Anything not consumed by a script is forwarded to the underlying CLI, so
`python -m src.main train --help` documents the full flag set.
