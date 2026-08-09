# SesaML

Akan (Twi) automatic speech recognition — training, evaluation, export and a web
app, with every run reproducibly logged.

SesaML trains a DeepSpeech2-style CTC model from scratch on Akan speech corpora,
and can also serve a fine-tuned Whisper model for comparison. Every command
writes its config, metrics, predictions and logs into `outputs/`, so experiments
stay comparable long after the terminal scrollback is gone.

```bash
scripts/setup_env.sh                  # venv + dependencies
scripts/download_dataset.sh           # fetch a corpus into data/
scripts/train.sh --epochs 5           # train, logging to outputs/
scripts/evaluate.sh                   # WER / CER on held-out audio
scripts/serve_app.sh                  # try it in a browser
```

## Layout

| | |
| --- | --- |
| [src/](src/info.md) | the pipeline — data, models, training, inference |
| [scripts/](scripts/info.md) | entry points for every workflow |
| [app/](app/info.md) | Gradio web front-end |
| [data/](data/info.md) | checked-in corpus; downloaded audio stays untracked |
| [outputs/](outputs/info.md) | run logs, metrics and checkpoints |
| [tests/](tests/info.md) | fast dependency-light test suite |
| [docs/](docs/info.md) | architecture, diarization blueprint, and roadmap |
| `notebooks/` | exploratory analysis |

Each folder has its own `info.md` with the details.

## Getting started

Requires **Python ≥ 3.10, < 3.14** (Python **3.11** or **3.12** recommended; Python 3.14+ is unsupported by audio C-extension dependencies):

```bash
git clone https://github.com/nkwabyte/SesaML.git && cd SesaML

# Standard setup:
scripts/setup_env.sh

# Or specify a Python version manager binary (pyenv, brew, asdf, etc.):
PYTHON=python3.11 scripts/setup_env.sh
```

That creates `.venv`, installs `requirements.txt`, seeds `.env` from
`env.template` and creates the `outputs/` tree. Add a HuggingFace token to
`.env` if you want the gated corpus or private models:

```
HF_TOKEN=hf_...
MODEL_REPO_ID=CiBeDL/twi_trained_whisper
```

## Data

Three Akan corpora are wired up, all 16 kHz Twi with the same orthography:

| corpus | clips | audio | domain | text column | licence |
| --- | --- | --- | --- | --- | --- |
| `ghanaopendata/twi-speech-text-multispeaker-16k` | 15,560 | ~19 h | religious | `text` | open |
| `Lagyamfi/akan_audio_processed` | 2,446 | ~3 h | Bible | `sentence` | MIT (gated) |
| `ghananlpcommunity/twi-health-asr-gemini-500hrs` | 59,291 | **~494 h** | health / media | `transcription` | **CC-BY-NC-4.0** |

The transcription column is auto-detected, so corpora that name it `text`,
`sentence`, `transcription` or `transcript` all load without configuration.

```bash
# 1. Download all full Akan corpora (ghanaopendata, Lagyamfi, ghananlpcommunity 500h)
scripts/download_all_datasets.sh
# 2. Download a small smoke-test slice (e.g. 100 samples each across all 3 corpora)
scripts/download_all_datasets.sh --num-samples 100
# 3. Download with an explicit HuggingFace token for gated access
scripts/download_all_datasets.sh --token hf_...
```

Audio lands in `data/` and is never committed. The manifest describing each
download — row counts, columns, sample transcripts — is written to
`outputs/runs/<run_id>/dataset_manifest.json`, which **is** committed, so a
dataset's provenance survives in git even though its audio does not.

> `Lagyamfi/akan_audio_processed` advertises 26,906 rows across 22 splits, but
> that is 2,446 unique clips duplicated 11× — the base `train`/`test` plus ten
> augmentation copies. The downloader and trainer use the base splits only; see
> [scripts/info.md](scripts/info.md#datasets).

The health corpus is 25× more audio than the other two combined, and the only
one outside the religious domain — but it comes with three caveats, so it is
opt-in rather than a default. See [data/info.md](data/info.md#the-health-corpus).

## Training

```bash
scripts/train.sh                                    # both corpora combined
scripts/train.sh --epochs 30 --batch-size 16
EPOCHS=50 LEARNING_RATE=3e-4 scripts/train.sh
```

By default this trains on the base splits of both corpora and validates on
Lagyamfi's held-out `test` split, so per-epoch WER/CER is measured on audio the
model has not seen. Corpora are given as repeatable `repo[:split]` specs:

```bash
python -m src.main train \
  --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k:train \
  --hf-dataset Lagyamfi/akan_audio_processed:train \
  --val-dataset Lagyamfi/akan_audio_processed:test
```

To bring in the health corpus, add it as another spec — but drop the batch size,
because its clips are 30 s rather than ~4 s:

```bash
python -m src.main train --batch-size 2 \
  --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k:train \
  --hf-dataset ghananlpcommunity/twi-health-asr-gemini-500hrs:train \
  --val-dataset Lagyamfi/akan_audio_processed:test
```

Every run probes each corpus first and logs clip duration, estimated hours and
vocabulary coverage, warning when clips are long enough to threaten GPU memory
or when transcripts contain characters the CTC vocabulary drops.

### Staged plan

Concatenating all three is not a merge of equals — the health corpus is 95.6% of
the audio, so it dominates the gradient and the other two act as domain and
clean-label supplements. That is the right end state (22 h is not enough to train
this model from scratch; 494 h is), but it is worth reaching in two steps:

1. **Baseline on the two small corpora** — `scripts/train.sh`, ~22 h, `--batch-size 10`.
   Hours per run, so pipeline bugs are cheap to find.
2. **Add the health corpus** once the baseline is sound. Its 30 s clips force
   `--batch-size 2` on everything, and shuffled batches then pad the short clips
   to 2,400 frames — roughly 19% of compute lost to padding. Length-bucketed
   batching and per-corpus validation metrics fix that and are worth adding
   before committing to the long runs.

A CSV corpus works too — `--csv-path data/corpus/verified_data.csv`, with
`audio_path` and `text` columns.

Each run writes per-step and per-epoch metrics, keeps `best_model.pt` and
`last_model.pt`, and survives Ctrl-C with its partial weights and metrics intact.

## Evaluating and exporting

```bash
scripts/evaluate.sh                            # newest checkpoint, loss/WER/CER
scripts/evaluate.sh --model-path outputs/checkpoints/train-.../best_model.pt
scripts/export_model.sh --format torchscript   # .pt
scripts/export_model.sh --format state_dict    # .pth
scripts/export_model.sh --format executorch    # .pte, for on-device
python scripts/summarize_runs.py               # compare every run so far
```

Evaluation saves per-sample reference/hypothesis pairs alongside the headline
numbers, so you can read what the model actually got wrong. Exports record
format, size and SHA-256 in a tracked manifest even though the binary itself is
not committed.

## The app

```bash
scripts/serve_app.sh          # http://127.0.0.1:7860
scripts/serve_app.sh --share  # public tunnel
```

Upload or record a clip, pick `deepspeech` (your newest checkpoint) or `whisper`
(a HuggingFace repo), and get a transcript. Models load once per backend and are
cached. `app/app.py` exposes `demo` at module level, so it doubles as a
HuggingFace Space entrypoint. See [app/info.md](app/info.md).

## The models

Multiple CTC architectures share a uniform interface (`src/models/`):

- `deepspeech` (default) — DeepSpeech2-style residual CNN + bidirectional GRU ([src/models/deepspeech.py](src/models/deepspeech.py)), 2× time subsampling.
- `conformer` — Conformer-S encoder with attention, depthwise convolution and macaron feed-forwards ([src/models/conformer.py](src/models/conformer.py)), ~10M parameters, 4× time subsampling.
- `conformer-medium` — Conformer-M encoder, ~30M parameters, 4× time subsampling.

Pass `--architecture` to train, evaluate, or transcribe:

```bash
scripts/train.sh --architecture conformer
ARCHITECTURE=conformer-medium scripts/train.sh
```

Every checkpoint stores its `model_meta.json` in `outputs/checkpoints/<run_id>/`, so evaluation, export, and the web app auto-detect the architecture automatically.

Inputs are 128-bin mel spectrograms at 16 kHz with SpecAugment-style frequency and time masking during training.

The character vocabulary is 41 symbols — `a-z`, `0-9`, apostrophe, the Akan
characters `ɛ` and `ɔ`, space, and the CTC blank. Punctuation is dropped during
encoding; digits are kept because the health corpus writes dosages and dates
numerically. The three corpora sit at 96–98% coverage of this vocabulary.

Defaults live in [src/config.py](src/config.py) — audio, architecture, training
hyperparameters and paths, all in one place.

## Outputs

Every command opens a run directory:

```
outputs/
├── logs/<run_id>.log        full text log
├── runs/<run_id>/           config, metrics.jsonl, metrics.csv, predictions,
│                            summary.json — all tracked in git
├── runs/index.jsonl         one summary line per run
├── checkpoints/<run_id>/    weights        — not tracked
└── exports/<run_id>/        exported models — not tracked
```

Logs, metrics and configs are committed so experiments are reviewable in code
review; model binaries are not. Full contract in [outputs/info.md](outputs/info.md).

## Testing

```bash
scripts/run_tests.sh
```

Fast, synthetic, no data or checkpoint required. Covers text encoding, model
shapes, audio transforms, greedy decoding and WER/CER.

## Requirements

- **Python**: `≥ 3.10, < 3.14` (**3.11** or **3.12** recommended; 3.14+ is unsupported by `numba`/`llvmlite` dependencies).
- **PyTorch**: PyTorch 2.9, torchaudio (`requirements.txt` is fully pinned).
- **System Libraries**: `portaudio` (`brew install portaudio` on macOS, required for live microphone recording via `PyAudio`).
- **Compute**: Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), or CPU (`cpu`). The device is selected automatically based on availability.
