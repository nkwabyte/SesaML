# SesaML

Akan (Twi) automatic speech recognition — training, evaluation, speaker
diarization, versioned model exports and a web app, with every run reproducibly
logged.

SesaML trains Conformer and DeepSpeech2 CTC models from scratch on Akan speech
corpora, and can also serve a fine-tuned Whisper model for comparison. Every
command writes its config, metrics, predictions and logs into `outputs/`, so
experiments stay comparable long after the terminal scrollback is gone.

**Current model:** Conformer-CTC, **WER 0.3898 / CER 0.1277** on a held-out split —
trained from scratch on 12.3 hours of Akan audio, fine-tuned with a warm restart,
and decoded with beam search plus a Twi character n-gram language model. The
language model alone accounts for a **17.2% relative WER reduction**, with no
retraining (0.4705 greedy → 0.3898).

> Corpus reference — *Wɔbɛtumi akɔ dan a ɛtoa wɔn so no ne ne yɔnko…*
> Model output — *wɔbɛtumi akɔdan a ɛtoa wɔn so no ne onyankoa…*

Every measured number is in [docs/asr/results.md](docs/asr/results.md), including a
negative result worth reading: adding 100 hours of a 500-hour corpus made the
model strictly worse.

```bash
scripts/setup/setup_env.sh                 # venv + dependencies
scripts/datasets/download_all_datasets.sh  # fetch the corpora into data/
scripts/asr/train.sh --epochs 80           # train, logging to outputs/
scripts/asr/evaluate.sh                    # WER / CER on held-out audio
python scripts/lm/build_lm.py              # Twi n-gram LM used for decoding
python -m src.main models list             # which model version is served
scripts/asr/serve_app.sh                   # try it in a browser
```

## Layout

| | |
| --- | --- |
| [src/](src/info.md) | the pipeline — data, models, training, inference |
| [scripts/](scripts/info.md) | entry points for every workflow |
| [app/](app/info.md) | Gradio web front-end |
| [data/](data/info.md) | checked-in corpus; downloaded audio stays untracked |
| [outputs/](outputs/info.md) | run logs, metrics, checkpoints and the model registry |
| [tests/](tests/info.md) | fast dependency-light test suite (187 tests) |
| [docs/](docs/info.md) | measured results, architecture, diarization, and roadmap |
| `notebooks/` | exploratory analysis |

Each folder has its own `info.md` with the details.

## Getting started

Requires **Python ≥ 3.10, < 3.14** (Python **3.11** or **3.12** recommended; Python 3.14+ is unsupported by audio C-extension dependencies):

```bash
git clone https://github.com/nkwabyte/SesaML.git && cd SesaML

# Standard setup:
scripts/setup/setup_env.sh

# Or specify a Python version manager binary (pyenv, brew, asdf, etc.):
PYTHON=python3.11 scripts/setup/setup_env.sh
```

That creates `.venv`, installs `requirements.txt`, seeds `.env` from
`env.template` and creates the `outputs/` tree. Add a HuggingFace token to
`.env` if you want the gated corpus or private models:

```
HF_TOKEN=hf_...
```

`MODEL_REPO_ID` is optional and empty by default — the app serves the models
trained in this repo, from `outputs/asr/registry/`. Set it only to add a Whisper
baseline alongside them for comparison.

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
scripts/datasets/download_all_datasets.sh
# 2. Download a small smoke-test slice (e.g. 100 samples each across all 3 corpora)
scripts/datasets/download_all_datasets.sh --num-samples 100
# 3. Download with an explicit HuggingFace token for gated access
scripts/datasets/download_all_datasets.sh --token hf_...
```

Audio lands in `data/` and is never committed. The manifest describing each
download — row counts, columns, sample transcripts — is written to
`outputs/asr/runs/<run_id>/dataset_manifest.json`, which **is** committed, so a
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
scripts/asr/train.sh                                    # both corpora combined
scripts/asr/train.sh --epochs 30 --batch-size 16
EPOCHS=50 LEARNING_RATE=3e-4 scripts/asr/train.sh
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

Every run probes each corpus first and logs clip duration, estimated hours and
vocabulary coverage, and **aborts** when a corpus has no usable audio or no
usable transcripts rather than training on it.

A CSV corpus works too — `--csv-path path/to/manifest.csv`, with an audio-path
column and a transcription column (several spellings of each are recognised).
There is no default corpus: a data source must be named.

Each run writes per-step and per-epoch metrics, keeps `best_model.pt` and
`last_model.pt`, survives Ctrl-C with its partial weights intact, and can be
continued with `--resume outputs/asr/checkpoints/<run_id>/last_model.pt`.

### On the 500-hour health corpus

The obvious plan — throw all three corpora at it, since 494 h must beat 12 h —
**was tried and failed**. A 113-hour run (100 h of it health data) plateaued at
WER 1.0 while the 12.3-hour run reached 0.51. Nine times the data produced a
strictly worse model.

The health corpus is 59,291 clips of *exactly* 30 s with Gemini-generated
transcripts, and CTC cannot align ~400 characters over 750 encoder frames from a
random initialisation. Before spending GPU time on it again: fine-tune from the
working checkpoint rather than from scratch, and segment those 30 s clips into
utterances. Numbers in [docs/asr/results.md](docs/asr/results.md), plan in
[docs/roadmap_next_steps.md](docs/roadmap_next_steps.md).

Batches are length-bucketed by default, which is what makes mixing corpora
affordable at all: pairing a 0.04 s clip with a 30 s one pads nearly everything
to 30 s, and measured padding efficiency goes from **20% to 97%** with bucketing
on. Disable with `--no-bucket-batches`.

## Model versions

Every finished run is archived in the model registry at `outputs/asr/registry/`, and
the app, CLI and evaluation serve the **promoted** version — not whichever file
is newest.

```bash
python -m src.main models list                                # what exists, what is served
python -m src.main models rollback --architecture conformer   # back to the last good one
python -m src.main models promote --architecture conformer --version v002
```

Publishing and promoting are separate on purpose. A run is promoted only if it
beats the served version's WER, so a collapsed iteration is archived for
comparison without ever reaching the demo:

```
Published conformer/v002 (WER 1.0000)
Not promoted; conformer/v001 (WER 0.5100) remains current.
```

Rollback follows promotion history rather than version numbers, so it returns to
the last version actually served. The app exposes the same switch as a dropdown.
Each version also records the audio front-end it was trained on — loading 80-mel
weights under a 128-mel config degrades silently instead of raising, so the
registry compares and warns.

`registry.json` and each version's `metadata.json` are tracked in git; the `.pt`
weights are not.

## Speaker diarization

Turns a recording of several people into a speaker-attributed transcript —
*who spoke when* joined to *what was said*.

```bash
python -m src.main diarize --audio recording.wav --backend auto --num-speakers 2
```

```
[00:00.0 - 00:07.6] SPEAKER_00: ɔbɛtumi akɔdan a ɛtɔa wɔn nsono ne onyankon adi...
[00:08.3 - 00:08.8] SPEAKER_01: ucirie
[00:13.6 - 00:21.0] SPEAKER_02: awɔde asɔfodie nsafoɔ a no m ɛbeɛa na me yɛ atum afiri sooaa
```

Three backends, picked with `--backend`:

| Backend | Speaker model | Needs a licence? |
| --- | --- | --- |
| `pyannote` | `speaker-diarization-community-1` | Yes — three gated repos |
| `ecapa` | SpeechBrain ECAPA-TDNN | No — **default** |
| `spectral` | MFCC statistics | No, and no downloads at all |

`auto` takes the best that loads, so a missing licence downgrades the demo
instead of ending it. Pass `--num-speakers` when the count is known; automatic
speaker counting is the weak point of the embedding backends. Full write-up in
[docs/asr/speaker_diarization.md](docs/asr/speaker_diarization.md).

## Evaluating and exporting

```bash
scripts/asr/evaluate.sh                            # served model version, loss/WER/CER
scripts/asr/evaluate.sh --model-path outputs/asr/registry/conformer/v001/model.pt
scripts/asr/export_model.sh --format torchscript   # .pt
scripts/asr/export_model.sh --format state_dict    # .pth
scripts/asr/export_model.sh --format executorch    # .pte, for on-device
python scripts/maintenance/summarize_runs.py               # compare every run so far
```

Evaluation saves per-sample reference/hypothesis pairs alongside the headline
numbers, so you can read what the model actually got wrong. Exports record
format, size and SHA-256 in a tracked manifest even though the binary itself is
not committed.

## The app

```bash
scripts/asr/serve_app.sh          # http://127.0.0.1:7860
scripts/asr/serve_app.sh --share  # public tunnel
```

Two tabs: **Transcribe** for a single block of text, and **Speaker Diarization**
for a colour-coded, speaker-attributed transcript with a turn-taking timeline.

Pick `ctc` (the promoted registry version) or `whisper` (a HuggingFace repo). The
badge names the exact version and its WER, and a **Model version** dropdown
switches which one is served — the live fallback if a newly promoted model
misbehaves mid-demo. Models load once and are cached. Both `app.py` and
`app/app.py` expose `demo` at module level, so either doubles as a HuggingFace
Space entrypoint. See [app/info.md](app/info.md).

## The models

Multiple CTC architectures share a uniform interface (`src/asr/models/`):

- `deepspeech` (default) — DeepSpeech2-style residual CNN + bidirectional GRU ([src/asr/models/deepspeech.py](src/asr/models/deepspeech.py)), 2× time subsampling.
- `conformer` — Conformer-S encoder with attention, depthwise convolution and macaron feed-forwards ([src/asr/models/conformer.py](src/asr/models/conformer.py)), 8.4M parameters, 4× time subsampling. **This is the trained model: WER 0.510, CER 0.166.**
- `conformer-medium` — Conformer-M encoder, ~27M parameters, 4× time subsampling.

Pass `--architecture` to train, evaluate, or transcribe:

```bash
scripts/asr/train.sh --architecture conformer
ARCHITECTURE=conformer-medium scripts/asr/train.sh
```

Every checkpoint stores its architecture metadata beside the weights, so evaluation, export and the web app detect the architecture automatically — in a run directory as `model_meta.json`, in a registry version as `metadata.json`.

Inputs are **80-band log-mel spectrograms** at 16 kHz (25 ms window, 10 ms hop), normalized per utterance, with SpecAugment applied afterwards so masks fill with the feature mean. 80 rather than 128 because 128 mel filters over a 400-point FFT leaves four of them empty.

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
│                            summary.json
├── checkpoints/<run_id>/    weights, incl. resumable state — not tracked
├── registry/                versioned model exports; metadata tracked,
│                            weights not
└── exports/<run_id>/        exported models — not tracked
```

Run artifacts are machine-generated and accumulate quickly, so they are not
committed; use `python scripts/maintenance/summarize_runs.py` to compare runs locally. The
registry is the exception — its `registry.json` and per-version `metadata.json`
*are* tracked, because they record which model was served when and at what WER.
Full contract in [outputs/info.md](outputs/info.md).

Training leaves three checkpoint files per run and the resumable one is ~3× the
weights, so a few iterations is gigabytes. `scripts/maintenance/clean_outputs.py` prunes what
is no longer in use:

```bash
python scripts/maintenance/clean_outputs.py            # dry run: what would go
python scripts/maintenance/clean_outputs.py --apply    # delete it
```

It never touches `registry/`, protects the runs that produced published versions
(read from the registry, not hard-coded), and removes a run's weights only when
they are byte-identical to a published version — so `last_model.pt`, which
carries the optimizer state needed to resume, always survives. Run directories
and logs are kept for real training runs and dropped only for smoke tests: a
failed run's weights are worthless, but its metrics are the evidence for why it
failed.

## Testing

```bash
scripts/run_tests.sh
```

187 tests, fast and synthetic — no data, checkpoint or network required. Covers
text encoding, model shapes, audio transforms, greedy decoding, WER/CER, corpus
guards, ragged batches, length bucketing, speaker diarization, the model
registry and the outputs cleaner.

## Requirements

- **Python**: `≥ 3.10, < 3.14` (**3.11** or **3.12** recommended; 3.14+ is unsupported by `numba`/`llvmlite` dependencies).
- **PyTorch**: PyTorch 2.9, torchaudio (`requirements.txt` is fully pinned).
- **System Libraries**: `portaudio` (`brew install portaudio` on macOS, required for live microphone recording via `PyAudio`).
- **Compute**: Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), or CPU (`cpu`). The device is selected automatically based on availability.
