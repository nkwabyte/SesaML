#!/usr/bin/env bash
# Starts a DeepSpeech CTC training run.
#
# Logs, per-epoch metrics and predictions are written to outputs/runs/<run_id>/,
# weights to outputs/checkpoints/<run_id>/ (git-ignored).
#
# Usage:
#   scripts/train.sh                                    # both Akan corpora combined
#   scripts/train.sh --epochs 30 --batch-size 16
#   scripts/train.sh --csv-path data/manifest.csv       # audio-path + transcription columns
#   scripts/train.sh --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k
#   scripts/train.sh --resume outputs/checkpoints/<run_id>/last_model.pt --epochs 40
#
# There is no default corpus inside `python -m src.main train`: it used to fall
# back to data/corpus/verified_data.csv, which is a text-only translation table
# with no audio. This script is what supplies the real corpora.
#
# By default this trains on the base (non-augmented) splits of both corpora and
# validates on Lagyamfi's held-out test split. The augmented *_Aug splits are
# deliberately left out: they are copies of the same 2,446 clips, and the
# pipeline already applies SpecAugment on the fly.
#
# Environment overrides: ARCHITECTURE, HF_DATASETS, VAL_DATASET, EPOCHS, BATCH_SIZE,
# LEARNING_RATE, NUM_WORKERS, RESUME.
# Any extra flags are forwarded to `python -m src.main train`.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

ARCHITECTURE="${ARCHITECTURE:-deepspeech}"
# Space-separated list of repo[:split] specs.
HF_DATASETS="${HF_DATASETS:-ghanaopendata/twi-speech-text-multispeaker-16k:train Lagyamfi/akan_audio_processed:train}"
VAL_DATASET="${VAL_DATASET:-Lagyamfi/akan_audio_processed:test}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

ARGS=(train --architecture "${ARCHITECTURE}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LEARNING_RATE}")

# Only inject the default corpora when the caller did not pick a data source.
if [[ "$*" != *"--csv-path"* && "$*" != *"--hf-dataset"* ]]; then
  for spec in ${HF_DATASETS}; do
    ARGS+=(--hf-dataset "${spec}")
  done
fi

if [[ -n "${VAL_DATASET}" && "$*" != *"--val-"* && "$*" != *"--csv-path"* ]]; then
  ARGS+=(--val-dataset "${VAL_DATASET}")
fi

if [[ -n "${NUM_WORKERS:-}" && "$*" != *"--num-workers"* ]]; then
  ARGS+=(--num-workers "${NUM_WORKERS}")
fi

if [[ -n "${RESUME:-}" && "$*" != *"--resume"* ]]; then
  ARGS+=(--resume "${RESUME}")
fi

log "Training: epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LEARNING_RATE}"
"${PYTHON}" -m src.main "${ARGS[@]}" "$@"

RUN_DIR="$(latest_run train || true)"
[[ -n "${RUN_DIR}" ]] && log "Run artifacts: ${RUN_DIR}"
