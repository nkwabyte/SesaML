#!/usr/bin/env bash
# Starts a DeepSpeech CTC training run.
#
# Logs, per-epoch metrics and predictions are written to outputs/runs/<run_id>/,
# weights to outputs/checkpoints/<run_id>/ (git-ignored).
#
# Usage:
#   scripts/train.sh                                    # HF dataset, default hyperparameters
#   scripts/train.sh --epochs 30 --batch-size 16
#   scripts/train.sh --csv-path data/corpus/verified_data.csv
#   scripts/train.sh --val-split test                   # enables per-epoch WER/CER
#
# Environment overrides: HF_DATASET, EPOCHS, BATCH_SIZE, LEARNING_RATE.
# Any extra flags are forwarded to `python -m src.main train`.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

HF_DATASET="${HF_DATASET:-ghanaopendata/twi-speech-text-multispeaker-16k}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

ARGS=(train --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LEARNING_RATE}")

# Only inject the default HF dataset when the caller did not pick a data source.
if [[ "$*" != *"--csv-path"* && "$*" != *"--hf-dataset"* ]]; then
  ARGS+=(--hf-dataset "${HF_DATASET}")
fi

log "Training: epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LEARNING_RATE}"
"${PYTHON}" -m src.main "${ARGS[@]}" "$@"

RUN_DIR="$(latest_run train || true)"
[[ -n "${RUN_DIR}" ]] && log "Run artifacts: ${RUN_DIR}"
