#!/usr/bin/env bash
# Evaluates a trained checkpoint and stores loss/WER/CER plus per-sample
# predictions under outputs/runs/<run_id>/.
#
# Usage:
#   scripts/evaluate.sh
#   scripts/evaluate.sh --model-path outputs/checkpoints/train-20260809-120000/best_model.pt
#   scripts/evaluate.sh --csv-path data/corpus/verified_data.csv
#
# Environment overrides: HF_DATASET, SPLIT, MODEL_PATH.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

HF_DATASET="${HF_DATASET:-ghanaopendata/twi-speech-text-multispeaker-16k}"
SPLIT="${SPLIT:-train}"

ARGS=(evaluate --split "${SPLIT}")

if [[ "$*" != *"--csv-path"* && "$*" != *"--hf-dataset"* ]]; then
  ARGS+=(--hf-dataset "${HF_DATASET}")
fi

if [[ -n "${MODEL_PATH:-}" && "$*" != *"--model-path"* ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi

log "Evaluating on split '${SPLIT}'"
"${PYTHON}" -m src.main "${ARGS[@]}" "$@"

RUN_DIR="$(latest_run evaluate || true)"
[[ -n "${RUN_DIR}" ]] && log "Metrics and predictions: ${RUN_DIR}"
