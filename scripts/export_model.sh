#!/usr/bin/env bash
# Exports a trained checkpoint to a deployable artifact.
#
# The binary lands in outputs/exports/<run_id>/ (git-ignored); the manifest
# describing it - format, size, SHA-256, source checkpoint - is stored in
# outputs/runs/<run_id>/export_manifest.json and IS tracked.
#
# Usage:
#   scripts/export_model.sh                                     # TorchScript .pt
#   scripts/export_model.sh --format state_dict                 # plain .pth
#   scripts/export_model.sh --format executorch                 # on-device .pte
#   scripts/export_model.sh --model-path outputs/checkpoints/train-.../best_model.pt

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

FORMAT="${FORMAT:-torchscript}"
ARGS=(export)

if [[ "$*" != *"--format"* ]]; then
  ARGS+=(--format "${FORMAT}")
fi

if [[ -n "${MODEL_PATH:-}" && "$*" != *"--model-path"* ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi

log "Exporting model (format=${FORMAT})"
"${PYTHON}" -m src.main "${ARGS[@]}" "$@"

RUN_DIR="$(latest_run export || true)"
[[ -n "${RUN_DIR}" ]] && log "Export manifest: ${RUN_DIR}"
