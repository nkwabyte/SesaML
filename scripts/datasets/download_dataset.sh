#!/usr/bin/env bash
# Downloads a HuggingFace speech dataset into data/ and records a manifest under outputs/.
#
# Usage:
#   scripts/datasets/download_dataset.sh
#   scripts/datasets/download_dataset.sh --dataset ghanaopendata/twi-speech-text-multispeaker-16k --split train
#   scripts/datasets/download_dataset.sh --num-samples 50           # small smoke-test slice
#
# All flags are forwarded to scripts/datasets/download_dataset.py (see --help).

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
bootstrap

log "Downloading dataset (this can take a while on first run)"
exec "${PYTHON}" "${SCRIPTS_DIR}/datasets/download_dataset.py" "$@"
