#!/usr/bin/env bash
# Downloads a HuggingFace speech dataset into data/ and records a manifest under outputs/.
#
# Usage:
#   scripts/download_dataset.sh
#   scripts/download_dataset.sh --dataset ghanaopendata/twi-speech-text-multispeaker-16k --split train
#   scripts/download_dataset.sh --num-samples 50           # small smoke-test slice
#
# All flags are forwarded to scripts/download_dataset.py (see --help).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

log "Downloading dataset (this can take a while on first run)"
exec "${PYTHON}" "${SCRIPT_DIR}/download_dataset.py" "$@"
