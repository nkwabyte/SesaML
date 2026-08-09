#!/usr/bin/env bash
# Downloads all supported Akan speech datasets into data/ at a go.
#
# Usage:
#   scripts/download_all_datasets.sh                         # Download all full corpora
#   scripts/download_all_datasets.sh --num-samples 100        # Download small smoke-test slice (100 samples each)
#   scripts/download_all_datasets.sh --token hf_...           # With explicit HuggingFace token

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

NUM_SAMPLES=""
TOKEN_ARG=""
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --num-samples=*)
      NUM_SAMPLES="${1#*=}"
      shift
      ;;
    --token)
      TOKEN_ARG="$2"
      shift 2
      ;;
    --token=*)
      TOKEN_ARG="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: scripts/download_all_datasets.sh [--num-samples N] [--token HF_TOKEN]"
      echo "Downloads ghanaopendata, Lagyamfi, and ghananlpcommunity Twi speech datasets into data/."
      exit 0
      ;;
    *)
      EXTRA_FLAGS+=("$1")
      shift
      ;;
  esac
done

CORPORA=(
  "ghanaopendata/twi-speech-text-multispeaker-16k"
  "Lagyamfi/akan_audio_processed"
  "ghananlpcommunity/twi-health-asr-gemini-500hrs"
)

log "Starting bulk download of all ${#CORPORA[@]} Akan speech datasets into data/..."

for repo in "${CORPORA[@]}"; do
  log "========================================================"
  log "Downloading dataset: ${repo}"
  log "========================================================"

  CMD=("${PYTHON}" "${SCRIPT_DIR}/download_dataset.py" "--dataset" "${repo}")
  if [[ -n "${NUM_SAMPLES}" ]]; then
    CMD+=("--num-samples" "${NUM_SAMPLES}")
  fi
  if [[ -n "${TOKEN_ARG}" ]]; then
    CMD+=("--token" "${TOKEN_ARG}")
  fi
  if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_FLAGS[@]}")
  fi

  "${CMD[@]}" || warn "Dataset '${repo}' encountered an error or warning (continuing with remaining datasets)."
done

log "========================================================"
log "Bulk dataset download sequence finished!"
