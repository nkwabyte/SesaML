#!/usr/bin/env bash
# Downloads all supported Akan speech datasets into data/ at a go.
#
# Usage:
#   scripts/download_all_datasets.sh                         # Download all full corpora
#   scripts/download_all_datasets.sh --num-samples 100        # Download small smoke-test slice (100 samples each)
#   scripts/download_all_datasets.sh --token hf_...           # With explicit HuggingFace token

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

exec "${PYTHON}" "${SCRIPT_DIR}/download_all_datasets.py" "$@"
