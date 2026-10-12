#!/usr/bin/env bash
# Downloads all supported Akan speech datasets into data/ at a go.
#
# Usage:
#   scripts/datasets/download_all_datasets.sh                         # Download all full corpora
#   scripts/datasets/download_all_datasets.sh --num-samples 100        # Download small smoke-test slice (100 samples each)
#   scripts/datasets/download_all_datasets.sh --token hf_...           # With explicit HuggingFace token

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
bootstrap

exec "${PYTHON}" "${SCRIPTS_DIR}/datasets/download_all_datasets.py" "$@"
