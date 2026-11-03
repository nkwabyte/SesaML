#!/usr/bin/env bash
# SESA: local Gradio page for the best Akan model (conformer-medium/v003).
#
#   scripts/asr/serve_sesa.sh
#   scripts/asr/serve_sesa.sh --port 7860
#
# Record or upload audio in the browser. Sample clips are on the page.

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
bootstrap

"${PYTHON}" -c "import gradio, torch, soundfile" >/dev/null 2>&1 \
  || die "gradio, torch, or soundfile is missing. Run scripts/setup/setup_env.sh"

test -f "${REPO_ROOT}/outputs/asr/registry/conformer-medium/v003/model.pt" \
  || die "conformer-medium/v003 weights are not on this machine"
test -f "${REPO_ROOT}/outputs/asr/lm/twi_char.json" \
  || die "character language model outputs/asr/lm/twi_char.json is missing"

log "SESA will serve conformer-medium/v003 with beam search and the character LM"
exec "${PYTHON}" "${REPO_ROOT}/app/sesa.py" "$@"
