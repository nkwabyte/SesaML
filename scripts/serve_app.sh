#!/usr/bin/env bash
# Serves the Gradio transcription app from app/app.py.
#
# Usage:
#   scripts/serve_app.sh                     # http://127.0.0.1:7860
#   scripts/serve_app.sh --port 8080
#   scripts/serve_app.sh --share             # public tunnel
#   scripts/serve_app.sh --host 0.0.0.0      # reachable on your network
#
# Transcriptions made in the app are logged to outputs/runs/app-<timestamp>/.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

"${PYTHON}" -c "import gradio" >/dev/null 2>&1 || die "gradio is not installed - run scripts/setup_env.sh"

log "Starting Gradio app (Ctrl-C to stop)"
exec "${PYTHON}" "${REPO_ROOT}/app/app.py" "$@"
