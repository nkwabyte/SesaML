#!/usr/bin/env bash
# Transcribes an audio file and stores the transcript under outputs/runs/<run_id>/.
#
# Usage:
#   scripts/transcribe.sh path/to/audio.wav
#   scripts/transcribe.sh path/to/audio.wav --model-type whisper
#   scripts/transcribe.sh path/to/audio.wav --noise-reduction
#
# Any extra flags are forwarded to `python -m src.main transcribe`.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

[[ $# -ge 1 ]] || die "Usage: scripts/transcribe.sh <audio-file> [extra flags]"

AUDIO="$1"
shift
[[ -f "${AUDIO}" ]] || die "Audio file not found: ${AUDIO}"

log "Transcribing ${AUDIO}"
"${PYTHON}" -m src.main transcribe --audio "${AUDIO}" "$@"

RUN_DIR="$(latest_run transcribe || true)"
[[ -n "${RUN_DIR}" ]] && log "Transcript saved to: ${RUN_DIR}"
