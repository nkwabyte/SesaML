#!/usr/bin/env bash
# Transcribes an audio file and stores the transcript under outputs/asr/runs/<run_id>/.
#
# Usage:
#   scripts/asr/transcribe.sh path/to/audio.wav
#   scripts/asr/transcribe.sh path/to/audio.wav --model-type whisper
#   scripts/asr/transcribe.sh path/to/audio.wav --noise-reduction
#
# Any extra flags are forwarded to `python -m src.main transcribe`.

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
bootstrap

[[ $# -ge 1 ]] || die "Usage: scripts/asr/transcribe.sh <audio-file> [extra flags]"

AUDIO="$1"
shift
[[ -f "${AUDIO}" ]] || die "Audio file not found: ${AUDIO}"

log "Transcribing ${AUDIO}"
"${PYTHON}" -m src.main transcribe --audio "${AUDIO}" "$@"

RUN_DIR="$(latest_run transcribe || true)"
[[ -n "${RUN_DIR}" ]] && log "Transcript saved to: ${RUN_DIR}"
