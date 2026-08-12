#!/usr/bin/env bash
# Serves the Gradio app (transcription + speaker diarization) from app/app.py.
#
# Usage:
#   scripts/asr/serve_app.sh                     # http://127.0.0.1:7860
#   scripts/asr/serve_app.sh --port 8080
#   scripts/asr/serve_app.sh --share             # public tunnel
#   scripts/asr/serve_app.sh --host 0.0.0.0      # reachable on your network
#
# Prints which model version is about to be served before starting, because the
# app happily runs on random weights when nothing has been trained - it produces
# fluent-looking gibberish rather than an error, which is a bad thing to
# discover in front of an audience.
#
# Requests made in the app are logged to outputs/asr/runs/app-<timestamp>/.

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
bootstrap

"${PYTHON}" -c "import gradio" >/dev/null 2>&1 || die "gradio is not installed - run scripts/setup/setup_env.sh"

# Pre-flight: report the served model and the fallback available behind it.
"${PYTHON}" - <<'PY'
import warnings
warnings.filterwarnings("ignore")

from src.config import PipelineConfig
from src.utils.model_registry import ModelRegistry

config = PipelineConfig()
registry = ModelRegistry(config.paths.domain_dir)
version = registry.resolve()

if version is None:
    print("  WARNING: no published model version.")
    print("  The CTC backend will run on random weights and produce gibberish.")
    print("  Train one (scripts/asr/train.sh), publish an existing checkpoint")
    print("  (python -m src.main models publish ...), or use the Whisper backend.")
else:
    metrics = version.metrics
    scores = " ".join(
        f"{name.upper()}={metrics[name]:.3f}"
        for name in ("wer", "cer") if metrics.get(name) is not None
    )
    print(f"  Serving {version.architecture}/{version.version} {scores or '(no metrics)'}"
          f" from run {version.run_id or 'unknown'}")

    previous = registry.previous(version.architecture)
    if previous is not None:
        print(f"  Fallback available: {previous.architecture}/{previous.version}"
              f" (python -m src.main models rollback --architecture {version.architecture})")

    for problem in version.incompatibilities(config):
        print(f"  WARNING: feature mismatch - {problem}")
PY

log "Starting Gradio app (Ctrl-C to stop)"
exec "${PYTHON}" "${REPO_ROOT}/app/app.py" "$@"
