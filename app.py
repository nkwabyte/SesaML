"""
HuggingFace Spaces entry point.

Spaces looks for `app.py` at the repository root and expects a module-level
`demo`, but the real interface lives in `app/app.py` so that the CLI and the
Space run exactly the same code. This file only re-exports it.

It is loaded by file path rather than imported as `app.app`: this module is
itself named `app`, and a top-level module shadows the same-named package
directory, so a plain import would find this file again instead of the package.

Previously this file was a separate Whisper-only demo that duplicated the app
and could not see the models trained in this repo. Everything it did - and
speaker diarization besides - is now in the shared interface.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("sesaml_app", ROOT / "app" / "app.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# What Spaces (and `python app.py`) look for.
demo = _module.demo
build_demo = _module.build_demo

if __name__ == "__main__":
    _module.main()
