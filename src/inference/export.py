"""Model export helpers.

Exported artifacts (`.pt`, `.pth`, `.pte`) are written to
`outputs/exports/<run_id>/` and are deliberately excluded from version control;
the manifest describing them is stored with the run and IS tracked.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..config import PipelineConfig

EXPORT_FORMATS = ("torchscript", "state_dict", "executorch")


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of an exported artifact, so the tracked manifest identifies the untracked binary."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def example_input(config: PipelineConfig, time_steps: int = 400) -> torch.Tensor:
    """Dummy spectrogram batch matching the model's expected (batch, 1, n_mels, time) input."""
    return torch.randn(1, 1, config.audio.n_mels, time_steps)


def export_state_dict(model: nn.Module, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination


def export_torchscript(model: nn.Module, destination: Path, sample: torch.Tensor) -> Path:
    """Scripts the model, falling back to tracing when scripting is unsupported."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, sample, strict=False)
    scripted.save(str(destination))
    return destination


def export_executorch(model: nn.Module, destination: Path, sample: torch.Tensor) -> Path:
    """Exports an ExecuTorch `.pte` binary for on-device inference."""
    try:
        from executorch.exir import to_edge
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ExecuTorch is not installed. Install it with `pip install executorch` "
            "to export .pte models."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    exported = torch.export.export(model, (sample,))
    program = to_edge(exported).to_executorch()
    with destination.open("wb") as handle:
        handle.write(program.buffer)
    return destination


def export_model(
    model: nn.Module,
    export_dir: Path,
    export_format: str = "torchscript",
    config: Optional[PipelineConfig] = None,
    basename: str = "speech_recognition_model",
    time_steps: int = 400
) -> Dict[str, Any]:
    """
    Exports `model` in the requested format and returns a manifest entry
    describing the artifact (path, size, checksum).
    """
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format '{export_format}'. Expected one of {EXPORT_FORMATS}.")

    config = config or PipelineConfig()
    export_dir = Path(export_dir)
    sample = example_input(config, time_steps=time_steps)
    model = model.to("cpu").eval()

    if export_format == "state_dict":
        path = export_state_dict(model, export_dir / f"{basename}.pth")
    elif export_format == "torchscript":
        path = export_torchscript(model, export_dir / f"{basename}.pt", sample)
    else:
        path = export_executorch(model, export_dir / f"{basename}.pte", sample)

    return {
        "format": export_format,
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "input_shape": list(sample.shape),
        "n_mels": config.audio.n_mels,
        "sample_rate": config.audio.sample_rate,
    }
