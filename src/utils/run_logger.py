"""Run-scoped logging and artifact storage.

Every training, evaluation, export or transcription run gets its own directory
under ``outputs/`` so logs, metrics and predictions stay reproducible and
comparable across runs:

    outputs/
    ├── logs/<run_id>.log          full text log of the run
    ├── runs/<run_id>/             config, metrics and predictions (tracked)
    ├── runs/index.jsonl           one summary line per completed run
    ├── checkpoints/<run_id>/      training weights (NOT tracked)
    └── exports/<run_id>/          exported .pt/.pte/.pth models (NOT tracked)
"""

import csv
import json
import logging
import platform
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


SENSITIVE_PARAM_KEYS = {"token", "hf_token", "auth_token", "api_key", "secret", "password", "access_token"}

# Catches credentials that land in values rather than obviously-named keys - e.g. a token
# echoed inside an error message or a URL - which key-name matching alone would miss.
SECRET_VALUE_PATTERN = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"
)


def _scrub_secrets(value: Any) -> Any:
    """Replaces credential-shaped substrings inside a value with a redaction marker."""
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    return value


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Redacts sensitive values such as tokens and API keys from recorded run parameters."""
    sanitized: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_params(value)
        elif isinstance(key, str) and any(s in key.lower() for s in SENSITIVE_PARAM_KEYS) and value:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, (list, tuple)):
            sanitized[key] = type(value)(_scrub_secrets(item) for item in value)
        else:
            sanitized[key] = _scrub_secrets(value)
    return sanitized


def _make_stream_utf8(stream) -> None:
    """
    Best-effort switch of a text stream to UTF-8 with lossy fallback.

    Only meaningful on Windows, where the console encoding is cp1252 and any
    Twi transcript containing ɛ or ɔ raises on write.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def load_weights(path, map_location=None) -> Dict[str, Any]:
    """
    Reads model weights from either checkpoint layout.

    `speech_recognition_model.pt` is a plain state_dict, while `last_model.pt`
    and `best_model.pt` wrap one alongside optimizer and schedule state so a run
    can be resumed. Inference and export want only the weights, and should not
    fail because they were handed the resumable file.
    """
    import torch

    state = torch.load(str(path), map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state


def resolve_path(path) -> Path:
    """Resolves a path relative to the project root so runs launched from any
    working directory still write into the same ``outputs/`` tree."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _relative_to_root(path: Path) -> str:
    """Renders a path relative to the project root when possible, for portable records."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _json_default(obj: Any) -> str:
    """Fallback serializer for values that json cannot encode natively."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _git_revision() -> Dict[str, Optional[str]]:
    """Best-effort capture of the git commit/branch the run was launched from."""
    info: Dict[str, Optional[str]] = {"commit": None, "branch": None, "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "git": _git_revision(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        env["torch"] = None
    return env


class RunManager:
    """Owns the output directory, logger and metric files for a single run."""

    def __init__(
        self,
        kind: str = "train",
        config: Any = None,
        run_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        console: bool = True
    ):
        self.kind = kind
        self.config = config
        self.params = _sanitize_params(dict(params or {}))
        self.started_at = datetime.now()
        self.run_id = run_id or f"{kind}-{self.started_at.strftime(RUN_ID_FORMAT)}"

        base = output_dir or getattr(getattr(config, "paths", None), "output_dir", "outputs")
        self.output_dir = resolve_path(base)
        self.run_dir = self.output_dir / "runs" / self.run_id
        self.logs_dir = self.output_dir / "logs"
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.run_id
        self.export_dir = self.output_dir / "exports" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.logs_dir / f"{self.run_id}.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.metrics_csv_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.index_path = self.output_dir / "runs" / "index.jsonl"

        self._metric_fields: List[str] = []
        self._metric_rows: List[Dict[str, Any]] = []
        self._finished = False

        self.logger = self._build_logger(console)
        self.write_json("config.json", self.describe())
        self.logger.info("Run %s started (kind=%s)", self.run_id, self.kind)
        self.logger.info("Run directory: %s", self.run_dir)

    # ------------------------------------------------------------------ setup

    def _build_logger(self, console: bool) -> logging.Logger:
        logger = logging.getLogger(f"sesaml.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console:
            # Windows consoles default to cp1252, which cannot encode the Akan
            # vowels ɛ and ɔ - logging a single real transcript would raise
            # UnicodeEncodeError and take the training run down with it. Retarget
            # stdout at UTF-8 where possible, and fall back to replacing the
            # characters rather than letting a log line kill an hour of training.
            _make_stream_utf8(sys.stdout)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def describe(self) -> Dict[str, Any]:
        """Full provenance record written to ``config.json``."""
        config_dict = asdict(self.config) if is_dataclass(self.config) else self.config
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "params": self.params,
            "config": config_dict,
            "environment": _environment(),
        }

    # ---------------------------------------------------------------- metrics

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, stage: Optional[str] = None) -> Dict[str, Any]:
        """Appends one metric row to ``metrics.jsonl`` and ``metrics.csv``."""
        row: Dict[str, Any] = {
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(self.elapsed_seconds, 2),
        }
        if step is not None:
            row["step"] = step
        if stage is not None:
            row["stage"] = stage
        for key, value in metrics.items():
            row[key] = value.item() if hasattr(value, "item") else value

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")

        self._metric_rows.append(row)
        self._write_metrics_csv(row)
        return row

    def _write_metrics_csv(self, row: Dict[str, Any]) -> None:
        """Keeps a flat CSV alongside the JSONL, rewriting it when new columns appear."""
        new_fields = [key for key in row if key not in self._metric_fields]
        if new_fields and self._metric_fields:
            self._metric_fields.extend(new_fields)
            with self.metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._metric_fields)
                writer.writeheader()
                writer.writerows(self._metric_rows)
            return

        write_header = not self._metric_fields
        self._metric_fields.extend(new_fields)
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._metric_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @property
    def metric_rows(self) -> List[Dict[str, Any]]:
        return list(self._metric_rows)

    # -------------------------------------------------------------- artifacts

    def write_json(self, name: str, payload: Any) -> Path:
        """Writes a JSON artifact into the run directory and returns its path."""
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def log_predictions(
        self,
        predictions: Sequence[Dict[str, Any]],
        name: str = "predictions.json",
        preview: int = 3
    ) -> Path:
        """Stores reference/hypothesis pairs and logs a small preview."""
        path = self.write_json(name, list(predictions))
        for sample in list(predictions)[:preview]:
            self.logger.info(
                "sample | ref='%s' | hyp='%s'",
                sample.get("reference", ""),
                sample.get("hypothesis", "")
            )
        self.logger.info("Wrote %d predictions to %s", len(predictions), path)
        return path

    def ensure_checkpoint_dir(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir

    def ensure_export_dir(self) -> Path:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        return self.export_dir

    def checkpoint_path(self, filename: str) -> Path:
        return self.ensure_checkpoint_dir() / filename

    def export_path(self, filename: str) -> Path:
        return self.ensure_export_dir() / filename

    # ----------------------------------------------------------------- finish

    def finish(self, status: str = "completed", summary: Optional[Dict[str, Any]] = None) -> Path:
        """Writes ``summary.json`` and appends the run to ``outputs/runs/index.jsonl``."""
        if self._finished:
            return self.summary_path

        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(self.elapsed_seconds, 2),
            "params": self.params,
            "run_dir": _relative_to_root(self.run_dir),
            "log_file": _relative_to_root(self.log_path),
        }
        record.update(summary or {})
        # Re-sanitize after merging: a caller's summary (an error message quoting a
        # signed URL, say) can carry credentials the constructor never saw.
        record = _sanitize_params(record)
        self.write_json("summary.json", record)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")

        self.logger.info("Run %s %s in %.1fs", self.run_id, status, record["duration_sec"])
        self.logger.info("Summary: %s", self.summary_path)
        for handler in self.logger.handlers:
            handler.flush()
        self._finished = True
        return self.summary_path

    def __enter__(self) -> "RunManager":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            self.logger.exception("Run %s failed: %s", self.run_id, exc)
            self.finish(status="failed", summary={"error": f"{exc_type.__name__}: {exc}"})
        else:
            self.finish(status="completed")
        return False


CHECKPOINT_PREFERENCE = ("best_model.pt", "speech_recognition_model.pt", "last_model.pt")


def latest_checkpoint(output_dir: str = "outputs") -> Optional[Path]:
    """
    Newest checkpoint written by a training run, i.e. the most recent
    ``outputs/checkpoints/<run_id>/`` directory, preferring ``best_model.pt``.
    Falls back to loose files placed directly in ``outputs/checkpoints/``.
    Returns None when nothing has been trained yet.
    """
    checkpoints_root = resolve_path(output_dir) / "checkpoints"
    if not checkpoints_root.is_dir():
        return None

    run_dirs = sorted(
        (d for d in checkpoints_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    for run_dir in run_dirs:
        for name in CHECKPOINT_PREFERENCE:
            candidate = run_dir / name
            if candidate.is_file():
                return candidate

    loose = [f for f in checkpoints_root.glob("*.pt*") if f.is_file()]
    if loose:
        return max(loose, key=lambda f: f.stat().st_mtime)
    return None


def resolve_checkpoint(explicit: Optional[str], config: Any) -> str:
    """
    Picks the checkpoint to load: an explicit path wins, otherwise the newest
    run's weights, otherwise the configured default location.
    """
    if explicit:
        return explicit

    paths = getattr(config, "paths", None)
    discovered = latest_checkpoint(getattr(paths, "output_dir", "outputs"))
    if discovered is not None:
        return str(discovered)

    return str(resolve_path(getattr(paths, "models_dir", "outputs/checkpoints")) / config.training.model_name)


MODEL_META_FILENAME = "model_meta.json"


def save_model_meta(checkpoint_dir: Path, meta: Dict[str, Any]) -> Path:
    """
    Records which architecture produced the weights in a directory.

    Without this, loading a Conformer checkpoint into the default DeepSpeech2
    architecture fails with an opaque shape mismatch - or worse, silently
    succeeds if the shapes happen to line up.
    """
    path = Path(checkpoint_dir) / MODEL_META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=_json_default)
    return path


def load_model_meta(checkpoint_path: Any) -> Optional[Dict[str, Any]]:
    """Reads the architecture metadata sitting beside a checkpoint, if any."""
    if not checkpoint_path:
        return None
    meta_path = Path(checkpoint_path).parent / MODEL_META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_run_index(output_dir: str = "outputs") -> List[Dict[str, Any]]:
    """Reads every recorded run summary from ``outputs/runs/index.jsonl``."""
    index_path = resolve_path(output_dir) / "runs" / "index.jsonl"
    if not index_path.exists():
        return []
    runs = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs
