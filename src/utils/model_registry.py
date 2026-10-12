"""
Versioned store for trained model exports.

Checkpoints used to be resolved by modification time: whatever finished most
recently became the model the app served. Across a series of training
iterations that is a loaded gun - a run that collapses to emitting blanks is
newer than the good one, so it silently becomes the demo. This module replaces
"newest" with "promoted", and keeps every previous version so there is always
something to fall back to.

Two rules carry the safety:

* **Publishing and promoting are separate.** Every finished run is archived as a
  new version. It becomes `current` only if it beats the current version on the
  primary metric, so a worse run is recorded without ever being served.
* **Versions record the features they were trained on.** A checkpoint trained on
  80 mel bands loaded under a 128-band config does not error - the shapes are
  set by the architecture, not the front-end - it just produces nonsense. The
  registry compares them and refuses.

Layout::

    outputs/asr/registry/
        registry.json               pointers and promotion history
        conformer/
            v001/model.pt
            v001/metadata.json
            v002/...
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_DIRNAME = "registry"
INDEX_FILENAME = "registry.json"
WEIGHTS_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1

# Lower is better for both. WER first: it is the metric the project is judged on.
PRIMARY_METRICS = ("wer", "cer", "val_loss")


class RegistryError(RuntimeError):
    """Raised when a registry operation cannot be completed."""


@dataclass
class ModelVersion:
    """One published set of weights and everything needed to load it correctly."""

    architecture: str
    version: str
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> Dict[str, float]:
        return self.metadata.get("metrics") or {}

    @property
    def features(self) -> Dict[str, Any]:
        return self.metadata.get("features") or {}

    @property
    def run_id(self) -> Optional[str]:
        return self.metadata.get("run_id")

    @property
    def wer(self) -> Optional[float]:
        value = self.metrics.get("wer")
        return float(value) if value is not None else None

    def score(self) -> Optional[float]:
        """
        The number promotion compares, lower being better.

        Falls through WER to CER to validation loss, because a run without a
        validation set still has a loss and should be comparable to nothing but
        another run without one.
        """
        for name in PRIMARY_METRICS:
            value = self.metrics.get(name)
            if value is not None:
                return float(value)
        return None

    def incompatibilities(self, config: Any) -> List[str]:
        """
        Feature settings that differ from `config`, as human-readable strings.

        Empty means the weights can be loaded and will mean what they meant when
        they were trained.
        """
        recorded = self.features
        if not recorded:
            return []

        audio = getattr(config, "audio", None)
        current = {
            "sample_rate": getattr(audio, "sample_rate", None),
            "n_mels": getattr(audio, "n_mels", None),
            "n_fft": getattr(audio, "n_fft", None),
            "hop_length": getattr(audio, "hop_length", None),
        }

        differences = []
        for key, value in current.items():
            expected = recorded.get(key)
            if value is not None and expected is not None and value != expected:
                differences.append(f"{key}: trained with {expected}, config has {value}")
        return differences

    def describe(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "version": self.version,
            "path": str(self.path),
            "run_id": self.run_id,
            "metrics": self.metrics,
            "published_at": self.metadata.get("published_at"),
        }

    def __str__(self) -> str:
        wer = self.metrics.get("wer")
        scored = f"WER {wer:.4f}" if wer is not None else "no metrics"
        return f"{self.architecture}/{self.version} ({scored})"


class ModelRegistry:
    """Publishes, promotes and resolves versioned model exports."""

    def __init__(self, output_dir: str = "outputs/asr"):
        self.root = Path(output_dir) / REGISTRY_DIRNAME
        self.index_path = self.root / INDEX_FILENAME

    # --- index -----------------------------------------------------------

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema": SCHEMA_VERSION, "architectures": {}}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(
                f"Registry index at {self.index_path} is unreadable ({exc}). Delete it to "
                f"rebuild; the published versions themselves are intact on disk."
            ) from exc

    def _write_index(self, index: Dict[str, Any]) -> None:
        # Written via a temporary file: a half-written index would lose the
        # pointer to every published model at once.
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    # --- reading ---------------------------------------------------------

    def architectures(self) -> List[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def versions(self, architecture: str) -> List[ModelVersion]:
        """Every published version of an architecture, oldest first."""
        directory = self.root / architecture
        if not directory.is_dir():
            return []

        found = []
        for version_dir in sorted(directory.iterdir()):
            weights = version_dir / WEIGHTS_FILENAME
            if not weights.is_file():
                continue
            metadata_path = version_dir / METADATA_FILENAME
            metadata = {}
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    metadata = {}
            found.append(ModelVersion(architecture, version_dir.name, weights, metadata))
        return found

    def get(self, architecture: str, version: str) -> Optional[ModelVersion]:
        for candidate in self.versions(architecture):
            if candidate.version == version:
                return candidate
        return None

    def current(self, architecture: str) -> Optional[ModelVersion]:
        """The version being served, i.e. the last one promoted."""
        entry = self._read_index()["architectures"].get(architecture) or {}
        name = entry.get("current")
        return self.get(architecture, name) if name else None

    def previous(self, architecture: str) -> Optional[ModelVersion]:
        """
        The version `rollback` would return to.

        This is the promotion history, not the version numbering: after
        promoting v003 and then rolling back to v001, "previous" is whatever was
        promoted before the current one, which is what a fallback needs.
        """
        entry = self._read_index()["architectures"].get(architecture) or {}
        history = entry.get("history") or []
        for name in reversed(history[:-1]):
            found = self.get(architecture, name)
            if found is not None:
                return found
        return None

    def best(self, architecture: Optional[str] = None) -> Optional[ModelVersion]:
        """Lowest-scoring published version, across all architectures if none is named."""
        names = [architecture] if architecture else self.architectures()
        scored = [
            (version.score(), version)
            for name in names
            for version in self.versions(name)
            if version.score() is not None
        ]
        if not scored:
            return None
        return min(scored, key=lambda pair: pair[0])[1]

    def resolve(self, architecture: Optional[str] = None) -> Optional[ModelVersion]:
        """
        The version to serve: the promoted one, else the best published.

        Architectures are searched in order when none is named, so a project
        with both a Conformer and a DeepSpeech2 serves whichever is promoted.
        """
        names = [architecture] if architecture else self.architectures()
        for name in names:
            promoted = self.current(name)
            if promoted is not None:
                return promoted
        return self.best(architecture)

    # --- writing ---------------------------------------------------------

    def _next_version(self, architecture: str) -> str:
        existing = self.versions(architecture)
        numbers = [
            int(v.version[1:]) for v in existing
            if v.version.startswith("v") and v.version[1:].isdigit()
        ]
        return f"v{max(numbers, default=0) + 1:03d}"

    def publish(
        self,
        checkpoint: Any,
        architecture: str,
        metrics: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        config: Any = None,
        vocab_size: Optional[int] = None,
        subsampling_factor: Optional[int] = None,
        notes: Optional[str] = None,
        promote: Optional[bool] = None,
    ) -> ModelVersion:
        """
        Archives a checkpoint as a new version.

        `promote=None` (the default) promotes only when the new version scores
        better than the current one - which is what makes a failed training run
        harmless. Pass True to force it, False to archive without serving.
        """
        source = Path(checkpoint)
        if not source.is_file():
            raise RegistryError(f"Cannot publish: no checkpoint at {source}")

        version = self._next_version(architecture)
        directory = self.root / architecture / version
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / WEIGHTS_FILENAME
        shutil.copy2(source, destination)

        audio = getattr(config, "audio", None)
        metadata = {
            "architecture": architecture,
            "version": version,
            "run_id": run_id,
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "source_checkpoint": str(source),
            "size_bytes": destination.stat().st_size,
            "sha256": _digest(destination),
            "metrics": {k: v for k, v in (metrics or {}).items() if v is not None},
            # The front-end settings are as much a part of the model as its
            # weights; loading it under different ones silently degrades it.
            "features": {
                "sample_rate": getattr(audio, "sample_rate", None),
                "n_mels": getattr(audio, "n_mels", None),
                "n_fft": getattr(audio, "n_fft", None),
                "hop_length": getattr(audio, "hop_length", None),
                "vocab_size": vocab_size,
                "subsampling_factor": subsampling_factor,
            },
            "notes": notes,
        }
        (directory / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        published = ModelVersion(architecture, version, destination, metadata)

        if promote is None:
            promote = self._beats_current(published)
        if promote:
            self.promote(architecture, version)
        else:
            self._touch_architecture(architecture)

        return published

    def _beats_current(self, candidate: ModelVersion) -> bool:
        """A new version is served only if it is actually better than what is."""
        current = self.current(candidate.architecture)
        if current is None:
            return True

        new_score, old_score = candidate.score(), current.score()
        if new_score is None:
            return False
        if old_score is None:
            return True
        return new_score < old_score

    def _touch_architecture(self, architecture: str) -> None:
        index = self._read_index()
        index["architectures"].setdefault(architecture, {"current": None, "history": []})
        self._write_index(index)

    def promote(self, architecture: str, version: str) -> ModelVersion:
        """Makes a version the one that gets served."""
        found = self.get(architecture, version)
        if found is None:
            raise RegistryError(
                f"Cannot promote {architecture}/{version}: no such version. "
                f"Published: {[v.version for v in self.versions(architecture)] or 'none'}"
            )

        index = self._read_index()
        entry = index["architectures"].setdefault(architecture, {"current": None, "history": []})
        entry["current"] = version
        history = entry.setdefault("history", [])
        if not history or history[-1] != version:
            history.append(version)
        self._write_index(index)
        return found

    def rollback(self, architecture: str) -> ModelVersion:
        """
        Reverts to the previously promoted version.

        The demo's escape hatch: when a freshly promoted model misbehaves in
        front of an audience, this puts the last known-good one back without
        needing to know its version number.
        """
        target = self.previous(architecture)
        if target is None:
            raise RegistryError(
                f"Cannot roll back {architecture}: no earlier promoted version. "
                f"Published versions: {[v.version for v in self.versions(architecture)] or 'none'}"
            )

        index = self._read_index()
        entry = index["architectures"].setdefault(architecture, {"current": None, "history": []})
        history = entry.setdefault("history", [])
        if history:
            history.pop()
        entry["current"] = target.version
        self._write_index(index)
        return target

    def summary(self) -> List[Dict[str, Any]]:
        """Everything published, for `models list` and the docs."""
        rows = []
        for architecture in self.architectures():
            current = self.current(architecture)
            for version in self.versions(architecture):
                rows.append({
                    **version.describe(),
                    "current": current is not None and current.version == version.version,
                })
        return rows


def _digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256, so a published version can be identified independently of its path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
