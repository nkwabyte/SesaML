import json
import os
from typing import Any, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import ConcatDataset, Dataset
import torchaudio

from ...utils.audio_io import audio_duration_bytes, load_audio_bytes
from ...utils.noise_reduction import reduce_audio_noise
from .dataset import MissingAudioError

# Corpora disagree on what they call the transcription column:
# ghanaopendata/twi-speech-text-multispeaker-16k uses "text",
# Lagyamfi/akan_audio_processed uses "sentence".
TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript")

DEFAULT_SPLIT = "train"

# Where scripts/datasets/download_dataset.py writes its save_to_disk copies.
DEFAULT_LOCAL_DIR = os.path.join("data", "datasets")


def parse_dataset_spec(spec: str, default_split: str = DEFAULT_SPLIT) -> Tuple[str, str]:
    """
    Splits a `repo/name:split` specification into its parts.
    The split is optional, so `repo/name` yields (`repo/name`, default_split).
    Repository ids contain `/` but never `:`, so the separator is unambiguous.
    """
    if ":" in spec:
        name, _, split = spec.rpartition(":")
        return name.strip(), (split.strip() or default_split)
    return spec.strip(), default_split


class HuggingFaceAkanDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace speech datasets (e.g.
    `ghanaopendata/twi-speech-text-multispeaker-16k` or
    `Lagyamfi/akan_audio_processed`).

    The transcription column is auto-detected, so corpora that name it
    `sentence` rather than `text` load without silently producing empty labels.
    """

    def __init__(
        self,
        dataset_name: str = "ghanaopendata/twi-speech-text-multispeaker-16k",
        split: str = DEFAULT_SPLIT,
        sample_rate: int = 16000,
        apply_noise_reduction: bool = False,
        hf_dataset: Optional[Any] = None,
        text_column: Optional[str] = None,
        token: Optional[str] = None,
        allow_missing_audio: bool = False,
        local_dir: str = DEFAULT_LOCAL_DIR,
        limit_rows: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.apply_noise_reduction = apply_noise_reduction
        self.allow_missing_audio = allow_missing_audio
        self.source = f"{dataset_name}:{split}"
        self.limit_rows = limit_rows

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            self.hf_dataset = self._load(dataset_name, split, token, local_dir)

        # Taking an evenly-spaced slice rather than the first N rows: corpora are
        # frequently grouped by speaker or session, so a head slice can be one
        # voice while a stride spans the whole set.
        if limit_rows and limit_rows < len(self.hf_dataset) and hasattr(self.hf_dataset, "select"):
            step = len(self.hf_dataset) / limit_rows
            self.hf_dataset = self.hf_dataset.select(
                [int(i * step) for i in range(limit_rows)]
            )
            self.source = f"{self.source} [{limit_rows} of {int(step * limit_rows)} rows]"

        self.text_column = text_column or self._detect_text_column()
        self._decodes_own_audio = self._disable_audio_decoding()

    @staticmethod
    def local_path(dataset_name: str, split: str, local_dir: str = DEFAULT_LOCAL_DIR) -> str:
        """
        Where `scripts/datasets/download_dataset.py` parked this corpus.

        It saves to `<local_dir>/<repo__name>/<split>`, replacing the repository
        separator, so training can find a corpus that is already on disk.
        """
        return os.path.join(local_dir, dataset_name.replace("/", "__"), split)

    def _load(self, dataset_name: str, split: str, token: Optional[str], local_dir: str):
        """
        Prefers the on-disk copy, falling back to the Hub.

        The download script saves corpora with `save_to_disk` and caches them
        under `data/hf_cache`, but `load_dataset` consults neither by default -
        so training re-downloaded hundreds of hours of audio that was already
        sitting in the project directory, and failed outright on a machine with
        no network or no access token.
        """
        from datasets import load_dataset, load_from_disk

        path = self.local_path(dataset_name, split, local_dir)
        if os.path.isdir(path):
            self.source = path
            return load_from_disk(path)

        resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        cache_dir = os.path.join("data", "hf_cache")
        self.source = f"{dataset_name}:{split} (hub)"
        return load_dataset(
            dataset_name,
            split=split,
            token=resolved_token,
            cache_dir=cache_dir if os.path.isdir(cache_dir) else None,
        )

    def _disable_audio_decoding(self) -> bool:
        """
        Hands back the encoded bytes instead of letting `datasets` decode them.

        Since datasets 4.x the Audio feature decodes through torchcodec, which
        needs an FFmpeg shared library it does not ship; where that is missing,
        *every* row access raises - including reads of the text column, because
        indexing decodes the whole row. The container bytes are all we need, and
        soundfile turns them into samples with no native dependency beyond the
        one already pinned. Returns True when the cast succeeded.
        """
        if "audio" not in (getattr(self.hf_dataset, "column_names", []) or []):
            return False
        # In-memory mocks (lists of dicts, as the tests use) have no cast_column
        # and are already decoded; __getitem__ handles that form directly.
        if not hasattr(self.hf_dataset, "cast_column"):
            return False

        from datasets import Audio

        # Deliberately not guarded: a cast that fails here means every later row
        # access raises deep inside the DataLoader, where the cause is far less
        # obvious than it is at construction time.
        self.hf_dataset = self.hf_dataset.cast_column("audio", Audio(decode=False))
        return True

    def _detect_text_column(self) -> str:
        """Finds the transcription column, failing loudly rather than training on empty labels."""
        columns = list(getattr(self.hf_dataset, "column_names", []) or [])
        if not columns and len(self.hf_dataset) > 0 and hasattr(self.hf_dataset[0], "keys"):
            columns = list(self.hf_dataset[0].keys())
        for candidate in TEXT_COLUMNS:
            if candidate in columns:
                return candidate
        raise ValueError(
            f"No transcription column found in '{self.dataset_name}' (split '{self.split}'). "
            f"Looked for {TEXT_COLUMNS}, dataset has {columns}. "
            f"Pass text_column=... explicitly."
        )

    @staticmethod
    def _audio_bytes(audio_item: dict) -> Optional[bytes]:
        """
        Encoded audio for a row, whether stored inline or as a path on disk.

        Corpora saved with `save_to_disk` embed the bytes; ones loaded from a
        local cache directory may instead point at a file.
        """
        data = audio_item.get("bytes")
        if data:
            return data
        path = audio_item.get("path")
        if path and audio_item.get("array") is None and os.path.exists(path):
            with open(path, "rb") as handle:
                return handle.read()
        return None

    def texts(self) -> List[str]:
        """
        Every transcript, without decoding any audio.

        Reading a row decodes the whole row, so the audio column is dropped
        first - the same trick probe() uses. Needed to work out which samples
        CTC can align, which depends only on text length and clip duration.
        """
        try:
            drop = [c for c in self.hf_dataset.column_names if c != self.text_column]
            source = self.hf_dataset.remove_columns(drop) if drop else self.hf_dataset
            return [str(row[self.text_column] or "") for row in source]
        except Exception:
            return [str(self.hf_dataset[i].get(self.text_column, "")) for i in range(len(self))]

    def durations(self, cache_dir: str = "data/durations") -> List[float]:
        """
        Per-row clip length in seconds, for length-bucketed batching.

        Three sources, cheapest first: a `duration` column when the corpus ships
        one, a cached JSON from a previous pass, or a sweep of the audio headers.
        The sweep costs a full read of every row's bytes, so it is cached - on
        the 59k-row health corpus it is minutes, and paying it once per corpus
        instead of once per run is the difference between bucketing being worth
        it and not.
        """
        for column in ("duration", "length", "seconds"):
            if column in (getattr(self.hf_dataset, "column_names", []) or []):
                return [float(v) for v in self.hf_dataset[column]]

        suffix = f"__n{self.limit_rows}" if self.limit_rows else ""
        key = f"{self.dataset_name.replace('/', '__')}__{self.split}{suffix}.json"
        cache_path = os.path.join(cache_dir, key)
        total = len(self)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            if len(cached) == total:
                return [float(v) for v in cached]

        values = []
        for idx in range(total):
            audio_item = self.hf_dataset[idx].get("audio", {})
            encoded = self._audio_bytes(audio_item) if isinstance(audio_item, dict) else None
            if encoded is not None:
                try:
                    values.append(audio_duration_bytes(encoded))
                    continue
                except Exception:
                    pass
            array = audio_item.get("array") if isinstance(audio_item, dict) else None
            rate = (audio_item.get("sampling_rate") if isinstance(audio_item, dict) else None) or self.sample_rate
            values.append(len(array) / rate if array is not None else 0.0)

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump([round(v, 4) for v in values], handle)
        return values

    def describe(self) -> dict:
        """Provenance record for the run log."""
        return {
            "dataset": self.dataset_name,
            "source": self.source,
            "split": self.split,
            "rows": len(self),
            "text_column": self.text_column,
            "decoder": "soundfile" if self._decodes_own_audio else "datasets",
        }

    def probe(self, n_samples: int = 16, n_text_samples: int = 256, text_transform: Optional[Any] = None) -> dict:
        """
        Samples the corpus to report clip duration and vocabulary coverage.

        Corpora differ wildly in clip length - 30s clips produce ~7x the mel
        frames of a 4s clip and will exhaust GPU memory at the same batch size.
        Characters outside the CTC vocabulary are dropped silently during
        encoding, so a corpus full of digits trains against labels that do not
        match its audio. Both are cheap to measure up front and expensive to
        discover mid-run.
        """
        from .text_transform import TextTransform

        total = len(self)
        if total == 0:
            return {**self.describe(), "sampled": 0}

        step = max(1, total // max(1, n_samples))
        indices = list(range(0, total, step))[:n_samples]

        durations = []
        missing = 0
        for idx in indices:
            # Reading the header is the cheap part; only n_samples rows here.
            audio_item = self.hf_dataset[idx].get("audio", {})
            if not isinstance(audio_item, dict):
                missing += 1
                continue
            encoded = self._audio_bytes(audio_item)
            if encoded is not None:
                try:
                    durations.append(audio_duration_bytes(encoded))
                except Exception:
                    missing += 1
                continue
            array = audio_item.get("array")
            rate = audio_item.get("sampling_rate") or self.sample_rate
            if array is not None and rate:
                durations.append(len(array) / rate)
            else:
                missing += 1

        # Text needs no decoding, so sample far more widely than the audio. A
        # stride that lands on a periodic pattern would otherwise miss whole
        # categories of character (digits in particular).
        text_step = max(1, total // max(1, n_text_samples))
        text_indices = list(range(0, total, text_step))[:n_text_samples]
        try:
            drop = [c for c in self.hf_dataset.column_names if c != self.text_column]
            texts = [str(row[self.text_column]) for row in
                     self.hf_dataset.select(text_indices).remove_columns(drop)]
        except Exception:
            texts = [str(self.hf_dataset[i].get(self.text_column, "")) for i in text_indices]

        transform = text_transform or TextTransform()
        kept = chars = 0
        empty = 0
        dropped: dict = {}
        for text in texts:
            if not transform.text_to_int(text):
                empty += 1
            for ch in text.lower():
                chars += 1
                if ("<SPACE>" if ch == " " else ch) in transform.char_map:
                    kept += 1
                else:
                    dropped[ch] = dropped.get(ch, 0) + 1

        report = {
            **self.describe(),
            "sampled": len(indices),
            "audio_missing": missing,
            "text_sampled": len(texts),
            "empty_labels": empty,
            "coverage": round(kept / chars, 4) if chars else None,
            "dropped_chars": sorted(dropped.items(), key=lambda kv: -kv[1])[:10],
            "digits_dropped": sum(v for k, v in dropped.items() if k.isdigit()),
        }
        if durations:
            durations.sort()
            report.update({
                "duration_min_sec": round(durations[0], 2),
                "duration_median_sec": round(durations[len(durations) // 2], 2),
                "duration_max_sec": round(durations[-1], 2),
                "est_hours": round(sum(durations) / len(durations) * total / 3600, 1),
            })
        return report

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        item = self.hf_dataset[idx]
        audio_item = item.get("audio", {})
        text = str(item.get(self.text_column, ""))

        if isinstance(audio_item, dict):
            array = audio_item.get("array")
            sr = audio_item.get("sampling_rate", self.sample_rate)
            path = audio_item.get("path", f"sample_{idx}")
            encoded = self._audio_bytes(audio_item)

            if encoded is not None:
                waveform, sr = load_audio_bytes(encoded, target_sample_rate=self.sample_rate)
            elif array is not None:
                waveform = torch.tensor(array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
            elif self.allow_missing_audio:
                waveform = torch.randn(1, self.sample_rate)
            else:
                raise MissingAudioError(
                    f"Row {idx} of '{self.dataset_name}' (split '{self.split}') has an "
                    f"audio column with no decoded array. The corpus may have been "
                    f"downloaded without its audio files, or the column needs casting "
                    f"with datasets.Audio(). Pass allow_missing_audio=True to train on "
                    f"synthetic noise instead (smoke tests only)."
                )
        elif self.allow_missing_audio:
            waveform = torch.randn(1, self.sample_rate)
            sr = self.sample_rate
            path = f"sample_{idx}"
        else:
            raise MissingAudioError(
                f"Row {idx} of '{self.dataset_name}' (split '{self.split}') has no 'audio' "
                f"column (found {list(item.keys())}). A speech corpus is required for "
                f"training; a text-only corpus cannot supply the acoustic input."
            )

        if sr != self.sample_rate and waveform.ndim > 0:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.apply_noise_reduction:
            waveform = reduce_audio_noise(waveform, sample_rate=self.sample_rate)

        return waveform, self.sample_rate, text, str(path)


def load_hf_datasets(
    specs: Sequence[str],
    sample_rate: int = 16000,
    default_split: str = DEFAULT_SPLIT,
    apply_noise_reduction: bool = False,
    token: Optional[str] = None,
    allow_missing_audio: bool = False,
    local_dir: str = DEFAULT_LOCAL_DIR,
    limit_rows: Optional[int] = None
) -> List[HuggingFaceAkanDataset]:
    """Builds one HuggingFaceAkanDataset per `repo/name:split` specification."""
    datasets = []
    for spec in specs:
        name, split = parse_dataset_spec(spec, default_split)
        datasets.append(
            HuggingFaceAkanDataset(
                dataset_name=name,
                split=split,
                sample_rate=sample_rate,
                apply_noise_reduction=apply_noise_reduction,
                token=token,
                allow_missing_audio=allow_missing_audio,
                local_dir=local_dir,
                limit_rows=limit_rows
            )
        )
    return datasets


def combine_datasets(datasets: Sequence[Dataset]) -> Dataset:
    """
    Concatenates several corpora into a single training set.
    A single dataset is returned unchanged so the common case stays simple.
    """
    if not datasets:
        raise ValueError("No datasets to combine.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
