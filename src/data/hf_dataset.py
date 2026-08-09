import os
from typing import Any, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import ConcatDataset, Dataset
import torchaudio

from ..utils.noise_reduction import reduce_audio_noise

# Corpora disagree on what they call the transcription column:
# ghanaopendata/twi-speech-text-multispeaker-16k uses "text",
# Lagyamfi/akan_audio_processed uses "sentence".
TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript")

DEFAULT_SPLIT = "train"


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
        token: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.apply_noise_reduction = apply_noise_reduction

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            from datasets import load_dataset
            resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            self.hf_dataset = load_dataset(dataset_name, split=split, token=resolved_token)

        self.text_column = text_column or self._detect_text_column()

    def _detect_text_column(self) -> str:
        """Finds the transcription column, failing loudly rather than training on empty labels."""
        columns = list(getattr(self.hf_dataset, "column_names", []) or [])
        for candidate in TEXT_COLUMNS:
            if candidate in columns:
                return candidate
        raise ValueError(
            f"No transcription column found in '{self.dataset_name}' (split '{self.split}'). "
            f"Looked for {TEXT_COLUMNS}, dataset has {columns}. "
            f"Pass text_column=... explicitly."
        )

    def describe(self) -> dict:
        """Provenance record for the run log."""
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "rows": len(self),
            "text_column": self.text_column,
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
        for idx in indices:
            # Decoding audio is the expensive part, so only n_samples rows here.
            audio_item = self.hf_dataset[idx].get("audio", {})
            if isinstance(audio_item, dict):
                array = audio_item.get("array")
                rate = audio_item.get("sampling_rate") or self.sample_rate
                if array is not None and rate:
                    durations.append(len(array) / rate)

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
        dropped: dict = {}
        for text in texts:
            for ch in text.lower():
                chars += 1
                if ("<SPACE>" if ch == " " else ch) in transform.char_map:
                    kept += 1
                else:
                    dropped[ch] = dropped.get(ch, 0) + 1

        report = {
            **self.describe(),
            "sampled": len(indices),
            "text_sampled": len(texts),
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

            if array is not None:
                waveform = torch.tensor(array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                waveform = torch.randn(1, self.sample_rate)
        else:
            waveform = torch.randn(1, self.sample_rate)
            sr = self.sample_rate
            path = f"sample_{idx}"

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
    token: Optional[str] = None
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
                token=token
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
