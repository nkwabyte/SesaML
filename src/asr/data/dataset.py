import os
from typing import List, Tuple, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .text_transform import TextTransform
from ...utils.audio_io import audio_duration, load_audio
from ...utils.noise_reduction import reduce_audio_noise

# CSV corpora disagree on their column names the same way HuggingFace ones do.
# Accepting the common spellings avoids forcing every corpus through a rename.
AUDIO_COLUMNS = ("audio_path", "audio", "path", "file", "filename", "wav", "filepath")
TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "label")


class MissingAudioError(FileNotFoundError):
    """Raised when a row points at audio that is not on disk."""


class AkanAudioDataset(Dataset):
    """
    Dataset class for Akan speech-to-text audio and transcriptions.
    Can be initialized from a CSV file, a pandas DataFrame, or a directory of WAV files.

    Both columns are auto-detected and their absence is a hard error. A CSV
    without them used to fall through to empty transcripts and synthetic noise,
    which trains happily to a CTC loss of exactly 0.0 (the model just learns to
    emit blanks) and looks like textbook convergence in the logs.
    """

    def __init__(
        self,
        csv_file: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        audio_dir: Optional[str] = None,
        audio_col: Optional[str] = None,
        text_col: Optional[str] = None,
        sample_rate: int = 16000,
        apply_noise_reduction: bool = False,
        allow_missing_audio: bool = False
    ):
        self.sample_rate = sample_rate
        self.apply_noise_reduction = apply_noise_reduction
        self.audio_dir = audio_dir
        self.allow_missing_audio = allow_missing_audio
        self.source = csv_file or "<dataframe>"

        if dataframe is not None:
            self.data = dataframe.copy()
        elif csv_file is not None and os.path.exists(csv_file):
            # keep_default_na=False stops pandas turning a blank transcript cell
            # into NaN, which str() would then render as the literal "nan" and
            # feed to CTC as a three-character target.
            self.data = pd.read_csv(csv_file, keep_default_na=False, dtype=str)
        elif csv_file is not None:
            raise FileNotFoundError(f"Dataset CSV not found: {csv_file}")
        else:
            raise ValueError("AkanAudioDataset requires either csv_file= or dataframe=.")

        columns = list(self.data.columns)
        self.audio_col = self._resolve_column(audio_col, AUDIO_COLUMNS, columns, "audio path")
        self.text_col = self._resolve_column(text_col, TEXT_COLUMNS, columns, "transcription")
        # A DataFrame passed in directly has not been through the read_csv guard
        # above, so normalise its NaNs the same way.
        self.data[self.text_col] = self.data[self.text_col].fillna("").astype(str)
        self.data[self.audio_col] = self.data[self.audio_col].fillna("").astype(str)

    def _resolve_column(
        self,
        explicit: Optional[str],
        candidates: Tuple[str, ...],
        columns: List[str],
        role: str
    ) -> str:
        """Finds a column by name or by convention, failing loudly rather than yielding blanks."""
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(
                    f"{role} column '{explicit}' not found in {self.source}. "
                    f"Available columns: {columns}"
                )
            return explicit

        lowered = {str(c).strip().lower(): c for c in columns}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]

        raise ValueError(
            f"No {role} column found in {self.source}. Looked for {candidates}, "
            f"but the file has {columns}. This is usually a text-only corpus being "
            f"passed where an audio manifest is expected - a CSV for training needs "
            f"one column of audio file paths and one of transcriptions. "
            f"Pass audio_col=/text_col= explicitly to override."
        )

    def describe(self) -> dict:
        """Provenance record for the run log, mirroring HuggingFaceAkanDataset.describe()."""
        return {
            "dataset": self.source,
            "rows": len(self),
            "audio_column": self.audio_col,
            "text_column": self.text_col,
        }

    def resolve_path(self, audio_path: str) -> str:
        if self.audio_dir and not os.path.isabs(audio_path):
            return os.path.join(self.audio_dir, audio_path)
        return audio_path

    def texts(self) -> List[str]:
        """Every transcript, for the CTC feasibility check."""
        return [str(v) for v in self.data[self.text_col].tolist()]

    def durations(self) -> List[float]:
        """
        Per-row clip length in seconds, for length-bucketed batching.

        Headers only, so this does not decode the audio; rows whose file is
        missing report 0.0 and simply sort to the front of their pool.
        """
        values = []
        for idx in range(len(self)):
            path = self.resolve_path(str(self.data.iloc[idx][self.audio_col]))
            try:
                values.append(audio_duration(path))
            except Exception:
                values.append(0.0)
        return values

    def probe(self, n_samples: int = 16, n_text_samples: int = 256, text_transform=None) -> dict:
        """
        Samples the corpus for missing audio and vocabulary coverage before training.

        The audio check is what catches a manifest whose paths are all stale or
        relative to a different machine - cheap here, an hour wasted otherwise.
        """
        total = len(self)
        if total == 0:
            return {**self.describe(), "sampled": 0}

        step = max(1, total // max(1, n_samples))
        indices = list(range(0, total, step))[:n_samples]

        durations = []
        missing = 0
        for idx in indices:
            full_path = self.resolve_path(str(self.data.iloc[idx][self.audio_col]))
            if not os.path.exists(full_path):
                missing += 1
                continue
            try:
                durations.append(audio_duration(full_path))
            except Exception:
                # Present on disk but undecodable is the same problem for training.
                missing += 1

        text_step = max(1, total // max(1, n_text_samples))
        text_indices = list(range(0, total, text_step))[:n_text_samples]
        texts = [str(self.data.iloc[i][self.text_col]) for i in text_indices]

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
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.data.iloc[idx]
        audio_path = str(row[self.audio_col])
        text = str(row[self.text_col])
        full_path = self.resolve_path(audio_path)

        if os.path.exists(full_path):
            waveform, _ = load_audio(full_path, target_sample_rate=self.sample_rate)
        elif self.allow_missing_audio:
            # Opt-in only, for smoke tests that exercise shapes without real audio.
            waveform = torch.randn(1, self.sample_rate)
        else:
            raise MissingAudioError(
                f"Audio file not found for row {idx} of {self.source}: {full_path!r} "
                f"(column '{self.audio_col}'). Pass audio_dir= if the manifest stores "
                f"relative paths, or allow_missing_audio=True to substitute noise."
            )

        if self.apply_noise_reduction:
            waveform = reduce_audio_noise(waveform, sample_rate=self.sample_rate)

        return waveform, self.sample_rate, text, full_path


class AudioCollator:
    """
    Picklable collate function binding the text and audio transforms.

    DataLoader workers start with `spawn` on macOS and Windows, which cannot
    pickle a collate function defined as a local closure - it fails the moment
    num_workers > 0. A module-level callable survives the pickling.
    """

    def __init__(self, text_transform: TextTransform, audio_transforms: nn.Module, stride: int = 2):
        self.text_transform = text_transform
        self.audio_transforms = audio_transforms
        self.stride = stride

    def __call__(self, batch: List[Tuple[torch.Tensor, int, str, str]]):
        return data_processing(batch, self.text_transform, self.audio_transforms, stride=self.stride)


class EmptyLabelError(ValueError):
    """Raised when a whole batch encodes to zero-length CTC targets."""


def data_processing(
    batch: List[Tuple[torch.Tensor, int, str, str]],
    text_transform: TextTransform,
    audio_transforms: nn.Module,
    stride: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to prepare batches for PyTorch model and CTCLoss.
    Pads variable-length audio spectrograms and text label sequences.
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, sr, text, _ in batch:
        # Compute MelSpectrogram: shape (channels, n_mels, time) -> squeeze(0) -> transpose to (time, n_mels)
        spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        label_seq = torch.Tensor(text_transform.text_to_int(text))
        labels.append(label_seq)

        # Input length downsampled by CNN stride
        input_lengths.append(spec.shape[0] // stride)
        label_lengths.append(len(label_seq))

    # Pad spectrograms: (batch, time, n_mels) -> unsqueeze(1) -> transpose to (batch, channel=1, n_mels, time)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    if not any(length > 0 for length in label_lengths):
        # CTC treats a zero-length target as "emit blanks", whose loss the model
        # drives to exactly 0.0. Training would appear to converge perfectly
        # while learning nothing, so refuse the batch instead.
        raise EmptyLabelError(
            f"All {len(batch)} transcripts in this batch encode to empty label "
            f"sequences. The text column is likely blank, or every character in "
            f"it falls outside the CTC vocabulary. Sample: "
            f"{[text for _, _, text, _ in batch[:3]]!r}"
        )

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(label_lengths, dtype=torch.long)
