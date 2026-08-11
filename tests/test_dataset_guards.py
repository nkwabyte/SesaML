"""
Regression tests for the corpus guards.

A 30-epoch run once trained to a CTC loss of exactly 0.0 because a text-only
translation CSV was passed where an audio manifest was expected: the missing
columns fell through to empty transcripts, the missing audio fell through to
synthetic noise, and zero-length CTC targets are minimised by emitting blanks.
Every assertion here covers one link in that chain.
"""

import csv
import os
import random
import struct
import wave

import pandas as pd
import pytest
import torch

from src.config import PipelineConfig
from src.data.dataset import (
    AkanAudioDataset,
    EmptyLabelError,
    MissingAudioError,
    data_processing,
)
from src.data.text_transform import TextTransform
from src.main import valid_transforms_for


def write_wav(path, seconds=1.0, sample_rate=16000):
    random.seed(0)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = int(seconds * sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", random.randint(-2000, 2000)) for _ in range(frames)))
    return str(path)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def clips(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return [write_wav(audio_dir / f"clip{i}.wav") for i in range(4)]


def test_text_only_csv_is_rejected(tmp_path):
    """The exact failure: a translation corpus with no audio or text column."""
    path = write_csv(
        tmp_path / "translations.csv",
        ["English", "Akuapem Twi"],
        [["Good morning.", "Maakye"], ["How are you?", "Wo ho te sɛn"]],
    )
    with pytest.raises(ValueError, match="No audio path column"):
        AkanAudioDataset(csv_file=path)


def test_missing_text_column_is_rejected(tmp_path, clips):
    path = write_csv(tmp_path / "no_text.csv", ["audio_path"], [[c] for c in clips])
    with pytest.raises(ValueError, match="No transcription column"):
        AkanAudioDataset(csv_file=path)


def test_alternate_column_names_are_detected(tmp_path, clips):
    """Corpora spell these columns half a dozen ways; all should load."""
    path = write_csv(tmp_path / "alt.csv", ["path", "sentence"], [[c, "maakye"] for c in clips])
    dataset = AkanAudioDataset(csv_file=path)
    assert dataset.audio_col == "path"
    assert dataset.text_col == "sentence"
    assert len(dataset) == len(clips)


def test_explicit_columns_override_detection(tmp_path, clips):
    path = write_csv(tmp_path / "two.csv", ["wav", "text", "other"], [[c, "maakye", "x"] for c in clips])
    dataset = AkanAudioDataset(csv_file=path, audio_col="wav", text_col="other")
    assert dataset.text_col == "other"


def test_unknown_explicit_column_is_rejected(tmp_path, clips):
    path = write_csv(tmp_path / "one.csv", ["audio_path", "text"], [[c, "maakye"] for c in clips])
    with pytest.raises(ValueError, match="not found"):
        AkanAudioDataset(csv_file=path, text_col="nope")


def test_missing_audio_raises_instead_of_returning_noise(tmp_path):
    """The silent torch.randn fallback is what let a run train on pure noise."""
    path = write_csv(
        tmp_path / "stale.csv",
        ["audio_path", "text"],
        [["/nonexistent/clip.wav", "maakye"]],
    )
    dataset = AkanAudioDataset(csv_file=path)
    with pytest.raises(MissingAudioError, match="Audio file not found"):
        dataset[0]


def test_missing_audio_fallback_is_opt_in(tmp_path):
    path = write_csv(tmp_path / "stale.csv", ["audio_path", "text"], [["/nonexistent/clip.wav", "maakye"]])
    dataset = AkanAudioDataset(csv_file=path, allow_missing_audio=True)
    waveform, sample_rate, text, _ = dataset[0]
    assert waveform.shape == (1, sample_rate)
    assert text == "maakye"


def test_real_audio_loads_at_the_configured_sample_rate(tmp_path, clips):
    path = write_csv(tmp_path / "good.csv", ["audio_path", "text"], [[c, "maakye"] for c in clips])
    dataset = AkanAudioDataset(csv_file=path, sample_rate=16000)
    waveform, sample_rate, text, resolved = dataset[0]
    assert sample_rate == 16000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 16000
    assert os.path.exists(resolved)


def test_blank_cells_do_not_become_the_string_nan(tmp_path, clips):
    """pandas reads an empty cell as NaN, whose str() is 'nan' - a 3-char target."""
    path = write_csv(tmp_path / "blank.csv", ["audio_path", "text"], [[clips[0], ""]])
    dataset = AkanAudioDataset(csv_file=path)
    _, _, text, _ = dataset[0]
    assert text == ""


def test_dataframe_input_also_normalises_nan(clips):
    frame = pd.DataFrame({"audio_path": [clips[0]], "text": [float("nan")]})
    dataset = AkanAudioDataset(dataframe=frame)
    _, _, text, _ = dataset[0]
    assert text == ""


def test_probe_reports_missing_audio_and_empty_labels(tmp_path, clips):
    path = write_csv(
        tmp_path / "mixed.csv",
        ["audio_path", "text"],
        [[clips[0], "maakye"], ["/nonexistent/a.wav", ""]],
    )
    report = AkanAudioDataset(csv_file=path).probe()
    assert report["audio_missing"] == 1
    assert report["empty_labels"] == 1
    assert report["rows"] == 2
    assert report["duration_median_sec"] == 1.0


def test_probe_reports_out_of_vocabulary_characters(tmp_path, clips):
    path = write_csv(tmp_path / "punct.csv", ["audio_path", "text"], [[clips[0], "maakye!!!"]])
    report = AkanAudioDataset(csv_file=path).probe()
    assert report["coverage"] < 1.0
    assert ("!", 3) in report["dropped_chars"]


def test_collator_rejects_an_all_empty_batch():
    """Zero-length CTC targets drive the loss to exactly 0.0 while learning nothing."""
    transform = TextTransform()
    transforms = valid_transforms_for(PipelineConfig())
    batch = [(torch.randn(1, 16000), 16000, "", "a.wav") for _ in range(4)]
    with pytest.raises(EmptyLabelError, match="empty label"):
        data_processing(batch, transform, transforms, stride=2)


def test_collator_accepts_a_partially_empty_batch():
    transform = TextTransform()
    transforms = valid_transforms_for(PipelineConfig())
    batch = [
        (torch.randn(1, 16000), 16000, "maakye", "a.wav"),
        (torch.randn(1, 16000), 16000, "", "b.wav"),
    ]
    _, labels, input_lengths, label_lengths = data_processing(batch, transform, transforms, stride=2)
    assert label_lengths.tolist() == [6, 0]
    assert labels.shape[0] == 2
    assert input_lengths.shape[0] == 2


def test_missing_csv_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        AkanAudioDataset(csv_file=str(tmp_path / "does_not_exist.csv"))
