"""
Tests for the HuggingFace loader's decoding and corpus-resolution behaviour.

Two failures on the training server motivated these. `datasets` 4.x decodes its
Audio feature through torchcodec, which needs an FFmpeg shared library that is
frequently absent - and because indexing a row decodes every column, even
reading the transcript raised. Separately, the download script saves corpora
under data/datasets/ while the training path called `load_dataset`, so a corpus
already on disk was re-fetched from the Hub.
"""

import os

import pytest
import torch

from src.data.hf_dataset import (
    DEFAULT_LOCAL_DIR,
    HuggingFaceAkanDataset,
    parse_dataset_spec,
)
from src.utils.audio_io import audio_duration_bytes, load_audio_bytes


def wav_bytes(seconds=1.0, sample_rate=16000, freq=440.0):
    """A real WAV container in memory, so decoding is exercised for real."""
    import io
    import math
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = int(seconds * sample_rate)
        handle.writeframes(b"".join(
            struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sample_rate)))
            for i in range(frames)
        ))
    return buffer.getvalue()


def test_load_audio_bytes_decodes_without_torchcodec():
    waveform, sample_rate = load_audio_bytes(wav_bytes(seconds=0.5))
    assert sample_rate == 16000
    assert waveform.shape == (1, 8000)
    assert waveform.dtype == torch.float32


def test_load_audio_bytes_resamples():
    waveform, sample_rate = load_audio_bytes(
        wav_bytes(seconds=1.0, sample_rate=8000), target_sample_rate=16000
    )
    assert sample_rate == 16000
    assert waveform.shape[1] == 16000


def test_audio_duration_bytes_reads_the_header():
    assert audio_duration_bytes(wav_bytes(seconds=2.5)) == pytest.approx(2.5, abs=0.01)


def test_getitem_prefers_encoded_bytes_over_decoding():
    """The bytes path is what avoids datasets' torchcodec dependency entirely."""
    rows = [
        {"audio": {"bytes": wav_bytes(seconds=1.0), "path": None}, "sentence": "maakye"},
        {"audio": {"bytes": wav_bytes(seconds=2.0), "path": None}, "sentence": "wo ho te sɛn"},
    ]
    dataset = HuggingFaceAkanDataset(hf_dataset=rows, sample_rate=16000, text_column="sentence")

    waveform, sample_rate, text, _ = dataset[1]
    assert sample_rate == 16000
    assert waveform.shape == (1, 32000)
    assert text == "wo ho te sɛn"


def test_getitem_still_accepts_a_decoded_array():
    """Datasets that already decoded (torchcodec present) must keep working."""
    rows = [{"audio": {"array": [0.0] * 16000, "sampling_rate": 16000, "path": "a.wav"},
             "sentence": "maakye"}]
    dataset = HuggingFaceAkanDataset(hf_dataset=rows, sample_rate=16000, text_column="sentence")

    waveform, sample_rate, _, _ = dataset[0]
    assert waveform.shape == (1, 16000)
    assert sample_rate == 16000


def test_missing_audio_raises_rather_than_substituting_noise():
    from src.data.dataset import MissingAudioError

    rows = [{"audio": {"array": None, "bytes": None, "path": None}, "sentence": "maakye"}]
    dataset = HuggingFaceAkanDataset(hf_dataset=rows, sample_rate=16000, text_column="sentence")

    with pytest.raises(MissingAudioError):
        dataset[0]


def test_probe_counts_undecodable_rows_and_empty_labels():
    rows = [
        {"audio": {"bytes": wav_bytes(seconds=1.0), "path": None}, "sentence": "maakye"},
        {"audio": {"bytes": b"not audio at all", "path": None}, "sentence": ""},
    ]
    dataset = HuggingFaceAkanDataset(hf_dataset=rows, sample_rate=16000, text_column="sentence")

    report = dataset.probe(n_samples=2, n_text_samples=2)
    assert report["audio_missing"] == 1
    assert report["empty_labels"] == 1
    assert report["rows"] == 2


def test_missing_transcription_column_is_rejected():
    rows = [{"audio": {"bytes": wav_bytes(), "path": None}, "notes": "maakye"}]
    with pytest.raises(ValueError, match="No transcription column"):
        HuggingFaceAkanDataset(hf_dataset=rows)


def test_local_path_matches_the_download_script_layout():
    """download_dataset.py writes <save_dir>/<repo__name>/<split>."""
    path = HuggingFaceAkanDataset.local_path("Lagyamfi/akan_audio_processed", "train")
    assert path == os.path.join(DEFAULT_LOCAL_DIR, "Lagyamfi__akan_audio_processed", "train")


def test_local_corpus_is_preferred_over_the_hub(tmp_path, monkeypatch):
    """A corpus already on disk must not be re-fetched from the Hub."""
    import datasets

    corpus = tmp_path / "Some__corpus" / "train"
    corpus.mkdir(parents=True)

    monkeypatch.setattr(datasets, "load_from_disk", lambda path: [
        {"audio": {"bytes": wav_bytes(), "path": None}, "sentence": "maakye"}
    ])

    def explode(*args, **kwargs):
        raise AssertionError("load_dataset was called despite a local copy existing")

    monkeypatch.setattr(datasets, "load_dataset", explode)

    dataset = HuggingFaceAkanDataset(
        dataset_name="Some/corpus", split="train", local_dir=str(tmp_path)
    )
    assert len(dataset) == 1
    assert dataset.source == str(corpus)


@pytest.mark.parametrize("spec,expected", [
    ("repo/name", ("repo/name", "train")),
    ("repo/name:test", ("repo/name", "test")),
    ("repo/name:", ("repo/name", "train")),
])
def test_dataset_spec_parsing(spec, expected):
    assert parse_dataset_spec(spec, "train") == expected
