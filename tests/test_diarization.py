"""
Tests for speaker diarization.

The turn post-processing carries most of the risk here: a diarizer's raw output
is fragmented and overlapping, and feeding that straight to a CTC model yields a
transcript chopped mid-word. The backend tests use the `spectral` diarizer,
which needs no downloads, so the suite runs offline.
"""

import math

import pytest
import torch

from src.diarization import (
    SpeakerTurn,
    build_diarizer,
    clean_turns,
    detect_speech,
    drop_short,
    format_timestamp,
    format_transcript,
    merge_adjacent,
    resolve_overlaps,
    window_regions,
)
from src.diarization.backends import DiarizationError, EcapaDiarizer, _gated_repos
from src.diarization.turns import relabel_by_first_appearance

SAMPLE_RATE = 16000


def voice(seconds: float, f0: float, seed: int) -> torch.Tensor:
    """A harmonic stack standing in for a voice; f0 separates the speakers."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal = sum(
        (1.0 / (k ** 0.8)) * torch.sin(2 * math.pi * f0 * k * t + torch.rand(1, generator=generator) * 6.28)
        for k in range(1, 25)
    )
    return (signal / signal.abs().max() * 0.3).float()


def silence(seconds: float) -> torch.Tensor:
    """Quiet, but not digital zero - real recordings always have a noise floor."""
    return 0.001 * torch.randn(int(seconds * SAMPLE_RATE))


@pytest.fixture
def two_speakers():
    """A alternating with B: 0-2, 2.6-4.6, 5.2-6.8, 7.4-9.2 seconds."""
    return torch.cat([
        voice(2.0, 110, 1), silence(0.6),
        voice(2.0, 210, 2), silence(0.6),
        voice(1.6, 110, 3), silence(0.6),
        voice(1.8, 210, 4),
    ])


# --- turn post-processing -------------------------------------------------

def test_merge_adjacent_joins_the_same_speaker_across_a_short_pause():
    turns = [SpeakerTurn("A", 0.0, 1.0), SpeakerTurn("A", 1.2, 2.0)]
    merged = merge_adjacent(turns, max_gap=0.5)
    assert merged == [SpeakerTurn("A", 0.0, 2.0)]


def test_merge_adjacent_keeps_a_long_pause_separate():
    turns = [SpeakerTurn("A", 0.0, 1.0), SpeakerTurn("A", 3.0, 4.0)]
    assert len(merge_adjacent(turns, max_gap=0.5)) == 2


def test_merge_adjacent_never_joins_different_speakers():
    turns = [SpeakerTurn("A", 0.0, 1.0), SpeakerTurn("B", 1.05, 2.0)]
    assert len(merge_adjacent(turns, max_gap=0.5)) == 2


def test_drop_short_removes_backchannels():
    turns = [SpeakerTurn("A", 0.0, 0.2), SpeakerTurn("B", 1.0, 3.0)]
    assert drop_short(turns, min_duration=0.35) == [SpeakerTurn("B", 1.0, 3.0)]


def test_resolve_overlaps_splits_at_the_midpoint():
    turns = [SpeakerTurn("A", 0.0, 2.0), SpeakerTurn("B", 1.0, 3.0)]
    resolved = resolve_overlaps(turns)
    assert resolved == [SpeakerTurn("A", 0.0, 1.5), SpeakerTurn("B", 1.5, 3.0)]


def test_resolve_overlaps_keeps_both_speakers_present():
    """Dropping the quieter speaker would silently lose half a conversation."""
    turns = [SpeakerTurn("A", 0.0, 5.0), SpeakerTurn("B", 1.0, 2.0)]
    speakers = {t.speaker for t in resolve_overlaps(turns)}
    assert speakers == {"A", "B"}


def test_every_instant_has_at_most_one_speaker_after_cleaning():
    turns = [
        SpeakerTurn("A", 0.0, 3.0),
        SpeakerTurn("B", 2.0, 5.0),
        SpeakerTurn("A", 4.5, 6.0),
    ]
    cleaned = clean_turns(turns)
    for earlier, later in zip(cleaned, cleaned[1:]):
        assert earlier.end <= later.start + 1e-6


def test_relabel_orders_speakers_by_first_appearance():
    turns = [SpeakerTurn("cluster_7", 5.0, 6.0), SpeakerTurn("cluster_2", 0.0, 1.0)]
    relabelled = relabel_by_first_appearance(turns)
    by_start = sorted(relabelled, key=lambda t: t.start)
    assert by_start[0].speaker == "SPEAKER_00"
    assert by_start[1].speaker == "SPEAKER_01"


def test_clean_turns_on_empty_input():
    assert clean_turns([]) == []


# --- voice activity detection ---------------------------------------------

def test_vad_finds_each_speech_region(two_speakers):
    regions = detect_speech(two_speakers, SAMPLE_RATE)
    assert len(regions) == 4
    starts = [round(s, 1) for s, _ in regions]
    assert starts == pytest.approx([0.0, 2.5, 5.1, 7.3], abs=0.25)


def test_vad_handles_a_file_that_is_entirely_speech():
    regions = detect_speech(voice(3.0, 150, 9), SAMPLE_RATE)
    assert len(regions) == 1
    assert regions[0][1] - regions[0][0] > 2.5


def test_vad_never_returns_nothing_for_a_quiet_file():
    """A uniformly quiet recording is still worth transcribing."""
    regions = detect_speech(silence(2.0), SAMPLE_RATE)
    assert regions and regions[0][1] > regions[0][0]


def test_window_regions_splits_long_turns_with_overlap():
    windows = window_regions([(0.0, 5.0)], window=1.5, hop=0.75)
    assert len(windows) > 3
    assert all(end - start <= 1.5 + 1e-6 for start, end in windows)
    assert windows[1][0] < windows[0][1], "windows should overlap"


def test_window_regions_keeps_a_short_region_whole():
    assert window_regions([(1.0, 2.0)], window=1.5, hop=0.75) == [(1.0, 2.0)]


def test_window_regions_drops_slivers():
    assert window_regions([(0.0, 0.2)], window=1.5, hop=0.75, min_window=0.5) == []


# --- backends -------------------------------------------------------------

def test_spectral_backend_separates_two_voices(two_speakers):
    turns = build_diarizer("spectral").diarize(two_speakers, SAMPLE_RATE)
    assert len({t.speaker for t in turns}) == 2
    assert len(turns) == 4
    # Alternating, and the same voice must get the same label both times.
    labels = [t.speaker for t in turns]
    assert labels[0] == labels[2] and labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_backend_respects_an_explicit_speaker_count(two_speakers):
    turns = build_diarizer("spectral").diarize(two_speakers, SAMPLE_RATE, num_speakers=1)
    assert len({t.speaker for t in turns}) == 1


def test_backend_handles_a_single_speaker_monologue():
    turns = build_diarizer("spectral").diarize(voice(4.0, 130, 11), SAMPLE_RATE)
    assert len({t.speaker for t in turns}) == 1


def test_backend_handles_audio_shorter_than_one_window():
    turns = build_diarizer("spectral").diarize(voice(0.6, 130, 12), SAMPLE_RATE)
    assert len(turns) <= 1


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown diarization backend"):
        build_diarizer("nonexistent")


def test_auto_backend_falls_back_to_something_that_loads():
    """A missing licence must downgrade the demo, not end it."""
    diarizer = build_diarizer("auto")
    assert diarizer.name in ("pyannote", "ecapa", "spectral")


# --- error reporting ------------------------------------------------------

def test_gated_repo_is_extracted_from_the_error_message():
    """Naming the wrong repo sends people to re-accept conditions they already have."""
    exc = RuntimeError(
        "Cannot access gated repo for url "
        "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/x.npz. "
        "Access to model pyannote/speaker-diarization-community-1 is restricted"
    )
    assert "pyannote/speaker-diarization-community-1" in _gated_repos(exc)


def test_gated_repo_is_extracted_through_a_chained_cause():
    inner = RuntimeError("Access to model pyannote/segmentation-3.0 is restricted")
    outer = RuntimeError("pipeline failed")
    outer.__cause__ = inner
    assert "pyannote/segmentation-3.0" in _gated_repos(outer)


# --- formatting -----------------------------------------------------------

def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00.0"
    assert format_timestamp(65.4) == "01:05.4"


def test_format_transcript_labels_each_speaker():
    result = {"utterances": [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "transcript": "maakye"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.0, "transcript": "wo ho te sɛn"},
    ]}
    text = format_transcript(result)
    assert "SPEAKER_00: maakye" in text
    assert "SPEAKER_01: wo ho te sɛn" in text
    assert "[00:00.0 - 00:02.0]" in text


def test_format_transcript_without_timestamps():
    result = {"utterances": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "transcript": "maakye"}]}
    assert format_transcript(result, with_timestamps=False) == "SPEAKER_00: maakye"


def test_speaker_turn_serialises_for_the_api():
    turn = SpeakerTurn("SPEAKER_00", 1.0, 3.5)
    assert turn.to_dict() == {"speaker": "SPEAKER_00", "start": 1.0, "end": 3.5, "duration": 2.5}


# --- device selection -----------------------------------------------------

@pytest.mark.parametrize("requested,expected", [
    ("cpu", "cpu"),
    ("cuda", "cuda"),
    ("cuda:1", "cuda:1"),
    ("mps", "cpu"),      # Apple Silicon: speechbrain has no branch for it
    ("xpu", "cpu"),
])
def test_ecapa_narrows_the_device_to_what_speechbrain_handles(requested, expected):
    """
    speechbrain 1.1.0 sets device_type only for "cpu" or "cuda*". Any other
    value leaves the attribute unset and the constructor raises AttributeError
    several frames later - which is what the app hit on Apple Silicon, since
    device detection returns "mps" there.
    """
    assert EcapaDiarizer._supported_device(requested) == expected
