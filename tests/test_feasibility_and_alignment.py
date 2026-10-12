"""
Tests for CTC feasibility, forced alignment and utterance segmentation.

These three exist because of one measured failure: training on the 500-hour
health corpus plateaued at WER 1.0. The measurements behind them are in
docs/asr/results.md - 19% of one corpus could not align at all, and the corpus that
failed packs about seven utterances into every training example.
"""

import math

import pytest
import torch

from src.asr.data.alignment import TokenSpan, forced_align, frames_to_seconds
from src.asr.data.feasibility import (
    COMFORTABLE_HEADROOM,
    assess,
    encoder_frames,
    feasible_indices,
    minimum_frames,
)
from src.asr.data.segmenting import split_on_pauses, summarise

BLANK = 0
SPACE = 1


# --- feasibility ----------------------------------------------------------

def test_encoder_frames_matches_the_front_end():
    """1 second at a 160-sample hop is 100 mel frames, 25 after 4x subsampling."""
    assert encoder_frames(1.0, sample_rate=16000, hop_length=160, subsampling=4) == 25
    assert encoder_frames(1.0, sample_rate=16000, hop_length=160, subsampling=2) == 50


def test_minimum_frames_counts_one_per_character():
    assert minimum_frames([1, 2, 3]) == 3


def test_minimum_frames_charges_a_blank_between_repeats():
    """
    CTC collapses repeated labels, so "aa" needs a blank between the two a's.
    Twi doubles vowels freely (aa, ee, ɔɔ), so ignoring this under-counts.
    """
    assert minimum_frames([1, 1]) == 3
    assert minimum_frames([1, 1, 1]) == 5
    assert minimum_frames([1, 2, 2, 3]) == 5


def test_minimum_frames_on_empty_target():
    assert minimum_frames([]) == 0


def test_assess_counts_infeasible_samples():
    """A target needing more frames than the audio provides cannot train."""
    report = assess(
        durations=[10.0, 0.1],            # 250 frames, then 2
        encoded_targets=[[1, 2, 3], [1, 2, 3, 4, 5]],
        sample_rate=16000, hop_length=160, subsampling=4,
    )
    assert report.total == 2
    assert report.infeasible == 1
    assert report.infeasible_ratio == 0.5


def test_assess_flags_tight_but_possible_samples():
    """Headroom under ~2 aligns in principle and struggles in practice."""
    # 40 frames for 30 characters -> headroom 1.33
    report = assess([1.6], [list(range(1, 31))], 16000, 160, 4)
    assert report.infeasible == 0
    assert report.tight == 1
    assert report.median_headroom < COMFORTABLE_HEADROOM


def test_assess_counts_empty_targets_separately():
    report = assess([5.0], [[]], 16000, 160, 4)
    assert report.empty == 1
    assert report.infeasible == 0


def test_feasible_indices_drops_the_unalignable():
    durations = [10.0, 0.1, 10.0]
    targets = [[1, 2], [1, 2, 3, 4, 5, 6], [3, 4]]
    assert feasible_indices(durations, targets, 16000, 160, 4, min_headroom=1.0) == [0, 2]


def test_feasible_indices_can_demand_more_headroom():
    """Raising the bar trades corpus size for alignment difficulty."""
    durations = [1.6, 10.0]                       # 40 frames, 250 frames
    targets = [list(range(1, 31)), [1, 2, 3]]     # headroom 1.33 and 83
    assert feasible_indices(durations, targets, 16000, 160, 4, min_headroom=2.0) == [1]


def test_feasible_indices_drops_empty_targets():
    assert feasible_indices([5.0], [[]], 16000, 160, 4) == []


# --- forced alignment -----------------------------------------------------

def confident(frames: int, tokens, classes: int = 6) -> torch.Tensor:
    """Log-probs that put nearly all mass on a chosen token per frame."""
    probs = torch.full((frames, classes), 0.01)
    for t, token in enumerate(tokens):
        probs[t, token] = 10.0
    return probs.log_softmax(dim=-1)


def test_alignment_recovers_a_known_layout():
    """Frames 0-1 say token 2, frames 2-3 token 3, with a blank between."""
    log_probs = confident(5, [2, 2, BLANK, 3, 3])
    spans = forced_align(log_probs, [2, 3], blank=BLANK)

    assert [s.token for s in spans] == [2, 3]
    assert spans[0].start == 0 and spans[0].end == 2
    assert spans[1].start == 3 and spans[1].end == 5


def test_alignment_is_monotonic_and_covers_every_target():
    torch.manual_seed(0)
    targets = [1, 2, 3, 4, 5]
    spans = forced_align(torch.randn(40, 6).log_softmax(-1), targets, blank=BLANK)

    assert [s.token for s in spans] == targets
    assert [s.index for s in spans] == list(range(len(targets)))
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end <= later.start, "alignment must not go backwards"


def test_alignment_refuses_an_impossible_target():
    """Two frames cannot hold five characters; better to say so than guess."""
    assert forced_align(torch.randn(2, 6).log_softmax(-1), [1, 2, 3, 4, 5], blank=BLANK) == []


def test_alignment_separates_repeated_characters():
    """"aa" must occupy two spans with a gap, or CTC would collapse it to "a"."""
    spans = forced_align(torch.randn(8, 4).log_softmax(-1), [1, 1], blank=BLANK)
    assert len(spans) == 2
    assert spans[0].end <= spans[1].start


def test_alignment_of_an_empty_target():
    assert forced_align(torch.randn(10, 6).log_softmax(-1), [], blank=BLANK) == []


def test_frames_to_seconds_inverts_the_front_end():
    assert frames_to_seconds(25, hop_length=160, subsampling=4, sample_rate=16000) == pytest.approx(1.0)


# --- segmentation ---------------------------------------------------------

def spans_from(layout):
    """(index, token, start, end) tuples -> TokenSpans with a neutral score."""
    return [TokenSpan(index=i, token=tok, start=s, end=e, score=-1.0)
            for i, tok, s, e in layout]


def test_segmentation_splits_at_the_widest_pause():
    # "ab cd" with a wide gap in the middle; 0.04s per frame.
    spans = spans_from([
        (0, 2, 0, 5), (1, 3, 5, 10), (2, SPACE, 10, 12),
        (3, 4, 300, 305), (4, 5, 305, 310),
    ])
    segments = split_on_pauses(spans, "ab cd", seconds_per_frame=0.04,
                               max_seconds=4.0, min_seconds=0.1, space_token=SPACE)
    assert len(segments) == 2
    assert segments[0].text == "ab"
    assert segments[1].text == "cd"


def test_segmentation_never_cuts_inside_a_word():
    """
    The first version split "biom" into "b" and "iom" because the widest gap
    happened to fall mid-word. Training on those halves teaches nonsense.
    """
    spans = spans_from([
        (0, 2, 0, 2), (1, 3, 400, 402), (2, 4, 402, 404),   # widest gap is inside "abc"
        (3, SPACE, 404, 406), (4, 5, 406, 408),
    ])
    segments = split_on_pauses(spans, "abc d", seconds_per_frame=0.04,
                               max_seconds=1.0, min_seconds=0.01, space_token=SPACE)
    for segment in segments:
        assert not segment.text.startswith(("b", "c")) or " " in segment.text, segment.text
    # Whatever the split, no piece may begin partway through "abc".
    assert all(seg.text in ("abc", "d", "abc d") for seg in segments), [s.text for s in segments]


def test_segmentation_keeps_a_short_clip_whole():
    spans = spans_from([(0, 2, 0, 5), (1, 3, 5, 10)])
    segments = split_on_pauses(spans, "ab", seconds_per_frame=0.04,
                               max_seconds=30.0, min_seconds=0.1, space_token=SPACE)
    assert len(segments) == 1
    assert segments[0].text == "ab"


def test_segmentation_drops_slivers():
    spans = spans_from([(0, 2, 0, 1)])
    assert split_on_pauses(spans, "a", seconds_per_frame=0.04, min_seconds=5.0) == []


def test_segmentation_will_not_split_continuous_speech():
    """No gap wide enough means one long segment, not an arbitrary cut."""
    spans = spans_from([(i, 2 + (i % 3), i * 10, (i + 1) * 10) for i in range(40)])
    segments = split_on_pauses(spans, "x" * 40, seconds_per_frame=0.04,
                               max_seconds=2.0, min_gap=0.5, min_seconds=0.1)
    assert len(segments) == 1


def test_segmentation_on_empty_alignment():
    assert split_on_pauses([], "anything", seconds_per_frame=0.04) == []


def test_summarise_reports_corpus_shape():
    spans = spans_from([(0, 2, 0, 25), (1, SPACE, 25, 30), (2, 3, 400, 425)])
    segments = split_on_pauses(spans, "a b", seconds_per_frame=0.04,
                               max_seconds=2.0, min_seconds=0.1, space_token=SPACE)
    report = summarise(segments)
    assert report["segments"] == len(segments)
    assert report["total_seconds"] > 0


def test_summarise_on_nothing():
    assert summarise([])["segments"] == 0
