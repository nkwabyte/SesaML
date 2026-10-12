"""
Whether a training sample can contribute a gradient at all.

CTC aligns a label sequence to encoder frames monotonically, so it needs at
least one frame per label character - and a blank frame between any two
identical adjacent characters. A sample that violates this has no valid
alignment: its loss is infinite, `zero_infinity=True` turns that into a zero,
and the sample occupies a batch slot while contributing nothing.

That is not hypothetical here. Measured on the corpora in use:

    ghanaopendata   19% of samples infeasible
    health          0% infeasible, but median headroom 1.90 frames per character
    lagyamfi        0% infeasible, median headroom 2.93

A fifth of one corpus was silently training on nothing. Headroom below about 2
is where alignment becomes hard even when it is possible, which is the regime
the 30-second health clips sit in.

Nothing here decodes audio: durations come from the same cache that
length-bucketing already builds, and target lengths from the text.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

# Below this many frames per character, a valid alignment exists but the model
# has very little room to place blanks. Empirical, from the three corpora above:
# the two that train well sit at or above ~1.6, the one that failed at 1.90 with
# a fifth of its mass under 1.5.
COMFORTABLE_HEADROOM = 2.0


def encoder_frames(seconds: float, sample_rate: int, hop_length: int, subsampling: int) -> int:
    """Encoder output frames for a clip of `seconds`, after the front-end and subsampling."""
    mel_frames = int(seconds * sample_rate / max(1, hop_length))
    return mel_frames // max(1, subsampling)


def minimum_frames(encoded: Sequence[int]) -> int:
    """
    Frames a label sequence needs at minimum.

    One per character, plus one blank between each pair of identical adjacent
    characters - CTC collapses repeats, so "aa" cannot be emitted without a
    blank between the two a's. Ignoring that under-counts the requirement for
    exactly the doubled letters Twi uses freely (`aa`, `ɛɛ`, `oo`).
    """
    if not encoded:
        return 0
    required = len(encoded)
    required += sum(1 for a, b in zip(encoded, encoded[1:]) if a == b)
    return required


@dataclass
class FeasibilityReport:
    """How much of a corpus can actually train."""

    total: int
    infeasible: int
    tight: int
    empty: int
    median_headroom: float
    p05_headroom: float

    @property
    def infeasible_ratio(self) -> float:
        return self.infeasible / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "sampled": self.total,
            "infeasible": self.infeasible,
            "infeasible_pct": round(100 * self.infeasible_ratio, 1),
            "tight": self.tight,
            "empty_targets": self.empty,
            "headroom_median": round(self.median_headroom, 2),
            "headroom_p05": round(self.p05_headroom, 2),
        }


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def assess(
    durations: Sequence[float],
    encoded_targets: Sequence[Sequence[int]],
    sample_rate: int,
    hop_length: int,
    subsampling: int,
) -> FeasibilityReport:
    """Measures how many samples can align, and with how much room to spare."""
    infeasible = tight = empty = 0
    headrooms: List[float] = []

    for seconds, encoded in zip(durations, encoded_targets):
        if not encoded:
            empty += 1
            continue
        frames = encoder_frames(seconds, sample_rate, hop_length, subsampling)
        needed = minimum_frames(encoded)
        if frames < needed:
            infeasible += 1
            headrooms.append(frames / needed)
            continue
        headroom = frames / needed
        headrooms.append(headroom)
        if headroom < COMFORTABLE_HEADROOM:
            tight += 1

    return FeasibilityReport(
        total=len(durations),
        infeasible=infeasible,
        tight=tight,
        empty=empty,
        median_headroom=_percentile(headrooms, 0.5),
        p05_headroom=_percentile(headrooms, 0.05),
    )


def feasible_indices(
    durations: Sequence[float],
    encoded_targets: Sequence[Sequence[int]],
    sample_rate: int,
    hop_length: int,
    subsampling: int,
    min_headroom: float = 1.0,
) -> List[int]:
    """
    Indices worth training on.

    `min_headroom` of 1.0 keeps everything that can align at all. Raising it
    trades corpus size for alignment difficulty, which is the knob for a corpus
    whose samples are feasible on paper but hard in practice.
    """
    keep = []
    for index, (seconds, encoded) in enumerate(zip(durations, encoded_targets)):
        if not encoded:
            continue
        needed = minimum_frames(encoded)
        frames = encoder_frames(seconds, sample_rate, hop_length, subsampling)
        if needed and frames / needed >= min_headroom:
            keep.append(index)
    return keep
