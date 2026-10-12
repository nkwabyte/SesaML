"""
Cutting long recordings into single-utterance training examples.

The 500-hour health corpus is 30-second windows of broadcast media. Each window
holds around seven separate utterances - measured with the VAD, against one per
clip in both corpora that train successfully - carrying a single ~394-character
transcript spanning all of them. CTC then has to work out, with no supervision
on the boundaries, which characters belong to which stretch of speech. That is a
far harder search than aligning one utterance, and it is why training on this
corpus plateaus while 12 hours of single-utterance audio trains fine.

Splitting the audio alone does not help: the text has to be split with it, at
the same points. Forced alignment supplies exactly that correspondence, so cuts
are made in the *gaps* between aligned characters - the pauses - and each piece
keeps the text actually spoken in it.

The output is short, single-utterance clips of the shape the model already
learns from.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .alignment import TokenSpan

# A gap shorter than this is a pause within an utterance, not between two.
DEFAULT_MIN_GAP = 0.30
# Long enough to hold a sentence, short enough to align comfortably.
DEFAULT_MAX_SECONDS = 12.0
# Below this a segment carries too little speech to be worth a training slot.
DEFAULT_MIN_SECONDS = 0.8


@dataclass
class Segment:
    """One utterance: a time range in the source clip and the text spoken in it."""

    start: float
    end: float
    text: str
    score: float          # mean alignment log-probability, a confidence proxy
    first_char: int
    last_char: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
            "score": round(self.score, 4),
        }


def _gap_seconds(left: TokenSpan, right: TokenSpan, seconds_per_frame: float) -> float:
    return max(0.0, (right.start - left.end) * seconds_per_frame)


def split_on_pauses(
    spans: Sequence[TokenSpan],
    text: str,
    seconds_per_frame: float,
    min_gap: float = DEFAULT_MIN_GAP,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    pad: float = 0.10,
    space_token: Optional[int] = None,
) -> List[Segment]:
    """
    Cuts an aligned clip into utterances at the widest pauses.

    Splits greedily: the longest piece is broken at its widest internal gap,
    repeatedly, until everything is under `max_seconds` or no gap wide enough
    remains. Cutting at the widest available pause rather than the first one
    over a threshold keeps the boundaries at real utterance ends instead of at
    incidental breaths.

    Cuts land only on word boundaries. `space_token` marks where those are;
    without it a split falls wherever the widest gap happens to be, which in
    practice lands mid-word - the first version of this produced
    "...woakwaaba b" and "iom wɔ" as two training examples, teaching the model
    that a word ends in the middle of "biom".
    """
    if not spans:
        return []

    pieces: List[List[TokenSpan]] = [list(spans)]
    changed = True

    def splittable(piece: List[TokenSpan], i: int) -> bool:
        """A cut after position i is allowed only between words."""
        if space_token is None:
            return True
        return piece[i].token == space_token or piece[i + 1].token == space_token

    while changed:
        changed = False
        expanded: List[List[TokenSpan]] = []
        for piece in pieces:
            duration = (piece[-1].end - piece[0].start) * seconds_per_frame
            if duration <= max_seconds or len(piece) < 2:
                expanded.append(piece)
                continue

            widest_index, widest_gap = -1, 0.0
            for i in range(len(piece) - 1):
                if not splittable(piece, i):
                    continue
                gap = _gap_seconds(piece[i], piece[i + 1], seconds_per_frame)
                if gap > widest_gap:
                    widest_index, widest_gap = i, gap

            if widest_index < 0 or widest_gap < min_gap:
                # Nothing to cut on: a continuous stretch longer than the target.
                expanded.append(piece)
                continue

            expanded.append(piece[: widest_index + 1])
            expanded.append(piece[widest_index + 1:])
            changed = True
        pieces = expanded

    segments: List[Segment] = []
    for piece in pieces:
        start = max(0.0, piece[0].start * seconds_per_frame - pad)
        end = piece[-1].end * seconds_per_frame + pad
        if end - start < min_seconds:
            continue

        first, last = piece[0].index, piece[-1].index
        chunk = text[first: last + 1].strip()
        if not chunk:
            continue

        segments.append(Segment(
            start=start,
            end=end,
            text=chunk,
            score=sum(s.score for s in piece) / len(piece),
            first_char=first,
            last_char=last,
        ))

    return segments


def segment_clip(
    log_probs,
    encoded: Sequence[int],
    text: str,
    blank: int,
    seconds_per_frame: float,
    space_token: Optional[int] = None,
    **kwargs,
) -> List[Segment]:
    """
    Aligns a clip's transcript to its audio and returns the utterances found.

    Returns an empty list when the clip cannot be aligned, which is the honest
    outcome for a sample the model has no valid path through - the same samples
    `feasibility` flags from durations alone.
    """
    from .alignment import forced_align

    spans = forced_align(log_probs, encoded, blank=blank)
    if not spans:
        return []
    return split_on_pauses(
        spans, text, seconds_per_frame, space_token=space_token, **kwargs
    )


def summarise(segments: Sequence[Segment]) -> dict:
    """Corpus-level shape of a segmentation run, for the run log."""
    if not segments:
        return {"segments": 0}
    durations = sorted(s.duration for s in segments)
    scores = sorted(s.score for s in segments)
    middle = len(durations) // 2
    return {
        "segments": len(segments),
        "duration_median": round(durations[middle], 2),
        "duration_min": round(durations[0], 2),
        "duration_max": round(durations[-1], 2),
        "total_seconds": round(sum(durations), 1),
        "score_median": round(scores[middle], 3),
        "score_p05": round(scores[max(0, int(len(scores) * 0.05))], 3),
    }
