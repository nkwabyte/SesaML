"""
Speaker turns: the interchange format between diarization and ASR.

Every diarization backend produces the same thing - a list of (speaker, start,
end) intervals - so the transcription side never learns which backend ran. The
post-processing here matters more than it looks: diarizers emit fragmented and
sometimes overlapping turns, and feeding those to a CTC model verbatim produces
a transcript chopped mid-word.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class SpeakerTurn:
    """One continuous stretch of speech attributed to one speaker."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, object]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


def merge_adjacent(turns: Sequence[SpeakerTurn], max_gap: float = 0.5) -> List[SpeakerTurn]:
    """
    Joins consecutive turns from the same speaker separated by at most `max_gap`.

    Diarizers cut on brief pauses, so a single sentence often arrives as four or
    five turns. Transcribing each in isolation costs accuracy - a CTC model has
    no context across calls, and word fragments at the cut points decode badly -
    so the pieces are rejoined before any audio is sliced.
    """
    if not turns:
        return []

    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    merged = [ordered[0]]
    for turn in ordered[1:]:
        last = merged[-1]
        if turn.speaker == last.speaker and turn.start - last.end <= max_gap:
            merged[-1] = SpeakerTurn(last.speaker, last.start, max(last.end, turn.end))
        else:
            merged.append(turn)
    return merged


def drop_short(turns: Sequence[SpeakerTurn], min_duration: float = 0.35) -> List[SpeakerTurn]:
    """
    Removes turns too brief to carry a word.

    Sub-350ms turns are almost always backchannels ("mm", "yes") or diarizer
    jitter at a speaker change. They also subsample to too few frames for CTC to
    emit anything, so keeping them adds empty rows to the transcript.
    """
    return [turn for turn in turns if turn.duration >= min_duration]


def resolve_overlaps(turns: Sequence[SpeakerTurn]) -> List[SpeakerTurn]:
    """
    Rewrites overlapping turns so each instant belongs to exactly one speaker.

    Real conversations overlap, and pyannote reports that honestly, but the
    transcript format here is one line per turn. Two shapes of overlap need
    different treatment:

    * partial - the turns are split at the midpoint of the overlap.
    * containment - a short interjection inside a long turn splits the long one
      in two, so the interjection survives and the speaker who was interrupted
      keeps the rest of their turn. Treating this as a partial overlap would
      truncate the interjection to nothing and drop it from the transcript
      entirely, which is how one side of a conversation goes missing.
    """
    pending = deque(sorted(turns, key=lambda t: (t.start, t.end)))
    resolved: List[SpeakerTurn] = []

    while pending:
        turn = pending.popleft()
        if turn.duration <= 0:
            continue

        if not resolved or turn.start >= resolved[-1].end:
            resolved.append(turn)
            continue

        previous = resolved[-1]

        if turn.speaker == previous.speaker:
            resolved[-1] = SpeakerTurn(previous.speaker, previous.start, max(previous.end, turn.end))
        elif turn.end < previous.end:
            resolved[-1] = SpeakerTurn(previous.speaker, previous.start, turn.start)
            resolved.append(turn)
            # The remainder of the interrupted turn re-enters the queue so it is
            # checked against whatever comes next.
            pending.appendleft(SpeakerTurn(previous.speaker, turn.end, previous.end))
        else:
            boundary = (turn.start + previous.end) / 2
            resolved[-1] = SpeakerTurn(previous.speaker, previous.start, boundary)
            resolved.append(SpeakerTurn(turn.speaker, boundary, turn.end))

    return [turn for turn in resolved if turn.duration > 0]


def clean_turns(
    turns: Sequence[SpeakerTurn],
    max_gap: float = 0.5,
    min_duration: float = 0.35
) -> List[SpeakerTurn]:
    """The full post-processing chain applied to every backend's raw output."""
    return drop_short(merge_adjacent(resolve_overlaps(turns), max_gap=max_gap), min_duration=min_duration)


def relabel_by_first_appearance(turns: Sequence[SpeakerTurn]) -> List[SpeakerTurn]:
    """
    Renames speakers to SPEAKER_00, 01, ... in order of first speaking.

    Clustering assigns arbitrary ids, so the same recording can label the same
    person differently between runs. Ordering by first appearance makes the
    transcript stable and readable.
    """
    mapping: Dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"SPEAKER_{len(mapping):02d}"
    return [SpeakerTurn(mapping[t.speaker], t.start, t.end) for t in turns]
