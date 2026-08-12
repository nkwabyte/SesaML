"""
CTC forced alignment: where in the audio each label character was spoken.

A CTC model trained without any alignment supervision still contains one
implicitly - the most likely monotonic path through its per-frame posteriors
that emits exactly the known text. Recovering that path turns "here is 30
seconds and 400 characters" into "this character was spoken at this frame",
which is what makes it possible to cut a long recording into utterances without
losing the correspondence between audio and text.

The Viterbi pass is implemented here rather than taken from
`torchaudio.functional.forced_align`, which is deprecated and slated for
removal in the very version this project pins.

The standard CTC trick: the target is expanded to
`blank, c1, blank, c2, ..., blank` and the path may, at each frame, stay on the
current symbol, advance one, or - only when moving between two *different*
non-blank characters - skip the blank between them. The skip is what allows
adjacent characters to occupy consecutive frames; forbidding it between
identical characters is what stops "aa" collapsing to "a".
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

NEG_INF = -1e30


@dataclass
class TokenSpan:
    """One label character, and the frame range it was aligned to."""

    index: int      # position in the original label sequence
    token: int      # vocabulary id
    start: int      # first frame, inclusive
    end: int        # last frame, exclusive
    score: float    # mean log-probability over those frames

    @property
    def frames(self) -> int:
        return max(0, self.end - self.start)


def _expand(targets: Sequence[int], blank: int) -> List[int]:
    """blank, t0, blank, t1, ... blank"""
    expanded = [blank]
    for token in targets:
        expanded.append(token)
        expanded.append(blank)
    return expanded


def forced_align(
    log_probs: torch.Tensor,
    targets: Sequence[int],
    blank: int,
) -> List[TokenSpan]:
    """
    Aligns `targets` to `log_probs` (T, C) and returns one span per target token.

    Returns an empty list when no alignment exists - when the frames cannot
    accommodate the targets plus their required blanks - rather than returning a
    misleading best-effort path. Callers should treat that as "this sample is
    not alignable", which is the same condition `feasibility` predicts from
    durations alone.
    """
    if not len(targets):
        return []

    frames, _ = log_probs.shape
    expanded = _expand(targets, blank)
    states = len(expanded)

    if frames < _minimum_frames(expanded, blank):
        return []

    device = log_probs.device
    symbols = torch.tensor(expanded, device=device, dtype=torch.long)

    # A skip from s-2 to s is legal only onto a non-blank whose predecessor
    # non-blank differs; otherwise the two identical characters would merge.
    can_skip = torch.zeros(states, dtype=torch.bool, device=device)
    for s in range(2, states):
        if expanded[s] != blank and expanded[s] != expanded[s - 2]:
            can_skip[s] = True

    scores = torch.full((states,), NEG_INF, device=device)
    scores[0] = log_probs[0, expanded[0]]
    if states > 1:
        scores[1] = log_probs[0, expanded[1]]

    backpointers = torch.zeros((frames, states), dtype=torch.int8, device=device)

    for t in range(1, frames):
        stay = scores
        advance = torch.cat([torch.full((1,), NEG_INF, device=device), scores[:-1]])
        skip = torch.cat([torch.full((2,), NEG_INF, device=device), scores[:-2]])
        skip = torch.where(can_skip, skip, torch.full_like(skip, NEG_INF))

        stacked = torch.stack([stay, advance, skip])
        best, choice = stacked.max(dim=0)
        scores = best + log_probs[t].index_select(0, symbols)
        backpointers[t] = choice.to(torch.int8)

    # A valid path ends on the final character or the trailing blank.
    last = states - 1
    state = last if scores[last] >= scores[last - 1] else last - 1
    if scores[state] <= NEG_INF / 2:
        return []

    path = [0] * frames
    for t in range(frames - 1, 0, -1):
        path[t] = state
        # The stored choice says how this state was reached from t-1:
        # 0 stayed, 1 advanced one symbol, 2 skipped a blank.
        state -= int(backpointers[t, state].item())
    path[0] = state

    return _spans(path, expanded, log_probs, blank)


def _minimum_frames(expanded: Sequence[int], blank: int) -> int:
    """Frames the expanded sequence needs: every symbol once, blanks skippable."""
    required = 0
    previous = None
    for symbol in expanded:
        if symbol == blank:
            # A blank is only mandatory between two identical characters.
            continue
        if previous is not None and symbol == previous:
            required += 1
        required += 1
        previous = symbol
    return required


def _spans(
    path: Sequence[int],
    expanded: Sequence[int],
    log_probs: torch.Tensor,
    blank: int,
) -> List[TokenSpan]:
    """Turns a per-frame state path into one span per non-blank target token."""
    spans: List[TokenSpan] = []
    current_state: Optional[int] = None
    start = 0

    for t, state in enumerate(list(path) + [-1]):
        if state != current_state:
            if current_state is not None and expanded[current_state] != blank:
                token = expanded[current_state]
                window = log_probs[start:t, token]
                spans.append(TokenSpan(
                    index=(current_state - 1) // 2,
                    token=token,
                    start=start,
                    end=t,
                    score=float(window.mean()) if window.numel() else NEG_INF,
                ))
            current_state = state if state >= 0 else None
            start = t

    return spans


def frames_to_seconds(frame: int, hop_length: int, subsampling: int, sample_rate: int) -> float:
    """Converts an encoder frame index back to a time in the original audio."""
    return frame * subsampling * hop_length / sample_rate
