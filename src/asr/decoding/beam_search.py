"""
CTC prefix beam search with language-model shallow fusion.

Greedy decoding takes the most likely character at every frame independently,
which is why the model's errors are plausible-sounding misspellings: nothing in
the decoder prefers a spelling that exists in Twi over one that does not.

Prefix beam search keeps several candidate transcripts alive and scores each by
the total probability of *every* frame-alignment that collapses to it - which is
what CTC actually defines, and what greedy decoding approximates with a single
path. A language model is then folded in as the candidates are extended:

    score(prefix) = log P_ctc(prefix) + alpha * log P_lm(prefix) + beta * len(prefix)

`alpha` sets how much the language model is trusted. `beta` is a length bonus
that offsets the LM's inherent preference for shorter strings - without it,
raising alpha makes transcripts progressively terser rather than more correct.

The distinction that makes this CTC rather than ordinary beam search: a prefix's
probability is tracked as two numbers, the mass ending in a blank and the mass
ending in a real character. Emitting the same character twice extends the prefix
only when the previous frame ended in a blank; otherwise CTC collapses them.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

NEG_INF = -float("inf")


def _log_sum_exp(a: float, b: float) -> float:
    """Adds two probabilities held in log space without leaving log space."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class _Beam:
    """One candidate prefix, with its probability split by how it ends."""

    blank: float = NEG_INF        # log P(prefix, last frame emitted a blank)
    non_blank: float = NEG_INF    # log P(prefix, last frame emitted a real char)
    lm_score: float = 0.0         # cached, so the LM is queried once per prefix

    @property
    def total(self) -> float:
        return _log_sum_exp(self.blank, self.non_blank)


def beam_search_decode(
    log_probs,
    index_to_char: Dict[int, str],
    blank: int,
    language_model=None,
    beam_width: int = 25,
    alpha: float = 0.5,
    beta: float = 1.0,
    prune_threshold: float = -12.0,
) -> str:
    """
    Decodes one utterance's per-frame log-probabilities to a string.

    `log_probs` is (time, classes). `index_to_char` maps class ids to the
    characters they emit, space included; the blank has no character.

    `prune_threshold` skips characters whose frame log-probability is below it,
    which is what keeps the search affordable: without it every frame considers
    all ~41 symbols against every beam, and almost all of that work is spent on
    characters with no realistic chance.
    """
    frames = len(log_probs)
    if frames == 0:
        return ""

    beams: Dict[str, _Beam] = {"": _Beam(blank=0.0, non_blank=NEG_INF, lm_score=0.0)}

    for t in range(frames):
        frame = log_probs[t]
        candidates: Dict[str, _Beam] = {}

        # Only characters with a real chance at this frame, so the inner loop is
        # over a handful of symbols rather than the whole vocabulary.
        live = [
            index for index in range(len(frame))
            if float(frame[index]) > prune_threshold or index == blank
        ]

        for prefix, beam in beams.items():
            for index in live:
                probability = float(frame[index])

                if index == blank:
                    # A blank never extends the prefix; it only moves mass into
                    # the "ended in blank" half.
                    entry = candidates.setdefault(prefix, _Beam(lm_score=beam.lm_score))
                    entry.blank = _log_sum_exp(entry.blank, beam.total + probability)
                    continue

                char = index_to_char.get(index)
                if char is None:
                    continue

                last = prefix[-1] if prefix else None

                if char == last:
                    # Repeating the last character: extends the prefix only from
                    # the blank-ending mass. From the non-blank mass CTC collapses
                    # it, so that path stays on the same prefix.
                    same = candidates.setdefault(prefix, _Beam(lm_score=beam.lm_score))
                    same.non_blank = _log_sum_exp(same.non_blank, beam.non_blank + probability)

                    if beam.blank == NEG_INF:
                        continue
                    extended = prefix + char
                    entry = candidates.get(extended)
                    if entry is None:
                        entry = _Beam(lm_score=_lm_score(language_model, prefix, char, beam.lm_score))
                        candidates[extended] = entry
                    entry.non_blank = _log_sum_exp(entry.non_blank, beam.blank + probability)
                else:
                    extended = prefix + char
                    entry = candidates.get(extended)
                    if entry is None:
                        entry = _Beam(lm_score=_lm_score(language_model, prefix, char, beam.lm_score))
                        candidates[extended] = entry
                    entry.non_blank = _log_sum_exp(entry.non_blank, beam.total + probability)

        beams = dict(sorted(
            candidates.items(),
            key=lambda item: _rank(item[0], item[1], alpha, beta),
            reverse=True,
        )[:beam_width])

    if not beams:
        return ""
    best = max(beams.items(), key=lambda item: _rank(item[0], item[1], alpha, beta))
    return best[0]


def _lm_score(language_model, prefix: str, char: str, running: float) -> float:
    """Extends a prefix's cached LM score by one character."""
    if language_model is None:
        return 0.0
    return running + language_model.log_prob(char, prefix)


def _rank(prefix: str, beam: _Beam, alpha: float, beta: float) -> float:
    """The fused score a beam is kept or discarded on."""
    return beam.total + alpha * beam.lm_score + beta * len(prefix)


def decode_batch(
    log_probs,
    index_to_char: Dict[int, str],
    blank: int,
    lengths: Optional[Sequence[int]] = None,
    **kwargs,
) -> List[str]:
    """Decodes a (batch, time, classes) tensor, honouring per-item lengths."""
    results = []
    for i in range(len(log_probs)):
        frames = int(lengths[i]) if lengths is not None else len(log_probs[i])
        results.append(
            beam_search_decode(log_probs[i][:frames], index_to_char, blank, **kwargs)
        )
    return results
