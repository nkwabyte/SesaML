"""
A character n-gram language model over Twi, for CTC decoding.

The model's errors are overwhelmingly plausible-sounding misspellings - it hears
the sounds and writes them wrong. Greedy CTC decoding has no way to prefer a
spelling that exists in the language over one that does not, because it picks
the most likely character at every frame independently. An n-gram model supplies
exactly that missing preference.

**Character-level, not word-level.** Twi is agglutinative and there is no large
Twi lexicon to hand, so a word model would be mostly out-of-vocabulary. A
character model has no OOV case at all and targets spelling directly, which is
the error being fixed.

Scored with stupid backoff rather than Kneser-Ney: it is not a normalised
probability, but for rescoring - where only the relative ordering of candidate
prefixes matters - it performs comparably and has no discounting parameters to
get subtly wrong.

Written in-repo rather than using KenLM because `kenlm` and `pyctcdecode` pin
`numpy<2`, and this project runs numpy 2.3.3 under torch 2.9. Downgrading numpy
to add a decoder would risk the training environment for the sake of a component
that is a few hundred lines.
"""

import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# Probability mass discount applied each time the model backs off to a shorter
# context. 0.4 is the value from Brants et al., who introduced stupid backoff.
BACKOFF = 0.4

DEFAULT_ORDER = 6
# Start-of-sequence padding, so the first characters of an utterance are scored
# in context rather than from nothing.
BOS = "\x02"


class CharNGramLM:
    """A character n-gram model with stupid backoff."""

    def __init__(self, order: int = DEFAULT_ORDER, backoff: float = BACKOFF):
        self.order = order
        self.backoff = backoff
        # context -> {next_char: count}
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: Dict[str, int] = defaultdict(int)
        self.vocabulary: set = set()

    # --- training --------------------------------------------------------

    def train(self, lines: Iterable[str]) -> "CharNGramLM":
        """Counts every n-gram up to `order` in the given lines."""
        for line in lines:
            text = BOS * (self.order - 1) + line
            self.vocabulary.update(line)
            for i in range(self.order - 1, len(text)):
                char = text[i]
                # Every order from full context down to unigram, so backoff has
                # something to fall back to at each level.
                for n in range(self.order):
                    context = text[i - n:i]
                    self.counts[context][char] += 1
                    self.context_totals[context] += 1
        return self

    def prune(self, min_count: int = 2) -> "CharNGramLM":
        """
        Drops n-grams seen fewer than `min_count` times.

        Singletons are mostly typos and proper nouns; they inflate the model
        several-fold while contributing almost nothing to rescoring. Unigrams are
        kept regardless, because they are the final backoff and must be complete.
        """
        for context in list(self.counts):
            if not context:
                continue
            kept = {c: n for c, n in self.counts[context].items() if n >= min_count}
            if kept:
                self.counts[context] = kept
                self.context_totals[context] = sum(kept.values())
            else:
                del self.counts[context]
                del self.context_totals[context]
        return self

    # --- scoring ---------------------------------------------------------

    def log_prob(self, char: str, context: str) -> float:
        """
        log P(char | context), backing off to shorter contexts as needed.

        Returns a large negative number rather than -inf for a character never
        seen at all, so a single unknown symbol cannot veto an entire beam.
        """
        context = context[-(self.order - 1):] if self.order > 1 else ""
        penalty = 0.0

        while True:
            table = self.counts.get(context)
            if table and char in table:
                return math.log(table[char] / self.context_totals[context]) + penalty
            if not context:
                break
            context = context[1:]
            penalty += math.log(self.backoff)

        return -20.0 + penalty

    def score(self, text: str) -> float:
        """Total log-probability of a string, for diagnostics and perplexity."""
        padded = BOS * (self.order - 1) + text
        return sum(
            self.log_prob(padded[i], padded[:i])
            for i in range(self.order - 1, len(padded))
        )

    def perplexity(self, lines: Iterable[str]) -> float:
        """
        Per-character perplexity. Lower is better.

        The number to watch when choosing an order or a pruning threshold: it
        says how surprised the model is by held-out Twi, independent of any
        decoding parameters.
        """
        total_log_prob = 0.0
        total_chars = 0
        for line in lines:
            if not line:
                continue
            total_log_prob += self.score(line)
            total_chars += len(line)
        if not total_chars:
            return float("inf")
        return math.exp(-total_log_prob / total_chars)

    # --- persistence -----------------------------------------------------

    def save(self, path: str) -> str:
        """Writes the model as JSON, so it is portable and inspectable."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "order": self.order,
            "backoff": self.backoff,
            "vocabulary": sorted(self.vocabulary),
            "counts": {ctx: dict(table) for ctx, table in self.counts.items()},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "CharNGramLM":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        model = cls(order=payload["order"], backoff=payload.get("backoff", BACKOFF))
        model.vocabulary = set(payload.get("vocabulary", []))
        for context, table in payload["counts"].items():
            model.counts[context] = dict(table)
            model.context_totals[context] = sum(table.values())
        return model

    def describe(self) -> dict:
        return {
            "order": self.order,
            "contexts": len(self.counts),
            "vocabulary": len(self.vocabulary),
            "total_counts": sum(self.context_totals.get("", {}) if isinstance(
                self.context_totals.get(""), dict) else [self.context_totals.get("", 0)]),
        }
