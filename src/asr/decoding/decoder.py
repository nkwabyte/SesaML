"""
One decoder, shared by inference and evaluation.

Transcription and evaluation must decode identically, or the WER a run reports
describes a different system than the one the app serves. Both go through
`build_decoder` so there is one place where the choice is made.

The language model is loaded once and held; it is a few megabytes of JSON and
re-reading it per utterance would dominate the decode.
"""

import os
from typing import Callable, Dict, List, Optional

from .beam_search import beam_search_decode
from .language_model import CharNGramLM


class CTCDecoder:
    """Decodes per-frame log-probabilities to text, with or without an LM."""

    def __init__(
        self,
        text_transform,
        decoder: str = "beam",
        lm_path: Optional[str] = None,
        beam_width: int = 25,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.text_transform = text_transform
        self.blank = text_transform.blank_label
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.language_model: Optional[CharNGramLM] = None
        self.decoder = decoder

        self.index_to_char: Dict[int, str] = {
            index: (" " if char == "<SPACE>" else char)
            for index, char in text_transform.index_map.items()
            if char != "<BLANK>"
        }

        if decoder == "beam" and lm_path:
            if os.path.isfile(lm_path):
                self.language_model = CharNGramLM.load(lm_path)
            else:
                # Beam search without an LM still helps a little, and is a far
                # better outcome than refusing to transcribe because a model
                # file is missing.
                self.decoder = "beam"
                self.language_model = None

    @property
    def name(self) -> str:
        if self.decoder != "beam":
            return "greedy"
        return "beam+lm" if self.language_model is not None else "beam"

    def describe(self) -> dict:
        return {
            "decoder": self.name,
            "beam_width": self.beam_width if self.decoder == "beam" else None,
            "alpha": self.alpha if self.language_model else None,
            "beta": self.beta if self.language_model else None,
            "lm_contexts": len(self.language_model.counts) if self.language_model else None,
        }

    def decode(self, log_probs) -> str:
        """Decodes one utterance's (time, classes) log-probabilities."""
        if self.decoder != "beam":
            from ..training.evaluator import greedy_decoder

            return self.text_transform.int_to_text(
                greedy_decoder(log_probs.unsqueeze(1), blank_label=self.blank)[0]
            )

        return beam_search_decode(
            log_probs,
            self.index_to_char,
            self.blank,
            language_model=self.language_model,
            beam_width=self.beam_width,
            alpha=self.alpha,
            beta=self.beta,
        )

    def decode_batch(self, log_probs, lengths=None) -> List[str]:
        """Decodes a (batch, time, classes) tensor, honouring per-item lengths."""
        results = []
        for i in range(len(log_probs)):
            frames = int(lengths[i]) if lengths is not None else log_probs.shape[1]
            frames = min(frames, log_probs.shape[1])
            results.append(self.decode(log_probs[i][:frames]))
        return results


def build_decoder(config, text_transform) -> CTCDecoder:
    """Builds the decoder described by a PipelineConfig."""
    decoding = getattr(config, "decoding", None)
    if decoding is None:
        return CTCDecoder(text_transform, decoder="greedy")

    return CTCDecoder(
        text_transform,
        decoder=decoding.decoder,
        lm_path=decoding.lm_path,
        beam_width=decoding.beam_width,
        alpha=decoding.alpha,
        beta=decoding.beta,
    )
