"""
Model architectures and the registry that selects between them.

Every CTC architecture shares one contract so the trainer, evaluator, decoder
and exporter work with any of them unchanged:

    forward(x: (batch, 1, n_mels, time), lengths: (batch,) | None)
        -> logits (batch, time // subsampling_factor, n_class)

`subsampling_factor` is the piece that cannot be assumed: the collate function
derives CTC input_lengths from it, and a mismatch makes CTC fail or train
against wrong alignments.
"""

from typing import Any

import torch.nn as nn

from .deepspeech import SpeechRecognitionModel, ResCNN, CNNLayerNorm
from .conformer import ConformerCTC, ConvSubsampling, conformer_ctc_medium, conformer_ctc_small
from .whisper_model import WhisperASR

# Architecture name -> human description, for CLI help and run logs.
ARCHITECTURES = {
    "deepspeech": "DeepSpeech2-style residual CNN + bidirectional GRU (CTC)",
    "conformer": "Conformer-S encoder: attention + depthwise conv (CTC)",
    "conformer-medium": "Conformer-M encoder, ~3x the parameters of conformer",
}

DEFAULT_ARCHITECTURE = "deepspeech"


def build_architecture(name: str, n_class: int, config: Any) -> nn.Module:
    """
    Instantiates an architecture by name using the shared PipelineConfig.

    `n_class` comes from the live TextTransform vocabulary rather than config,
    so the model and the label encoding can never disagree.
    """
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {', '.join(sorted(ARCHITECTURES))}."
        )

    model_config = config.model
    audio_config = config.audio

    if name == "deepspeech":
        return SpeechRecognitionModel(
            n_cnn_layers=model_config.n_cnn_layers,
            n_rnn_layers=model_config.n_rnn_layers,
            rnn_dim=model_config.rnn_dim,
            n_class=n_class,
            # Both architectures read the feature count from the audio config, so
            # changing n_mels cannot leave one of them built for the old width.
            n_feats=audio_config.n_mels,
            stride=model_config.stride,
            dropout=model_config.dropout,
        )

    builder = conformer_ctc_small if name == "conformer" else conformer_ctc_medium
    return builder(n_class=n_class, n_feats=audio_config.n_mels, dropout=model_config.dropout)


def subsampling_factor(name: str) -> int:
    """
    Time reduction the named architecture applies, before instantiating it.

    The collate function needs this to compute CTC input_lengths, and building a
    whole model just to read one attribute would be wasteful.
    """
    if name == "deepspeech":
        return SpeechRecognitionModel.subsampling_factor
    if name in ("conformer", "conformer-medium"):
        return ConformerCTC.subsampling_factor
    raise ValueError(f"Unknown architecture '{name}'.")


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_ARCHITECTURE",
    "build_architecture",
    "subsampling_factor",
    "SpeechRecognitionModel",
    "ConformerCTC",
    "ConvSubsampling",
    "conformer_ctc_small",
    "conformer_ctc_medium",
    "ResCNN",
    "CNNLayerNorm",
    "WhisperASR",
]
