"""
Regression tests for variable-length batches.

Run train-20260810-171239 died on the first real batch with
`AssertionError: Expected key_padded_mask.shape[1] to be 21, but got 20`.
torchaudio's Conformer sizes its padding mask by `lengths.max()` and asserts
that width matches the encoder input, but the collate function floors lengths to
`time // 4` while the two stride-2 convolutions ceil - so any batch whose longest
clip does not divide evenly leaves the mask a frame short. Every clip in a real
corpus has a different length, so this fires immediately on real audio and never
on the equal-length tensors the unit tests used.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import PipelineConfig
from src.asr.data.dataset import data_processing
from src.asr.data.text_transform import TextTransform
from src.main import valid_transforms_for
from src.asr.models import build_architecture, subsampling_factor

ARCHITECTURES = ["deepspeech", "conformer", "conformer-medium"]

# Durations chosen so the padded length subsamples unevenly - the case the
# equal-length fixtures in the older tests never produce.
RAGGED_BATCHES = [
    (1.0, 2.3, 3.7),
    (2.0, 2.1),
    (1.5, 7.9, 2.5, 3.3),
    (0.9, 1.0, 10.4),
]


def make_batch(durations, text="wo ho", sample_rate=16000):
    return [
        (torch.randn(1, int(d * sample_rate)), sample_rate, text, f"clip{i}.wav")
        for i, d in enumerate(durations)
    ]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("durations", RAGGED_BATCHES)
def test_forward_survives_ragged_batch(architecture, durations):
    text_transform = TextTransform()
    config = PipelineConfig()
    config.model.architecture = architecture
    stride = subsampling_factor(architecture)

    spectrograms, _, input_lengths, _ = data_processing(
        make_batch(durations), text_transform, valid_transforms_for(config), stride=stride
    )
    model = build_architecture(architecture, text_transform.vocab_size, config)
    output = model(spectrograms, input_lengths)

    assert output.shape[0] == len(durations)
    assert output.shape[2] == text_transform.vocab_size
    # CTC needs at least as many frames as the longest declared input length.
    assert output.shape[1] >= int(input_lengths.max())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_ctc_loss_is_finite_on_a_ragged_batch(architecture):
    text_transform = TextTransform()
    config = PipelineConfig()
    config.model.architecture = architecture
    stride = subsampling_factor(architecture)

    spectrograms, labels, input_lengths, label_lengths = data_processing(
        make_batch((1.0, 2.3, 3.7)), text_transform, valid_transforms_for(config), stride=stride
    )
    model = build_architecture(architecture, text_transform.vocab_size, config)
    output = F.log_softmax(model(spectrograms, input_lengths), dim=2).transpose(0, 1)

    loss = nn.CTCLoss(blank=text_transform.blank_label)(output, labels, input_lengths, label_lengths)
    assert torch.isfinite(loss), f"{architecture} produced a non-finite CTC loss on a ragged batch"
    assert loss.item() > 0.0
