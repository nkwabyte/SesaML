"""
Tests for the feature front-end and the training-loop guarantees.

These cover the changes that decide whether the model can learn at all: log
compression and normalization of the input, length-aware recurrence, and the
bookkeeping that lets a long run be resumed instead of restarted.
"""

import math

import pytest
import torch
import torch.nn as nn

from src.config import PipelineConfig
from src.data.audio_transforms import LogMelNormalize
from src.data.text_transform import TextTransform
from src.main import train_transforms_for, valid_transforms_for
from src.models import build_architecture
from src.models.deepspeech import BidirectionalGRU, SpeechRecognitionModel


def test_mel_filters_are_all_populated():
    """n_mels must fit the FFT's frequency resolution or filters come out empty."""
    import torchaudio

    config = PipelineConfig()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels,
    )
    empty = int((mel.mel_scale.fb.sum(dim=0) == 0).sum())
    assert empty == 0, f"{empty} mel filters are all-zero; n_mels is too high for n_fft"


def test_features_are_log_compressed_and_normalized():
    """Raw mel power spans orders of magnitude; the encoder needs it standardized."""
    config = PipelineConfig()
    transforms = valid_transforms_for(config)
    waveform = torch.randn(1, config.audio.sample_rate * 2) * 0.1

    features = transforms(waveform)

    assert features.shape[1] == config.audio.n_mels
    # Per-utterance CMVN over the time axis: roughly zero mean, unit variance.
    assert abs(float(features.mean())) < 0.1
    assert 0.5 < float(features.std()) < 1.5
    # Log compression means negatives exist; a power spectrogram is non-negative.
    assert float(features.min()) < 0.0


def test_normalization_is_invariant_to_recording_gain():
    """Two takes of the same audio at different volumes must yield the same features."""
    transform = LogMelNormalize()
    mel = torch.rand(1, 80, 50) + 0.1

    quiet = transform(mel * 0.01)
    loud = transform(mel * 100.0)

    assert torch.allclose(quiet, loud, atol=1e-3)


def test_specaugment_only_applies_to_training_features():
    config = PipelineConfig()
    waveform = torch.randn(1, config.audio.sample_rate) * 0.1

    torch.manual_seed(0)
    train_out = train_transforms_for(config)(waveform)
    valid_out = valid_transforms_for(config)(waveform)

    assert train_out.shape == valid_out.shape
    # Masking zeroes whole bands, so the augmented copy cannot match exactly.
    assert not torch.allclose(train_out, valid_out)


def test_bigru_output_is_invariant_to_padding():
    """
    The same frames must encode identically however much padding follows them.

    A bidirectional GRU's backward pass starts at the last frame, so on a padded
    batch it consumes the padding first and carries that state into the real
    speech - meaning an utterance decoded differently depending on which clips
    happened to share its batch. Packing confines the recurrence to real frames.
    """
    torch.manual_seed(0)
    gru = BidirectionalGRU(rnn_dim=16, hidden_size=16, dropout=0.0, batch_first=True).eval()

    speech = torch.randn(1, 30, 16)
    padded = torch.cat([speech, torch.randn(1, 70, 16)], dim=1)
    lengths = torch.tensor([30])

    with torch.no_grad():
        assert torch.allclose(gru(speech, lengths), gru(padded, lengths)[:, :30], atol=1e-5)
        # And confirm the guard is load-bearing: without lengths it does drift.
        drift = (gru(speech) - gru(padded)[:, :30]).abs().max()
    assert float(drift) > 1e-3


def test_deepspeech_threads_lengths_into_every_gru_layer():
    """The model must pass lengths down, not just accept the argument and drop it."""
    torch.manual_seed(0)
    model = SpeechRecognitionModel(n_class=41, n_cnn_layers=1, n_rnn_layers=2, rnn_dim=32, n_feats=80)
    model.eval()

    clip = torch.randn(1, 1, 80, 200)
    padded = torch.zeros(1, 1, 80, 400)
    padded[:, :, :, :200] = clip
    lengths = torch.tensor([100])

    with torch.no_grad():
        a = model(clip, lengths)
        b = model(padded, lengths)

    frames = int(lengths[0])
    drift = (a[:, :frames] - b[:, :frames]).abs().max(dim=2).values[0]

    # The convolutional front-end reaches a few frames past the boundary, so
    # some bleed there is inherent. What must hold is that it stays local: an
    # unpacked GRU contaminates every frame instead, including the first, and
    # does so at ~0.33 - three orders of magnitude above this bound. The
    # interior is not asserted bit-identical because packing a longer tensor
    # changes the reduction order, which moves the last few decimal places.
    assert float(drift[: frames - 15].max()) < 1e-3, "padding is leaking into the interior"
    assert float(drift.max()) < 0.1


@pytest.mark.parametrize("architecture", ["deepspeech", "conformer"])
def test_models_accept_the_configured_feature_width(architecture):
    config = PipelineConfig()
    config.model.architecture = architecture
    text_transform = TextTransform()

    model = build_architecture(architecture, text_transform.vocab_size, config)
    features = torch.randn(2, 1, config.audio.n_mels, 120)
    output = model(features, torch.tensor([30, 20]))

    assert output.shape[0] == 2
    assert output.shape[2] == text_transform.vocab_size


def test_gradient_clipping_bounds_the_update():
    """The clip must actually bind, or a single bad batch can undo an epoch."""
    model = nn.Linear(4, 4)
    loss = (model(torch.randn(8, 4)) * 1e6).sum()
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    clipped = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters()))

    assert float(norm) > 5.0, "test needs a gradient that actually exceeds the threshold"
    assert clipped <= 5.0 + 1e-3


def test_ctc_loss_tolerates_targets_longer_than_input():
    """
    zero_infinity keeps one impossible sample from poisoning the batch.

    Clips where the transcript cannot fit the subsampled frames exist in every
    corpus; without this the whole batch's gradient becomes NaN.
    """
    torch.manual_seed(0)
    logits = torch.randn(5, 2, 41).log_softmax(dim=2)
    targets = torch.randint(0, 40, (2, 20))
    input_lengths = torch.tensor([5, 5])
    target_lengths = torch.tensor([3, 20])  # second is impossible in 5 frames

    strict = nn.CTCLoss(blank=40)(logits, targets, input_lengths, target_lengths)
    tolerant = nn.CTCLoss(blank=40, zero_infinity=True)(logits, targets, input_lengths, target_lengths)

    assert not torch.isfinite(strict)
    assert torch.isfinite(tolerant)


# --- progress bars must never kill a run ----------------------------------

def test_training_loop_survives_a_broken_output_stream():
    """
    A 100-epoch run died two hours in with OSError(22) raised from tqdm's write
    to a stdout whose reader had gone away. A progress bar is cosmetic; losing
    one must never cost a model.
    """
    from src.utils.progress import progress

    class BrokenStream:
        def write(self, *args):
            raise OSError(22, "Invalid argument")

        def flush(self, *args):
            raise OSError(22, "Invalid argument")

        def isatty(self):
            return True

    total = 0
    for value in progress(range(500), file=BrokenStream(), disable=False, mininterval=0):
        total += value
    assert total == sum(range(500))


def test_progress_is_off_when_no_terminal_is_attached(monkeypatch):
    """Detached runs (schtasks, nohup, a closed SSH session) draw no bar."""
    from src.utils import progress as progress_module

    monkeypatch.delenv("SESAML_PROGRESS", raising=False)
    monkeypatch.setattr(progress_module.sys, "stderr", None)
    monkeypatch.setattr(progress_module.sys, "stdout", None)
    assert progress_module.progress_enabled() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("yes", True), ("off", False)])
def test_progress_env_override(monkeypatch, value, expected):
    from src.utils import progress as progress_module

    monkeypatch.setenv("SESAML_PROGRESS", value)
    assert progress_module.progress_enabled() is expected
