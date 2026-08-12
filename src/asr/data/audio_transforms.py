"""
Mel front-end shared by training, evaluation and inference.

The pipeline is log-mel -> per-utterance normalization -> SpecAugment, which is
the standard ASR ordering and matters more than it looks:

* Raw mel *power* spans four orders of magnitude (measured 0 to ~2600 on this
  corpus, mean ~230). Feeding that into a convolution with no compression makes
  the loudest frames dominate every gradient and wastes most of the network's
  dynamic range. Log compression is what turns it into a roughly Gaussian
  feature the encoder can actually use.
* Normalizing per utterance removes per-recording gain and channel differences,
  which is the cheapest speaker/microphone invariance available. Crowd-sourced
  Akan corpora are recorded on wildly different phones, so this is not optional.
* SpecAugment comes last, and masks with 0.0 - which after normalization is the
  feature mean, i.e. genuinely uninformative rather than an artificial silence.
"""

import torch
import torch.nn as nn
import torchaudio


class LogMelNormalize(nn.Module):
    """
    Log-compresses a mel spectrogram and normalizes it per utterance.

    Cepstral mean and variance normalization is computed over the time axis of
    this single clip, so it needs no corpus statistics and behaves identically
    at training and inference time.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # (..., n_mels, time). log1p keeps the zero floor of a power spectrogram
        # finite; a bare log would send silent bins to -inf.
        features = torch.log(mel + self.eps)
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True)
        return (features - mean) / (std + self.eps)


def _mel_spectrogram(sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> nn.Module:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )


def get_train_audio_transforms(
    sample_rate: int = 16000,
    n_mels: int = 80,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    n_fft: int = 400,
    hop_length: int = 160,
    time_mask_ratio: float = 0.2
) -> nn.Sequential:
    """
    Training front-end: log-mel, per-utterance normalization, then SpecAugment.

    `time_mask_ratio` caps a time mask at a fraction of the utterance. Without
    it a 100-frame mask erases a full second, which on a short clip removes more
    speech than remains and yields a target the audio no longer supports.
    """
    return nn.Sequential(
        _mel_spectrogram(sample_rate, n_mels, n_fft, hop_length),
        LogMelNormalize(),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
        torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, p=time_mask_ratio),
    )


def get_valid_audio_transforms(
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> nn.Sequential:
    """Validation/inference front-end: the same features, without augmentation."""
    return nn.Sequential(
        _mel_spectrogram(sample_rate, n_mels, n_fft, hop_length),
        LogMelNormalize(),
    )
