import torch
import torch.nn as nn
import torchaudio

def get_train_audio_transforms(
    sample_rate: int = 16000,
    n_mels: int = 128,
    freq_mask_param: int = 30,
    time_mask_param: int = 100
) -> nn.Sequential:
    """
    Returns audio transforms for training: MelSpectrogram with Frequency and Time Masking (SpecAugment).
    """
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
        torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    )

def get_valid_audio_transforms(
    sample_rate: int = 16000,
    n_mels: int = 128
) -> nn.Sequential:
    """
    Returns audio transforms for validation/inference: MelSpectrogram.
    """
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    )
