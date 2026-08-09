import os
from typing import List, Tuple, Optional, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

from .text_transform import TextTransform
from .audio_transforms import get_train_audio_transforms, get_valid_audio_transforms
from ..utils.noise_reduction import reduce_audio_noise

class AkanAudioDataset(Dataset):
    """
    Dataset class for Akan speech-to-text audio and transcriptions.
    Can be initialized from a CSV file, a pandas DataFrame, or a directory of WAV files.
    """

    def __init__(
        self,
        csv_file: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        audio_dir: Optional[str] = None,
        audio_col: str = "audio_path",
        text_col: str = "text",
        sample_rate: int = 16000,
        apply_noise_reduction: bool = False
    ):
        self.sample_rate = sample_rate
        self.apply_noise_reduction = apply_noise_reduction
        self.audio_dir = audio_dir

        if dataframe is not None:
            self.data = dataframe.copy()
        elif csv_file is not None and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = pd.DataFrame(columns=[audio_col, text_col])

        self.audio_col = audio_col
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.data.iloc[idx]
        audio_path = str(row[self.audio_col]) if self.audio_col in row else ""
        text = str(row[self.text_col]) if self.text_col in row else ""

        if self.audio_dir and not os.path.isabs(audio_path):
            full_path = os.path.join(self.audio_dir, audio_path)
        else:
            full_path = audio_path

        if os.path.exists(full_path):
            waveform, sr = torchaudio.load(full_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
        else:
            # Fallback mock 1-second waveform if audio file is not present locally
            waveform = torch.randn(1, self.sample_rate)

        if self.apply_noise_reduction:
            waveform = reduce_audio_noise(waveform, sample_rate=self.sample_rate)

        return waveform, self.sample_rate, text, full_path


class AudioCollator:
    """
    Picklable collate function binding the text and audio transforms.

    DataLoader workers start with `spawn` on macOS and Windows, which cannot
    pickle a collate function defined as a local closure - it fails the moment
    num_workers > 0. A module-level callable survives the pickling.
    """

    def __init__(self, text_transform: TextTransform, audio_transforms: nn.Module, stride: int = 2):
        self.text_transform = text_transform
        self.audio_transforms = audio_transforms
        self.stride = stride

    def __call__(self, batch: List[Tuple[torch.Tensor, int, str, str]]):
        return data_processing(batch, self.text_transform, self.audio_transforms, stride=self.stride)


def data_processing(
    batch: List[Tuple[torch.Tensor, int, str, str]],
    text_transform: TextTransform,
    audio_transforms: nn.Module,
    stride: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to prepare batches for PyTorch model and CTCLoss.
    Pads variable-length audio spectrograms and text label sequences.
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, sr, text, _ in batch:
        # Compute MelSpectrogram: shape (channels, n_mels, time) -> squeeze(0) -> transpose to (time, n_mels)
        spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        label_seq = torch.Tensor(text_transform.text_to_int(text))
        labels.append(label_seq)

        # Input length downsampled by CNN stride
        input_lengths.append(spec.shape[0] // stride)
        label_lengths.append(len(label_seq))

    # Pad spectrograms: (batch, time, n_mels) -> unsqueeze(1) -> transpose to (batch, channel=1, n_mels, time)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    if labels and any(len(l) > 0 for l in labels):
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    else:
        labels = torch.zeros((len(batch), 1), dtype=torch.long)

    return spectrograms, labels, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(label_lengths, dtype=torch.long)
