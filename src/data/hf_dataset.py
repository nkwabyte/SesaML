from typing import Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import torchaudio

from .text_transform import TextTransform
from ..utils.noise_reduction import reduce_audio_noise

class HuggingFaceAkanDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace speech datasets (e.g. `ghanaopendata/twi-speech-text-multispeaker-16k`).
    Loads audio arrays/waveforms and Akan text transcriptions.
    """

    def __init__(
        self,
        dataset_name: str = "ghanaopendata/twi-speech-text-multispeaker-16k",
        split: str = "train",
        sample_rate: int = 16000,
        apply_noise_reduction: bool = False,
        hf_dataset: Optional[Any] = None
    ):
        self.sample_rate = sample_rate
        self.apply_noise_reduction = apply_noise_reduction

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            from datasets import load_dataset
            self.hf_dataset = load_dataset(dataset_name, split=split)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        item = self.hf_dataset[idx]
        audio_item = item.get("audio", {})
        text = str(item.get("text", ""))

        if isinstance(audio_item, dict):
            array = audio_item.get("array")
            sr = audio_item.get("sampling_rate", self.sample_rate)
            path = audio_item.get("path", f"sample_{idx}")

            if array is not None:
                waveform = torch.tensor(array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                waveform = torch.randn(1, self.sample_rate)
        else:
            waveform = torch.randn(1, self.sample_rate)
            sr = self.sample_rate
            path = f"sample_{idx}"

        if sr != self.sample_rate and waveform.ndim > 0:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.apply_noise_reduction:
            waveform = reduce_audio_noise(waveform, sample_rate=self.sample_rate)

        return waveform, self.sample_rate, text, str(path)
