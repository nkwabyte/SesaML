import numpy as np
import torch

def reduce_audio_noise(waveform: torch.Tensor, sample_rate: int = 16000, prop_decrease: float = 1.0) -> torch.Tensor:
    """
    Applies spectral gate noise reduction using `noisereduce` library.
    Expects waveform tensor of shape (channels, samples) or (samples,).
    """
    try:
        import noisereduce as nr
    except ImportError:
        # If noisereduce is not installed, return waveform unchanged
        return waveform

    audio_np = waveform.numpy()
    if audio_np.ndim == 1:
        reduced_np = nr.reduce_noise(y=audio_np, sr=sample_rate, prop_decrease=prop_decrease)
    else:
        # Multichannel
        reduced_np = np.stack([
            nr.reduce_noise(y=channel, sr=sample_rate, prop_decrease=prop_decrease)
            for channel in audio_np
        ])

    return torch.from_numpy(reduced_np)
