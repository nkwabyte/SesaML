"""
Audio file I/O with a backend that actually exists at runtime.

torchaudio 2.9 removed its built-in sox/soundfile backends and made
`torchaudio.load`/`info`/`save` thin delegates to `torchcodec`. Where torchcodec
is not installed - it is not a dependency of torchaudio, so a plain
`pip install torchaudio==2.9.0` does not bring it - every one of those calls
raises ImportError on the first real audio file. `soundfile` is a pinned
dependency and reads WAV/FLAC/OGG directly, so it is the primary backend here
and torchaudio is tried first only so that formats soundfile lacks (mp3 on some
builds) still work where torchcodec is present.
"""

import io
from typing import Tuple

import torch


class AudioLoadError(RuntimeError):
    """Raised when no available backend could decode an audio file."""


def _load_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    import soundfile as sf

    # always_2d keeps mono and stereo on the same code path; soundfile returns
    # (frames, channels) where torch convention is (channels, frames).
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data).transpose(0, 1).contiguous(), int(sample_rate)


def _load_torchaudio(path: str) -> Tuple[torch.Tensor, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    return waveform, int(sample_rate)


def load_audio(path: str, target_sample_rate: int = None, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Reads an audio file into a (channels, frames) float32 tensor.

    Resamples to `target_sample_rate` when given, and downmixes to mono by
    default - the mel front-end expects a single channel, and a stereo file
    would otherwise silently double the spectrogram's channel dimension.
    """
    errors = []
    waveform = None
    for backend in (_load_torchaudio, _load_soundfile):
        try:
            waveform, sample_rate = backend(path)
            break
        except Exception as exc:  # ImportError, format errors, missing codecs
            errors.append(f"{backend.__name__}: {type(exc).__name__}: {exc}")

    if waveform is None:
        raise AudioLoadError(
            f"Could not decode {path!r} with any available backend. Tried:\n  "
            + "\n  ".join(errors)
            + "\nInstall `torchcodec` for torchaudio's decoder, or ensure `soundfile` "
              "supports this format (it does not read mp3 on all platforms)."
        )

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sample_rate and sample_rate != target_sample_rate:
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return waveform, sample_rate


def load_audio_bytes(
    data: bytes,
    target_sample_rate: int = None,
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Decodes an audio file already held in memory.

    HuggingFace corpora store the encoded container inline in their Arrow
    tables. Decoding those bytes directly avoids `datasets`' own decoder, which
    since 4.x delegates to torchcodec and therefore needs a matching FFmpeg
    shared library - absent on plenty of machines that can still train happily.
    """
    import soundfile as sf

    with io.BytesIO(data) as buffer:
        array, sample_rate = sf.read(buffer, dtype="float32", always_2d=True)

    waveform = torch.from_numpy(array).transpose(0, 1).contiguous()

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sample_rate and sample_rate != target_sample_rate:
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return waveform, int(sample_rate)


def audio_duration_bytes(data: bytes) -> float:
    """Duration of in-memory audio, from the header alone where the format allows."""
    import soundfile as sf

    with io.BytesIO(data) as buffer:
        info = sf.info(buffer)
    return float(info.frames) / float(info.samplerate)


def audio_duration(path: str) -> float:
    """
    Returns a file's duration in seconds, reading the header only where possible.

    Used by corpus probes, which touch many files and must not pay for full
    decodes just to report a median clip length.
    """
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        waveform, sample_rate = load_audio(path)
        return waveform.shape[-1] / float(sample_rate)
