"""
Voice activity detection and windowing for the embedding-based diarizers.

pyannote brings its own neural segmentation model. The backends that do not -
because that model is gated - need speech regions from somewhere, and this is
that somewhere: an energy gate calibrated per recording, then fixed windows cut
from the speech it finds.

The gate is adaptive rather than a fixed dB threshold because the corpora here
are crowd-sourced on phones, where absolute levels vary by an order of magnitude
between recordings. A per-file percentile adapts; a constant does not.
"""

from typing import List, Sequence, Tuple

import torch


def frame_energies(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0
) -> Tuple[torch.Tensor, float]:
    """Root-mean-square energy per frame, in dB, plus the hop in seconds."""
    samples = waveform.reshape(-1)
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))

    if samples.numel() < frame_length:
        return torch.zeros(0), hop_length / sample_rate

    frames = samples.unfold(0, frame_length, hop_length)
    rms = frames.pow(2).mean(dim=1).clamp(min=1e-12).sqrt()
    return 20 * torch.log10(rms), hop_length / sample_rate


def detect_speech(
    waveform: torch.Tensor,
    sample_rate: int,
    margin_db: float = 8.0,
    dynamic_range_db: float = 35.0,
    min_speech: float = 0.35,
    min_silence: float = 0.30,
    pad: float = 0.10
) -> List[Tuple[float, float]]:
    """
    Returns (start, end) speech regions in seconds.

    The threshold is anchored from both ends of the file's own energy
    distribution: `margin_db` above the noise floor, but never more than
    `dynamic_range_db` below the speech peak. Anchoring on the floor alone fails
    on a recording that is mostly speech - the low percentile then sits inside
    the speech itself and the gate never closes - while anchoring on the peak
    alone fails on a quiet recording with one loud moment. Taking the stricter
    of the two handles both.
    """
    energies, hop = frame_energies(waveform, sample_rate)
    if energies.numel() == 0:
        duration = waveform.shape[-1] / sample_rate
        return [(0.0, duration)] if duration > 0 else []

    floor = torch.quantile(energies, 0.10)
    peak = torch.quantile(energies, 0.95)
    threshold = torch.maximum(floor + margin_db, peak - dynamic_range_db)
    speech = energies > threshold

    regions: List[Tuple[float, float]] = []
    start = None
    for idx, active in enumerate(speech.tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            regions.append((start * hop, idx * hop))
            start = None
    if start is not None:
        regions.append((start * hop, len(speech) * hop))

    regions = _bridge_gaps(regions, min_silence)
    regions = [(s, e) for s, e in regions if e - s >= min_speech]

    duration = waveform.shape[-1] / sample_rate
    padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in regions]

    # Everything below the gate means the file is uniformly quiet, not silent;
    # treating it as one region beats returning an empty transcript.
    return padded or [(0.0, duration)]


def _bridge_gaps(regions: Sequence[Tuple[float, float]], min_silence: float) -> List[Tuple[float, float]]:
    """Joins regions separated by less than `min_silence` of quiet."""
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        last_start, last_end = merged[-1]
        if start - last_end < min_silence:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def window_regions(
    regions: Sequence[Tuple[float, float]],
    window: float = 1.5,
    hop: float = 0.75,
    min_window: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Cuts speech regions into overlapping windows for speaker embedding.

    Embeddings need roughly a second of speech to be stable, but a turn can run
    for a minute - so regions are cut into windows, each labelled independently,
    and consecutive same-label windows become one turn. The overlap lets a
    speaker change land inside a window without losing the boundary entirely.
    """
    windows: List[Tuple[float, float]] = []
    for start, end in regions:
        if end - start <= window:
            if end - start >= min_window:
                windows.append((start, end))
            continue
        position = start
        while position < end:
            stop = min(position + window, end)
            if stop - position >= min_window:
                windows.append((position, stop))
            if stop >= end:
                break
            position += hop
    return windows
