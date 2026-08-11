"""Speaker diarization: who spoke when, joined to the SesaML ASR transcript."""

from .backends import (
    BACKENDS,
    DEFAULT_BACKEND,
    BaseDiarizer,
    DiarizationError,
    EcapaDiarizer,
    PyannoteDiarizer,
    SpectralDiarizer,
    build_diarizer,
)
from .pipeline import (
    DiarizedTranscriber,
    Utterance,
    diarize_and_transcribe,
    format_timestamp,
    format_transcript,
)
from .segmentation import detect_speech, window_regions
from .turns import SpeakerTurn, clean_turns, drop_short, merge_adjacent, resolve_overlaps

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "BaseDiarizer",
    "DiarizationError",
    "EcapaDiarizer",
    "PyannoteDiarizer",
    "SpectralDiarizer",
    "build_diarizer",
    "DiarizedTranscriber",
    "Utterance",
    "diarize_and_transcribe",
    "format_timestamp",
    "format_transcript",
    "detect_speech",
    "window_regions",
    "SpeakerTurn",
    "clean_turns",
    "drop_short",
    "merge_adjacent",
    "resolve_overlaps",
]
