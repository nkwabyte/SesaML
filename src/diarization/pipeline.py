"""
Speaker-attributed transcription: "who spoke when" joined to "what was said".

The order matters. Diarizing first and transcribing each turn separately - the
approach here - gives every utterance a speaker without the ASR model needing
any notion of speakers, and works with a plain CTC model that emits no
timestamps. The alternative, transcribing first and assigning speakers to words
afterwards, needs word-level alignment the CTC decoder here does not produce.

The cost is that the ASR model sees each turn in isolation, so it has no
cross-turn context. For a character-level CTC model with no language model that
costs almost nothing, since it has no long-range context to lose.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..config import PipelineConfig
from ..inference.transcribe import Transcriber
from ..utils.audio_io import load_audio
from .backends import DEFAULT_BACKEND, BaseDiarizer, build_diarizer
from .turns import SpeakerTurn


@dataclass
class Utterance:
    """One speaker turn with its transcription."""

    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "transcript": self.text,
        }


def format_timestamp(seconds: float) -> str:
    """mm:ss.s, which reads better than raw seconds in a transcript."""
    minutes, remainder = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{remainder:04.1f}"


class DiarizedTranscriber:
    """
    Runs a diarization backend and an ASR backend over one recording.

    Both are held open between calls: loading the ECAPA encoder and a Conformer
    checkpoint takes seconds, which is fine once and unacceptable per request in
    an interactive app.
    """

    def __init__(
        self,
        model_type: str = "deepspeech",
        model_path: Optional[str] = None,
        backend: str = DEFAULT_BACKEND,
        config: Optional[PipelineConfig] = None,
        transcriber: Optional[Transcriber] = None,
        diarizer: Optional[BaseDiarizer] = None,
        token: Optional[str] = None,
        whisper_repo: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.transcriber = transcriber or Transcriber(
            model_type=model_type,
            model_path=model_path,
            config=self.config,
            **({"whisper_repo": whisper_repo} if whisper_repo else {}),
        )
        self.diarizer = diarizer or build_diarizer(
            backend, token=token, device=self.config.device
        )

    def describe(self) -> Dict[str, Any]:
        return {"asr": self.transcriber.describe(), "diarization": self.diarizer.describe()}

    def transcribe(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        apply_noise_reduction: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns speaker-attributed utterances plus timing and provenance.

        The file is decoded once and every turn sliced from that one waveform;
        re-reading per turn would dominate the runtime on a long recording.
        """
        started = time.time()
        waveform, sample_rate = load_audio(
            audio_path, target_sample_rate=self.config.audio.sample_rate
        )
        audio_seconds = waveform.shape[-1] / sample_rate

        diarization_started = time.time()
        turns = self.diarizer.diarize(waveform, sample_rate, num_speakers=num_speakers)
        diarization_seconds = time.time() - diarization_started

        # No detectable speech turns: transcribe the whole file rather than
        # returning nothing, so a single-speaker clip still produces output.
        if not turns:
            turns = [SpeakerTurn("SPEAKER_00", 0.0, audio_seconds)]

        asr_started = time.time()
        utterances = []
        for turn in turns:
            start = max(0, int(turn.start * sample_rate))
            end = min(waveform.shape[-1], int(turn.end * sample_rate))
            text = self.transcriber.transcribe_waveform(
                waveform[:, start:end],
                sample_rate=sample_rate,
                apply_noise_reduction=apply_noise_reduction,
            )
            utterances.append(Utterance(turn.speaker, turn.start, turn.end, text.strip()))
        asr_seconds = time.time() - asr_started

        # Turns whose audio decoded to nothing are dropped here rather than in
        # the loop, so the timing above still reflects the real work done.
        spoken = [u for u in utterances if u.text]
        elapsed = time.time() - started

        return {
            "audio_path": audio_path,
            "audio_seconds": round(audio_seconds, 2),
            "speakers": sorted({u.speaker for u in spoken}),
            "num_speakers": len({u.speaker for u in spoken}),
            "utterances": [u.to_dict() for u in spoken],
            "empty_turns": len(utterances) - len(spoken),
            "timing": {
                "total_sec": round(elapsed, 2),
                "diarization_sec": round(diarization_seconds, 2),
                "asr_sec": round(asr_seconds, 2),
                "realtime_factor": round(elapsed / audio_seconds, 2) if audio_seconds else None,
            },
            **self.describe(),
        }


def format_transcript(result: Dict[str, Any], with_timestamps: bool = True) -> str:
    """Renders the result as the speaker-labelled transcript people actually read."""
    lines = []
    for utterance in result.get("utterances", []):
        if with_timestamps:
            stamp = f"[{format_timestamp(utterance['start'])} - {format_timestamp(utterance['end'])}] "
        else:
            stamp = ""
        lines.append(f"{stamp}{utterance['speaker']}: {utterance['transcript']}")
    return "\n".join(lines)


def diarize_and_transcribe(
    audio_path: str,
    model_type: str = "deepspeech",
    model_path: Optional[str] = None,
    backend: str = DEFAULT_BACKEND,
    num_speakers: Optional[int] = None,
    config: Optional[PipelineConfig] = None,
    apply_noise_reduction: bool = False,
) -> Dict[str, Any]:
    """One-shot helper for the CLI; the app holds a DiarizedTranscriber open instead."""
    pipeline = DiarizedTranscriber(
        model_type=model_type, model_path=model_path, backend=backend, config=config
    )
    return pipeline.transcribe(
        audio_path, num_speakers=num_speakers, apply_noise_reduction=apply_noise_reduction
    )
