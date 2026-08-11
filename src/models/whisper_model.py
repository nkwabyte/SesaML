import os
from typing import Optional, Union, Dict, Any
import torch

class WhisperASR:
    """
    Wrapper for fine-tuned HuggingFace Whisper automatic speech recognition model.
    Used for inference and evaluation on Akan audio.
    """

    # Whisper decodes autoregressively, so on audio it cannot model it falls
    # into repetition loops - emitting the same syllable until it hits the token
    # limit. That is the single ugliest failure mode in a live demo, and it is
    # cheap to bound: forbid repeating any 4-token span, and penalise repeats.
    GENERATE_KWARGS = {"no_repeat_ngram_size": 4, "repetition_penalty": 1.15}

    def __init__(
        self,
        model_name_or_path: str = None,
        device: Optional[str] = None
    ):
        if not model_name_or_path:
            raise ValueError(
                "WhisperASR needs a repository id. This project serves its own trained "
                "models; Whisper is an optional comparison baseline, so there is no default."
            )
        self.model_name_or_path = model_name_or_path
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        elif device in ["cuda", "0"]:
            self.device = 0
        else:
            self.device = -1

        self.pipeline = None

    def load_pipeline(self):
        """Lazy loader for HuggingFace ASR pipeline."""
        if self.pipeline is None:
            from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

            try:
                model = WhisperForConditionalGeneration.from_pretrained(self.model_name_or_path)
                processor = WhisperProcessor.from_pretrained(self.model_name_or_path)
                self.pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=self.device
                )
            except Exception as e:
                # Fallback to standard pipeline initialization by repo id
                self.pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=self.model_name_or_path,
                    device=self.device
                )

    def transcribe(self, audio: Union[str, bytes]) -> str:
        """Transcribe audio file path or audio bytes to text."""
        self.load_pipeline()
        return self._text(self.pipeline(audio, generate_kwargs=self.GENERATE_KWARGS))

    def transcribe_waveform(self, waveform, sample_rate: int = 16000) -> str:
        """
        Transcribe samples already in memory.

        Diarization hands over one speaker turn at a time, sliced from an
        already-decoded file; writing each slice back to disk just so the
        pipeline could re-read it would dominate the runtime.
        """
        self.load_pipeline()

        array = waveform
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        if array.ndim > 1:
            array = array.mean(axis=0)

        return self._text(self.pipeline(
            {"raw": array, "sampling_rate": sample_rate},
            generate_kwargs=self.GENERATE_KWARGS,
        ))

    @staticmethod
    def _text(result: Any) -> str:
        """Normalises the pipeline's several return shapes to a plain string."""
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "")
        return str(result)
