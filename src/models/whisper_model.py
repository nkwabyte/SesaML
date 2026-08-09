import os
from typing import Optional, Union, Dict, Any
import torch

class WhisperASR:
    """
    Wrapper for fine-tuned HuggingFace Whisper automatic speech recognition model.
    Used for inference and evaluation on Akan audio.
    """

    def __init__(
        self,
        model_name_or_path: str = "CiBeDL/twi_trained_whisper",
        device: Optional[str] = None
    ):
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
        result = self.pipeline(audio)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        elif isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "")
        return str(result)
