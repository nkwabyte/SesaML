import os
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

from ..config import PipelineConfig
from ..data.text_transform import TextTransform
from ..data.audio_transforms import get_valid_audio_transforms
from ..models import build_architecture, SpeechRecognitionModel
from ..models.whisper_model import WhisperASR
from ..training.evaluator import greedy_decoder
from ..utils.audio_io import load_audio
from ..utils.noise_reduction import reduce_audio_noise
from ..utils.run_logger import load_model_meta, load_weights, resolve_checkpoint

DEFAULT_WHISPER_REPO = "CiBeDL/twi_trained_whisper"

def load_deepspeech_model(
    model_path: str,
    config: Optional[PipelineConfig] = None,
    text_transform: Optional[TextTransform] = None
) -> torch.nn.Module:
    """Loads state dict into configured or auto-detected CTC architecture."""
    config = config or PipelineConfig()
    text_transform = text_transform or TextTransform()
    device = torch.device(config.device)

    meta = load_model_meta(model_path)
    arch_name = (meta and meta.get("architecture")) or config.model.architecture

    n_class = text_transform.vocab_size
    model = build_architecture(arch_name, n_class, config).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(load_weights(model_path, map_location=device))
    model.eval()
    return model


class Transcriber:
    """
    Holds a loaded ASR model so repeated transcriptions reuse it.

    The one-shot `transcribe_audio()` helper below reloads weights on every
    call, which is fine for the CLI but far too slow for the Gradio app - the
    app keeps one Transcriber per backend instead.
    """

    def __init__(
        self,
        model_type: str = "deepspeech",
        model_path: Optional[str] = None,
        whisper_repo: str = DEFAULT_WHISPER_REPO,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.model_type = model_type.lower()
        self.whisper_repo = whisper_repo
        self.device = torch.device(self.config.device)
        self.text_transform = TextTransform()

        if self.model_type == "whisper":
            self.model_path = whisper_repo
            self._whisper = WhisperASR(model_name_or_path=whisper_repo, device=self.config.device)
            self._model = None
        else:
            self.model_path = resolve_checkpoint(model_path, self.config)
            self._whisper = None
            self._model = load_deepspeech_model(
                model_path=self.model_path,
                config=self.config,
                text_transform=self.text_transform
            )

        # Inference must use exactly the features training used - n_fft and
        # hop_length included, or the model sees a different time resolution
        # than it was trained on.
        self._audio_transforms = get_valid_audio_transforms(
            sample_rate=self.config.audio.sample_rate,
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length
        )

    @property
    def is_trained(self) -> bool:
        """False when the DeepSpeech backend fell back to random weights."""
        return self.model_type == "whisper" or os.path.exists(self.model_path)

    def describe(self) -> Dict[str, Any]:
        """Model provenance, surfaced in the app UI and written to the run log."""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": str(self.device),
            "sample_rate": self.config.audio.sample_rate,
            "trained_weights_found": self.is_trained,
        }

    def transcribe(self, audio_path: str, apply_noise_reduction: bool = False) -> str:
        """Transcribes one audio file to Akan text."""
        if self.model_type == "whisper":
            return self._whisper.transcribe(audio_path)

        waveform, _ = load_audio(audio_path, target_sample_rate=self.config.audio.sample_rate)

        if apply_noise_reduction:
            waveform = reduce_audio_noise(waveform, sample_rate=self.config.audio.sample_rate)

        # (channels, n_mels, time) -> squeeze -> transpose to (time, n_mels)
        # -> unsqueeze batch and channel -> (1, 1, n_mels, time)
        spec = self._audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spec = spec.unsqueeze(0).unsqueeze(1).transpose(2, 3).to(self.device)

        with torch.no_grad():
            output = self._model(spec)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, class)
            decoded_outputs = greedy_decoder(output, blank_label=self.text_transform.blank_label)
            return self.text_transform.int_to_text(decoded_outputs[0])


def transcribe_audio(
    audio_path: str,
    model_type: str = "deepspeech",
    model_path: Optional[str] = None,
    whisper_repo: str = DEFAULT_WHISPER_REPO,
    config: Optional[PipelineConfig] = None,
    apply_noise_reduction: bool = False
) -> str:
    """
    Transcribes an Akan audio file to Akan text.
    Supports either 'deepspeech' (custom PyTorch CTC model) or 'whisper'
    (HuggingFace transformer pipeline).
    """
    transcriber = Transcriber(
        model_type=model_type,
        model_path=model_path,
        whisper_repo=whisper_repo,
        config=config
    )
    return transcriber.transcribe(audio_path, apply_noise_reduction=apply_noise_reduction)
