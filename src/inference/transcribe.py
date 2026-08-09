import os
from typing import Optional, Union
import torch
import torch.nn.functional as F
import torchaudio

from ..config import PipelineConfig
from ..data.text_transform import TextTransform
from ..data.audio_transforms import get_valid_audio_transforms
from ..models.deepspeech import SpeechRecognitionModel
from ..models.whisper_model import WhisperASR
from ..training.evaluator import greedy_decoder
from ..utils.noise_reduction import reduce_audio_noise
from ..utils.run_logger import resolve_checkpoint

def load_deepspeech_model(
    model_path: str,
    config: Optional[PipelineConfig] = None,
    text_transform: Optional[TextTransform] = None
) -> SpeechRecognitionModel:
    """Loads state dict into SpeechRecognitionModel architecture."""
    config = config or PipelineConfig()
    text_transform = text_transform or TextTransform()
    device = torch.device(config.device)

    n_class = text_transform.vocab_size
    model = SpeechRecognitionModel(
        n_cnn_layers=config.model.n_cnn_layers,
        n_rnn_layers=config.model.n_rnn_layers,
        rnn_dim=config.model.rnn_dim,
        n_class=n_class,
        n_feats=config.model.n_feats,
        stride=config.model.stride,
        dropout=config.model.dropout
    ).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model

def transcribe_audio(
    audio_path: str,
    model_type: str = "deepspeech",
    model_path: Optional[str] = None,
    whisper_repo: str = "CiBeDL/twi_trained_whisper",
    config: Optional[PipelineConfig] = None,
    apply_noise_reduction: bool = False
) -> str:
    """
    Transcribes Akan audio file to Akan text.
    Supports either 'deepspeech' (custom PyTorch CTC model) or 'whisper' (HuggingFace transformer pipeline).
    """
    config = config or PipelineConfig()
    device_str = config.device

    if model_type.lower() == "whisper":
        whisper_asr = WhisperASR(model_name_or_path=whisper_repo, device=device_str)
        return whisper_asr.transcribe(audio_path)

    # Default: deepspeech model
    text_transform = TextTransform()
    model_path = resolve_checkpoint(model_path, config)

    device = torch.device(device_str)
    model = load_deepspeech_model(model_path=model_path, config=config, text_transform=text_transform)

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != config.audio.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.audio.sample_rate)
        waveform = resampler(waveform)

    if apply_noise_reduction:
        waveform = reduce_audio_noise(waveform, sample_rate=config.audio.sample_rate)

    valid_audio_transforms = get_valid_audio_transforms(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels
    )

    # (channels, n_mels, time) -> squeeze -> transpose to (time, n_mels) -> unsqueeze batch and channel -> (1, 1, n_mels, time)
    spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    spec = spec.unsqueeze(0).unsqueeze(1).transpose(2, 3).to(device)

    with torch.no_grad():
        output = model(spec)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, class)
        decoded_outputs = greedy_decoder(output, blank_label=text_transform.blank_label)
        predicted_text = text_transform.int_to_text(decoded_outputs[0])

    return predicted_text
