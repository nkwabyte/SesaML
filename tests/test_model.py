import torch
import pytest
from src.asr.models import build_architecture, subsampling_factor, ARCHITECTURES
from src.asr.models.deepspeech import SpeechRecognitionModel
from src.asr.models.conformer import ConformerCTC, conformer_ctc_small

def test_speech_recognition_model_forward():
    batch_size = 2
    n_mels = 128
    time_steps = 100
    n_class = 35

    model = SpeechRecognitionModel(
        n_cnn_layers=2,
        n_rnn_layers=2,
        rnn_dim=128,
        n_class=n_class,
        n_feats=n_mels,
        stride=2
    )

    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    output = model(dummy_input)

    assert output.shape[0] == batch_size
    assert output.shape[1] == time_steps // 2
    assert output.shape[2] == n_class

def test_conformer_ctc_forward():
    batch_size = 2
    n_mels = 128
    time_steps = 100
    n_class = 35

    model = ConformerCTC(
        n_class=n_class,
        n_feats=n_mels,
        encoder_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
    )

    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    dummy_lengths = torch.tensor([50, 25], dtype=torch.long)
    output = model(dummy_input, lengths=dummy_lengths)

    assert output.shape[0] == batch_size
    assert output.shape[1] == time_steps // 4
    assert output.shape[2] == n_class

def test_architecture_registry():
    from src.config import PipelineConfig
    config = PipelineConfig()

    for name in ARCHITECTURES:
        assert subsampling_factor(name) in (2, 4)
        model = build_architecture(name, n_class=35, config=config)
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "subsampling_factor")

