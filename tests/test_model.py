import torch
import pytest
from src.models.deepspeech import SpeechRecognitionModel

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

    # Dummy spectrogram input: (batch, channel=1, n_mels, time)
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    output = model(dummy_input)

    # Output shape should be (batch_size, downsampled_time, n_class)
    assert output.shape[0] == batch_size
    assert output.shape[2] == n_class
