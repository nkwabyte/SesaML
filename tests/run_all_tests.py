import unittest
import torch
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.text_transform import TextTransform
from src.data.audio_transforms import get_train_audio_transforms, get_valid_audio_transforms
from src.data.dataset import AkanAudioDataset, data_processing
from src.models.deepspeech import SpeechRecognitionModel
from src.training.evaluator import greedy_decoder, Evaluator
from src.utils.metrics import calculate_wer, calculate_cer

class TestSesaML(unittest.TestCase):

    def test_text_transform(self):
        tt = TextTransform()
        text = "ɛdeɛn na ɛrekɔso"
        encoded = tt.text_to_int(text)
        decoded = tt.int_to_text(encoded)
        self.assertIn("ɛ", decoded)
        self.assertIn("ɔ", decoded)
        self.assertEqual(decoded, text)

    def test_metrics(self):
        ref = "ɛdeɛn na ɛrekɔso"
        hyp = "ɛdeɛn na ɛrekɔso"
        self.assertEqual(calculate_wer(ref, hyp), 0.0)
        self.assertEqual(calculate_cer(ref, hyp), 0.0)

        hyp_diff = "ɛdeɛn na birekɔso"
        self.assertGreater(calculate_wer(ref, hyp_diff), 0.0)

    def test_audio_transforms(self):
        train_tf = get_train_audio_transforms(16000, 128)
        valid_tf = get_valid_audio_transforms(16000, 128)

        waveform = torch.randn(1, 16000)
        train_spec = train_tf(waveform)
        valid_spec = valid_tf(waveform)

        self.assertEqual(train_spec.shape[1], 128)
        self.assertEqual(valid_spec.shape[1], 128)

    def test_deepspeech_model(self):
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

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], n_class)

    def test_greedy_decoder(self):
        probs = torch.randn(5, 2, 35)  # (time, batch, class)
        decoded = greedy_decoder(probs, blank_label=34)
        self.assertEqual(len(decoded), 2)
        self.assertIsInstance(decoded[0], list)

    def test_hf_dataset_mock(self):
        from src.data.hf_dataset import HuggingFaceAkanDataset
        mock_hf_data = [
            {"audio": {"array": [0.1, -0.2, 0.3], "sampling_rate": 16000, "path": "test.wav"}, "text": "Medaase"}
        ]
        ds = HuggingFaceAkanDataset(hf_dataset=mock_hf_data)
        self.assertEqual(len(ds), 1)
        waveform, sr, text, path = ds[0]
        self.assertEqual(sr, 16000)
        self.assertEqual(text, "Medaase")
        self.assertEqual(path, "test.wav")

if __name__ == "__main__":
    unittest.main()
