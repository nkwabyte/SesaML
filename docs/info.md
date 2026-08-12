# Documentation Index (`docs/`)

This directory contains technical documentation, architecture deep dives, deployment blueprints, and future development roadmaps for SesaML.

---

## Model Documentation

- **[`deepspeech.md`](asr/deepspeech.md)**: DeepSpeech2 CNN + BiGRU CTC architecture design, parameter footprint, and baseline selection rationale.
- **[`conformer.md`](asr/conformer.md)**: Conformer Convolution-augmented Transformer architecture, 4× subsampling, self-attention masking, and efficiency design.
- **[`whisper.md`](asr/whisper.md)**: Fine-tuned Whisper encoder-decoder Transformer, pre-trained multilingual transfer learning, and benchmarking rationale.

---

## Systems & Features

- **[`speaker_diarization.md`](asr/speaker_diarization.md)**: System design and code implementation for speaker diarization ("Who Spoke When") integrated into the Gradio web UI or a production Django REST API backend.
- **[`noise_robustness.md`](asr/noise_robustness.md)**: Strategy for real-world background noise handling: on-the-fly SNR noise augmentation during training vs production pre-filtering (Silero VAD & spectral gating).
- **[`roadmap_next_steps.md`](roadmap_next_steps.md)**: Technical roadmap for full 500-hour health corpus training, length-bucketed batching, text normalization, KenLM beam search CTC decoding, and ONNX/ExecuTorch deployment.
