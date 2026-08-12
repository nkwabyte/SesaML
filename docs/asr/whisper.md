# Whisper Fine-Tuning Architecture

## Overview

The Whisper integration in SesaML (`src/asr/models/whisper_model.py`) wraps OpenAI's Whisper encoder-decoder Transformer models via HuggingFace `transformers`. It allows fine-tuning and inference on pre-trained multilingual Whisper checkpoints (such as `openai/whisper-small`).

## Architecture Specification

- **Encoder**: 80-channel Mel Spectrogram input passed through a 2D convolutional stem followed by a Transformer encoder stack.
- **Decoder**: Autoregressive Transformer decoder predicting text tokens using cross-attention over encoder representations.
- **Task & Language Prompting**:
  - Configurable language prompt (e.g. `<|sn|>` for Akan/Twi or `<|en|>`).
  - Task prompt (`transcribe` vs `translate`).

## Design Rationale & Why We Selected It

1. **Multilingual Transfer Learning**:
   - Pre-trained on 680,000 hours of weakly supervised audio, Whisper brings strong acoustic priors and robust noise tolerance.
2. **Autoregressive Decoding**:
   - Unlike CTC models which assume conditional independence between output frames, Whisper's Transformer decoder learns explicit language model transitions, producing coherent punctuation and capitalization.
3. **Benchmarking & Comparison**:
   - Serving fine-tuned Whisper checkpoints alongside custom-trained CTC models (`deepspeech`, `conformer`) provides a gold-standard baseline to evaluate WER, latency, and model size trade-offs.

## Trade-offs & Considerations

- **Inference Latency**: Autoregressive decoding token-by-token is significantly slower than greedy CTC decoding.
- **Hallucinations**: Decoder-based models can hallucinate text on background noise or non-speech audio segments.
- **Model Footprint**: Requires larger memory footprint (e.g. Whisper Small is ~244M parameters vs Conformer Small at ~10M parameters).
