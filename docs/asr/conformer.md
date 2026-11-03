# Conformer CTC Architecture

## Overview

The Conformer CTC implementation in SesaML (`src/asr/models/conformer.py`) is a Convolution-augmented Transformer encoder paired with a Connectionist Temporal Classification (CTC) head. It leverages `torchaudio.models.Conformer` for high-accuracy streaming and offline speech recognition.

## Architecture Specification

- **Input**: Log-mel spectrogram, `(batch, 1, n_mels, time)`. Trained models use **80** mel bands (`AudioConfig.n_mels`). The constructor's default of 128 is not what `build_architecture` passes.
- **Subsampling Front-End (`ConvSubsampling`)**:
  - Two stacked 2D convolutions (stride 2 each) reducing time sequence length by **4×** ($T \to T/4$).
  - Linear projection to the model embedding dimension (`encoder_dim`).
- **Conformer Encoder Layers** (`torchaudio.models.Conformer`):
  - Macaron-style feed-forward modules.
  - Multi-head self-attention with relative positional encodings.
  - Depthwise separable convolution, kernel size 31.
- **Configurations that have been trained**:
  - `conformer` (small): `encoder_dim` 144, 16 layers, 4 heads, FFN 576. **8,380,409** parameters.
  - `conformer-medium`: `encoder_dim` 256, 16 layers, 4 heads, FFN 1024. **26,281,257** parameters.

The served model is `conformer-medium/v003`. Measured test numbers are in [`results.md`](results.md#10-the-path-under-030).
- **Length-Aware Attention Masking**:
  - Accepts `input_lengths` tensor to dynamically mask padding frames, preventing self-attention layers from attending to padded frames.

## Design Rationale & Why We Selected It

1. **State-of-the-Art Hybrid Architecture**:
   - Conformer combines the global context learning of Self-Attention with the local feature extraction capabilities of Convolution.
2. **Computational Efficiency (4× Subsampling)**:
   - By downsampling the time sequence length by 4× upfront, a 30-second audio clip (~3000 spectrogram frames) is reduced to ~375 encoder frames. This cuts self-attention quadratic compute ($O(T^2)$) by **16×**, allowing memory-efficient training on long clips.
3. **Superior Acoustic Representation**:
   - Depthwise separable convolutions excel at capturing local shift-invariant speech patterns (phoneme transitions), making it highly effective for tonal and agglutinative languages like Akan (Twi).

## Trade-offs & Considerations

- **Length Alignment**: The DataLoader collator must calculate CTC `input_lengths = spec_len // 4` to match the 4× subsampling stride. The model then trims to the longest real length, because the convolutions can round the frame count the other way.
- **Variable Length Masking**: Masking padding frames in multi-head attention is strictly required during batched training to prevent loss divergence.
