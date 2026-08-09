# Conformer CTC Architecture

## Overview

The Conformer CTC implementation in SesaML (`src/models/conformer.py`) is a Convolution-augmented Transformer encoder paired with a Connectionist Temporal Classification (CTC) head. It leverages `torchaudio.models.Conformer` for high-accuracy streaming and offline speech recognition.

## Architecture Specification

- **Input**: Log-Mel Spectrogram features ($B \times 1 \times T \times F$, where $F = 128$).
- **Subsampling Front-End (`ConvSubsampling`)**:
  - Two stacked 2D convolutions (stride 2 each) reducing time sequence length by **4×** ($T \to T/4$).
  - Linear projection to the model embedding dimension ($D_{model}$).
- **Conformer Encoder Layers**:
  - Macaron-style Feed-Forward Modules (FFN with Swish activation and Half-Step residual connections).
  - Multi-Head Self-Attention (MHSA) with relative positional encodings.
  - Depthwise Separable Convolutional Modules (1D Depthwise Conv + BatchNorm + Swish + Pointwise Conv).
- **Configurations Supported**:
  - `conformer` (Small): $D_{model}=256$, 4 attention heads, 16 depthwise conv kernel size, 12 layers (~10M parameters).
  - `conformer-medium` (Medium): $D_{model}=512$, 8 attention heads, 31 depthwise conv kernel size, 16 layers (~30M parameters).
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

- **Length Alignment**: The DataLoader collator must calculate CTC `input_lengths = spec_len // 4` to match the 4× subsampling stride.
- **Variable Length Masking**: Masking padding frames in multi-head attention is strictly required during batched training to prevent loss divergence.
