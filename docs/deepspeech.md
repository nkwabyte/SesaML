# DeepSpeech2 CTC Architecture

## Overview

The DeepSpeech2 implementation in SesaML (`src/models/deepspeech.py`) is a Convolutional-Recurrent Neural Network with a Connectionist Temporal Classification (CTC) loss function, inspired by Baidu's DeepSpeech 2 architecture.

## Architecture Specification

- **Input**: Log-Mel Spectrogram features ($B \times 1 \times T \times F$, where $F = 128$ mel frequency bins).
- **Subsampling Convolutional Layers**:
  - 2D Conv layer (kernel $3 \times 3$, stride $2 \times 2$, padding 1) + BatchNorm + Hardtanh
  - 2D Conv layer (kernel $3 \times 3$, stride $2 \times 1$, padding 1) + BatchNorm + Hardtanh
  - **Subsampling factor**: Reduces time frames by **2×** ($T \to T/2$).
- **Recurrent Encoder**:
  - 5 stacked Bidirectional GRU (BiGRU) layers ($D=512$ hidden units per direction).
  - Batch normalization across sequence time frames.
- **Classification Head**:
  - Linear projection layer mapping hidden dimension ($2 \times 512 = 1024$) to the 41-token vocabulary size (Akan characters $a\text{--}z, \varepsilon, \jmath, 0\text{--}9$, space, apostrophe, blank).

## Design Rationale & Why We Selected It

1. **Lightweight Baseline**:
   - Total parameter count: **~23 Million parameters**.
   - Trains rapidly even on modest GPU compute and Apple Silicon (`mps`).
2. **Fast Iteration & Debugging**:
   - Because the time sequence length is only reduced by 2× and GRU operations are lightweight, pipeline bugs, collator logic, and dataset preprocessing can be validated in minutes rather than hours.
3. **Robust CTC Alignment**:
   - CTC loss handles unaligned speech input natively without needing frame-level forced alignment transcripts.

## Trade-offs & Limitations

- **Recurrent Bottleneck**: BiGRU layers lack the long-range parallel attention mechanisms of Transformers.
- **Sequence Length**: 2× subsampling means long audio clips (e.g. 30-second clips from the health corpus) produce relatively long hidden sequences (~750 frames), causing high GPU memory usage during training.
