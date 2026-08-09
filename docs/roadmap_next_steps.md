# SesaML Roadmap & Next Steps

This document outlines the planned technical improvements and architectural milestones for SesaML on Akan (Twi) Automatic Speech Recognition.

---

## Phase 1: Scaled Training & Optimization

### 1. Full 500-Hour Health Corpus Training
- **Goal**: Train `conformer-medium` on the combined 500+ hour dataset (`ghananlpcommunity/twi-health-asr-gemini-500hrs` + `ghanaopendata` + `Lagyamfi`).
- **Optimization Needed**:
  - Implement **Length-Bucketed Batching** (`src/data/dataset.py`) to group clips of similar duration in each batch, eliminating ~19% compute waste caused by padding 4s clips to 30s.
  - Mixed-precision FP16 / BF16 training via PyTorch `torch.cuda.amp` or `torch.mps` autocast.

### 2. Akan Text Preprocessing & Number Normalization
- **Goal**: Add a number-to-words normalizer (`src/data/text_transform.py`) to convert numeric digits (`2 capsules` $\to$ `capsules mmienu`) into spelled-out Twi words.
- **Impact**: Removes ambiguity where the model must map the acoustic sound of Twi numbers to ASCII digit symbols.

### 3. N-Gram Language Model & Beam Search Decoding
- **Goal**: Integrate a KenLM or PyTorch 4-gram language model trained on Twi Wikipedia & Bible text corpora into the CTC greedy decoder.
### 4. Background Noise Data Augmentation
- **Goal**: Add on-the-fly SNR noise mixing (`src/data/audio_transforms.py`) using background noise datasets (e.g. MUSAN / environmental noise).
- **Impact**: Forces the model to learn noise-invariant acoustic representations natively, maintaining low WER in clinical, market, and field environments.

---

## Phase 2: Production Deployment & Edge Optimization

### 1. ONNX & ExecuTorch Model Quantization
- **Goal**: Export `conformer` models to ONNX Runtime and INT8 ExecuTorch (`.pte`) formats.
- **Target**: Real-time on-device offline transcription for mobile devices (iOS / Android) and low-power embedded hardware.

### 2. Silero VAD & Production Speech Enhancement
- **Goal**: Integrate Silero Voice Activity Detection (VAD) and spectral noise suppression into production serving (`app/app.py` & REST API).
- **Target**: Trims non-speech background hum/silence before ASR, preventing false CTC character insertions.

### 3. Multi-GPU & Distributed Data Parallel (DDP)
- **Goal**: Update `src/training/trainer.py` to support `torch.nn.parallel.DistributedDataParallel` for multi-GPU training on Hetzner / AWS cloud instances.

---

## Summary Milestone Checklist

| Milestone | Target | Priority | Status |
| :--- | :--- | :--- | :--- |
| **Conformer CTC Architecture** | Multi-head attention + 4× subsampling | High | Completed |
| **Apple Silicon MPS Support** | Auto-detected device selection | High | Completed |
| **Cross-Platform Installers** | `ensure_python.sh` & `ensure_python.ps1` | High | Completed |
| **Length-Bucketed DataLoader** | Minimize padding waste on 30s clips | High | Planned |
| **Noise Augmentation Training** | On-the-fly MUSAN SNR mixing | High | Documented |
| **KenLM CTC Beam Search Decoder** | Language model rescoring | Medium | Planned |
| **Speaker Diarization Pipeline** | `pyannote.audio` + Django / Gradio | Medium | Documented |
