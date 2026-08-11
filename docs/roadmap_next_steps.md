# SesaML Roadmap & Next Steps

Planned technical improvements and architectural milestones for SesaML on Akan (Twi) Automatic Speech Recognition.

Items marked **Completed** were implemented and measured; see [`docs/results.md`](results.md) for the numbers.

---

## Phase 0: Completed

### 1. Length-Bucketed Batching — *completed*
Batches clips of similar duration together ([`src/data/bucketing.py`](../src/data/bucketing.py)). Measured on the real corpus length mixture, the fraction of batched frames carrying audio rather than padding rose from **20% to 97%**; on the mixed 113-hour run the training log reported **99% vs 88% shuffled**.

The estimate in the previous draft ("~19% compute waste") was **backwards** — waste was ~80%, because ghanaopendata's median clip is under a second while the health corpus is uniformly 30 seconds. Mixing them at random pads nearly everything to 30s.

### 2. Mixed-Precision Training — *completed*
CUDA AMP with `GradScaler` ([`src/training/trainer.py`](../src/training/trainer.py)). The encoder runs in fp16; `log_softmax` and the CTC loss stay in fp32, which is where the numerical sensitivity lives. Gradients are unscaled before clipping.

### 3. Log-Mel Features & Per-Utterance Normalization — *completed*
The front-end previously fed **raw mel power** (measured range 0–2568, mean 233) straight into the encoder. It now applies log compression and per-utterance CMVN ([`src/data/audio_transforms.py`](../src/data/audio_transforms.py)), with SpecAugment last so masks fill with the feature mean.

`n_mels` also dropped 128 → 80: against a 400-point FFT (201 bins), 128 mel filters left **four filterbanks entirely empty**.

### 4. Speaker Diarization — *completed*
Three backends, colour-coded Gradio tab, `diarize` CLI command. See [`docs/speaker_diarization.md`](speaker_diarization.md).

### 5. Training Reliability — *completed*
Resume-from-checkpoint, corpus probes that abort on a bad dataset before epoch 1, gradient clipping, `zero_infinity` CTC, and offline corpus resolution.

### 6. Versioned Model Registry — *completed*
Every finished run is archived under `outputs/registry/` and promoted only if it beats the served version's WER ([`src/utils/model_registry.py`](../src/utils/model_registry.py)). Checkpoints previously resolved by modification time, which — given that a 113-hour run produced a *worse* model than a 12-hour one — meant one bad iteration would silently replace a working demo.

Rollback follows promotion history, and is exposed on the CLI (`models rollback`) and as a dropdown in the app. Versions also record the front-end settings they were trained on, because loading 80-mel weights under a 128-mel config degrades silently rather than raising.

---

## Phase 1: Scaled Training & Optimization

### 1. Make the 500-Hour Health Corpus Usable — *blocked, needs work*
A 5-epoch run over 113 hours (100h of it from the health corpus) **failed to learn**: WER stayed at 1.0 and training loss plateaued at ~2.75. See [`docs/results.md`](results.md).

The corpus is 59,291 clips of **exactly 30.00 seconds** with Gemini-generated transcriptions. Two plausible causes, in order of suspicion:

- **Utterance length.** CTC must align ~400 characters over 750 encoder frames from a random initialisation. Long-form CTC training normally needs either curriculum learning (short clips first) or segmentation into utterances.
- **Label quality.** Pseudo-labels may not align with the audio well enough to train from scratch.

Next steps, cheapest first:
1. **Fine-tune, don't train from scratch** — start from the `run-clean` checkpoint (WER 0.51) rather than random weights.
2. **Segment the 30s clips** into utterances with the VAD already written for diarization ([`src/diarization/segmentation.py`](../src/diarization/segmentation.py)), producing short clips the model can align.
3. **Verify a sample of the pseudo-labels** against the audio before spending more GPU time.

### 2. Akan Text Preprocessing & Number Normalization
Convert numeric digits (`2 capsules` → `capsules mmienu`) to spelled-out Twi. The model currently must map the *sound* of a Twi number to an ASCII digit.

Partially addressed: typographic variants are now folded (`’` → `'`), which had been silently gluing elisions like `m’abankɛseɛ` into one word. Corpus probes report that ~4% of characters are dropped, and confirm they are **all punctuation** — legitimately not spoken.

### 3. N-Gram Language Model & Beam Search Decoding
Integrate a KenLM 4-gram model over Twi Wikipedia and Bible text into the CTC decoder. **Highest-expected-value item on this list**: the model decodes greedily with no lexicon, so its errors are overwhelmingly plausible-sounding misspellings that a language model would fix.

### 4. Background Noise Data Augmentation
On-the-fly SNR mixing with MUSAN or environmental noise, for clinical, market, and field recordings.

---

## Phase 2: Production Deployment & Edge Optimization

### 1. ONNX & ExecuTorch Quantization
Export Conformer to ONNX Runtime and INT8 ExecuTorch (`.pte`) for on-device offline transcription.

### 2. Silero VAD in Production Serving
The energy-based VAD written for diarization works and is dependency-free, but a neural VAD is more robust on noisy field audio.

### 3. Multi-GPU / DDP
`DistributedDataParallel` in the trainer. Lower priority than it looks: the current RTX 4000 Ada is **power-limited, not memory-limited** (69.9W of a 70W cap at 98% utilisation with 12GB of 20GB used), so a second GPU helps more than a bigger batch on one.

### 4. Diarization Error Rate Evaluation
Every diarization claim is currently qualitative — none of the three corpora carry speaker labels. A small hand-labelled multi-speaker Akan set would make DER measurable.

---

## Summary Milestone Checklist

| Milestone | Target | Priority | Status |
| :--- | :--- | :--- | :--- |
| **Conformer CTC Architecture** | Multi-head attention + 4× subsampling | High | Completed |
| **Apple Silicon MPS Support** | Auto-detected device selection | High | Completed |
| **Cross-Platform Installers** | `ensure_python.sh` & `ensure_python.ps1` | High | Completed |
| **Log-Mel + CMVN Front-End** | Compressed, normalized features | High | Completed |
| **Length-Bucketed DataLoader** | 20% → 97% padding efficiency | High | Completed |
| **Mixed-Precision Training** | CUDA AMP, fp32 loss | High | Completed |
| **Resume From Checkpoint** | Optimizer + schedule state | High | Completed |
| **Corpus Probes / Fail-Fast** | Abort on a bad dataset in 0.1s | High | Completed |
| **Speaker Diarization Pipeline** | 3 backends + Gradio + CLI | High | Completed |
| **Versioned Model Registry** | Promote-on-improvement + rollback | High | Completed |
| **First Trained Model** | WER 0.51 / CER 0.166 on 12.3h (`conformer/v001`) | High | Completed |
| **500h Health Corpus Training** | Fine-tune or segment 30s clips | High | Blocked — see Phase 1.1 |
| **KenLM CTC Beam Search Decoder** | Language model rescoring | High | Planned |
| **Noise Augmentation Training** | On-the-fly MUSAN SNR mixing | Medium | Documented |
| **Twi Number Normalization** | Digits → spelled-out Twi | Medium | Partial |
| **DER Evaluation Set** | Labelled multi-speaker Akan audio | Medium | Planned |
