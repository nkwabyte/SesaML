# Noise Robustness Architecture & Strategy

Real-world deployment environments (clinical health centers, market chatter, street traffic, mobile phone microphones) contain significant ambient background noise. This document details the dual-strategy in SesaML for handling noise during **training** and **production inference**.

---

## 1. Training Pipeline: On-the-Fly Noise Augmentation

This section is a design. The SNR mixer is not in `src/asr/data/audio_transforms.py`. The train front-end that actually runs is log-mel, per-utterance CMVN, then SpecAugment, and, since 2026-10-31, a speed perturbation at 0.9 / 1.0 / 1.1 that was measured and did not beat the served model ([`results.md` §10](results.md#10-the-path-under-030)).

Training the model exclusively on clean studio audio causes severe performance degradation when deployed in noisy environments. The sketch below is how real-world noise would be injected on the fly.

```
                           Clean Audio Waveform
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Random SNR Mixing (5-20dB) │◄─── Background Noise Corpus
                     └──────────────┬───────────────┘     (MUSAN / RIR / Environmental)
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Mel-Spectrogram & SpecAug  │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │    Conformer / DeepSpeech    │
                     └──────────────────────────────┘
```

### Proposed Training Augmentation (`src/asr/data/audio_transforms.py`)

```python
import torch
import torchaudio.transforms as T

class NoiseAugmentation:
    """Adds background noise to speech waveforms at random SNRs during training."""
    def __init__(self, noise_dataset_path: str, min_snr_db: float = 5.0, max_snr_db: float = 20.0, p: float = 0.5):
        self.noise_dataset_path = noise_dataset_path
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return waveform

        # 1. Load random noise clip (e.g. street, market, hospital chatter)
        noise_waveform = self._load_random_noise_clip(waveform.size(-1))
        
        # 2. Pick random SNR
        snr_db = torch.empty(1).uniform_(self.min_snr_db, self.max_snr_db).item()
        
        # 3. Mix speech and noise at calculated SNR
        clean_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)
        scale = (clean_power / (noise_power + 1e-8)) * (10 ** (-snr_db / 20.0))
        
        return waveform + scale * noise_waveform
```

### Benefits
- **Native Invariance**: The model learns to ignore non-speech frequencies and transient background clicks automatically.
- **No Extra Inference Latency**: Robustness is baked into the neural network weights—no extra pre-processing delays during live transcription.

---

## 2. Production Inference Pipeline: Noise Filtering & VAD

For live production (Gradio Web UI, Django REST API, mobile edge), audio is passed through a lightweight noise suppression and Voice Activity Detection (VAD) stage prior to ASR & Diarization.

```
Raw Field Audio ──► [ 1. Silero VAD (Trim Silence/Hum) ] ──► [ 2. DeepFilterNet / Spectral Gate ] ──► [ 3. SesaML ASR ]
```

### Component Breakdown

1. **Silero Voice Activity Detection (VAD)**:
   - Strips non-speech audio (applause, air conditioner hum, background silence) before processing.
   - Prevents CTC greedy decoder from outputting false character insertions on quiet background static.

2. **Spectral Gating (`src/utils/noise_reduction.py`)**:
   - Built-in spectral gate noise suppressor (activated via `--noise-reduction` flag in `transcribe.sh` / CLI).
   - Fast, zero-dependency pre-filter that estimates noise floor over non-speech frames and attenuates stationary noise.

3. **Deep Learning Speech Enhancement (DeepFilterNet)**:
   - For severe non-stationary background noise (clapping, machinery, overlapping voices), a lightweight DeepFilterNet / RNNoise model runs as a pre-filtering stage in the API.

---

## 3. Summary of Recommendations

| Pipeline Stage | Technique | Objective | Status |
| :--- | :--- | :--- | :--- |
| **Training** | `SpecAugment` (Time/Freq Masking) | Prevents overfitting to specific spectral features | Supported (`audio_transforms.py`) |
| **Training** | Speed perturbation 0.9 / 1.0 / 1.1, waveform only, train front-end only | Measured on `p7-speed` (`conformer-medium/v006`). Lagyamfi test beam+LM 0.2920 against the served model's 0.2887. Not part of the served weights | In the train transform by default. See [`results.md`](results.md#10-the-path-under-030) |
| **Training** | On-the-Fly Noise Injection (MUSAN) | The design in §1. It is not implemented in `audio_transforms.py` | Planned |
| **Inference** | Spectral Gating Noise Reduction | Removes stationary background hum/hiss | Supported (`--noise-reduction`) |
| **Inference** | Silero Voice Activity Detection (VAD) | Trims silence & prevents false text insertions | Planned |
