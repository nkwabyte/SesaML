# Documentation Index (`docs/`)

This directory contains technical documentation, architecture deep dives, deployment blueprints, and future development roadmaps for SesaML.

---

## Measured results

- **[`results.md`](asr/results.md)**: Every WER and CER the repository has measured. The served model is `conformer-medium/v003` at **0.2887** beam+LM word error rate on the Lagyamfi test set. §10 is the run-by-run record through the speed-perturbation result on 2026-10-31.
- **[`court_flow.md`](asr/court_flow.md)**: The product is customary-court speech. Lagyamfi stays a regression check. This is the order of work after 0.2887, and the implementation plan for it. Status is kept in [`court_flow_scratchpad.md`](asr/court_flow_scratchpad.md).
- **[`ghanaopenai_twi_multispeaker_tts.md`](asr/ghanaopenai_twi_multispeaker_tts.md)**: How to add `ghanaopenai/twi-multispeaker-tts` (91 h, not downloaded) to the training mix. Not a replacement for any current corpus. Not started.

## Model Documentation

- **[`deepspeech.md`](asr/deepspeech.md)**: DeepSpeech2 CNN + BiGRU CTC architecture design, parameter footprint, and baseline selection rationale.
- **[`conformer.md`](asr/conformer.md)**: Conformer Convolution-augmented Transformer architecture, 4× subsampling, self-attention masking, and efficiency design.
- **[`whisper.md`](asr/whisper.md)**: Fine-tuned Whisper encoder-decoder Transformer, pre-trained multilingual transfer learning, and benchmarking rationale.

---

## Systems & Features

- **[`speaker_diarization.md`](asr/speaker_diarization.md)**: System design and code implementation for speaker diarization ("Who Spoke When") integrated into the Gradio web UI or a production Django REST API backend.
- **[`noise_robustness.md`](asr/noise_robustness.md)**: What the train front-end actually does (SpecAugment, and speed perturbation, which was measured and did not help), the noise-injection design that is not built yet, and the spectral gate used at inference.
- **[`roadmap_next_steps.md`](roadmap_next_steps.md)**: What is already measured, what is still blocked (the 500-hour health corpus), and what is left after the 0.30 result.
