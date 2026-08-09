# Source

The SesaML pipeline: audio in, Akan text out.

```
src/
├── config.py            all tunables — audio, model, training, paths, device auto-detection
├── main.py              CLI: train / evaluate / transcribe / export
├── data/
│   ├── text_transform.py   char ↔ int mapping for CTC (a-z, 0-9, ɛ, ɔ, ', space)
│   ├── audio_transforms.py MelSpectrogram, SpecAugment masking
│   ├── dataset.py          CSV-backed dataset + collate function
│   └── hf_dataset.py       HuggingFace corpora, multi-corpus concatenation
├── models/
│   ├── __init__.py         architecture registry (build_architecture, subsampling_factor)
│   ├── conformer.py        Conformer-CTC encoder (attention + depthwise conv)
│   ├── deepspeech.py       DeepSpeech2-style CNN + BiGRU CTC model
│   └── whisper_model.py    wrapper around a fine-tuned HF Whisper
├── training/
│   ├── trainer.py          training loop, length-aware forwarding, checkpointing, metric logging
│   └── evaluator.py        CTC loss, length-aware forwarding, greedy decoding, WER/CER
├── inference/
│   ├── transcribe.py       Transcriber (cached) + transcribe_audio (auto-detects architecture)
│   └── export.py           TorchScript / state dict / ExecuTorch export
└── utils/
    ├── metrics.py          WER/CER (jiwer, with a pure-Python fallback)
    ├── noise_reduction.py  optional spectral gate pre-processing
    └── run_logger.py       RunManager — outputs/ contract, save_model_meta, load_model_meta
```

## Entry point

```bash
python -m src.main train --architecture conformer --device mps
python -m src.main evaluate --architecture conformer --device mps
python -m src.main transcribe --audio clip.wav --model-type deepspeech --architecture conformer
python -m src.main export --architecture conformer --format torchscript
```

The [scripts/](../scripts/info.md) wrappers are the friendlier way in — they
handle the venv, `.env` and sensible defaults.

## Things worth knowing

**`run_logger.py` is the spine.** Every command opens a `RunManager`, which owns
the run directory, the logger and the metric files. If you add a command, open a
run for it too — that is what keeps `outputs/` complete. See
[outputs/info.md](../outputs/info.md).

**Architecture registry & `model_meta.json`.** `build_architecture()` builds CTC models by name (`deepspeech`, `conformer`, `conformer-medium`). Checkpoints save `model_meta.json` in their directory so `load_deepspeech_model()`, evaluation, export, and the web app auto-detect the architecture without manual configuration. Time subsampling factors (2× for DeepSpeech, 4× for Conformer) are derived via `subsampling_factor(arch)` so CTC sequence lengths align exactly.

**Apple Silicon MPS & Device Auto-Detection.** `PipelineConfig` automatically selects `mps` (Metal) on Apple Silicon Macs, `cuda` on NVIDIA GPUs, or `cpu`. The `--device` flag overrides this per command.

**The vocabulary is small and fixed.** `TextTransform` covers `a-z`, `0-9`,
apostrophe, and the Akan characters `ɛ` and `ɔ`, plus space and a CTC blank —
41 symbols. Characters outside it (including punctuation) are dropped during
encoding, which is what you want for ASR. Uppercase `Ɔ`/`Ɛ` lowercase correctly.

Digits are in the vocabulary because the health corpus writes dosages and dates
numerically; dropping them would leave the audio saying a number that the label
does not contain, which trains the model to skip speech. Note the flip side: the
model must now learn that the sound of *mmienu* (or English *two*) maps to `2`.
Spelling numbers out in Twi during preprocessing would be cleaner still.
`include_digits=False` restores the 31-symbol vocabulary, and letter indices are
identical either way — digits are appended last.

**Text columns differ across corpora.** `HuggingFaceAkanDataset` auto-detects
`text` / `sentence` / `transcription` / `transcript` and raises if none is
present, rather than yielding empty labels that would silently poison CTC
training.

**Checkpoints resolve automatically.** `resolve_checkpoint()` picks the newest
`outputs/checkpoints/<run_id>/best_model.pt` unless you pass an explicit path.

**Two ways to transcribe.** `transcribe_audio()` loads weights per call — fine
for the CLI. `Transcriber` holds a loaded model; use it for anything that
transcribes repeatedly, like the Gradio app.
