# SesaML — Measured Results

Every figure here was produced by this repository. Where something is unverified or a negative result, it says so.

Hardware: **NVIDIA RTX 4000 SFF Ada**, 20GB, 70W cap. Windows, Python 3.11, torch 2.9.0+cu130.

---

## 1. Headline

| | |
|---|---|
| **Best model** | Conformer-CTC, 8.4M parameters |
| **WER** | **0.510** |
| **CER** | **0.166** |
| Training data | 12.3 hours (Lagyamfi 3.2h + ghanaopendata 9.1h) |
| Validation | Lagyamfi test split, 259 clips, held out |
| Training time | 80 epochs × ~88s = **~2 hours** |

From a random initialisation, on twelve hours of audio, in two hours of training. CER 0.166 means roughly **five characters in six are correct**.

### What it actually produces

Corpus reference:
> Wɔbɛtumi akɔ dan a ɛtoa wɔn so no ne ne yɔnko…

Model output:
> ɔbɛtumi akɔdan a ɛtɔa wɔn nsono…

Recognisably the same sentence, with spelling errors. This is exactly the error profile an n-gram language model fixes, which is why KenLM rescoring is the highest-value next step.

---

## 2. Learning curve (`run-clean`)

| Epoch | train_loss | val_loss | WER | CER |
|---|---|---|---|---|
| 1 | 2.78 | 2.98 | 1.000 | 1.000 |
| 5 | 2.17 | 1.85 | 0.952 | 0.584 |
| 9 | 1.89 | 1.30 | 0.865 | 0.395 |
| 78 | 0.601 | 0.571 | 0.502 | 0.163 |
| 80 | 0.588 | 0.570 | 0.510 | 0.166 |

Validation loss tracks training loss to the end (0.588 vs 0.570) — **no overfitting**, so the model is data-limited, not capacity-limited. More audio should help more than a bigger model.

The WER 1.0 / CER 1.0 at epoch 1 is normal and expected: CTC first learns to emit blanks, then starts emitting characters.

---

## 3. Negative result: the 500-hour corpus

A second run (`run-big`) over **113.3 hours** — the clean corpora plus a 100h slice of `ghananlpcommunity/twi-health-asr-gemini-500hrs` — **failed to learn**.

| Epoch | train_loss | val_loss | WER | CER |
|---|---|---|---|---|
| 1 | 2.919 | 4.557 | 0.981 | 0.947 |
| 2 | 2.877 | 2.952 | 1.000 | 1.000 |
| 5 | 2.750 | 2.967 | 1.000 | 1.000 |

Loss plateaued; the model never escaped emitting blanks. **9× the data produced a strictly worse model than 12.3 hours.**

The health corpus is 59,291 clips of **exactly 30.00 seconds** with Gemini-generated transcriptions. Most likely cause is utterance length: CTC must align ~400 characters over 750 encoder frames from scratch. Remedies are in [`roadmap_next_steps.md`](roadmap_next_steps.md#1-make-the-500-hour-health-corpus-usable-blocked-needs-work) — fine-tune from the working checkpoint rather than from random weights, and segment the 30s clips into utterances.

**This is the useful slide.** More data is not automatically better; utterance length and label quality decide whether it can be used at all.

---

## 4. What made the working run possible

A prior 30-epoch run had reported a training loss of **exactly 0.0** and looked converged. It had learned nothing.

| Bug | Effect | Fix |
|---|---|---|
| Text-only CSV used as an audio manifest | Empty labels + synthetic noise. CTC loss → 0.0 because zero-length targets are minimised by emitting blanks | Columns auto-detected; missing ones are a hard error. **Fails in 0.1s instead of 50 minutes** |
| `torchaudio.load` broken | torchaudio 2.9 delegates I/O to `torchcodec`, which it does not depend on. Every audio read raised | soundfile-backed loader with a torchaudio fallback |
| Conformer crashed on ragged batches | torchaudio's mask is sized by `lengths.max()` but asserted against the input's time dim; collate floors while convolutions ceil | Trim to the longest real length |
| Blank CSV cells | pandas → `NaN` → the literal string `"nan"` as a 3-character CTC target | `keep_default_na=False` |
| Raw mel power into the encoder | Values spanning 0–2568 with no compression or normalization | Log-mel + per-utterance CMVN |
| 4 of 128 mel filters empty | 128 mel bands over a 201-bin FFT | `n_mels` 128 → 80 |
| DeepSpeech2 ignored `lengths` | BiGRU ran its backward pass over padding; measured activation drift **0.33** | `pack_padded_sequence` |
| `datasets` 4.x audio decoding | Requires torchcodec + FFmpeg; indexing *any* column raised | Decode container bytes with soundfile |
| Corpora re-downloaded | Training called `load_dataset()` while the download script wrote to `data/datasets/` | Prefer the local copy |
| Windows console cp1252 | Logging a Twi `ɛ`/`ɔ` would crash the run | UTF-8 with lossy fallback |
| `’` dropped from the vocabulary | Silently glued Twi elisions (`m’abankɛseɛ`) into one word | Fold typographic variants |

Test suite: **15 → 130 tests.**

---

## 5. Published model exports

Every finished training run is archived in the **model registry** under `outputs/registry/`, and the app, CLI and evaluation all serve the *promoted* version — never simply the newest.

### Current exports

| | Architecture | Version | WER | CER | Run | Training data |
|---|---|---|---|---|---|---|
| **→** | conformer | **v001** | **0.510** | **0.166** | `run-clean` | 12.3h (Lagyamfi + ghanaopendata), 80 epochs, bs 32, lr 8e-4 |

`→` marks the version being served. `run-big` (113h, WER 1.0) is deliberately **not** published as a serving version — see §3.

### Why promotion is separate from publishing

Checkpoints used to resolve by modification time, so the most recent run was served whether or not it was any good. Given §3 — where 9× the data produced a strictly worse model — that is a live hazard across a series of iterations.

The registry publishes every run but promotes one only if it **beats the current version's WER**. A collapsed run is archived for comparison and never reaches the demo. Concretely, publishing a WER 1.0 run against the current 0.51:

```
Published conformer/v002 (WER 1.0000)
Not promoted; conformer/v001 (WER 0.5100) remains current.
```

### Rollback

```bash
python -m src.main models list                                    # what exists, what is served
python -m src.main models rollback --architecture conformer       # back to the last good one
python -m src.main models promote --architecture conformer --version v002
```

Rollback follows **promotion history**, not version numbers, so it returns to the last version that was actually served. The Gradio app exposes the same switch in a **Model version** dropdown — the live escape hatch if a newly promoted model misbehaves mid-demo.

### What a version records

Each version stores its weights plus a `metadata.json`: WER/CER/val_loss, originating run id, SHA-256, size, and the **front-end settings it was trained on** (`sample_rate`, `n_mels`, `n_fft`, `hop_length`, `vocab_size`, `subsampling_factor`).

That last group is not bookkeeping. Loading 80-mel weights under a 128-mel config does **not** raise — the shapes are set by the architecture, not the front-end — it silently degrades transcription. The registry compares them and warns in both the CLI and the app.

In git: `registry.json` and every `metadata.json` are **tracked**; the `.pt` weights are not. A clone therefore carries the full provenance of which model was served when, without the binaries.

---

## 6. Throughput and hardware

| Measurement | Value |
|---|---|
| Training throughput (conformer, bs=32) | 555 batches/epoch, ~88s/epoch |
| GPU memory at bs=32 | 12.1 GB of 20.5 GB |
| GPU power | **69.9W of a 70W cap**, 98% utilisation |
| Padding efficiency, unbucketed | 20% |
| Padding efficiency, bucketed | 97% (99% on the mixed run) |

The card is **power-limited, not memory-limited**. Raising the batch size past 32 buys almost no throughput and risks OOM — bs=64 would need ~24GB. This is the number to quote when asked "why not a bigger batch?"

### Corpora

| Corpus | Clips | Hours | Clip length | Transcript column |
|---|---|---|---|---|
| Lagyamfi/akan_audio_processed (train) | 2,187 | 3.2 | 2.4–8.8s | `sentence` |
| Lagyamfi/akan_audio_processed (test) | 259 | 0.3 | 2.2–11.5s | `sentence` |
| ghanaopendata/twi-speech-text-multispeaker-16k | 15,560 | 9.1 | 0.04–7.0s | `text` |
| ghananlpcommunity/twi-health-asr-gemini-500hrs | 59,291 | 494.1 | 30.00s exactly | `transcription` |

Vocabulary coverage is ~96% on all four; the ~4% dropped is **entirely punctuation**, which is not spoken and correctly excluded from CTC targets.

---

## 7. Speaker diarization

Three backends on a 28.3s Akan clip assembled from six corpus recordings:

| Backend | Speakers found | Turns | Load | Runtime |
|---|---|---|---|---|
| pyannote | 3 | 7 | 4s | 6.9s |
| ecapa (thr 0.90) | 3 | 7 | 66s first run | 7.1s |
| ecapa (thr 0.75) | 5 | 8 | — | — |
| ecapa (thr 0.60) | 6 | 8 | — | — |
| spectral | 1 | 4 | instant | 0.01s |

pyannote and ECAPA-at-0.90 agree independently, which is what set the default threshold. VAD boundaries land within **0.1s** of ground truth on a synthetic 4-region signal.

**Caveat for the slide:** the corpus clips carry no speaker labels, so "3 speakers" is agreement between two backends, not verified truth. Diarization Error Rate is not measured — no labelled multi-speaker Akan audio exists in these corpora.

Full end-to-end output is in [`speaker_diarization.md`](speaker_diarization.md#7-measured-end-to-end-result).

---

## 8. Reproducing

```bash
# The working model
scripts/train.sh --architecture conformer \
  --hf-dataset Lagyamfi/akan_audio_processed:train \
  --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k:train \
  --val-dataset Lagyamfi/akan_audio_processed:test \
  --epochs 80 --batch-size 32 --lr 8e-4

# Evaluate
python -m src.main evaluate --architecture conformer \
  --hf-dataset Lagyamfi/akan_audio_processed:test \
  --model-path outputs/checkpoints/run-clean/best_model.pt

# Speaker-attributed transcript
python -m src.main diarize --audio recording.wav --backend auto

# Model versions: what is published, what is served, and how to fall back
python -m src.main models list
python -m src.main models rollback --architecture conformer

# Demo app
scripts/serve_app.sh
```

Continue a run with `--resume outputs/checkpoints/<run_id>/last_model.pt`.
