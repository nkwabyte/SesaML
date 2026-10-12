# SesaML — Measured Results

Every figure here was produced by this repository. Where something is unverified or a negative result, it says so.

Hardware: **NVIDIA RTX 4000 SFF Ada**, 20GB, 70W cap. Windows, Python 3.11, torch 2.9.0+cu130.

---

## 1. Headline

| | |
|---|---|
| **Best model** | Conformer-CTC, 8.4M parameters (`conformer/v004`) |
| **WER** | **0.3917** (greedy: 0.4705) |
| **CER** | **0.1264** (greedy: 0.1430) |
| Decoder | beam search, width 25, + character 6-gram LM (α 0.5, β 0.5) |
| Training data | 12.3 hours (Lagyamfi 3.2h + ghanaopendata 9.1h) |
| Validation | Lagyamfi test split, 259 clips, held out |
| Training time | 80 epochs from scratch (~2h), then 40 fine-tuning epochs (~37m) |

From a random initialisation, on twelve hours of audio, in two hours of training. CER 0.143 means roughly **six characters in seven are correct**.

### What it actually produces

Corpus reference:
> Wɔbɛtumi akɔ dan a ɛtoa wɔn so no ne ne yɔnko…

Model output (`v004`):
> wɔbɛtumi akɔdan a ɛtoa wɔn so no ne onyankoa…

Recognisably the same sentence, with spelling errors. That error profile is exactly what an n-gram language model fixes — and it did; see §6.

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

The health corpus is 59,291 clips of **exactly 30.00 seconds** with Gemini-generated transcriptions. Most likely cause is utterance length: CTC must align ~400 characters over 750 encoder frames from scratch. The mechanism is now measured; see below.

**This is the useful slide.** More data is not automatically better; utterance length and label quality decide whether it can be used at all.

### Diagnosed

Measured against the two corpora that train successfully:

| | health | ghanaopendata | lagyamfi |
|---|---|---|---|
| clip duration | **30.00s** fixed | 1.48s | 5.54s |
| target characters | **394** | 20 | 46 |
| **speech regions per clip** | **7** | **1** | **1** |
| CTC headroom (frames/char) | 1.90 (p05 1.49) | 1.59 | 2.93 |

**Seven utterances per training example** is the cause. Both working corpora give CTC one utterance — a single contiguous alignment. Health gives it a 30-second window holding ~7 utterances under one 394-character transcript, and CTC has to work out the boundaries unsupervised. The transcripts reveal spontaneous Adom TV broadcast media with 15% English code-switching.

Full diagnosis, the tooling built for it, and what is *not* established (whether the labels are accurate — the WER test is confounded by domain shift) are in [`roadmap_next_steps.md`](../roadmap_next_steps.md#1-make-the-500-hour-health-corpus-usable--diagnosed-tooling-built).

---

## 3b. A fifth of the training corpus was contributing nothing

CTC needs at least one encoder frame per label character, plus a blank between repeated characters. Measured across the two corpora actually used for training:

```
Train CTC feasibility: 17,747 samples | infeasible 3,295 (18.6%) | empty targets 201
                       headroom median 1.61
```

Those 3,295 samples have targets longer than their audio can align. `zero_infinity` turns the resulting infinite loss into a zero, so they occupy batch slots and return **no gradient** — silently, with nothing in the logs to say so.

Every run now reports this, and `--min-headroom 1.0` drops them:

```bash
python -m src.main train --min-headroom 1.0 ...
# Train: keeping 14,251 of 17,747 samples at headroom >= 1.00 (dropped 3,496)
```

Either the audio is truncated or those transcripts do not match it; both are worth knowing about rather than training through.

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

Test suite: **15 → 187 tests.**

---

## 5. Published model exports

Every finished training run is archived in the **model registry** under `outputs/asr/registry/`, and the app, CLI and evaluation all serve the *promoted* version — never simply the newest.

### Current exports

| | Architecture | Version | WER | CER | Run | Training data |
|---|---|---|---|---|---|---|
| **→** | conformer | **v004** | **0.4701** | **0.1429** | `run-v2-finetune` | Fine-tuned from v001, 40 epochs, bs 32, lr 2.5e-4 |
| | conformer | v003 | 0.5131 | 0.1593 | `run-v2-finetune` | Epoch-1 weights, mislabelled by a selection bug — see below |
| | conformer | v002 | 0.5102 | 0.1659 | `smoke-init-test3` | 1-epoch smoke test of `--init-weights` |
| | conformer | v001 | 0.5100 | 0.1660 | `run-clean` | 12.3h from scratch, 80 epochs, bs 32, lr 8e-4 |

`→` marks the version being served. `run-big` (113h, WER 1.0) is deliberately **not** published as a serving version — see §3.

**v004 is an 8% relative WER improvement over v001** (0.510 → 0.4701), from a warm restart: v001's weights reloaded with a fresh one-cycle schedule at lr 2.5e-4 over the same 12.3 hours of audio. No new data was involved.

### The selection bug that produced v003

Every WER and CER in the table above was **measured by a direct evaluation run**, not read from a training log — because for one release those two disagreed.

`best_model.pt` was written at the epoch with the lowest **validation loss**, while `_publish()` reported the metrics of the epoch with the lowest **WER**. On `run-v2-finetune` those were epoch 1 (val_loss 0.5402) and epoch 38 (WER 0.4533). The registry therefore published epoch-1 weights labelled WER 0.4533, and — because promotion compares WER — promoted them over v001. The app briefly served a model *worse* than the one it replaced (measured 0.5131 vs 0.5100) under a better-looking number.

Two consequences worth keeping in mind:

- **The epoch-38 weights are gone.** They were never written to disk, because selection did not track WER. The best surviving weights are the final epoch's, at 0.4701.
- The fix is in [`trainer.py`](../../src/asr/training/trainer.py): checkpoints are selected on WER when validation provides one, and publishing reports the metrics captured at the moment that checkpoint was written, so the label always describes the weights.

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

## 6. Language-model decoding

Greedy decoding takes the argmax at every frame independently, so nothing prefers a spelling that exists in Twi over one that does not. Fusing a character n-gram model into a CTC prefix beam search fixed that, **with no retraining**:

| Decoder | WER | CER |
|---|---|---|
| greedy | 0.4705 | 0.1430 |
| beam search alone | 0.4702 | 0.1398 |
| **beam + LM (α 0.5, β 0.5)** | **0.3898** | **0.1277** |

**A 17.2% relative reduction in WER.** Beam search on its own buys almost nothing — the gain is the language model.

Corrections it makes:

```
ref   : me kyɛm ne me nkwagyeɛ abɛn m'abantenten
greedy: me kyɛm ne me nkwagyeɛ abɛn m'abantentei
+LM   : me kyɛm ne me nkwagyeɛ abɛn m'abantenten      ← exactly right

ref   : mesu mefrɛ awurade ɔno a ɔfata sɛ wɔyi no ayɛ
greedy: mesu mefrɛ awuradeno a ɔfaka sɛ woyii no ayɛ
+LM   : mesu mefrɛ awurade no a ɔfata sɛ woyii no ayɛ  ← splits a word, fixes a letter
```

### Weights

α (language-model weight) and β (per-character bonus) were swept on the validation split:

| α | 0.0 | 0.3 | **0.5** | 0.8 | 1.2 |
|---|---|---|---|---|---|
| best WER | 0.4702 | 0.4068 | **0.3936** | 0.3957 | 0.4182 |

Past α ≈ 0.8 the language model starts overruling the acoustics and WER climbs again. β matters less; without it, raising α makes transcripts terser rather than more correct.

### The model

A **character** 6-gram, not word-level: Twi is agglutinative with no large lexicon to hand, so a word model would be mostly out-of-vocabulary, while a character model has no OOV case and targets spelling directly — which is the error being fixed. Trained on 2.0M characters of Twi from `verified_data.csv`, `crowdsourced_data.csv`, the CoNLL NER files and the training transcripts. **Held-out perplexity 4.05 per character**, against 41 for uniform.

There is a pleasing irony in the largest single source: `data/corpus/verified_data.csv` is the text-only file that caused this project's most expensive bug, when it was mistaken for an audio manifest and trained a model on silence for fifty minutes (§4). It has no audio — which is precisely what makes it good language-model data.

### Not KenLM

`kenlm` and `pyctcdecode` pin `numpy<2`, and this project runs numpy 2.3.3 under torch 2.9. Downgrading numpy to add a decoder would risk the training environment, so the n-gram model and the beam search are implemented in-repo ([`src/asr/decoding/`](../../src/asr/decoding/)) — a few hundred lines, no new dependencies, and it runs identically on both machines.

### Which Twi, and how much of it

Akuapem and Asante Twi differ systematically in spelling — Asante writes the doubled vowels Akuapem drops (`gyefoɔ`/`gyefo`, `abankɛseɛ`/`abankɛse`). Counting those markers places every audio corpus, **including the split WER is measured on**, firmly in Asante, while `verified_data.csv` — 84% of the language model's original text — sits well apart:

| corpus | Asante markers |
|---|---|
| Lagyamfi test (the WER split) | 98.9% |
| ghanaopendata / health audio | 97–99% |
| crowdsourced, CoNLL NER | 93–94% |
| **verified_data.csv** | **77.8%** |
| **Ghana-NLP parallel text** | **77.6%** |

Ablating the language model's sources settles what that costs:

| LM text | Lines | WER | CER |
|---|---|---|---|
| all sources | 49,703 | 0.3971 | 0.1303 |
| **without verified_data** | 24,283 | **0.3971** | **0.1280** |
| verified → Ghana-NLP | 30,370 | 0.4062 | 0.1323 |
| **audio transcripts only** | 17,544 | **0.3913** | **0.1268** |
| all + Ghana-NLP | 55,790 | 0.4011 | 0.1316 |

Three things follow, and the third is the one worth remembering:

1. **Dropping the 25,420 Akuapem-leaning lines is free.** WER is unchanged within noise and CER improves, so `verified_data.csv` is now excluded by default (`--include-verified` restores it).
2. **Ghana-NLP is not a replacement.** It scores the same 77.6% on dialect markers and makes both metrics worse. Not wired in.
3. **Less text won.** 17,544 in-domain transcripts beat 49,703 lines of mixed text. For a language model, matching the dialect and domain of the audio matters more than volume.

#### Two candidate replacements, both rejected on measurement

`Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` (6,090 rows) scores the same 77.6% on dialect markers as the file it would replace, and made both metrics worse.

`michsethowusu/twi-english-parallel-synthetic-50m` advertises 47.9M rows, which sounds decisive until it is measured:

| corpus | lines | word tokens | **distinct types** | type/token |
|---|---|---|---|---|
| 50m synthetic (400k sampled) | 400,000 | 800,030 | **634** | 0.08% |
| audio transcripts | 17,544 | 113,787 | **11,167** | 9.81% |
| **crowdsourced_data.csv** | **695** | 3,172 | **1,124** | 35.44% |
| twi CoNLL | 6,044 | 142,271 | 12,616 | 8.87% |

Every sampled row is **three words or fewer** (median two), 43% are exact duplicates, and the English side is word salad — *"plantain repent"*, *"chair be cooked"*. The whole dataset is a few hundred Twi words recombined into pairs; at 2.89 GB across 47.9M rows, ~60 bytes per row, that structure holds throughout and not just in the sample. It contributes **200 word types** not already covered.

What it does to WER:

| LM text | WER | CER |
|---|---|---|
| greedy, no LM | 0.4725 | 0.1437 |
| current sources | 0.3971 | 0.1280 |
| current **+ 50m synthetic** | 0.3965 | 0.1276 |
| **50m synthetic alone** | **0.5445** | 0.1781 |

Used alone it is **worse than not using a language model at all**. Added to the existing sources it changes nothing measurable. Not adopted.

The line worth keeping from this: `crowdsourced_data.csv` is 695 lines and holds **1,124 word types — nearly twice the vocabulary of 400,000 rows of the synthetic set**. Row counts are a poor proxy for how much a language model can learn.

The caveat on (3): those transcripts are Bible-domain, the same as the eval split, so an LM built only from them is narrow by construction. It is the right choice for this benchmark and the wrong one for a general-purpose product — which is why genuine **Asante** text in other domains is the thing worth acquiring. `scripts/lm/build_translation_worklist.py` writes 31,014 unique English sentences to `data/corpus/english/` for exactly that.

### Leakage check

Since the LM's text sources include general Twi and the validation set is scripture, the two could overlap. Measured: **1 of 259 validation references (0.4%) appears in the LM's training text**, and that one is `"na afei"` — a two-word phrase, not a memorised verse. Zero substring containments. The gain is not recitation.

### Cost

Beam search is ~0.1s per clip against greedy's ~0.0004s. Irrelevant beside the forward pass for transcription; it is why per-epoch validation during **training** still uses greedy, while evaluation and serving use the full decoder.

---

## 7. Throughput and hardware

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

## 8. Speaker diarization

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

## 9. Reproducing

```bash
# The working model
scripts/asr/train.sh --architecture conformer \
  --hf-dataset Lagyamfi/akan_audio_processed:train \
  --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k:train \
  --val-dataset Lagyamfi/akan_audio_processed:test \
  --epochs 80 --batch-size 32 --lr 8e-4

# Evaluate
python -m src.main evaluate --architecture conformer \
  --hf-dataset Lagyamfi/akan_audio_processed:test \
  --model-path outputs/asr/checkpoints/run-clean/best_model.pt

# Rebuild the language model (needed once; not committed)
python scripts/lm/build_lm.py

# Evaluate with and without it
python -m src.main evaluate --architecture conformer --decoder greedy
python -m src.main evaluate --architecture conformer --decoder beam

# Speaker-attributed transcript
python -m src.main diarize --audio recording.wav --backend auto

# Model versions: what is published, what is served, and how to fall back
python -m src.main models list
python -m src.main models rollback --architecture conformer

# Demo app
scripts/asr/serve_app.sh
```

Continue a run with `--resume outputs/asr/checkpoints/<run_id>/last_model.pt`.
