# SesaML — Measured Results

Every figure here was produced by this repository. Where something is unverified or a negative result, it says so.

Hardware for §§2–9: **NVIDIA RTX 4000 SFF Ada**, 20GB, 70W cap. Windows, Python 3.11, torch 2.9.0+cu130. The runs in §10 were measured on an **NVIDIA RTX A6000** (48GB) on `knust-gpu`.

---

## 1. Headline

The model the project serves is **conformer-medium/v003** (`p5-ctrl`), 26,281,257 parameters. The headline number is a separate `evaluate` run, not the training log.

| | |
|---|---|
| **Served model** | Conformer-CTC medium, 26,281,257 parameters (`conformer-medium/v003`) |
| **WER** | **0.2887** beam+LM (greedy: 0.3350) |
| **CER** | **0.0828** beam+LM (greedy: 0.0924) |
| Decoder | beam search, width 25, + character 6-gram LM (α 0.5, β 0.5) |
| Test set | Lagyamfi test, 259 clips, held out (`evaluate-20261016-003148`, `evaluate-20261016-003142`) |
| Training data | ghanaopendata train + Lagyamfi `train_core` + WAXAL v2 train, 31,268 samples kept at headroom ≥ 1.0 |
| Selection | best dev greedy WER, epoch 10 of 12. Dev greedy was 0.2899; that is not the headline |
| Registry label | test greedy 0.335 / 0.0924. The beam figure lives in the version notes |

The registry stores the greedy test WER. Training publishes the best **dev** greedy WER and will auto-promote when that number is lower than the stored label, which is how a worse model can replace v003. §10 records the three runs that did that and were rolled back. The day-to-day log of every run is [`data/PLAN_WER_0.30.md`](../../data/PLAN_WER_0.30.md).

### Earlier baseline

The first write-up of this file, measured on the Ada card, is still the history of the small model:

| | |
|---|---|
| **Model** | Conformer-CTC, 8,380,409 parameters (`conformer/v004` at the time) |
| **WER** | **0.3917** (greedy: 0.4705) |
| **CER** | **0.1264** (greedy: 0.1430) |
| Decoder | beam search, width 25, + character 6-gram LM (α 0.5, β 0.5) |
| Training data | 12.3 hours (Lagyamfi 3.2h + ghanaopendata 9.1h) |
| Validation | Lagyamfi test split, 259 clips, held out |
| Training time | 80 epochs from scratch (~2h), then 40 fine-tuning epochs (~37m) |

A re-evaluation on 2026-10-14 (`evaluate-20261014-110832`, `evaluate-20261014-110827`) measured the same beam WER, 0.3917 / 0.1264, and greedy 0.4713 / 0.1431. From a random initialisation, on twelve hours of audio, in two hours of training. CER 0.143 means roughly **six characters in seven are correct**.

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

`→` marks what this table's write-up was serving. That is no longer current. `run-big` (113h, WER 1.0) is deliberately **not** published as a serving version — see §3. The registry served today is `conformer/v006` for the small model and **`conformer-medium/v003`** for the model the app should serve. Both are in §10, with test-set numbers from `evaluate` rather than from the training log.

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
# Headline number for the served model. Training validation stays greedy;
# this beam run is the number to quote.
python -m src.main evaluate --architecture conformer-medium \
  --model-path outputs/asr/registry/conformer-medium/v003/model.pt \
  --hf-dataset Lagyamfi/akan_audio_processed:test \
  --decoder beam

# The early small-model baseline
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

---

## 10. The path under 0.30

Every row is an `evaluate` run on **Lagyamfi test, 259 clips**, unless the row names another set. Beam is CTC prefix beam search, width 25, with the character 6-gram at α 0.5, β 0.5 (`outputs/asr/lm/twi_char.json`). Greedy is what training-time validation reports. On this test set the project treats about **±1.1 points** as one standard error, and does not promote a model for a smaller move.

The small model below is 8,380,409 parameters. The medium model is 26,281,257. Both counts are the figure the training log prints at startup.

| Run | Registry | What changed | Greedy WER / CER | Beam+LM WER / CER | After the test |
|---|---|---|---|---|---|
| `p1-clean-baseline` | `conformer/v005` | From scratch, ghanaopendata + Lagyamfi `train_core`, 80 epochs. The registry file still stores the **dev** greedy 0.4717; the test numbers are the evaluate run | 0.4871 / 0.1500 | 0.3977 / 0.1289 | Kept as the clean init. Not served |
| `p2-waxal-ft` | `conformer/v006` | Warm start of v005, plus WAXAL v1 train, 30 epochs, best epoch 25 | 0.4125 / 0.1212 | 0.3622 / 0.1100 | **Served small model** |
| `p3-medium` | `conformer-medium/v001` | Same three corpora, medium model from scratch, 40 epochs, best epoch 39 | 0.3540 / 0.0962 | 0.3131 / 0.0897 | Replaced by v002 |
| `p4-medium-ft` | `conformer-medium/v002` | Warm start of v001, WAXAL v1, 30 epochs, lr 1.5e-4, best epoch 26 | 0.3430 / 0.0935 | 0.3030 / 0.0862 | 0.3 pt above 0.30, inside the noise. Replaced by v003 |
| **`p5-ctrl`** | **`conformer-medium/v003`** | Warm start of v002, real data only, WAXAL **v2** train (17,317 clips), 12 epochs, lr 1e-4, best epoch 10 | **0.3350 / 0.0924** | **0.2887 / 0.0828** | **Served.** 1.4 pt under v002, just outside the noise |
| `p5-pseudo` | `conformer-medium/v004` | v003 recipe plus 7,772 pseudo-labelled WAXAL clips (41 h, top 40% by `seq_lp`) | 0.3492 / 0.0945 | 0.2975 / 0.0857 | Negative. Rolled back |
| `p6-cs` | `conformer-medium/v005` | v003 recipe plus 8,003 code-switch clips (`train_cap7`, 13.4 h) | 0.3352 / 0.0921 | 0.2933 / 0.0841 | No change on Twi test. Rolled back |
| `p7-speed` | `conformer-medium/v006` | v003 recipe plus train-only speed perturbation 0.9 / 1.0 / 1.1, warm start from **v003**, 12 epochs, best dev epoch 2 | 0.3324 / 0.0895 | 0.2920 / 0.0835 | Negative. Rolled back |

Evaluate run ids for the served model and for `p7-speed`: `evaluate-20261016-003142` (v003 greedy), `evaluate-20261016-003148` (v003 beam), `p7-speed-test-greedy`, `p7-speed-test-beam`. The earlier rows are the `evaluate-20261014-*` and `evaluate-20261015-*` / `evaluate-20261016-*` directories under `outputs/asr/runs/`.

### What each negative result actually did

**Pseudo-labels (`v004`).** WAXAL v2 test greedy was 0.3790 / 0.1288 against v003's 0.3801 / 0.1298. Beam on Lagyamfi test was 0.9 pt worse. The gain that took v002 from 0.3030 to v003's 0.2887 came from the extra real WAXAL v2 audio and the extra fine-tune, not from the pseudo-labels.

**Code-switching (`v005`).** Lagyamfi test did not move. A prompt-disjoint holdout of 434 clips did: greedy 0.6597 / 0.2770 against v003's 0.8691 / 0.5049 (`evaluate-20261016-115910`, `evaluate-20261016-105826`). Beam on that holdout is 0.6925 / 0.3191 for v005 and 0.8496 / 0.5128 for v003, so the Twi-only language model hurts the code-switched model. WAXAL v2 test greedy improved to 0.3676 / 0.1251. v005 is the code-switch option. It is not the served Twi model. 0.66 on the holdout is still poor: the corpus has 1,165 sentences.

**Speed perturbation (`v006`, 2026-10-31).** One change from the v003 recipe: each training waveform is resampled at 0.9×, 1.0×, or 1.1× before the mel spectrogram. `SpeedPerturbation` is the first stage of `get_train_audio_transforms` only. Evaluation and inference use `get_valid_audio_transforms`, which does not include it. The run started from v003's weights, on GPU 0, and finished in 2951 seconds. Beam is 0.3 pt worse than v003 and greedy is 0.3 pt better. Both are inside the noise. This run was not scored on WAXAL test.

Dev greedy on the 300-clip Lagyamfi dev set, which is what selected the checkpoint:

| Epoch | Train loss | Dev WER | Dev CER |
|---|---|---|---|
| 1 | 0.512 | 0.2976 | 0.0760 |
| **2** | **0.502** | **0.2935** | **0.0748** |
| 3 | 0.508 | 0.3071 | 0.0767 |
| 4 | 0.519 | 0.3028 | 0.0752 |
| 5 | 0.511 | 0.3112 | 0.0801 |
| 6 | 0.497 | 0.3066 | 0.0790 |
| 7 | 0.484 | 0.3043 | 0.0789 |
| 8 | 0.466 | 0.2953 | 0.0747 |
| 9 | 0.452 | 0.2977 | 0.0755 |
| 10 | 0.439 | 0.2959 | 0.0754 |
| 11 | 0.433 | 0.2962 | 0.0748 |
| 12 | 0.424 | 0.2945 | 0.0751 |

Train loss fell through epoch 12. Dev WER never beat epoch 2. A longer run of this fine-tune is not supported by that curve. v003's own dev greedy best was 0.2899, so the speed-perturbed model was behind on the selection set as well as on the test beam number.

`AudioConfig.speed_factors` defaults to `(0.9, 1.0, 1.1)`, so a later training run includes the layer unless that tuple is emptied. v003 itself was trained before the layer existed.

### Promotion compares the wrong pair of numbers

`ModelRegistry` promotes when the new run's WER is lower than the served version's stored WER. v003's stored WER is the test greedy **0.335**. A training run publishes its best **dev** greedy WER, which sits near 0.29. `p7-speed` therefore logged `Published conformer-medium/v006 (WER 0.2935) and promoted it to current` even though the test beam number was worse. The same thing happened to `v004` and `v005`. Each time, `models rollback --architecture conformer-medium` returned the served pointer to v003, and the version's `metadata.json` was then relabelled so `wer` / `cer` are the test greedy figures and the dev figures sit in `training_log_dev_greedy_wer` / `training_log_dev_greedy_cer`.

`conformer/v005` has not been relabelled. Its registry WER, 0.4717, is the dev number. The test numbers in the table above are the evaluate run.

### Language model, after the acoustic model moved

§6 is the measurement on the early small model. A later ablation on `conformer/v006` (2026-10-15, same beam settings) did not find a better language model: dropping Akuapem-leaning `verified_data.csv` was free, adding WAXAL transcripts or `michsethowusu/twi-english-parallel-synthetic-50m` made things worse, and an α/β re-sweep stayed inside the noise of α 0.5, β 0.5. The shipped LM is the one the headline uses. It has read the Lagyamfi **dev** transcripts, so dev beam numbers are leaky. Test has not: 1 of 259 references overlaps, and that one is the two-word phrase "na afei".

### Where the measured record stops

The order of work after this table is [`court_flow.md`](court_flow.md) (2026-10-31). Lagyamfi test stays the regression number. The next measurement is the court gold sheet, drafted with v003, then scored. Dropping Fante-marked WAXAL clips is the first training change in that plan, and it waits on the sheet. Checkpoint averaging is not scheduled. The run log that produced the table remains [`data/PLAN_WER_0.30.md`](../../data/PLAN_WER_0.30.md). The 500-hour health corpus is still the negative result in §3. It has not been retrained.
