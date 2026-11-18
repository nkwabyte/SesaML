# SesaML Roadmap & Next Steps

Planned technical improvements and architectural milestones for SesaML on Akan (Twi) Automatic Speech Recognition.

Items marked **Completed** were implemented and measured; see [`docs/asr/results.md`](asr/results.md) for the numbers.

## Where the measurements stand (2026-10-31)

The served model is **`conformer-medium/v003`**: Lagyamfi test beam+LM **0.2887 / 0.0828**, greedy 0.3350 / 0.0924, 259 clips. The small model currently served is `conformer/v006` at beam+LM 0.3622 / 0.1100. The full table, including the runs that were rolled back, is [`results.md` §10](asr/results.md#10-the-path-under-030).

Measured and not adopted, because none of them beat v003 on the Lagyamfi test beam number by more than the ±1.1-point noise:

- Pseudo-labels on 41 h of WAXAL (`conformer-medium/v004`, beam 0.2975).
- Code-switch fine-tune (`conformer-medium/v005`, beam 0.2933). It does help a prompt-disjoint code-switch holdout (greedy 0.6597 against v003's 0.8691) and stays available for that.
- Train-only speed perturbation at 0.9 / 1.0 / 1.1 (`conformer-medium/v006`, beam 0.2920). The layer is now the default in the train front-end. v003 was trained without it.
- Language-model text and α/β. The shipped character 6-gram at α 0.5, β 0.5 is the one the headline uses.

The next work is the court-first flow in [`asr/court_flow.md`](asr/court_flow.md). Lagyamfi test stays a regression check. It is no longer the set that selects or promotes a model. The blocking step is a native speaker filling the court sheet, after the pack is re-drafted with v003. Dropping Fante-marked WAXAL clips is the first acoustic experiment, and only after that sheet can score it. Checkpoint averaging is not in the flow. A longer fine-tune is not the default next step: on `p7-speed`, training loss fell through epoch 12 while dev WER never beat epoch 2.

The 500-hour health corpus is unchanged from Phase 1.1 below. It has not been made trainable.

`ghanaopenai/twi-multispeaker-tts` is a separate 91-hour broadcast set with transcripts, checked on the Hub on 2026-11-02 and not downloaded. It is a candidate addition, not a replacement. The work is in [`asr/ghanaopenai_twi_multispeaker_tts.md`](asr/ghanaopenai_twi_multispeaker_tts.md). The raw 30-second clips are the same shape that failed on the health corpus, so they have to be cut to utterances before any training run.

---

## Phase 0: Completed

### 1. Length-Bucketed Batching — *completed*
Batches clips of similar duration together ([`src/asr/data/bucketing.py`](../src/asr/data/bucketing.py)). Measured on the real corpus length mixture, the fraction of batched frames carrying audio rather than padding rose from **20% to 97%**; on the mixed 113-hour run the training log reported **99% vs 88% shuffled**.

The estimate in the previous draft ("~19% compute waste") was **backwards** — waste was ~80%, because ghanaopendata's median clip is under a second while the health corpus is uniformly 30 seconds. Mixing them at random pads nearly everything to 30s.

### 2. Mixed-Precision Training — *completed*
CUDA AMP with `GradScaler` ([`src/asr/training/trainer.py`](../src/asr/training/trainer.py)). The encoder runs in fp16; `log_softmax` and the CTC loss stay in fp32, which is where the numerical sensitivity lives. Gradients are unscaled before clipping.

### 3. Log-Mel Features & Per-Utterance Normalization — *completed*
The front-end previously fed **raw mel power** (measured range 0–2568, mean 233) straight into the encoder. It now applies log compression and per-utterance CMVN ([`src/asr/data/audio_transforms.py`](../src/asr/data/audio_transforms.py)), with SpecAugment last so masks fill with the feature mean.

`n_mels` also dropped 128 → 80: against a 400-point FFT (201 bins), 128 mel filters left **four filterbanks entirely empty**.

### 4. Speaker Diarization — *completed*
Three backends, colour-coded Gradio tab, `diarize` CLI command. See [`docs/asr/speaker_diarization.md`](asr/speaker_diarization.md).

### 5. Training Reliability — *completed*
Resume-from-checkpoint, corpus probes that abort on a bad dataset before epoch 1, gradient clipping, `zero_infinity` CTC, and offline corpus resolution.

### 6. Versioned Model Registry — *completed*
Every finished run is archived under `outputs/asr/registry/` and promoted only if it beats the served version's WER ([`src/utils/model_registry.py`](../src/utils/model_registry.py)). Checkpoints previously resolved by modification time, which — given that a 113-hour run produced a *worse* model than a 12-hour one — meant one bad iteration would silently replace a working demo.

Rollback follows promotion history, and is exposed on the CLI (`models rollback`) and as a dropdown in the app. Versions also record the front-end settings they were trained on, because loading 80-mel weights under a 128-mel config degrades silently rather than raising.

---

## Phase 1: Scaled Training & Optimization

### 1. Make the 500-Hour Health Corpus Usable — *diagnosed; tooling built*

A 5-epoch run over 113 hours (100h of it from the health corpus) **failed to learn**: WER stayed at 1.0 and training loss plateaued at ~2.75 nats per character. The cause is now measured rather than guessed.

#### What the corpus actually is

| | health | ghanaopendata | lagyamfi |
|---|---|---|---|
| clip duration | **30.00s** fixed | 1.48s | 5.54s |
| target characters | **394** | 20 | 46 |
| **speech regions per clip** | **7** | **1** | **1** |
| CTC headroom (frames/char) | 1.90 (p05 1.49) | 1.59 | 2.93 |
| English word ratio | **0.15** | 0.00 | 0.08 |

The transcripts show what it is: spontaneous **Adom TV broadcast media**, code-switched Twi/English — *"wɔ Adom TV bɛdwema dwumadie yi so… multimedia can brand"*, *"in the first place"*.

#### Why CTC cannot learn from it

**The dominant cause is seven utterances per training example.** Both corpora that train successfully give CTC one utterance per sample — a single contiguous alignment. The health corpus gives it a 30-second window holding ~7 separate utterances split by pauses, under one 394-character transcript. CTC must discover, with no supervision on the boundaries, which characters belong to which stretch of speech. Plateauing at 2.75 nats/char — better than the 3.71 of uniform guessing, never finding an alignment — is what that failure looks like.

Secondary: headroom of 1.90 frames per character (1.49 at p05) leaves very little room to place blanks, against 2.93 for the corpus that trains comfortably.

**What is not established:** whether the Gemini labels are accurate. `v004` scores WER 0.987 on health against 0.688 on a Lagyamfi control, but that comparison is confounded — a model trained on read scripture fails on broadcast media whether the labels are right or wrong. This needs a native speaker to check a sample before anyone concludes the labels are bad.

#### Tooling built

- [`src/asr/data/feasibility.py`](../src/asr/data/feasibility.py) — measures and filters samples CTC cannot align, counting the blank required between repeated characters (Twi doubles vowels freely, so ignoring that under-counts).
- [`src/asr/data/alignment.py`](../src/asr/data/alignment.py) — CTC forced alignment via Viterbi over the expanded lattice. Written here rather than taken from `torchaudio.functional.forced_align`, which is deprecated and slated for removal in the version this project pins; validated identical to that reference on 40/40 random cases.
- [`src/asr/data/segmenting.py`](../src/asr/data/segmenting.py) — cuts a long clip into utterances at aligned pauses, splitting the *text* at the same points so audio and transcript stay in correspondence. Cuts land only on word boundaries.

#### Measured on real health audio

Aligned 8/8 clips, producing 25 utterances from 8 thirty-second windows (median 8.2s). Two honest caveats:

- **Alignment confidence is low** — median score −6.7, i.e. the aligner is genuinely unsure, consistent with a model trained on read scripture being asked to align broadcast media.
- **The 30s windows are arbitrary cuts from a continuous broadcast**, so their transcripts begin and end mid-sentence. Segmentation inherits that truncation at the edges.

#### Next steps, in order

1. **Align with a domain-appropriate model.** The segmentation machinery works; its weak link is using `v004` as the aligner. A multilingual model (MMS, wav2vec2) would align this audio far better.
2. **Filter segments by alignment confidence** before training on them, and measure what fraction survives.
3. **Fine-tune, never train from scratch** on this corpus.
4. **The product target is court speech, not this corpus.** Decided in [`court_flow.md`](asr/court_flow.md). Health is Adom TV broadcast media. Training on it optimises a different distribution from both Lagyamfi and the court recordings. It stays blocked until a speaker has checked the labels, and it is not the next training run.

### 1b. CTC Feasibility Filtering — *completed*

Measured across the corpora in use, **18.6% of the training set (3,295 of 17,747 samples) has a CTC target longer than its audio can align**, plus 201 empty targets. `zero_infinity` turns their infinite loss into a zero, so they fill batch slots and return no gradient — nearly a fifth of the corpus training on nothing, silently.

Every run now reports feasibility, and `--min-headroom 1.0` drops the unalignable samples. See [`docs/asr/results.md`](asr/results.md).

### 2. Akan Text Preprocessing & Number Normalization
Convert numeric digits (`2 capsules` → `capsules mmienu`) to spelled-out Twi. The model currently must map the *sound* of a Twi number to an ASCII digit.

Partially addressed: typographic variants are now folded (`’` → `'`), which had been silently gluing elisions like `m’abankɛseɛ` into one word. Corpus probes report that ~4% of characters are dropped, and confirm they are **all punctuation** — legitimately not spoken.

### 3. N-Gram Language Model & Beam Search Decoding — *completed*
Done, and it was the highest-value item on the early small model: **WER 0.4705 → 0.3898, a 17.2% relative reduction, with no retraining.** A character 6-gram over 2.0M characters of Twi, fused into a CTC prefix beam search ([`src/asr/decoding/`](../src/asr/decoding/)). Full numbers, the α/β sweep and the leakage check are in [`results.md`](asr/results.md#6-language-model-decoding). The same decoder on the served medium model is the 0.2887 headline. A later text and α/β ablation did not beat the shipped LM; that result is in [`results.md` §10](asr/results.md#10-the-path-under-030).

Not KenLM: it and `pyctcdecode` pin `numpy<2` against this project's numpy 2.3.3 under torch 2.9, so the n-gram model and beam search are implemented in-repo instead.

**Where to take it further:** a *word*-level LM with a lexicon would likely beat the character model, but needs a Twi wordlist. More Twi text would help directly — the current 2.0M characters is small for an n-gram model.

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
| **N-Gram LM + Beam Search Decoder** | 0.4705 → 0.3898 WER on the early small model; 0.2887 on the served medium model | High | Completed |
| **WAXAL fine-tune + medium Conformer** | Served `conformer-medium/v003`, Lagyamfi test beam+LM 0.2887 / 0.0828 | High | Completed |
| **Pseudo-labels, code-switch, speed perturbation** | None beat v003 on Lagyamfi test beam by more than the noise. Rolled back | High | Measured, not adopted |
| **Court-first flow** | Stop promoting on Lagyamfi. Score customary-court audio, then one change at a time | High | Planned — [`court_flow.md`](asr/court_flow.md) |
| **Noise Augmentation Training** | On-the-fly MUSAN SNR mixing | Medium | Documented |
| **Twi Number Normalization** | Digits → spelled-out Twi | Medium | Partial |
| **DER Evaluation Set** | Labelled multi-speaker Akan audio | Medium | Planned |
