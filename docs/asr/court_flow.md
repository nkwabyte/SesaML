# Court-first flow

The product is Asante Twi speech from customary-court and council hearings, transcribed toward English. The requirements PDF in `data/` names that use. Lagyamfi is Bible reading. It produced the benchmark number. It is the wrong set to keep optimising.

Recorded 2026-10-31. This page is the order of work. [`court_flow_scratchpad.md`](court_flow_scratchpad.md) is where the phase status is kept. [`results.md`](results.md) keeps the measurements. [`data/PLAN_WER_0.30.md`](../../data/PLAN_WER_0.30.md) keeps the run log that got to 0.30.

## Decision

`conformer-medium/v003` stays the served model. Lagyamfi test stays a regression check. It stops being the set that selects a checkpoint, promotes a version, or decides that a change worked.

No training run starts until a native speaker has filled a court sheet and that sheet has been scored. Another fine-tune chosen on Lagyamfi dev would be rejected or promoted for the wrong reason, which is what happened to `p7-speed`.

## Why the Bible set binds

The hours do not. In the v003 recipe, Lagyamfi `train_core` is 1,887 clips from a 3.2-hour train set. The kept mix is 31,268 clips: that slice, ghanaopendata (9.52 h), and WAXAL v2 train (17,317 clips, 93.65 h).

The decision does. Checkpoint selection is the 300-clip Lagyamfi dev set. The published number is Lagyamfi test, 259 clips, beam+LM **0.2887 / 0.0828**. About ±1.1 points on that test is one standard error. A change inside that band is treated as noise, so a model that helps court speech and leaves the Bible number flat is recorded as no gain.

Promotion makes it mechanical. `ModelRegistry.publish` promotes when the new version's stored WER is lower than the served version's. Training writes its best **dev** greedy WER into that field. v003's stored WER is the test greedy **0.335**. A dev number near 0.29 therefore promotes. `p7-speed` did: dev greedy 0.2935 became `conformer-medium/v006` and had to be rolled back. Test beam was 0.2920, worse than 0.2887 and inside the noise. The same auto-promote happened to `v004` and `v005`.

The language model was chosen the same way. 17,544 in-domain transcripts beat 49,703 lines of mixed text, and that domain is scripture. The shipped character 6-gram is the right model for the Bible number and the wrong default for court audio. Adding WAXAL transcripts, Ghana-NLP parallel text, or the synthetic 50-million-row set did not help the Bible number. Volume is not a substitute for domain.

## What has been measured on court audio

`data/Splitted Audio Files` is 3,785 clips of exactly 10.00 s (10.5 h): Gyaase council meetings, Mpentuase vs Bomso, and Ghana Immigration vs Mpentuase. No transcripts. The 2026-10-15 audit found no unsplit originals of these hearings on this machine.

`scripts/asr/court_gold.py` built a 120-clip pack (20 min) under `data/court_gold/`. Treatments: 50 Whisper drafts, 50 Conformer drafts, 20 from scratch. The private key (`data/court_gold/private/key.csv`) stays on this Ubuntu machine. It is not committed, and it is not on `knust-gpu`.

The drafts that were scored in the plan were **not** the served medium model. Whisper's median confidence was −0.34, against −0.03 on Lagyamfi, and 17% of its court drafts were the memorised line `na sane firi wɔn so kɔ soro`. Small Conformer v006 beam produced a median of 4 words per 10-second clip, 90 of 120 under 6 words. Seventeen degenerate drafts were blanked. **v003 has not been scored on this pack.** `court_gold.py draft` defaults `architecture` to `conformer` and leaves `model_path` empty, and `resolve_checkpoint` then serves the small conformer's current version.

Code-switch is a related measurement, not a court measurement. On a 434-clip prompt-disjoint holdout, v003 greedy is 0.8691 / 0.5049. The code-switch fine-tune (`v005`) reaches 0.6597 / 0.2770 greedy, and beam with the Twi LM is worse (0.6925). Court speech mixes English into Twi. That holdout is scripted read speech, so it is a proxy.

## The flow

1. **Freeze v003.** Report Lagyamfi test beam whenever a later model is compared. Do not promote on Lagyamfi dev.
2. **Put v003 drafts on the existing pack, then have the sheet filled.** Twenty of the 120 clips are from scratch. That split measures whether the corrector was pulled toward the draft. It is too small to promote on.
3. **Score the sheet and read the errors before any training.** Tag mid-word cuts, English inside Twi, overlap, silence, and empty hypotheses. `src/asr/data/segmenting.py` cannot recut these files: it splits a transcript and the audio together, and these clips have no transcript. The unsplit hearings are not on disk either. Until they turn up, references follow the pack guideline and omit a word cut off by the 10-second boundary. The rate of those cuts is a caveat on the score.
4. **Grow the from-scratch set before a training comparison.** Aim for 100 from-scratch references. The ±1.1-point rule was estimated on 259 Bible clips and does not apply to 20 court clips. New clips in this batch are from scratch, so the draft cannot anchor them.
5. **One acoustic change.** Drop WAXAL v2 train rows with `fante_marker` set. Everything else stays the v003 recipe, initialised from v003. The court from-scratch set picks the winner. Lagyamfi test is only the regression check. About 16% of the WAXAL transcripts carried Fante-style markers when the corpus was audited; the filtered run must print the share and the hours actually removed.
6. **Swap the language model with the acoustics held still.** Build it from the corrected court transcripts and any other Asante court or council text that is lexically rich. Distinct word types are the test, not row count. Decode the same gold clips with the Bible LM and with the court LM. The Bible LM stays in place for the Lagyamfi regression number.
7. **Add hours last.** Agreement-filtered pseudo-labels of court audio wait until two systems have both been scored on the from-scratch clips and a person has checked a sample. The drafts in hand are blank or a memorised verse. The health corpus stays out. A 113-hour run that was mostly health finished at WER 1.0, and its labels have not been checked by ear.

Checkpoint averaging and a longer fine-tune of the v003 recipe are not in this flow. `p7-speed` already showed training loss falling for 12 epochs while dev WER never beat epoch 2.

## Implementation plan

Work runs on this Ubuntu machine and on `knust-gpu`. Nothing in this plan is copied to the MacBook. The GPU server keeps only the three training corpora (`Lagyamfi__akan_audio_processed`, `ghanaopendata__twi-speech-text-multispeaker-16k`, `ug-waxal__akan-transcribed-v2`) until a later phase actually trains. Court audio stays on Ubuntu until that run.

Each phase has one change. A phase does not start until the gate above it is open.

### P0 — Freeze the served pointer

**Status:** decided here. No code yet.

**Work:** none beyond this document. `conformer-medium` current remains `v003`.

**Gate:** open. P1 is the next code change, and it does not need the speaker.

### P1 — Training must not promote

**Status:** not started.

**Work:** `Trainer._publish` calls `ModelRegistry.publish` without `promote=`, so the default promotes when dev greedy beats the stored test greedy. Pass `promote=False` from the trainer. Promotion stays an explicit `python -m src.main models promote` after the deciding evaluation. Record the dev figures as `training_log_dev_greedy_wer` and `training_log_dev_greedy_cer`, which is the relabel already applied by hand to `v004`, `v005`, and `v006`.

**Files:** `src/asr/training/trainer.py`, `src/utils/model_registry.py` only if the metric keys need a helper, and a test beside the existing registry tests.

**Runs on:** Ubuntu. No GPU.

**Done when:** a test publishes a version whose dev WER is 0.29 against a current version stored at 0.335, and `current` does not change.

**Out of scope:** changing which epoch is best. The run may still track Lagyamfi dev internally. Serving does not follow that number.

### P2 — Draft the pack with v003

**Status:** not started. The sheet's `corrected` column is the blocker for everything after P4, but P2 has to happen first so the draft the speaker sees is the served model.

**Work:**

- Point `scripts/asr/court_gold.py draft` at `outputs/asr/registry/conformer-medium/v003/model.pt`.
- Set `config.model.architecture` to `conformer-medium` and the decoder to beam width 25, α 0.5, β 0.5, LM `outputs/asr/lm/twi_char.json`. Refuse to write drafts unless the loaded decoder name is `beam+lm` and `describe()` shows that checkpoint.
- Write the checkpoint path and the decoder description into `private/drafts_conformer.json`. The file on disk today has neither, so a later reader cannot see which weights produced it.
- Rebuild `pack/transcription_sheet.csv` only if every `corrected` cell is empty. If any cell has text, stop. Do not overwrite a speaker's work.
- Leave the Whisper JSONL as it is unless it is missing. Those drafts are the other arm of the anchoring test.

**Runs on:** Ubuntu. A 10-second clip through v003 on CPU is about a second once the weights are loaded. 120 clips, greedy and beam, do not need the GPU server.

**Done when:** the json names v003, the sheet still has an empty `corrected` column, and the blanked-draft count is printed.

**Do not:** copy `private/key.csv`, `.env`, or `data/` to the MacBook or to `knust-gpu`.

### P3 — Speaker fills the sheet

**Status:** blocked on P2.

**Work:** give the speaker `data/court_gold/pack/` (`audio/`, `transcription_sheet.csv`, `GUIDELINES.md`). They fill `corrected`. The guidelines already cover Asante spelling, English words left in English, `[?]` for an unclear word, `[skip]` for silence, and words cut off by the 10-second boundary.

**Done when:** the filled sheet is back on Ubuntu, outside the pack the speaker can edit, so a later rebuild cannot destroy it.

### P4 — Score the pilot and write it down

**Status:** blocked on P3.

**Work:** `python scripts/asr/court_gold.py --root data/court_gold score --sheet <filled sheet>`. Record, in [`results.md`](results.md), the WER/CER for Whisper, Conformer greedy, and Conformer beam on all usable rows, on the from-scratch rows, and on each shown-draft arm, plus the corrector-change CER. Then read the notes and count mid-word cuts, English, overlap, silence, and empty hypotheses.

**Done when:** those numbers are in `results.md` with the clip counts, and the from-scratch count is stated on its own. No model is promoted from this table.

**Gate to P5:** at least 100 from-scratch references exist. The pilot has 20. Close the gap with `court_gold.py select --scratch` equal to `--n`, a new anonymised pack, and another sheet. Those clips get no draft. If the pilot's notes say the 10-second cuts destroy most of the speech, stop and look for the unsplit hearings before collecting the larger set. Do not invent a segmenter pass over text these files do not have.

### P5 — Drop Fante-marked WAXAL

**Status:** blocked on the P4 gate.

**Work:** one change against the v003 recipe. Same three corpora, same headroom 1.0, same init `conformer-medium/v003/model.pt`, validation still logged on Lagyamfi dev so the run has a curve. Training rows of `ug-waxal/akan-transcribed-v2` with `fante_marker` true are removed. Print how many clips and hours that drops before epoch 1. Publish with P1's `promote=False`.

**Then evaluate, separately:** Lagyamfi test greedy and beam+LM (regression), and the court from-scratch clips with the same beam settings (the decision). The epoch that is compared is the one selected for the court set, not the Bible-dev best, once that comparison is possible. If the court set is still too small to pick an epoch, report the Bible-dev checkpoint on both sets and do not promote.

**Done when:** both numbers are in `results.md`. Promote only if the court from-scratch WER improves on a paired comparison of the same clips and Lagyamfi test beam has not fallen apart. "Fallen apart" means a move well outside the ±1.1-point band, not a 0.3-point wiggle.

**Runs on:** `knust-gpu`, GPU 0 only. The three training corpora are already there. Do not put the health corpus, WAXAL v1, the pseudo-label set, or the raw court audio on that machine for this run.

### P6 — Court language model

**Status:** blocked on P3. Independent of P5. Run it after P4 so there is court text, and do not combine it with the Fante drop in one run.

**Work:** a second character n-gram at `outputs/asr/lm/court_char.json`, built through `scripts/lm/build_lm.py` from the corrected court transcripts. Extra Asante text is allowed only when its distinct word types justify the lines, the same standard that rejected the synthetic corpus (634 types in 400,000 rows) and kept `crowdsourced_data.csv` (1,124 types in 695 lines). Decode the gold clips twice: shipped `twi_char.json`, then `court_char.json`. Acoustics stay v003, or the P5 checkpoint if that one was promoted.

**Done when:** both decodes are in `results.md`. The Bible LM remains the one used for the Lagyamfi regression. Serving switches only if the court decode is better and the regression holds.

### P7 — More court hours

**Status:** blocked until P4's from-scratch scores exist for two systems and a person has checked a sample of any proposed labels.

**Work:** not specified yet. The agreement filter and `scripts/asr/ctc_label.py` are the tools already in the repo. Pseudo-labels on WAXAL were a negative result (v004 beam 0.2975 against v003's 0.2887) on audio that had human transcripts. Court drafts without that check are not a training set.

**Health:** still the negative result in [`roadmap_next_steps.md`](../roadmap_next_steps.md). It is not a way around the Bible set. Adom TV is a different domain from court, and the Gemini labels are unchecked.

## What this plan does not do

- No further Bible-only fine-tune, and no checkpoint averaging, as the next experiment.
- No training on the current court drafts.
- No health-corpus training.
- No copy of datasets, the private key, or `.env` to the MacBook.
- No `--delete` sync between Ubuntu and `knust-gpu`.
