# Adding `ghanaopenai/twi-multispeaker-tts`

Recorded 2026-11-02. This dataset is a candidate **addition** to the training mix. It does not replace Lagyamfi, WAXAL v2, ghanaopendata, the lab code-switch set, or the court recordings. Nothing here has been downloaded or trained.

The product goal is one acoustic model that transcribes Asante Twi and handles English mixed into that Twi. This corpus is spontaneous broadcast speech with that mix. The court sheet is still the set that decides a promotion. See [`court_flow.md`](court_flow.md).

## What was checked

It is not on this Ubuntu machine and not on `knust-gpu`. A name search of `data/` and the GPU `data/` tree found no `ghanaopenai` path. The corpus already in training, `ghanaopendata/twi-speech-text-multispeaker-16k`, is a different dataset.

Hub page checked 2026-11-02 (`huggingface.co/datasets/ghanaopenai/twi-multispeaker-tts`, revision `7024f08d40194a002feb8004f65983e544a5f18f`, card last modified 2026-10-13):

| | |
|---|---|
| License | CC-BY-NC-4.0 |
| Split | `train` only, 10,960 clips |
| Duration claimed | 91.33 hours |
| Download size on the card | 5.83 GB parquet. Hub storage reported 11.7 GB |
| Audio | 16 kHz, Demucs-isolated vocals |
| Text | `transcription` (source transcript) and `griot_transcription` (`Qlerqly/griot-nano-1` via Sherpa-ONNX). Also `ipa` |
| Quality columns | `utmos`, DNSMOS (`dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovr`), `spk_count`, `spk_min_sim`, `spk_med_sim`, `speech_duration`, `speech_ratio` |
| Card quality filter | median UTMOS 2.05. `utmos >= 2.2` is 33.50 h / 4,040 clips. `utmos >= 2.5` is 11.30 h / 1,360 clips |

Eight rows were read from the datasets server (indexes 0–7), not a corpus-wide sample. Every one of those rows has `duration` 30.0. Speech inside the window ranges from 6.4 s to 29.0 s. The text is spontaneous Asante Twi with English left in English (`household name`, `create awareness`, `your health is your wealth`, `Adom TV`, `Franko`). Clip ids look like YouTube ids (`-EuNwtB64u4_005`). On those eight rows the Griot column is not cleaner than `transcription`. That is not a measured corpus comparison.

Overlap with `ghananlpcommunity/twi-health-asr-gemini-500hrs` was not checked. Both are fixed 30-second broadcast windows and the sample names Adom TV, so treat overlap as unknown until the ids are compared.

## Why it is worth adding

The served recipe is mostly read or prompted speech: Bible, short ghanaopendata clips, and WAXAL picture descriptions. This set is the other kind. It has audio and a transcript for speech that is not scripture, and the English is in the sentence rather than in a separate code-switch script. `conformer-medium/v005` already showed that 13.4 hours of scripted code-switch can sit in one model without moving Lagyamfi test (beam 0.2933 against 0.2887) while cutting a code-switch holdout from 0.8691 to 0.6597 greedy. Broadcast code-switching is the hours that set does not have.

The license is non-commercial. A training run for this research project can use it. Do not ship a commercial system on the assumption that this license allows it.

## Why the raw clips cannot go into the loader

A 100-hour slice of the same shape, the health corpus, trained to word error 1.0. Those clips are exactly 30.00 s, about seven speech regions under one ~394-character transcript. CTC never found the boundaries. [`results.md`](results.md) §3 and [`roadmap_next_steps.md`](../roadmap_next_steps.md) Phase 1.1 record that. This card describes the same window length. Adding these 10,960 files as one example each would repeat that run.

`src/asr/data/segmenting.py` can cut a window only when the text and the audio describe the same speech. Both text columns here are machine output. Aligning the wrong string splits the audio in the wrong places and trains the error in. The health aligner, a scripture model, had a median score of −6.7 on broadcast audio. Do not treat a low score as a successful cut.

## Work for the next agent

Do this on the Ubuntu machine. Do not copy the dataset to the MacBook. Do not put it on `knust-gpu` until a training run is actually launched. The GPU server keeps only the three corpora the current recipe reads, until that run.

One change when training finally happens. Do not combine this addition with the Fante-marked WAXAL drop or a new language model.

### 1. Download and probe

Save it where the loader already looks: `data/datasets/ghanaopenai__twi-multispeaker-tts`. Decode audio the way `src/asr/data/hf_dataset.py` does (`Audio(decode=False)` and `src/utils/audio_io.py`). Do not call `torchaudio.load`.

Before any segment or training step, print:

- Duration histogram. Confirm whether every clip is exactly 30 s or the eight-row sample was misleading.
- Speech regions per clip, with the same VAD count used on the health corpus.
- CTC headroom of each text column (`src/asr/data/feasibility.py`, `--min-headroom 1.0`).
- English-token rate, and the share of lines with a Fante-style marker, using the same marker test as `scripts/datasets/build_waxal_dataset.py`.
- Distinct word types in each text column. Row count is not the measure.
- Character error rate between `transcription` and `griot_transcription`. The card calls Griot the accurate column. The eight rows above do not show that. Pick the column the alignment will trust only after this number, and say which column in the run log.
- How many YouTube-style ids also occur in the health corpus, if that corpus stores ids. Overlap means these hours are not new.

Write the probe numbers into this file. They are not known yet.

### 2. Cut utterances, then check a sample by ear

Split each 30-second window at pauses so each training example is one utterance with the text spoken in it. Drop cuts whose alignment score is in the range that failed on health. Drop clips whose two transcripts disagree badly, unless a person has chosen the right one.

Listen to a sample of the surviving pairs before calling the hours real. The health labels are still unchecked for the same reason: a model score on broadcast audio cannot tell a bad transcript from a domain shift.

### 3. Train only the segmented subset

Add the segmented split as one more `--hf-dataset` on top of the current mix (ghanaopendata train, Lagyamfi `train_core`, WAXAL v2 train) and keep the code-switch `train_cap7` slice in the mix, so the one model does not forget English-in-Twi. Initialise from `conformer-medium/v005` if the goal is that single code-switching model, and say so in the run id. Validation can stay Lagyamfi dev so the curve exists. That dev number must not promote.

`Trainer._publish` still promotes when dev greedy beats v003's stored test greedy of 0.335. Until [`court_flow.md`](court_flow.md) P1 passes `promote=False`, roll the pointer back to the previous served version before anyone uses the demo. Score Lagyamfi test as the regression (hold the ±1.1 band) and the 434-clip code-switch holdout as the code-switch number. Score the court from-scratch clips when the sheet exists. Do not promote on Lagyamfi alone.

The text columns are not a language-model source until a person has checked them. Machine transcripts in the character model teach the recogniser its own mistakes. That experiment is separate from the acoustic addition.

## Status

Not started. Not downloaded. No training run. The court sheet is still empty (120 rows, `corrected` filled on 0 as of 2026-11-02) and this dataset does not unblock it.
