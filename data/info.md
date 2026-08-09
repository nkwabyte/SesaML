# Data

```
data/
├── info.md            this file                          — tracked
├── corpus/            checked-in Akan corpus & CSVs      — tracked
├── hf_cache/          HuggingFace download cache         — NOT tracked
└── datasets/          datasets saved via save_to_disk()  — NOT tracked
```

Only `corpus/` and this file are committed. Everything else under `data/` is
downloaded content and stays out of git — it is reproducible from the dataset
repositories and far too large to version.

## Fetching the corpora

```bash
scripts/download_dataset.sh                                        # ghanaopendata (public)
scripts/download_dataset.sh --dataset Lagyamfi/akan_audio_processed # gated, needs HF_TOKEN
```

Downloads land in `hf_cache/` and `datasets/<repo__name>/<split>/`. Each download
also writes a manifest (rows, columns, sample transcripts) to
`outputs/runs/<run_id>/dataset_manifest.json`, which **is** tracked — so the
provenance of a dataset survives in git even though its audio does not.

| corpus | clips | audio | clip length | domain | text column | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `ghanaopendata/twi-speech-text-multispeaker-16k` | 15,560 | ~19 h | ~4 s | religious | `text` | public |
| `Lagyamfi/akan_audio_processed` | 2,446 | ~3 h | 1.5–16 s | Bible | `sentence` | MIT, gated (`auto`) |
| `ghananlpcommunity/twi-health-asr-gemini-500hrs` | 59,291 | ~494 h | 30 s | health / media | `transcription` | CC-BY-NC-4.0 |

All three are 16 kHz Twi and share the same orthography, so they concatenate
cleanly. The transcription column is auto-detected — see
[src/info.md](../src/info.md).

`Lagyamfi/akan_audio_processed` reports 26,906 rows across 22 splits, but that is
the same 2,446 clips duplicated 11× (ten `*_Aug` augmentation copies). The
downloader skips those by default; see [scripts/info.md](../scripts/info.md).

## The health corpus

`ghananlpcommunity/twi-health-asr-gemini-500hrs` is 25× more audio than the other
two combined, and the only corpus outside the religious domain. It is the single
biggest available win for this model — but it is opt-in, for three reasons.

**30-second clips.** Every clip is exactly 30 s, versus ~4 s elsewhere. That is
~2,400 mel frames instead of ~360 — a 6.7× longer sequence through five
bidirectional GRU layers. At the default `--batch-size 10` this will exhaust GPU
memory; start at `--batch-size 2`. Training runs probe each corpus and warn about
this before the first batch.

**Digits — handled.** 22% of its transcripts contain digits (dosages, dates,
phone numbers). The CTC vocabulary originally had no digit symbols, so
`2 capsules` became ` capsules` in the label while the audio still said it. The
vocabulary now includes `0-9`, lifting coverage from 95.8% to 96.3%. The residual
trade-off: the model must learn that the sound of *mmienu* maps to the character
`2`. Spelling numbers out in Twi would be cleaner, and remains the better fix if
you have a number-to-words mapping.

**Licence and labels.** CC-BY-NC-4.0 is non-commercial, unlike the other corpora
— a model trained on it inherits that restriction. The `gemini` in its name
indicates machine-generated transcriptions rather than human-verified ones, so
expect some label noise beyond the digit issue.

Transcripts also code-switch heavily into English (`multimedia`, `acknowledge`,
`capsules`), which the Latin-alphabet vocabulary handles fine and which reflects
how Twi is actually spoken.

```bash
scripts/download_dataset.sh --dataset ghananlpcommunity/twi-health-asr-gemini-500hrs --num-samples 200
```

Start with `--num-samples`: the full corpus is 57 GB.
