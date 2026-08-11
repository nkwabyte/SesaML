# Speaker Diarization — Architecture & Implementation

How SesaML answers **"who spoke when"** (diarization) and joins it to **"what was said"** (ASR) to produce speaker-attributed Akan transcripts.

Implemented in [`src/diarization/`](../src/diarization/). Every number in this document was measured on this repository, on real Akan audio; where something is unverified it says so.

---

## 1. Pipeline

Diarization runs **first**, and each speaker turn is transcribed separately.

```
                          ┌──────────────────────┐
                          │     Audio input      │
                          │  (decoded once)      │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  1. Diarization backend         │
                    │     pyannote / ecapa / spectral │
                    └────────────────┬────────────────┘
                                     │  raw speaker turns
                    ┌────────────────▼────────────────┐
                    │  2. Turn post-processing        │
                    │     overlaps → merge → drop     │
                    │     short → relabel by order    │
                    └────────────────┬────────────────┘
                                     │  clean, non-overlapping turns
                    ┌────────────────▼────────────────┐
                    │  3. Per-turn ASR                │
                    │     slice waveform → Conformer  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  SPEAKER_00: ɔbɛtumi akɔdan...  │
                    │  SPEAKER_01: ti yɛn brɛ ne adom │
                    └─────────────────────────────────┘
```

### Why diarize first

The alternative — transcribe the whole file, then assign speakers to words — needs **word-level timestamps**. Our CTC decoder emits characters with no alignment output, so those timestamps do not exist. Diarizing first sidesteps the problem entirely and works with any ASR model.

The cost is that the ASR model sees each turn in isolation, losing cross-turn context. For a character-level CTC model with no language model, that costs almost nothing — it has no long-range context to lose. It would matter for Whisper.

---

## 2. Backends

Three, in descending order of quality and ascending order of how likely they are to run on a given machine. Selected with `--backend`, or `auto` to take the best that loads.

| Backend | Speaker model | Gated? | Load | Runtime (28s clip) | Use when |
|---|---|---|---|---|---|
| `pyannote` | `speaker-diarization-community-1` | **Yes** — 3 repos | ~4s | 6.9s | Accuracy matters; licences accepted |
| `ecapa` | SpeechBrain ECAPA-TDNN | No | ~66s first run | 7.1s | **Default.** No licence needed |
| `spectral` | MFCC statistics | No | instant | 0.01s | Offline, no downloads, last resort |

### pyannote

The neural pipeline the literature reports numbers for. Handles overlapped speech and estimates the speaker count itself.

**It is gated, and this bites hard.** The pipeline pulls *several* gated repositories, and accepting the conditions on the one named in the URL is not enough. In practice three were needed:

1. `pyannote/speaker-diarization-3.1`
2. `pyannote/segmentation-3.0`
3. `pyannote/speaker-diarization-community-1` ← pyannote.audio 4.x redirects here

Because the blocking repository is rarely the one you asked for, [`backends.py`](../src/diarization/backends.py) parses the 403 and names the repository the hub actually refused, following the exception chain.

> **API note:** pyannote.audio 3 returns an `Annotation`; version 4 returns a `DiarizeOutput` wrapper and has no `.itertracks()`. Both are handled. The `itertracks` call in the previous version of this document targets the 3.x API only.

### ecapa (default)

SpeechBrain ECAPA-TDNN embeddings over our own VAD, clustered agglomeratively on cosine distance. Public weights, no licence. Classic pre-neural pipeline: **it assigns exactly one speaker to every instant**, so it cannot represent two people talking at once. That is the main quality gap against pyannote.

### spectral

MFCC mean and standard deviation as the speaker embedding (coefficient 0 dropped — it tracks loudness, i.e. microphone distance, not identity). No downloads at all. It exists so the pipeline runs anywhere and a demo never hard-fails, not because it is competitive.

---

## 3. Turn post-processing

A diarizer's raw output is fragmented and overlapping. Feeding it straight to a CTC model produces a transcript chopped mid-word. [`turns.py`](../src/diarization/turns.py) applies four steps:

1. **`resolve_overlaps`** — every instant gets exactly one speaker.
   - *Partial* overlap → split at the midpoint.
   - *Containment* (a short interjection inside a long turn) → **split the long turn in two** so the interjection survives. Treating this as a partial overlap truncates the interjection to zero length and drops it, which is how one side of a conversation goes missing. This was a real bug, caught by a test.
2. **`merge_adjacent`** — rejoin same-speaker turns separated by ≤ 0.5s. Diarizers cut on brief pauses, so one sentence often arrives as four or five turns.
3. **`drop_short`** — discard turns under 0.35s. Almost always backchannels or diarizer jitter, and too few frames for CTC to decode.
4. **`relabel_by_first_appearance`** — rename to `SPEAKER_00, 01, …` in speaking order, so labels are stable between runs.

---

## 4. Voice activity detection

`pyannote` brings its own neural segmentation. The other two need speech regions, from [`segmentation.py`](../src/diarization/segmentation.py): frame RMS energy in dB, gated adaptively.

The threshold is anchored from **both ends** of the file's own energy distribution — `max(floor + 8dB, peak − 35dB)`:

- Anchoring on the noise floor alone fails on a recording that is mostly speech: the low percentile sits inside the speech itself and the gate never closes. *This was the first implementation, and it returned the entire file as one region.*
- Anchoring on the peak alone fails on a quiet recording with one loud moment.

Taking the stricter of the two handles both. Measured against ground truth on a 4-region test signal, boundaries land within **0.1s**.

Speech regions are then cut into 1.5s windows at 0.75s hop for embedding — long enough to embed stably, overlapped so a speaker change inside a window is not lost.

---

## 5. Choosing the speaker count

**This is the weak point of the embedding backends, and it should be stated plainly in any presentation.**

With `num_speakers` given, clustering is a fixed-size cut and is reliable. Without it, the cut is made on a cosine distance threshold, and that threshold does not generalise. Measured on a 28s Akan clip assembled from six different corpus recordings:

| ECAPA distance threshold | Speakers found |
|---|---|
| 0.60 | 6 |
| 0.75 | 5 |
| **0.90 (default)** | **3** — agrees with pyannote |

pyannote independently found 3 speakers and 7 turns on the same audio. The default was fitted to that agreement, on **one recording** — so:

- Pass `--num-speakers` whenever the count is known. The Gradio app exposes it as a slider.
- Prefer `pyannote` when it is not.

A same-speaker/different-speaker distance study on six corpus clips showed **overlapping distributions** (same-speaker up to 1.03, different-speaker from 0.55), but that ground truth was unreliable — the clips were picked by stride, with no speaker labels — so it should not be read as a measurement of ECAPA's separability.

---

## 6. Usage

### CLI

```bash
python -m src.main diarize --audio recording.wav                       # auto backend
python -m src.main diarize --audio recording.wav --backend pyannote --num-speakers 2
python -m src.main diarize --audio recording.wav --backend spectral --device cpu
```

Writes `diarization.json` and `transcript.txt` into `outputs/runs/<run_id>/`.

### Gradio app

```bash
scripts/serve_app.sh          # or: python app/app.py --share
```

The **Speaker Diarization** tab gives a colour-coded transcript, a proportional turn-taking timeline, a speaker-count slider, backend selection, and the raw JSON.

### Python

```python
from src.diarization import DiarizedTranscriber, format_transcript

pipeline = DiarizedTranscriber(model_type="deepspeech", backend="ecapa")
result = pipeline.transcribe("recording.wav", num_speakers=2)
print(format_transcript(result))
```

`DiarizedTranscriber` holds both models open — loading ECAPA and a Conformer checkpoint takes seconds, which is fine once and unacceptable per request.

### Output shape

```json
{
  "audio_seconds": 28.32,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "num_speakers": 3,
  "utterances": [
    {"speaker": "SPEAKER_00", "start": 0.03, "end": 7.59, "duration": 7.56,
     "transcript": "ɔbɛtumi akɔdan a ɛtɔa wɔn nsono ne onyankon adi..."}
  ],
  "empty_turns": 0,
  "timing": {"total_sec": 12.4, "diarization_sec": 6.85, "asr_sec": 5.5, "realtime_factor": 0.44},
  "asr": {"model_type": "deepspeech", "model_path": "...", "trained_weights_found": true},
  "diarization": {"backend": "pyannote"}
}
```

---

## 7. Measured end-to-end result

Trained Conformer (`run-clean`, WER 0.51 / CER 0.166) + pyannote, on a 28.3s Akan clip built from six corpus recordings:

```
[00:00.0 - 00:07.6] SPEAKER_00: ɔbɛtumi akɔdan a ɛtɔa wɔn nsono ne onyankon adi a wɔdi mfeɛ...
[00:08.3 - 00:08.8] SPEAKER_01: ucirie
[00:12.0 - 00:13.1] SPEAKER_01: ɔ o
[00:13.6 - 00:21.0] SPEAKER_02: awɔde asɔfodie nsafoɔ a no m ɛbeɛa na me yɛ atum afiri sooaa
[00:21.0 - 00:22.0] SPEAKER_00: ti yɛn brɛ ne adom
[00:22.0 - 00:24.7] SPEAKER_02: ahyɛdeɛ akyeɛ awurade adwuma mumoao
[00:26.5 - 00:27.8] SPEAKER_00: pafrei ɔdii adanseɛ s
```

Against the corpus reference for the first clip — *"Wɔbɛtumi akɔ dan a ɛtoa wɔn so no ne ne yɔnko…"* — the model produces *"ɔbɛtumi akɔdan a ɛtɔa wɔn nsono…"*: recognisably the same sentence.

**Read this honestly.** Turn boundaries are good and speaker attribution is plausible, but the source clips have no speaker labels, so "3 speakers" is not verified ground truth — only agreement between two independent backends.

---

## 8. Limitations

- **No overlapped-speech transcription.** Overlaps are resolved to one speaker per instant before ASR. pyannote *detects* overlap; we discard that information.
- **Automatic speaker counting is unreliable** on the embedding backends (§5).
- **No speaker identification** — labels are `SPEAKER_00`, not names. Enrolment against known voices is not implemented.
- **Turns under 0.35s are dropped**, so very short backchannels never appear.
- **ASR quality bounds everything.** At WER 0.51 the transcript is recognisable, not accurate; diarization cannot improve it.
- **Not evaluated with DER.** Diarization Error Rate needs labelled multi-speaker Akan audio, which none of the three corpora provide. Every diarization claim here is qualitative.

---

## 9. Production: async REST API

For long recordings, run the pipeline in a worker rather than a request. Sketch, not implemented:

```python
# models.py
class AudioTranscriptionJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.FileField(upload_to="audio_jobs/")
    status = models.CharField(max_length=20, default="PENDING", choices=[
        ("PENDING", "Pending"), ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"), ("FAILED", "Failed"),
    ])
    architecture = models.CharField(max_length=50, default="conformer")
    created_at = models.DateTimeField(auto_now_add=True)


class SpeakerUtterance(models.Model):
    # models.CASCADE, not a bare CASCADE - the previous draft would NameError.
    job = models.ForeignKey(AudioTranscriptionJob, related_name="utterances",
                            on_delete=models.CASCADE)
    speaker_label = models.CharField(max_length=50)
    start_time = models.FloatField()
    end_time = models.FloatField()
    transcript = models.TextField()
```

```python
# tasks.py
@shared_task
def process_diarized_transcription(job_id: str):
    job = AudioTranscriptionJob.objects.get(id=job_id)
    job.status = "PROCESSING"; job.save(update_fields=["status"])
    try:
        # Built once per worker process, not per job: loading the models is
        # seconds and would otherwise dominate every request.
        result = get_pipeline().transcribe(job.audio_file.path)
        SpeakerUtterance.objects.bulk_create([
            SpeakerUtterance(
                job=job, speaker_label=u["speaker"],
                start_time=u["start"], end_time=u["end"], transcript=u["transcript"],
            ) for u in result["utterances"]
        ])
        job.status = "COMPLETED"
    except Exception:
        job.status = "FAILED"
        raise
    finally:
        job.save(update_fields=["status"])
```

The worker must hold one `DiarizedTranscriber` per process. Constructing it per job would load ECAPA and the Conformer every time.

---

## 10. What changed from the first draft

The original design was sound; these were the corrections found while implementing it.

| Issue | Resolution |
|---|---|
| Called `Transcriber.transcribe_segment()` / `transcribe_waveform()` — **neither existed** | Both implemented; the pipeline uses `transcribe_waveform` so the file is decoded once, not per turn |
| `Transcriber(model_path=...)` / `Transcriber(architecture=...)` — wrong signature | Corrected throughout |
| `on_delete=CASCADE` — undefined name | `on_delete=models.CASCADE` |
| Assumed pyannote just works | It is gated behind **three** repos; two non-gated backends added so the demo never hard-fails |
| `annotation.itertracks()` | pyannote 4 returns `DiarizeOutput`; both APIs handled |
| No overlap handling | Overlaps resolved before ASR, with containment split correctly |
| Loaded the pipeline per call | Both models cached in `DiarizedTranscriber` and in the app |
| No speaker-count control | `--num-speakers` on the CLI, slider in the app |

---

## 11. Tests

[`tests/test_diarization.py`](../tests/test_diarization.py) — 27 tests, no network or downloads (they use the `spectral` backend):

- Turn algebra: merging, dropping, overlap resolution, containment, relabelling
- VAD: region detection against ground truth, all-speech files, silent files
- Windowing: overlap, short regions, slivers
- Backends: two-voice separation, explicit speaker count, monologue, sub-window audio
- Error reporting: gated-repo extraction, including through a chained cause
