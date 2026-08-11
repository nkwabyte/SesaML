# App

Gradio web front-end for the SesaML Akan speech-to-text models.

```
app/
├── app.py     the Gradio interface
└── info.md    this file
```

## Running it

```bash
scripts/serve_app.sh                 # http://127.0.0.1:7860
scripts/serve_app.sh --port 8080
scripts/serve_app.sh --share         # public Gradio tunnel
scripts/serve_app.sh --host 0.0.0.0  # reachable on your network
```

Or directly: `python app/app.py --share`.

## What it does

Upload a clip or record from the microphone, pick a backend, and get an Akan
transcript. Optional spectral-gate noise reduction is applied before inference.

| backend | weights |
| --- | --- |
| `deepspeech` | newest checkpoint in `outputs/checkpoints/`, resolved automatically |
| `whisper` | Optional comparison baseline — shown only when `$MODEL_REPO_ID` is set |

If no checkpoint has been trained yet, the app defaults to Whisper and shows a
warning on the DeepSpeech option — an untrained DeepSpeech model produces
gibberish rather than failing, which is easy to mistake for a broken model.

Models are loaded once per backend and cached, so only the first transcription
pays the load cost. This is why the app uses `Transcriber` from
[src/inference/transcribe.py](../src/inference/transcribe.py) rather than the
one-shot `transcribe_audio()` helper, which reloads weights on every call.

## Output

Every app session opens a run under `outputs/runs/app-<timestamp>/` and logs
each transcription — backend, elapsed time, word and character counts — to that
run's `metrics.jsonl`, with the full text log in `outputs/logs/`. See
[outputs/info.md](../outputs/info.md).

## Deployment

`app.py` exposes `demo` at module level, so it works unchanged as a HuggingFace
Space entrypoint. `MODEL_REPO_ID` is optional; set `HF_TOKEN` as a
Space secret.
