# Outputs

Everything a run produces lands here. Each invocation of `src/main.py` (train,
evaluate, transcribe, export), `scripts/datasets/download_dataset.py`, or a session of
the Gradio app creates a run with an id like `train-20260809-101500` and writes
all of its artifacts under that id.

Artifacts are grouped by **model domain**. `asr/` is speech recognition; phase
two's translation model becomes `outputs/translation/` beside it. Two models can
then both publish a `v001` without one overwriting the other. `logs/` and the
third-party `models/` cache stay shared.

```
outputs/
├── logs/<run_id>.log            full text log of the run        — SHARED
├── models/                      downloaded third-party weights  — SHARED
│                                (SpeechBrain ECAPA for diarization)
└── asr/                         speech recognition — the domain tree
    ├── runs/
    │   ├── index.jsonl          one summary line per completed run
    │   └── <run_id>/
    │       ├── config.json          config snapshot, CLI params, git commit, env
    │       ├── metrics.jsonl        every logged metric row, append-only
    │       ├── metrics.csv          same rows, flat, for spreadsheets/plots
    │       ├── history.json         per-epoch metrics (training runs)
    │       ├── model_summary.json   architecture, parameter counts, device
    │       ├── predictions/         per-epoch reference/hypothesis samples
    │       ├── predictions.json     evaluation reference/hypothesis samples
    │       ├── evaluation.json      loss / WER / CER for an evaluation run
    │       ├── transcript.txt       transcription result
    │       ├── dataset_manifest.json rows, columns, sample transcripts (downloads)
    │       ├── export_manifest.json exported artifact: format, size, SHA-256
    │       └── summary.json         status, duration, headline metrics
    ├── checkpoints/<run_id>/    training weights + model_meta.json — NOT in git
    │                            best_model.pt (weights), last_model.pt
    │                            (weights + optimizer + schedule, ~3x larger)
    ├── registry/                versioned model exports — the served models
    │   ├── registry.json        promotion pointer and history  — tracked
    │   └── <arch>/<version>/
    │       ├── model.pt         published weights              — NOT in git
    │       └── metadata.json    WER/CER, run id, SHA-256,
    │                            training features               — tracked
    ├── exports/<run_id>/        exported .pt/.pth/.pte          — NOT in git
    └── lm/twi_char.json         character n-gram LM             — NOT in git
```

`PathConfig.domain_dir` (default `outputs/asr`) is what every run path hangs
off. `outputs/.gitignore` is written against `*/` rather than a fixed domain
name, so a new domain inherits the right rules without anyone editing it.

## What is tracked

Run artifacts are machine-generated and accumulate quickly, so `runs/` and
`logs/` are **not** committed — a few runs a day would bury real diffs in churn.
Use `python scripts/maintenance/summarize_runs.py` to compare runs locally.

The registry is the exception: `registry.json` and every version's
`metadata.json` **are** tracked, because they record which model was served
when and at what WER. That provenance is worth keeping in git even though the
weights are not.

Model binaries are never committed: `checkpoints/`, `exports/` and `models/`
are ignored, as are `*.pt`, `*.pth`, `*.pte`, `*.ckpt`, `*.bin`,
`*.safetensors`, `*.onnx` and `*.tflite` anywhere in the tree. The rules live in
`outputs/.gitignore`.

Each `export_manifest.json` records the SHA-256 of its (untracked) binary, so a
committed run still identifies exactly which artifact it produced.

Run kinds are `train`, `evaluate`, `transcribe`, `export`, `download` and `app`.

## Browsing runs

```bash
python scripts/maintenance/summarize_runs.py              # table of all runs
python scripts/maintenance/summarize_runs.py --kind train --limit 5
less outputs/logs/train-20260809-101500.log
```

`metrics.csv` loads directly into pandas:

```python
import pandas as pd
pd.read_csv("outputs/asr/runs/train-20260809-101500/metrics.csv")
```

## Cleaning up

Training leaves three checkpoint files per run, and the resumable one is roughly
three times the size of the weights. After a handful of iterations that is
gigabytes of superseded runs.

```bash
python scripts/maintenance/clean_outputs.py                    # dry run: what would go
python scripts/maintenance/clean_outputs.py --apply            # delete it
python scripts/maintenance/clean_outputs.py --keep run-big --apply
python scripts/maintenance/clean_outputs.py --checkpoints-only --apply
```

It never touches `registry/`, and it protects the runs that produced published
versions — read out of the registry metadata rather than hard-coded, so
protection follows whatever has actually been published.

Run directories and logs are kept for real training runs and removed only for
smoke tests, verification and downloads. That is deliberate: a failed run's
weights are worthless, but its metrics are the evidence for why it failed.
