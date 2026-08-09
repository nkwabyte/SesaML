# Outputs

Everything a run produces lands here. Each invocation of `src/main.py` (train,
evaluate, transcribe, export), `scripts/download_dataset.py`, or a session of
the Gradio app creates a run with an id like `train-20260809-101500` and writes
all of its artifacts under that id.

```
outputs/
├── logs/<run_id>.log            full text log of the run
├── runs/
│   ├── index.jsonl              one summary line per completed run
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
├── checkpoints/<run_id>/        training weights          — NOT in git
└── exports/<run_id>/            exported .pt/.pth/.pte    — NOT in git
```

## What is tracked

Logs, metrics, configs, predictions and summaries are committed so experiments
stay reviewable and comparable in git. Model binaries are not: `checkpoints/`
and `exports/` are ignored, as are `*.pt`, `*.pth`, `*.pte`, `*.ckpt`, `*.bin`,
`*.safetensors`, `*.onnx` and `*.tflite` anywhere in the tree. The rules live in
`outputs/.gitignore`.

Each `export_manifest.json` records the SHA-256 of its (untracked) binary, so a
committed run still identifies exactly which artifact it produced.

Run kinds are `train`, `evaluate`, `transcribe`, `export`, `download` and `app`.

## Browsing runs

```bash
python scripts/summarize_runs.py              # table of all runs
python scripts/summarize_runs.py --kind train --limit 5
less outputs/logs/train-20260809-101500.log
```

`metrics.csv` loads directly into pandas:

```python
import pandas as pd
pd.read_csv("outputs/runs/train-20260809-101500/metrics.csv")
```
