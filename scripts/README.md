# Scripts

Entry points for the common SesaML workflows. Every script can be run from any
directory — they resolve the repository root themselves, load `.env`, and
activate `.venv` when it exists.

```bash
chmod +x scripts/*.sh      # once, after cloning
```

| Script | What it does |
| --- | --- |
| `setup_env.sh` | Creates `.venv`, installs `requirements.txt`, seeds `.env` and the `outputs/` tree. |
| `download_dataset.sh` | Downloads a HuggingFace speech dataset into `data/` and writes a manifest to `outputs/`. |
| `train.sh` | Starts a training run; logs, metrics and checkpoints go to `outputs/`. |
| `evaluate.sh` | Scores a checkpoint (loss/WER/CER) and saves per-sample predictions. |
| `transcribe.sh` | Transcribes one audio file and stores the transcript. |
| `export_model.sh` | Exports a checkpoint to `.pt` / `.pth` / `.pte` plus a tracked manifest. |
| `run_tests.sh` | Runs the test suite (pytest, falling back to `tests/run_all_tests.py`). |
| `summarize_runs.py` | Prints a table of all recorded runs from `outputs/runs/index.jsonl`. |
| `common.sh` | Shared bash helpers; sourced by the others, not run directly. |

## Typical flow

```bash
scripts/setup_env.sh
scripts/download_dataset.sh --num-samples 100        # small slice first
scripts/train.sh --epochs 5 --batch-size 8
scripts/evaluate.sh                                  # uses the newest checkpoint
scripts/export_model.sh --format torchscript
python scripts/summarize_runs.py
```

`evaluate.sh`, `transcribe.sh` and `export_model.sh` default to the newest
`outputs/checkpoints/<run-id>/best_model.pt`. Pass `--model-path` to pin a
specific checkpoint.

## Configuration

`.env` (copied from `env.template`) is loaded automatically. Useful variables:

- `HF_TOKEN` — HuggingFace access token for gated datasets/models
- `MODEL_REPO_ID` — repository id used by the Gradio app in `app.py`

Scripts also honour a few overrides so you don't have to repeat flags:

```bash
EPOCHS=30 BATCH_SIZE=16 scripts/train.sh
HF_DATASET=some/other-corpus scripts/download_dataset.sh
MODEL_PATH=outputs/checkpoints/train-20260809-101500/best_model.pt scripts/evaluate.sh
```

Anything not consumed by a script is forwarded to the underlying CLI, so
`python -m src.main train --help` documents the full flag set.
