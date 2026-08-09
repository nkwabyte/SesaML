#!/usr/bin/env python
"""
Downloads a HuggingFace speech dataset (default: the Ghana Open Data Twi corpus)
into the local `data/` directory and records a manifest under `outputs/`.

The audio itself stays in `data/` (git-ignored); only the manifest, row counts
and column schema are written to the run directory so downloads stay auditable.

Examples:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --split train --num-samples 100
    python scripts/download_dataset.py --dataset some/other-corpus --no-save-to-disk
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.utils.run_logger import RunManager, resolve_path

DEFAULT_DATASET = "ghanaopendata/twi-speech-text-multispeaker-16k"
TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset repository id")
    parser.add_argument("--config-name", default=None, help="Dataset configuration name, when the dataset defines several")
    parser.add_argument("--split", default=None, action="append", help="Split to download (repeatable). Defaults to every available split.")
    parser.add_argument("--cache-dir", default="data/hf_cache", help="HuggingFace cache directory")
    parser.add_argument("--save-dir", default="data/datasets", help="Directory the dataset is saved to on disk")
    parser.add_argument("--no-save-to-disk", action="store_true", help="Only populate the cache, skip save_to_disk()")
    parser.add_argument("--num-samples", type=int, default=None, help="Keep only the first N rows of each split (smoke tests)")
    parser.add_argument("--token", default=None, help="HuggingFace token (defaults to $HF_TOKEN)")
    parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")
    return parser


def resolve_splits(dataset: str, config_name, requested, token, logger):
    """Uses the requested splits, otherwise discovers what the dataset offers."""
    if requested:
        return requested
    try:
        from datasets import get_dataset_split_names
        splits = list(get_dataset_split_names(dataset, config_name=config_name, token=token))
        logger.info("Discovered splits: %s", ", ".join(splits))
        return splits
    except Exception as exc:
        logger.warning("Could not list splits (%s); falling back to 'train'", exc)
        return ["train"]


def preview_transcripts(split_dataset, limit: int = 3):
    """Reads a few transcripts without decoding audio columns."""
    text_column = next((c for c in split_dataset.column_names if c in TEXT_COLUMNS), None)
    if text_column is None:
        return text_column, []

    drop = [c for c in split_dataset.column_names if c != text_column]
    subset = split_dataset.select(range(min(limit, len(split_dataset)))).remove_columns(drop)
    return text_column, [str(row[text_column]) for row in subset]


def main() -> int:
    args = build_parser().parse_args()
    config = PipelineConfig()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    cache_dir = resolve_path(args.cache_dir)
    save_root = resolve_path(args.save_dir) / args.dataset.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)

    with RunManager(kind="download", config=config, run_id=args.run_id, params=vars(args)) as run:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("The 'datasets' package is required: pip install datasets") from exc

        run.logger.info("Dataset: %s", args.dataset)
        run.logger.info("Cache directory: %s", cache_dir)
        run.logger.info("Authenticated: %s", bool(token))

        splits = resolve_splits(args.dataset, args.config_name, args.split, token, run.logger)
        manifest = {
            "dataset": args.dataset,
            "config_name": args.config_name,
            "cache_dir": str(cache_dir),
            "save_dir": None if args.no_save_to_disk else str(save_root),
            "num_samples_per_split": args.num_samples,
            "splits": [],
        }

        for split in splits:
            run.logger.info("Loading split '%s'", split)
            split_dataset = load_dataset(
                args.dataset,
                name=args.config_name,
                split=split,
                cache_dir=str(cache_dir),
                token=token,
            )

            if args.num_samples:
                keep = min(args.num_samples, len(split_dataset))
                split_dataset = split_dataset.select(range(keep))
                run.logger.info("Truncated '%s' to %d rows", split, keep)

            target = None
            if not args.no_save_to_disk:
                target = save_root / split
                target.parent.mkdir(parents=True, exist_ok=True)
                split_dataset.save_to_disk(str(target))
                run.logger.info("Saved '%s' to %s", split, target)

            text_column, samples = preview_transcripts(split_dataset)
            split_info = {
                "split": split,
                "num_rows": len(split_dataset),
                "columns": list(split_dataset.column_names),
                "text_column": text_column,
                "saved_to": str(target) if target else None,
                "sample_transcripts": samples,
            }
            manifest["splits"].append(split_info)
            run.log_metrics({"split": split, "num_rows": len(split_dataset)}, stage="download")
            run.logger.info("Split '%s': %d rows, columns=%s", split, len(split_dataset), split_dataset.column_names)

        total_rows = sum(item["num_rows"] for item in manifest["splits"])
        manifest["total_rows"] = total_rows
        run.write_json("dataset_manifest.json", manifest)
        run.finish(status="completed", summary={
            "dataset": args.dataset,
            "total_rows": total_rows,
            "splits": [item["split"] for item in manifest["splits"]],
            "save_dir": manifest["save_dir"],
        })

        print("\n" + "=" * 40)
        print("DATASET READY:")
        print(f"Dataset:  {args.dataset}")
        print(f"Rows:     {total_rows}")
        print(f"Location: {manifest['save_dir'] or cache_dir}")
        print(f"Manifest: {run.run_dir / 'dataset_manifest.json'}")
        print("=" * 40)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
