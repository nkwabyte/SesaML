#!/usr/bin/env python
"""
Prints a table of every recorded run from `outputs/runs/index.jsonl`.

Examples:
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --kind train --limit 10
    python scripts/summarize_runs.py --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_logger import load_run_index

COLUMNS = ("run_id", "kind", "status", "duration_sec", "best_metric", "final_wer", "final_cer")


def format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", default=None, help="Filter by run kind (train, evaluate, export, transcribe, download)")
    parser.add_argument("--limit", type=int, default=20, help="Show only the most recent N runs")
    parser.add_argument("--output-dir", default="outputs", help="Outputs directory to read")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit raw JSON instead of a table")
    args = parser.parse_args()

    runs = load_run_index(args.output_dir)
    if args.kind:
        runs = [run for run in runs if run.get("kind") == args.kind]
    runs = runs[-args.limit:]

    if not runs:
        print("No runs recorded yet. Start one with scripts/train.sh")
        return 0

    if args.as_json:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
        return 0

    rows = [[format_cell(run.get(column)) for column in COLUMNS] for run in runs]
    widths = [max(len(COLUMNS[i]), *(len(row[i]) for row in rows)) for i in range(len(COLUMNS))]

    header = "  ".join(name.ljust(widths[i]) for i, name in enumerate(COLUMNS))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
