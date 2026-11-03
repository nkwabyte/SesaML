#!/usr/bin/env python
"""
Turns pseudo-labelled clips into a trainable HuggingFace dataset, keeping only the confident ones.

Input is the JSONL written by scripts/asr/ctc_label.py over a pool from build_pseudo_pool.py.
Each kept clip becomes a row with the model's transcript as `sentence`, written exactly like the
real corpora (16 kHz mono FLAC bytes, `data/datasets/<repo__name>/<split>`), so training takes
`--hf-dataset <name>:train` with no loader changes. The rows are marked `pseudo=True` and carry
the teacher's confidence, because these transcripts are model output, not ground truth, and must
never appear in an evaluation set.

Selection keeps the top `--keep` fraction by `seq_lp` (log-probability per character), and drops
clips whose speech rate is implausible or whose transcript is too short to be useful. Held-out
speakers are re-checked here as a second guard, independently of the pool builder.

    python scripts/datasets/build_pseudo_dataset.py --labels outputs/asr/pseudo/waxal_pool.jsonl \\
        --pool data/waxal_pool --keep 0.4 --name ug-waxal/pseudo-v1
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_waxal_dataset import TARGET_RATE, encode_clip  # noqa: E402

MIN_CHARS = 20
CHARS_PER_SEC = (4.0, 20.0)


def speaker_key(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def main() -> int:
    from datasets import Audio, Dataset, Features, Value

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="JSONL from ctc_label.py")
    p.add_argument("--pool", required=True, help="Pool directory (manifest.csv + audios/)")
    p.add_argument("--keep", type=float, default=0.4, help="Share of clips to keep, most confident first")
    p.add_argument("--name", default="ug-waxal/pseudo-v1")
    p.add_argument("--out-dir", default="data/datasets")
    p.add_argument("--heldout-speakers", default="data/waxal/heldout_speakers.txt")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    save_root = Path(args.out_dir) / args.name.replace("/", "__")
    if save_root.exists() and any(save_root.iterdir()):
        sys.exit(f"{save_root} already exists; choose a new --name.")
    pool = Path(args.pool)
    speakers = {Path(r["audio_path"]).name: speaker_key(r["speaker_id"])
                for r in csv.DictReader(open(pool / "manifest.csv", encoding="utf-8"))}
    heldout = {speaker_key(l) for l in open(args.heldout_speakers, encoding="utf-8") if l.strip()}

    records = [r for r in map(json.loads, open(args.labels, encoding="utf-8")) if "teacher" in r]
    usable = [r for r in records
              if len(r["teacher"]) >= MIN_CHARS and CHARS_PER_SEC[0] <= r["chars_per_sec"] <= CHARS_PER_SEC[1]
              and r["seq_lp"] > float("-inf") and Path(r["path"]).name in speakers]
    usable.sort(key=lambda r: r["seq_lp"], reverse=True)
    kept = usable[: int(len(usable) * args.keep)]
    leaked = [r for r in kept if speakers[Path(r["path"]).name] in heldout]
    if leaked:
        sys.exit(f"{len(leaked)} selected clips belong to held-out speakers; refusing to build.")
    print(f"{len(records)} labelled | {len(usable)} usable (length/rate filter) | keeping top {args.keep:.0%} = {len(kept)}")

    paths = [str(pool / "audios" / Path(r["path"]).name) for r in kept]
    with ProcessPoolExecutor(args.workers) as pool_exec:
        encoded = list(pool_exec.map(encode_clip, paths, chunksize=32))

    features = Features({
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "sentence": Value("string"), "speaker_id": Value("string"), "seconds": Value("float32"),
        "seq_lp": Value("float32"), "pseudo": Value("bool"),
    })
    rows = [(r, e) for r, e in zip(kept, encoded) if e[2] > 1e-4]
    data = Dataset.from_dict({
        "audio": [{"bytes": e[0], "path": Path(r["path"]).name} for r, e in rows],
        "sentence": [r["teacher"] for r, _ in rows],
        "speaker_id": [speakers[Path(r["path"]).name] for r, _ in rows],
        "seconds": [e[1] for _, e in rows],
        "seq_lp": [r["seq_lp"] for r, _ in rows],
        "pseudo": [True] * len(rows),
    }, features=features).cast_column("audio", Audio(sampling_rate=TARGET_RATE, decode=False))
    data.save_to_disk(str(save_root / "train"))
    report = {"name": args.name, "clips": len(rows), "hours": round(sum(e[1] for _, e in rows) / 3600, 1),
              "speakers": len({speakers[Path(r['path']).name] for r, _ in rows}), "keep": args.keep,
              "lowest_kept_seq_lp": round(min(r["seq_lp"] for r, _ in rows), 4),
              "median_seq_lp": round(sorted(r["seq_lp"] for r, _ in rows)[len(rows) // 2], 4),
              "labels": args.labels, "note": "pseudo-labels: model transcripts, never an evaluation set"}
    (save_root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
