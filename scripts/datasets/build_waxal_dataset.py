#!/usr/bin/env python
"""
Builds a trainable HuggingFace dataset from the clips recovered by extract_akan_zip.py.

Training reads corpora from `data/datasets/<repo__name>/<split>` (save_to_disk), as
undecoded audio bytes plus a text column, so this writes exactly that and
`--hf-dataset ug-waxal/akan-transcribed:train` works with no loader changes.

Each clip is decoded, mixed to mono, resampled to 16 kHz and stored as FLAC: the
source is 44.1 kHz stereo mp3, about 8x the bytes the model can use.

Splits are by speaker. Unlike Lagyamfi, this corpus carries a speaker id, and a
random split would put the same voice on both sides of the line and flatter the
held-out score. Clips describing the same photograph still recur across splits
(the file names carry an image number), so topic vocabulary overlaps even though
voices do not; the report says so rather than hiding it.

With --keep-splits the dev and test sets of an earlier build are frozen exactly (same clips),
so scores measured on them stay comparable. Clips that were not in that build are added to
train, except those from dev/test speakers, which are left out: putting a held-out speaker's
new clips in train would break the speaker separation the split exists to guarantee.

Reports CTC feasibility alongside, because ~19 s clips under one long transcript
are the shape that failed on the health corpus.

Examples:
    python scripts/datasets/build_waxal_dataset.py
    python scripts/datasets/build_waxal_dataset.py --limit 200 --name smoke/waxal --workers 4
"""

import argparse
import csv
import io
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TARGET_RATE = 16000
SILENCE_PEAK = 1e-4
# Spellings that mark Fante rather than Asante Twi. A heuristic, not a dialect
# classifier: it flags clips for a native speaker, and the training split can drop them.
FANTE_MARKERS = re.compile(r"dz|\bebien|\benyim|\bekyir|\bebiasa|\bebaasa")


def encode_clip(path: str):
    """(flac bytes, seconds, peak) for one source clip, resampled to 16 kHz mono."""
    import numpy as np
    import soundfile as sf
    import soxr

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != TARGET_RATE:
        mono = soxr.resample(mono, rate, TARGET_RATE)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    buffer = io.BytesIO()
    sf.write(buffer, mono, TARGET_RATE, format="FLAC")
    return buffer.getvalue(), len(mono) / TARGET_RATE, peak


def split_by_speaker(rows, dev_frac: float, test_frac: float, seed: int):
    """Assigns whole speakers to test, then dev, then train, by clip share."""
    by_speaker = defaultdict(list)
    for index, row in enumerate(rows):
        by_speaker[row["speaker_id"]].append(index)
    speakers = sorted(by_speaker)
    random.Random(seed).shuffle(speakers)

    total = len(rows)
    quota = {"test": test_frac * total, "dev": dev_frac * total}
    assignment = {"train": [], "dev": [], "test": []}
    for speaker in speakers:
        clips = by_speaker[speaker]
        # Skip a speaker who would overshoot a quota badly (one voice has 558 clips).
        for name in ("test", "dev"):
            if len(assignment[name]) + len(clips) <= quota[name] * 1.25 and len(assignment[name]) < quota[name]:
                assignment[name].extend(clips)
                break
        else:
            assignment["train"].extend(clips)
    return assignment


def frozen_assignment(rows, previous: Path):
    """
    Indices per split that reproduce an earlier build's dev/test and send every other
    clip from a train speaker to train. Returns (assignment, excluded_clip_count).
    """
    from datasets import load_from_disk

    speaker_split, frozen = {}, {}
    for split in ("train", "dev", "test"):
        data = load_from_disk(str(previous / split))
        for speaker in set(data["speaker_id"]):
            speaker_split[str(speaker)] = split
        if split != "train":
            frozen[split] = {a["path"] for a in data.remove_columns(
                [c for c in data.column_names if c != "audio"])["audio"]}
    assignment = {"train": [], "dev": [], "test": []}
    excluded = 0
    for index, row in enumerate(rows):
        name = Path(row["audio_path"]).name
        for split in ("dev", "test"):
            if name in frozen[split]:
                assignment[split].append(index)
                break
        else:
            if speaker_split.get(str(row["speaker_id"]), "train") == "train":
                assignment["train"].append(index)
            else:
                excluded += 1       # a dev/test speaker's clip that was not in the frozen set
    for split in ("dev", "test"):
        if len(assignment[split]) != len(frozen[split]):
            sys.exit(f"Frozen {split} set has {len(frozen[split])} clips but only {len(assignment[split])} "
                     "were found in this manifest; refusing to change the held-out set.")
    return assignment, excluded


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/waxal_akan", help="Output directory of extract_akan_zip.py")
    p.add_argument("--name", default="ug-waxal/akan-transcribed", help="Dataset name; saved under data/datasets/<name with / as __>")
    p.add_argument("--out-dir", default="data/datasets")
    p.add_argument("--dev-frac", type=float, default=0.04)
    p.add_argument("--test-frac", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-seconds", type=float, default=35.0, help="Drop clips longer than this")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="Use at most N clips (smoke test)")
    p.add_argument("--keep-splits", default=None, metavar="DIR",
                   help="An earlier build (data/datasets/<name>): freeze its dev/test clips, add new clips to train")
    return p


def main() -> int:
    from datasets import Audio, Dataset, Features, Value

    from src.asr.data.feasibility import assess
    from src.asr.data.text_transform import TextTransform

    args = build_parser().parse_args()
    root = Path(args.root)
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    print(f"{len(rows)} manifest rows")

    paths = [str(root / r["audio_path"]) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        encoded = list(pool.map(encode_clip, paths, chunksize=32))

    dropped = {"silent": 0, "too_long": 0}
    keep = []
    for row, (blob, seconds, peak) in zip(rows, encoded):
        if peak < SILENCE_PEAK:
            dropped["silent"] += 1
        elif seconds > args.max_seconds:
            dropped["too_long"] += 1
        else:
            keep.append({**row, "bytes": blob, "seconds": seconds})
    print(f"kept {len(keep)} | dropped {dropped}")

    save_root = Path(args.out_dir) / args.name.replace("/", "__")
    if save_root.exists() and any(save_root.iterdir()):
        sys.exit(f"{save_root} already exists; choose a new --name (a running job may have it memory-mapped).")
    excluded = 0
    if args.keep_splits:
        assignment, excluded = frozen_assignment(keep, Path(args.keep_splits))
        print(f"frozen dev/test from {args.keep_splits}; {excluded} new clips from dev/test speakers left out")
    else:
        assignment = split_by_speaker(keep, args.dev_frac, args.test_frac, args.seed)
    earlier = root / "manifest.before_tar_recovery.csv"
    from_zip = {Path(r["audio_path"]).name for r in csv.DictReader(open(earlier, encoding="utf-8"))} if earlier.exists() else None
    transform = TextTransform()
    # Built as a plain struct and cast afterwards: Audio.encode_example imports
    # torchcodec, which needs an FFmpeg this machine does not have, while the
    # storage cast does not.
    features = Features({
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "sentence": Value("string"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "seconds": Value("float32"),
        "fante_marker": Value("bool"),
        "source": Value("string"),
    })

    report = {"name": args.name, "seed": args.seed, "dropped": dropped,
              "kept_splits_from": args.keep_splits, "new_clips_from_heldout_speakers_excluded": excluded, "splits": {}}
    for split, indices in assignment.items():
        items = [keep[i] for i in indices]
        data = Dataset.from_dict({
            "audio": [{"bytes": it["bytes"], "path": Path(it["audio_path"]).name} for it in items],
            "sentence": [it["text"] for it in items],
            "speaker_id": [str(it["speaker_id"]) for it in items],
            "gender": [it["gender"] for it in items],
            "seconds": [it["seconds"] for it in items],
            "fante_marker": [bool(FANTE_MARKERS.search(it["text"].lower())) for it in items],
            # zip clips are 44.1 kHz originals; tar clips are 16 kHz re-encodes of the same recordings
            "source": ["unknown" if from_zip is None else ("zip" if Path(it["audio_path"]).name in from_zip else "tar") for it in items],
        }, features=features).cast_column("audio", Audio(sampling_rate=TARGET_RATE, decode=False))
        data.save_to_disk(str(save_root / split))

        feas = assess([it["seconds"] for it in items], [transform.text_to_int(it["text"]) for it in items],
                      TARGET_RATE, 160, 4)
        report["splits"][split] = {
            "clips": len(items),
            "hours": round(sum(it["seconds"] for it in items) / 3600, 2),
            "speakers": len({it["speaker_id"] for it in items}),
            "fante_marker_share": round(sum(bool(FANTE_MARKERS.search(it["text"].lower())) for it in items) / max(1, len(items)), 3),
            "feasibility_4x": {"infeasible": feas.infeasible, "empty": feas.empty, "tight": feas.tight,
                               "median_headroom": round(feas.median_headroom, 2), "p05_headroom": round(feas.p05_headroom, 2)},
        }
        print(split, report["splits"][split])

    speakers = {s: {it["speaker_id"] for it in (keep[i] for i in idx)} for s, idx in assignment.items()}
    report["speaker_overlap"] = {"train&dev": len(speakers["train"] & speakers["dev"]),
                                 "train&test": len(speakers["train"] & speakers["test"]),
                                 "dev&test": len(speakers["dev"] & speakers["test"])}
    images = lambda idx: {re.search(r"image_(\d+)", keep[i]["audio_path"]).group(1) for i in idx if re.search(r"image_(\d+)", keep[i]["audio_path"])}
    report["image_overlap_with_train"] = {s: len(images(assignment[s]) & images(assignment["train"])) for s in ("dev", "test")}
    report["images_in_split"] = {s: len(images(assignment[s])) for s in assignment}
    (save_root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("speaker overlap:", report["speaker_overlap"], "| image overlap with train:", report["image_overlap_with_train"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
