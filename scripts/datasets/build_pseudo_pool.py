#!/usr/bin/env python
"""
Selects and extracts a pool of untranscribed WAXAL recordings for pseudo-labelling.

`waxal/Akan_master.xlsx` lists 195,285 recordings (Batch1..Batch10) with no transcripts. They are
the largest data we have, but they overlap what is already labelled and held out, so a careless
draw leaks:

  * 99.9% of the transcribed clips are cut from recordings that are also in these batches, so
    those recordings are removed (`--transcribed` sheet);
  * every speaker in the WAXAL dev or test split is removed (`--heldout-speakers`), because the
    same voices describing the same photographs would put the held-out sets in the training data.

Selection is spread across speakers (`--max-per-speaker`) so that the pool adds voices rather than
repeating the few most prolific ones. Only batches whose audio can be carved out of `Akan_.tar`
by header offset are used (3 to 9; the tar is several tars glued end to end, so sequential `tar`
cannot reach them). The recordings are written as they are stored (16 kHz mp3, ~18 s each).

    python scripts/datasets/build_pseudo_pool.py --n 20000 --out data/waxal_pool

Needs `openpyxl`.
"""

import argparse
import csv
import json
import mmap
import random
import re
from collections import defaultdict
from pathlib import Path

BLOCK = 512
BATCH_DIR = re.compile(r"batch(\d+)_audios", re.I)
TRANSCRIBED_DIR = "selected transcribed audios"


def speaker_key(value) -> str:
    """Speaker ids compare equal across sources: the sheets hold floats (328.0), the datasets strings (328)."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def index_tar(tar_path: str) -> dict:
    """{basename: (batch, data_offset, size)} for batch mp3 members, found by header magic."""
    found, with_handle = {}, open(tar_path, "rb")
    mm = mmap.mmap(with_handle.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    while True:
        at = mm.find(b"ustar", pos)
        if at < 0:
            break
        pos = at + 5
        header = at - 257
        if header < 0 or header % BLOCK:
            continue
        name = mm[header:header + 100].split(b"\0")[0].decode("utf-8", "replace")
        if not name.lower().endswith(".mp3") or TRANSCRIBED_DIR in name:
            continue
        m = BATCH_DIR.search(name)
        if not m:
            continue
        try:
            size = int(mm[header + 124:header + 136].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            continue
        found.setdefault(name.rsplit("/", 1)[-1], (int(m.group(1)), header + BLOCK, size))   # first copy wins
    mm.close()
    with_handle.close()
    return found


def main() -> int:
    import openpyxl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default="data/waxal/Akan_master.xlsx")
    p.add_argument("--tar", default="data/Akan_.tar")
    p.add_argument("--heldout-speakers", default="data/waxal/heldout_speakers.txt")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--max-per-speaker", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/waxal_pool")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out.")
    (out / "audios").mkdir(parents=True, exist_ok=True)

    heldout = {speaker_key(l) for l in open(args.heldout_speakers, encoding="utf-8") if l.strip()}
    book = openpyxl.load_workbook(args.master, read_only=True)
    def rows(sheet):
        it = book[sheet].iter_rows(values_only=True)
        head = [str(c).strip() if c is not None else "" for c in next(it)]
        col = {n: head.index(n) for n in ("Filename", "SPEAKER_ID")}
        for r in it:
            if r[col["Filename"]]:
                yield str(r[col["Filename"]]), speaker_key(r[col["SPEAKER_ID"]])
    transcribed = {name for name, _ in rows("Transcribed")}
    print(f"{len(transcribed)} transcribed recordings, {len(heldout)} held-out speakers")

    recordings = []
    for sheet in book.sheetnames:
        if sheet.startswith("Batch"):
            recordings += [(name, spk, sheet) for name, spk in rows(sheet)]
    print(f"{len(recordings)} recordings listed in the batch sheets")

    available = index_tar(args.tar)
    print(f"{len(available)} batch mp3 members carvable from the tar")

    stats = defaultdict(int)
    kept = []
    for name, spk, sheet in recordings:
        if name in transcribed:
            stats["excluded: already transcribed"] += 1
        elif spk in heldout:
            stats["excluded: dev/test speaker"] += 1
        elif name not in available:
            stats["unavailable: not carvable from the tar"] += 1
        else:
            kept.append((name, spk, sheet))
    stats["eligible"] = len(kept)

    rng = random.Random(args.seed)
    by_speaker = defaultdict(list)
    for item in kept:
        by_speaker[item[1]].append(item)
    chosen = []
    for spk, items in by_speaker.items():
        rng.shuffle(items)
        chosen += items[: args.max_per_speaker]
    rng.shuffle(chosen)
    stats["eligible speakers"] = len(by_speaker)
    stats["after per-speaker cap"] = len(chosen)
    chosen = chosen[: args.n]

    # The exclusions are the whole point of this pool; a silent failure of either is a leak.
    leaked_speakers = [n for n, spk, _ in chosen if spk in heldout]
    leaked_clips = [n for n, _, _ in chosen if n in transcribed]
    if leaked_speakers or leaked_clips or not stats.get("excluded: dev/test speaker"):
        raise SystemExit(f"Exclusion check failed: {len(leaked_speakers)} recordings from held-out speakers, "
                         f"{len(leaked_clips)} already-transcribed recordings, "
                         f"{stats.get('excluded: dev/test speaker', 0)} removed by the speaker rule (expected > 0).")

    with open(args.tar, "rb") as handle, open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "speaker_id", "batch", "bytes"])
        for name, spk, sheet in chosen:
            _, offset, size = available[name]
            handle.seek(offset)
            data = handle.read(size)
            (out / "audios" / name).write_bytes(data)
            writer.writerow([f"audios/{name}", spk, sheet, size])

    est_hours = sum(available[n][2] for n, _, _ in chosen) / 6035 / 3600
    report = {**stats, "selected": len(chosen), "selected speakers": len({s for _, s, _ in chosen}),
              "estimated hours (bytes / 6,035 B/s)": round(est_hours, 1), "seed": args.seed,
              "by batch": {b: sum(1 for _, _, s in chosen if s == b) for b in sorted({s for _, _, s in chosen})}}
    (out / "pool_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
