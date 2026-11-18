#!/usr/bin/env python
"""
Fills the gaps left by extract_akan_zip.py with clips carved out of `Akan_.tar`.

`Akan.zip` lost 2,729 of its 18,787 transcribed clips, but the same clips sit in the
tail of `Akan_.tar` under `selected transcribed audios/`. Plain `tar` cannot reach
them: the file is several tars glued end to end, and GNU tar stops at the first
junction. So the headers are located directly (the `ustar` magic at byte 257 of each
512-byte block) and each member's bytes are read from just after its header.

Tar stores no checksum of the payload, so recovered clips are verified two ways: each
must decode in full, and a sample of clips present in both the zip extraction and the
tar are compared as audio. They are not byte-identical - the tar copies are 16 kHz,
lower-bitrate re-encodes of the zip's 44.1 kHz clips - so "agree" means equal duration
and a waveform correlation above 0.9 once both are resampled to 16 kHz (a clip against
itself shifted by a wrong offset scores ~0). The zip copies are the higher-fidelity ones
and are kept wherever both exist.

Appends to an existing extraction (the --root written by extract_akan_zip.py) and never
overwrites a file. Needs `soundfile`, `soxr`, `numpy` and `openpyxl`.

    python scripts/datasets/recover_akan_tar.py --tar data/Akan_.tar --root data/waxal_akan
"""

import argparse
import csv
import json
import mmap
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_akan_zip import fold  # noqa: E402

MEMBER_DIR = "selected transcribed audios"
BLOCK = 512


def scan_members(tar_path: str, wanted: set):
    """{basename: (data_offset, size)} for wanted mp3 members, found by header magic."""
    found = {}
    with open(tar_path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
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
            base = name.rsplit("/", 1)[-1]
            if base not in wanted or MEMBER_DIR not in name:
                continue
            try:
                size = int(mm[header + 124:header + 136].split(b"\0")[0].strip() or b"0", 8)
            except ValueError:
                continue
            found.setdefault(base, (header + BLOCK, size))
        mm.close()
    return found


def same_audio(zip_bytes: bytes, tar_bytes: bytes) -> bool:
    """True when two encodings are the same recording: same length, correlated waveform."""
    import io

    import numpy as np
    import soundfile as sf
    import soxr

    def load(blob):
        samples, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        return soxr.resample(mono, rate, 16000) if rate != 16000 else mono

    a, b = load(zip_bytes), load(tar_bytes)
    if abs(len(a) - len(b)) > 1600:
        return False
    n = min(len(a), len(b)) - 6000
    best = max(np.corrcoef(a[3000 + s:3000 + s + n], b[3000:3000 + n])[0, 1] for s in range(-2000, 2001, 100))
    return bool(best > 0.9)


def main() -> int:
    import io

    import openpyxl
    import soundfile as sf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar", default="data/Akan_.tar")
    p.add_argument("--root", default="data/waxal_akan", help="Existing extract_akan_zip.py output")
    p.add_argument("--cross-check", type=int, default=400, help="Clips present in both sources to compare byte for byte")
    args = p.parse_args()

    root = Path(args.root)
    audio_dir = root / "audios"
    have = {f.name for f in audio_dir.iterdir()}

    rows = openpyxl.load_workbook(root / "transcripts.xlsx", read_only=True).active.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    col = {n: header.index(n) for n in ("Full Filename", "Transcriptions", "SPEAKER_ID", "GENDER", "AGE")}
    table = {r[col["Full Filename"]]: r for r in rows if r[col["Full Filename"]]}
    missing = {n for n in table if n not in have}
    print(f"{len(table)} transcribed clips in the sheet; {len(have)} extracted; {len(missing)} to recover")

    index = scan_members(args.tar, set(table))
    print(f"{len(index)} of the sheet's clips found in the tar")

    with open(args.tar, "rb") as handle:
        def payload(name):
            offset, size = index[name]
            handle.seek(offset)
            return handle.read(size)

        recovered, undecodable, written = [], [], 0
        with open(root / "manifest.csv", "a", newline="", encoding="utf-8") as manifest:
            writer = csv.writer(manifest)
            for name in sorted(missing & set(index)):
                data = payload(name)
                try:
                    with io.BytesIO(data) as buffer:
                        sf.read(buffer, dtype="float32")
                except Exception:
                    undecodable.append(name)
                    continue
                text, changed = fold(str(table[name][col["Transcriptions"]] or "").strip())
                if not text:
                    continue
                (audio_dir / name).write_bytes(data)
                r = table[name]
                writer.writerow([f"audios/{name}", text, r[col["SPEAKER_ID"]], r[col["GENDER"]], r[col["AGE"]]])
                recovered.append(name)
                written += changed

        both = sorted((set(table) & have) & set(index))
        sample = random.Random(0).sample(both, min(args.cross_check, len(both)))
        agree = sum(same_audio((audio_dir / n).read_bytes(), payload(n)) for n in sample)

    report = {
        "sheet_clips": len(table), "already_extracted": len(have), "found_in_tar": len(index),
        "recovered_and_verified_decodable": len(recovered), "recovered_but_undecodable": len(undecodable),
        "still_missing": len(missing - set(recovered)),
        "cross_check_sampled": len(sample), "cross_check_same_audio": agree,
        "lookalike_characters_folded": written,
    }
    (root / "recover_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
