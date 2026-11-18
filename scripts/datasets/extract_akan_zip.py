#!/usr/bin/env python
"""
Recovers the transcribed WAXAL Akan clips from `Akan.zip` and writes a training manifest.

`Akan.zip` is two archives joined end to end: a 1.96 GB fragment of the original
zip, then a complete zip whose central directory is the one readers see. Every
tool applies the fragment's length as a uniform offset shift, which is right for
the second archive and wrong for the entries whose data sits in the fragment -
unzip and Python's zipfile fail those with "Bad magic number". Measured on the
shipped file: 16,057 of 18,787 clips are recovered and verified, resolving at
either the shifted or the unshifted offset, and 2,730 are not in the file at
all (the fragment is cut off before them).

Each entry is therefore tried at both offsets and accepted only when the local
header name, the CRC and the size all agree with the central directory. Nothing
unverified is written.

Writes only into --out, which must not already hold files, using base names
only so an archive path cannot escape it. The spreadsheet join and the Unicode
fold need `openpyxl`, which is not a project dependency.

Examples:
    python scripts/datasets/extract_akan_zip.py --out data/waxal_akan
    python scripts/datasets/extract_akan_zip.py --out /tmp/probe --limit 50
"""

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

AUDIO_MEMBER = "/audios/"
SHEET_MEMBER = "selected transcribed audios.xlsx"

# Lookalike code points used in place of the Twi letters. The 41-character CTC
# vocabulary would drop them silently, deleting a letter from the target.
UNICODE_FOLD = {
    "ↄ": "ɔ",   # ↄ  LATIN SMALL LETTER REVERSED C   -> ɔ
    "Ↄ": "Ɔ",   # Ↄ  LATIN CAPITAL LETTER REVERSED C -> Ɔ
    "ℇ": "ɛ",   # ℇ  EULER CONSTANT                  -> ɛ
}


def local_name(handle, offset: int, utf8: bool):
    """The file name in the local header at `offset`, or None if there is no header."""
    handle.seek(offset)
    header = handle.read(30)
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        return None, 0
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    name = handle.read(name_len).decode("utf-8" if utf8 else "cp437", "replace")
    return name, offset + 30 + name_len + extra_len


def read_verified(handle, info, candidates):
    """Bytes of the entry at the first candidate offset that passes every check, else None."""
    for offset in candidates:
        if offset < 0:
            continue
        name, data_at = local_name(handle, offset, bool(info.flag_bits & 0x800))
        if name != info.filename:
            continue
        handle.seek(data_at)
        raw = handle.read(info.compress_size)
        try:
            data = zlib.decompress(raw, -15) if info.compress_type == 8 else raw
        except zlib.error:
            continue
        if len(data) == info.file_size and (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC:
            return data
    return None


def fold(text: str):
    """Maps lookalike code points to the real letters; returns the text and how many were changed."""
    changed = sum(text.count(ch) for ch in UNICODE_FOLD)
    for src, dst in UNICODE_FOLD.items():
        text = text.replace(src, dst)
    return text, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="data/Akan.zip", help="Path to Akan.zip")
    parser.add_argument("--out", required=True, help="Output directory (created; must be empty if it exists)")
    parser.add_argument("--limit", type=int, default=None, help="Extract at most N clips (smoke test)")
    return parser


def main() -> int:
    import zipfile

    args = build_parser().parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; refusing to mix extractions. Choose a new --out.")
    audio_dir = out / "audios"
    audio_dir.mkdir(parents=True, exist_ok=True)

    archive = zipfile.ZipFile(args.zip)
    infos = archive.infolist()
    # The first entry of a well-formed archive sits at offset 0, so wherever
    # zipfile placed it is exactly the shift it applied to every offset.
    shift = min(i.header_offset for i in infos)
    print(f"{len(infos)} entries; reader-applied shift {shift:,} bytes")

    recovered, lost, bad_crc = {}, [], 0
    with open(args.zip, "rb") as handle:
        sheet = next((i for i in infos if i.filename.endswith(SHEET_MEMBER)), None)
        if sheet is None:
            sys.exit(f"{SHEET_MEMBER} not found in the archive")
        sheet_bytes = read_verified(handle, sheet, (sheet.header_offset, sheet.header_offset - shift))
        if sheet_bytes is None:
            sys.exit("The transcript spreadsheet itself failed verification; nothing can be labelled.")
        (out / "transcripts.xlsx").write_bytes(sheet_bytes)

        mp3s = [i for i in infos if AUDIO_MEMBER in i.filename and i.filename.lower().endswith(".mp3")]
        if args.limit:
            mp3s = mp3s[: args.limit]
        for n, info in enumerate(mp3s, 1):
            data = read_verified(handle, info, (info.header_offset, info.header_offset - shift))
            if data is None:
                lost.append(info.filename)
                continue
            base = Path(info.filename).name
            (audio_dir / base).write_bytes(data)
            recovered[base] = len(data)
            if n % 2000 == 0:
                print(f"  {n}/{len(mp3s)} processed, {len(recovered)} recovered")

    try:
        import openpyxl
    except ImportError:
        sys.exit("Audio extracted, but the manifest needs openpyxl (pip install openpyxl).")

    rows = openpyxl.load_workbook(out / "transcripts.xlsx", read_only=True).active.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    col = {name: header.index(name) for name in ("Full Filename", "Transcriptions", "SPEAKER_ID", "GENDER", "AGE")}

    written = folded = missing_text = 0
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "text", "speaker_id", "gender", "age"])
        for row in rows:
            name = row[col["Full Filename"]]
            if name not in recovered:
                continue
            text, changed = fold(str(row[col["Transcriptions"]] or "").strip())
            if not text:
                missing_text += 1
                continue
            folded += changed
            writer.writerow([f"audios/{name}", text, row[col["SPEAKER_ID"]], row[col["GENDER"]], row[col["AGE"]]])
            written += 1

    report = {
        "entries_in_archive": len(infos),
        "mp3_attempted": len(mp3s),
        "mp3_recovered": len(recovered),
        "mp3_lost": len(lost),
        "manifest_rows": written,
        "rows_without_transcript": missing_text,
        "lookalike_characters_folded": folded,
        "bytes_written": sum(recovered.values()),
        "lost_examples": lost[:5],
    }
    (out / "extract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
