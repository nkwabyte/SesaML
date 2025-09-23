#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import soundfile as sf
import librosa
from mutagen import File as MutaFile  # works for mp3/m4a/ogg, etc.

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma", ".opus"}

def list_audio_files(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS])

def probe_with_soundfile(p: Path):
    # Returns dict or None if unsupported by soundfile
    try:
        info = sf.info(str(p))
        duration = float(info.frames) / float(info.samplerate) if info.samplerate > 0 else None
        return {
            "format": info.format,           # e.g., 'WAV'
            "subtype": info.subtype,         # e.g., 'PCM_16'
            "channels": info.channels,
            "sample_rate": info.samplerate,
            "duration_sec": duration,
        }
    except Exception:
        return None

def probe_with_mutagen(p: Path):
    try:
        m = MutaFile(str(p))
        if m is None:
            return None
        # Channels / sample_rate / length are best-effort
        duration = float(getattr(m.info, "length", None) or 0.0) or None
        samplerate = int(getattr(m.info, "sample_rate", 0) or 0) or None
        channels = int(getattr(m.info, "channels", 0) or 0) or None

        # Rough container/codec name
        if p.suffix.lower() == ".mp3":
            fmt = "MP3"
        elif p.suffix.lower() in [".m4a", ".aac"]:
            fmt = "M4A/AAC"
        elif p.suffix.lower() == ".ogg":
            # mutagen could be Vorbis or Opus inside ogg
            name = m.mime[0] if getattr(m, "mime", None) else "OGG"
            fmt = "OGG" if "ogg" in name else name
        elif p.suffix.lower() == ".opus":
            fmt = "OPUS"
        else:
            fmt = (m.mime[0] if getattr(m, "mime", None) else p.suffix.upper().strip("."))

        return {
            "format": fmt,
            "subtype": None,
            "channels": channels,
            "sample_rate": samplerate,
            "duration_sec": duration,
        }
    except Exception:
        return None

def get_duration_librosa(p: Path):
    try:
        dur = librosa.get_duration(path=str(p))
        return float(dur)
    except Exception:
        return None

def infer_source_domain(path: Path, root: Path, domain_from: str, mapping: dict | None):
    rel = path.relative_to(root)
    parts = [part.lower() for part in rel.parts]

    # Mapping file takes precedence (match if any key is a substring of the rel path)
    if mapping:
        rel_str = str(rel).lower()
        for key, value in mapping.items():
            if key.lower() in rel_str:
                return value

    if domain_from == "topdir":
        # Use the first directory under root as domain (if exists)
        return parts[0] if len(parts) > 1 else "unknown"
    elif domain_from == "parent":
        # Use the immediate parent folder
        return path.parent.name.lower() if path.parent != root else "unknown"
    else:
        # Try to find a meaningful token
        candidates = {"customary court", "court", "radio", "telephone", "phone", "street", "interview"}
        rel_str = " ".join(parts)
        for c in candidates:
            if c.replace(" ", "") in rel_str.replace("_", "").replace("-", ""):
                return c
        return "unknown"

def main():
    ap = argparse.ArgumentParser(description="Build raw audio manifest")
    ap.add_argument("--input_dir", type=Path, required=True, help="Root folder of raw audio (recursive)")
    ap.add_argument("--out_csv", type=Path, required=True, help="Where to write the manifest CSV")
    ap.add_argument("--domain_from", type=str, default="topdir",
                    choices=["topdir", "parent", "auto"], help="How to infer source_domain if no mapping is supplied")
    ap.add_argument("--domain_map", type=Path, default=None,
                    help='Optional JSON mapping file: {"customary_court": "customary court", "radio": "radio_broadcast"}')
    args = ap.parse_args()

    root = args.input_dir.resolve()
    files = list_audio_files(root)
    if not files:
        print(f"No audio files found in {root}")
        return

    mapping = None
    if args.domain_map and args.domain_map.exists():
        with open(args.domain_map, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    rows = []
    for p in tqdm(files, desc="Scanning"):
        # Basic file facts
        rel_path = str(p.relative_to(root))
        ext = p.suffix.lower().lstrip(".")
        size = p.stat().st_size

        # Try soundfile first (great for wav/flac/ogg). Fall back to mutagen.
        meta = probe_with_soundfile(p) or probe_with_mutagen(p) or {}
        # If duration still None, try librosa as last resort
        if meta.get("duration_sec") is None:
            meta["duration_sec"] = get_duration_librosa(p)

        # Infer domain
        domain = infer_source_domain(p, root, args.domain_from, mapping)

        rows.append({
            "rel_path": rel_path,
            "ext": ext,
            "format": meta.get("format"),
            "subtype": meta.get("subtype"),
            "channels": meta.get("channels"),
            "sample_rate": meta.get("sample_rate"),
            "duration_sec": meta.get("duration_sec"),
            "file_size_bytes": size,
            "source_domain": domain,
        })

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"✔ Manifest written to {args.out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main()