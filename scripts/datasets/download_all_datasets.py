#!/usr/bin/env python
"""
Downloads all supported Akan speech datasets into local `data/` directory at a go.

Cross-platform (Windows, macOS, Linux).

Examples:
    python scripts/download_all_datasets.py
    python scripts/download_all_datasets.py --num-samples 100
    python scripts/download_all_datasets.py --token hf_...
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

CORPORA = [
    "ghanaopendata/twi-speech-text-multispeaker-16k",
    "Lagyamfi/akan_audio_processed",
    "ghananlpcommunity/twi-health-asr-gemini-500hrs",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-samples", type=int, default=None, help="Keep only N samples per split (smoke test)")
    parser.add_argument("--token", default=None, help="HuggingFace access token")
    args, unknown = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    download_script = repo_root / "scripts" / "download_dataset.py"

    print(f"[sesaml] Starting bulk download of {len(CORPORA)} Akan datasets into data/...")

    for repo in CORPORA:
        print("=" * 64)
        print(f"[sesaml] Downloading dataset: {repo}")
        print("=" * 64)

        cmd = [sys.executable, str(download_script), "--dataset", repo]
        if args.num_samples:
            cmd.extend(["--num-samples", str(args.num_samples)])
        if args.token:
            cmd.extend(["--token", args.token])
        cmd.extend(unknown)

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[sesaml] Warning: Dataset '{repo}' exited with code {res.returncode}. Continuing with remaining datasets...")

    print("=" * 64)
    print("[sesaml] Bulk dataset download sequence complete!")


if __name__ == "__main__":
    main()
