#!/usr/bin/env python
"""
Builds the English worklist to be translated into Asante Twi.

Measured on the validation split, the language model gains nothing from more
text of the wrong dialect or domain: dropping `verified_data.csv` (25,420
Akuapem-leaning lines) left WER unchanged and improved CER, and adding
`Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` made both worse. The best model came from
17,544 in-domain transcripts alone.

So the useful text to add is **Asante Twi**, not simply more Twi. This collects
the English side of the parallel corpora already in the repository - sentences
whose Twi already exists in the wrong variety, or not at all - so they can be
translated into Asante and used to grow the language model.

Two files are written to data/corpus/english/:

  * `to_translate.csv` - id, english, twi (blank), source. Fill the twi column.
  * `to_translate.txt` - one English sentence per line, for machine-translation
    tools that take plain text.

Both are deduplicated on the normalised English, so the same sentence is not
paid for twice.

    python scripts/lm/build_translation_worklist.py
    python scripts/lm/build_translation_worklist.py --limit 5000
    python scripts/lm/build_translation_worklist.py --no-ghananlp
"""

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join("data", "corpus", "english")

# Columns holding the English side, across the several spellings in use.
ENGLISH_COLUMNS = ("English", "1. English Sentence/Phrase", "english", "text")

# Long enough to carry structure a language model can learn from, short enough
# to translate reliably. One-word entries teach an n-gram almost nothing.
MIN_WORDS = 3
MAX_WORDS = 40


def normalise(text: str) -> str:
    """Lowercased, whitespace-collapsed, for deduplication only."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def from_csv(path: str, label: str):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        column = next((c for c in fields if c in ENGLISH_COLUMNS), fields[0] if fields else None)
        if column is None:
            return
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                yield value, label


def from_ghananlp():
    """The English side of Ghana-NLP's parallel set."""
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT", token=os.environ.get("HF_TOKEN")
        )["train"]
        for row in dataset:
            value = str(row.get("text") or "").strip()
            if value:
                yield value, "ghana-nlp"
    except Exception as exc:
        print(f"  Ghana-NLP unavailable ({type(exc).__name__}); continuing without it")


def usable(text: str) -> bool:
    words = text.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep at most N sentences, evenly spread across the sources")
    parser.add_argument("--no-ghananlp", dest="ghananlp", action="store_false", default=True)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.chdir(ROOT)

    sources = [
        from_csv(os.path.join("data", "corpus", "verified_data.csv"), "verified_data"),
        from_csv(os.path.join("data", "corpus", "crowdsourced_data.csv"), "crowdsourced"),
    ]
    if args.ghananlp:
        sources.append(from_ghananlp())

    seen = set()
    rows = []
    counts = {}
    for source in sources:
        for text, label in source:
            if not usable(text):
                continue
            key = normalise(text)
            if key in seen:
                continue
            seen.add(key)
            rows.append((text, label))
            counts[label] = counts.get(label, 0) + 1

    if args.limit and len(rows) > args.limit:
        step = len(rows) / args.limit
        rows = [rows[int(i * step)] for i in range(args.limit)]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "to_translate.csv")
    txt_path = os.path.join(args.output_dir, "to_translate.txt")

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "english", "twi", "source"])
        for index, (text, label) in enumerate(rows, start=1):
            writer.writerow([index, text, "", label])

    with open(txt_path, "w", encoding="utf-8") as handle:
        for text, _ in rows:
            handle.write(text.replace("\n", " ") + "\n")

    print("English sentences awaiting Asante Twi translation:")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<16} {count:>7,}")
    print(f"  {'TOTAL':<16} {len(rows):>7,} unique, {MIN_WORDS}-{MAX_WORDS} words each\n")
    print(f"  {csv_path}   (fill in the 'twi' column)")
    print(f"  {txt_path}   (one sentence per line)")
    print("\nOnce translated, add the file to scripts/lm/build_lm.py and rebuild:")
    print("  python scripts/lm/build_lm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
