#!/usr/bin/env python
"""
Builds the Twi character n-gram language model used for CTC decoding.

Text is drawn from every Twi source in the repository:

* `data/corpus/english/to_translate.csv` - the Asante Twi translations, once its
  `twi` column is filled (see scripts/lm/build_translation_worklist.py). Included
  automatically when present.
* `data/corpus/crowdsourced_data.csv` - a smaller Twi translation set, Asante.
* `data/corpus/twi/*.txt` - CoNLL-tagged NER data, one token per line with a
  label; sentences are reassembled from the tokens and the tags discarded.
* The transcripts of the audio corpora, when they are on disk. Measured, these
  are the single most valuable source: an LM built from them alone beat one
  built from three times as much mixed text.

`data/corpus/verified_data.csv` is **excluded by default**. Its 25,420 lines
lean Akuapem while every audio corpus - and the validation split WER is measured
on - is Asante, and an ablation showed dropping it left WER unchanged (0.3971)
while improving CER (0.1303 -> 0.1280). `--include-verified` puts it back.
`Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT` was tested as a replacement and made both
worse (WER 0.4062), so it is not wired in.

Every line is normalised through `TextTransform`, so the model is built over
exactly the character set the decoder can emit - no accents or punctuation the
CTC vocabulary would never produce.

    python scripts/lm/build_lm.py                       # build with defaults
    python scripts/lm/build_lm.py --order 8 --min-count 3
    python scripts/lm/build_lm.py --no-audio-transcripts
"""

import argparse
import csv
import os
import sys
from typing import Iterator, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.asr.data.text_transform import TextTransform          # noqa: E402
from src.asr.decoding.language_model import DEFAULT_ORDER, CharNGramLM  # noqa: E402

DEFAULT_OUTPUT = os.path.join("outputs", "asr", "lm", "twi_char.json")

TWI_CSV_COLUMNS = ("Akuapem Twi", "1. Twi Translation", "twi", "Twi")
AUDIO_CORPORA = (
    ("data/datasets/Lagyamfi__akan_audio_processed/train", "sentence"),
    ("data/datasets/ghanaopendata__twi-speech-text-multispeaker-16k/train", "text"),
)


def from_csv(path: str) -> Iterator[str]:
    """Twi sentences from a two-column translation CSV."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        column = next((c for c in (reader.fieldnames or []) if c in TWI_CSV_COLUMNS), None)
        if column is None:
            # Fall back to the last column: these files are English,Twi pairs.
            column = (reader.fieldnames or [None])[-1]
        if column is None:
            return
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                yield value


def from_conll(path: str) -> Iterator[str]:
    """
    Sentences reassembled from a CoNLL-tagged file.

    The file is one `token TAG` pair per line with blank lines between
    sentences, so the tokens of each sentence are joined and the tags dropped.
    """
    if not os.path.isfile(path):
        return
    tokens: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    yield " ".join(tokens)
                    tokens = []
                continue
            tokens.append(stripped.split()[0])
    if tokens:
        yield " ".join(tokens)


def from_audio_corpus(path: str, column: str) -> Iterator[str]:
    """Transcripts of a downloaded speech corpus, without decoding its audio."""
    if not os.path.isdir(path):
        return
    try:
        from datasets import load_from_disk

        dataset = load_from_disk(path)
        drop = [c for c in dataset.column_names if c != column]
        source = dataset.remove_columns(drop) if drop else dataset
        for row in source:
            value = str(row.get(column) or "").strip()
            if value:
                yield value
    except Exception as exc:  # a missing corpus must not stop the build
        print(f"  skipped {path}: {type(exc).__name__}")


def from_translations(path: str) -> Iterator[str]:
    """The Asante Twi column of the translation worklist, once it is filled in."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("twi") or "").strip()
            if value:
                yield value


def collect(include_audio: bool, include_verified: bool = False) -> Iterator[str]:
    sources = [
        ("english/to_translate.csv", from_translations,
         os.path.join("data", "corpus", "english", "to_translate.csv")),
        ("crowdsourced_data.csv", from_csv, os.path.join("data", "corpus", "crowdsourced_data.csv")),
        ("twi/train.txt", from_conll, os.path.join("data", "corpus", "twi", "train.txt")),
        ("twi/dev.txt", from_conll, os.path.join("data", "corpus", "twi", "dev.txt")),
        ("twi/test.txt", from_conll, os.path.join("data", "corpus", "twi", "test.txt")),
    ]
    if include_verified:
        sources.insert(0, ("verified_data.csv (Akuapem)", from_csv,
                           os.path.join("data", "corpus", "verified_data.csv")))

    for label, reader, path in sources:
        count = 0
        for line in reader(path):
            count += 1
            yield line
        print(f"  {label:<26} {count:>7,} lines")

    if include_audio:
        for path, column in AUDIO_CORPORA:
            count = 0
            for line in from_audio_corpus(path, column):
                count += 1
                yield line
            print(f"  {os.path.basename(path):<26} {count:>7,} transcripts")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER, help="n-gram order")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Drop n-grams seen fewer times than this (unigrams are always kept)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-audio-transcripts", dest="include_audio", action="store_false",
                        default=True, help="Build from written Twi only")
    parser.add_argument("--include-verified", action="store_true", default=False,
                        help="Include data/corpus/verified_data.csv. Off by default: it leans "
                             "Akuapem while the audio is Asante, and an ablation showed it does "
                             "not improve WER and slightly worsens CER")
    parser.add_argument("--holdout", type=int, default=2000,
                        help="Lines held out to report perplexity on")
    args = parser.parse_args()

    os.chdir(ROOT)
    transform = TextTransform()

    print("Collecting Twi text:")
    normalised: List[str] = []
    for line in collect(args.include_audio, args.include_verified):
        # Through the vocabulary, so the model covers exactly what can be emitted.
        text = transform.int_to_text(transform.text_to_int(line)).strip()
        if text:
            normalised.append(text)

    if not normalised:
        print("No Twi text found. Is data/corpus/ populated?")
        return 1

    holdout = normalised[-args.holdout:] if args.holdout else []
    training = normalised[:-args.holdout] if args.holdout else normalised
    characters = sum(len(line) for line in training)
    print(f"\n  {len(training):,} lines, {characters:,} characters for training")
    print(f"  {len(holdout):,} lines held out\n")

    model = CharNGramLM(order=args.order)
    model.train(training)
    contexts_before = len(model.counts)
    model.prune(min_count=args.min_count)

    print(f"Model: order {args.order}, {contexts_before:,} contexts -> "
          f"{len(model.counts):,} after pruning at min-count {args.min_count}")
    if holdout:
        print(f"Held-out perplexity per character: {model.perplexity(holdout):.2f}")

    path = model.save(args.output)
    print(f"Wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
