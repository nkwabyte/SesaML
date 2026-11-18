#!/usr/bin/env python
"""
Cleans the Ghana English-Twi code-switched corpus and cuts a prompt-disjoint hold-out.

The corpus has 1,544 distinct sentences read by ~33 speakers each, and every sentence of its own
test split is also in train, so its test score would only measure memorised sentences. This
build holds out whole *sentences*: `--holdout-sentences` of them are removed from train entirely,
and the hold-out is made of those sentences read by the corpus's validation+test speakers
(unseen voices as well as unseen text).

Cleaning (rows are dropped, never patched; audio bytes are copied untouched):
  * rows with `error` set or prompt_set != "standard" (malformed);
  * clips over --max-seconds (the trainer's limit) or empty after normalisation;
  * text: curly quotes folded, hyphens and punctuation become spaces (so "Block-i" is two words),
    lower-cased. The directory split is trusted, not the per-row `split` column.

    python scripts/datasets/build_codeswitch_dataset.py --name cs/ghana-codeswitch-v1
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SRC = "data/datasets/responsibleailab__ghana-english-twi-code-switched-speech-corpus"


def normalise(text: str) -> str:
    text = (text or "").replace("’", "'").replace("‘", "'").replace("ↄ", "ɔ").replace("ℇ", "ɛ").lower()
    text = re.sub(r"[^a-z0-9ɛɔ' ]", " ", text)
    text = re.sub(r"(?<![a-z0-9ɛɔ])'|'(?![a-z0-9ɛɔ])", " ", text)      # keep apostrophes inside words only
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    from datasets import Audio, Dataset, load_from_disk

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC)
    p.add_argument("--name", default="cs/ghana-codeswitch-v1")
    p.add_argument("--out-dir", default="data/datasets")
    p.add_argument("--holdout-sentences", type=int, default=150)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    save_root = Path(args.out_dir) / args.name.replace("/", "__")
    if save_root.exists() and any(save_root.iterdir()):
        sys.exit(f"{save_root} already exists; choose a new --name.")

    def load(split):
        d = load_from_disk(f"{args.src}/{split}")
        d = d.cast_column("audio", Audio(decode=False))
        keep, drops, texts = [], Counter(), {}
        meta = d.remove_columns(["audio"])
        for i, r in enumerate(meta):
            t = normalise(r["transcript"])
            if r["error"]:
                drops["error set"] += 1
            elif r["prompt_set"] != "standard":
                drops["prompt_set != standard"] += 1
            elif not r["duration"] or r["duration"] > args.max_seconds:
                drops["too long / no duration"] += 1
            elif len(t) < 3:
                drops["empty text"] += 1
            else:
                keep.append(i)
                texts[i] = t
        print(f"{split}: {len(d)} rows, kept {len(keep)}, dropped {dict(drops)}")
        return d, keep, texts

    parts = {s: load(s) for s in ("train", "validation", "test")}
    train_sents = {parts["train"][2][i] for i in parts["train"][1]}
    rng = random.Random(args.seed)
    eval_sents = sorted({t for s in ("validation", "test") for t in parts[s][2].values()} & train_sents)
    rng.shuffle(eval_sents)
    holdout = set(eval_sents[: args.holdout_sentences])
    if len(holdout) < args.holdout_sentences:
        sys.exit(f"only {len(holdout)} sentences are shared between train and the eval speakers")

    def build(split, wanted):
        d, keep, texts = parts[split]
        idx = [i for i in keep if wanted(texts[i])]
        sub = d.select(idx)
        return Dataset.from_dict({
            "audio": [a for a in sub["audio"]],
            "sentence": [texts[i] for i in idx],
            "speaker_id": [str(x) for x in sub["speaker_id"]],
            "seconds": [float(x) for x in sub["duration"]],
        }).cast_column("audio", Audio(sampling_rate=16000, decode=False))

    train = build("train", lambda t: t not in holdout)
    held = [build(s, lambda t: t in holdout) for s in ("validation", "test")]
    from datasets import concatenate_datasets
    holdout_ds = concatenate_datasets(held)
    # Hard checks: no held-out sentence in train; no speaker on both sides.
    tr_sent, ho_sent = set(train["sentence"]), set(holdout_ds["sentence"])
    tr_spk, ho_spk = set(train["speaker_id"]), set(holdout_ds["speaker_id"])
    if tr_sent & ho_sent or tr_spk & ho_spk:
        sys.exit(f"LEAK: {len(tr_sent & ho_sent)} shared sentences, {len(tr_spk & ho_spk)} shared speakers")
    train.save_to_disk(str(save_root / "train"))
    holdout_ds.save_to_disk(str(save_root / "holdout"))
    report = {"name": args.name, "train_clips": len(train), "train_hours": round(sum(train["seconds"]) / 3600, 1),
              "train_speakers": len(tr_spk), "train_sentences": len(tr_sent),
              "holdout_clips": len(holdout_ds), "holdout_hours": round(sum(holdout_ds["seconds"]) / 3600, 2),
              "holdout_speakers": len(ho_spk), "holdout_sentences": len(ho_sent), "seed": args.seed,
              "note": "holdout = unseen sentences read by unseen speakers; train excludes those sentences"}
    (save_root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
