#!/usr/bin/env python
"""
Transcribes audio with a fine-tuned Whisper teacher so a Conformer can train on it.

Every clip gets a transcript and a confidence, and nothing is filtered at label
time: the keep/drop thresholds should come from measuring the teacher against
references, not from a guess. So the workflow is two steps.

  1. Calibrate on audio that already has references (a corpus you hold out):

         python scripts/asr/pseudo_label.py --teacher <whisper dir> \\
             --hf-dataset Lagyamfi/akan_audio_processed:test --out outputs/asr/pseudo/lagyamfi_test.jsonl

     The summary prints teacher WER/CER and, for each confidence band, how many
     clips survive and what WER they carry - the trade-off a threshold buys.

  2. Label the unlabelled audio, then export only what clears the threshold:

         python scripts/asr/pseudo_label.py --teacher <whisper dir> \\
             --audio-dir "data/Splitted Audio Files" --out outputs/asr/pseudo/court.jsonl \\
             --export-manifest outputs/asr/pseudo/court.csv --min-logprob -0.35

Sources: --hf-dataset REPO[:SPLIT] (local save_to_disk copy), --manifest CSV with an
audio-path column (optional text column) plus --audio-dir for relative paths, or
--audio-dir alone for a directory of audio files.

Whisper takes at most 30 s. Longer clips are skipped and counted, never truncated:
a transcript of the first 30 s attached to the whole clip is exactly the silent
mislabelling this project has been bitten by. Decoding defaults to the setting that
scored best on the held-out split (greedy with a repeat guard); beam search was
measurably worse. The checkpoint ships no preprocessor config, so the feature
extractor comes from --feature-extractor.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
MAX_SECONDS = 30.0
# Rates outside this range mean a loop, a hallucination over silence, or a
# truncation, whatever the model's own confidence says.
CHARS_PER_SEC_RANGE = (2.0, 25.0)
CONFIDENCE_BANDS = (-0.15, -0.25, -0.35, -0.5, -0.8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True, help="Directory of the fine-tuned Whisper checkpoint")
    p.add_argument("--feature-extractor", default="openai/whisper-medium", help="Source of the 80-mel feature extractor")
    p.add_argument("--hf-dataset", default=None, metavar="REPO[:SPLIT]", help="Local HuggingFace corpus (under data/datasets)")
    p.add_argument("--manifest", default=None, help="CSV with an audio-path column and an optional text column")
    p.add_argument("--audio-dir", default=None, help="Directory of audio files, or the root for relative manifest paths")
    p.add_argument("--out", required=True, help="JSONL file to write (appended to when --resume)")
    p.add_argument("--resume", action="store_true", help="Skip clips already present in --out")
    p.add_argument("--limit", type=int, default=None, help="Label at most N clips (evenly spaced), for smoke tests")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4, help="0 disables the repeat guard")
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--export-manifest", default=None, help="Write audio_path,text for clips that clear the thresholds")
    p.add_argument("--min-logprob", type=float, default=None, help="Keep clips whose mean token log-probability is at least this")
    return p


def build_dataset(args):
    """Returns a dataset yielding (waveform, sample_rate, text, path), plus the reference flag."""
    import pandas as pd
    from src.asr.data.dataset import AkanAudioDataset

    chosen = [bool(args.hf_dataset), bool(args.manifest), bool(args.audio_dir) and not args.manifest]
    if sum(chosen) != 1:
        sys.exit("Give exactly one source: --hf-dataset, --manifest, or --audio-dir on its own.")

    if args.hf_dataset:
        from src.asr.data.hf_dataset import load_hf_datasets
        return load_hf_datasets([args.hf_dataset], sample_rate=16000)[0], True

    if args.manifest:
        data = AkanAudioDataset(csv_file=args.manifest, audio_dir=args.audio_dir, sample_rate=16000)
        return data, True

    root = Path(args.audio_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        sys.exit(f"No audio files under {root}")
    frame = pd.DataFrame({"audio_path": [str(p) for p in files], "text": [""] * len(files)})
    return AkanAudioDataset(dataframe=frame, sample_rate=16000), False


class Indexed:
    """Wraps a dataset so each item carries its index; a bad clip is reported, not fatal."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        try:
            waveform, sr, text, path = self.dataset[i]
            # HuggingFace rows carry no file name; the index is the only stable handle.
            path = str(path) if path not in (None, "None", "") else f"row_{i}"
            return {"index": i, "audio": waveform.squeeze(0).numpy(), "text": text, "path": path, "error": None}
        except Exception as exc:  # decode failures must be visible, and must not stop a long run
            return {"index": i, "audio": None, "text": "", "path": f"row_{i}", "error": f"{type(exc).__name__}: {exc}"}


def collate(items):
    return items


def folded(transform, text: str) -> str:
    return transform.int_to_text(transform.text_to_int(text)).strip()


def label(args, dataset, indices):
    import torch
    from torch.utils.data import DataLoader
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    from src.asr.data.text_transform import TextTransform

    tokenizer = WhisperTokenizer.from_pretrained(args.teacher)
    extractor = WhisperFeatureExtractor.from_pretrained(args.feature_extractor)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device).eval()
    transform = TextTransform()
    text_ids = tokenizer.eos_token_id  # ids at or above this are special tokens, not text

    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}
    if args.no_repeat_ngram_size:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = args.repetition_penalty

    loader = DataLoader(Indexed(dataset, indices), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate, shuffle=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = {"too_long": 0, "decode_error": 0}
    written = 0

    with open(out_path, "a" if args.resume else "w", encoding="utf-8") as sink:
        for batch in loader:
            usable = []
            for item in batch:
                if item["error"]:
                    skipped["decode_error"] += 1
                    sink.write(json.dumps({"index": item["index"], "path": item["path"], "skipped": item["error"]}) + "\n")
                elif len(item["audio"]) / 16000 > MAX_SECONDS:
                    skipped["too_long"] += 1
                    sink.write(json.dumps({"index": item["index"], "path": item["path"], "skipped": "longer than 30s"}) + "\n")
                else:
                    usable.append(item)
            if not usable:
                continue

            features = extractor([u["audio"] for u in usable], sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                result = model.generate(
                    features.input_features.to(args.device, dtype),
                    return_dict_in_generate=True, output_logits=True, **generate_kwargs,
                )
            scores = model.compute_transition_scores(result.sequences, result.logits, normalize_logits=True)
            generated = result.sequences[:, -scores.shape[1]:]

            for row, item in enumerate(usable):
                keep_token = generated[row] < text_ids
                n_tokens = int(keep_token.sum())
                mean_lp = float(scores[row][keep_token].float().mean()) if n_tokens else float("-inf")
                raw = tokenizer.decode(generated[row], skip_special_tokens=True).strip()
                hyp = folded(transform, raw)
                seconds = len(item["audio"]) / 16000
                rate = len(hyp) / seconds if seconds else 0.0
                record = {
                    "index": item["index"], "path": item["path"], "seconds": round(seconds, 2),
                    "teacher_raw": raw, "teacher": hyp, "mean_logprob": round(mean_lp, 4),
                    "tokens": n_tokens, "chars_per_sec": round(rate, 2),
                    "implausible_rate": not (CHARS_PER_SEC_RANGE[0] <= rate <= CHARS_PER_SEC_RANGE[1]),
                }
                if item["text"]:
                    record["reference"] = folded(transform, item["text"])
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            sink.flush()
            print(f"  {written} labelled", end="\r", flush=True)
    print()
    return skipped


def summarise(path: Path, skipped: dict) -> list:
    import jiwer

    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    labelled = [r for r in records if "teacher" in r]
    print(f"\n{len(labelled)} labelled | skipped: {skipped} | {len(records) - len(labelled)} skip records in file")
    if not labelled:
        return labelled
    print("mean_logprob median %.3f | implausible chars/sec: %d | empty hypotheses: %d" % (
        statistics.median(r["mean_logprob"] for r in labelled),
        sum(r["implausible_rate"] for r in labelled), sum(not r["teacher"] for r in labelled)))

    refd = [r for r in labelled if r.get("reference") and r["teacher"]]
    if not refd:
        print("No references in this source: thresholds cannot be calibrated here.")
        return labelled
    ref, hyp = [r["reference"] for r in refd], [r["teacher"] for r in refd]
    print(f"\nTeacher vs reference on {len(refd)} clips: WER {jiwer.wer(ref, hyp):.4f} | CER {jiwer.cer(ref, hyp):.4f}")
    print("\n  min mean_logprob | kept clips | WER of kept | CER of kept")
    for threshold in CONFIDENCE_BANDS:
        kept = [r for r in refd if r["mean_logprob"] >= threshold and not r["implausible_rate"]]
        if len(kept) >= 20:
            kr, kh = [r["reference"] for r in kept], [r["teacher"] for r in kept]
            print(f"  {threshold:>16.2f} | {len(kept):>5} ({100 * len(kept) / len(refd):4.0f}%) | {jiwer.wer(kr, kh):.4f}      | {jiwer.cer(kr, kh):.4f}")
        else:
            print(f"  {threshold:>16.2f} | {len(kept):>5}  too few to measure")
    return labelled


def export(labelled: list, args) -> None:
    kept = [r for r in labelled if r["teacher"] and not r["implausible_rate"]
            and (args.min_logprob is None or r["mean_logprob"] >= args.min_logprob)]
    import csv
    with open(args.export_manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "text", "mean_logprob"])
        for r in kept:
            writer.writerow([r["path"], r["teacher"], r["mean_logprob"]])
    print(f"\nExported {len(kept)} of {len(labelled)} clips to {args.export_manifest} "
          f"(min_logprob={args.min_logprob}). These are model transcripts, not ground truth.")


def main() -> int:
    args = build_parser().parse_args()
    dataset, _ = build_dataset(args)
    indices = list(range(len(dataset)))
    if args.limit and args.limit < len(indices):
        step = len(indices) / args.limit
        indices = [indices[int(k * step)] for k in range(args.limit)]
    if args.resume and Path(args.out).exists():
        done = {json.loads(line)["index"] for line in open(args.out, encoding="utf-8")}
        indices = [i for i in indices if i not in done]
        print(f"Resuming: {len(done)} already in {args.out}, {len(indices)} to do")

    skipped = label(args, dataset, indices)
    labelled = summarise(Path(args.out), skipped)
    if args.export_manifest:
        export(labelled, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
