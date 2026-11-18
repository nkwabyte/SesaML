#!/usr/bin/env python
"""
Transcribes audio with a trained Conformer-CTC and scores how far each transcript can be trusted.

This is the CTC counterpart of pseudo_label.py (which drives the Whisper teacher). A
transcript is only worth training on if something can tell the good ones from the bad, and a
model's own confidence is a weak signal when it is wrong with conviction (Whisper was). So
every clip gets several candidate confidence measures, and when the source has references the
summary ranks the measures by what they actually buy:

    python scripts/asr/ctc_label.py --model-path outputs/asr/registry/conformer-medium/v001/model.pt \\
        --architecture conformer-medium --hf-dataset ug-waxal/akan-transcribed-v2:test \\
        --out outputs/asr/pseudo/waxal_test_ctc.jsonl

For each measure it keeps the most confident 20/40/60/80/100% of clips and reports the WER of
what is kept, which is the trade-off a pseudo-labelling threshold would make.

Measures (higher = more confident):
    seq_lp     log-probability per character of the greedy transcript under CTC (all alignments)
    emit_conf  mean frame probability over the frames that emit a character
    agree      1 - CER between the greedy and the beam+LM transcript (needs --beam)

Sources are the same as pseudo_label.py: --hf-dataset, --manifest, or --audio-dir.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

KEEP_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=None, help="Checkpoint (default: the served registry version)")
    p.add_argument("--architecture", default="conformer-medium")
    p.add_argument("--hf-dataset", default=None, metavar="REPO[:SPLIT]")
    p.add_argument("--manifest", default=None)
    p.add_argument("--audio-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="Evenly spaced subset of N clips")
    p.add_argument("--beam", action="store_true", help="Also decode with beam search + LM (slow) for the agree measure")
    p.add_argument("--lm", default="outputs/asr/lm/twi_char_A_clean.json")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p


def label(args, dataset, indices):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from pseudo_label import Indexed, collate
    from src.asr.data.text_transform import TextTransform
    from src.asr.decoding.decoder import CTCDecoder
    from src.asr.inference.transcribe import Transcriber
    from src.config import PipelineConfig

    config = PipelineConfig()
    config.model.architecture = args.architecture
    config.device = args.device
    transcriber = Transcriber(model_type="deepspeech", model_path=args.model_path, config=config)
    print("model:", transcriber.describe()["model_path"])
    transform = TextTransform()
    blank = transform.blank_label
    greedy = CTCDecoder(transform, decoder="greedy")
    beam = CTCDecoder(transform, decoder="beam", lm_path=args.lm,
                      beam_width=config.decoding.beam_width, alpha=config.decoding.alpha,
                      beta=config.decoding.beta) if args.beam else None
    fold = lambda s: transform.int_to_text(transform.text_to_int(s)).strip()

    loader = DataLoader(Indexed(dataset, indices), batch_size=8, num_workers=args.num_workers,
                        collate_fn=collate, shuffle=False)
    # Written as it goes: a pool run is an hour long and must survive an interruption.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = open(out, "w", encoding="utf-8")
    records, skipped, written = [], 0, 0
    for batch in loader:
        for item in batch:
            if item["error"] or item["audio"] is None:
                skipped += 1
                records.append({"index": item["index"], "path": item["path"], "skipped": item["error"]})
                continue
            wave = torch.from_numpy(item["audio"]).unsqueeze(0)
            seconds = wave.shape[-1] / 16000
            spec = transcriber._audio_transforms(wave).squeeze(0).transpose(0, 1)
            spec = spec.unsqueeze(0).unsqueeze(1).transpose(2, 3).to(transcriber.device)
            with torch.no_grad():
                logp = F.log_softmax(transcriber._model(spec), dim=2)[0]          # (T, C)
            best_lp, best_idx = logp.max(dim=1)
            text = greedy.decode(logp)
            labels = transform.text_to_int(text)
            emitting = best_idx != blank
            emit_conf = float(best_lp[emitting].exp().mean()) if emitting.any() else 0.0
            if labels and len(labels) <= logp.shape[0]:
                nll = F.ctc_loss(logp.unsqueeze(1), torch.tensor([labels]), torch.tensor([logp.shape[0]]),
                                 torch.tensor([len(labels)]), blank=blank, reduction="sum", zero_infinity=True)
                seq_lp = float(-nll) / len(labels)
            else:
                seq_lp = float("-inf")
            record = {"index": item["index"], "path": item["path"], "seconds": round(seconds, 2),
                      "teacher": fold(text), "seq_lp": round(seq_lp, 4), "emit_conf": round(emit_conf, 4),
                      "chars_per_sec": round(len(text) / seconds, 2) if seconds else 0.0}
            if beam is not None:
                record["beam"] = fold(beam.decode(logp))
            if item["text"]:
                record["reference"] = fold(item["text"])
            records.append(record)
        for r in records[written:]:
            sink.write(json.dumps(r, ensure_ascii=False) + "\n")
        sink.flush()
        written = len(records)
        print(f"  {len(records)} clips", end="\r", flush=True)
    print()
    sink.close()
    return records, skipped


def summarise(records) -> None:
    import jiwer

    ok = [r for r in records if "teacher" in r]
    refd = [r for r in ok if r.get("reference") and r["teacher"]]
    print(f"\n{len(ok)} labelled, {len(records) - len(ok)} skipped, {len(refd)} with reference and a non-empty transcript")
    if len(refd) < 30:
        print("Too few references to calibrate.")
        return
    refs = [r["reference"] for r in refd]
    print(f"Greedy WER {jiwer.wer(refs, [r['teacher'] for r in refd]):.4f} | CER {jiwer.cer(refs, [r['teacher'] for r in refd]):.4f}")
    if all("beam" in r for r in refd):
        print(f"Beam+LM WER {jiwer.wer(refs, [r['beam'] for r in refd]):.4f} | CER {jiwer.cer(refs, [r['beam'] for r in refd]):.4f}")
        for r in refd:
            r["agree"] = 1.0 - min(1.0, jiwer.cer(r["teacher"], r["beam"])) if r["beam"] else 0.0

    measures = [m for m in ("seq_lp", "emit_conf", "agree") if all(m in r for r in refd)]
    print("\nWER of the kept clips (the most confident share, ranked by each measure); hours kept in brackets")
    header = f"{'measure':<11}" + "".join(f"{'top %d%%' % round(100 * f):>16}" for f in KEEP_FRACTIONS)
    print(header)
    total_h = sum(r["seconds"] for r in refd) / 3600
    for m in measures:
        ranked = sorted(refd, key=lambda r: r[m], reverse=True)
        cells = []
        for f in KEEP_FRACTIONS:
            kept = ranked[: max(5, int(len(ranked) * f))]
            wer = jiwer.wer([r["reference"] for r in kept], [r["teacher"] for r in kept])
            cells.append(f"{wer:.3f} ({sum(r['seconds'] for r in kept) / 3600:.1f}h)")
        print(f"{m:<11}" + "".join(f"{c:>16}" for c in cells))
    print(f"(all {len(refd)} clips = {total_h:.1f} h)")


def main() -> int:
    args = build_parser().parse_args()
    from pseudo_label import build_dataset

    dataset, _ = build_dataset(args)
    indices = list(range(len(dataset)))
    if args.limit and args.limit < len(indices):
        step = len(indices) / args.limit
        indices = [indices[int(k * step)] for k in range(args.limit)]
    records, skipped = label(args, dataset, indices)
    summarise(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
