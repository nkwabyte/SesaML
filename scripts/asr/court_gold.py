#!/usr/bin/env python
"""
Builds, and later scores against, a small human-corrected reference set for the
court/council recordings, which have audio but no transcripts.

A native speaker corrects a machine draft instead of typing from scratch, which is
much faster, but a draft anchors the corrector: reference text drifts toward the
drafting system's errors and flatters it. So each clip is assigned one of three
treatments at random, and the assignment is kept out of the transcriber's sheet:

    whisper    draft from the Whisper teacher
    conformer  draft from the served Conformer (beam + LM)
    none       transcribed from scratch (the unbiased control)

`score` then reports every system on the from-scratch clips (unbiased) and on the
drafted ones, so the size of the anchoring effect is measured rather than assumed.

Steps (state lives under --root, default data/court_gold):

    select   pick clips, write the anonymised pack and the private key   (laptop)
    draft    Conformer drafts on the model-input clips                   (GPU server)
    sheet    merge drafts into the spreadsheet the transcriber fills in
    score    WER/CER of each system against the corrected sheet

Whisper drafts come from scripts/asr/pseudo_label.py --audio-dir <root>/model_input.
The pack is anonymised: file names in the source reveal the parties to each hearing.
"""

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NAME = re.compile(r"^(?P<group>.+?) \((?P<index>\d+)\)(?:\.\w+)*\.wav$")
MIN_SPEECH_FRACTION = 0.5
MIN_GAP = 3            # clips apart (10 s each), so neighbours do not share a speaker turn
SKIP_MARK = "[skip]"
# Whisper falls into this memorised sentence on audio it cannot follow (it opens
# 17% of the court drafts and the worst WAXAL ones). A draft like that is not a
# transcript, so it is not shown to the corrector.
HALLUCINATIONS = ("na sane firi wɔn so kɔ soro",)
MIN_DRAFT_WORDS = 3


def usable_draft(text: str) -> bool:
    return len(text.split()) >= MIN_DRAFT_WORDS and not any(h in text.lower() for h in HALLUCINATIONS)

GUIDELINES = """\
# Transcription guidelines

You will hear short (10 second) clips cut from recorded council/court proceedings.
Open `transcription_sheet.csv`, play `audio/<audio_file>`, and fill the `corrected` column.

* Write what is said, in Asante Twi spelling, the way you would write it normally.
* If a draft is shown, fix it. If it is empty, type the transcript yourself.
* Do not add punctuation or capital letters; they are ignored. Keep apostrophes (m'abusuafo).
* The clips are cut at fixed times, so a clip may start or end in the middle of a word.
  Leave a cut-off word out unless you can clearly tell what it is.
* English words spoken inside Twi: write them as spoken, in English spelling.
* A word you cannot make out: write `[?]` in its place. Silence or no speech: write `[skip]`.
* Overlapping speakers: transcribe the clearer speaker only, and add a note.
* Use the `notes` column for anything unusual. Please do not look up or change clip order.
"""


def speech_fraction(samples, rate: int) -> float:
    """Share of 30 ms frames above the file's own speech threshold (floor+8 dB, within 35 dB of peak)."""
    import numpy as np

    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    frame, hop = int(rate * 0.03), int(rate * 0.01)
    if len(mono) < frame:
        return 0.0
    count = 1 + (len(mono) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    rms = np.sqrt(np.maximum((mono[idx] ** 2).mean(axis=1), 1e-12))
    db = 20 * np.log10(rms)
    threshold = max(np.quantile(db, 0.10) + 8.0, np.quantile(db, 0.95) - 35.0)
    return float((db > threshold).mean())


def allocate(counts: dict, n: int) -> dict:
    """Square-root-proportional quotas (small groups are not starved), largest remainder."""
    weights = {g: math.sqrt(c) for g, c in counts.items()}
    total = sum(weights.values())
    raw = {g: n * w / total for g, w in weights.items()}
    quota = {g: int(v) for g, v in raw.items()}
    for g in sorted(raw, key=lambda g: raw[g] - quota[g], reverse=True)[: n - sum(quota.values())]:
        quota[g] += 1
    return quota


def cmd_select(args) -> None:
    import soundfile as sf
    import soxr

    src, root = Path(args.src), Path(args.root)
    groups = defaultdict(list)
    for path in src.glob("*.wav"):
        m = NAME.match(path.name)
        if m:
            groups[m["group"]].append((int(m["index"]), path))
    if not groups:
        sys.exit(f"No clips named '<group> (<n>).wav' under {src}")
    quota = allocate({g: len(v) for g, v in groups.items()}, args.n)
    print("quota per group:", quota)

    rng = random.Random(args.seed)
    chosen = []
    for group, clips in sorted(groups.items()):
        order = clips[:]
        rng.shuffle(order)
        taken, rejected = [], 0
        for index, path in order:
            if len(taken) >= quota[group]:
                break
            if any(abs(index - t[0]) < MIN_GAP for t in taken):
                continue
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            if speech_fraction(samples, rate) < MIN_SPEECH_FRACTION:
                rejected += 1
                continue
            taken.append((index, path))
        print(f"  {group:<44} picked {len(taken):>3} of quota {quota[group]:>3} ({rejected} rejected as mostly silent)")
        chosen += [(group, i, p) for i, p in taken]

    rng.shuffle(chosen)
    treatments = ["none"] * args.scratch
    rest = len(chosen) - args.scratch
    treatments += ["whisper"] * (rest // 2) + ["conformer"] * (rest - rest // 2)
    rng.shuffle(treatments)

    for sub in ("pack/audio", "model_input", "private"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    with open(root / "private" / "key.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold_id", "source_file", "group", "clip_index", "treatment"])
        for n, ((group, index, path), treatment) in enumerate(zip(chosen, treatments), 1):
            gid = f"gold_{n:03d}"
            shutil.copy(path, root / "pack" / "audio" / f"{gid}.wav")
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            mono = soxr.resample(samples.mean(axis=1), rate, 16000)
            sf.write(root / "model_input" / f"{gid}.wav", mono, 16000, subtype="PCM_16")
            writer.writerow([gid, path.name, group, index, treatment])
    (root / "pack" / "GUIDELINES.md").write_text(GUIDELINES, encoding="utf-8")
    counts = defaultdict(int)
    for t in treatments:
        counts[t] += 1
    print(f"wrote {len(chosen)} clips to {root} | treatments {dict(counts)} | ~{len(chosen) * 10 / 60:.0f} min of audio")


def cmd_draft(args) -> None:
    """Conformer drafts (greedy and beam+LM) for every model-input clip."""
    from src.config import PipelineConfig
    from src.asr.inference.transcribe import Transcriber

    root = Path(args.root)
    clips = sorted((root / "model_input").glob("*.wav"))
    out = {}
    for decoder in ("greedy", "beam"):
        config = PipelineConfig()
        config.model.architecture = "conformer"
        config.decoding.decoder = decoder
        if args.lm:
            config.decoding.lm_path = args.lm
        transcriber = Transcriber(model_type="deepspeech", model_path=args.model_path, config=config)
        print(decoder, "->", transcriber.describe()["model_path"])
        for clip in clips:
            out.setdefault(clip.stem, {})[f"conformer_{decoder}"] = transcriber.transcribe(str(clip))
    (root / "private").mkdir(parents=True, exist_ok=True)
    (root / "private" / "drafts_conformer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"drafted {len(out)} clips")


def load_whisper(path: Path) -> dict:
    drafts = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "teacher" in r:
            drafts[Path(r["path"]).stem] = r["teacher"]
    return drafts


def cmd_sheet(args) -> None:
    root = Path(args.root)
    key = list(csv.DictReader(open(root / "private" / "key.csv", encoding="utf-8")))
    conformer = json.load(open(root / "private" / "drafts_conformer.json", encoding="utf-8"))
    whisper = load_whisper(Path(args.whisper))
    missing = [k["gold_id"] for k in key if k["gold_id"] not in whisper or k["gold_id"] not in conformer]
    if missing:
        sys.exit(f"Drafts missing for {len(missing)} clips (e.g. {missing[:3]}); run both drafting steps first.")

    with open(root / "pack" / "transcription_sheet.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold_id", "audio_file", "draft", "corrected", "notes"])
        shown_source = {}
        for k in sorted(key, key=lambda k: k["gold_id"]):
            gid, how = k["gold_id"], k["treatment"]
            draft = {"whisper": whisper[gid], "conformer": conformer[gid]["conformer_beam"]}.get(how, "")
            if draft and not usable_draft(draft):
                draft = ""
            shown_source[gid] = how if draft else "none"
            writer.writerow([gid, f"audio/{gid}.wav", draft, "", ""])
    with open(root / "private" / "drafts_all.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold_id", "treatment", "shown_source", "whisper", "conformer_greedy", "conformer_beam"])
        for k in key:
            g = k["gold_id"]
            writer.writerow([g, k["treatment"], shown_source[g], whisper[g], conformer[g]["conformer_greedy"], conformer[g]["conformer_beam"]])
    blanked = sum(1 for k in key if k["treatment"] != "none" and shown_source[k["gold_id"]] == "none")
    print(f"wrote {root / 'pack' / 'transcription_sheet.csv'} ({len(key)} rows) and private/drafts_all.csv; "
          f"{blanked} degenerate drafts were blanked, {sum(v != 'none' for v in shown_source.values())} drafts shown")


def cmd_score(args) -> None:
    import jiwer

    from src.asr.data.text_transform import TextTransform

    root = Path(args.root)
    transform = TextTransform()
    fold = lambda s: transform.int_to_text(transform.text_to_int(s)).strip()
    drafts = {r["gold_id"]: r for r in csv.DictReader(open(root / "private" / "drafts_all.csv", encoding="utf-8"))}
    sheet = list(csv.DictReader(open(args.sheet, encoding="utf-8-sig")))

    rows = []
    for r in sheet:
        text = (r.get("corrected") or "").strip()
        reference = fold(text.replace("[?]", " "))
        if not text or SKIP_MARK in text or not reference:
            continue
        rows.append({**drafts[r["gold_id"]], "reference": reference, "shown": fold(r.get("draft") or "")})
    print(f"{len(rows)} usable references of {len(sheet)} sheet rows "
          f"({sum('[?]' in (r.get('corrected') or '') for r in sheet)} contain [?] gaps, which count as removed words)")
    if not rows:
        return

    systems = ("whisper", "conformer_greedy", "conformer_beam")
    print("\nWER / CER against the corrected references")
    print(f"{'subset':<34}{'clips':>6}  " + "  ".join(f"{s:>22}" for s in systems))
    for label, subset in (("ALL", rows),
                          ("RANDOM CONTROL: from scratch", [r for r in rows if r["treatment"] == "none"]),
                          ("whisper draft was shown", [r for r in rows if r["shown_source"] == "whisper"]),
                          ("conformer draft was shown", [r for r in rows if r["shown_source"] == "conformer"]),
                          ("draft blanked as degenerate", [r for r in rows if r["treatment"] != "none" and r["shown_source"] == "none"])):
        if len(subset) < 5:
            print(f"{label:<34}{len(subset):>6}  too few to score")
            continue
        refs = [r["reference"] for r in subset]
        cells = []
        for s in systems:
            hyps = [fold(r[s]) for r in subset]
            cells.append(f"{jiwer.wer(refs, hyps):>10.3f} / {jiwer.cer(refs, hyps):<9.3f}")
        print(f"{label:<34}{len(subset):>6}  " + "  ".join(f"{c:>22}" for c in cells))

    shown = [r for r in rows if r["shown"]]
    if shown:
        print(f"\nHow much the corrector changed the draft they were shown: CER {jiwer.cer([r['reference'] for r in shown], [r['shown'] for r in shown]):.3f} "
              f"over {len(shown)} clips (near 0 = accepted as-is, which would bias scores toward that system)")
    print("\nCompare each system's score on the random control against the clips where its own draft was shown;"
          " a large gap is the anchoring effect. 'Blanked' clips are where the drafting system failed, so they are"
          " harder than average and are not a fair control.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/court_gold")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--src", default="data/Splitted Audio Files")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--scratch", type=int, default=20)
    s.add_argument("--seed", type=int, default=21)
    d = sub.add_parser("draft")
    d.add_argument("--model-path", default=None, help="Defaults to the served registry version")
    d.add_argument("--lm", default=None, help="LM json for the beam pass (default: configured LM)")
    h = sub.add_parser("sheet")
    h.add_argument("--whisper", required=True, help="JSONL from pseudo_label.py over <root>/model_input")
    c = sub.add_parser("score")
    c.add_argument("--sheet", required=True, help="The transcription sheet, filled in")
    args = p.parse_args()
    {"select": cmd_select, "draft": cmd_draft, "sheet": cmd_sheet, "score": cmd_score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
