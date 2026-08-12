#!/usr/bin/env python
"""
Prune outputs/ down to the artifacts that are actually in use.

Training leaves a lot behind: three checkpoint files per run, of which the
resumable one is three times the size of the weights, plus a run directory and a
log. After a handful of iterations that is gigabytes, nearly all of it from runs
that were abandoned, failed, or were smoke tests.

What is never deleted:

* **outputs/asr/registry/** - the published model exports. This is what the app and
  CLI serve, so removing it breaks the demo.
* **The runs that produced published versions.** Their paths are read out of the
  registry metadata rather than hard-coded, so protection follows whatever has
  actually been published.
* Anything named with --keep.

Dry run by default. Nothing is removed without --apply.

    python scripts/maintenance/clean_outputs.py                    # what would go
    python scripts/maintenance/clean_outputs.py --apply            # do it
    python scripts/maintenance/clean_outputs.py --keep run-big --apply
    python scripts/maintenance/clean_outputs.py --checkpoints-only --apply
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Run ids matching these are disposable regardless of anything else: they are
# the throwaway names used by smoke tests, verification and dataset downloads.
DISPOSABLE_PATTERNS = (
    re.compile(r"^smoke-"),
    re.compile(r"^verify-"),
    re.compile(r"^demo-"),
    re.compile(r"^download-"),
)


def human(size: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def protected_runs(outputs: Path) -> set:
    """
    Run ids that produced a published model version.

    Read from the registry rather than listed here, so this keeps working as
    versions are published and promoted across future training iterations.
    """
    from src.utils.model_registry import ModelRegistry

    keep = set()
    registry = ModelRegistry(str(outputs))
    for architecture in registry.architectures():
        for version in registry.versions(architecture):
            if version.run_id:
                keep.add(version.run_id)
            # The source path is recorded too; its parent is the run directory.
            # Split on both separators: a version published on the Windows GPU
            # box records `outputs\asr\checkpoints\<run>\...`, and PurePath on POSIX
            # treats that whole string as one filename, yielding an empty parent.
            source = version.metadata.get("source_checkpoint")
            if source:
                parts = [p for p in source.replace("\\", "/").split("/") if p]
                if len(parts) >= 2:
                    keep.add(parts[-2])
    return keep


def is_disposable(name: str) -> bool:
    return any(pattern.match(name) for pattern in DISPOSABLE_PATTERNS)


def published_digests(outputs: Path) -> dict:
    """SHA-256 -> "arch/version" for every published set of weights."""
    from src.utils.model_registry import ModelRegistry

    digests = {}
    registry = ModelRegistry(str(outputs))
    for architecture in registry.architectures():
        for version in registry.versions(architecture):
            digest = version.metadata.get("sha256")
            if digest:
                digests[digest] = f"{architecture}/{version.version}"
    return digests


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_weights(outputs: Path, keep: set) -> list:
    """
    Checkpoint files inside protected runs that are byte-identical to a
    published version.

    Publishing copies the weights, so after every run the same bytes sit in both
    the run directory and the registry. Removing the copy is safe by
    construction - the hashes are compared, not assumed - while anything unique
    to the run survives. `last_model.pt` in particular carries optimizer and
    schedule state that the registry never stores, so it is exactly the kind of
    file this must not touch.
    """
    digests = published_digests(outputs)
    if not digests:
        return []

    duplicates = []
    for run_id in sorted(keep):
        directory = outputs / "checkpoints" / run_id
        if not directory.is_dir():
            continue
        for weights in sorted(directory.glob("*.pt*")):
            published_as = digests.get(file_digest(weights))
            if published_as:
                duplicates.append(
                    (weights, weights.stat().st_size, f"already published as {published_as}")
                )
    return duplicates


def plan(domain_root: Path, keep: set, checkpoints_only: bool, dedupe: bool = True,
         outputs: Path = None) -> list:
    """
    Returns (path, size, reason) for everything that would be removed.

    `domain_root` holds the per-model artifacts (outputs/asr/, later
    outputs/translation/); `outputs` is the shared root that owns logs/ and
    defaults to the domain root when not given.
    """
    removals = []
    outputs = outputs or domain_root

    if dedupe:
        removals.extend(duplicate_weights(domain_root, keep))

    for directory in sorted((domain_root / "checkpoints").glob("*")):
        if not directory.is_dir():
            continue
        if directory.name in keep:
            continue
        reason = "smoke/verify run" if is_disposable(directory.name) else "superseded run"
        removals.append((directory, directory_size(directory), reason))

    # Exported artifacts follow the same rule as checkpoints: an export from a
    # smoke test is disposable, one from a published run is the deliverable.
    for directory in sorted((domain_root / "exports").glob("*")):
        if not directory.is_dir() or directory.name in keep:
            continue
        if is_disposable(directory.name):
            removals.append((directory, directory_size(directory), "smoke/verify export"))

    if not checkpoints_only:
        for directory in sorted((domain_root / "runs").glob("*")):
            if not directory.is_dir() or directory.name in keep:
                continue
            if is_disposable(directory.name):
                removals.append((directory, directory_size(directory), "smoke/verify run"))

        for log in sorted((outputs / "logs").glob("*.log")):
            if log.stem in keep:
                continue
            if is_disposable(log.stem):
                removals.append((log, log.stat().st_size, "smoke/verify log"))

    # Editor and OS droppings, wherever they landed.
    for junk in outputs.rglob(".DS_Store"):
        removals.append((junk, junk.stat().st_size, "OS metadata"))

    return removals


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", default="outputs", help="Outputs directory to prune")
    parser.add_argument("--domain", default="asr",
                        help="Model domain whose artifacts to prune (asr, translation, ...). "
                             "logs/ is shared and pruned alongside whichever domain is given.")
    parser.add_argument("--apply", action="store_true", help="Actually delete; otherwise dry run")
    parser.add_argument("--keep", action="append", default=[], metavar="RUN_ID",
                        help="Protect a run id in addition to the published ones. Repeatable.")
    parser.add_argument("--checkpoints-only", action="store_true",
                        help="Leave run directories and logs alone; prune weights only")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false", default=True,
                        help="Keep run-directory weights even when byte-identical to a published version")
    args = parser.parse_args()

    outputs = (ROOT / args.outputs) if not Path(args.outputs).is_absolute() else Path(args.outputs)
    if not outputs.is_dir():
        print(f"No outputs directory at {outputs}")
        return 1

    domain_root = outputs / args.domain
    if not domain_root.is_dir():
        print(f"No '{args.domain}' artifacts at {domain_root}")
        return 1

    keep = set(args.keep) | protected_runs(domain_root)
    removals = plan(domain_root, keep, args.checkpoints_only, dedupe=args.dedupe, outputs=outputs)

    print(f"Protected run ids: {', '.join(sorted(keep)) or 'none'}")
    print(f"Never touched:     {domain_root.name}/registry/ (the served model exports)\n")

    if not removals:
        print("Nothing to clean.")
        return 0

    total = 0
    for path, size, reason in removals:
        total += size
        print(f"  {human(size):>8}  {path.relative_to(outputs.parent)}  ({reason})")

    print(f"\n{'Removed' if args.apply else 'Would remove'} {len(removals)} items, {human(total)}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    for path, _, _ in removals:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    # Keep the directory skeleton so the pipeline has somewhere to write.
    for directory in [outputs / "logs"] + [domain_root / n for n in ("runs", "checkpoints", "exports")]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
