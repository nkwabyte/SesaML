import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.text_transform import TextTransform
from src.data.audio_transforms import get_train_audio_transforms, get_valid_audio_transforms
from src.data.dataset import AkanAudioDataset, AudioCollator
from src.data.hf_dataset import parse_dataset_spec
from src.inference.export import EXPORT_FORMATS, export_model
from src.inference.transcribe import load_deepspeech_model, transcribe_audio
from src.models import ARCHITECTURES, build_architecture, subsampling_factor
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.run_logger import RunManager, load_weights, resolve_checkpoint

# Beyond this many mel frames a clip dominates GPU memory at normal batch sizes.
LONG_CLIP_FRAMES = 1200
MIN_VOCAB_COVERAGE = 0.97


def _is_punctuation(ch: str) -> bool:
    """True for marks a speaker does not pronounce, which CTC targets shed on purpose."""
    return not ch.isalnum() and not ch.isspace()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SesaML - Akan Audio Speech-to-Text Transcriber & Trainer")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    arch_choices = list(ARCHITECTURES.keys())

    # Transcribe sub-command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe Akan audio file to text")
    transcribe_parser.add_argument("--audio", "-a", required=True, help="Path to input audio file (WAV/MP3/FLAC)")
    transcribe_parser.add_argument("--model-type", choices=["deepspeech", "whisper"], default="deepspeech", help="Model type to use for transcription")
    transcribe_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="CTC model architecture when model-type is deepspeech")
    transcribe_parser.add_argument("--model-path", default=None, help="Path to model checkpoint (.pt / .pth)")
    transcribe_parser.add_argument("--whisper-repo", default="CiBeDL/twi_trained_whisper", help="HuggingFace repository ID for Whisper model")
    transcribe_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    transcribe_parser.add_argument("--noise-reduction", action="store_true", help="Apply spectral gate noise reduction before transcription")
    transcribe_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train DeepSpeech or Conformer CTC model on Akan dataset")
    train_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="Model architecture: deepspeech, conformer, or conformer-medium")
    train_parser.add_argument("--csv-path", default=None, help="Path to dataset CSV file containing 'audio_path' and 'text' columns")
    train_parser.add_argument("--hf-dataset", action="append", default=None, metavar="REPO[:SPLIT]", help="HuggingFace dataset to train on. Repeat the flag to combine corpora, e.g. --hf-dataset ghanaopendata/twi-speech-text-multispeaker-16k --hf-dataset Lagyamfi/akan_audio_processed:train")
    train_parser.add_argument("--split", default="train", help="Default split for --hf-dataset entries that omit one")
    train_parser.add_argument("--val-dataset", action="append", default=None, metavar="REPO[:SPLIT]", help="HuggingFace dataset used for validation, e.g. Lagyamfi/akan_audio_processed:test")
    train_parser.add_argument("--val-split", default=None, help="Validation split of the training dataset (enables per-epoch WER/CER logging)")
    train_parser.add_argument("--val-csv-path", default=None, help="Path to validation CSV file (enables per-epoch WER/CER logging)")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    train_parser.add_argument("--resume", default=None, metavar="CHECKPOINT", help="Continue training from a run's last_model.pt or best_model.pt, restoring optimizer and schedule state")
    train_parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes (default: one per core, capped at 8)")
    train_parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=None, help="Disable CUDA mixed precision (enabled by default on CUDA)")
    train_parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm (default 5.0; 0 disables clipping)")
    train_parser.add_argument("--allow-no-validation", action="store_true", help="Permit a multi-epoch run with no validation set (no WER/CER will be computed)")
    train_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Evaluate sub-command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model WER and CER")
    eval_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="Model architecture: deepspeech, conformer, or conformer-medium")
    eval_parser.add_argument("--csv-path", default=None, help="Path to evaluation dataset CSV file")
    eval_parser.add_argument("--hf-dataset", action="append", default=None, metavar="REPO[:SPLIT]", help="HuggingFace dataset to evaluate on. Repeat to combine corpora.")
    eval_parser.add_argument("--split", default="train", help="Default split for --hf-dataset entries that omit one")
    eval_parser.add_argument("--model-path", default=None, help="Path to model checkpoint")
    eval_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    eval_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Export sub-command
    export_parser = subparsers.add_parser("export", help="Export a trained checkpoint to a deployable model file")
    export_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="Model architecture: deepspeech, conformer, or conformer-medium")
    export_parser.add_argument("--model-path", default=None, help="Path to trained checkpoint (state dict)")
    export_parser.add_argument("--format", dest="export_format", choices=list(EXPORT_FORMATS), default="torchscript", help="Export format: torchscript (.pt), state_dict (.pth) or executorch (.pte)")
    export_parser.add_argument("--basename", default="speech_recognition_model", help="Base filename for the exported artifact")
    export_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    export_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    return parser

def build_dataset(config: PipelineConfig, hf_datasets, split, csv_path):
    """
    Builds the dataset to train or evaluate on. One or more HuggingFace corpora
    (`repo[:split]` specs) are concatenated; otherwise a CSV-backed dataset is used.
    Returns the dataset plus a per-corpus description for the run log.

    There is deliberately no default corpus. This used to fall back to
    `data/corpus/verified_data.csv`, which is a text-only translation table with
    no audio: a bare `python -m src.main train` silently trained on it for 30
    epochs. scripts/train.sh supplies the real corpora, and anyone bypassing the
    script should have to say what they mean.
    """
    if hf_datasets:
        specs = [hf_datasets] if isinstance(hf_datasets, str) else list(hf_datasets)
        from src.data.hf_dataset import combine_datasets, load_hf_datasets
        parts = load_hf_datasets(specs, sample_rate=config.audio.sample_rate, default_split=split)
        return combine_datasets(parts), [part.probe() for part in parts]

    if not csv_path:
        raise ValueError(
            "No data source given. Pass --hf-dataset REPO[:SPLIT] (repeatable) or "
            "--csv-path PATH to a manifest with audio-path and transcription columns. "
            "scripts/train.sh supplies the project's default Akan corpora."
        )

    dataset = AkanAudioDataset(csv_file=csv_path, sample_rate=config.audio.sample_rate)
    # Probing costs a few seconds and catches a corpus whose audio paths are all
    # stale before the run burns an hour producing a blank-emitting model.
    return dataset, [dataset.probe()]

def log_corpus_report(run, report: dict, role: str, config: PipelineConfig) -> None:
    """
    Logs a corpus probe, then aborts on the failures that otherwise train to a
    perfect-looking zero loss, and warns about the ones that merely degrade it.
    """
    run.logger.info("%s corpus: %s", role, report)

    name = report.get("dataset")
    sampled = report.get("sampled") or 0

    if report.get("rows") == 0:
        raise ValueError(f"{role} corpus '{name}' is empty - nothing to {role.lower()} on.")

    # A corpus whose sampled audio is entirely absent will either raise per-row
    # mid-epoch or, worse, train on noise. Fail now, with the path in hand.
    missing = report.get("audio_missing")
    if missing is not None and sampled and missing == sampled:
        raise FileNotFoundError(
            f"{role} corpus '{name}': none of the {sampled} sampled audio files exist. "
            f"The manifest's paths are wrong, relative to another machine, or the audio "
            f"was never downloaded. Check the '{report.get('audio_column')}' column."
        )
    if missing:
        run.logger.warning(
            "%s corpus '%s': %d of %d sampled audio files are missing; those rows will "
            "raise during training.", role, name, missing, sampled
        )

    # Zero-length CTC targets are minimised by emitting blanks, so the loss goes
    # to exactly 0.0 and the run looks converged while having learned nothing.
    text_sampled = report.get("text_sampled") or 0
    empty = report.get("empty_labels")
    if empty is not None and text_sampled and empty == text_sampled:
        raise ValueError(
            f"{role} corpus '{name}': all {text_sampled} sampled transcripts encode to "
            f"empty label sequences. Column '{report.get('text_column')}' is blank or "
            f"holds text entirely outside the CTC vocabulary. Training on this yields a "
            f"CTC loss of 0.0 and a model that only emits blanks."
        )
    if empty:
        run.logger.warning(
            "%s corpus '%s': %d of %d sampled transcripts are empty after encoding.",
            role, name, empty, text_sampled
        )

    median = report.get("duration_median_sec")
    if median and median * config.audio.frames_per_second > LONG_CLIP_FRAMES:
        run.logger.warning(
            "%s clips average %.1fs (~%d mel frames). That is %.0fx the sequence length of a "
            "typical 4s clip; reduce --batch-size (try 2) if training runs out of memory.",
            report.get("dataset"), median, median * config.audio.frames_per_second,
            median / 4.0
        )

    digits = report.get("digits_dropped") or 0
    if digits:
        run.logger.warning(
            "%s transcripts contain digits the vocabulary is dropping - the audio says them "
            "but the labels will not. TextTransform includes 0-9 by default, so this means "
            "include_digits=False or a custom char list is in use.",
            report.get("dataset")
        )

    # Punctuation is not spoken, so dropping it from CTC targets is correct and
    # accounts for essentially all of the ~4% these corpora lose. Warning on the
    # raw coverage would fire on every healthy corpus and train people to ignore
    # it; what matters is whether *letters* are going missing.
    dropped = report.get("dropped_chars") or []
    lost_letters = [(ch, n) for ch, n in dropped if not _is_punctuation(ch)]
    coverage = report.get("coverage")

    if lost_letters:
        run.logger.warning(
            "%s drops %d non-punctuation characters the audio still says: %s. "
            "These are outside the CTC vocabulary, so the labels will not match "
            "the speech.", report.get("dataset"), sum(n for _, n in lost_letters), lost_letters
        )
    elif coverage is not None and coverage < MIN_VOCAB_COVERAGE:
        run.logger.info(
            "%s vocabulary coverage is %.1f%%; all dropped characters are punctuation (%s).",
            report.get("dataset"), 100 * coverage, dropped
        )


def loader_kwargs(config: PipelineConfig) -> dict:
    """
    DataLoader settings shared by every command.

    Decoding and mel-transforming audio on the main process starves the GPU,
    badly so for long clips, hence num_workers. pin_memory only helps for CUDA
    transfers and warns on other backends, so it is conditional.
    """
    kwargs = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory and config.device == "cuda",
    }
    if config.training.num_workers > 0:
        kwargs["persistent_workers"] = True
        # Four batches queued per worker keeps the GPU fed across the jitter in
        # decode time between a 1-second clip and a 30-second one.
        kwargs["prefetch_factor"] = 4
    return kwargs


def train_transforms_for(config: PipelineConfig):
    """Training front-end built from the audio config, so every caller agrees on the features."""
    return get_train_audio_transforms(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
        freq_mask_param=config.audio.freq_mask_param,
        time_mask_param=config.audio.time_mask_param,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        time_mask_ratio=config.audio.time_mask_ratio,
    )


def valid_transforms_for(config: PipelineConfig):
    """Validation/inference front-end; identical features, no augmentation."""
    return get_valid_audio_transforms(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
    )


def build_model(config: PipelineConfig, n_class: int) -> torch.nn.Module:
    return build_architecture(config.model.architecture, n_class, config)

def run_transcribe(args, config: PipelineConfig) -> None:
    if getattr(args, "architecture", None):
        config.model.architecture = args.architecture
    if getattr(args, "device", None):
        config.device = args.device
    with RunManager(kind="transcribe", config=config, run_id=args.run_id, params=vars(args)) as run:
        run.logger.info("Transcribing %s using %s (arch=%s, device=%s)", args.audio, args.model_type, config.model.architecture, config.device)
        transcript = transcribe_audio(
            audio_path=args.audio,
            model_type=args.model_type,
            model_path=args.model_path,
            whisper_repo=args.whisper_repo,
            config=config,
            apply_noise_reduction=args.noise_reduction
        )
        run.write_json("transcription.json", {
            "audio_path": args.audio,
            "model_type": args.model_type,
            "architecture": config.model.architecture,
            "device": config.device,
            "model_path": args.model_path,
            "whisper_repo": args.whisper_repo if args.model_type == "whisper" else None,
            "noise_reduction": args.noise_reduction,
            "transcript": transcript,
        })
        run.write_text("transcript.txt", transcript)
        run.log_metrics({"characters": len(transcript), "words": len(transcript.split())}, stage="transcribe")
        run.logger.info("Transcript: %s", transcript)

        print("\n" + "=" * 40)
        print("TRANSCRIPTION RESULT:")
        print(transcript)
        print("=" * 40)
        print(f"Saved to: {run.run_dir}")

def run_train(args, config: PipelineConfig) -> None:
    if getattr(args, "architecture", None):
        config.model.architecture = args.architecture
    if getattr(args, "device", None):
        config.device = args.device
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.use_amp is not None:
        config.training.use_amp = args.use_amp
    if args.grad_clip is not None:
        config.training.grad_clip_norm = args.grad_clip

    run = RunManager(kind="train", config=config, run_id=args.run_id, params=vars(args))
    try:
        text_transform = TextTransform()
        train_transforms = train_transforms_for(config)
        valid_transforms = valid_transforms_for(config)

        dataset, sources = build_dataset(config, args.hf_dataset, args.split, args.csv_path)
        for source in sources:
            log_corpus_report(run, source, "Train", config)
        run.logger.info("Training samples: %d | Architecture: %s", len(dataset), config.model.architecture)

        stride = subsampling_factor(config.model.architecture)
        train_collate = AudioCollator(text_transform, train_transforms, stride=stride)

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=train_collate,
            **loader_kwargs(config)
        )

        val_loader = None
        val_sources = []
        if args.val_dataset or args.val_split or args.val_csv_path:
            # --val-dataset names its own corpora; --val-split reuses the training
            # corpora with a different split.
            val_specs = args.val_dataset
            if not val_specs and args.val_split and args.hf_dataset:
                val_specs = [
                    f"{parse_dataset_spec(spec, args.split)[0]}:{args.val_split}"
                    for spec in args.hf_dataset
                ]

            val_dataset, val_sources = build_dataset(
                config,
                val_specs,
                args.val_split or args.split,
                args.val_csv_path
            )
            for source in val_sources:
                log_corpus_report(run, source, "Validation", config)
            run.logger.info("Validation samples: %d", len(val_dataset))

            valid_collate = AudioCollator(text_transform, valid_transforms, stride=stride)

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                collate_fn=valid_collate,
                **loader_kwargs(config)
            )
        elif config.training.epochs > 1 and not args.allow_no_validation:
            # Without a validation set there is no WER, no CER, and best-model
            # selection silently falls back to training loss - which is exactly
            # how a run that had learned nothing still reported a "best epoch".
            raise ValueError(
                f"Refusing to train {config.training.epochs} epochs with no validation set: "
                f"there would be no WER/CER and the best checkpoint would be chosen on "
                f"training loss alone. Pass --val-dataset REPO:SPLIT, --val-split SPLIT or "
                f"--val-csv-path PATH, or --allow-no-validation to proceed anyway."
            )

        model = build_model(config, text_transform.vocab_size)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            text_transform=text_transform,
            run=run,
            resume_from=args.resume
        )
        trainer.train()

        run.finish(status=trainer.status, summary={
            "train_samples": len(dataset),
            "val_samples": len(val_loader.dataset) if val_loader else 0,
            "train_corpora": sources,
            "val_corpora": val_sources,
            **trainer.summary,
        })
        print(f"\nCheckpoints: {run.checkpoint_dir}")
        print(f"Logs & metrics: {run.run_dir}")
    except BaseException as exc:
        run.logger.exception("Training run failed")
        run.finish(status="failed", summary={"error": f"{type(exc).__name__}: {exc}"})
        raise

def run_evaluate(args, config: PipelineConfig) -> None:
    if getattr(args, "architecture", None):
        config.model.architecture = args.architecture
    if getattr(args, "device", None):
        config.device = args.device
    with RunManager(kind="evaluate", config=config, run_id=args.run_id, params=vars(args)) as run:
        text_transform = TextTransform()
        valid_transforms = valid_transforms_for(config)

        dataset, sources = build_dataset(config, args.hf_dataset, args.split, args.csv_path)
        for source in sources:
            log_corpus_report(run, source, "Evaluation", config)
        run.logger.info("Evaluation samples: %d | Architecture: %s | Device: %s", len(dataset), config.model.architecture, config.device)

        stride = subsampling_factor(config.model.architecture)
        collate_fn = AudioCollator(text_transform, valid_transforms, stride=stride)

        val_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs(config)
        )

        model_path = resolve_checkpoint(args.model_path, config)
        model = build_model(config, text_transform.vocab_size)
        if os.path.exists(model_path):
            model.load_state_dict(load_weights(model_path, map_location=config.device))
            run.logger.info("Loaded checkpoint %s", model_path)
        else:
            run.logger.warning("Checkpoint %s not found - evaluating an untrained model", model_path)

        criterion = torch.nn.CTCLoss(blank=text_transform.blank_label)
        evaluator = Evaluator(model, criterion, config.device, text_transform)
        metrics = evaluator.evaluate(val_loader, return_predictions=True)
        predictions = metrics.pop("predictions", [])

        run.log_metrics(metrics, stage="evaluate")
        run.log_predictions(predictions[: config.training.max_logged_predictions])
        run.write_json("evaluation.json", {"model_path": model_path, "corpora": sources, **metrics})
        run.finish(status="completed", summary={"model_path": model_path, "corpora": sources, **metrics})

        print("\n" + "=" * 40)
        print("EVALUATION RESULTS:")
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"Average WER:     {metrics['wer']:.4f}")
        print(f"Average CER:     {metrics['cer']:.4f}")
        print("=" * 40)
        print(f"Saved to: {run.run_dir}")

def run_export(args, config: PipelineConfig) -> None:
    if getattr(args, "architecture", None):
        config.model.architecture = args.architecture
    if getattr(args, "device", None):
        config.device = args.device
    with RunManager(kind="export", config=config, run_id=args.run_id, params=vars(args)) as run:
        text_transform = TextTransform()
        model_path = resolve_checkpoint(args.model_path, config)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}. Train a model first or pass --model-path."
            )

        model = load_deepspeech_model(model_path=model_path, config=config, text_transform=text_transform)
        run.logger.info("Exporting %s as %s", model_path, args.export_format)

        manifest = export_model(
            model=model,
            export_dir=run.ensure_export_dir(),
            export_format=args.export_format,
            config=config,
            basename=args.basename
        )
        manifest["source_checkpoint"] = model_path
        manifest["vocab_size"] = text_transform.vocab_size
        run.write_json("export_manifest.json", manifest)
        run.finish(status="completed", summary=manifest)

        print("\n" + "=" * 40)
        print("EXPORT COMPLETE:")
        print(f"Artifact: {manifest['path']}")
        print(f"Size:     {manifest['size_bytes'] / 1e6:.2f} MB")
        print(f"SHA256:   {manifest['sha256']}")
        print("=" * 40)
        print(f"Manifest: {run.run_dir / 'export_manifest.json'}")

def main():
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig()

    handlers = {
        "transcribe": run_transcribe,
        "train": run_train,
        "evaluate": run_evaluate,
        "export": run_export,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return

    handler(args, config)

if __name__ == "__main__":
    main()
