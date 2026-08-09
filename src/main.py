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
from src.utils.run_logger import RunManager, resolve_checkpoint

# torchaudio's MelSpectrogram defaults to n_fft=400, hop_length=n_fft//2.
MEL_HOP_LENGTH = 200
# Beyond this many mel frames a clip dominates GPU memory at normal batch sizes.
LONG_CLIP_FRAMES = 1200
MIN_VOCAB_COVERAGE = 0.97

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
    """
    if hf_datasets:
        specs = [hf_datasets] if isinstance(hf_datasets, str) else list(hf_datasets)
        from src.data.hf_dataset import combine_datasets, load_hf_datasets
        parts = load_hf_datasets(specs, sample_rate=config.audio.sample_rate, default_split=split)
        return combine_datasets(parts), [part.probe() for part in parts]

    csv_file = csv_path or os.path.join(config.paths.corpus_dir, "verified_data.csv")
    dataset = AkanAudioDataset(csv_file=csv_file, sample_rate=config.audio.sample_rate)
    return dataset, [{"csv": csv_file, "rows": len(dataset)}]

def log_corpus_report(run, report: dict, role: str, config: PipelineConfig) -> None:
    """Logs a corpus probe and warns about the two things that silently bite."""
    run.logger.info("%s corpus: %s", role, report)

    median = report.get("duration_median_sec")
    if median and median * config.audio.sample_rate / MEL_HOP_LENGTH > LONG_CLIP_FRAMES:
        run.logger.warning(
            "%s clips average %.1fs (~%d mel frames). That is %.0fx the sequence length of a "
            "typical 4s clip; reduce --batch-size (try 2) if training runs out of memory.",
            report.get("dataset"), median, median * config.audio.sample_rate / MEL_HOP_LENGTH,
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

    coverage = report.get("coverage")
    if coverage is not None and coverage < MIN_VOCAB_COVERAGE:
        run.logger.warning(
            "%s vocabulary coverage is %.1f%% (dropped: %s)",
            report.get("dataset"), 100 * coverage, report.get("dropped_chars")
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
        kwargs["prefetch_factor"] = 2
    return kwargs


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

    run = RunManager(kind="train", config=config, run_id=args.run_id, params=vars(args))
    try:
        text_transform = TextTransform()
        train_transforms = get_train_audio_transforms(config.audio.sample_rate, config.audio.n_mels)
        valid_transforms = get_valid_audio_transforms(config.audio.sample_rate, config.audio.n_mels)

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

        model = build_model(config, text_transform.vocab_size)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            text_transform=text_transform,
            run=run
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
        valid_transforms = get_valid_audio_transforms(config.audio.sample_rate, config.audio.n_mels)

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
            model.load_state_dict(torch.load(model_path, map_location=config.device))
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
