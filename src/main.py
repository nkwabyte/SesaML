import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.text_transform import TextTransform
from src.data.audio_transforms import get_train_audio_transforms, get_valid_audio_transforms
from src.data.bucketing import LengthBucketedBatchSampler, dataset_lengths, padding_efficiency
from src.diarization import BACKENDS as DIARIZATION_BACKENDS
from src.diarization import DiarizedTranscriber, format_transcript
from src.data.dataset import AkanAudioDataset, AudioCollator
from src.data.hf_dataset import parse_dataset_spec
from src.inference.export import EXPORT_FORMATS, export_model
from src.inference.transcribe import load_deepspeech_model, transcribe_audio
from src.models import ARCHITECTURES, build_architecture, subsampling_factor
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.model_registry import ModelRegistry
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
    transcribe_parser.add_argument("--whisper-repo", default=None, help="HuggingFace repository ID for the Whisper comparison baseline. No default: this project serves its own trained models from outputs/registry/")
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
    train_parser.add_argument("--init-weights", default=None, metavar="CHECKPOINT", help="Path to pretrained model weights (.pt or checkpoint) to initialize model parameters before training (for fine-tuning on new corpora)")
    train_parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes (default: one per core, capped at 8)")
    train_parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=None, help="Disable CUDA mixed precision (enabled by default on CUDA)")
    train_parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm (default 5.0; 0 disables clipping)")
    train_parser.add_argument("--allow-no-validation", action="store_true", help="Permit a multi-epoch run with no validation set (no WER/CER will be computed)")
    train_parser.add_argument("--no-bucket-batches", dest="bucket_batches", action="store_false", default=True, help="Disable length-bucketed batching (batches clips of similar duration together to avoid paying compute for padding)")
    train_parser.add_argument("--limit-rows", type=int, default=None, metavar="N", help="Use at most N evenly-spaced rows from each --hf-dataset, for bounding a run against a very large corpus")
    train_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Diarize sub-command
    diarize_parser = subparsers.add_parser("diarize", help="Transcribe a multi-speaker recording with speaker labels")
    diarize_parser.add_argument("--audio", "-a", required=True, help="Path to input audio file (WAV/MP3/FLAC)")
    diarize_parser.add_argument("--model-type", choices=["deepspeech", "whisper"], default="deepspeech", help="ASR backend used for each speaker turn")
    diarize_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="CTC model architecture when model-type is deepspeech")
    diarize_parser.add_argument("--model-path", default=None, help="Path to model checkpoint (.pt / .pth)")
    diarize_parser.add_argument("--backend", choices=list(DIARIZATION_BACKENDS) + ["auto"], default="auto", help="Diarization backend: pyannote (best, gated), ecapa (public weights), spectral (no downloads), or auto")
    diarize_parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers, when known. Improves the ecapa and spectral backends considerably")
    diarize_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    diarize_parser.add_argument("--noise-reduction", action="store_true", help="Apply spectral gate noise reduction before transcription")
    diarize_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Evaluate sub-command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model WER and CER")
    eval_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="Model architecture: deepspeech, conformer, or conformer-medium")
    eval_parser.add_argument("--csv-path", default=None, help="Path to evaluation dataset CSV file")
    eval_parser.add_argument("--hf-dataset", action="append", default=None, metavar="REPO[:SPLIT]", help="HuggingFace dataset to evaluate on. Repeat to combine corpora.")
    eval_parser.add_argument("--split", default="train", help="Default split for --hf-dataset entries that omit one")
    eval_parser.add_argument("--model-path", default=None, help="Path to model checkpoint")
    eval_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    eval_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Models sub-command: the versioned registry of trained exports
    models_parser = subparsers.add_parser("models", help="List, promote or roll back versioned model exports")
    models_sub = models_parser.add_subparsers(dest="models_command")

    models_sub.add_parser("list", help="Show every published version and which one is served")

    show_parser = models_sub.add_parser("show", help="Full metadata for one version")
    show_parser.add_argument("--architecture", required=True, choices=arch_choices)
    show_parser.add_argument("--version", default=None, help="Version id (default: the promoted one)")

    publish_parser = models_sub.add_parser("publish", help="Archive a checkpoint as a new version")
    publish_parser.add_argument("--model-path", required=True, help="Checkpoint to publish")
    publish_parser.add_argument("--architecture", required=True, choices=arch_choices)
    publish_parser.add_argument("--wer", type=float, default=None, help="Word error rate, used to decide promotion")
    publish_parser.add_argument("--cer", type=float, default=None, help="Character error rate")
    publish_parser.add_argument("--run-id-of", dest="source_run_id", default=None, help="Run that produced these weights")
    publish_parser.add_argument("--notes", default=None, help="Free-text note stored with the version")
    publish_parser.add_argument("--promote", action="store_true", help="Serve it regardless of whether it beats the current version")

    promote_parser = models_sub.add_parser("promote", help="Make a published version the one that is served")
    promote_parser.add_argument("--architecture", required=True, choices=arch_choices)
    promote_parser.add_argument("--version", required=True, help="Version id, e.g. v002")

    rollback_parser = models_sub.add_parser("rollback", help="Revert to the previously promoted version")
    rollback_parser.add_argument("--architecture", required=True, choices=arch_choices)

    models_parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    models_parser.add_argument("--device", default=None, help=argparse.SUPPRESS)

    # Export sub-command
    export_parser = subparsers.add_parser("export", help="Export a trained checkpoint to a deployable model file")
    export_parser.add_argument("--architecture", choices=arch_choices, default="deepspeech", help="Model architecture: deepspeech, conformer, or conformer-medium")
    export_parser.add_argument("--model-path", default=None, help="Path to trained checkpoint (state dict)")
    export_parser.add_argument("--format", dest="export_format", choices=list(EXPORT_FORMATS), default="torchscript", help="Export format: torchscript (.pt), state_dict (.pth) or executorch (.pte)")
    export_parser.add_argument("--basename", default="speech_recognition_model", help="Base filename for the exported artifact")
    export_parser.add_argument("--device", default=None, help="Compute device to use (cpu, cuda, mps)")
    export_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    return parser

def build_dataset(config: PipelineConfig, hf_datasets, split, csv_path, limit_rows=None):
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
        parts = load_hf_datasets(
            specs, sample_rate=config.audio.sample_rate, default_split=split, limit_rows=limit_rows
        )
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


def build_train_loader(config: PipelineConfig, dataset, collate_fn, run, bucket: bool) -> DataLoader:
    """
    Training loader, length-bucketed unless asked otherwise.

    Bucketing needs every clip's duration up front, which for a corpus with no
    duration column means one header sweep (cached afterwards). That is worth it
    whenever clip lengths vary: batching a 0.2s clip with a 30s one pads the
    short one out to the long one's length and pays full compute for silence.
    """
    kwargs = loader_kwargs(config)

    if not bucket:
        return DataLoader(
            dataset=dataset, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=collate_fn, **kwargs
        )

    try:
        lengths = dataset_lengths(dataset, sample_rate=config.audio.sample_rate)
    except TypeError as exc:
        run.logger.warning("Falling back to shuffled batching: %s", exc)
        return DataLoader(
            dataset=dataset, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=collate_fn, **kwargs
        )

    sampler = LengthBucketedBatchSampler(
        lengths, batch_size=config.training.batch_size, shuffle=True
    )

    # Report what bucketing bought, against what plain shuffling would have cost.
    shuffled = [list(range(i, min(i + config.training.batch_size, len(lengths))))
                for i in range(0, len(lengths), config.training.batch_size)]
    bucketed = list(iter(sampler))
    run.logger.info(
        "Length-bucketed batching: %.0f%% of batched frames carry audio (%.0f%% shuffled). "
        "Clips span %.2fs to %.2fs.",
        100 * padding_efficiency(lengths, bucketed),
        100 * padding_efficiency(lengths, shuffled),
        min(lengths), max(lengths)
    )

    return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn, **kwargs)


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

        dataset, sources = build_dataset(
            config, args.hf_dataset, args.split, args.csv_path, limit_rows=args.limit_rows
        )
        for source in sources:
            log_corpus_report(run, source, "Train", config)
        run.logger.info("Training samples: %d | Architecture: %s", len(dataset), config.model.architecture)

        stride = subsampling_factor(config.model.architecture)
        train_collate = AudioCollator(text_transform, train_transforms, stride=stride)

        train_loader = build_train_loader(config, dataset, train_collate, run, args.bucket_batches)

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

        if getattr(args, "init_weights", None):
            weights_path = resolve_checkpoint(args.init_weights, config)
            weights = load_weights(weights_path, map_location=config.device)
            model.load_state_dict(weights)
            run.logger.info("Initialized model weights for fine-tuning from %s", weights_path)

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

def run_diarize(args, config: PipelineConfig) -> None:
    if getattr(args, "architecture", None):
        config.model.architecture = args.architecture
    if getattr(args, "device", None):
        config.device = args.device

    with RunManager(kind="diarize", config=config, run_id=args.run_id, params=vars(args)) as run:
        run.logger.info(
            "Diarizing %s (asr=%s, backend=%s, speakers=%s)",
            args.audio, args.model_type, args.backend, args.num_speakers or "auto"
        )
        pipeline = DiarizedTranscriber(
            model_type=args.model_type,
            model_path=args.model_path,
            backend=args.backend,
            config=config,
        )
        run.logger.info("Pipeline: %s", pipeline.describe())

        result = pipeline.transcribe(
            args.audio,
            num_speakers=args.num_speakers,
            apply_noise_reduction=args.noise_reduction,
        )
        transcript = format_transcript(result)

        run.write_json("diarization.json", result)
        run.write_text("transcript.txt", transcript)
        run.log_metrics(
            {
                "num_speakers": result["num_speakers"],
                "utterances": len(result["utterances"]),
                "audio_seconds": result["audio_seconds"],
                **result["timing"],
            },
            stage="diarize"
        )
        run.finish(status="completed", summary={
            "audio_path": args.audio,
            "num_speakers": result["num_speakers"],
            "utterances": len(result["utterances"]),
            "speakers": result["speakers"],
            "timing": result["timing"],
        })

        print("\n" + "=" * 60)
        print(f"SPEAKER-ATTRIBUTED TRANSCRIPT ({result['num_speakers']} speakers)")
        print("=" * 60)
        print(transcript)
        print("=" * 60)
        print(f"Saved to: {run.run_dir}")


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

def run_models(args, config: PipelineConfig) -> None:
    """
    Registry management. Deliberately does not open a RunManager: listing
    versions is a read, and it should not litter outputs/runs/ with a directory
    per invocation.
    """
    registry = ModelRegistry(config.paths.output_dir)
    command = getattr(args, "models_command", None) or "list"

    if command == "list":
        rows = registry.summary()
        if not rows:
            print(
                "No published model versions yet.\n"
                "Training publishes automatically; or publish an existing checkpoint with:\n"
                "  python -m src.main models publish --model-path <ckpt> --architecture conformer --wer 0.51"
            )
            return
        print(f"{'':2} {'ARCHITECTURE':<18} {'VERSION':<8} {'WER':>7} {'CER':>7}  {'RUN':<22} PUBLISHED")
        for row in rows:
            marker = "->" if row["current"] else "  "
            wer = row["metrics"].get("wer")
            cer = row["metrics"].get("cer")
            print(
                f"{marker} {row['architecture']:<18} {row['version']:<8} "
                f"{wer if wer is None else f'{wer:.4f}':>7} {cer if cer is None else f'{cer:.4f}':>7}  "
                f"{(row['run_id'] or '-'):<22} {row['published_at'] or '-'}"
            )
        print("\n'->' marks the version served by the app, CLI and evaluation.")
        return

    if command == "show":
        version = (
            registry.get(args.architecture, args.version) if args.version
            else registry.current(args.architecture)
        )
        if version is None:
            raise SystemExit(f"No such version for {args.architecture}.")
        print(json.dumps(version.metadata, indent=2, ensure_ascii=False))
        mismatches = version.incompatibilities(config)
        if mismatches:
            print("\nWARNING - this version does not match the current audio config:")
            for line in mismatches:
                print(f"  - {line}")
            print("Loading it under these settings will degrade transcription quality.")
        return

    if command == "publish":
        published = registry.publish(
            args.model_path,
            architecture=args.architecture,
            metrics={"wer": args.wer, "cer": args.cer},
            run_id=args.source_run_id,
            config=config,
            vocab_size=TextTransform().vocab_size,
            subsampling_factor=subsampling_factor(args.architecture),
            notes=args.notes,
            promote=True if args.promote else None,
        )
        current = registry.current(args.architecture)
        served = current is not None and current.version == published.version
        print(f"Published {published}")
        print("Promoted to current." if served else f"Not promoted; {current} remains current.")
        return

    if command == "promote":
        print(f"Promoted {registry.promote(args.architecture, args.version)} to current.")
        return

    if command == "rollback":
        print(f"Rolled back: {registry.rollback(args.architecture)} is now current.")
        return

    raise SystemExit(f"Unknown models command '{command}'. Try: list, show, publish, promote, rollback.")


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
        "diarize": run_diarize,
        "train": run_train,
        "evaluate": run_evaluate,
        "models": run_models,
        "export": run_export,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return

    handler(args, config)

if __name__ == "__main__":
    main()
