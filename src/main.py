import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.text_transform import TextTransform
from src.data.audio_transforms import get_train_audio_transforms, get_valid_audio_transforms
from src.data.dataset import AkanAudioDataset, data_processing
from src.inference.export import EXPORT_FORMATS, export_model
from src.inference.transcribe import load_deepspeech_model, transcribe_audio
from src.models.deepspeech import SpeechRecognitionModel
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.run_logger import RunManager, resolve_checkpoint

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SesaML - Akan Audio Speech-to-Text Transcriber & Trainer")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Transcribe sub-command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe Akan audio file to text")
    transcribe_parser.add_argument("--audio", "-a", required=True, help="Path to input audio file (WAV/MP3/FLAC)")
    transcribe_parser.add_argument("--model-type", choices=["deepspeech", "whisper"], default="deepspeech", help="Model type to use for transcription")
    transcribe_parser.add_argument("--model-path", default=None, help="Path to DeepSpeech model checkpoint (.pt / .pth)")
    transcribe_parser.add_argument("--whisper-repo", default="CiBeDL/twi_trained_whisper", help="HuggingFace repository ID for Whisper model")
    transcribe_parser.add_argument("--noise-reduction", action="store_true", help="Apply spectral gate noise reduction before transcription")
    transcribe_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train DeepSpeech CTC model on Akan dataset")
    train_parser.add_argument("--csv-path", default=None, help="Path to dataset CSV file containing 'audio_path' and 'text' columns")
    train_parser.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name (e.g. ghanaopendata/twi-speech-text-multispeaker-16k)")
    train_parser.add_argument("--split", default="train", help="HuggingFace dataset split used for training")
    train_parser.add_argument("--val-split", default=None, help="HuggingFace split used for validation (enables per-epoch WER/CER logging)")
    train_parser.add_argument("--val-csv-path", default=None, help="Path to validation CSV file (enables per-epoch WER/CER logging)")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Evaluate sub-command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model WER and CER")
    eval_parser.add_argument("--csv-path", default=None, help="Path to evaluation dataset CSV file")
    eval_parser.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name (e.g. ghanaopendata/twi-speech-text-multispeaker-16k)")
    eval_parser.add_argument("--split", default="train", help="HuggingFace dataset split to evaluate on")
    eval_parser.add_argument("--model-path", default=None, help="Path to model checkpoint")
    eval_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    # Export sub-command
    export_parser = subparsers.add_parser("export", help="Export a trained checkpoint to a deployable model file")
    export_parser.add_argument("--model-path", default=None, help="Path to trained checkpoint (state dict)")
    export_parser.add_argument("--format", dest="export_format", choices=list(EXPORT_FORMATS), default="torchscript", help="Export format: torchscript (.pt), state_dict (.pth) or executorch (.pte)")
    export_parser.add_argument("--basename", default="speech_recognition_model", help="Base filename for the exported artifact")
    export_parser.add_argument("--run-id", default=None, help="Explicit run id for the outputs/ directory")

    return parser

def build_dataset(config: PipelineConfig, hf_dataset, split, csv_path):
    """Builds either a HuggingFace-backed or CSV-backed dataset."""
    if hf_dataset:
        from src.data.hf_dataset import HuggingFaceAkanDataset
        return HuggingFaceAkanDataset(dataset_name=hf_dataset, split=split, sample_rate=config.audio.sample_rate)

    csv_file = csv_path or os.path.join(config.paths.corpus_dir, "verified_data.csv")
    return AkanAudioDataset(csv_file=csv_file, sample_rate=config.audio.sample_rate)

def build_model(config: PipelineConfig, n_class: int) -> SpeechRecognitionModel:
    return SpeechRecognitionModel(
        n_cnn_layers=config.model.n_cnn_layers,
        n_rnn_layers=config.model.n_rnn_layers,
        rnn_dim=config.model.rnn_dim,
        n_class=n_class,
        n_feats=config.model.n_feats,
        stride=config.model.stride,
        dropout=config.model.dropout
    )

def run_transcribe(args, config: PipelineConfig) -> None:
    with RunManager(kind="transcribe", config=config, run_id=args.run_id, params=vars(args)) as run:
        run.logger.info("Transcribing %s using %s", args.audio, args.model_type)
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
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    run = RunManager(kind="train", config=config, run_id=args.run_id, params=vars(args))
    try:
        text_transform = TextTransform()
        train_transforms = get_train_audio_transforms(config.audio.sample_rate, config.audio.n_mels)
        valid_transforms = get_valid_audio_transforms(config.audio.sample_rate, config.audio.n_mels)

        dataset = build_dataset(config, args.hf_dataset, args.split, args.csv_path)
        run.logger.info("Training samples: %d", len(dataset))

        def train_collate(batch):
            return data_processing(batch, text_transform, train_transforms, stride=config.model.stride)

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=train_collate
        )

        val_loader = None
        if args.val_split or args.val_csv_path:
            val_dataset = build_dataset(
                config,
                args.hf_dataset if args.val_split else None,
                args.val_split,
                args.val_csv_path
            )
            run.logger.info("Validation samples: %d", len(val_dataset))

            def valid_collate(batch):
                return data_processing(batch, text_transform, valid_transforms, stride=config.model.stride)

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                collate_fn=valid_collate
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
            **trainer.summary,
        })
        print(f"\nCheckpoints: {run.checkpoint_dir}")
        print(f"Logs & metrics: {run.run_dir}")
    except BaseException as exc:
        run.logger.exception("Training run failed")
        run.finish(status="failed", summary={"error": f"{type(exc).__name__}: {exc}"})
        raise

def run_evaluate(args, config: PipelineConfig) -> None:
    with RunManager(kind="evaluate", config=config, run_id=args.run_id, params=vars(args)) as run:
        text_transform = TextTransform()
        valid_transforms = get_valid_audio_transforms(config.audio.sample_rate, config.audio.n_mels)

        dataset = build_dataset(config, args.hf_dataset, args.split, args.csv_path)
        run.logger.info("Evaluation samples: %d", len(dataset))

        def collate_fn(batch):
            return data_processing(batch, text_transform, valid_transforms, stride=config.model.stride)

        val_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn
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
        run.write_json("evaluation.json", {"model_path": model_path, **metrics})
        run.finish(status="completed", summary={"model_path": model_path, **metrics})

        print("\n" + "=" * 40)
        print("EVALUATION RESULTS:")
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"Average WER:     {metrics['wer']:.4f}")
        print(f"Average CER:     {metrics['cer']:.4f}")
        print("=" * 40)
        print(f"Saved to: {run.run_dir}")

def run_export(args, config: PipelineConfig) -> None:
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
