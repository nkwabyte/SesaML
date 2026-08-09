#!/usr/bin/env python
"""
Gradio front-end for the SesaML Akan speech-to-text models.

Run locally:
    scripts/serve_app.sh
    python app/app.py --share            # public tunnel

The app keeps one loaded model per backend, so only the first transcription
pays the load cost. Every transcription is appended to the run directory under
`outputs/`, exactly like the CLI commands.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.config import PipelineConfig
from src.inference.transcribe import DEFAULT_WHISPER_REPO, Transcriber
from src.utils.run_logger import RunManager, latest_checkpoint

MODEL_CHOICES = ("deepspeech", "whisper")

DESCRIPTION = """
# 🇬🇭 SesaML — Akan Speech-to-Text

Upload an audio clip or record from your microphone to transcribe Akan (Twi) speech.

- **deepspeech** — the DeepSpeech2 CTC model trained in this repo, loaded from the newest checkpoint in `outputs/checkpoints/`.
- **whisper** — a fine-tuned Whisper model pulled from the HuggingFace Hub.
"""

config = PipelineConfig()
# Whisper repo can be overridden per-deployment (HuggingFace Spaces sets this).
WHISPER_REPO = os.environ.get("MODEL_REPO_ID") or DEFAULT_WHISPER_REPO

run = RunManager(kind="app", config=config, params={"whisper_repo": WHISPER_REPO})
_transcribers: Dict[Tuple[str, str], Transcriber] = {}


def get_transcriber(model_type: str) -> Transcriber:
    """Loads a backend on first use and reuses it afterwards."""
    key = (model_type, WHISPER_REPO)
    if key not in _transcribers:
        run.logger.info("Loading %s backend...", model_type)
        started = time.time()
        _transcribers[key] = Transcriber(
            model_type=model_type,
            whisper_repo=WHISPER_REPO,
            config=config
        )
        run.logger.info("Loaded %s in %.1fs", model_type, time.time() - started)
    return _transcribers[key]


def model_status(model_type: str) -> str:
    """Markdown badge describing which weights are actually in use."""
    if model_type == "whisper":
        return f"**Whisper** — `{WHISPER_REPO}` (downloaded from HuggingFace on first use)"

    checkpoint = latest_checkpoint(config.paths.output_dir)
    if checkpoint is None:
        return (
            "⚠️ **No trained checkpoint found.** DeepSpeech will run with random weights "
            "and produce gibberish. Train one first with `scripts/train.sh`, or switch to Whisper."
        )
    return f"**DeepSpeech** — `{checkpoint}`"


def transcribe(audio_path: Optional[str], model_type: str, noise_reduction: bool) -> Tuple[str, str]:
    """Gradio callback: returns the transcript and a short run detail line."""
    if not audio_path:
        raise gr.Error("Please upload an audio file or record a clip first.")

    started = time.time()
    try:
        transcriber = get_transcriber(model_type)
        transcript = transcriber.transcribe(audio_path, apply_noise_reduction=noise_reduction)
    except Exception as exc:
        run.logger.exception("Transcription failed for %s", audio_path)
        raise gr.Error(f"Transcription failed: {exc}") from exc

    elapsed = time.time() - started
    run.log_metrics(
        {
            "model_type": model_type,
            "noise_reduction": noise_reduction,
            "characters": len(transcript),
            "words": len(transcript.split()),
            "seconds": round(elapsed, 2),
        },
        stage="transcribe"
    )
    run.logger.info("[%s] %s -> %r (%.2fs)", model_type, Path(audio_path).name, transcript, elapsed)

    if not transcript.strip():
        detail = f"Empty transcript — {model_type} produced no output in {elapsed:.2f}s."
    else:
        detail = f"{model_type} · {elapsed:.2f}s · {len(transcript.split())} words"
    return transcript, detail


def build_demo() -> gr.Blocks:
    # Default to Whisper when nothing has been trained yet, so a fresh clone
    # produces real transcriptions instead of noise from random weights.
    default_model = "deepspeech" if latest_checkpoint(config.paths.output_dir) else "whisper"

    with gr.Blocks(title="SesaML — Akan Speech-to-Text", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Akan audio"
                )
                model_type = gr.Radio(
                    choices=list(MODEL_CHOICES),
                    value=default_model,
                    label="Model"
                )
                noise_reduction = gr.Checkbox(
                    value=False,
                    label="Apply noise reduction",
                    info="Spectral gate pre-processing; helps on noisy recordings."
                )
                submit = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                status = gr.Markdown(model_status(default_model))
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=8,
                    show_copy_button=True,
                    placeholder="The Akan transcription will appear here."
                )
                detail_output = gr.Markdown()

        model_type.change(fn=model_status, inputs=model_type, outputs=status)
        submit.click(
            fn=transcribe,
            inputs=[audio_input, model_type, noise_reduction],
            outputs=[transcript_output, detail_output]
        )

        gr.Markdown(f"Session log: `{run.log_path}`")

    return demo


# Module-level so HuggingFace Spaces (which imports this file) finds the app.
demo = build_demo()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), help="Interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)), help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio tunnel")
    args = parser.parse_args()

    run.logger.info("Starting Gradio app on %s:%d (share=%s)", args.host, args.port, args.share)
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    finally:
        run.finish(status="completed", summary={"transcriptions": len(run.metric_rows)})


if __name__ == "__main__":
    main()
