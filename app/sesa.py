#!/usr/bin/env python
"""
SESA — local Akan (Twi) transcription.

Record from the microphone or upload a clip. The page serves the best measured
model, conformer-medium/v003, decoded with beam search and the character
language model (Lagyamfi test WER 0.2887).

    scripts/asr/serve_sesa.sh
    python app/sesa.py
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.asr.inference.transcribe import Transcriber
from src.config import PipelineConfig
from src.utils.run_logger import RunManager

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "outputs/asr/registry/conformer-medium/v003/model.pt"
LANGUAGE_MODEL = REPO_ROOT / "outputs/asr/lm/twi_char.json"

# Short clips from demo_audio/, cut from the same scripture readings the model
# was measured on. The second column is the corpus reference, so a click fills
# the audio and the text it should come out as.
EXAMPLES = [
    [
        str(REPO_ROOT / "demo_audio/clip_0.wav"),
        "na nhyira ne n’animuonyam din daa, na n’animuonyam nhyɛ asase nyinaa ma. Amen ne Amen.",
    ],
    [
        str(REPO_ROOT / "demo_audio/clip_1.wav"),
        "na m’agyenkwa Nyankopɔn ho na me honhom ho asane no;",
    ],
    [
        str(REPO_ROOT / "demo_audio/clip_2.wav"),
        "Na wɔn a wahyira wɔn na ɛbɛtena asase so, na wɔn a wabɔ wɔn dua no, wɔbɛyi wɔn hɔ.",
    ],
    [
        str(REPO_ROOT / "demo_audio/clip_3.wav"),
        "ɔtɛ sɛ ne nsa bɛka mmɔborɔni",
    ],
]

_transcriber: Optional[Transcriber] = None
_run: Optional[RunManager] = None


def _config() -> PipelineConfig:
    config = PipelineConfig()
    config.model.architecture = "conformer-medium"
    config.decoding.decoder = "beam"
    config.decoding.lm_path = str(LANGUAGE_MODEL)
    config.decoding.beam_width = 25
    config.decoding.alpha = 0.5
    config.decoding.beta = 0.5
    return config


def get_transcriber() -> Transcriber:
    """Loads v003 once. The first transcription pays the load; later ones reuse it."""
    global _transcriber
    if _transcriber is None:
        if not WEIGHTS.is_file():
            raise gr.Error(f"Best-model weights are missing: {WEIGHTS}")
        if not LANGUAGE_MODEL.is_file():
            raise gr.Error(f"Character language model is missing: {LANGUAGE_MODEL}")
        _transcriber = Transcriber(
            model_type="conformer-medium",
            model_path=str(WEIGHTS),
            config=_config(),
        )
        decoder = _transcriber.decoder.name
        if decoder != "beam+lm":
            raise gr.Error(
                f"SESA expected beam search with the language model, and got {decoder}."
            )
    return _transcriber


def get_run() -> RunManager:
    global _run
    if _run is None:
        _run = RunManager(
            kind="sesa",
            config=_config(),
            params={"model_path": str(WEIGHTS), "decoder": "beam"},
        )
    return _run


def transcribe(audio_path: Optional[str]) -> Tuple[str, str]:
    """One block of Akan text for the whole clip."""
    if not audio_path:
        raise gr.Error("Record a clip or upload an audio file first.")

    started = time.time()
    try:
        text = get_transcriber().transcribe(audio_path)
    except Exception as exc:
        get_run().logger.exception("Transcription failed for %s", audio_path)
        raise gr.Error(f"Transcription failed: {exc}") from exc

    elapsed = time.time() - started
    get_run().log_metrics(
        {
            "mode": "transcribe",
            "model": "conformer-medium/v003",
            "characters": len(text),
            "words": len(text.split()),
            "seconds": round(elapsed, 2),
        },
        stage="transcribe",
    )
    get_run().logger.info("%s -> %r (%.2fs)", Path(audio_path).name, text, elapsed)

    if not text.strip():
        return "", f"No speech recognised · {elapsed:.1f}s"
    return text, f"conformer-medium/v003 · beam + LM · {elapsed:.1f}s · {len(text.split())} words"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SESA") as demo:
        gr.Markdown(
            """
# SESA

Akan (Twi) speech to text. Record from the microphone or upload a file.
The model is **conformer-medium/v003**, beam search with the character
language model. On the Lagyamfi test set that system scores **0.2887** word
error rate. The first transcription loads the weights; the next ones reuse them.
            """.strip()
        )
        audio = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
        )
        reference = gr.Textbox(
            label="Reference for the sample",
            lines=2,
            interactive=False,
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[audio, reference],
            label="Sample audio",
            examples_per_page=4,
        )
        transcribe_button = gr.Button("Transcribe", variant="primary")
        transcript = gr.Textbox(
            label="Transcript",
            lines=6,
            show_copy_button=True,
            placeholder="The Akan transcript will appear here.",
        )
        status = gr.Markdown()
        transcribe_button.click(
            fn=transcribe,
            inputs=audio,
            outputs=[transcript, status],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    finally:
        if _run is not None:
            _run.finish(status="completed")


if __name__ == "__main__":
    main()
