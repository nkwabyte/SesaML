#!/usr/bin/env python
"""
Gradio front-end for the SesaML Akan speech-to-text models.

Run locally:
    scripts/asr/serve_app.sh
    python app/app.py --share            # public tunnel

Two modes:
  * Transcribe — one block of Akan text for the whole clip.
  * Diarize    — "who spoke when" joined to "what was said", one line per
                 speaker turn, for recordings with more than one voice.

Models are loaded once and reused, so only the first request of each kind pays
the load cost. Every request is appended to the run directory under `outputs/`,
exactly like the CLI commands.
"""

import argparse
import html
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.config import PipelineConfig
from src.asr.diarization import BACKENDS, DiarizationError, DiarizedTranscriber
from src.asr.diarization.backends import resolve_token
from src.asr.inference.transcribe import DEFAULT_WHISPER_REPO, Transcriber
from src.asr.models import ARCHITECTURES
from src.utils.model_registry import ModelRegistry
from src.utils.run_logger import RunManager, latest_checkpoint, load_model_meta

WHISPER_REPO = os.environ.get("MODEL_REPO_ID") or DEFAULT_WHISPER_REPO

# Whisper is offered only when a repository is configured. It is a comparison
# baseline, not this project's model, and a fresh clone should not silently
# download a third-party checkpoint and present its output as ours.
MODEL_CHOICES = ("ctc", "whisper") if WHISPER_REPO else ("ctc",)

# Distinct hues that stay legible on the light and dark Gradio themes alike.
SPEAKER_COLORS = (
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#4d7c0f",
)

DESCRIPTION = f"""
# 🇬🇭 SesaML: Akan Speech-to-Text

Upload an audio clip or record from your microphone to transcribe Akan (Twi) speech.

- **ctc** — the Conformer model trained in this repo, served from the promoted
  version in the model registry (`outputs/asr/registry/`).
{"- **whisper** — comparison baseline from `" + WHISPER_REPO + "`." if WHISPER_REPO else ""}

Use the **Speaker Diarization** tab for recordings with more than one speaker.
"""

config = PipelineConfig()

run = RunManager(kind="app", config=config, params={"whisper_repo": WHISPER_REPO})
registry = ModelRegistry(config.paths.domain_dir)
_transcribers: Dict[str, Transcriber] = {}
_pipelines: Dict[Tuple[str, str], DiarizedTranscriber] = {}


def get_transcriber(model_type: str) -> Transcriber:
    """Loads an ASR backend on first use and reuses it afterwards."""
    if model_type not in _transcribers:
        run.logger.info("Loading %s backend...", model_type)
        started = time.time()
        _transcribers[model_type] = Transcriber(
            model_type="whisper" if model_type == "whisper" else "deepspeech",
            whisper_repo=WHISPER_REPO,
            config=config,
        )
        run.logger.info("Loaded %s in %.1fs", model_type, time.time() - started)
    return _transcribers[model_type]


def get_pipeline(model_type: str, backend: str) -> DiarizedTranscriber:
    """One DiarizedTranscriber per (ASR, diarization) pair; both are expensive to load."""
    key = (model_type, backend)
    if key not in _pipelines:
        run.logger.info("Loading diarization pipeline (asr=%s, backend=%s)...", model_type, backend)
        started = time.time()
        _pipelines[key] = DiarizedTranscriber(
            transcriber=get_transcriber(model_type),
            backend=backend,
            config=config,
            token=resolve_token(),
        )
        run.logger.info("Loaded pipeline in %.1fs", time.time() - started)
    return _pipelines[key]


def model_status(model_type: str) -> str:
    """
    Markdown badge describing which weights are actually in use.

    Names the registry version, not just a path: across a series of training
    iterations "which model is this?" is the question, and a run directory name
    does not answer it.
    """
    if model_type == "whisper":
        return f"**Whisper** — `{WHISPER_REPO}` (downloaded from HuggingFace on first use)"

    version = registry.resolve()
    if version is not None:
        metrics = version.metrics
        scores = " · ".join(
            f"{name.upper()} {metrics[name]:.3f}"
            for name in ("wer", "cer") if metrics.get(name) is not None
        )
        previous = registry.previous(version.architecture)
        fallback = f"<br/>Fallback available: `{previous.version}`" if previous else ""
        warnings = version.incompatibilities(config)
        alert = (
            "<br/>⚠️ **Feature mismatch:** " + "; ".join(warnings)
            if warnings else ""
        )
        return (
            f"**{version.architecture} `{version.version}`** — {scores or 'no metrics'}"
            f"<br/>run `{version.run_id or 'unknown'}`{fallback}{alert}"
        )

    checkpoint = latest_checkpoint(config.paths.domain_dir)
    if checkpoint is None:
        return (
            "⚠️ **No trained checkpoint found.** The CTC model will run with random weights "
            "and produce gibberish. Train one with `scripts/asr/train.sh`, or switch to Whisper."
        )
    meta = load_model_meta(checkpoint)
    arch = (meta and meta.get("architecture")) or "unknown"
    described = ARCHITECTURES.get(arch, arch)
    return f"**{arch}** (unpublished) — {described}<br/>`{checkpoint}`"


def version_choices() -> List[str]:
    """Every published version, newest first, for the demo's model picker."""
    labels = []
    for architecture in registry.architectures():
        current = registry.current(architecture)
        for version in reversed(registry.versions(architecture)):
            wer = version.metrics.get("wer")
            served = current is not None and current.version == version.version
            labels.append(
                f"{architecture}/{version.version}"
                f"{' (current)' if served else ''}"
                f"{'' if wer is None else f' — WER {wer:.3f}'}"
            )
    return labels


def select_version(label: Optional[str]) -> str:
    """
    Switches the served version from the UI.

    The point is the live demo: if a freshly promoted model misbehaves in front
    of an audience, the previous one is one dropdown away rather than a restart
    and a command line.
    """
    if not label:
        return model_status("ctc")

    architecture, _, rest = label.partition("/")
    version = rest.split(" ")[0]
    try:
        registry.promote(architecture, version)
    except Exception as exc:
        raise gr.Error(f"Could not switch version: {exc}") from exc

    # Drop cached models so the next request loads the newly promoted weights.
    _transcribers.clear()
    _pipelines.clear()
    run.logger.info("Switched served model to %s/%s", architecture, version)
    return model_status("ctc")


def transcribe(audio_path: Optional[str], model_type: str, noise_reduction: bool) -> Tuple[str, str]:
    """Plain transcription: one block of text for the whole clip."""
    if not audio_path:
        raise gr.Error("Please upload an audio file or record a clip first.")

    started = time.time()
    try:
        transcript = get_transcriber(model_type).transcribe(
            audio_path, apply_noise_reduction=noise_reduction
        )
    except Exception as exc:
        run.logger.exception("Transcription failed for %s", audio_path)
        raise gr.Error(f"Transcription failed: {exc}") from exc

    elapsed = time.time() - started
    run.log_metrics(
        {
            "mode": "transcribe",
            "model_type": model_type,
            "noise_reduction": noise_reduction,
            "characters": len(transcript),
            "words": len(transcript.split()),
            "seconds": round(elapsed, 2),
        },
        stage="transcribe",
    )
    run.logger.info("[%s] %s -> %r (%.2fs)", model_type, Path(audio_path).name, transcript, elapsed)

    if not transcript.strip():
        return transcript, f"Empty transcript — {model_type} produced no output in {elapsed:.2f}s."
    return transcript, f"{model_type} · {elapsed:.2f}s · {len(transcript.split())} words"


def speaker_palette(speakers: List[str]) -> Dict[str, str]:
    return {speaker: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, speaker in enumerate(sorted(speakers))}


def render_conversation(result: Dict) -> str:
    """Speaker-coloured transcript; the thing an audience actually reads."""
    utterances = result.get("utterances", [])
    if not utterances:
        return "_No speech was transcribed. Try the other diarization backend, or check the audio._"

    colors = speaker_palette(result.get("speakers", []))
    rows = []
    for utterance in utterances:
        color = colors.get(utterance["speaker"], "#475569")
        rows.append(
            f'<div style="margin:0 0 10px 0;padding:8px 12px;border-left:4px solid {color};'
            f'background:rgba(127,127,127,0.07);border-radius:0 6px 6px 0">'
            f'<div style="font-size:12px;color:{color};font-weight:600">'
            f'{html.escape(utterance["speaker"])}'
            f'<span style="opacity:.65;font-weight:400"> · {utterance["start"]:.1f}s – {utterance["end"]:.1f}s</span>'
            f"</div>"
            f'<div style="font-size:15px;line-height:1.5">{html.escape(utterance["transcript"])}</div>'
            f"</div>"
        )
    return "".join(rows)


def render_timeline(result: Dict) -> str:
    """A proportional bar per speaker turn, so turn-taking is visible at a glance."""
    utterances = result.get("utterances", [])
    total = result.get("audio_seconds") or 0
    if not utterances or not total:
        return ""

    colors = speaker_palette(result.get("speakers", []))
    blocks = []
    for utterance in utterances:
        left = 100 * utterance["start"] / total
        width = max(0.4, 100 * (utterance["end"] - utterance["start"]) / total)
        color = colors.get(utterance["speaker"], "#475569")
        blocks.append(
            f'<div title="{html.escape(utterance["speaker"])} {utterance["start"]:.1f}s–{utterance["end"]:.1f}s" '
            f'style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;top:0;bottom:0;'
            f'background:{color};border-radius:2px"></div>'
        )

    legend = " ".join(
        f'<span style="margin-right:14px;font-size:12px">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{color};'
        f'border-radius:2px;margin-right:5px"></span>{html.escape(speaker)}</span>'
        for speaker, color in colors.items()
    )
    return (
        f'<div style="margin:4px 0 8px 0">{legend}</div>'
        f'<div style="position:relative;height:26px;background:rgba(127,127,127,.12);'
        f'border-radius:4px;overflow:hidden">{"".join(blocks)}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;opacity:.6;margin-top:3px">'
        f"<span>0.0s</span><span>{total:.1f}s</span></div>"
    )


def diarize(
    audio_path: Optional[str],
    model_type: str,
    backend: str,
    num_speakers: int,
    noise_reduction: bool,
) -> Tuple[str, str, str, Dict]:
    """Diarization callback: coloured transcript, timeline, summary line, raw JSON."""
    if not audio_path:
        raise gr.Error("Please upload an audio file or record a clip first.")

    try:
        pipeline = get_pipeline(model_type, backend)
    except DiarizationError as exc:
        # The gated-model case lands here; the message names the exact repository
        # to accept, so it is shown verbatim rather than summarised away.
        raise gr.Error(str(exc)) from exc

    try:
        result = pipeline.transcribe(
            audio_path,
            num_speakers=num_speakers or None,
            apply_noise_reduction=noise_reduction,
        )
    except Exception as exc:
        run.logger.exception("Diarization failed for %s", audio_path)
        raise gr.Error(f"Diarization failed: {exc}") from exc

    timing = result["timing"]
    run.log_metrics(
        {
            "mode": "diarize",
            "model_type": model_type,
            "backend": backend,
            "num_speakers": result["num_speakers"],
            "utterances": len(result["utterances"]),
            "audio_seconds": result["audio_seconds"],
            "seconds": timing["total_sec"],
        },
        stage="diarize",
    )
    run.logger.info(
        "[diarize/%s+%s] %s -> %d speakers, %d utterances (%.2fs)",
        model_type, backend, Path(audio_path).name,
        result["num_speakers"], len(result["utterances"]), timing["total_sec"],
    )

    summary = (
        f"**{result['num_speakers']} speaker(s)** · {len(result['utterances'])} turns · "
        f"{result['audio_seconds']:.1f}s audio in {timing['total_sec']:.1f}s "
        f"({timing['realtime_factor']}× realtime) — "
        f"diarization {timing['diarization_sec']:.1f}s, ASR {timing['asr_sec']:.1f}s"
    )
    return render_conversation(result), render_timeline(result), summary, result


def build_demo() -> gr.Blocks:
    # Ask the registry, not the filesystem. Checking outputs/asr/checkpoints/ for a
    # loose file meant that pruning superseded runs - which leaves the published
    # version untouched - made the app fall back to Whisper despite a perfectly
    # good trained model being published and served.
    has_trained_model = registry.resolve() is not None or latest_checkpoint(config.paths.domain_dir)
    default_model = "ctc" if has_trained_model or "whisper" not in MODEL_CHOICES else "whisper"
    samples = sorted(str(p) for p in Path("demo_audio").glob("*.wav")) if Path("demo_audio").is_dir() else []

    with gr.Blocks(title="SesaML — Akan Speech-to-Text", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.Tab("Transcribe"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"], type="filepath", label="Akan audio"
                        )
                        model_type = gr.Radio(
                            choices=list(MODEL_CHOICES), value=default_model, label="Model"
                        )
                        noise_reduction = gr.Checkbox(
                            value=False,
                            label="Apply noise reduction",
                            info="Spectral gate pre-processing; helps on noisy recordings.",
                        )
                        submit = gr.Button("Transcribe", variant="primary")
                    with gr.Column(scale=1):
                        status = gr.Markdown(model_status(default_model))
                        with gr.Accordion("Model version", open=False):
                            gr.Markdown(
                                "Every training run is archived. Switching here changes the "
                                "model served everywhere — the live fallback if a new one "
                                "misbehaves."
                            )
                            version_picker = gr.Dropdown(
                                choices=version_choices(),
                                value=next(iter(version_choices()), None),
                                label="Served version",
                                interactive=True,
                            )
                        transcript_output = gr.Textbox(
                            label="Transcript", lines=8, show_copy_button=True,
                            placeholder="The Akan transcription will appear here.",
                        )
                        detail_output = gr.Markdown()

                if samples:
                    gr.Examples(examples=[[s] for s in samples], inputs=[audio_input], label="Sample clips")

                model_type.change(fn=model_status, inputs=model_type, outputs=status)
                version_picker.change(fn=select_version, inputs=version_picker, outputs=status)
                submit.click(
                    fn=transcribe,
                    inputs=[audio_input, model_type, noise_reduction],
                    outputs=[transcript_output, detail_output],
                )

            with gr.Tab("Speaker Diarization"):
                gr.Markdown(
                    "Splits a recording into speaker turns, then transcribes each turn — "
                    "**who spoke when** joined to **what was said**."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        d_audio = gr.Audio(
                            sources=["upload", "microphone"], type="filepath",
                            label="Conversation audio (2+ speakers)",
                        )
                        d_model = gr.Radio(
                            choices=list(MODEL_CHOICES), value=default_model, label="ASR model"
                        )
                        d_backend = gr.Radio(
                            choices=list(BACKENDS), value="ecapa", label="Diarization backend",
                            info="pyannote is most accurate but needs accepted HuggingFace licences; "
                                 "ecapa needs none; spectral needs no downloads at all.",
                        )
                        d_speakers = gr.Slider(
                            minimum=0, maximum=8, step=1, value=0, label="Number of speakers",
                            info="0 = detect automatically. Setting the true count makes the "
                                 "ecapa and spectral backends markedly more reliable.",
                        )
                        d_noise = gr.Checkbox(value=False, label="Apply noise reduction")
                        d_submit = gr.Button("Diarize & Transcribe", variant="primary")
                    with gr.Column(scale=2):
                        d_summary = gr.Markdown()
                        d_timeline = gr.HTML()
                        d_conversation = gr.HTML()
                        with gr.Accordion("Raw JSON", open=False):
                            d_json = gr.JSON()

                if samples:
                    conversation_samples = [[s] for s in samples if "conversation" in s] or [[s] for s in samples]
                    gr.Examples(examples=conversation_samples, inputs=[d_audio], label="Sample conversations")

                d_submit.click(
                    fn=diarize,
                    inputs=[d_audio, d_model, d_backend, d_speakers, d_noise],
                    outputs=[d_conversation, d_timeline, d_summary, d_json],
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
        run.finish(status="completed", summary={"requests": len(run.metric_rows)})


if __name__ == "__main__":
    main()
