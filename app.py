import os
import gradio as gr
import torch
from transformers import pipeline

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID")  # set in Space variables
if not MODEL_REPO_ID:
    raise RuntimeError("Missing MODEL_REPO_ID space variable")

# HF_TOKEN is a Space secret; huggingface_hub/transformers will pick it up automatically.
# (No need to hardcode token in code.)
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_REPO_ID,
    tokenizer=MODEL_REPO_ID,
    feature_extractor=MODEL_REPO_ID,
    device=0 if torch.cuda.is_available() else -1,
)

def transcribe(audio_path):
    if audio_path is None:
        return ""
    out = asr(audio_path)
    return out["text"]

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload audio"),
    outputs=gr.Textbox(label="Transcript"),
    title="Akan Whisper ASR",
)

if __name__ == "__main__":
    demo.launch()