# Speaker Diarization Architecture & Implementation Guide

This document specifies how to integrate **Speaker Diarization** ("Who Spoke When") with the **SesaML ASR pipeline** ("What Was Said") to produce speaker-attributed Akan transcripts.

---

## 1. System Architecture Overview

ASR models (DeepSpeech2, Conformer, Whisper) transcribe audio into text but do not track individual speaker identities. A production diarization pipeline combines a **Speaker Embedding & Segmentation Model** (e.g. `pyannote.audio` or `speechbrain`) with the **SesaML Transcriber**.

```
                         ┌─────────────────────────────────────────┐
                         │              Audio Input                │
                         └────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │   1. Diarization Pipeline     │                 │   2. SesaML ASR Pipeline      │
     │   (pyannote.audio / ECAPA)    │                 │   (Conformer / Whisper)       │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     ▼                                                 ▼
        [Speaker Timestamps]                              [Word/Segment Timestamps]
        • 00:00 - 00:04: Speaker 0                       • 00:00 - 00:04: "Medaase..."
        • 00:04 - 00:09: Speaker 1                       • 00:04 - 00:09: "Wo ho te sen?"
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   3. Alignment & Merger     │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   Speaker Labeled Output    │
                               │   [Speaker 0]: Medaase...   │
                               │   [Speaker 1]: Wo ho te sen?│
                               └─────────────────────────────┘
```

---

## 2. Option A: Gradio Web App Integration (`app/app.py`)

In the Gradio interface, a checkbox `"Enable Speaker Diarization"` delegates audio processing to `pyannote.audio` before invoking `Transcriber.transcribe()`.

### Implementation Outline (`app/app.py`)

```python
import torch
from pyannote.audio import Pipeline
from src.inference.transcribe import Transcriber

class DiarizedTranscriber:
    def __init__(self, model_path: str, hf_token: str = None):
        self.transcriber = Transcriber(model_path=model_path)
        # Load pyannote speaker diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            self.diarization_pipeline.to(torch.device("mps"))

    def transcribe_with_speakers(self, audio_path: str) -> str:
        # Step 1: Run Diarization
        diarization = self.diarization_pipeline(audio_path)
        
        results = []
        # Step 2: Slice audio per speaker turn & transcribe with SesaML
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Transcribe segment (turn.start -> turn.end)
            segment_text = self.transcriber.transcribe_segment(
                audio_path, start_sec=turn.start, end_sec=turn.end
            )
            results.append(f"[{speaker}] ({turn.start:.1f}s - {turn.end:.1f}s): {segment_text}")
            
        return "\n".join(results)
```

---

## 3. Option B: Production Django REST API Integration

For enterprise backend deployments, long audio files are processed asynchronously via a **Django REST Framework (DRF)** API backed by **Celery** and **Redis**.

### System Architecture

- **Django Web Server**: Handles API authentication, upload endpoints, and transcript storage.
- **Celery Worker Pool**: Processes audio diarization and GPU inference in background tasks.
- **Redis / PostgreSQL**: Stores task status, speaker turns, and structured transcript JSON.

### Database Schema (`models.py`)

```python
from django.db import models
import uuid

class AudioTranscriptionJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.FileField(upload_to="audio_jobs/")
    status = models.CharField(
        max_length=20, 
        choices=[("PENDING", "Pending"), ("PROCESSING", "Processing"), ("COMPLETED", "Completed"), ("FAILED", "Failed")],
        default="PENDING"
    )
    architecture = models.CharField(max_length=50, default="conformer")
    created_at = models.DateTimeField(auto_now_add=True)

class SpeakerUtterance(models.Model):
    job = models.ForeignKey(AudioTranscriptionJob, related_name="utterances", on_delete=CASCADE)
    speaker_label = models.CharField(max_length=50) # e.g. "SPEAKER_00"
    start_time = models.FloatField()
    end_time = models.FloatField()
    transcript = models.TextField()
```

### API Endpoint & Celery Task (`tasks.py` & `views.py`)

#### 1. REST Endpoint (`views.py`)
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioTranscriptionJob
from .tasks import process_diarized_transcription

class TranscribeAudioView(APIView):
    def post(self, request):
        file_obj = request.FILES.get("audio")
        arch = request.data.get("architecture", "conformer")
        
        job = AudioTranscriptionJob.objects.create(audio_file=file_obj, architecture=arch)
        
        # Dispatch background Celery worker task
        process_diarized_transcription.delay(str(job.id))
        
        return Response({"job_id": str(job.id), "status": "PENDING"}, status=status.HTTP_202_ACCEPTED)
```

#### 2. Celery Worker Task (`tasks.py`)
```python
from celery import shared_task
from pyannote.audio import Pipeline
import torchaudio
from .models import AudioTranscriptionJob, SpeakerUtterance
from src.inference.transcribe import Transcriber

@shared_task
def process_diarized_transcription(job_id: str):
    job = AudioTranscriptionJob.objects.get(id=job_id)
    job.status = "PROCESSING"
    job.save()
    
    try:
        # Load Diarization & SesaML Transcriber
        diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        transcriber = Transcriber(architecture=job.architecture)
        
        diarization = diarizer(job.audio_file.path)
        waveform, sr = torchaudio.load(job.audio_file.path)
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Extract slice waveform
            start_frame = int(turn.start * sr)
            end_frame = int(turn.end * sr)
            segment_wave = waveform[:, start_frame:end_frame]
            
            text = transcriber.transcribe_waveform(segment_wave, sample_rate=sr)
            
            SpeakerUtterance.objects.create(
                job=job,
                speaker_label=speaker,
                start_time=turn.start,
                end_time=turn.end,
                transcript=text
            )
            
        job.status = "COMPLETED"
        job.save()
    except Exception as e:
        job.status = "FAILED"
        job.save()
        raise e
```

#### 3. Structured JSON Response Format
```json
{
  "job_id": "8f3b2c1a-4e5d-6f7a-8b9c-0d1e2f3a4b5c",
  "status": "COMPLETED",
  "speakers_count": 2,
  "utterances": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.5,
      "transcript": "Medaase wo nkyeɛsoɔ no ho."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.8,
      "end": 7.2,
      "transcript": "Wo ho te sen ɛnnɛ anopa?"
    }
  ]
}
```
