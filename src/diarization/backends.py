"""
Diarization backends, in descending order of quality and ascending order of
how likely they are to actually run on a given machine.

* `pyannote`  - the neural pipeline the literature reports numbers for. Best,
                but every model it needs is gated behind a HuggingFace licence
                acceptance, so it fails closed on a fresh clone.
* `ecapa`     - SpeechBrain's ECAPA-TDNN speaker embeddings (public, not gated)
                over our own VAD, clustered agglomeratively. Clearly weaker than
                pyannote on overlapped speech, entirely adequate for a meeting
                or interview recording.
* `spectral`  - MFCC statistics as the embedding. No downloads, no extra
                dependencies, works offline on any machine. Weakest of the
                three, and the reason a demo never hard-fails.

They share one interface - `diarize(waveform, sample_rate) -> [SpeakerTurn]` -
so the transcription side never learns which one ran.
"""

import os
import re
from typing import List, Optional, Sequence

import torch

from .segmentation import detect_speech, window_regions
from .turns import SpeakerTurn, clean_turns, relabel_by_first_appearance

BACKENDS = ("pyannote", "ecapa", "spectral")
DEFAULT_BACKEND = "ecapa"

# pyannote.audio 4 renamed the community pipeline; 3.x still serves the old
# name. Tried in order, newest first.
PYANNOTE_PIPELINES = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
)
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Matches "Access to model <repo> is restricted" and the resolve URL in a 403.
_GATED_PATTERNS = (
    re.compile(r"Access to model ([\w\-.]+/[\w\-.]+) is restricted"),
    re.compile(r"huggingface\.co/([\w\-.]+/[\w\-.]+)/resolve/"),
)


class DiarizationError(RuntimeError):
    """Raised when a backend cannot run at all, with what to do about it."""


def _gated_repos(exc: BaseException) -> List[str]:
    """
    Pulls the repository ids out of a HuggingFace 403, including from the cause.

    A gated-repo failure names the blocked repository in its message, and that
    is rarely the pipeline the caller asked for - it is one of the models the
    pipeline pulls in. Reporting the wrong one sends people to accept conditions
    they have already accepted.
    """
    found: List[str] = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and len(found) < 8:
        text = str(current)
        for pattern in _GATED_PATTERNS:
            for repo in pattern.findall(text):
                if repo not in seen:
                    seen.add(repo)
                    found.append(repo)
        current = current.__cause__ or current.__context__
    return found


def resolve_token(token: Optional[str] = None) -> Optional[str]:
    """HuggingFace token from the argument, the environment, or the project .env."""
    if token:
        return token
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]

    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition("=")
                if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                    return value.strip().strip('"').strip("'") or None
    return None


class BaseDiarizer:
    """Common post-processing so every backend returns comparable turns."""

    name = "base"

    def __init__(self, max_gap: float = 0.5, min_duration: float = 0.35):
        self.max_gap = max_gap
        self.min_duration = min_duration

    def diarize(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        num_speakers: Optional[int] = None
    ) -> List[SpeakerTurn]:
        raw = self._diarize(waveform, sample_rate, num_speakers)
        cleaned = clean_turns(raw, max_gap=self.max_gap, min_duration=self.min_duration)
        return relabel_by_first_appearance(cleaned)

    def _diarize(self, waveform, sample_rate, num_speakers) -> Sequence[SpeakerTurn]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name}


class PyannoteDiarizer(BaseDiarizer):
    """
    The `pyannote/speaker-diarization-3.1` pipeline.

    Handles overlapped speech and estimates the speaker count itself, which the
    embedding-clustering backends do far more crudely.
    """

    name = "pyannote"

    def __init__(
        self,
        token: Optional[str] = None,
        device: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed. `pip install pyannote.audio`, or "
                "select the 'ecapa' backend, which needs no gated model."
            ) from exc

        resolved = resolve_token(token)
        self.pipeline = None
        errors = []

        # pyannote.audio 4 serves the 3.1 pipeline out of a renamed repository,
        # so the name that works depends on the installed version. Trying both
        # keeps this working across the upgrade in either direction.
        for name in (pipeline_name,) if pipeline_name else PYANNOTE_PIPELINES:
            try:
                self.pipeline = Pipeline.from_pretrained(name, token=resolved)
                self.pipeline_name = name
                break
            except Exception as exc:
                errors.append((name, exc))

        if self.pipeline is None:
            # Name the repository the hub actually refused. The pipeline pulls
            # several gated models, and accepting the conditions on the one in
            # the URL is not enough if a dependency is still locked - so the
            # message has to point at the blocker rather than at a guess.
            blocked = sorted({repo for _, exc in errors for repo in _gated_repos(exc)})
            listed = "\n  ".join(f"https://hf.co/{repo}" for repo in blocked) or "\n  ".join(
                f"https://hf.co/{name}" for name, _ in errors
            )
            raise DiarizationError(
                "pyannote could not load its models. Accept the conditions, signed in as the "
                f"owner of HF_TOKEN, at:\n  {listed}\n"
                "Each pipeline pulls several gated repositories and every one needs accepting. "
                "The 'ecapa' backend needs no licence at all and is the default. "
                f"Last error: {type(errors[-1][1]).__name__}: {str(errors[-1][1])[:160]}"
            ) from errors[-1][1]

        if device:
            self.pipeline.to(torch.device(device))

    def _diarize(self, waveform, sample_rate, num_speakers) -> Sequence[SpeakerTurn]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        options = {"num_speakers": num_speakers} if num_speakers else {}
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, **options)

        # pyannote 3 returns an Annotation directly; 4 wraps it in a DiarizeOutput
        # alongside the embeddings and an overlap-free variant. Overlaps are kept
        # here and resolved by this package's own post-processing, so that every
        # backend goes through the same rules.
        annotation = getattr(output, "speaker_diarization", output)

        return [
            SpeakerTurn(str(speaker), float(segment.start), float(segment.end))
            for segment, _, speaker in annotation.itertracks(yield_label=True)
        ]


class EmbeddingClusteringDiarizer(BaseDiarizer):
    """
    VAD, then one speaker embedding per window, then agglomerative clustering.

    This is the classic pre-neural pipeline. It cannot represent two people
    talking at once - every instant is assigned exactly one speaker - which is
    the main quality gap against pyannote.
    """

    name = "embedding"

    def __init__(
        self,
        window: float = 1.5,
        hop: float = 0.75,
        distance_threshold: float = 0.9,
        max_speakers: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window = window
        self.hop = hop
        self.distance_threshold = distance_threshold
        self.max_speakers = max_speakers

    def embed(self, segments: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise NotImplementedError

    def _diarize(self, waveform, sample_rate, num_speakers) -> Sequence[SpeakerTurn]:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        regions = detect_speech(waveform, sample_rate)
        windows = window_regions(regions, window=self.window, hop=self.hop)
        if not windows:
            return []

        # One window cannot be clustered, and two speakers cannot be told apart
        # from a single embedding - so a short clip is one speaker by definition.
        if len(windows) == 1:
            return [SpeakerTurn("SPEAKER_00", windows[0][0], windows[0][1])]

        chunks = [
            waveform[int(start * sample_rate):int(end * sample_rate)]
            for start, end in windows
        ]
        embeddings = self.embed(chunks, sample_rate)
        labels = self._cluster(embeddings, num_speakers)

        return [
            SpeakerTurn(f"cluster_{label}", start, end)
            for (start, end), label in zip(windows, labels)
        ]

    def _cluster(self, embeddings: torch.Tensor, num_speakers: Optional[int]) -> List[int]:
        """
        Agglomerative clustering on cosine distance.

        With `num_speakers` unknown the cut is made on a distance threshold
        rather than a fixed count, so a monologue stays one speaker instead of
        being split to satisfy an assumed two.

        Deciding the speaker count from embeddings alone is the weak point of
        this whole approach. Measured on a 28s Akan clip assembled from six
        different corpus recordings, thresholds of 0.6 and 0.75 found six and
        five speakers while pyannote found three; 0.9 also finds three. Hence
        the default - but it was fitted on one recording, so pass `num_speakers`
        whenever it is known, and prefer the pyannote backend when it is not.
        """
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        features = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()

        if num_speakers and num_speakers > 1:
            model = AgglomerativeClustering(n_clusters=min(num_speakers, len(features)), metric="cosine", linkage="average")
        elif num_speakers == 1:
            return [0] * len(features)
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            )

        labels = model.fit_predict(features)

        # A runaway threshold can shatter the recording into dozens of speakers;
        # keep the largest clusters and fold the rest into the nearest of them.
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > self.max_speakers:
            keep = set(unique[np.argsort(-counts)][: self.max_speakers])
            centroids = {k: features[labels == k].mean(axis=0) for k in keep}
            labels = [
                label if label in keep
                else max(centroids, key=lambda k: float(centroids[k] @ features[i]))
                for i, label in enumerate(labels)
            ]
        return [int(label) for label in labels]

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "window_sec": self.window,
            "hop_sec": self.hop,
            "distance_threshold": self.distance_threshold,
        }


class EcapaDiarizer(EmbeddingClusteringDiarizer):
    """SpeechBrain ECAPA-TDNN embeddings - public weights, no licence to accept."""

    name = "ecapa"

    def __init__(self, device: Optional[str] = None, savedir: str = "outputs/models/ecapa", **kwargs):
        super().__init__(**kwargs)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise DiarizationError(
                "speechbrain is not installed. `pip install speechbrain`, or select the "
                "'spectral' backend, which needs no downloads at all."
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL,
            savedir=savedir,
            run_opts={"device": self.device},
        )

    def embed(self, segments, sample_rate: int) -> torch.Tensor:
        # ECAPA is trained at 16 kHz, which is also this project's sample rate,
        # so no resampling is needed on the pipeline's own audio.
        padded = torch.nn.utils.rnn.pad_sequence(
            [chunk.reshape(-1) for chunk in segments], batch_first=True
        ).to(self.device)
        lengths = torch.tensor(
            [len(chunk) / padded.shape[1] for chunk in segments], device=self.device
        )
        with torch.no_grad():
            return self.encoder.encode_batch(padded, lengths).squeeze(1).cpu()


class SpectralDiarizer(EmbeddingClusteringDiarizer):
    """
    MFCC mean and standard deviation as the speaker embedding.

    Vocal-tract shape shows up in the cepstral means, which is why this
    separates two clearly different voices at all. It has no notion of speaker
    identity beyond that, so it degrades quickly with similar voices or noise -
    it exists so the pipeline runs anywhere, not because it is competitive.
    """

    name = "spectral"

    def __init__(self, n_mfcc: int = 30, distance_threshold: float = 0.35, **kwargs):
        kwargs.setdefault("distance_threshold", distance_threshold)
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
        self._mfcc = None

    def _transform(self, sample_rate: int):
        import torchaudio

        if self._mfcc is None:
            self._mfcc = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
            )
        return self._mfcc

    def embed(self, segments, sample_rate: int) -> torch.Tensor:
        transform = self._transform(sample_rate)
        vectors = []
        for chunk in segments:
            coefficients = transform(chunk.reshape(1, -1)).squeeze(0)
            # Drop MFCC 0: it tracks loudness, which says more about the
            # microphone distance than about who is speaking.
            coefficients = coefficients[1:]
            vectors.append(torch.cat([coefficients.mean(dim=1), coefficients.std(dim=1)]))
        return torch.stack(vectors)


def build_diarizer(
    backend: str = DEFAULT_BACKEND,
    token: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> BaseDiarizer:
    """
    Instantiates a backend by name.

    `auto` walks the list from best to most-available and returns the first that
    loads, which is what the app uses so a missing licence downgrades the demo
    instead of ending it.
    """
    backend = (backend or DEFAULT_BACKEND).lower()

    if backend == "auto":
        errors = []
        for candidate in BACKENDS:
            try:
                return build_diarizer(candidate, token=token, device=device, **kwargs)
            except Exception as exc:
                errors.append(f"{candidate}: {type(exc).__name__}")
        raise DiarizationError("No diarization backend could load. Tried " + ", ".join(errors))

    if backend == "pyannote":
        return PyannoteDiarizer(token=token, device=device, **kwargs)
    if backend == "ecapa":
        return EcapaDiarizer(device=device, **kwargs)
    if backend == "spectral":
        return SpectralDiarizer(**kwargs)

    raise ValueError(f"Unknown diarization backend '{backend}'. Available: {BACKENDS} or 'auto'.")
