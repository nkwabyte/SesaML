import os
from dataclasses import dataclass, field
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    # 80 log-mel bands over a 400-sample (25ms) window with a 160-sample (10ms)
    # hop - the standard ASR front-end. n_mels was 128 against the same 400-point
    # FFT, whose 201 frequency bins cannot support that many triangular filters:
    # four of them came out entirely empty, wasting input dimensions and warning
    # on every run.
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    # SpecAugment. freq_mask_param is ~1/3 of n_mels, matching the paper's LB
    # policy; time masks are additionally capped at a fraction of the utterance
    # so a short clip is not erased outright.
    freq_mask_param: int = 27
    time_mask_param: int = 100
    time_mask_ratio: float = 0.2
    apply_noise_reduction: bool = False

    @property
    def frames_per_second(self) -> float:
        """Mel frames produced per second of audio; used to size clips and batches."""
        return self.sample_rate / self.hop_length

@dataclass
class ModelConfig:
    # Which architecture to build; see src/models/__init__.py ARCHITECTURES.
    architecture: str = "deepspeech"
    n_cnn_layers: int = 3
    n_rnn_layers: int = 5
    rnn_dim: int = 512
    stride: int = 2
    dropout: float = 0.1
    # The input feature count is AudioConfig.n_mels, not a field here: two
    # separately-editable copies of the same number silently produce a shape
    # mismatch the moment one is changed. Likewise n_class comes from the live
    # TextTransform vocabulary, so the model and the label encoding cannot
    # disagree - see build_architecture().

@dataclass
class TrainingConfig:
    batch_size: int = 10
    epochs: int = 10
    learning_rate: float = 5e-4
    # Decoding and mel-transforming audio on the main process starves the GPU.
    # Default to the machine's cores, capped: past ~8 workers the gain is small
    # and each one holds a copy of the dataset.
    num_workers: int = field(default_factory=lambda: min(8, max(1, (os.cpu_count() or 2) - 1)))
    pin_memory: bool = True
    # CTC gradients spike when a batch mixes very short and very long targets;
    # unclipped, one such batch can undo an epoch of progress.
    grad_clip_norm: float = 5.0
    # Mixed precision on CUDA. The encoder runs in fp16 while log_softmax and the
    # CTC loss stay in fp32, which is where the numerical sensitivity lives.
    use_amp: bool = True
    logging_freq: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    model_name: str = "speech_recognition_model.pt"
    # Number of reference/hypothesis pairs persisted per evaluation pass
    max_logged_predictions: int = 50

@dataclass
class PathConfig:
    data_dir: str = "data"
    corpus_dir: str = "data/corpus"
    # Everything a run produces lives under output_dir. Only checkpoints/ and
    # exports/ are excluded from version control (see outputs/.gitignore).
    output_dir: str = "outputs"
    runs_dir: str = "outputs/runs"
    logs_dir: str = "outputs/logs"
    checkpoints_dir: str = "outputs/checkpoints"
    exports_dir: str = "outputs/exports"
    models_dir: str = "outputs/checkpoints"

def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = field(default_factory=get_default_device)

    def to_dict(self) -> dict:
        """Serializable snapshot of the whole configuration, stored with every run."""
        from dataclasses import asdict
        return asdict(self)
