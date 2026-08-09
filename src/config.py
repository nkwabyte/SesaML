import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    freq_mask_param: int = 30
    time_mask_param: int = 100
    apply_noise_reduction: bool = False

@dataclass
class ModelConfig:
    n_cnn_layers: int = 3
    n_rnn_layers: int = 5
    rnn_dim: int = 512
    n_feats: int = 128
    stride: int = 2
    dropout: float = 0.1
    # Fallbacks only. The CLI builds models from TextTransform.vocab_size so the
    # vocabulary stays the single source of truth: 4 special + 26 letters +
    # 10 digits = 40 symbols, plus the CTC blank at index 40.
    n_class: int = 41
    blank_label: int = 40  # Always n_class - 1

@dataclass
class TrainingConfig:
    batch_size: int = 10
    epochs: int = 10
    learning_rate: float = 5e-4
    num_workers: int = 1
    pin_memory: bool = True
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

@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        """Serializable snapshot of the whole configuration, stored with every run."""
        from dataclasses import asdict
        return asdict(self)
