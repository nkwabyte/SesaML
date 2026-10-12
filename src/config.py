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
    # Which architecture to build; see src/asr/models/__init__.py ARCHITECTURES.
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
class DecodingConfig:
    """
    How per-frame CTC posteriors become text.

    Greedy decoding takes the argmax at every frame independently, so it has no
    way to prefer a spelling that exists in Twi over one that does not - which is
    exactly the model's error profile. Beam search with an n-gram language model
    fused in fixed that: measured 0.4725 -> 0.3936 WER, a 16.7% relative gain,
    with no retraining.

    alpha and beta were swept on the validation split; 0.5/0.5 won. Raising
    alpha past ~0.8 makes the language model overrule the acoustics and WER
    climbs again.
    """

    # "greedy" or "beam". Beam is ~0.1s per clip against greedy's ~0.0004s,
    # which is irrelevant next to the forward pass but matters in a tight loop.
    decoder: str = "beam"
    lm_path: str = "outputs/asr/lm/twi_char.json"
    beam_width: int = 25
    alpha: float = 0.5          # language model weight
    beta: float = 0.5           # per-character bonus, offsetting the LM's brevity bias

    @property
    def uses_lm(self) -> bool:
        return self.decoder == "beam" and bool(self.lm_path)


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
    checkpoint_dir: str = "outputs/asr/checkpoints"
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

    # Artifacts are grouped by model domain, so the translation model added in
    # phase two cannot collide with speech recognition in checkpoints/, runs/ or
    # the registry: two models both called "v001" would otherwise overwrite each
    # other's weights. `domain_dir` is what run and registry paths hang off.
    domain: str = "asr"
    domain_dir: str = "outputs/asr"

    runs_dir: str = "outputs/asr/runs"
    checkpoints_dir: str = "outputs/asr/checkpoints"
    exports_dir: str = "outputs/asr/exports"
    models_dir: str = "outputs/asr/checkpoints"
    registry_dir: str = "outputs/asr/registry"
    lm_dir: str = "outputs/asr/lm"

    # Shared across domains: one flat stream of run logs keyed by run id, so a
    # single directory still answers "what happened in run X".
    logs_dir: str = "outputs/logs"

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
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = field(default_factory=get_default_device)

    def to_dict(self) -> dict:
        """Serializable snapshot of the whole configuration, stored with every run."""
        from dataclasses import asdict
        return asdict(self)
