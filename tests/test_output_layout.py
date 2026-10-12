"""
Tests that run artifacts land in the per-domain output tree.

Artifacts are grouped by model domain - `outputs/asr/` today, and
`outputs/translation/` once phase two exists - so that two models cannot
overwrite each other's checkpoints or publish a colliding `v001`. Everything
downstream hangs off `PathConfig.domain_dir`, which makes it a single point of
failure worth pinning down.

`Trainer._publish` in particular catches every exception and downgrades it to a
log warning, on the reasoning that a finished checkpoint should not be lost to a
registry problem. The cost is that a wrong registry path is invisible at
runtime: training reports success and simply does not publish. Only a test
notices.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import PipelineConfig
from src.asr.data.text_transform import TextTransform
from src.asr.models import build_architecture, subsampling_factor
from src.asr.training.trainer import Trainer
from src.utils.model_registry import ModelRegistry
from src.utils.run_logger import RunManager


def _scratch_config(tmp_path):
    """A configuration whose whole output tree lives inside tmp_path."""
    config = PipelineConfig()
    config.device = "cpu"
    config.model.architecture = "conformer"
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.use_amp = False
    config.training.num_workers = 0

    root = tmp_path / "outputs"
    config.paths.output_dir = str(root)
    config.paths.domain_dir = str(root / "asr")
    config.paths.logs_dir = str(root / "logs")
    config.paths.checkpoints_dir = str(root / "asr" / "checkpoints")
    return config, root


def _tiny_loader(config, text_transform, batches=2):
    """
    Batches shaped like the collator's output: spectrogram, labels, lengths.

    `AudioCollator` emits input lengths already divided by the architecture's
    subsampling factor (`spec.shape[0] // stride`), not raw frame counts, so the
    same convention is used here.
    """
    torch.manual_seed(0)
    frames, n_mels = 160, config.audio.n_mels
    stride = subsampling_factor(config.model.architecture)
    spectrograms = torch.randn(batches * config.training.batch_size, 1, n_mels, frames)
    # Short targets so CTC can align them inside the subsampled frames.
    labels = torch.randint(1, 20, (batches * config.training.batch_size, 4))
    input_lengths = torch.full((len(spectrograms),), frames // stride, dtype=torch.long)
    label_lengths = torch.full((len(labels),), 4, dtype=torch.long)

    dataset = TensorDataset(spectrograms, labels, input_lengths, label_lengths)
    return DataLoader(dataset, batch_size=config.training.batch_size, collate_fn=_collate)


def _collate(items):
    spectrograms = torch.stack([i[0] for i in items])
    labels = torch.stack([i[1] for i in items])
    input_lengths = torch.stack([i[2] for i in items])
    label_lengths = torch.stack([i[3] for i in items])
    return spectrograms, labels, input_lengths, label_lengths


def _train_once(tmp_path):
    config, root = _scratch_config(tmp_path)
    text_transform = TextTransform()
    model = build_architecture(
        config.model.architecture, text_transform.vocab_size, config
    )
    loader = _tiny_loader(config, text_transform)
    run = RunManager(kind="train", config=config, console=False)
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        text_transform=text_transform,
        run=run,
    )
    trainer.train()
    return config, root, trainer, run


def test_training_writes_every_artifact_into_the_domain_tree(tmp_path):
    """
    Checkpoints, run records and the registry belong to the domain; logs are
    shared. A path regression here is silent, so it is asserted rather than
    eyeballed.
    """
    config, root, trainer, run = _train_once(tmp_path)

    assert (root / "asr" / "checkpoints").is_dir(), "checkpoints left the domain tree"
    assert (root / "asr" / "runs" / run.run_id).is_dir(), "run directory left the domain tree"
    assert (root / "logs" / f"{run.run_id}.log").is_file(), "logs are shared, not per-domain"

    # Nothing may be written to the pre-reorganization locations.
    assert not (root / "checkpoints").exists()
    assert not (root / "runs").exists()
    assert not (root / "registry").exists()


def test_training_publishes_into_the_domain_registry(tmp_path):
    """
    `_publish` swallows its exceptions, so a wrong registry root would show up
    only as a warning in a log nobody reads. This asserts the version is really
    on disk under the domain.
    """
    config, root, trainer, run = _train_once(tmp_path)

    registry = ModelRegistry(config.paths.domain_dir)
    versions = registry.versions("conformer")

    assert versions, "training completed without publishing anything"
    assert (root / "asr" / "registry").is_dir()

    published = versions[-1]
    assert published.path.is_file(), "published version has no weights on disk"


def test_published_metrics_describe_the_checkpoint_that_was_saved(tmp_path):
    """
    Guards the regression that promoted a worse model: the checkpoint was chosen
    on val_loss while the published metrics were taken from the best-WER epoch,
    so v003 carried epoch-1 weights labelled with an epoch-38 WER. The metrics
    must come from the epoch actually written to best_model.pt.
    """
    config, root, trainer, run = _train_once(tmp_path)

    registry = ModelRegistry(config.paths.domain_dir)
    published = registry.versions("conformer")[-1]

    assert trainer.best_metrics, "no metrics were captured for the saved checkpoint"
    for name in ("wer", "cer"):
        recorded = published.metrics.get(name)
        captured = trainer.best_metrics.get(name)
        if captured is None:
            continue
        assert recorded == captured, (
            f"published {name}={recorded} does not describe the saved checkpoint "
            f"({name}={captured})"
        )


def test_a_second_domain_cannot_collide_with_asr(tmp_path):
    """
    The point of the split: a translation model publishing its own v001 must not
    touch the ASR registry. Verified with the registry directly, since no second
    model exists yet.
    """
    config, root, trainer, run = _train_once(tmp_path)

    asr = ModelRegistry(str(root / "asr"))
    translation = ModelRegistry(str(root / "translation"))

    assert asr.versions("conformer"), "ASR registry should hold the trained model"
    assert not translation.versions("conformer"), "domains share a registry"
    assert asr.root != translation.root
