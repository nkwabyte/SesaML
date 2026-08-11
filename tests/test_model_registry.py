"""
Tests for the versioned model registry.

The registry exists because checkpoints used to be resolved by modification
time, so the most recent training run was served whether or not it was any
good. The properties that matter are therefore about *not* serving things: a
worse run must not be promoted, a rollback must reach the last good version,
and weights trained on different features must be flagged rather than loaded
silently.
"""

import json
import pathlib

import pytest

from src.config import PipelineConfig
from src.utils.model_registry import ModelRegistry, ModelVersion, RegistryError
from src.utils.run_logger import load_model_meta


@pytest.fixture
def weights(tmp_path):
    """A stand-in checkpoint; the registry only ever copies and hashes it."""
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not really a torch file, but it is bytes")
    return path


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(str(tmp_path / "outputs"))


def publish(registry, weights, wer, run_id="run", architecture="conformer", **kwargs):
    return registry.publish(
        weights, architecture=architecture, metrics={"wer": wer, "cer": wer / 3},
        run_id=run_id, **kwargs
    )


# --- publishing -----------------------------------------------------------

def test_first_version_is_published_and_promoted(registry, weights):
    version = publish(registry, weights, 0.5)
    assert version.version == "v001"
    assert registry.current("conformer").version == "v001"
    assert version.path.is_file()


def test_versions_are_numbered_in_order(registry, weights):
    assert [publish(registry, weights, 0.5).version for _ in range(3)] == ["v001", "v002", "v003"]


def test_publishing_records_a_checksum_and_size(registry, weights):
    version = publish(registry, weights, 0.5)
    assert len(version.metadata["sha256"]) == 64
    assert version.metadata["size_bytes"] == weights.stat().st_size


def test_publishing_a_missing_checkpoint_is_rejected(registry, tmp_path):
    with pytest.raises(RegistryError, match="no checkpoint"):
        registry.publish(tmp_path / "nope.pt", architecture="conformer")


# --- the promotion guard --------------------------------------------------

def test_a_better_run_is_promoted(registry, weights):
    publish(registry, weights, 0.50, run_id="first")
    publish(registry, weights, 0.40, run_id="better")
    assert registry.current("conformer").version == "v002"


def test_a_worse_run_is_archived_but_not_promoted(registry, weights):
    """The whole point: a collapsed run must never become the served model."""
    publish(registry, weights, 0.50, run_id="good")
    publish(registry, weights, 1.00, run_id="collapsed")

    assert registry.current("conformer").version == "v001"
    assert len(registry.versions("conformer")) == 2, "the bad run should still be archived"


def test_an_equal_run_does_not_displace_the_incumbent(registry, weights):
    publish(registry, weights, 0.50)
    publish(registry, weights, 0.50)
    assert registry.current("conformer").version == "v001"


def test_a_run_without_metrics_never_displaces_a_scored_one(registry, weights):
    publish(registry, weights, 0.50)
    registry.publish(weights, architecture="conformer", metrics={}, run_id="unscored")
    assert registry.current("conformer").version == "v001"


def test_promotion_can_be_forced(registry, weights):
    publish(registry, weights, 0.50)
    registry.publish(weights, architecture="conformer", metrics={"wer": 1.0}, promote=True)
    assert registry.current("conformer").version == "v002"


def test_promotion_can_be_suppressed(registry, weights):
    publish(registry, weights, 0.50)
    registry.publish(weights, architecture="conformer", metrics={"wer": 0.1}, promote=False)
    assert registry.current("conformer").version == "v001"


def test_cer_breaks_the_tie_when_wer_is_absent(registry, weights):
    registry.publish(weights, architecture="conformer", metrics={"cer": 0.30})
    registry.publish(weights, architecture="conformer", metrics={"cer": 0.20})
    assert registry.current("conformer").version == "v002"


# --- rollback -------------------------------------------------------------

def test_rollback_returns_to_the_previous_promotion(registry, weights):
    publish(registry, weights, 0.50)
    publish(registry, weights, 0.40)
    assert registry.current("conformer").version == "v002"

    rolled = registry.rollback("conformer")
    assert rolled.version == "v001"
    assert registry.current("conformer").version == "v001"


def test_rollback_follows_promotion_history_not_version_numbers(registry, weights):
    """After a forced promotion, the fallback is the last *served* version."""
    publish(registry, weights, 0.50)             # v001 promoted
    publish(registry, weights, 0.90)             # v002 archived only
    registry.promote("conformer", "v002")        # forced

    assert registry.rollback("conformer").version == "v001"


def test_rollback_without_history_is_refused(registry, weights):
    publish(registry, weights, 0.5)
    with pytest.raises(RegistryError, match="no earlier promoted version"):
        registry.rollback("conformer")


def test_promoting_an_unknown_version_is_refused(registry, weights):
    publish(registry, weights, 0.5)
    with pytest.raises(RegistryError, match="no such version"):
        registry.promote("conformer", "v099")


# --- resolution -----------------------------------------------------------

def test_resolve_prefers_the_promoted_version(registry, weights):
    publish(registry, weights, 0.50)
    registry.publish(weights, architecture="conformer", metrics={"wer": 0.1}, promote=False)
    assert registry.resolve("conformer").version == "v001"


def test_resolve_finds_any_architecture_when_none_is_named(registry, weights):
    publish(registry, weights, 0.5, architecture="deepspeech")
    assert registry.resolve().architecture == "deepspeech"


def test_resolve_on_an_empty_registry_is_none(registry):
    assert registry.resolve() is None
    assert registry.current("conformer") is None
    assert registry.versions("conformer") == []


def test_best_ignores_which_version_is_promoted(registry, weights):
    publish(registry, weights, 0.50)
    registry.publish(weights, architecture="conformer", metrics={"wer": 0.10}, promote=False)
    assert registry.best("conformer").version == "v002"


# --- feature compatibility ------------------------------------------------

def test_matching_feature_settings_report_no_incompatibility(registry, weights):
    config = PipelineConfig()
    version = registry.publish(weights, architecture="conformer", config=config)
    assert version.incompatibilities(config) == []


def test_a_changed_mel_count_is_reported(registry, weights):
    """Loading 80-mel weights under a 128-mel config does not error - it degrades."""
    trained = PipelineConfig()
    version = registry.publish(weights, architecture="conformer", config=trained)

    changed = PipelineConfig()
    changed.audio.n_mels = 128
    problems = version.incompatibilities(changed)

    assert len(problems) == 1
    assert "n_mels" in problems[0]
    assert "trained with 80" in problems[0]


def test_a_version_without_recorded_features_claims_no_incompatibility(registry, weights):
    version = registry.publish(weights, architecture="conformer")
    version.metadata["features"] = {}
    assert version.incompatibilities(PipelineConfig()) == []


# --- metadata round-trip --------------------------------------------------

def test_load_model_meta_reads_the_registry_layout(registry, weights):
    """Architecture detection must work for registry versions, not just run dirs."""
    version = registry.publish(
        weights, architecture="conformer", config=PipelineConfig(),
        vocab_size=41, subsampling_factor=4,
    )
    meta = load_model_meta(version.path)

    assert meta["architecture"] == "conformer"
    # "features" is flattened so callers see one shape from either layout.
    assert meta["vocab_size"] == 41
    assert meta["subsampling_factor"] == 4
    assert meta["n_mels"] == PipelineConfig().audio.n_mels


def test_summary_marks_the_served_version(registry, weights):
    publish(registry, weights, 0.50)
    publish(registry, weights, 0.90)
    rows = registry.summary()
    assert [row["current"] for row in rows] == [True, False]


def test_a_corrupt_index_is_reported_not_swallowed(registry, weights):
    publish(registry, weights, 0.5)
    registry.index_path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="unreadable"):
        registry.current("conformer")


def test_index_survives_being_rewritten(registry, weights):
    publish(registry, weights, 0.5)
    publish(registry, weights, 0.4)
    index = json.loads(registry.index_path.read_text(encoding="utf-8"))
    assert index["architectures"]["conformer"]["history"] == ["v001", "v002"]


def test_architectures_are_isolated_from_each_other(registry, weights):
    publish(registry, weights, 0.90, architecture="deepspeech")
    publish(registry, weights, 0.50, architecture="conformer")

    assert registry.current("deepspeech").version == "v001"
    assert registry.current("conformer").version == "v001"
    assert sorted(registry.architectures()) == ["conformer", "deepspeech"]


# --- outputs cleanup ------------------------------------------------------

def test_cleanup_protects_runs_that_produced_published_versions(registry, weights, tmp_path):
    """Protection is derived from the registry, so it follows what is published."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from clean_outputs import protected_runs

    registry.publish(weights, architecture="conformer", run_id="run-clean")
    assert "run-clean" in protected_runs(registry.root.parent)


def test_cleanup_dedupes_only_byte_identical_weights(registry, weights, tmp_path):
    """
    last_model.pt carries optimizer state the registry never stores, so a
    dedupe that matched on filename rather than content would destroy the
    ability to resume a run.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from clean_outputs import duplicate_weights

    outputs = registry.root.parent
    run_dir = outputs / "checkpoints" / "run-clean"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(weights.read_bytes())          # identical
    (run_dir / "last_model.pt").write_bytes(weights.read_bytes() + b"++")  # unique

    registry.publish(weights, architecture="conformer", run_id="run-clean")

    removable = [path.name for path, _, _ in duplicate_weights(outputs, {"run-clean"})]
    assert removable == ["best_model.pt"]
    assert (run_dir / "last_model.pt").exists()


def test_cleanup_finds_nothing_to_dedupe_on_an_empty_registry(tmp_path):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from clean_outputs import duplicate_weights

    (tmp_path / "checkpoints").mkdir()
    assert duplicate_weights(tmp_path, {"anything"}) == []
