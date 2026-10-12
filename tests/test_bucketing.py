"""
Tests for length-bucketed batching.

The corpora this trains on differ enormously in clip length - ghanaopendata's
median is under a second, the health corpus is uniformly 30 - so a randomly
shuffled batch pads almost everything out to 30 seconds and pays full compute
for the silence. These tests pin down that bucketing fixes the padding without
destroying the randomness that training depends on.
"""

import pytest

from src.asr.data.bucketing import (
    LengthBucketedBatchSampler,
    dataset_lengths,
    padding_efficiency,
)

# Roughly the real mixture: many sub-second clips, a tail of long ones.
MIXED = [0.2] * 200 + [1.0] * 200 + [5.0] * 100 + [30.0] * 100


def all_indices(sampler):
    return sorted(i for batch in sampler for i in batch)


def test_every_item_appears_exactly_once_per_epoch():
    sampler = LengthBucketedBatchSampler(MIXED, batch_size=16)
    assert all_indices(sampler) == list(range(len(MIXED)))


def test_batches_respect_the_batch_size():
    sampler = LengthBucketedBatchSampler(MIXED, batch_size=16)
    batches = list(iter(sampler))
    assert all(len(b) <= 16 for b in batches)
    assert sum(len(b) for b in batches) == len(MIXED)
    assert len(batches) == len(sampler)


def test_drop_last_yields_only_full_batches():
    lengths = [1.0] * 50
    sampler = LengthBucketedBatchSampler(lengths, batch_size=16, drop_last=True)
    batches = list(iter(sampler))
    assert all(len(b) == 16 for b in batches)
    assert len(batches) == len(sampler) == 3


def test_bucketing_beats_shuffling_on_padding_waste():
    """The whole point: batched frames should carry audio, not silence."""
    sampler = LengthBucketedBatchSampler(MIXED, batch_size=16, seed=1)
    bucketed = list(iter(sampler))

    shuffled = [list(range(i, min(i + 16, len(MIXED)))) for i in range(0, len(MIXED), 16)]
    # Interleave so each shuffled batch mixes lengths, as random order would.
    interleaved = [[j for j in range(i, len(MIXED), len(MIXED) // 16)][:16]
                   for i in range(len(MIXED) // 16)]

    bucketed_eff = padding_efficiency(MIXED, bucketed)
    mixed_eff = padding_efficiency(MIXED, interleaved)

    # Measured on this mixture: ~0.20 mixed, ~0.97 bucketed. The margins below
    # are loose enough to survive a different shuffle, tight enough that losing
    # the sort would fail them.
    assert mixed_eff < 0.35, f"baseline unexpectedly efficient ({mixed_eff:.2f})"
    assert bucketed_eff > 0.90, f"bucketing left {1 - bucketed_eff:.0%} padding waste"
    assert bucketed_eff > mixed_eff * 3
    assert padding_efficiency(MIXED, shuffled) <= 1.0


def test_batch_order_is_shuffled_between_epochs():
    """Sorted-by-length batch order would correlate the gradient across an epoch."""
    sampler = LengthBucketedBatchSampler(MIXED, batch_size=16, seed=7)
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first != second, "successive epochs produced identical batches"
    assert sorted(i for b in first for i in b) == sorted(i for b in second for i in b)


def test_batch_order_is_not_monotonic_in_length():
    sampler = LengthBucketedBatchSampler(MIXED, batch_size=16, seed=3)
    batches = list(iter(sampler))
    means = [sum(MIXED[i] for i in b) / len(b) for b in batches]
    assert means != sorted(means), "batches are emitted shortest-to-longest"


def test_shuffle_disabled_is_deterministic():
    a = LengthBucketedBatchSampler(MIXED, batch_size=16, shuffle=False)
    b = LengthBucketedBatchSampler(MIXED, batch_size=16, shuffle=False)
    assert list(iter(a)) == list(iter(b))


def test_padding_efficiency_is_one_for_uniform_lengths():
    lengths = [4.0] * 32
    batches = [list(range(16)), list(range(16, 32))]
    assert padding_efficiency(lengths, batches) == pytest.approx(1.0)


def test_padding_efficiency_matches_a_hand_computed_case():
    # One batch of 1s + 3s: 4s real, 6s padded.
    assert padding_efficiency([1.0, 3.0], [[0, 1]]) == pytest.approx(4.0 / 6.0)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        LengthBucketedBatchSampler([], batch_size=8)


def test_invalid_batch_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        LengthBucketedBatchSampler([1.0], batch_size=0)


def test_dataset_lengths_unwraps_concatenated_corpora():
    from torch.utils.data import ConcatDataset

    class Fake:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, idx):
            return self.values[idx]

        def durations(self):
            return self.values

    combined = ConcatDataset([Fake([1.0, 2.0]), Fake([30.0])])
    assert dataset_lengths(combined) == [1.0, 2.0, 30.0]


def test_dataset_without_durations_is_reported_clearly():
    class NoDurations:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return None

    with pytest.raises(TypeError, match="cannot report clip durations"):
        dataset_lengths(NoDurations())
