"""
Length-bucketed batching.

Padding is charged at the length of the longest clip in a batch, so a randomly
shuffled batch that happens to pair a 0.2-second clip with a 30-second one makes
every short clip cost as much as the long one. On these Akan corpora that is not
a rounding error: ghanaopendata's median clip is under a second while the health
corpus is uniformly 30 seconds, so mixing them at random wastes the large
majority of every batch on padding.

Sorting the whole corpus by length would fix the padding and break training -
each batch would come from one narrow slice of the data, so the gradient would
be correlated within a batch and the epoch would walk from short clips to long.
The standard compromise, used here, is a "sortish" sampler: shuffle, cut the
shuffled order into pools of several batches, sort within each pool, and shuffle
the resulting batches. Batches are then internally uniform in length while the
sequence of batches stays random.
"""

import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """
    Yields batches of similarly-sized clips.

    `pool_factor` sets how many batches are sorted together. Larger values pack
    batches more tightly but make the batch order less random. Measured on the
    real length mixture, batched frames that carry audio rather than padding:
    20% unbucketed, 69% at a pool of 4 batches, 87% at 16, 97% at 64. 64 is the
    default because even then a pool is a few thousand clips out of tens of
    thousands - a random enough sample of the corpus that sorting inside it
    costs nothing in gradient quality.
    """

    def __init__(
        self,
        lengths: Sequence[float],
        batch_size: int,
        pool_factor: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(lengths) == 0:
            raise ValueError("Cannot bucket an empty dataset.")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.pool_size = max(batch_size, batch_size * pool_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seeds the shuffle so successive epochs see different groupings."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start:start + self.pool_size]
            pool.sort(key=lambda i: self.lengths[i])
            for offset in range(0, len(pool), self.batch_size):
                batch = pool[offset:offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def padding_efficiency(lengths: Sequence[float], batches: Sequence[Sequence[int]]) -> float:
    """
    Fraction of batched frames that carry real audio rather than padding.

    1.0 means no padding at all; 0.2 means four fifths of the compute is spent
    on silence. Used to report what bucketing bought, and by its tests.
    """
    real = padded = 0.0
    for batch in batches:
        if not len(batch):
            continue
        longest = max(lengths[i] for i in batch)
        real += sum(lengths[i] for i in batch)
        padded += longest * len(batch)
    return real / padded if padded else 0.0


def dataset_lengths(dataset, sample_rate: int = 16000) -> List[float]:
    """
    Per-item clip durations for any dataset the pipeline builds.

    ConcatDataset is unwrapped so several corpora bucket as one pool - which is
    the case that matters most, since the corpora differ from each other far
    more than they vary internally.
    """
    from torch.utils.data import ConcatDataset

    if isinstance(dataset, ConcatDataset):
        values: List[float] = []
        for part in dataset.datasets:
            values.extend(dataset_lengths(part, sample_rate=sample_rate))
        return values

    if hasattr(dataset, "durations"):
        return dataset.durations()

    raise TypeError(
        f"{type(dataset).__name__} cannot report clip durations, so its batches "
        f"cannot be length-bucketed. Pass --no-bucket-batches to train without it."
    )
