"""Deterministic seeded train/selection/test partition for the T6 optimization loop.

See plans/tier1-prompt-optimization-prd.md T6: the corpus is split once per
run keyed by `split_seed` so a run is fully reproducible; TEST is held out
separately and scored exactly once at the end of the loop (never used for
acceptance decisions), which is why it gets the largest share.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ['CorpusSplit', 'split_corpus']


@dataclass(frozen=True)
class CorpusSplit:
    """A disjoint, exhaustive partition of a corpus into train/selection/test."""

    train: list[Any]
    selection: list[Any]
    test: list[Any]


def split_corpus(
    items: Iterable[Any],
    *,
    seed: int,
    ratios: tuple[int, int, int] = (2, 1, 7),
) -> CorpusSplit:
    """Deterministically partition *items* into a :class:`CorpusSplit`.

    Shuffles a copy of *items* with ``random.Random(seed)`` (never mutates
    the input, never touches the global ``random`` module state — so this is
    safe to call repeatedly, e.g. once per optimization run, without cross-run
    interference) then slices it by *ratios* (default 2:1:7 —
    train:selection:test). Slicing a single shuffled sequence into three
    contiguous, non-overlapping parts makes the partition disjoint and
    exhaustive by construction — every item appears in exactly one split.

    ``train``/``selection`` sizes are floor-divided from *ratios*; ``test``
    gets whatever remains (``n - train_n - selection_n``), so it silently
    absorbs the rounding remainder and — being the largest ratio share by
    convention — stays the largest split.
    """
    items_list = list(items)
    n = len(items_list)
    train_ratio, selection_ratio, test_ratio = ratios
    total_ratio = train_ratio + selection_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError(f'split_corpus: ratios must sum to a positive number, got {ratios}')

    rng = random.Random(seed)
    shuffled = items_list[:]
    rng.shuffle(shuffled)

    train_n = (n * train_ratio) // total_ratio
    selection_n = (n * selection_ratio) // total_ratio
    # TEST absorbs the remainder so the partition is always exhaustive even
    # when n isn't evenly divisible by total_ratio.
    test_n = n - train_n - selection_n

    train = shuffled[:train_n]
    selection = shuffled[train_n:train_n + selection_n]
    test = shuffled[train_n + selection_n:]

    return CorpusSplit(train=train, selection=selection, test=test)
