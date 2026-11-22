#!/usr/bin/env python3
"""Delete test and probe litter graphs from the FalkorDB the config names, which is production.

A graph is deleted only when its name fully matches one of ``LITTER_PATTERNS``;
that allowlist is the script's whole blast radius, and every other graph is
kept and reported. ``list`` writes a listing for review; ``delete`` removes only
the graphs that listing reviewed and that are still litter, one at a time, and
stops if the server restarts. Runbook: OPERATIONS.md "Deleting test litter from
production".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import re
from collections.abc import Iterable
from dataclasses import dataclass

LITTER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        '_probe',
        '_probe4748b_[0-9a-f]{8}',
        'probe_e1_gw[0-9]+',
        'probe_e1_master',
        'sweep_e4_gw[0-9]+',
        '_test_(?:3334_falkor_fulltext|530_list_indices_integration|4340_read_pagination'
        '|3561_episode_uuid|3769_drop_vector_indices|2207_merge_entities)_[0-9a-f]{8}',
        'evalmem_test_[0-9a-f]{8}',
    )
)


class NotLitter(ValueError):
    """A graph the allowlist does not name, so it is never deleted."""

    def __init__(self, name: str) -> None:
        super().__init__(f'graph {name!r} is not on LITTER_PATTERNS, so it is never deleted')
        self.name = name


def is_litter(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in LITTER_PATTERNS)


def require_litter(name: str) -> str:
    if not is_litter(name):
        raise NotLitter(name)
    return name


@dataclass(frozen=True)
class CleanupPlan:
    delete: tuple[str, ...]
    keep: tuple[str, ...]


def plan_cleanup(names: Iterable[str]) -> CleanupPlan:
    ordered = tuple(names)
    return CleanupPlan(
        delete=tuple(name for name in ordered if is_litter(name)),
        keep=tuple(name for name in ordered if not is_litter(name)),
    )
