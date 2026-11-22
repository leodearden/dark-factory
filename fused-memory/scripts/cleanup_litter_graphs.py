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

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol

from shared.safe_io import atomic_write_text

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


class GraphServer(Protocol):
    def graph_names(self) -> tuple[str, ...]: ...

    def node_count(self, name: str) -> int: ...

    def delete(self, name: str) -> None: ...

    def run_id(self) -> str: ...


def _frozen(counts: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(counts))


@dataclass(frozen=True)
class Census:
    run_id: str
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'counts', _frozen(self.counts))


def take_census(server: GraphServer, names: Iterable[str]) -> Census:
    run_id = server.run_id()
    return Census(run_id, {name: server.node_count(name) for name in names})


@dataclass(frozen=True)
class Listing:
    target: str
    run_id: str
    taken_at: str
    delete: Mapping[str, int]
    keep: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'delete', _frozen(self.delete))
        object.__setattr__(self, 'keep', _frozen(self.keep))

    def to_json(self) -> dict[str, object]:
        return {
            'target': self.target,
            'run_id': self.run_id,
            'taken_at': self.taken_at,
            'delete': dict(self.delete),
            'keep': dict(self.keep),
        }


class UnreadableListing(ValueError):
    """A reviewed listing that cannot be read back as a Listing."""

    def __init__(self, path: Path, defect: Exception) -> None:
        super().__init__(f'cannot read the reviewed listing {path}: {defect}')


def build_listing(target_uri: str, server: GraphServer) -> Listing:
    census = take_census(server, server.graph_names())
    plan = plan_cleanup(census.counts)
    return Listing(
        target=target_uri,
        run_id=census.run_id,
        taken_at=datetime.now(UTC).isoformat(timespec='seconds'),
        delete={name: census.counts[name] for name in plan.delete},
        keep={name: census.counts[name] for name in plan.keep},
    )


def write_listing(path: Path, listing: Listing) -> None:
    atomic_write_text(path, json.dumps(listing.to_json(), sort_keys=True, indent=2) + '\n')


def read_listing(path: Path) -> Listing:
    try:
        return _listing_from_json(json.loads(Path(path).read_text(encoding='utf-8')))
    except (OSError, ValueError) as defect:
        raise UnreadableListing(path, defect) from defect


def _listing_from_json(raw: object) -> Listing:
    if not isinstance(raw, dict):
        raise ValueError(f'expected a JSON object, found {type(raw).__name__}')
    return Listing(
        target=_text(raw, 'target'),
        run_id=_text(raw, 'run_id'),
        taken_at=_text(raw, 'taken_at'),
        delete=_counts(raw, 'delete'),
        keep=_counts(raw, 'keep'),
    )


def _required(raw: dict[str, object], key: str) -> object:
    if key not in raw:
        raise ValueError(f'it has no {key!r} key')
    return raw[key]


def _text(raw: dict[str, object], key: str) -> str:
    value = _required(raw, key)
    if not isinstance(value, str):
        raise ValueError(f'{key!r} is a {type(value).__name__}, not a string')
    return value


def _counts(raw: dict[str, object], key: str) -> dict[str, int]:
    value = _required(raw, key)
    if not isinstance(value, dict) or not all(_is_count(count) for count in value.values()):
        raise ValueError(f'{key!r} is not a map of graph name to node count')
    return value


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
