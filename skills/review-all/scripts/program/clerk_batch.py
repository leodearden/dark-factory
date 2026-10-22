#!/usr/bin/env python3
"""Clerk batches for a /review-all Phase 6: ``plans/review-all-program-tooling-prd.md`` §4 H4."""
from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import state


@dataclass(frozen=True)
class Impact:
    task: int
    check: str
    kind: Any
    from_path: str
    to_path: str | None


def _checks(row: state.TaskRow) -> Iterator[tuple[int, Mapping[str, Any]]]:
    entries = row.metadata.get('delivered_checks')
    if not isinstance(entries, list | tuple):
        return
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            yield index, entry


def _check_name(entry: Mapping[str, Any], index: int) -> str:
    name = entry.get('name')
    return name if isinstance(name, str) and name else f'delivered_checks[{index}]'


def _anchors(entry: Mapping[str, Any], move: state.Move) -> bool:
    paths = entry.get('paths')
    named = isinstance(paths, list | tuple) and move.from_path in [
        path for path in paths if isinstance(path, str)
    ]
    pattern = entry.get('pattern')
    return named or (
        isinstance(pattern, str) and any(symbol and symbol in pattern for symbol in move.symbols)
    )


def impact_scan(moves: Sequence[state.Move], rows: Sequence[state.TaskRow]) -> tuple[Impact, ...]:
    """Each (move, task, check) whose check names the move's ``from`` in ``paths`` or whose ``pattern`` contains a moved symbol."""
    ordered = sorted(rows, key=lambda row: row.task_id)
    return tuple(
        Impact(row.task_id, _check_name(entry, index), entry.get('kind'), move.from_path, move.to_path)
        for move in moves for row in ordered for index, entry in _checks(row)
        if _anchors(entry, move)
    )
