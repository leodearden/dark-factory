"""Derive one verify host's on and off blocks over a report window from its lifecycle rows.

The timeline is fleet-wide: the transitions are every store's host lifecycle
rows for one host, and the first verify is the host's earliest merge_verify
start in any store. Blocks tile [since, until) exactly.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

__all__ = [
    'BLOCK_RULE',
    'Block',
    'BlockOrigin',
    'HostState',
    'LifecycleOrigin',
    'Transition',
    'derive_blocks',
]

BLOCK_RULE = (
    'a block is a maximal run of one host state over [since, until), fleet-wide: '
    'verify_host_admitted rows for the host turn it on and verify_host_retired rows turn it off, '
    'in any store; consecutive rows of one kind (one toggle reloaded in each orchestrator) '
    'collapse to the earliest; before the first such row the host is on from its first '
    'merge_verify start in any store and off before it'
)


class HostState(StrEnum):
    ON = 'on'
    OFF = 'off'


class LifecycleOrigin(StrEnum):
    ADMITTED = 'admitted'
    RETIRED = 'retired'


class BlockOrigin(StrEnum):
    WINDOW_START = 'window_start'
    FIRST_VERIFY = 'first_verify'
    ADMITTED = 'admitted'
    RETIRED = 'retired'


@dataclass(frozen=True)
class Transition:
    """One host lifecycle row: at *at* the host entered *state*."""

    at: datetime
    state: HostState
    origin: LifecycleOrigin


@dataclass(frozen=True)
class Block:
    """[start, end) in one *state*, entered at *opened_at* by *opened_by* (which may precede start)."""

    state: HostState
    start: datetime
    end: datetime
    opened_by: BlockOrigin
    opened_at: datetime

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400

    def contains(self, at: datetime) -> bool:
        return self.start <= at < self.end


@dataclass(frozen=True)
class _Boundary:
    at: datetime
    state: HostState
    opened_by: BlockOrigin


def _boundaries(transitions: Iterable[Transition], first_verify_at: datetime | None) -> list[_Boundary]:
    """The state changes in time order: the first verify if it precedes every row, then the rows collapsed."""
    ordered = [_Boundary(t.at, t.state, BlockOrigin(t.origin.value)) for t in sorted(transitions, key=lambda t: t.at)]
    if first_verify_at is not None and (not ordered or first_verify_at < ordered[0].at):
        ordered.insert(0, _Boundary(first_verify_at, HostState.ON, BlockOrigin.FIRST_VERIFY))
    collapsed: list[_Boundary] = []
    for boundary in ordered:
        if not collapsed or boundary.state != collapsed[-1].state:
            collapsed.append(boundary)
    return collapsed


def _clip(opening: _Boundary, later: Sequence[_Boundary], since: datetime, until: datetime) -> tuple[Block, ...]:
    """Cut [since, until) at each later change of state, dropping empty segments."""
    blocks: list[Block] = []
    current, start = opening, since
    for boundary in later:
        if boundary.at >= until:
            break
        if boundary.state == current.state:
            continue
        if boundary.at > start:
            blocks.append(Block(current.state, start, boundary.at, current.opened_by, current.at))
        current, start = boundary, boundary.at
    blocks.append(Block(current.state, start, until, current.opened_by, current.at))
    return tuple(blocks)


def derive_blocks(
    transitions: Iterable[Transition],
    first_verify_at: datetime | None,
    since: datetime,
    until: datetime,
) -> tuple[Block, ...]:
    """The blocks tiling [since, until), each opened by the row or verify that set its state."""
    if until <= since:
        raise ValueError(f'until {until.isoformat()} is not after since {since.isoformat()}')
    boundaries = _boundaries(transitions, first_verify_at)
    prior = [b for b in boundaries if b.at <= since]
    opening = prior[-1] if prior else _Boundary(since, HostState.OFF, BlockOrigin.WINDOW_START)
    return _clip(opening, [b for b in boundaries if b.at > since], since, until)
