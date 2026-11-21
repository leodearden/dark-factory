"""Builders shared by the resource report's tests: project roots, memory holds, consumer collections.

A hold is the event the orchestrator's admission gate writes, its sample an
``asdict`` of a real ``shared.psi.PsiSample``. A collection is written as the
sampler writes it: one row per consumer figure under ``sampler.metrics``'
spec, the consumer encoded by ``sampler.consumers``, into a store built from
``sampler.store.SCHEMA``. So no metric or consumer name is spelled here.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Collection, Mapping
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import _checkout as _checkout
from sampler.consumers import Consumer, Figure, encode_consumer
from sampler.host_memory import ReclaimThreadKind
from sampler.metrics import FIGURE_METRICS, RECLAIM_METRICS, MetricSpec
from sampler.store import SCHEMA
from shared.project_layout import runs_db
from shared.psi import PsiSample


def _readability(spec: MetricSpec) -> MetricSpec:
    if spec.readability is None:
        raise AssertionError(f'{spec.name} declares no readability metric')
    return spec.readability


READ_OK = _readability(FIGURE_METRICS[Figure.ANON].spec)

_INSERT = (
    'INSERT INTO samples (ts, metric, value, window_mean, window_max, leaf)'
    ' VALUES (?, ?, ?, NULL, NULL, ?)'
)


HOLD_SAMPLE = PsiSample(cpu_some10=57.34, mem_some10=9.57, mem_full10=3.7, io_some10=10.25, read_ok=True)
"""The PSI sample every fixture hold carries, its own cgroup aside."""


def memory_hold(
    at: datetime, own_cgroup: str, *, arm: str = 'mem_full_avg10', task_id: str = '5653'
) -> dict[str, Any]:
    """A ``dispatch_deferred`` event for ``make_runs_db``, deferred by *arm* at *at*."""
    sample = replace(HOLD_SAMPLE, own_cgroup=own_cgroup, own_read_ok=bool(own_cgroup))
    return {
        'timestamp': at.isoformat(),
        'event_type': 'dispatch_deferred',
        'task_id': task_id,
        'data': {'metric': arm, 'in_flight': 30, 'floor': 3, 'sample': asdict(sample)},
    }


def project_root(tmp_path: Path, name: str) -> Path:
    """``<tmp_path>/<name>``, with the directory its runs.db lives in already made."""
    root = tmp_path / name
    runs_db(root).parent.mkdir(parents=True)
    return root


def samples_db(path: Path) -> Path:
    """An empty load-samples store at *path*, with the sampler's own schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def write_collection(
    db: Path,
    ts: int,
    consumers: Mapping[Consumer, Mapping[Figure, float]],
    *,
    unreadable: Collection[Consumer] = frozenset(),
    reclaim: Mapping[ReclaimThreadKind, float] | None = None,
    after_gap: bool = False,
) -> None:
    """One consumer collection at *ts*; stall figures are given in seconds, as stored.

    Every consumer named, in *consumers* or *unreadable*, gets a readability
    row: 0 when it is in *unreadable*, else 1. Each *reclaim* thread kind's CPU
    increment (µs) is a host-level row. *after_gap* writes the collection as
    the sampler writes the one after a gap, its first included: every
    consumer flagged unreadable (D3); give it gauges only.
    """
    rows: list[tuple[int, str, float, str | None]] = []
    named = {*consumers, *unreadable}
    flagged = named if after_gap else unreadable

    def add(spec: MetricSpec, value: float, leaf: str | None) -> None:
        rows.append((ts, spec.name, float(value), leaf))

    for consumer in named:
        add(READ_OK, 0 if consumer in flagged else 1, encode_consumer(consumer))
    for consumer, figures in consumers.items():
        for figure, value in figures.items():
            add(FIGURE_METRICS[figure].spec, value, encode_consumer(consumer))
    for kind, usec in (reclaim or {}).items():
        add(RECLAIM_METRICS[kind], usec, None)
    conn = sqlite3.connect(db)
    try:
        conn.execute('PRAGMA synchronous = OFF')  # a throwaway fixture store needs no fsync
        conn.executemany(_INSERT, rows)
        conn.commit()
    finally:
        conn.close()
