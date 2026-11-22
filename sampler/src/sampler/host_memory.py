"""The host's reclaim threads and memory totals: C1's host-level figures.

PRD ``plans/per-consumer-resource-recording-prd.md`` C1. The kernel's
background reclaim (``kswapd<N>``) and compaction (``kcompactd<N>``) threads
are read from ``/proc/<pid>/stat`` through ``shared.proc_group``, and their
CPU becomes per-kind increments over the interval by ``sampler.increments``'
D3 counter and gap rules, which ``sampler.consumers`` applies to cgroups. Host used memory is
``MemTotal - MemAvailable`` from ``/proc/meminfo``; host program memory and
file cache come from the root cgroup's ``memory.stat``.

A figure that cannot be read is ``None``, never 0. This module names no
metric: ``sampler.metrics`` imports it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from shared.cgroup_stats import CGROUP_ROOT, MEMORY_STAT, parse_keyed_counters
from shared.proc_group import read_cpu_ticks, read_stat_fields

from sampler.increments import counter_increment, is_gap

__all__ = [
    'MEMINFO',
    'PROC_ROOT',
    'HostMemoryStat',
    'ReclaimState',
    'ReclaimThread',
    'ReclaimThreadKind',
    'read_host_memory_stat',
    'read_host_used_bytes',
    'read_reclaim_threads',
    'reclaim_increments',
]

PROC_ROOT = Path('/proc')
MEMINFO = Path('/proc/meminfo')

_USEC_PER_TICK = 1_000_000 / os.sysconf('SC_CLK_TCK')


class ReclaimThreadKind(StrEnum):
    KSWAPD = 'kswapd'
    KCOMPACTD = 'kcompactd'


_RECLAIM_COMM = re.compile(rf'({"|".join(kind.value for kind in ReclaimThreadKind)})\d+')


@dataclass(frozen=True)
class ReclaimThread:
    pid: int
    kind: ReclaimThreadKind
    cpu_usec: int


def _reclaim_thread(entry: Path) -> ReclaimThread | None:
    fields = read_stat_fields(entry, skip_malformed=True)
    match = None if fields is None else _RECLAIM_COMM.fullmatch(fields.comm)
    if match is None:
        return None
    ticks = read_cpu_ticks(entry)
    if ticks is None:
        return None
    return ReclaimThread(int(entry.name), ReclaimThreadKind(match[1]), round(ticks * _USEC_PER_TICK))


def read_reclaim_threads(proc_root: str | Path = PROC_ROOT) -> tuple[ReclaimThread, ...]:
    """Every live kswapd/kcompactd thread with its CPU so far, in µs."""
    pids = sorted(int(name) for name in os.listdir(proc_root) if name.isdigit())
    threads = (_reclaim_thread(Path(proc_root) / str(pid)) for pid in pids)
    return tuple(thread for thread in threads if thread is not None)


_STATE_VERSION = 1


@dataclass(frozen=True)
class ReclaimState:
    """Each reclaim thread's CPU (µs) at ``ts``, keyed by pid."""

    ts: int
    cpu_usec: Mapping[int, int]

    def to_json(self) -> str:
        return json.dumps(
            {
                'version': _STATE_VERSION,
                'ts': self.ts,
                'cpu_usec': {str(pid): usec for pid, usec in self.cpu_usec.items()},
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> ReclaimState | None:
        """The decoded state, or ``None`` for any corrupt or foreign text (a gap)."""
        try:
            data = json.loads(text)
            if data['version'] != _STATE_VERSION:
                return None
            cpu = {int(pid): _strict_int(usec) for pid, usec in data['cpu_usec'].items()}
            return cls(_strict_int(data['ts']), MappingProxyType(cpu))
        except (ValueError, KeyError, TypeError, AttributeError):
            return None


def _strict_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f'not an integer: {value!r}')
    return value


def reclaim_increments(
    previous: ReclaimState | None,
    current: Iterable[ReclaimThread],
    *,
    now: int,
    nominal_interval: int,
) -> tuple[Mapping[ReclaimThreadKind, int] | None, ReclaimState]:
    """Per-kind CPU increments (µs) since ``previous``; ``None`` after a gap or on a first run."""
    threads = tuple(current)
    state = ReclaimState(now, MappingProxyType({t.pid: t.cpu_usec for t in threads}))
    if previous is None or is_gap(previous.ts, now, nominal_interval):
        return None, state
    increments: dict[ReclaimThreadKind, int] = {}
    for thread in threads:
        increment = counter_increment(previous.cpu_usec.get(thread.pid), thread.cpu_usec)
        increments[thread.kind] = increments.get(thread.kind, 0) + increment
    return MappingProxyType(increments), state


_KIB = 1024


def read_host_used_bytes(meminfo_path: str | Path = MEMINFO) -> int | None:
    """``MemTotal - MemAvailable`` in bytes; ``None`` when either is unreadable."""
    try:
        text = Path(meminfo_path).read_text(encoding='utf-8')
    except OSError:
        return None
    kib: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        fields = rest.split()
        if fields and fields[0].isdigit():
            kib[key] = int(fields[0])
    if 'MemTotal' not in kib or 'MemAvailable' not in kib:
        return None
    return (kib['MemTotal'] - kib['MemAvailable']) * _KIB


@dataclass(frozen=True)
class HostMemoryStat:
    anon_bytes: int | None
    file_bytes: int | None


def read_host_memory_stat(cgroup_root: str | Path = CGROUP_ROOT) -> HostMemoryStat:
    """Host program memory and file cache from the root cgroup's ``memory.stat``."""
    try:
        counters = parse_keyed_counters(
            (Path(cgroup_root) / MEMORY_STAT).read_text(encoding='utf-8')
        )
    except (OSError, ValueError):
        return HostMemoryStat(None, None)
    return HostMemoryStat(counters.get('anon'), counters.get('file'))
