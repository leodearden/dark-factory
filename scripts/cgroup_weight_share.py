"""Measure, over a window, what CPU each cgroup used and wanted and what its cpu.weight entitles it to among its siblings.

Read-only. Usually pointed at the user@1000.service tree described in
docs/cgroup-slice-topology.md.
"""
from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

USEC_PER_S = 1e6
NS_PER_S = 1e9


@dataclasses.dataclass(frozen=True)
class Claim:
    weight: int | None
    demand: float


def fair_shares(capacity: float, claims: Mapping[str, Claim]) -> dict[str, float] | None:
    """cgroup-v2 cpu.weight water-filling; None when any sibling has no weight."""
    unsatisfied: dict[str, tuple[int, float]] = {}
    for name, claim in claims.items():
        if claim.weight is None:
            return None
        unsatisfied[name] = (claim.weight, claim.demand)
    shares: dict[str, float] = {}
    remaining = capacity
    while satisfied := _satisfied_at(remaining, unsatisfied):
        for name in satisfied:
            _, shares[name] = unsatisfied.pop(name)
            remaining -= shares[name]
    shares.update(_split_by_weight(remaining, unsatisfied))
    return {name: shares[name] for name in claims}


def _split_by_weight(capacity: float, claims: Mapping[str, tuple[int, float]]) -> dict[str, float]:
    total = sum(weight for weight, _ in claims.values())
    return {name: capacity * weight / total for name, (weight, _) in claims.items()}


def _satisfied_at(capacity: float, claims: Mapping[str, tuple[int, float]]) -> list[str]:
    proportional = _split_by_weight(capacity, claims)
    return [name for name, (_, demand) in claims.items() if proportional[name] >= demand]


@dataclasses.dataclass(frozen=True)
class CgroupReading:
    usage_usec: int
    weight: int | None
    thread_wait_ns: Mapping[int, int]


@dataclasses.dataclass(frozen=True)
class Snapshot:
    readings: Mapping[str, CgroupReading]
    taken_at: float


@dataclasses.dataclass(frozen=True)
class CgroupWindow:
    used_cores: float
    wait_cores: float
    wanted_cores: float
    weight: int | None


def read_tree(cgroup_root: Path, proc_root: Path) -> Snapshot:
    """Read every cgroup under cgroup_root, keyed by its path relative to the root ('.' for the root)."""
    readings: dict[str, CgroupReading] = {}
    for dirpath, _, _ in os.walk(cgroup_root):
        directory = Path(dirpath)
        try:
            readings[directory.relative_to(cgroup_root).as_posix()] = _read_cgroup(directory, proc_root)
        except OSError:
            continue  # the cgroup was removed mid-walk
    return Snapshot(MappingProxyType(readings), time.monotonic())


def measure(before: Snapshot, after: Snapshot, seconds: float) -> dict[str, CgroupWindow]:
    """Per cgroup present in both snapshots: cores used, cores waited by its subtree, and its weight."""
    common = sorted(before.readings.keys() & after.readings.keys())
    own_wait_ns = {path: _own_wait_ns(before.readings[path], after.readings[path]) for path in common}
    windows: dict[str, CgroupWindow] = {}
    for path in common:
        used = (after.readings[path].usage_usec - before.readings[path].usage_usec) / (seconds * USEC_PER_S)
        wait = sum(ns for other, ns in own_wait_ns.items() if _is_within(other, path)) / (seconds * NS_PER_S)
        windows[path] = CgroupWindow(used, wait, used + wait, after.readings[path].weight)
    return windows


def _read_cgroup(directory: Path, proc_root: Path) -> CgroupReading:
    tids = [int(tid) for tid in (directory / 'cgroup.threads').read_text().split()]
    return CgroupReading(
        usage_usec=_read_usage_usec(directory / 'cpu.stat'),
        weight=_read_weight(directory / 'cpu.weight'),
        thread_wait_ns=MappingProxyType(_read_thread_waits(proc_root, tids)),
    )


def _read_usage_usec(cpu_stat: Path) -> int:
    for line in cpu_stat.read_text().splitlines():
        key, _, value = line.partition(' ')
        if key == 'usage_usec':
            return int(value)
    raise ValueError(f'{cpu_stat} has no usage_usec line')


def _read_weight(cpu_weight: Path) -> int | None:
    try:
        return int(cpu_weight.read_text())
    except FileNotFoundError:
        return None  # the parent has no cpu controller enabled for its children


def _read_thread_waits(proc_root: Path, tids: Iterable[int]) -> dict[int, int]:
    waits: dict[int, int] = {}
    for tid in tids:
        try:
            waits[tid] = int((proc_root / str(tid) / 'schedstat').read_text().split()[1])
        except OSError:
            continue  # the thread exited after cgroup.threads listed it
    return waits


def _own_wait_ns(before: CgroupReading, after: CgroupReading) -> int:
    common = before.thread_wait_ns.keys() & after.thread_wait_ns.keys()
    return sum(after.thread_wait_ns[tid] - before.thread_wait_ns[tid] for tid in common)


def _is_within(path: str, ancestor: str) -> bool:
    return ancestor == '.' or path == ancestor or path.startswith(f'{ancestor}/')
