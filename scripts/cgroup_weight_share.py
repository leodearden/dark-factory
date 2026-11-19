"""Measure, over a window, what CPU each cgroup used and wanted and what its cpu.weight entitles it to among its siblings.

Read-only. Usually pointed at the user@1000.service tree described in
docs/cgroup-slice-topology.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
import os
import posixpath
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from shared.psi import read_runqueue_ratio

USEC_PER_S = 1e6
NS_PER_S = 1e9
SIBLING_COLUMNS = ('path', 'weight', 'used', 'wait', 'wanted', 'share_of_parent', 'fair', 'used/fair')
CHAIN_COLUMNS = ('path', 'weight', 'used', 'parent_used', 'share_of_parent', 'fair', 'used/fair')


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
    """Read every cgroup under cgroup_root, keyed by its path relative to the root ('.' for the root).

    All usage counters are read in one pass before any thread, so they share the snapshot's time.
    """
    counters = dict(_read_counters(cgroup_root))
    taken_at = time.monotonic()
    readings = {
        path: CgroupReading(usage_usec, weight, MappingProxyType(_read_thread_waits(proc_root, tids)))
        for path, (usage_usec, weight, tids) in counters.items()
    }
    return Snapshot(MappingProxyType(readings), taken_at)


def measure(before: Snapshot, after: Snapshot, seconds: float) -> dict[str, CgroupWindow]:
    """Per cgroup present in both snapshots: cores used, cores waited by its subtree, and its weight."""
    if seconds <= 0:
        raise ValueError(f'window length must be positive, got {seconds}')
    common = sorted(before.readings.keys() & after.readings.keys())
    own_wait_ns = {path: _own_wait_ns(before.readings[path], after.readings[path]) for path in common}
    windows: dict[str, CgroupWindow] = {}
    for path in common:
        used = (after.readings[path].usage_usec - before.readings[path].usage_usec) / (seconds * USEC_PER_S)
        wait = sum(ns for other, ns in own_wait_ns.items() if _is_within(other, path)) / (seconds * NS_PER_S)
        windows[path] = CgroupWindow(used, wait, used + wait, after.readings[path].weight)
    return windows


@dataclasses.dataclass(frozen=True)
class ChainLevel:
    path: str
    weight: int | None
    used_cores: float
    parent_used_cores: float
    share_of_parent: float
    fair_cores: float | None
    used_vs_fair: float | None


def share_chain(windows: Mapping[str, CgroupWindow], leaf: str) -> list[ChainLevel]:
    """From leaf up to the root's child: each level's share of its parent against its weight-fair share."""
    if leaf not in windows:
        raise KeyError(f'{leaf} is not in the measured tree')
    levels: list[ChainLevel] = []
    path = leaf
    while path != '.':
        levels.append(_level(windows, path))
        path = _parent(path)
    return levels


def _level(windows: Mapping[str, CgroupWindow], path: str) -> ChainLevel:
    """Water-fill the parent's actual usage among path and its siblings, by weight, capped at each one's wanted."""
    parent = _parent(path)
    siblings = {other: window for other, window in windows.items() if other != '.' and _parent(other) == parent}
    claims = {other: Claim(window.weight, window.wanted_cores) for other, window in siblings.items()}
    shares = fair_shares(windows[parent].used_cores, claims)
    fair = None if shares is None else shares[path]
    used = windows[path].used_cores
    parent_used = windows[parent].used_cores
    return ChainLevel(
        path=path,
        weight=windows[path].weight,
        used_cores=used,
        parent_used_cores=parent_used,
        share_of_parent=used / parent_used if parent_used else 0.0,
        fair_cores=fair,
        used_vs_fair=used / fair if fair else None,
    )


def host_share(levels: Iterable[ChainLevel]) -> float:
    return math.prod(level.share_of_parent for level in levels)


def main(argv: Sequence[str] | None = None) -> int:
    """Print each window; exit 2 if any --chain was absent from some window, after printing the rest."""
    args = _parse_args(argv)
    chain_missing = False
    for _ in range(args.repeat):
        before = read_tree(args.cgroup_root, args.proc_root)
        time.sleep(args.window)
        after = read_tree(args.cgroup_root, args.proc_root)
        elapsed = after.taken_at - before.taken_at
        windows = measure(before, after, elapsed)
        print(_header(elapsed, args.proc_root))
        _print_table(_sibling_rows(windows, args.min_cores))
        for chain in args.chain:
            if chain in windows:
                _print_chain(share_chain(windows, chain), chain)
            else:
                chain_missing = True
                sys.stdout.flush()
                print(f'--chain {chain}: not in the measured tree under {args.cgroup_root}', file=sys.stderr)
        print(flush=True)
    return 2 if chain_missing else 0


def _print_chain(levels: Sequence[ChainLevel], leaf: str) -> None:
    print(f'chain {leaf}')
    _print_table([CHAIN_COLUMNS, *(_chain_row(level) for level in levels)])
    print(f'host_share {host_share(levels):.4f}')


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cgroup-root', type=Path, default=Path('/sys/fs/cgroup'))
    parser.add_argument('--proc-root', type=Path, default=Path('/proc'))
    parser.add_argument('--window', type=float, default=60.0, help='seconds per window')
    parser.add_argument('--repeat', type=int, default=1, help='number of windows')
    parser.add_argument('--chain', action='append', default=[], help='cgroup path relative to --cgroup-root')
    parser.add_argument('--min-cores', type=float, default=0.05, help='hide cgroups wanting fewer cores')
    return parser.parse_args(argv)


def _header(elapsed: float, proc_root: Path) -> str:
    runqueue = read_runqueue_ratio(proc_stat_path=proc_root / 'stat')
    ratio = f'{runqueue.ratio:.2f}' if runqueue.read_ok else '?'
    now = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    return f'# {now} window={elapsed:.1f}s procs_running/nproc {ratio}'


def _sibling_rows(windows: Mapping[str, CgroupWindow], min_cores: float) -> list[tuple[str, ...]]:
    shown = [path for path, window in windows.items() if window.wanted_cores >= min_cores]
    return [SIBLING_COLUMNS, *(_sibling_row(windows, path) for path in shown)]


def _sibling_row(windows: Mapping[str, CgroupWindow], path: str) -> tuple[str, ...]:
    window = windows[path]
    measured = (
        path, _number(window.weight), _number(window.used_cores),
        _number(window.wait_cores), _number(window.wanted_cores),
    )
    if path == '.':
        return (*measured, '-', '-', '-')
    level = _level(windows, path)
    return (*measured, _number(level.share_of_parent), _number(level.fair_cores), _number(level.used_vs_fair))


def _chain_row(level: ChainLevel) -> tuple[str, ...]:
    return (
        level.path, _number(level.weight), _number(level.used_cores), _number(level.parent_used_cores),
        _number(level.share_of_parent), _number(level.fair_cores), _number(level.used_vs_fair),
    )


def _number(value: float | None) -> str:
    if value is None:
        return '-'
    return str(value) if isinstance(value, int) else f'{value:.2f}'


def _print_table(rows: Sequence[tuple[str, ...]]) -> None:
    path_width = max(len(row[0]) for row in rows)
    for row in rows:
        print(f'{row[0]:<{path_width}} ' + ' '.join(f'{cell:>15}' for cell in row[1:]))


def _parent(path: str) -> str:
    return posixpath.dirname(path) or '.'


def _read_counters(cgroup_root: Path) -> Iterator[tuple[str, tuple[int, int | None, list[int]]]]:
    for dirpath, _, _ in os.walk(cgroup_root):
        directory = Path(dirpath)
        try:
            usage_usec = _read_usage_usec(directory / 'cpu.stat')
            weight = _read_weight(directory / 'cpu.weight')
            tids = [int(tid) for tid in (directory / 'cgroup.threads').read_text().split()]
        except OSError:
            continue  # the cgroup was removed mid-walk
        yield directory.relative_to(cgroup_root).as_posix(), (usage_usec, weight, tids)


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


if __name__ == '__main__':
    sys.exit(main())
