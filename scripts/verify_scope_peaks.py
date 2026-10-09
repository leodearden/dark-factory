#!/usr/bin/env python3
"""Record and summarise memory.peak of df-verify-* scopes (task 5415).

A verify scope is `--collect`ed the instant its last process exits, so its
cgroup memory.peak cannot be read after the gate finishes. `poll` therefore
samples live scopes, carries the monotonic peak per scope across ticks, and
appends one JSONL record per scope using the LAST observation once the scope
disappears. `report` turns those records into the per-project p50/p90/max,
the recommended laptop MemoryMax, and the 16/16 vs 8/8 thread-split
recommendation.

  verify_scope_peaks.py poll   --out peaks.jsonl [--interval 5]
  verify_scope_peaks.py report peaks.jsonl [--since 2026-10-09] [--scale dark-factory=2 --scale reify=0.6667]

One gate is several scopes (test / lint / type legs). Legs of one project and
role whose start times are within GATE_GAP_SECS of the previous leg's last
sighting form one gate; the gate figure is the largest leg (legs may be
concurrent, so the sum of legs is reported as the upper bound).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024**3
GATE_GAP_SECS = 60.0
RECOMMEND_HEADROOM = 1.25
SPLIT_SUM_LIMIT_GIB = 50.0
SCOPE_GLOB = '/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/df-verify-*.scope'
SCOPE_NAME = re.compile(r'^df-verify-(?P<slug>.+)-(?P<hash>[0-9a-f]{8})-(?P<uuid>[0-9a-f]+)\.scope$')
THREAD_ENV_KEYS = ('PYTEST_XDIST_AUTO_NUM_WORKERS', 'CARGO_BUILD_JOBS', 'NEXTEST_TEST_THREADS', 'MAKEFLAGS')
DEFAULT_SCALES = {'dark-factory': 16 / 8, 'reify': 16 / 24}


@dataclass
class ScopeRecord:
    unit: str
    slug: str
    role: str
    first_seen: float
    last_seen: float
    peak_bytes: int
    mem_total_gib: float
    mem_available_gib_at_start: float
    threads: dict[str, str] = field(default_factory=dict)


def classify_role(cwd_basename: str) -> str:
    if cwd_basename.startswith('_merge'):
        return 'merge'
    if cwd_basename.startswith('_lane-'):
        return 'task'
    return 'other'


def meminfo_gib(path: str = '/proc/meminfo') -> tuple[float, float]:
    values: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        key, _, rest = line.partition(':')
        values[key] = int(rest.split()[0])
    return values['MemTotal'] / 1024**2, values['MemAvailable'] / 1024**2


def _first_pid(scope_dir: str) -> str | None:
    try:
        pids = Path(scope_dir, 'cgroup.procs').read_text().split()
    except OSError:
        return None
    return pids[0] if pids else None


def _thread_env(pid: str) -> dict[str, str]:
    try:
        raw = Path(f'/proc/{pid}/environ').read_bytes()
    except OSError:
        return {}
    pairs = (item.partition(b'=') for item in raw.split(b'\0') if item)
    env = {key: value for key, _, value in pairs}
    return {k: env[k.encode()].decode('utf8', 'replace') for k in THREAD_ENV_KEYS if k.encode() in env}


def _role_of(pid: str | None) -> str:
    if pid is None:
        return 'other'
    try:
        return classify_role(os.path.basename(os.readlink(f'/proc/{pid}/cwd')))
    except OSError:
        return 'other'


def _read_peak(scope_dir: str) -> int | None:
    try:
        return int(Path(scope_dir, 'memory.peak').read_text())
    except (OSError, ValueError):
        return None


def poll(out: Path, interval: float, scope_glob: str) -> None:
    live: dict[str, ScopeRecord] = {}
    while True:
        now = time.time()
        seen: set[str] = set()
        for scope_dir in glob.glob(scope_glob):
            unit = os.path.basename(scope_dir)
            match = SCOPE_NAME.match(unit)
            peak = _read_peak(scope_dir)
            if match is None or peak is None:
                continue
            seen.add(unit)
            rec = live.get(unit)
            if rec is None:
                pid = _first_pid(scope_dir)
                total, available = meminfo_gib()
                rec = live[unit] = ScopeRecord(
                    unit=unit, slug=match['slug'], role=_role_of(pid), first_seen=now, last_seen=now,
                    peak_bytes=peak, mem_total_gib=total, mem_available_gib_at_start=available,
                    threads=_thread_env(pid) if pid else {},
                )
            rec.peak_bytes = max(rec.peak_bytes, peak)
            rec.last_seen = now
        for unit in [u for u in live if u not in seen]:
            with out.open('a') as f:
                f.write(json.dumps(asdict(live.pop(unit))) + '\n')
        time.sleep(interval)


@dataclass(frozen=True)
class Gate:
    slug: str
    role: str
    legs: tuple[ScopeRecord, ...]

    @property
    def max_leg_gib(self) -> float:
        return max(leg.peak_bytes for leg in self.legs) / GIB

    @property
    def sum_legs_gib(self) -> float:
        return sum(leg.peak_bytes for leg in self.legs) / GIB


def group_gates(records: list[ScopeRecord], gap: float = GATE_GAP_SECS) -> list[Gate]:
    gates: list[Gate] = []
    for key in sorted({(r.slug, r.role) for r in records}):
        legs = sorted((r for r in records if (r.slug, r.role) == key), key=lambda r: r.first_seen)
        current: list[ScopeRecord] = []
        last_end = 0.0
        for leg in legs:
            if current and leg.first_seen - last_end > gap:
                gates.append(Gate(*key, tuple(current)))
                current = []
            current.append(leg)
            last_end = max(last_end, leg.last_seen)
        if current:
            gates.append(Gate(*key, tuple(current)))
    return gates


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


@dataclass(frozen=True)
class ProjectSummary:
    slug: str
    gates: int
    p50: float
    p90: float
    max: float
    sum_legs_max: float
    recommended_max_gib: int
    scale: float
    scaled_p90: float
    threads: dict[str, set[str]]


def summarise(gates: list[Gate], scales: dict[str, float]) -> list[ProjectSummary]:
    out: list[ProjectSummary] = []
    for slug in sorted({g.slug for g in gates if g.role == 'merge'}):
        mine = [g for g in gates if g.slug == slug and g.role == 'merge']
        peaks = [g.max_leg_gib for g in mine]
        p90 = percentile(peaks, 0.9)
        scale = scales.get(slug, 1.0)
        threads: dict[str, set[str]] = {}
        for gate in mine:
            for leg in gate.legs:
                for k, v in leg.threads.items():
                    threads.setdefault(k, set()).add(v)
        out.append(ProjectSummary(
            slug=slug, gates=len(mine), p50=statistics.median(peaks), p90=p90, max=max(peaks),
            sum_legs_max=max(g.sum_legs_gib for g in mine), recommended_max_gib=math.ceil(p90 * RECOMMEND_HEADROOM),
            scale=scale, scaled_p90=p90 * scale, threads=threads,
        ))
    return out


def split_recommendation(summaries: list[ProjectSummary]) -> str:
    total = sum(s.scaled_p90 for s in summaries)
    verdict = '16/16' if total < SPLIT_SUM_LIMIT_GIB else '8/8 for Reify'
    return f'sum of scaled p90 = {total:.1f} GiB (limit {SPLIT_SUM_LIMIT_GIB:.0f}) -> {verdict}'


def render_report(records: list[ScopeRecord], scales: dict[str, float]) -> str:
    summaries = summarise(group_gates(records), scales)
    if not summaries:
        return 'no merge-role gates recorded\n'
    baselines = [r.mem_available_gib_at_start for r in records if r.role == 'merge']
    lines = [
        '| project | gates | p50 GiB | p90 GiB | max GiB | sum-of-legs max | MemoryMax (p90x1.25) | scale | scaled p90 | thread env in force |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for s in summaries:
        env = '; '.join(f'{k}={",".join(sorted(v))}' for k, v in sorted(s.threads.items())) or 'unrecorded'
        lines.append(
            f'| {s.slug} | {s.gates} | {s.p50:.1f} | {s.p90:.1f} | {s.max:.1f} | {s.sum_legs_max:.1f} '
            f'| {s.recommended_max_gib}G | {s.scale:.3g} | {s.scaled_p90:.1f} | {env} |'
        )
    lines += [
        '',
        split_recommendation(summaries),
        f'MemAvailable at scope start (merge legs): min {min(baselines):.0f} / median {statistics.median(baselines):.0f} '
        f'/ max {max(baselines):.0f} GiB of {records[0].mem_total_gib:.0f} GiB total',
    ]
    return '\n'.join(lines) + '\n'


def load_records(path: Path, since: float) -> list[ScopeRecord]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return [ScopeRecord(**row) for row in rows if row['first_seen'] >= since]


def _parse_scales(items: list[str]) -> dict[str, float]:
    scales = dict(DEFAULT_SCALES)
    for item in items:
        slug, _, factor = item.partition('=')
        scales[slug] = float(factor)
    return scales


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_poll = sub.add_parser('poll')
    p_poll.add_argument('--out', type=Path, required=True)
    p_poll.add_argument('--interval', type=float, default=5.0)
    p_poll.add_argument('--scope-glob', default=SCOPE_GLOB.format(uid=os.getuid()))
    p_rep = sub.add_parser('report')
    p_rep.add_argument('records', type=Path)
    p_rep.add_argument('--since', default='1970-01-01', help='YYYY-MM-DD (UTC) lower bound on scope start')
    p_rep.add_argument('--scale', action='append', default=[], metavar='SLUG=FACTOR')
    args = parser.parse_args(argv)
    if args.cmd == 'poll':
        poll(args.out, args.interval, args.scope_glob)
        return 0
    since = time.mktime(time.strptime(args.since, '%Y-%m-%d')) - time.timezone
    sys.stdout.write(render_report(load_records(args.records, since), _parse_scales(args.scale)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
