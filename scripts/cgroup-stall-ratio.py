#!/usr/bin/env python3
"""Who is actually starved for CPU: runqueue-wait ratio per (cgroup bucket, family, nice).

Reads /proc/<pid>/schedstat twice over a window and reports, per class, the CPU it
ran (run-cores), the CPU it wanted but waited for (wait-cores) and the stall ratio
wait/(run+wait). This is the honest "who is hurt" signal on a saturated host: PSI
`some` saturates near 50-75 and cannot say WHICH work is stalled; loadavg is not
demand. Used by the 2026-09-08 load-tuning session (agents at nice 10 inside
orchestrator-dark-factory.service stalled 89%, merge xdist workers 51%, task
xdist workers 57%) and by the follow-up gate that checks per-role cgroup
CPUWeight on verify scopes actually moves these numbers.

Buckets: df (orchestrator-dark-factory.service), reify (orchestrator-reify.service),
verify-scope:<project> (df-verify-<project>-* transient scopes), other.
Families: xdist worker, pytest master, claude-cli, rustc, cargo test binary, cargo,
orch-main, git, other. The verify ROLE is inferred from nice (5 merge / 15 task /
19 background, per shared/verify_admission.py::_NICE_TIERS), which holds both inside
a unit and inside a scope.

Read-only. Stdlib only. Prints a table; with --json also a trailing single-line JSON
of headline classes for a before_done note.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

ROLE_BY_NICE = {5: 'merge', 15: 'task', 19: 'background'}


def snap() -> dict[str, tuple[int, int, str, int, str]]:
    d = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cgroup') as f:
                cg = f.read().strip()
            with open(f'/proc/{pid}/schedstat') as f:
                run_ns, wait_ns = (int(x) for x in f.read().split()[:2])
            with open(f'/proc/{pid}/stat') as f:
                st = f.read()
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cl = f.read().replace(b'\0', b' ').decode('utf8', 'replace')
        except OSError:
            continue
        nice = int(st[st.rfind(')') + 2:].split()[16])
        d[pid] = (run_ns, wait_ns, cg, nice, cl)
    return d


def bucket(cg: str) -> str:
    if 'orchestrator-dark-factory.service' in cg:
        return 'df'
    if 'orchestrator-reify.service' in cg:
        return 'reify'
    m = re.search(r'df-verify-([A-Za-z0-9_]+)-', cg)
    if m:
        return f'verify-scope:{m.group(1)}'
    return 'other'


def family(cl: str) -> str:
    if 'exec(eval(sys.stdin.readline()))' in cl:
        return 'xdist-worker'
    if re.search(r'(^| )claude( |$)|/claude ', cl):
        return 'claude-cli'
    if 'orchestrator run --config' in cl:
        return 'orch-main'
    if re.search(r'rustc|rust-lld', cl):
        return 'rustc'
    if re.search(r'target/(debug|release)/deps/', cl):
        return 'test-binary'
    if 'cargo' in cl:
        return 'cargo'
    if 'pytest' in cl:
        return 'pytest-master'
    if re.search(r'^git |/git |git-', cl):
        return 'git'
    return 'other'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=float, default=20.0, help='seconds between the two samples')
    ap.add_argument('--json', action='store_true', help='also print a trailing single-line JSON of headline classes')
    ap.add_argument('--min-cores', type=float, default=0.05, help='hide rows below this run+wait')
    args = ap.parse_args()

    a = snap()
    t0 = time.monotonic()
    time.sleep(args.window)
    b = snap()
    dt = time.monotonic() - t0

    run: collections.Counter = collections.Counter()
    wait: collections.Counter = collections.Counter()
    n: collections.Counter = collections.Counter()
    for pid, (r2, w2, cg, nice, cl) in b.items():
        if pid not in a:
            continue
        r1, w1 = a[pid][0], a[pid][1]
        key = (bucket(cg), family(cl), nice)
        run[key] += r2 - r1
        wait[key] += w2 - w1
        n[key] += 1

    try:
        with open('/proc/pressure/cpu') as f:
            psi = f.readline().split()[1].split('=')[1]
        with open('/proc/stat') as f:
            runq = next(line for line in f if line.startswith('procs_running')).split()[1]
    except (OSError, StopIteration, IndexError):
        psi, runq = '?', '?'

    print(f'window {dt:.1f}s  psi cpu some avg10={psi}  procs_running={runq}')
    print(f'{"bucket":22s} {"family":14s} {"nice":>4s} {"role":10s} {"procs":>5s} {"run":>7s} {"wait":>7s} {"stall%":>7s}')
    rows = sorted(run, key=lambda k: -(run[k] + wait[k]))
    for k in rows:
        tot = run[k] + wait[k]
        if tot / 1e9 / dt < args.min_cores:
            continue
        b_, f_, ni = k
        print(f'{b_:22s} {f_:14s} {ni:4d} {ROLE_BY_NICE.get(ni, "-"):10s} {n[k]:5d} '
              f'{run[k] / 1e9 / dt:7.2f} {wait[k] / 1e9 / dt:7.2f} {100 * wait[k] / tot if tot else 0:6.0f}%')

    if args.json:
        def cls(pred):
            r = sum(run[k] for k in run if pred(k))
            w = sum(wait[k] for k in wait if pred(k))
            tot = r + w
            return {'run_cores': round(r / 1e9 / dt, 2), 'wait_cores': round(w / 1e9 / dt, 2),
                    'stall_pct': round(100 * w / tot) if tot else None}

        def is_verify(bk: str, proj: str) -> bool:
            return bk == proj or bk == f'verify-scope:{proj if proj != "df" else "dark_factory"}'

        headline = {
            'psi_cpu_some_avg10': psi, 'procs_running': runq, 'window_s': round(dt, 1),
            'df_agents': cls(lambda k: k[0] == 'df' and k[1] == 'claude-cli'),
            'df_merge_workers': cls(lambda k: is_verify(k[0], 'df') and k[1] == 'xdist-worker' and k[2] == 5),
            'df_task_workers': cls(lambda k: is_verify(k[0], 'df') and k[1] == 'xdist-worker' and k[2] == 15),
            'df_background_workers': cls(lambda k: is_verify(k[0], 'df') and k[1] == 'xdist-worker' and k[2] == 19),
            'reify_agents': cls(lambda k: k[0] == 'reify' and k[1] == 'claude-cli'),
            'reify_merge_verify': cls(lambda k: is_verify(k[0], 'reify') and k[1] in ('rustc', 'test-binary', 'cargo') and k[2] == 5),
            'reify_task_verify': cls(lambda k: is_verify(k[0], 'reify') and k[1] in ('rustc', 'test-binary', 'cargo') and k[2] == 15),
        }
        print(json.dumps(headline))
    return 0


if __name__ == '__main__':
    sys.exit(main())
