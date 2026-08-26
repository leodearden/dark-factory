#!/usr/bin/env python3
"""Offline replay study: measure deep speculative-stack verify behaviour from history.

Motivation (see plans/deep-speculative-verify-ahead-analysis-2026-07-22.md and its
2026-07-23 adversarial review): reify's event record contains essentially no true
>=3-item cumulative verifies — the pipeline never builds deeper than K=2 and the
speculation probe only relabels 2-item trees — so P(pass|depth>=2), the causal
verify-time growth (epsilon), and same-tree flake R at depth are all UNMEASURED.
This study measures them by retrodiction: reconstruct historical backlog episodes
from a runs.db snapshot, rebuild the adjacent stacks a deep-speculation mechanism
would have built (merge landed second-parents in land order, truncating at the
first file conflict), and run the real verify suite on each tree in an offline
clone. Within-episode controls (singles, duplicate runs) cancel toolchain drift.

Phases, in global priority order (early termination still yields a complete curve):
  A: full adjacent stack per episode (capped at STACK_CAP items)  -> P(pass|d)
  S: single I0 on the same base (every 2nd episode)               -> drift control
  B: 3-item prefix stack (episodes where A built >=4)             -> practical d=2
  C: duplicate re-run of A's exact tree (first DUP_CAP episodes)  -> same-tree flake

Everything runs against a dedicated clone (marker file required — this script
refuses to operate on a repo without it). The live repo and DB are never touched:
the DB is a snapshot copy, and all git surgery happens in the clone.

Usage:
  replay_deep_stack_study.py --db SNAPSHOT.db --repo CLONE_DIR --out STUDY_DIR --plan
  replay_deep_stack_study.py --db SNAPSHOT.db --repo CLONE_DIR --out STUDY_DIR --run

Results append to STUDY_DIR/results.jsonl (one JSON object per verify run);
re-invocation skips already-completed job ids, so the study is resumable.
Run the unit under SCHED_IDLE (systemd -p CPUSchedulingPolicy=idle) — the
script does not set scheduling policy itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import resource
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

CLONE_MARKER = '.replay-study-clone'
STACK_CAP = 6          # max items in a phase-A stack (queue rarely offers more)
DUP_CAP = 25           # max phase-C duplicate runs
SINGLE_EVERY = 2       # phase-S sampling: every Nth episode
VERIFY_TIMEOUT_S = 14400  # generous: SCHED_IDLE stretches wall time
# --scope all + DF_VERIFY_ROLE=merge: replicate the PRODUCTION MERGE verify
# (hooks/pre-merge-commit uses --scope all; the merge pipeline injects
# DF_VERIFY_ROLE=merge => profile=both). --scope branch is unusable here: every
# stack tip is already an ancestor of clone-time main, so branch scope resolves
# EMPTY and verify.sh exits 0 having run nothing ("nothing to verify") — this
# silently no-op'd 41/55 phase-A runs in the v1 (2026-07-23) results.
# NO --include-infra (v3, deliberate deviation from the production merge
# verify): tests/infra/run_all.sh runs BEFORE the cargo phases and
# test_prd_gate_corpus.sh fails deterministically on historical trees in the
# offline clone (environment-sensitive verdict assertions; 87/88 other infra
# tests pass), aborting verify before any Rust test executes — all 8 v2 runs
# were infra-blocked with zero workspace-test signal. The infra suite tests
# harness plumbing, is invariant to stacked CODE content, and so cannot carry
# depth signal; excluding it removes a constant false-fail floor from P(pass|d).
VERIFY_CMD = ['./scripts/verify.sh', 'test', '--scope', 'all']
VERIFY_ENV = {'DF_VERIFY_ROLE': 'merge'}
SCRIPT_VERSION = 5
# Files removed from the (disposable) study checkout after each stack build.
# Dropping --include-infra was not enough: each tree runs ITS OWN era's
# verify.sh, and some eras run tests/infra/run_all.sh unconditionally — then
# fail on env-sensitive infra assertions BEFORE any cargo test executes
# (3 v3 A-runs censored this way, a different infra test than v2's). Every
# era guards the step with `test -f tests/infra/run_all.sh`, so removing the
# file censors the infra suite structurally, era-proof.
STRIP_TREE_FILES = ['tests/infra/run_all.sh']
# Stale cross-era artifacts in the SHARED study target dir that production
# per-tree worktrees never see. structure_instance_e2e prefers
# target/release/reify when present (in production it never is — task 4390
# scoped reify-cli out of the release pass — so the test falls back to the
# debug binary the workspace pass just built fresh). An old-era tree's verify
# built the release CLI here, and every later tree whose golden disagreed with
# that binary's era failed the golden e2e — 54 v4 S/B/C runs censored,
# including 23/25 same-content C-vs-A flips (the tell). Deleting it per-job
# restores the production-identical fallback path.
STRIP_TARGET_FILES = ['target/release/reify']
NOOP_CPU_FLOOR_S = 60  # a "pass" burning less CPU than this is a suspect no-op
SCRUB_ENV = [k for k in os.environ if k.startswith('CLAUDE')]


def log(msg: str) -> None:
    print(f'[{dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}] {msg}',
          flush=True)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['git', '-C', str(repo), *args],
                          capture_output=True, text=True, check=check)


# ── Episode reconstruction ──────────────────────────────────────────────────

def load_spans(db_path: Path, since_days: int) -> list[tuple[str, dt.datetime, dt.datetime]]:
    """True queue-residency spans: each merge_queued paired with that task's NEXT
    merge_finalized of ANY terminal state (done/blocked/already_merged/...).

    Pairing per queued event — not first-queued -> first-done — because blocked ->
    requeue churn otherwise inflates residency ~10x (median 65 min vs true ~15 min)
    and collapses the whole window into one mega-episode.
    """
    con = sqlite3.connect(str(db_path))
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=since_days)
    open_q: dict[str, dt.datetime] = {}
    spans: list[tuple[str, dt.datetime, dt.datetime]] = []
    rows = con.execute(
        "SELECT event_type, timestamp, task_id FROM events "
        "WHERE event_type IN ('merge_queued','merge_finalized') ORDER BY id")
    for etype, ts, task_id in rows:
        if task_id is None:
            continue
        t = dt.datetime.fromisoformat(ts)
        tid = str(task_id)
        if etype == 'merge_queued':
            open_q.setdefault(tid, t)
        elif tid in open_q:
            start = open_q.pop(tid)
            if start >= cutoff and t > start:
                spans.append((tid, start, t))
    spans.sort(key=lambda r: r[1])
    return spans


def find_episodes(spans: list[tuple[str, dt.datetime, dt.datetime]]) -> list[list[str]]:
    """Backlog episodes: maximal co-residency chains that reach >=3 simultaneously
    co-queued distinct tasks (matching the analysis doc's episode definition).

    Returns each episode's distinct member tasks in arrival order — the order an
    adjacent-stack mechanism would have stacked them.
    """
    events: list[tuple[dt.datetime, int, str]] = []
    for tid, q, d in spans:
        events.append((q, +1, tid))
        events.append((d, -1, tid))
    events.sort(key=lambda e: (e[0], e[1]))
    episodes: list[list[str]] = []
    active: dict[str, int] = {}
    first_seen: dict[str, dt.datetime] = {}
    peak = 0
    for t, delta, tid in events:
        if delta > 0:
            active[tid] = active.get(tid, 0) + 1
            if len(active) >= 2:
                first_seen.setdefault(tid, t)
                for other in active:
                    first_seen.setdefault(other, t)
            peak = max(peak, len(active))
        else:
            n = active.get(tid, 0) - 1
            if n <= 0:
                active.pop(tid, None)
            else:
                active[tid] = n
            if not active:
                if peak >= 3 and len(first_seen) >= 3:
                    episodes.append(sorted(first_seen, key=lambda k: first_seen[k]))
                first_seen = {}
                peak = 0
    if peak >= 3 and len(first_seen) >= 3:
        episodes.append(sorted(first_seen, key=lambda k: first_seen[k]))
    return episodes


def merge_commit_index(repo: Path) -> dict[str, str]:
    """task label -> OLDEST 'Merge task/<label> into main' commit on main.

    Single history walk (vs one git-log --grep per task): oldest match is the
    first genuine landing, sidestepping pipeline-duplicate re-merges.
    """
    r = git(repo, 'log', 'main', '--format=%H %s')
    idx: dict[str, str] = {}
    for line in r.stdout.splitlines():
        sha, _, subj = line.partition(' ')
        m = re.match(r'Merge task/(\S+) into main', subj)
        if m:
            idx[m.group(1)] = sha  # newest-first walk: last write wins = oldest
    return idx


# ── Job planning ────────────────────────────────────────────────────────────

def plan_jobs(repo: Path, db_path: Path, since_days: int) -> list[dict]:
    spans = load_spans(db_path, since_days)
    episodes = find_episodes(spans)
    log(f'{len(spans)} queue-residency spans in window; '
        f'{len(episodes)} episodes reaching >=3 co-queued')
    idx = merge_commit_index(repo)
    plans = []
    for ep in episodes:
        resolved = [(tid, idx[tid]) for tid in ep if tid in idx][:STACK_CAP]
        if len(resolved) < 3:
            continue
        base = git(repo, 'rev-parse', f'{resolved[0][1]}^1').stdout.strip()
        tips = []
        for tid, mc in resolved:
            tip = git(repo, 'rev-parse', f'{mc}^2', check=False).stdout.strip()
            if tip:
                tips.append((tid, tip))
        if len(tips) < 3:
            continue
        ep_id = 'ep' + resolved[0][1][:8]
        plans.append({'episode': ep_id, 'base': base, 'tips': tips})
    jobs: list[dict] = []
    for phase in ('A', 'S', 'B', 'C'):
        for i, p in enumerate(plans):
            n = len(p['tips'])
            if phase == 'S' and i % SINGLE_EVERY:
                continue
            if phase == 'B' and n < 4:
                continue
            if phase == 'C' and len([j for j in jobs if j['phase'] == 'C']) >= DUP_CAP:
                break
            count = {'A': n, 'S': 1, 'B': 3, 'C': n}[phase]
            jobs.append({
                'job_id': f'{p["episode"]}-{phase}',
                'phase': phase,
                'episode': p['episode'],
                'base': p['base'],
                'tips': p['tips'][:count],
            })
    return jobs


# ── Execution ───────────────────────────────────────────────────────────────

def build_stack(repo: Path, base: str, tips: list[list[str] | tuple[str, str]]) -> tuple[str, int]:
    """Checkout base, merge tips in order, truncate at first conflict.

    Returns (tree_tip_sha, items_merged).
    """
    git(repo, 'merge', '--abort', check=False)
    git(repo, 'reset', '--hard', check=False)
    git(repo, 'clean', '-fd', '-e', 'target', '-e', 'node_modules',
        '-e', CLONE_MARKER, check=False)
    git(repo, 'checkout', '--detach', base)
    merged = 0
    for tid, tip in tips:
        r = git(repo, 'merge', '--no-ff', '--no-edit',
                '-m', f'replay-study merge task/{tid}', tip, check=False)
        if r.returncode != 0:
            git(repo, 'merge', '--abort', check=False)
            break
        merged += 1
    for rel in STRIP_TREE_FILES + STRIP_TARGET_FILES:
        (repo / rel).unlink(missing_ok=True)
    return git(repo, 'rev-parse', 'HEAD').stdout.strip(), merged


def run_verify(repo: Path, log_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
    env.update(VERIFY_ENV)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, 'w') as lf:
        try:
            proc = subprocess.run(VERIFY_CMD, cwd=str(repo), env=env,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=VERIFY_TIMEOUT_S)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_user = after.ru_utime - before.ru_utime
    cpu_sys = after.ru_stime - before.ru_stime
    return {
        'exit_code': exit_code,
        'passed': exit_code == 0,
        'timed_out': timed_out,
        'wall_s': round(time.monotonic() - t0, 1),
        'cpu_user_s': round(cpu_user, 1),
        'cpu_sys_s': round(cpu_sys, 1),
        # A "pass" that burned almost no CPU verified nothing (e.g. an empty
        # scope resolution) — flag it so aggregation can never mistake a
        # no-op exit 0 for a green verify.
        'suspect_noop': exit_code == 0 and (cpu_user + cpu_sys) < NOOP_CPU_FLOOR_S,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', required=True, type=Path, help='runs.db SNAPSHOT (never the live db)')
    ap.add_argument('--repo', required=True, type=Path, help='dedicated study clone of reify')
    ap.add_argument('--out', required=True, type=Path, help='study output dir')
    ap.add_argument('--since-days', type=int, default=30)
    ap.add_argument('--plan', action='store_true', help='print the job plan and exit')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--max-jobs', type=int, default=0, help='stop after N jobs this invocation')
    args = ap.parse_args()

    if not (args.repo / CLONE_MARKER).exists():
        log(f'REFUSING: {args.repo} lacks {CLONE_MARKER} — never run against a live repo')
        return 2

    jobs = plan_jobs(args.repo, args.db, args.since_days)
    if args.plan or not args.run:
        for j in jobs:
            print(json.dumps({**j, 'tips': [t for t, _ in j['tips']]}))
        counts = {p: sum(1 for j in jobs if j['phase'] == p) for p in 'ASBC'}
        log(f'{len(jobs)} jobs planned: '
            + ', '.join(f'{v} {k}' for k, v in counts.items()))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    results_path = args.out / 'results.jsonl'
    completed = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                completed.add(json.loads(line)['job_id'])
            except (json.JSONDecodeError, KeyError):
                continue
    log(f'{len(jobs)} jobs planned, {len(completed)} already complete')

    ran = 0
    for j in jobs:
        if j['job_id'] in completed:
            continue
        if args.max_jobs and ran >= args.max_jobs:
            break
        log(f'{j["job_id"]}: building {len(j["tips"])}-item stack on {j["base"][:10]}')
        tree, merged = build_stack(args.repo, j['base'], j['tips'])
        # Tree OBJECT hash (not commit sha): rebuilt merges get fresh commit
        # shas from timestamps, but identical content — phase C same-tree
        # comparison must key on this.
        tree_obj = git(args.repo, 'rev-parse', 'HEAD^{tree}').stdout.strip()
        rec = {**j, 'tips': [t for t, _ in j['tips']], 'tree': tree,
               'tree_object': tree_obj, 'script_version': SCRIPT_VERSION,
               'items_merged': merged, 'started_at': dt.datetime.now(dt.UTC).isoformat()}
        if merged == 0:
            rec.update(exit_code=None, passed=None, skipped='no_items_merged')
        else:
            log(f'{j["job_id"]}: verifying tree {tree[:10]} ({merged} items merged)')
            rec.update(run_verify(args.repo, args.out / 'logs' / f'{j["job_id"]}.log'))
            log(f'{j["job_id"]}: passed={rec["passed"]} wall={rec["wall_s"]}s '
                f'cpu={rec["cpu_user_s"] + rec["cpu_sys_s"]}s')
        with open(results_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        ran += 1
    log(f'invocation done: {ran} jobs run')
    return 0


if __name__ == '__main__':
    sys.exit(main())
