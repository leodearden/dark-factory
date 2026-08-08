#!/usr/bin/env python3
"""Mint the fable-architect-trial-v2 curated hard fixture pool (β1, task 3631).

Two modes:

``--census``
    Re-run the architect-exhaustion census across the three source checkouts'
    ``data/orchestrator/runs.db``, enrich each candidate from its
    ``.taskmaster/tasks/tasks.db``, resolve merge-SHA availability and the
    eval baseline commit, and print one row per candidate. This is the
    reproducibility fix for v1, whose sampling driver was gitignored and whose
    pool therefore could not be re-derived.

``--mint``
    Read the committed ``_meta/curation.json``, drive the canonical
    ``task_sampler`` minting pipeline (``capture_reference`` /
    ``build_fixture_record`` / ``pin_eval_branch``) over its ``include`` rows,
    overlay the v2 ceilings, and write the fixture JSONs plus the generated
    ``CURATION.md``.

Nothing in this script writes to ``orchestrator/src/orchestrator/evals/tasks/``
— the standing eval corpus is out of scope for β1 and stays byte-unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three source checkouts, keyed by the census project id. Both databases
# are read read-only; the checkouts are read via git only.
SOURCE_CHECKOUTS: dict[str, str] = {
    'reify': '/home/leo/src/reify',
    'dark_factory': '/home/leo/src/dark-factory',
    'know_live': '/home/leo/src/know-live',
}

# The recorded per-project distinct counts. The census asserts against these so
# a drifted db is caught loudly rather than silently re-scoping the pool.
EXPECTED_CENSUS_COUNTS: dict[str, int] = {
    'reify': 36, 'dark_factory': 4, 'know_live': 1,
}

# ---------------------------------------------------------------------------
# The census filter
# ---------------------------------------------------------------------------
#
# The 121-turn clause binds ONLY the max_turns arm. Budget exhaustion
# terminates at an arbitrary turn count (know-live 543 exhausted its budget at
# 113 turns and is in the recorded census), so gating it on 121 too would
# yield 23 candidates instead of the recorded 41. 121 is know-live's
# production ``max_turns.architect: 120`` ceiling plus one.
CENSUS_SQL = """\
SELECT DISTINCT task_id FROM events
WHERE event_type='invocation_end' AND role='architect'
  AND ( (json_extract(data,'$.subtype')='error_max_turns'
         AND json_extract(data,'$.turns')=121)
     OR  json_extract(data,'$.subtype')='error_max_budget_usd' )\
"""

# Same predicate, projecting the wall-clock instead of the id — the evidence
# behind the ``timeout_minutes`` derivation.
_CENSUS_DURATION_SQL = """\
SELECT duration_ms FROM events
WHERE event_type='invocation_end' AND role='architect'
  AND duration_ms IS NOT NULL
  AND ( (json_extract(data,'$.subtype')='error_max_turns'
         AND json_extract(data,'$.turns')=121)
     OR  json_extract(data,'$.subtype')='error_max_budget_usd' )\
"""

# Earliest architect invocation for a task — the anchor for baseline rung 3.
_FIRST_INVOCATION_SQL = """\
SELECT MIN(timestamp) FROM events
WHERE task_id = ? AND role='architect'\
"""


class BaselineUnresolved(RuntimeError):
    """No rung of the baseline ladder produced a commit.

    Raised rather than returning an empty SHA: ``run_architect_eval`` requires
    ``task['pre_task_commit']`` to create its eval worktree, so an empty value
    would surface as an opaque failure deep inside the runner instead of here,
    at the point where the cause is known.
    """


def connect_ro(db_path: Path | str) -> sqlite3.Connection:
    """Open *db_path* read-only, so a live orchestrator db is never locked.

    Mirrors ``task_sampler.enrich_candidates_from_task_db``'s access pattern.
    """
    conn = sqlite3.connect(f'file:{Path(db_path)}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def census_task_ids(runs_db: Path | str) -> list[str]:
    """Return the sorted distinct architect-exhaustion task ids in *runs_db*."""
    conn = connect_ro(runs_db)
    try:
        return sorted(str(row[0]) for row in conn.execute(CENSUS_SQL) if row[0])
    finally:
        conn.close()


def census_durations_ms(runs_db: Path | str) -> list[int]:
    """Return every ``duration_ms`` over the census population in *runs_db*."""
    conn = connect_ro(runs_db)
    try:
        return [int(row[0]) for row in conn.execute(_CENSUS_DURATION_SQL)]
    finally:
        conn.close()


def first_architect_invocation_ts(runs_db: Path | str, task_id: str) -> str | None:
    """Return the earliest architect-invocation timestamp for *task_id*."""
    conn = connect_ro(runs_db)
    try:
        row = conn.execute(_FIRST_INVOCATION_SQL, (task_id,)).fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path | str) -> str:
    """Run a git command in *cwd*, returning stripped stdout ('' on failure).

    Read-only queries only. A non-zero exit means "no answer" for every call
    site here (an unmatched grep, a revision that does not resolve), which the
    callers turn into an explicit fallback — never a silent wrong answer.
    """
    proc = subprocess.run(
        ['git', *args], cwd=str(cwd), capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ''


_MERGE_SUBJECT = 'Merge task/{task_id} into main'


def find_merge_sha(repo_root: Path | str, task_id: str) -> str | None:
    """Return the single ``Merge task/<id> into main`` SHA, else ``None``.

    ``None`` means planRate-only: either the task landed SPLIT/direct (no merge
    commit at all) or several merges carry the same id, in which case picking
    one would silently invent a reference for the judge to grade against.

    The grep is anchored on the full subject so ``Merge task/8030 into main``
    cannot answer a query for task ``803``.
    """
    subject = _MERGE_SUBJECT.format(task_id=task_id)
    out = _git(
        ['log', '--all', f'--grep=^{re.escape(subject)}$', '--extended-regexp',
         '--pretty=%H'],
        repo_root,
    )
    shas = [line.strip() for line in out.splitlines() if line.strip()]
    return shas[0] if len(shas) == 1 else None


def _autocommit_start_sha(repo_root: Path | str, task_id: str) -> str | None:
    """Return the EARLIEST ``set_task_status(<id>=in-progress)`` SHA on main.

    A task can be re-started, so the baseline is main at its FIRST start.
    Only the ``in-progress`` transition marks task start — ``done`` / ``blocked``
    / ``pending`` transitions are later or unrelated states.
    """
    subject = f'chore(tasks): auto-commit after set_task_status({task_id}=in-progress)'
    out = _git(
        ['log', 'main', f'--grep=^{re.escape(subject)}$', '--extended-regexp',
         '--pretty=%H'],
        repo_root,
    )
    shas = [line.strip() for line in out.splitlines() if line.strip()]
    # git log is newest-first; the earliest start is the last line.
    return shas[-1] if shas else None


def resolve_baseline(
    repo_root: Path | str, task_id: str, first_invocation_ts: str | None,
) -> tuple[str, str]:
    """Resolve the eval baseline commit for *task_id*; return ``(sha, rung)``.

    Three rungs, strongest provenance first:

    1. ``merge_first_parent`` — ``M^1`` of the task's single merge commit.
       Identical to what ``resolve_task_commits_from_merge`` returns, i.e.
       prior main.
    2. ``status_autocommit`` — the taskmaster ``set_task_status(<id>=in-progress)``
       auto-commit on main. A real, timestamped anchor at task start.
    3. ``timestamp_walk`` — the newest main commit strictly before the task's
       first architect invocation.

    The rung is returned alongside the SHA so the caller can stamp
    ``provenance.baseline_source`` and the weaker provenance stays visible
    rather than masquerading as merge-derived.

    Raises :class:`BaselineUnresolved` if every rung fails — never an empty SHA.
    """
    merge_sha = find_merge_sha(repo_root, task_id)
    if merge_sha:
        pre = _git(['rev-parse', f'{merge_sha}^1'], repo_root)
        if pre:
            return pre, 'merge_first_parent'

    start = _autocommit_start_sha(repo_root, task_id)
    if start:
        return start, 'status_autocommit'

    if first_invocation_ts:
        walked = _git(
            ['rev-list', '-n1', f'--before={first_invocation_ts}', 'main'],
            repo_root,
        )
        if walked:
            return walked, 'timestamp_walk'

    raise BaselineUnresolved(
        f'resolve_baseline: no baseline commit for task {task_id!r} in '
        f'{repo_root} — no single "Merge task/{task_id} into main", no '
        f'set_task_status({task_id}=in-progress) auto-commit on main, and no '
        f'main commit before first_invocation_ts={first_invocation_ts!r}. '
        f'A fixture with an empty pre_task_commit would fail inside '
        f'run_architect_eval; refusing to emit one.'
    )


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One census candidate, enriched for curation."""

    task_id: str
    project: str
    project_root: str
    title: str = ''
    description: str = ''
    status: str = ''
    complexity: str | None = None
    modules: list[str] = field(default_factory=list)
    merge_sha: str | None = None
    baseline_sha: str = ''
    baseline_source: str = ''

    @property
    def brief_chars(self) -> int:
        """Length of the brief. Recorded as EVIDENCE for the curation
        judgement, never as the rule — the criterion is whether the brief
        states an implementable goal."""
        return len(self.description or '')


def enrich_from_task_db(
    candidates: list[Candidate], db_path: Path | str,
) -> list[Candidate]:
    """Fill title / description / status / complexity / modules in place.

    Mirrors ``task_sampler.enrich_candidates_from_task_db``'s read-only access
    and best-effort degradation, extended to also read ``status`` (the
    cancelled / pending candidates need their own explicit curation decision).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f'WARNING: no task db at {db_path}; candidates stay stubs',
              file=sys.stderr)
        return candidates

    conn = connect_ro(db_path)
    try:
        rows = {
            str(r['id']): r for r in conn.execute(
                'SELECT id, title, description, status, metadata FROM tasks'
            )
        }
    finally:
        conn.close()

    for cand in candidates:
        row = rows.get(str(cand.task_id))
        if row is None:
            continue
        cand.title = row['title'] or ''
        cand.description = row['description'] or ''
        cand.status = row['status'] or ''
        raw_meta = row['metadata']
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except (TypeError, ValueError):
                meta = {}
            if isinstance(meta, dict):
                cand.complexity = meta.get('complexity') or cand.complexity
                mods = meta.get('modules')
                if isinstance(mods, list):
                    cand.modules = [str(m) for m in mods]
    return candidates


def collect_candidates(project: str, root: Path) -> list[Candidate]:
    """Census + enrich + resolve merge/baseline for one source checkout."""
    runs_db = root / 'data' / 'orchestrator' / 'runs.db'
    candidates = [
        Candidate(task_id=tid, project=project, project_root=str(root))
        for tid in census_task_ids(runs_db)
    ]
    enrich_from_task_db(candidates, root / '.taskmaster' / 'tasks' / 'tasks.db')
    for cand in candidates:
        cand.merge_sha = find_merge_sha(root, cand.task_id)
        ts = first_architect_invocation_ts(runs_db, cand.task_id)
        try:
            cand.baseline_sha, cand.baseline_source = resolve_baseline(
                root, cand.task_id, ts,
            )
        except BaselineUnresolved as exc:
            # Recorded, not swallowed: --census reports it and the row cannot
            # be curated as an include until a baseline exists.
            cand.baseline_source = f'UNRESOLVED ({exc.__class__.__name__})'
    return candidates


# ---------------------------------------------------------------------------
# --census
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def run_census(strict: bool = True) -> int:
    """Print the census table + per-project counts; return a process exit code."""
    all_candidates: list[Candidate] = []
    counts: dict[str, int] = {}
    durations: list[int] = []

    for project, root_str in SOURCE_CHECKOUTS.items():
        root = Path(root_str)
        if not root.exists():
            print(f'ERROR: source checkout {root} not found', file=sys.stderr)
            return 2
        cands = collect_candidates(project, root)
        counts[project] = len(cands)
        durations.extend(census_durations_ms(root / 'data' / 'orchestrator' / 'runs.db'))
        all_candidates.extend(cands)

    header = (f'{"project":<14}{"task":<7}{"status":<12}{"brief":>6}  '
              f'{"merge_sha":<12}{"baseline_rung":<20}title')
    print(header)
    print('-' * len(header))
    for c in all_candidates:
        merge = (c.merge_sha or '')[:10] or '-'
        print(f'{c.project:<14}{c.task_id:<7}{c.status or "?":<12}'
              f'{c.brief_chars:>6}  {merge:<12}{c.baseline_source:<20}'
              f'{(c.title or "")[:48]}')

    print()
    for project, n in counts.items():
        expected = EXPECTED_CENSUS_COUNTS[project]
        flag = 'OK' if n == expected else f'DRIFT (expected {expected})'
        print(f'{project:<14}{n:>4}  {flag}')
    total = sum(counts.values())
    print(f'{"TOTAL":<14}{total:>4}')

    referenced = sum(1 for c in all_candidates if c.merge_sha)
    print(f'\nmerge-SHA available (referenced): {referenced}/{total}; '
          f'SPLIT / planRate-only: {total - referenced}/{total}')

    if durations:
        minutes = [d / 60_000 for d in durations]
        print(f'\nduration over the census population (events.duration_ms), '
              f'n={len(minutes)}: '
              f'max={max(minutes):.1f}m p95={_percentile(minutes, 95):.1f}m '
              f'p50={_percentile(minutes, 50):.1f}m')

    if strict and counts != EXPECTED_CENSUS_COUNTS:
        print(f'\nCENSUS DRIFT: got {counts}, expected {EXPECTED_CENSUS_COUNTS}. '
              f'The recorded pool is no longer reproducible from these dbs; '
              f'refusing to silently re-scope it.', file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--census', action='store_true',
                      help='Re-run the architect-exhaustion census and print it')
    parser.add_argument('--no-strict', action='store_true',
                        help='Report census drift without a non-zero exit')
    args = parser.parse_args(argv)

    if args.census:
        return run_census(strict=not args.no_strict)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
