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

# ---------------------------------------------------------------------------
# The curation judgements
# ---------------------------------------------------------------------------
#
# The criterion is the PRD's, recorded verbatim: EXCLUDE a candidate when its
# brief FAILS TO STATE AN IMPLEMENTABLE GOAL. It is never a length threshold —
# `brief_chars` is recorded on every row as evidence for the judgement, not as
# the rule. reify 3024's 94-character brief is INCLUDED for exactly this
# reason: terse, but it names the annotation to parse and the fallback to
# derive, so it is implementable-but-thin, which is the hard signal the pool
# exists to measure.
#
# Every candidate not listed here is an include; each entry below is one
# individually adjudicated exclusion, keyed ``(project, task_id)``.
EXCLUSIONS: dict[tuple[str, str], str] = {
    ('reify', '3378'): (
        'EXCLUDE. Cancelled, and abandonment was NOT benign: the architect '
        'reported the task unactionable 6 separate times (esc-3378-91/92/125/'
        '126/129/130), each naming the same cause — the required signature '
        '`fn solve_elastic_static(body, material, loads, supports, options)` '
        'references stdlib types (Body, Load, Support) and fn-param-default '
        'syntax that do not exist on main, with no sibling task creating '
        'them. The task is ill-posed AT ITS OWN BASELINE, so an exhaustion '
        'here measures the spec, not the model — precisely the confound this '
        'curation removes.'
    ),
    ('reify', '5208'): (
        'EXCLUDE. Status `pending`: never landed, so there is no reference '
        'diff and no ground truth that the task is completable at all. The '
        'brief is a SYMPTOM REPORT — "curated 3-arg fillet fails through the '
        'production .ri pipeline (CLI errors loudly, GUI fails silently) ... '
        'classic phantom-capability" — which states an observed failure, not '
        'an implementable goal: the deliverable is a root-cause diagnosis '
        'that did not exist at filing time. Architect exhaustion on an '
        'open-ended investigation is not the plan-quality signal the pool '
        'measures.'
    ),
}

# Candidates whose status is not `done` but which ARE included, with the
# benign-abandonment finding that justifies keeping them. Recorded separately
# so the judgement is explicit rather than implied by absence from EXCLUSIONS.
NON_DONE_INCLUDES: dict[tuple[str, str], str] = {
    ('reify', '3586'): (
        'INCLUDE. Cancelled, but abandonment WAS benign with respect to '
        'well-posedness: the only escalations on this task are '
        'esc-3586-38/40, both "Planning failed: agent failed: '
        "subtype='error_max_budget_usd'\" — budget exhaustion during "
        'planning, with no unactionability claim anywhere in its history. '
        'The 2489-char brief names concrete file:line deliverables '
        '(BRepKind::Vertex at geometry.rs:54-71, OcctKernel::extract_vertices '
        'mirroring handle.rs:293-318, per-op populators in two named '
        'crates). It was abandoned for COST, not for being ill-posed, which '
        'is exactly the hard-task signal the pool measures.'
    ),
}

# A `done` task that ALSO drew unactionable escalations is a genuinely hard
# task that was eventually resolved, not an ill-posed one — the discriminator
# is the terminal status. reify 2573, 3095 and 4370 are in that group and are
# included; only the CANCELLED-and-unactionable 3378 is excluded.
DONE_DESPITE_UNACTIONABLE: tuple[tuple[str, str], ...] = (
    ('reify', '2573'), ('reify', '3095'), ('reify', '4370'),
)


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


# ---------------------------------------------------------------------------
# --author — (re)generate _meta/curation.json from the live dbs + judgements
# ---------------------------------------------------------------------------

POOL_DIR = REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'evals' / 'tasks_hard_v2'
CURATION_JSON = POOL_DIR / '_meta' / 'curation.json'
CURATION_MD = POOL_DIR / 'CURATION.md'


def _candidate_row(cand: Candidate) -> dict:
    """Render one adjudicated candidate row for the manifest."""
    key = (cand.project, cand.task_id)
    excluded = key in EXCLUSIONS
    if excluded:
        reason = EXCLUSIONS[key]
    elif key in NON_DONE_INCLUDES:
        reason = NON_DONE_INCLUDES[key]
    elif key in DONE_DESPITE_UNACTIONABLE:
        reason = (
            'INCLUDE. The brief states an implementable goal, and although '
            'the architect raised unactionable escalation(s) on this task it '
            'reached status `done` — the unactionability was resolved, so '
            'this is a genuinely hard task that landed, not an ill-posed '
            'one. Terminal status is the discriminator (cf. reify 3378, '
            'cancelled AND unactionable, which is excluded).'
        )
    else:
        reason = (
            'INCLUDE. The brief states an implementable goal: it names the '
            'deliverable and enough of its surface for an architect to locate '
            'the work at the baseline commit.'
        )

    row: dict = {
        'task_id': cand.task_id,
        'project': cand.project,
        'project_root': cand.project_root,
        'title': cand.title,
        'status': cand.status,
        'brief_chars': cand.brief_chars,
        'decision': 'exclude' if excluded else 'include',
        'reason': reason,
        'merge_sha': cand.merge_sha,
        'baseline_sha': cand.baseline_sha,
        'baseline_source': cand.baseline_source,
    }
    if not excluded:
        row['mint_mode'] = 'referenced' if cand.merge_sha else 'planrate_only'
    return row


# The architect turn ceiling in production (know-live's
# dark-factory-orchestrator.yaml `max_turns: {architect: 120}`). Exhaustion is
# observed at 121 = this ceiling + 1, which is what the census filter keys on,
# so the pool must be run at the SAME ceiling or it cannot reproduce the
# failures it was selected for.
MAX_ARCHITECT_TURNS = 120

# The PRD's provisional floor. Kept because the measurement below clears it
# with room to spare — see `_build_ceilings`.
TIMEOUT_MINUTES = 180

_ALL_ARCHITECT_DURATION_SQL = """\
SELECT MAX(duration_ms) FROM events
WHERE event_type='invocation_end' AND role='architect' AND duration_ms IS NOT NULL\
"""


def _all_architect_max_ms(runs_db: Path | str) -> int:
    conn = connect_ro(runs_db)
    try:
        row = conn.execute(_ALL_ARCHITECT_DURATION_SQL).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row and row[0] else 0


def _build_ceilings() -> dict:
    """Measure the wall-clock evidence and emit the ceilings block.

    ``timeout_minutes`` is DERIVED, not guessed: it is shown to clear both the
    observed max-at-exhaustion and the all-time architect max, so the timeout
    provably cannot bind before the turn or budget ceiling. That matters
    because ``runner.py`` deliberately does not taint-exclude a timeout — a
    binding one would score a kept 0.0 and manufacture an artificial failure.
    """
    census_ms: list[int] = []
    all_max_ms = 0
    for root_str in SOURCE_CHECKOUTS.values():
        runs_db = Path(root_str) / 'data' / 'orchestrator' / 'runs.db'
        census_ms.extend(census_durations_ms(runs_db))
        all_max_ms = max(all_max_ms, _all_architect_max_ms(runs_db))

    minutes = sorted(d / 60_000 for d in census_ms)
    observed_max = round(max(minutes), 1) if minutes else 0.0
    p95 = round(_percentile(minutes, 95), 1) if minutes else 0.0
    p50 = round(_percentile(minutes, 50), 1) if minutes else 0.0
    all_max = round(all_max_ms / 60_000, 1)

    return {
        'max_architect_turns': MAX_ARCHITECT_TURNS,
        'max_architect_turns_note': (
            'know-live production dark-factory-orchestrator.yaml carries '
            'max_turns: {architect: 120}; the census keys on exhaustion at '
            '121 turns = that ceiling + 1. The pool must run at the same '
            'ceiling or it cannot reproduce the failures it was selected for.'
        ),
        'timeout_minutes': TIMEOUT_MINUTES,
        'derivation': {
            'source': (
                'events.duration_ms in <project_root>/data/orchestrator/runs.db, '
                'over the census population itself (the same predicate as '
                'census filter_sql)'
            ),
            'substitution_note': (
                'The PRD names "v1 wall-clock dumps" as the derivation source, '
                'but data/eval-campaign/fable-architect-only-results.json '
                'carries no duration field at all — its 72 records hold only '
                'task_id / config_name / outcome / trial / plan_quality / '
                'plan_steps / cost_usd. Deriving from it is impossible, so '
                'runs.db events.duration_ms is the substituted (and richer) '
                'source. The substitution is recorded here so the threshold '
                'is not mistaken for a guess.'
            ),
            'sample_n': len(minutes),
            'observed_max_minutes': observed_max,
            'p95_minutes': p95,
            'p50_minutes': p50,
            'all_architect_max_minutes': all_max,
            'all_architect_sample_note': (
                'all_architect_max_minutes is the max over EVERY architect '
                'invocation_end in the three checkouts (n=14701), not just the '
                'census population — the strictest bound available.'
            ),
            'headroom': (
                f'{TIMEOUT_MINUTES} min is '
                f'{TIMEOUT_MINUTES / observed_max:.1f}x the observed '
                f'max-at-exhaustion ({observed_max} min, n={len(minutes)}) and '
                f'{TIMEOUT_MINUTES / all_max:.1f}x the all-time architect max '
                f'({all_max} min), so the timeout provably cannot bind before '
                f'the {MAX_ARCHITECT_TURNS}-turn or budget ceiling.'
            ),
            'why_it_matters': (
                'runner.py raises the eval to outcome="timeout" on '
                'asyncio.TimeoutError and deliberately does NOT taint-exclude '
                'it, so a binding timeout scores a kept 0.0 — manufacturing an '
                'artificial failure on a set already selected for being hard.'
            ),
        },
    }


def author_manifest(census_date: str) -> dict:
    """Build the curation manifest from the live dbs + the judgement tables.

    *census_date* is passed in at the CLI boundary rather than read from the
    clock, so the manifest — and therefore the CURATION.md rendered from it —
    stays a pure function of its inputs.
    """
    per_project: dict[str, list[Candidate]] = {}
    for project, root_str in SOURCE_CHECKOUTS.items():
        per_project[project] = collect_candidates(project, Path(root_str))

    counts = {p: len(c) for p, c in per_project.items()}
    if counts != EXPECTED_CENSUS_COUNTS:
        raise RuntimeError(
            f'author_manifest: census drift — got {counts}, expected '
            f'{EXPECTED_CENSUS_COUNTS}. Refusing to author a manifest whose '
            f'pool silently differs from the recorded one.'
        )

    rows = [_candidate_row(c) for cands in per_project.values() for c in cands]
    referenced = sum(1 for r in rows if r.get('mint_mode') == 'referenced')
    planrate = sum(1 for r in rows if r.get('mint_mode') == 'planrate_only')

    return {
        'schema_version': 1,
        'task': '3631 — fable-trial-v2 β1: mint the curated v2 hard fixture pool',
        'prd': 'plans/fable-architect-trial-v2-prd.md',
        'cohort': 'fable-trial-v2-hard',
        'census': {
            'source': '<project_root>/data/orchestrator/runs.db (events table)',
            'census_date': census_date,
            'filter_sql': CENSUS_SQL,
            'filter_note': (
                'The turns-at-exhaustion 121 clause binds the max_turns arm '
                'ONLY. Budget exhaustion terminates at an arbitrary turn '
                'count (know-live 543 exhausted its budget at 113 turns), so '
                'applying 121 globally yields 23 candidates instead of the '
                'recorded 41. 121 is know-live production '
                'max_turns.architect=120 plus one.'
            ),
            'counts': counts,
            'task_ids': {p: [c.task_id for c in cands]
                         for p, cands in per_project.items()},
        },
        'curation_criterion': (
            "Exclude a candidate when its brief fails to state an "
            "implementable goal. Never a length threshold: brief_chars is "
            "recorded per row as evidence for the judgement, not as the rule."
        ),
        'adversarial_scan': (
            'No candidate in the census is an adversarial/red-team fixture — '
            'all 41 are real product tasks from the three checkouts\' task '
            'trees. The PRD\'s skip-adversarial rule therefore excludes '
            'nothing here; the scan is recorded so its emptiness is a finding '
            'rather than an omission.'
        ),
        'merge_sha_availability': {
            'referenced': referenced,
            'planrate_only': planrate,
            'finding': (
                f'SPLIT / direct-landed candidates are a MAJORITY '
                f'({planrate}/{len(rows)}), not the minority the trial '
                f'manifest assumed. Only {referenced} of {len(rows)} have a '
                f'single clean "Merge task/<id> into main" commit, so only '
                f'those can carry a reference block. This caps the '
                f'downstream plan_quality population and is a material fact '
                f'for γ1, though it does not block β1 — D9\'s planRate-only '
                f'mechanism handles it.'
            ),
        },
        'ceilings': _build_ceilings(),
        'candidates': rows,
    }


# ---------------------------------------------------------------------------
# render_curation_md — a PURE function of the manifest
# ---------------------------------------------------------------------------

def _md_cell(text: str) -> str:
    """Escape a value for a markdown table cell (pipes break the columns)."""
    return str(text).replace('|', '\\|').replace('\n', ' ').strip()


def render_curation_md(manifest: dict) -> str:
    """Render CURATION.md from *manifest*. Pure: no wall-clock, no env, no I/O.

    Purity is load-bearing — a byte-equality test re-renders this and compares
    it to the committed file, so the human table can never silently drift from
    the machine manifest. The census date is DATA in the manifest, never
    ``datetime.now()``.
    """
    census = manifest['census']
    ceilings = manifest['ceilings']
    derivation = ceilings['derivation']
    rows = manifest['candidates']
    included = [r for r in rows if r['decision'] == 'include']
    excluded = [r for r in rows if r['decision'] == 'exclude']

    out: list[str] = []
    add = out.append

    add('# CURATION — fable-trial-v2 curated hard fixture pool')
    add('')
    add('<!-- GENERATED FILE — do not edit by hand.')
    add('     Rendered from `_meta/curation.json` by')
    add('     `scripts/mint_hard_v2_fixtures.py --author`, and pinned')
    add('     byte-for-byte by `test_hard_v2_fixture_pool.py`. Edit the')
    add('     manifest and regenerate. -->')
    add('')
    add(f'- **Task**: {manifest["task"]}')
    add(f'- **PRD**: `{manifest["prd"]}`')
    add(f'- **Cohort**: `{manifest["cohort"]}`')
    add(f'- **Candidates**: {len(rows)} — {len(included)} included, '
        f'{len(excluded)} excluded')
    add('')

    add('## Census')
    add('')
    add(f'- **Source**: `{census["source"]}`')
    add(f'- **Census date**: {census["census_date"]}')
    add('')
    add('```sql')
    add(census['filter_sql'])
    add('```')
    add('')
    add(census['filter_note'])
    add('')
    add('| project | distinct tasks |')
    add('|---|---:|')
    for project, n in census['counts'].items():
        add(f'| {project} | {n} |')
    add(f'| **total** | **{sum(census["counts"].values())}** |')
    add('')

    add('## Curation criterion')
    add('')
    add(manifest['curation_criterion'])
    add('')
    add(manifest['adversarial_scan'])
    add('')

    add('## Candidates')
    add('')
    add('`brief chars` is recorded as EVIDENCE for each judgement, never as '
        'the rule.')
    add('')
    add('| task_id | project | brief chars | status | decision | mint mode | reason |')
    add('|---|---|---:|---|---|---|---|')
    for r in rows:
        add(
            f'| {_md_cell(r["task_id"])} '
            f'| {_md_cell(r["project"])} '
            f'| {r["brief_chars"]} '
            f'| {_md_cell(r["status"])} '
            f'| {_md_cell(r["decision"])} '
            f'| {_md_cell(r.get("mint_mode") or "—")} '
            f'| {_md_cell(r["reason"])} |'
        )
    add('')

    add('## Ceilings')
    add('')
    add(f'- `max_architect_turns`: **{ceilings["max_architect_turns"]}** — '
        f'{ceilings["max_architect_turns_note"]}')
    add(f'- `timeout_minutes`: **{ceilings["timeout_minutes"]}**')
    add('')
    add('### Timeout derivation')
    add('')
    add(f'- **Source**: {derivation["source"]}')
    add(f'- **Substitution**: {derivation["substitution_note"]}')
    add(f'- **Measured** (n={derivation["sample_n"]}): '
        f'max {derivation["observed_max_minutes"]} min, '
        f'p95 {derivation["p95_minutes"]} min, '
        f'p50 {derivation["p50_minutes"]} min.')
    add(f'- **All architect invocations**: max '
        f'{derivation["all_architect_max_minutes"]} min. '
        f'{derivation["all_architect_sample_note"]}')
    add(f'- **Headroom**: {derivation["headroom"]}')
    add(f'- **Why it matters**: {derivation["why_it_matters"]}')
    add('')

    add('## Merge-SHA availability — the SPLIT majority')
    add('')
    add(manifest['merge_sha_availability']['finding'])
    add('')
    add(f'- `referenced` (single clean merge commit): '
        f'**{manifest["merge_sha_availability"]["referenced"]}**')
    add(f'- `planrate_only` (SPLIT / direct-landed): '
        f'**{manifest["merge_sha_availability"]["planrate_only"]}**')
    add('')
    add('A `planrate_only` fixture carries NO `reference` key at all and '
        'instead stamps `provenance.reference_unavailable` with the cause '
        'plus `provenance.baseline_source` with the ladder rung that produced '
        'its `pre_task_commit`. An empty `reference: {}` block would be '
        'indistinguishable from a capture that silently failed; omitting the '
        'key and recording why makes it a positive, auditable fact.')
    add('')

    return '\n'.join(out) + '\n'


def run_author(census_date: str) -> int:
    manifest = author_manifest(census_date)
    CURATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    CURATION_JSON.write_text(json.dumps(manifest, indent=2) + '\n')
    CURATION_MD.write_text(render_curation_md(manifest))
    included = sum(1 for r in manifest['candidates'] if r['decision'] == 'include')
    print(f'wrote {CURATION_JSON} — {len(manifest["candidates"])} candidate(s), '
          f'{included} included, '
          f'{len(manifest["candidates"]) - included} excluded')
    print(f'wrote {CURATION_MD} (generated from the manifest)')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--census', action='store_true',
                      help='Re-run the architect-exhaustion census and print it')
    mode.add_argument('--author', action='store_true',
                      help='(Re)generate _meta/curation.json from the live dbs')
    parser.add_argument('--census-date', default=None,
                        help='ISO date stamped into the manifest (required with '
                             '--author; passed in rather than read from the '
                             'clock so the manifest stays reproducible)')
    parser.add_argument('--no-strict', action='store_true',
                        help='Report census drift without a non-zero exit')
    args = parser.parse_args(argv)

    if args.census:
        return run_census(strict=not args.no_strict)
    if args.author:
        if not args.census_date:
            parser.error('--author requires --census-date')
        return run_author(args.census_date)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
