#!/usr/bin/env python3
"""Mint the fable-architect-trial-v2 curated hard fixture pool (β1, task 3631).

Four modes:

``--census``
    Re-run the architect-exhaustion census across the three source checkouts'
    ``data/orchestrator/runs.db``, enrich each candidate from its
    ``.taskmaster/tasks/tasks.db``, resolve merge-SHA availability and the
    eval baseline commit, and print one row per candidate. This is the
    reproducibility fix for v1, whose sampling driver was gitignored and whose
    pool therefore could not be re-derived.

``--render``
    Re-render ``CURATION.md`` from the COMMITTED ``_meta/curation.json`` — a
    pure function of the manifest, with no database access at all. This is the
    regeneration path after a manifest edit; ``--author`` is not, because it
    re-derives the manifest from the live dbs and refuses on census drift.

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
#
# The committed manifest's `census.counts` block IS this record once the
# manifest exists, so `expected_census_counts()` reads it from there — the
# driver and the artifact it produced cannot disagree. The literal below is
# the BOOTSTRAP value only, used before any manifest has been authored.
_BOOTSTRAP_CENSUS_COUNTS: dict[str, int] = {
    'reify': 36, 'dark_factory': 4, 'know_live': 1,
}


def expected_census_counts() -> dict[str, int]:
    """Return the recorded per-project census counts.

    Read from the committed ``_meta/curation.json`` when it exists (the single
    machine-readable source of truth), falling back to the bootstrap literal
    only when no manifest has been authored yet.
    """
    try:
        counts = json.loads(CURATION_JSON.read_text())['census']['counts']
    except (OSError, ValueError, KeyError, TypeError):
        return dict(_BOOTSTRAP_CENSUS_COUNTS)
    return {str(project): int(n) for project, n in counts.items()}

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

# ---------------------------------------------------------------------------
# The continuity back-fill
# ---------------------------------------------------------------------------
#
# Three fixtures from the STANDING corpus are re-banded into the v2 cohort so
# the v2 campaign overlaps v1's population. They are NOT census candidates —
# they are carried across verbatim from their canonical fixture and their
# lineage is machine-checked against it, so the v2 record can never become a
# divergent re-authoring of the same id.
CONTINUITY_SOURCE_DIR = 'orchestrator/src/orchestrator/evals/tasks'

CONTINUITY_FIXTURES: tuple[dict, ...] = (
    {
        'id': 'reify_task_12',
        'source_path': f'{CONTINUITY_SOURCE_DIR}/reify_task_12.json',
        'reason': (
            'The v1 trial graded plan_quality against a valid reference on '
            'exactly ONE fixture, so its plan-quality signal rested on n=1. '
            'Re-banding this fixture into the v2 cohort under a reference '
            'captured from its own committed pre/post SHAs is what closes '
            'that confound.'
        ),
    },
    {
        'id': 'reify_task_27',
        'source_path': f'{CONTINUITY_SOURCE_DIR}/reify_task_27.json',
        'reason': (
            'Second reify continuity anchor: a high-complexity task from the '
            'same repo as the v1 n=1 fixture, so a v1-to-v2 delta on reify is '
            'not read off a single record.'
        ),
    },
    {
        'id': 'df_task_18',
        'source_path': f'{CONTINUITY_SOURCE_DIR}/df_task_18.json',
        'reason': (
            'The dark-factory continuity anchor, so the overlap with v1 spans '
            'both repos rather than reify alone.'
        ),
    },
)

CONTINUITY_RATIONALE = (
    'These three fixtures are back-filled from the standing corpus so the v2 '
    'campaign shares part of its population with v1 and the two are '
    'comparable rather than merely adjacent. The v1 trial could grade '
    'plan_quality against a valid reference on only one fixture; re-banding '
    'these under references captured from their own committed pre/post SHAs '
    'closes that n=1 confound. Each record is built from the canonical '
    'fixture under evals/tasks/ — same pre_task_commit, same '
    'post_task_commit, same task_definition, same verify_commands — and the '
    'equality is asserted by test, so "reference by copy, do not duplicate '
    'content divergently" is machine-checked rather than conventional. '
    'Capturing the reference here (rather than copying a post-iota-2 fixture) '
    'is the same capture_reference call iota-2 makes for the standing corpus, '
    'so beta-1 is self-contained and creates no cross-task coupling.'
)

# A continuity fixture's baseline is INHERITED, not ladder-resolved: it comes
# from a fixture whose pre/post predate the ladder, and df_task_18's
# pre_task_commit is not its post's first parent, so stamping
# `merge_first_parent` would be a false provenance. Distinct rung name, so the
# weaker/other provenance stays visible instead of masquerading.
CONTINUITY_BASELINE_SOURCE = 'standing_fixture_inherited'


# The source dbs are LIVE: the three orchestrators keep running, so new
# architect exhaustions keep landing in `events` and the census count grows
# over time. That is expected and does NOT invalidate the pool — the manifest
# pins the exact recorded task_ids, so the curated 41 stay reproducible as
# long as none of them DISAPPEAR. Only a missing recorded id means the pool
# can no longer be re-derived.
CENSUS_DRIFT_GUIDANCE = (
    'The source dbs are live, so purely ADDITIVE drift (new exhaustions since '
    'the recorded census_date) is expected and harmless: the manifest pins the '
    'exact recorded task_ids and the curated pool is a dated snapshot, not a '
    'standing query. Refusing to re-author, which would silently pull '
    'uncurated candidates into the pool. Compare the live ids against '
    'census.task_ids in _meta/curation.json: if every recorded id is still '
    'present, the pool is intact and this exit is informational. If a recorded '
    'id is MISSING, the pool genuinely can no longer be re-derived from these '
    'dbs — investigate before touching the manifest.'
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


# The two subject spellings a landing merge is written under. BOTH are in live
# use — censused over reify's main: 2741 x "Merge task/N into main" and 74 x
# "Merge task/N: <subject>" — and matching only the first is what stamped a
# false `reference_unavailable` on fixtures that DO have a landing merge
# (measured: reify_task_4026, reify_task_2379, reify_task_2573).
_MERGE_SUBJECT_SPELLINGS = (
    # Fully-anchored: the whole subject is known.
    '^Merge task/{escaped_id} into main$',
    # Prefix-anchored: the subject tail is free text, so only the head can be
    # matched. The trailing space is load-bearing — see the docstring below.
    '^Merge task/{escaped_id}: ',
)


def _merge_subject_patterns(task_id: str) -> list[str]:
    """Return the anchored EREs matching *task_id*'s landing-merge subjects.

    ONE derivation, shared by every reader, mirroring the discipline of
    ``orchestrator/src/orchestrator/git_ops.py::_merge_subject`` — whose
    single-source-of-truth subject is consumed by ``git_ops.py::GitOps
    .find_merge_marker``, ``merge_to_main`` and ``advance_main`` so writer and
    reader can never silently drift. This driver only ever READS history, so
    it derives the accepted patterns rather than the written subject, but the
    property is the same: adding or amending a spelling happens here, once.
    """
    escaped = re.escape(task_id)
    return [s.format(escaped_id=escaped) for s in _MERGE_SUBJECT_SPELLINGS]


def find_merge_sha(repo_root: Path | str, task_id: str) -> str | None:
    """Return the task's single landing-merge SHA, else ``None``.

    Two subject spellings are accepted, per ``_merge_subject_patterns``:
    ``Merge task/<id> into main`` and ``Merge task/<id>: <subject>``.

    ``None`` means planRate-only: either the task landed SPLIT/direct (no merge
    commit at all) or several merges carry the same id — across the UNION of
    both spellings — in which case picking one would silently invent a
    reference for the judge to grade against.

    Both patterns are ``^``-anchored, which is what preserves substring-safety:
    ``Merge task/8030 …`` cannot answer a query for task ``803`` because the
    ``0`` after ``task/803`` falls where the pattern requires a space or a
    colon. ``git_ops.py::GitOps.find_merge_marker`` gets the same property from
    ``--fixed-strings``; that is not available here because the colon form's
    subject tail is unknown, so only a PREFIX can be matched and git's
    ``--grep`` has no anchoring under ``--fixed-strings``. Hence ERE plus
    ``^``, which also makes the trailing space in the colon pattern
    load-bearing: ``Merge task/803:no-space`` is not the landing-merge shape.
    """
    # git ORs multiple --grep args by default, so one invocation covers the
    # union and `len(shas) == 1` keeps its across-spellings ambiguity meaning.
    greps = [f'--grep={pattern}' for pattern in _merge_subject_patterns(task_id)]
    out = _git(
        ['log', '--all', *greps, '--extended-regexp', '--pretty=%H'],
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
    3. ``timestamp_walk`` — the newest FIRST-PARENT main commit strictly
       before the task's first architect invocation.

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
        # --first-parent is load-bearing. Without it rev-list traverses
        # EVERYTHING reachable from main, including commits that only ever
        # lived on a merged-in side branch, and returns a tree state that was
        # never a state of main. Measured: task 4026's base resolved to
        # e21d047026, a side-branch commit 245 commits from its true branch
        # point (3613bea224^1 = 794d321596) and absent from
        # `git rev-list --first-parent main`. Task 4086 had the same defect.
        # This rung is an approximation either way (see `base_approximation`),
        # but an approximation must at minimum name a real state of main.
        walked = _git(
            ['rev-list', '-n1', '--first-parent',
             f'--before={first_invocation_ts}', 'main'],
            repo_root,
        )
        if walked:
            return walked, 'timestamp_walk'

    raise BaselineUnresolved(
        f'resolve_baseline: no baseline commit for task {task_id!r} in '
        f'{repo_root} — no single landing merge under either accepted '
        f'spelling ("Merge task/{task_id} into main" / '
        f'"Merge task/{task_id}: <subject>"), no '
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

    # Same guard as the sampler's: a locked or schema-drifted db degrades to
    # recorded stubs rather than aborting with a raw traceback. The stubs are
    # not silently minted — `_run_mint` refuses an include row that resolves
    # to an empty title/description.
    try:
        conn = connect_ro(db_path)
        try:
            rows = {
                str(r['id']): r for r in conn.execute(
                    'SELECT id, title, description, status, metadata FROM tasks'
                )
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f'WARNING: read of {db_path} failed ({exc}); candidates stay stubs',
              file=sys.stderr)
        return candidates

    for cand in candidates:
        row = rows.get(str(cand.task_id))
        if row is None:
            continue
        # `or cand.<field>` preserves an already-populated value, matching
        # `enrich_candidates_from_task_db`: a blank column must not blank out
        # a field the caller already had.
        cand.title = row['title'] or cand.title
        cand.description = row['description'] or cand.description
        cand.status = row['status'] or cand.status
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
    recorded = expected_census_counts()
    for project, n in counts.items():
        expected = recorded.get(project)
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

    if strict and counts != recorded:
        print(f'\nCENSUS DRIFT: got {counts}, expected {recorded}. '
              f'{CENSUS_DRIFT_GUIDANCE}', file=sys.stderr)
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
SELECT MAX(duration_ms), COUNT(*) FROM events
WHERE event_type='invocation_end' AND role='architect' AND duration_ms IS NOT NULL\
"""


class WallClockEvidenceMissing(RuntimeError):
    """No ``duration_ms`` evidence, so ``timeout_minutes`` cannot be derived.

    Raised rather than dividing by a zero denominator: the whole point of the
    ceilings block is that the timeout is DERIVED from measured wall-clock and
    shown not to bind, so a run with no measurement must stop at the missing
    evidence — not emit a ZeroDivisionError from inside an f-string several
    frames from the cause, and certainly not a basis-free threshold.
    """


def _all_architect_duration_stats(runs_db: Path | str) -> tuple[int, int]:
    """Return ``(max_duration_ms, n)`` over EVERY architect invocation.

    The count is measured alongside the max so the note that reports the
    sample size can never carry a stale hardcoded n next to a fresh max.
    """
    conn = connect_ro(runs_db)
    try:
        row = conn.execute(_ALL_ARCHITECT_DURATION_SQL).fetchone()
    finally:
        conn.close()
    if not row:
        return 0, 0
    return (int(row[0]) if row[0] else 0), (int(row[1]) if row[1] else 0)


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
    all_n = 0
    for root_str in SOURCE_CHECKOUTS.values():
        runs_db = Path(root_str) / 'data' / 'orchestrator' / 'runs.db'
        census_ms.extend(census_durations_ms(runs_db))
        root_max_ms, root_n = _all_architect_duration_stats(runs_db)
        all_max_ms = max(all_max_ms, root_max_ms)
        all_n += root_n

    minutes = sorted(d / 60_000 for d in census_ms)
    observed_max = round(max(minutes), 1) if minutes else 0.0
    p95 = round(_percentile(minutes, 95), 1) if minutes else 0.0
    p50 = round(_percentile(minutes, 50), 1) if minutes else 0.0
    all_max = round(all_max_ms / 60_000, 1)

    # Both are headroom DENOMINATORS below. A checkout whose runs.db carries no
    # populated duration_ms (fresh, truncated, or schema-drifted) would turn
    # this deliberately loud script into an opaque ZeroDivisionError raised
    # from inside an f-string, so name the missing evidence instead.
    if observed_max <= 0 or all_max <= 0:
        raise WallClockEvidenceMissing(
            f'_build_ceilings: no wall-clock evidence to derive '
            f'timeout_minutes from — observed max-at-exhaustion '
            f'{observed_max} min over n={len(minutes)} census invocation(s), '
            f'all-architect max {all_max} min over n={all_n} invocation(s). '
            f'Expected populated events.duration_ms in '
            f'{[str(Path(r) / "data" / "orchestrator" / "runs.db") for r in SOURCE_CHECKOUTS.values()]}. '
            f'Refusing to emit a ceilings block whose headroom cannot be '
            f'computed: the threshold would then be a guess, which is exactly '
            f'what this derivation exists to prevent.'
        )

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
            'all_architect_sample_n': all_n,
            'all_architect_sample_note': (
                f'all_architect_max_minutes is the max over EVERY architect '
                f'invocation_end carrying a duration_ms in the three '
                f'checkouts (n={all_n}), not just the census population — the '
                f'strictest bound available. Measured alongside the max, so '
                f'the two can never be reported from different populations.'
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
    recorded = expected_census_counts()
    if counts != recorded:
        raise RuntimeError(
            f'author_manifest: census drift — got {counts}, expected '
            f'{recorded}. Refusing to author a manifest whose '
            f'pool silently differs from the recorded one. '
            f'{CENSUS_DRIFT_GUIDANCE}'
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
            'reproducibility_note': CENSUS_DRIFT_GUIDANCE,
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
        'continuity': {
            'rationale': CONTINUITY_RATIONALE,
            'baseline_source': CONTINUITY_BASELINE_SOURCE,
            'fixtures': [dict(entry) for entry in CONTINUITY_FIXTURES],
        },
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
    add('     `scripts/mint_hard_v2_fixtures.py --render`, and pinned')
    add('     byte-for-byte by `test_hard_v2_fixture_pool.py`. Edit the')
    add('     manifest and re-run --render. (--author re-derives the whole')
    add('     manifest from the LIVE runs.db files and refuses on census')
    add('     drift, so it is not the regeneration path.) -->')
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
    add(f'**Reproducibility**: {census["reproducibility_note"]}')
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

    continuity = manifest['continuity']
    add('## Continuity back-fill')
    add('')
    add(continuity['rationale'])
    add('')
    add('| fixture | source | why |')
    add('|---|---|---|')
    for entry in continuity['fixtures']:
        add(f'| `{_md_cell(entry["id"])}` '
            f'| `{_md_cell(entry["source_path"])}` '
            f'| {_md_cell(entry["reason"])} |')
    add('')
    add(f'These carry `provenance.baseline_source: '
        f'{continuity["baseline_source"]}` rather than a ladder rung — their '
        f'baseline is inherited from the canonical fixture, not resolved '
        f'here, and `df_task_18`\'s `pre_task_commit` is not its post '
        f'commit\'s first parent, so claiming `merge_first_parent` would be a '
        f'false provenance. They are the ONLY fixtures whose ids overlap the '
        f'standing corpus, and that overlap is pinned by test.')
    add('')

    return '\n'.join(out) + '\n'


def run_render() -> int:
    """Re-render ``CURATION.md`` from the COMMITTED manifest. No db access.

    This is the regeneration path the README and the generated-file header
    name. ``--author`` cannot serve as one: it re-derives the whole manifest
    from the three LIVE ``runs.db`` files and refuses on ANY census drift,
    while purely additive drift (new architect exhaustions, which keep
    landing) is expected and harmless — so ``--author`` becomes unrunnable
    over time, and after a manifest hand-edit it would in any case overwrite
    the edit rather than render it. ``--render`` is a pure function of the
    committed manifest, which is exactly what the byte-equality test pins.
    """
    if not CURATION_JSON.exists():
        raise RuntimeError(
            f'run_render: no manifest at {CURATION_JSON}. CURATION.md is '
            f'generated FROM the manifest, so there is nothing to render; '
            f'author the manifest first (--author).'
        )
    manifest = json.loads(CURATION_JSON.read_text())
    CURATION_MD.write_text(render_curation_md(manifest))
    print(f'wrote {CURATION_MD} (rendered from {CURATION_JSON}; no db access)')
    return 0


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


# ---------------------------------------------------------------------------
# --mint — drive the canonical task_sampler pipeline over the include rows
# ---------------------------------------------------------------------------

# The project → repo-stratum mapping is NOT re-declared here: it lives in
# task_sampler.repo_of_project(), the same table build_fixture_record derives
# each fixture id from. A local literal copy could drift from it silently.


def _is_ancestor_of_main(repo_root: Path | str, commit: str) -> bool:
    """True if *commit* is reachable from ``main`` (hence not GC-eligible)."""
    proc = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', commit, 'main'],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    return proc.returncode == 0


def _refs_pointing_at(repo_root: Path | str, commit: str) -> list[str]:
    """Return every ref that points at *commit* (empty if none do).

    The second half of the retrievability question. Ancestry of ``main`` is not
    the only thing that holds a commit off the GC path — an existing
    ``evals/<id>`` branch (exactly what ``pin_eval_branch`` would have created)
    does too. A task-branch tip that was squashed or rebased onto main is NOT
    an ancestor of main yet is still perfectly safe if a ref holds it, so
    reporting GC-eligibility from ancestry alone would raise a false alarm.
    """
    out = _git(['for-each-ref', '--points-at', commit, '--format=%(refname)'],
               repo_root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _import_sampler():
    """Import ``task_sampler`` from this repo's orchestrator package."""
    src = REPO_ROOT / 'orchestrator' / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from orchestrator.evals import task_sampler
    return task_sampler


async def _pin_or_record_failure(
    sampler, record: dict, project_root: str, post: str,
) -> None:
    """Pin ``evals/<id>`` at *post*, recording the failure rather than raising.

    The pin is a stability guarantee (it keeps *post* off the GC path), not a
    correctness precondition — the reference block already carries the SHA. So
    a read-only checkout degrades to a RECORDED failure rather than making the
    whole pool un-mintable. ``pin_eval_branch`` uses ``update-ref``, so
    re-pinning an already-pinned branch at the same SHA is a no-op.
    """
    try:
        await sampler.pin_eval_branch(project_root, record['id'], post)
        record['provenance']['eval_branch_pinned'] = True
    except (RuntimeError, OSError) as exc:
        record['provenance']['eval_branch_pinned'] = False
        record['provenance']['eval_branch_pin_error'] = (
            f'pin_eval_branch(evals/{record["id"]}) failed: {exc}'
        )

        # Retrievability has TWO independent guarantors: reachability from
        # main, or any ref pointing at the commit. Reporting from ancestry
        # alone would cry wolf on a task-branch tip that an `evals/<id>`
        # branch already holds — and a field that raises false alarms stops
        # being read.
        preamble = (
            'The reference block carries the post SHA directly and '
            'materialize_reference_diff works off the pre/post SHAs, not the '
            'branch; the pin only holds the commit off the GC path. '
        )
        if _is_ancestor_of_main(project_root, post):
            impact = (
                f'None for retrievability. {preamble}post_task_commit IS an '
                f'ancestor of main in this checkout, so it is reachable and '
                f'not GC-eligible regardless. Re-run --mint from a checkout '
                f'with write access to restore the belt-and-braces ref.'
            )
        elif (holders := _refs_pointing_at(project_root, post)):
            impact = (
                f'None for retrievability. {preamble}post_task_commit is NOT '
                f'an ancestor of main (it is a task-branch tip, not a merge '
                f'commit), but it is already held by {len(holders)} ref(s) — '
                f'{", ".join(holders)} — so it is not GC-eligible. The '
                f'intended evals/{record["id"]} pin is among them, i.e. the '
                f'pin this run could not write already exists at this exact '
                f'SHA and the update-ref would have been a no-op.'
            )
        else:
            impact = (
                f'ACTION NEEDED. {preamble}post_task_commit is NOT an '
                f'ancestor of main in this checkout and NO ref points at it, '
                f'so it is GC-eligible and the reference diff can be lost. '
                f'Re-pin from a checkout with write access before relying '
                f'on it.'
            )
        record['provenance']['eval_branch_pin_impact'] = impact
        print(f'  WARNING: could not pin evals/{record["id"]}: {exc}',
              file=sys.stderr)


async def _mint_continuity_one(sampler, entry: dict, sampled_at: str, seed: int,
                               ceilings: dict) -> dict:
    """Re-band one STANDING fixture into the v2 cohort.

    Built from the canonical fixture's OWN committed ``pre_task_commit`` /
    ``post_task_commit`` via ``capture_reference`` — the same call ι2 makes for
    the standing corpus — so β1 gets a valid reference block immediately with
    no dependency on a sibling task landing. Everything that identifies the
    task (``task_definition``, ``project``, ``project_root``,
    ``verify_commands``, ``modules``, ``complexity``, the frozen ``plan``) is
    carried across verbatim; only the v2 banding (cohort, ceilings,
    provenance) is added. Nothing is written back to ``evals/tasks/``.
    """
    source_path = REPO_ROOT / entry['source_path']
    src = json.loads(source_path.read_text())

    task_id = src['id'].split('_task_', 1)[1]
    cand = sampler.CompletedTaskCandidate(
        task_id=task_id,
        project=src['project'],
        project_root=src['project_root'],
        title=src['task_definition']['title'],
        description=src['task_definition']['description'],
        complexity=src.get('complexity'),
        modules=list(src.get('modules') or []),
        pre_commit=src['pre_task_commit'],
        post_commit=src['post_task_commit'],
        # The standing fixtures predate provenance capture and record no merge
        # commit, so there is none to carry. Recorded as None below.
        merge_sha='',
    )

    reference = await sampler.capture_reference(
        cand.project_root, cand.pre_commit, cand.post_commit,
    )
    if reference.files <= 0:
        # A reference-less continuity fixture would defeat the entire point of
        # the back-fill (closing v1's n=1 plan_quality confound), so an empty
        # capture is a hard stop, not a shrug.
        raise RuntimeError(
            f'_mint_continuity_one: capture_reference over '
            f'{cand.pre_commit}..{cand.post_commit} in {cand.project_root} '
            f'produced an EMPTY diff for {src["id"]}. The continuity fixture '
            f'exists to carry a valid reference; refusing to mint one that '
            f'cannot be graded against it.'
        )

    record = sampler.build_fixture_record(
        cand, reference, src['verify_commands'], plan=src.get('plan'),
        cohort='fable-trial-v2-hard', sampled_at=sampled_at, seed=seed,
        plan_source='standing_fixture',
    )

    # The id and name are DERIVED by build_fixture_record; a mismatch would
    # mean the v2 record is not the same fixture, which is the exact
    # divergence the continuity contract forbids.
    if record['id'] != src['id'] or record['name'] != src['name']:
        raise RuntimeError(
            f'_mint_continuity_one: derived id/name '
            f'({record["id"]!r}/{record["name"]!r}) does not match the source '
            f'({src["id"]!r}/{src["name"]!r}) — refusing to mint a divergent '
            f'copy under a continuity id.'
        )

    record['provenance']['merge_sha'] = None
    record['provenance']['derived_from'] = entry['source_path']
    record['provenance']['baseline_source'] = CONTINUITY_BASELINE_SOURCE

    # build_fixture_record stamps `{source:'landed', passed:True}`
    # unconditionally, on the documented premise "the task merged to main ⇒ its
    # gates passed at the post commit". MEASURE that premise instead of
    # asserting it: these three fixtures predate provenance capture and their
    # post commits are task-BRANCH TIPS, not merges reachable from main, so at
    # this SHA the premise is not established — and the canonical fixture
    # records no verify_outcome to inherit either. Same refusal to fabricate a
    # gate result as the planRate-only path in `_mint_one`.
    reachable = _is_ancestor_of_main(cand.project_root, cand.post_commit)
    record['provenance']['post_commit_reachable_from_main'] = reachable
    if not reachable:
        record['verify_outcome'] = {
            'source': 'landed_branch_tip',
            'passed': None,
            'commands': dict(src['verify_commands']),
            'reason': (
                f'post_task_commit {cand.post_commit} is NOT reachable from '
                f'main in {cand.project_root} (measured at mint time with '
                f'`git merge-base --is-ancestor`): it is the task-branch tip '
                f'the canonical fixture recorded, not a merge commit on main. '
                f'The "merged to main ⇒ gates passed at the post commit" '
                f'premise therefore cannot be established AT THIS SHA, and '
                f'{entry["source_path"]} carries no verify_outcome to inherit, '
                f'so `passed` is recorded as unknown rather than asserted. '
                f'This does not weaken the fixture: it is scored on plan '
                f'quality against its reference diff, never on a landed gate '
                f'result. The commands are carried for provenance only.'
            ),
        }
    record['provenance']['continuity_note'] = (
        f'Re-banded from the standing corpus, not censused. '
        f'pre_task_commit / post_task_commit / task_definition are carried '
        f'verbatim from {entry["source_path"]} and asserted equal by test; '
        f'the reference block was captured here from those same SHAs. The '
        f'baseline is INHERITED, not ladder-resolved.'
    )

    # Carry across any key the canonical fixture carries that the record
    # schema does not produce — `setup_commands` and `max_review_cycles` are
    # both read by the runner (setup_commands inside run_architect_eval's own
    # worktree creation), so dropping them would silently change how the
    # fixture runs in the very campaign this pool feeds.
    for key, value in src.items():
        if key not in record:
            record[key] = value

    await _pin_or_record_failure(
        sampler, record, cand.project_root, cand.post_commit,
    )

    record['max_architect_turns'] = ceilings['max_architect_turns']
    record['timeout_minutes'] = ceilings['timeout_minutes']
    return record


async def _mint_one(sampler, row: dict, sampled_at: str, seed: int,
                    ceilings: dict) -> dict:
    """Build one fixture record from an adjudicated ``include`` row."""
    repo = sampler.repo_of_project(row['project'])
    cand = sampler.CompletedTaskCandidate(
        task_id=row['task_id'],
        project=row['project'],
        project_root=row['project_root'],
        title=row['title'],
        description=row['description'],
        complexity=row.get('complexity'),
        modules=list(row.get('modules') or []),
        merge_sha=row['merge_sha'] or '',
    )

    if row['mint_mode'] == 'referenced':
        pre, post = await sampler.resolve_task_commits_from_merge(
            cand.project_root, row['merge_sha'],
        )
        cand.pre_commit, cand.post_commit = pre, post
        reference = await sampler.capture_reference(cand.project_root, pre, post)
    else:
        # planRate-only: no landed post SHA exists, so there is nothing to
        # diff and nothing to pin. The baseline comes off the recorded rung.
        cand.pre_commit = row['baseline_sha']
        cand.post_commit = ''
        reference = sampler.ReferenceCapture(
            post_task_commit='', files=0, insertions=0, deletions=0,
        )

    record = sampler.build_fixture_record(
        cand, reference, sampler.default_verify_commands(repo), plan=None,
        cohort='fable-trial-v2-hard', sampled_at=sampled_at, seed=seed,
    )
    record['provenance']['baseline_source'] = row['baseline_source']
    # The curation's terminal adjudication travels WITH the fixture. Without
    # it a reader of the JSON alone cannot tell a landed task from a cancelled
    # one, and the verify_outcome below would be an unattributable claim.
    record['provenance']['task_status'] = row['status']

    if row['mint_mode'] == 'referenced':
        await _pin_or_record_failure(
            sampler, record, cand.project_root, cand.post_commit,
        )
    else:
        # Omit the key entirely and record WHY. An empty `reference: {}` block
        # is indistinguishable from a capture that silently failed, which is
        # exactly the degradation D9 exists to abolish.
        record.pop('reference', None)
        record['provenance']['reference_unavailable'] = (
            f'No single landing merge for task {row["task_id"]} exists in '
            f'{row["project_root"]} under EITHER accepted subject spelling — '
            f'"Merge task/{row["task_id"]} into main" or '
            f'"Merge task/{row["task_id"]}: <subject>" (SPLIT / direct-landed, '
            f'or several merges carry this id across the two spellings) — so '
            f'no landed post-commit is available to diff against. This fixture '
            f'is planRate-only: it is scored on plan production, never on '
            f'reference-diff similarity. Its pre_task_commit came from '
            f'baseline rung {row["baseline_source"]!r}.'
        )
        # build_fixture_record stamps landed_verify_outcome unconditionally,
        # and its contract is "the task merged to main => its gates passed at
        # the post commit". That premise is exactly what fails here: there is
        # no post commit (and for a cancelled candidate the task never landed
        # at all), so `source: 'landed', passed: True` would be a fabricated
        # ground truth. Same reasoning as the popped `reference` above —
        # applied to the gate result instead of the diff.
        record['verify_outcome'] = {
            'source': 'unavailable',
            'passed': None,
            'commands': sampler.default_verify_commands(repo),
            'reason': (
                f'No landed post-commit is available for task '
                f'{row["task_id"]} (terminal status {row["status"]!r}), so '
                f'there is no commit at which these gates can be claimed to '
                f'have passed. This fixture is planRate-only and is never '
                f'scored against a landed verify outcome; the commands are '
                f'carried for provenance only.'
            ),
        }

    # runner.py reads both straight off the task record (task.get with 50 / 60
    # defaults), so a fixture missing them silently runs at the wrong ceiling.
    record['max_architect_turns'] = ceilings['max_architect_turns']
    record['timeout_minutes'] = ceilings['timeout_minutes']
    return record


async def _run_mint(sampled_at: str, seed: int) -> int:
    import asyncio  # noqa: F401  (imported for symmetry with the caller)

    sampler = _import_sampler()
    manifest = json.loads(CURATION_JSON.read_text())
    ceilings = manifest['ceilings']

    includes = [c for c in manifest['candidates'] if c['decision'] == 'include']
    # The manifest carries only the curation fields; re-read title/description
    # from each task db so the fixture's task_definition is the real brief.
    by_project: dict[str, list[dict]] = {}
    for row in includes:
        by_project.setdefault(row['project'], []).append(row)
    for project, rows in by_project.items():
        root = Path(SOURCE_CHECKOUTS[project])
        cands = [
            Candidate(task_id=r['task_id'], project=project, project_root=str(root))
            for r in rows
        ]
        enrich_from_task_db(cands, root / '.taskmaster' / 'tasks' / 'tasks.db')
        for row, cand in zip(rows, cands, strict=True):
            # The enrichment degrades to a stub (unreadable db, id absent from
            # the tasks table, blanked columns) rather than raising. Minting
            # that stub would write a fixture with an empty name and an empty
            # task_definition — an architect handed no brief at all — so the
            # stub is refused HERE, where the cause is known.
            if not (cand.title or '').strip() or not (cand.description or '').strip():
                raise RuntimeError(
                    f'_run_mint: include row {project}/{row["task_id"]} '
                    f'resolved to an empty title/description from '
                    f'{root}/.taskmaster/tasks/tasks.db (title={cand.title!r}, '
                    f'brief_chars={cand.brief_chars}). Refusing to mint a '
                    f'fixture with no brief; re-run --census to see whether '
                    f'the task id still resolves in that db.'
                )
            row['title'] = cand.title
            row['description'] = cand.description
            row['complexity'] = cand.complexity
            row['modules'] = cand.modules

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in includes:
        record = await _mint_one(sampler, row, sampled_at, seed, ceilings)
        (POOL_DIR / f'{record["id"]}.json').write_text(
            json.dumps(record, indent=2) + '\n',
        )
        written += 1
        print(f'  wrote {record["id"]}.json ({row["mint_mode"]}, '
              f'baseline={row["baseline_source"]})')

    for entry in manifest['continuity']['fixtures']:
        record = await _mint_continuity_one(
            sampler, entry, sampled_at, seed, ceilings,
        )
        (POOL_DIR / f'{record["id"]}.json').write_text(
            json.dumps(record, indent=2) + '\n',
        )
        written += 1
        print(f'  wrote {record["id"]}.json (continuity, from '
              f'{entry["source_path"]})')

    print(f'wrote {written} fixture(s) to {POOL_DIR}')
    return 0


def run_mint(sampled_at: str, seed: int) -> int:
    import asyncio
    return asyncio.run(_run_mint(sampled_at, seed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--census', action='store_true',
                      help='Re-run the architect-exhaustion census and print it')
    mode.add_argument('--author', action='store_true',
                      help='(Re)generate _meta/curation.json from the live dbs')
    mode.add_argument('--render', action='store_true',
                      help='Re-render CURATION.md from the COMMITTED manifest '
                           '(no db access) — the regeneration path after a '
                           'manifest edit')
    mode.add_argument('--mint', action='store_true',
                      help='Mint the fixture JSONs from the manifest include rows')
    parser.add_argument('--census-date', default=None,
                        help='ISO date stamped into the manifest (required with '
                             '--author; passed in rather than read from the '
                             'clock so the manifest stays reproducible)')
    parser.add_argument('--sampled-at', default=None,
                        help='ISO timestamp stamped into every minted fixture '
                             '(required with --mint; injected at the CLI '
                             'boundary, mirroring cli._run_eval_sample, so the '
                             'minting library stays wall-clock-free)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Recorded in each fixture\'s provenance')
    parser.add_argument('--no-strict', action='store_true',
                        help='Report census drift without a non-zero exit')
    args = parser.parse_args(argv)

    if args.census:
        return run_census(strict=not args.no_strict)
    if args.author:
        if not args.census_date:
            parser.error('--author requires --census-date')
        return run_author(args.census_date)
    if args.render:
        return run_render()
    if args.mint:
        if not args.sampled_at:
            parser.error('--mint requires --sampled-at')
        return run_mint(args.sampled_at, args.seed)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
