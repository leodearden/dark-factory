"""Stratified eval-fixture task sampler (revival ζ).

Re-cuts a small, near-HEAD eval-fixture corpus across BOTH repos
(dark_factory + reify), stratified by ``repo × kind × path``, plus a listing
command that prints the stratification counts. Mirrors the corpus-builder
pattern established by ``evals/prompt_opt/curator_corpus.py``: read-only
discovery, deterministic stratified sampling (seeded ``random.Random``,
round-robin across cells), dataclass records, a structured audit report with
named failures, and hermetic dependency-injected tests behind a thin operator
CLI.

Determinism is a hard requirement: the library takes an explicit ``seed`` and
``sampled_at`` at its boundary and never reads the wall-clock or unseeded
randomness itself (same discipline as ``curator_corpus``), so a cut is exactly
reproducible.

This module is intentionally split across the plan's steps — this first slice
carries the ``CompletedTaskCandidate`` record and the three pure classifiers
(``repo_of`` / ``classify_kind`` / ``classify_path``) that define the three
stratification axes. Sampling, fixture-record building, git glue, and the
audit/listing surface land in later slices.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from orchestrator.agents.triage import has_simple_task_blocker
from orchestrator.evals.snapshots import _run as _git_run
from orchestrator.evals.snapshots import get_diff_between_commits

logger = logging.getLogger(__name__)

__all__ = [
    'Cell',
    'CompletedTaskCandidate',
    'FixtureAuditReport',
    'ReferenceCapture',
    'SampleResult',
    'StratificationCounts',
    'audit_fixture_corpus',
    'build_fixture_record',
    'capture_reference',
    'cell_of',
    'classify_kind',
    'classify_path',
    'default_verify_commands',
    'discover_completed_tasks',
    'enrich_candidates_from_task_db',
    'format_stratification_table',
    'git_ref_exists',
    'landed_verify_outcome',
    'materialize_reference_diff',
    'pin_eval_branch',
    'repo_of',
    'resolve_task_commits_from_merge',
    'sample_stratified',
    'stratification_counts',
    'stratify',
]

# A stratification cell: the ``(repo, kind, path)`` 3-tuple.
Cell = tuple[str, str, str]

# The full axis product — every possible (repo, kind, path) cell. Used to
# surface EMPTY cells loudly (a cell the corpus offers nothing for) rather
# than letting thin coverage pass silently.
_REPOS = ('df', 'reify')
_KINDS = ('bugfix', 'feature', 'refactor')
_PATHS = ('simple', 'full')
_ALL_CELLS: tuple[Cell, ...] = tuple(
    (repo, kind, path) for repo in _REPOS for kind in _KINDS for path in _PATHS
)


@dataclass
class CompletedTaskCandidate:
    """One completed-task merge, a candidate for the eval-fixture corpus.

    Reconstructed read-only from a ``Merge task/<id> into main`` merge commit
    (see ``discover_completed_tasks``, a later slice): ``merge_sha`` is that
    commit M, ``post_commit`` is M (the landed state, where ``evals/<id>`` is
    pinned) and ``pre_commit`` is M^1 (prior main / the eval baseline). The
    classifier axes read only ``project`` / ``title`` / ``description`` /
    ``complexity``; the git fields carry the reference-diff provenance for
    fixture building.
    """

    task_id: str
    project: str
    project_root: str
    title: str = ''
    description: str = ''
    complexity: str | None = None
    modules: list[str] = field(default_factory=list)
    pre_commit: str = ''
    post_commit: str = ''
    merge_sha: str = ''


# ---------------------------------------------------------------------------
# repo_of — the repo axis (df / reify)
# ---------------------------------------------------------------------------

def repo_of(candidate: CompletedTaskCandidate) -> str:
    """Return the repo stratum (``'df'`` or ``'reify'``) for *candidate*.

    Normalises across the two spellings in play: fused-memory's project_id
    ``'dark_factory'`` and the fixture-JSON project string ``'dark-factory'``
    both map to ``'df'``; ``'reify'`` maps to ``'reify'``.

    Raises ``ValueError`` on an unrecognised project — repo is a hard
    stratification axis, so an unknown value is a loud error rather than a
    silent default (honours the loud-over-silent-degradation norm).
    """
    project = (candidate.project or '').strip().lower()
    if 'reify' in project:
        return 'reify'
    if 'dark' in project and 'factory' in project:
        return 'df'
    raise ValueError(
        f'repo_of: unrecognised project {candidate.project!r} for task '
        f'{candidate.task_id!r} (expected a dark_factory/dark-factory or reify project)'
    )


# ---------------------------------------------------------------------------
# classify_kind — the kind axis (bugfix / feature / refactor)
# ---------------------------------------------------------------------------

# Keyword stems matched with a LEADING word boundary (``\b<stem>``), so a stem
# also matches its inflections ('fix' → 'fixes'/'fixing'/'fixed') while a
# trailing word char keeps 'move' from matching 'remove'. The lists are a
# coarse heuristic for stratification only — misclassifications are acceptable
# and are surfaced in the sample notes, never silently load-bearing.
_BUGFIX_STEMS = (
    'bug', 'fix', 'broke', 'regress', 'crash', 'incorrect', 'wrong', 'fault',
    'revert', 'hotfix', 'patch',
)
_REFACTOR_STEMS = (
    'refactor', 'extract', 'rename', 'simplif', 'consolidat', 'dedup',
    'cleanup', 'clean up', 'restructur', 'reorganiz', 'reorganis',
)
_FEATURE_STEMS = (
    'add', 'implement', 'introduc', 'new', 'support', 'creat', 'feature',
    'enable', 'build',
)

# Documented default when no stem matches: 'feature'. Most net-new task work
# is feature-shaped, and a feature fixture is a safe home for an
# unclassifiable task (the kind axis is descriptive stratification, not a
# correctness gate).
_DEFAULT_KIND = 'feature'


def _matches_any(text: str, stems: tuple[str, ...]) -> bool:
    return any(re.search(r'\b' + re.escape(stem), text) for stem in stems)


def classify_kind(candidate: CompletedTaskCandidate) -> str:
    """Return the kind stratum: ``'bugfix'`` | ``'feature'`` | ``'refactor'``.

    A keyword heuristic over ``title + description`` (lower-cased). Precedence
    is **bugfix > refactor > feature**: an explicit bug/fix signal is the
    strongest intent marker (and bug fixes are the most consequential kind to
    label right for the corpus — they exercise diagnose-and-minimally-fix),
    then structural refactors, then net-new feature work. Falls back to the
    documented default (``'feature'``) when no stem matches.
    """
    text = f'{candidate.title}\n{candidate.description}'.lower()
    if _matches_any(text, _BUGFIX_STEMS):
        return 'bugfix'
    if _matches_any(text, _REFACTOR_STEMS):
        return 'refactor'
    if _matches_any(text, _FEATURE_STEMS):
        return 'feature'
    return _DEFAULT_KIND


# ---------------------------------------------------------------------------
# classify_path — the path axis (simple / full), mirroring production routing
# ---------------------------------------------------------------------------

def classify_path(candidate: CompletedTaskCandidate) -> str:
    """Return the path stratum: ``'simple'`` | ``'full'``.

    Reuses the production simple-task fast-path definition so the stratifier's
    path axis matches how the orchestrator would actually have routed the task
    (``orchestrator.agents.triage``): the path is ``'simple'`` iff
    ``complexity`` normalises to ``'simple'`` (case-insensitive, whitespace-
    stripped) AND no hard-blocker contradiction token
    (migration/architecture/integration test/design…new/implement…new feature)
    is present; otherwise ``'full'``.

    The veto (``has_simple_task_blocker``) is applied to ``title +
    description`` rather than the description alone. This is a deliberate
    conservative superset of production's description-only check: a blocker in
    the title is just as disqualifying, and checking more text can only push a
    candidate toward ``'full'`` (never toward a spurious ``'simple'``), which
    is the safe direction for an eval-corpus stratifier.
    """
    if (candidate.complexity or '').strip().lower() != 'simple':
        return 'full'
    text = f'{candidate.title}\n{candidate.description}'
    if has_simple_task_blocker(text):
        return 'full'
    return 'simple'


def cell_of(candidate: CompletedTaskCandidate) -> Cell:
    """Return the ``(repo, kind, path)`` stratification cell for *candidate*."""
    return (repo_of(candidate), classify_kind(candidate), classify_path(candidate))


# ---------------------------------------------------------------------------
# stratify + sample_stratified — deterministic round-robin cut over the band
# ---------------------------------------------------------------------------

def stratify(
    candidates: list[CompletedTaskCandidate],
) -> dict[Cell, list[CompletedTaskCandidate]]:
    """Bucket *candidates* into ``(repo, kind, path)`` cells.

    Returns a dict keyed by the 3-tuple; a cell with no candidates is simply
    absent (not a zero-valued key), mirroring ``curator_corpus``'s
    ``by_action`` bucketing. Empty-cell surfacing is the sampler's job
    (:func:`sample_stratified` notes), not this partition's.
    """
    cells: dict[Cell, list[CompletedTaskCandidate]] = {}
    for candidate in candidates:
        cells.setdefault(cell_of(candidate), []).append(candidate)
    return cells


@dataclass
class SampleResult:
    """Result of a stratified cut: the selection, per-cell counts, and notes.

    ``notes`` carries human-readable, loud-over-silent signals — a shortfall
    when the corpus can't fill the band floor, and one entry per EMPTY axis
    cell — so a thin cut is never silently truncated.
    """

    selected: list[CompletedTaskCandidate]
    cell_counts: dict[Cell, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _round_robin_select(
    cells: dict[Cell, list[CompletedTaskCandidate]], n: int, seed: int,
) -> list[CompletedTaskCandidate]:
    """Deterministically pick *n* candidates round-robin across *cells*.

    Mirrors ``curator_corpus._sample_stratified``: within each cell the
    candidates are ordered by ``task_id`` then shuffled with a per-cell seeded
    ``random.Random`` (no wall-clock / unseeded randomness), and the picker
    walks the cells in sorted order taking one per cell per round until *n* is
    reached or every cell is exhausted (a thin cell is thereby taken in full).
    Round-robin keeps a bounded selection representative even when one cell
    dominates the raw corpus.
    """
    shuffled: dict[Cell, list[CompletedTaskCandidate]] = {}
    for cell, items in cells.items():
        ordered = sorted(items, key=lambda c: c.task_id)
        rng = random.Random(f'{seed}:{cell}')
        rng.shuffle(ordered)
        shuffled[cell] = ordered

    selected: list[CompletedTaskCandidate] = []
    cursors = dict.fromkeys(shuffled, 0)
    cell_cycle = sorted(shuffled)
    while len(selected) < n:
        progressed = False
        for cell in cell_cycle:
            if len(selected) >= n:
                break
            cursor = cursors[cell]
            pool = shuffled[cell]
            if cursor < len(pool):
                selected.append(pool[cursor])
                cursors[cell] = cursor + 1
                progressed = True
        if not progressed:
            break  # every cell exhausted before reaching n
    return selected


def _coverage_notes(
    cells: dict[Cell, list[CompletedTaskCandidate]],
    total: int,
    target_low: int,
    selected_n: int,
) -> list[str]:
    """Loud-over-silent coverage notes: shortfall + every empty axis cell."""
    notes: list[str] = []
    if total < target_low:
        notes.append(
            f'shortfall: corpus has {total} candidate(s), below target floor '
            f'{target_low}; selected all {selected_n} (no silent truncation)'
        )
    for cell in _ALL_CELLS:
        if not cells.get(cell):
            notes.append(f'empty cell: {cell[0]}/{cell[1]}/{cell[2]}')
    return notes


def sample_stratified(
    candidates: list[CompletedTaskCandidate],
    target_low: int = 10,
    target_high: int = 14,
    seed: int = 0,
) -> SampleResult:
    """Deterministically sample *candidates* down to the ``[target_low,
    target_high]`` band, round-robin across ``(repo, kind, path)`` cells.

    - Rich corpus (``total > target_high``): sample down to exactly
      *target_high*, round-robin, so no single over-supplied cell dominates.
    - In-band or thin corpus (``total <= target_high``): take ALL candidates
      (a thin cell is taken in full). When ``total < target_low`` the floor
      can't be met — that shortfall is recorded in ``notes`` rather than
      silently truncated.

    Determinism is by explicit *seed* only (no wall-clock / unseeded random),
    so a cut is exactly reproducible; a different *seed* may reorder/repick.
    Empty axis cells are always surfaced in ``notes``.
    """
    cells = stratify(candidates)
    total = len(candidates)
    target_n = total if total <= target_high else target_high
    selected = _round_robin_select(cells, target_n, seed)
    cell_counts = dict(Counter(cell_of(c) for c in selected))
    notes = _coverage_notes(cells, total, target_low, len(selected))
    return SampleResult(selected=selected, cell_counts=cell_counts, notes=notes)


# ---------------------------------------------------------------------------
# Fixture-record building: reference block, verify outcome, canonical schema
# ---------------------------------------------------------------------------

# The canonical fixture `project` string per repo — matches the existing
# evals/tasks/*.json fixtures (df fixtures spell it 'dark-factory'). The field
# is provenance only: runner.load_task consumes project_root / verify_commands
# / task_definition / modules / plan, never `project` itself.
_CANONICAL_PROJECT = {'df': 'dark-factory', 'reify': 'reify'}

# Per-repo standard gate command sets — byte-for-byte the commands the existing
# fixtures already hardcode (df_task_12 / reify_task_27), so a landed task's
# gates are captured as the known-good per-repo standard rather than re-run.
_DEFAULT_VERIFY_COMMANDS: dict[str, dict[str, str]] = {
    'df': {
        'test': 'cd orchestrator && uv run pytest tests/ -x',
        'lint': 'cd orchestrator && uv run ruff check src/',
        'typecheck': 'cd orchestrator && uv run pyright src/',
    },
    'reify': {
        'test': 'cargo test --workspace',
        'lint': 'cargo clippy --workspace',
        'typecheck': '',
    },
}


@dataclass
class ReferenceCapture:
    """The judge's ``reference`` contender provenance for a fixture.

    Stores the landed ``post_task_commit`` plus a compact diff stat
    (``files`` / ``insertions`` / ``deletions``) — NOT the full inline diff,
    which would bloat the fixture JSON and which stays retrievable from the
    pinned ``evals/<id>`` branch via ``materialize_reference_diff`` (git glue,
    a later slice). Produced by ``capture_reference`` over ``pre..post``.
    """

    post_task_commit: str
    files: int
    insertions: int
    deletions: int


def default_verify_commands(repo: str) -> dict[str, str]:
    """Return the per-repo standard gate command set (``{test,lint,typecheck}``).

    ``df`` → pytest / ruff / pyright; ``reify`` → cargo test / clippy. Raises
    ``ValueError`` on an unknown repo (loud-over-silent). Returns a fresh copy
    so callers can't mutate the module-level template.
    """
    try:
        return dict(_DEFAULT_VERIFY_COMMANDS[repo])
    except KeyError:
        raise ValueError(
            f'default_verify_commands: unknown repo {repo!r} (expected df or reify)'
        ) from None


def landed_verify_outcome(verify_commands: dict[str, str]) -> dict:
    """Return the landed-status verify outcome for a fixture.

    Ground truth is free: the task merged to ``main`` ⇒ its gates passed at
    the post commit. So the outcome is ``{source:'landed', passed:True,
    commands:<verify_commands>}`` — never a live re-run of a 6-week-old SHA
    (env-dependent, slow, non-deterministic).
    """
    return {'source': 'landed', 'passed': True, 'commands': dict(verify_commands)}


def build_fixture_record(
    candidate: CompletedTaskCandidate,
    reference: ReferenceCapture,
    verify_commands: dict[str, str],
    plan: dict | None,
    cohort: str,
    sampled_at: str,
    seed: int = 0,
    plan_source: str | None = None,
) -> dict:
    """Emit the canonical eval-fixture record (a JSON-serializable dict).

    Matches the existing ``evals/tasks/*.json`` schema (so ``runner.load_task``
    + ``run_eval`` consume it unchanged) — ``id`` = ``<repo>_task_<task_id>``,
    ``name``, ``project``, ``project_root``, ``pre_task_commit``,
    ``post_task_commit``, ``task_definition``, ``verify_commands``, ``modules``,
    ``complexity`` — extended with the revival-ζ fields: ``cohort``,
    ``provenance`` (``merge_sha`` / ``sampled_at`` / ``seed`` / ``plan_source``),
    a compact ``reference`` block (post commit + diff stat), and a landed
    ``verify_outcome``.

    The frozen ``plan`` is captured opportunistically: when present it is
    embedded verbatim and ``provenance.plan_source`` defaults to ``'embedded'``;
    when ``None`` the record carries ``plan: null`` and
    ``provenance.plan_source == 'unavailable'`` (a completed task's
    ``.task/plan.json`` is gitignored, scrubbed post-merge, and not persisted
    in fused-memory, so it is frequently unrecoverable). Callers may pass an
    explicit *plan_source* (e.g. ``'transcript_archive'``) to record where a
    present plan was recovered from.
    """
    repo = repo_of(candidate)
    if plan_source is None:
        plan_source = 'embedded' if plan is not None else 'unavailable'
    return {
        'id': f'{repo}_task_{candidate.task_id}',
        'name': candidate.title,
        'project': _CANONICAL_PROJECT[repo],
        'project_root': candidate.project_root,
        'pre_task_commit': candidate.pre_commit,
        'post_task_commit': candidate.post_commit,
        'task_definition': {
            'title': candidate.title,
            'description': candidate.description,
        },
        'verify_commands': dict(verify_commands),
        'modules': list(candidate.modules),
        'complexity': candidate.complexity,
        'cohort': cohort,
        'provenance': {
            'merge_sha': candidate.merge_sha,
            'sampled_at': sampled_at,
            'seed': seed,
            'plan_source': plan_source,
        },
        'reference': {
            'post_task_commit': reference.post_task_commit,
            'diff_stat': {
                'files': reference.files,
                'insertions': reference.insertions,
                'deletions': reference.deletions,
            },
        },
        'verify_outcome': landed_verify_outcome(verify_commands),
        'plan': plan,
    }


# ---------------------------------------------------------------------------
# Git glue — merge-log discovery, commit resolution, reference capture, pin
# ---------------------------------------------------------------------------

# A completed-task merge subject: ``Merge task/<id> into main``. The id is the
# task id (may be non-numeric for legacy fixtures, so ``\S+``).
_MERGE_SUBJECT_RE = re.compile(r'Merge task/(\S+) into main')


async def resolve_task_commits_from_merge(
    repo_root: Path | str, merge_sha: str,
) -> tuple[str, str]:
    """Resolve ``(pre, post)`` = ``(M^1, M)`` for merge commit *merge_sha*.

    ``pre = M^1`` is prior main (the eval baseline
    ``snapshots.create_eval_worktree`` checks out); ``post = M`` is the landed
    state (where ``evals/<id>`` is pinned). Both are returned as full SHAs.
    """
    root = Path(repo_root)
    pre = await _git_run(['git', 'rev-parse', f'{merge_sha}^1'], cwd=root)
    post = await _git_run(['git', 'rev-parse', merge_sha], cwd=root)
    return pre, post


async def capture_reference(
    repo_root: Path | str, pre: str, post: str,
) -> ReferenceCapture:
    """Capture the compact ``pre..post`` reference block (no full inline diff).

    Parses ``git diff --numstat pre..post`` into a ``(files, insertions,
    deletions)`` diff stat. Binary files (numstat ``-`` columns) count toward
    ``files`` but contribute no line counts. The full diff stays retrievable
    via :func:`materialize_reference_diff` off the pinned branch.
    """
    root = Path(repo_root)
    numstat = await _git_run(['git', 'diff', '--numstat', f'{pre}..{post}'], cwd=root)
    files = insertions = deletions = 0
    for line in numstat.splitlines():
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        added, deleted = parts[0], parts[1]
        files += 1
        if added != '-':
            insertions += int(added)
        if deleted != '-':
            deletions += int(deleted)
    post_full = await _git_run(['git', 'rev-parse', post], cwd=root)
    return ReferenceCapture(
        post_task_commit=post_full,
        files=files,
        insertions=insertions,
        deletions=deletions,
    )


async def materialize_reference_diff(
    repo_root: Path | str, pre: str, post: str,
) -> str:
    """Return the full ``git diff pre..post`` — the judge's reference diff.

    Delegates to ``snapshots.get_diff_between_commits`` (no worktree needed),
    the same helper the eval pipeline already uses; the diff stays retrievable
    from the pinned ``evals/<id>`` branch so the judge grades a non-empty
    landed diff via its ``result.get('diff')`` reference-contender short-circuit.
    """
    return await get_diff_between_commits(Path(repo_root), pre, post)


async def pin_eval_branch(
    repo_root: Path | str, fixture_id: str, post: str,
) -> str:
    """Idempotently pin branch ``evals/<fixture_id>`` at *post*; return its name.

    Uses ``git update-ref`` (not ``git branch``), so a second call re-points
    the ref to the same SHA without erroring — the branch is a stable, retriev-
    able handle on the landed commit for the judge's reference contender.
    """
    root = Path(repo_root)
    branch = f'evals/{fixture_id}'
    post_full = await _git_run(['git', 'rev-parse', post], cwd=root)
    await _git_run(
        ['git', 'update-ref', f'refs/heads/{branch}', post_full], cwd=root,
    )
    return branch


async def discover_completed_tasks(
    repo_root: Path | str,
    since: str | None = None,
    *,
    project: str,
    project_root: str | None = None,
) -> list[CompletedTaskCandidate]:
    """Discover completed-task merges via the git log (read-only).

    Runs ``git log --grep='Merge task/' [--since=<since>]`` and parses each
    ``Merge task/<id> into main`` subject into a :class:`CompletedTaskCandidate`
    STUB carrying ``task_id`` + ``merge_sha`` + resolved ``pre_commit`` (M^1) /
    ``post_commit`` (M). ``title`` / ``description`` / ``complexity`` /
    ``modules`` are left empty for a downstream enrichment pass (the CLI fills
    them from the task DB before classifying); a stub alone classifies to the
    default cell.

    *project* / *project_root* tag the candidates with their repo of origin
    (``project_root`` defaults to *repo_root*), so branch pinning later lands
    in each fixture's own checkout.
    """
    root = Path(repo_root)
    resolved_root = project_root or str(root)
    cmd = ['git', 'log', '--grep=Merge task/', '--pretty=%H %s']
    if since:
        cmd.insert(2, f'--since={since}')
    out = await _git_run(cmd, cwd=root)

    candidates: list[CompletedTaskCandidate] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, subject = line.partition(' ')
        match = _MERGE_SUBJECT_RE.search(subject)
        if not match:
            continue
        pre = await _git_run(['git', 'rev-parse', f'{sha}^1'], cwd=root)
        candidates.append(CompletedTaskCandidate(
            task_id=match.group(1),
            project=project,
            project_root=resolved_root,
            pre_commit=pre,
            post_commit=sha,
            merge_sha=sha,
        ))
    return candidates


def enrich_candidates_from_task_db(
    candidates: list[CompletedTaskCandidate], db_path: Path | str,
) -> list[CompletedTaskCandidate]:
    """Fill ``title`` / ``description`` / ``complexity`` from a taskmaster db.

    :func:`discover_completed_tasks` yields STUBS — the merge log carries only
    the task id, not its title/description/complexity — so the three
    classifiers would collapse every stub into the default cell. This
    best-effort pass looks each *numeric* task id up in the ``tasks`` table at
    *db_path* (``<project_root>/.taskmaster/tasks/tasks.db``) and returns copies
    with those fields filled (``complexity`` read from the row's ``metadata``
    JSON blob). Opened ``mode=ro`` (mirroring
    ``curator_corpus.read_curator_decisions``) so it never locks the live db.

    Degrades to the UNCHANGED stub — never raises — for a non-numeric id
    (legacy fixtures), an id absent from the db, or a missing/unreadable db: a
    stub still classifies (to the default cell) rather than crashing the cut.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning(
            'enrich_candidates_from_task_db: no task db at %s; keeping %d '
            'candidate(s) as stubs', db_path, len(candidates),
        )
        return list(candidates)

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                'SELECT id, title, description, metadata FROM tasks'
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning(
            'enrich_candidates_from_task_db: read of %s failed (%s); keeping stubs',
            db_path, exc,
        )
        return list(candidates)

    by_id = {str(row['id']): row for row in rows}

    enriched: list[CompletedTaskCandidate] = []
    for cand in candidates:
        row = by_id.get(str(cand.task_id).strip())
        if row is None:
            enriched.append(cand)
            continue
        complexity = cand.complexity
        raw_meta = row['metadata']
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                meta = None
            if isinstance(meta, dict) and meta.get('complexity'):
                complexity = str(meta['complexity'])
        enriched.append(replace(
            cand,
            title=row['title'] or cand.title,
            description=row['description'] or cand.description,
            complexity=complexity,
        ))
    return enriched


# ---------------------------------------------------------------------------
# Listing + audit surface — stratification counts, table render, corpus audit
# ---------------------------------------------------------------------------

def _candidate_from_fixture(fixture: dict) -> CompletedTaskCandidate:
    """Reconstruct a classification-only candidate from a fixture dict.

    Only the fields the three classifiers read (``project`` / ``title`` /
    ``description`` / ``complexity``) are populated; the git/provenance fields
    are irrelevant to :func:`cell_of` and left empty. ``task_id`` carries the
    fixture ``id`` purely for error messages (``repo_of`` names it on an
    unrecognised project).
    """
    task_def = fixture.get('task_definition') or {}
    return CompletedTaskCandidate(
        task_id=str(fixture.get('id', '')),
        project=str(fixture.get('project', '')),
        project_root=str(fixture.get('project_root', '')),
        title=str(task_def.get('title', '')),
        description=str(task_def.get('description', '')),
        complexity=fixture.get('complexity'),
        modules=list(fixture.get('modules') or []),
    )


@dataclass
class StratificationCounts:
    """Per-cell counts of a fixture corpus, plus totals and plan-capture tally.

    ``cells`` maps each observed ``(repo, kind, path)`` cell to its count
    (Counter-style — an absent cell is simply not a key; :func:`format_
    stratification_table` fills the zeros over the full axis product).
    ``plan_captured`` counts fixtures carrying a non-``None`` ``plan`` (the
    subset usable by the downstream implementer eval). ``cohort`` records the
    scope filter applied (``None`` = all cohorts).
    """

    cells: dict[Cell, int]
    total: int
    plan_captured: int
    cohort: str | None = None


def stratification_counts(
    fixtures: list[dict], cohort: str | None = None,
) -> StratificationCounts:
    """Count *fixtures* per ``(repo, kind, path)`` cell (optionally cohort-scoped).

    Reclassifies each fixture through the same axis functions the sampler used
    (:func:`cell_of` over a :func:`_candidate_from_fixture` view), so the
    listing reflects the live classification rather than a stored label. When
    *cohort* is given, only fixtures whose ``cohort`` field matches are counted
    — letting ``eval-list-fixtures --cohort`` report the fresh near-HEAD
    stratification cleanly without the retained legacy fixtures muddying it.
    """
    scoped = [
        f for f in fixtures if cohort is None or f.get('cohort') == cohort
    ]
    cells = Counter(cell_of(_candidate_from_fixture(f)) for f in scoped)
    plan_captured = sum(1 for f in scoped if f.get('plan') is not None)
    return StratificationCounts(
        cells=dict(cells),
        total=len(scoped),
        plan_captured=plan_captured,
        cohort=cohort,
    )


def format_stratification_table(counts: StratificationCounts) -> str:
    """Render *counts* as a stable, human-readable fixed-width table.

    Iterates the full 12-cell axis product in a fixed order (so EVERY cell —
    including empty ones — renders as a stable row), then a totals line
    carrying the grand total and the plan-captured tally. Deterministic: the
    same :class:`StratificationCounts` always renders byte-identically (no
    dict-iteration-order or wall-clock dependence).
    """
    scope = counts.cohort if counts.cohort is not None else 'all cohorts'
    rule = f'  {"-" * 6} {"-" * 9} {"-" * 7} {"-" * 5}'
    lines = [
        f'stratification ({scope}):',
        f'  {"repo":<6} {"kind":<9} {"path":<7} {"count":>5}',
        rule,
    ]
    for cell in _ALL_CELLS:
        repo, kind, path = cell
        lines.append(
            f'  {repo:<6} {kind:<9} {path:<7} {counts.cells.get(cell, 0):>5}'
        )
    lines.append(rule)
    lines.append(
        f'  TOTAL: {counts.total}   plan_captured: {counts.plan_captured}'
    )
    return '\n'.join(lines)


# A predicate ``(project_root, ref) -> bool`` — injected into
# :func:`audit_fixture_corpus` so the branch-presence check is hermetic in
# tests. :func:`git_ref_exists` is the production wiring.
RefExistsFn = Callable[[str, str], bool]


def git_ref_exists(project_root: str, ref: str) -> bool:
    """Return True iff *ref* resolves to a commit in the repo at *project_root*.

    A synchronous, read-only ``git rev-parse --verify`` probe, used to wire
    :func:`audit_fixture_corpus`'s ``ref_exists`` predicate to a real checkout
    at the CLI boundary. Returns False (never raises) when the ref is absent or
    *project_root* isn't a git repo, so a missing pin surfaces as a named audit
    failure rather than crashing the listing command.
    """
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}'],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


@dataclass
class FixtureAuditReport:
    """Structured result of :func:`audit_fixture_corpus`'s integrity checks.

    ``ok`` is True only when every check passes; otherwise ``failures`` names
    each failing check (band violation, or a per-fixture
    ``missing_reference:<id>`` / ``missing_verify_outcome:<id>`` /
    ``missing_branch:<id>``) so a CLI/CI caller reports exactly what's wrong
    rather than a bare pass/fail. Mirrors ``curator_corpus.AuditReport``.
    """

    ok: bool
    count: int
    failures: list[str] = field(default_factory=list)


def audit_fixture_corpus(
    fixtures: list[dict],
    *,
    target_low: int = 10,
    target_high: int = 14,
    ref_exists: RefExistsFn | None = None,
) -> FixtureAuditReport:
    """Audit a fixture corpus for band + per-fixture completeness.

    Checks, each contributing a named reason to ``FixtureAuditReport.failures``
    when violated:

    - ``band``                       — the corpus size is within
      ``[target_low, target_high]`` (the revival-ζ [10,14] band).
    - ``missing_reference:<id>``      — every fixture carries a ``reference``
      block (the judge's reference-contender provenance).
    - ``missing_verify_outcome:<id>`` — every fixture carries a
      ``verify_outcome`` block (the deterministic gate's landed ground truth).
    - ``missing_branch:<id>``         — the fixture's ``evals/<id>`` branch is
      pinned in its own ``project_root``. Only checked when *ref_exists* is
      supplied; when ``None`` the branch check is SKIPPED (a structural-only
      audit for callers without a git checkout, e.g. a bare listing).

    Never raises: an empty corpus simply fails ``band``. The branch predicate
    is dependency-injected (:func:`git_ref_exists` is the production wiring) so
    the audit is hermetic in tests.
    """
    failures: list[str] = []
    count = len(fixtures)

    if not (target_low <= count <= target_high):
        failures.append(
            f'band: count {count} outside [{target_low},{target_high}]'
        )

    for fixture in fixtures:
        fid = fixture.get('id', '<unknown>')
        if not fixture.get('reference'):
            failures.append(f'missing_reference:{fid}')
        if not fixture.get('verify_outcome'):
            failures.append(f'missing_verify_outcome:{fid}')
        if ref_exists is not None:
            branch = f'evals/{fid}'
            if not ref_exists(str(fixture.get('project_root', '')), branch):
                failures.append(f'missing_branch:{fid}')

    return FixtureAuditReport(ok=not failures, count=count, failures=failures)
