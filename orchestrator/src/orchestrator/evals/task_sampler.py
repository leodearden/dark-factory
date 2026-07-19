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

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from orchestrator.agents.triage import has_simple_task_blocker

__all__ = [
    'Cell',
    'CompletedTaskCandidate',
    'SampleResult',
    'cell_of',
    'classify_kind',
    'classify_path',
    'repo_of',
    'sample_stratified',
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
