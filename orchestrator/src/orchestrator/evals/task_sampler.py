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

import re
from dataclasses import dataclass, field

from orchestrator.agents.triage import has_simple_task_blocker

__all__ = [
    'CompletedTaskCandidate',
    'classify_kind',
    'classify_path',
    'repo_of',
]


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
