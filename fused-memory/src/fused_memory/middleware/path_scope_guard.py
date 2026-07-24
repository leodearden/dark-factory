"""Multi-project path-scope guard.

Generalises the original task-1088 dark-factory-only guard (its back-compat
re-export shim module was retired in task 2208 / PRD D2) to a per-project
prefix registry built from configured ``known_project_roots``.

A candidate is rejected when its title / description / details / files
mention a path prefix owned by a project other than the one being filed
into.  The verdict's ``suggested_project`` field carries the owning
project_id of the first mismatched prefix so the caller can resubmit (or
the LLM can re-route under the multi-project Stage 2 prompt).

Wired into :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`
at the ``submit_task`` entry point.  Path-guard
rejections also fire a ``scope_violation`` escalation via
:class:`fused_memory.middleware.scope_violation_escalator.ScopeViolationEscalator`
when one is configured.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask


# ---------------------------------------------------------------------------
# Pattern cache
# ---------------------------------------------------------------------------

_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _build_pattern(prefixes: tuple[str, ...]) -> re.Pattern[str]:
    """Build (and cache) a compiled regex for the given prefix tuple.

    Each prefix is anchored by a leading word boundary: start-of-string OR
    a character that is NOT ``[A-Za-z0-9_\\-/.]``.  This means a prefix
    matches ONLY as a *leading* path component — at start-of-string or after
    whitespace, punctuation, quotes, colons, etc. — but NOT when preceded by a
    path separator (``/``) or a dot (``.``).

    Excluding ``/`` and ``.`` from the boundary class ensures that mid-path
    occurrences like ``vendor/corpus/expr.txt`` or ``x.corpus/foo`` do NOT
    match the bare prefix ``corpus/``, eliminating the false-positive routing
    of tasks whose text merely passes *through* a known-project directory.

    Note: ``\\-`` is kept escaped (literal hyphen, no range); ``/`` and ``.``
    are plain literals inside the class.

    Cache note: ``_PATTERN_CACHE`` is keyed by the prefix tuple; the boundary
    class is a compile-time constant (not a runtime variable), so no cache
    invalidation or versioning is needed when the pattern template changes —
    only new prefixes cause a new compile.
    """
    if prefixes not in _PATTERN_CACHE:
        alts = '|'.join(re.escape(p) for p in prefixes)
        pattern = re.compile(rf'(?:^|(?<=[^A-Za-z0-9_\-/.]))({alts})')
        _PATTERN_CACHE[prefixes] = pattern
    return _PATTERN_CACHE[prefixes]


def find_paths(text: str, prefixes: tuple[str, ...]) -> list[str]:
    """Scan *text* for any of *prefixes* and return ordered, deduplicated matches.

    Empty *text* or empty *prefixes* short-circuit to ``[]``.
    """
    if not text or not prefixes:
        return []
    pattern = _build_pattern(prefixes)
    seen: set[str] = set()
    result: list[str] = []
    for match in pattern.finditer(text):
        prefix = match.group(1)
        if prefix not in seen:
            seen.add(prefix)
            result.append(prefix)
    return result


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

# Kept stable for one merge cycle so existing callers branching on
# ``error_type='DarkFactoryPathScopeViolation'`` continue to work.  Renaming
# to a neutral name is a tracked followup.
_ERROR_TYPE: str = 'DarkFactoryPathScopeViolation'


@dataclass(frozen=True)
class PathGuardVerdict:
    """Outcome of a multi-project path-scope check.

    Mirrors the original dark-factory-only ``PathGuardVerdict`` shape so
    existing test fixtures keep working; adds ``suggested_project`` to
    carry the owning project_id of the first mismatched prefix.

    Fields:
        outcome: ``'ok'`` or ``'rejection'``.
        project_id: The project_id that was checked (the wrong one on rejection).
        matched_paths: Tuple of path prefixes found in the candidate that
            belonged to a project other than ``project_id``.
        suggested_project: project_id whose prefix matched first; ``None``
            when the matched prefix has no owner in the registry, or when
            multiple owners were matched and we cannot pick one.
        error_type: Stable error-type string for callers to branch on.
    """

    outcome: Literal['ok', 'rejection']
    project_id: str = ''
    matched_paths: tuple[str, ...] = field(default_factory=tuple)
    suggested_project: str | None = None
    error_type: str = _ERROR_TYPE

    @property
    def is_rejection(self) -> bool:
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Return a structured MCP error dict, or ``{}`` on ok.

        The error message names the matched paths, the wrong project_id,
        and the suggested target project (when known) so the caller can
        re-submit without a second round-trip to figure out where the
        task belongs.
        """
        if not self.is_rejection:
            return {}
        paths_str = ', '.join(self.matched_paths)
        if self.suggested_project:
            tail = f'Resubmit to the {self.suggested_project} project instead.'
        else:
            tail = (
                'No single suggested target — the matched prefixes belong to '
                'multiple projects or none.  Inspect matched_paths and route '
                'manually.'
            )
        return {
            'error': (
                f'{self.error_type}: task references paths owned by another '
                f'project ({paths_str}) but was filed under project '
                f'{self.project_id!r}. {tail}'
            ),
            'error_type': self.error_type,
            'project_id': self.project_id,
            'matched_paths': list(self.matched_paths),
            'suggested_project': self.suggested_project,
        }


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _aggregate_owner_mismatches(
    items: list[str],
    project_id: str,
    owner_of: Callable[[str], str | None],
) -> tuple[list[str], str | None]:
    """Classify *items* via *owner_of* and collect those owned by another project.

    Shared aggregation core for :func:`_resolve_mismatches` (matched
    prefixes from the regex-over-prose heuristic, classified via
    :meth:`ProjectPrefixRegistry.project_for_prefix`) and
    :func:`check_files_for_scope` (concrete file paths, classified via
    :meth:`ProjectPrefixRegistry.project_for_path`).  The two callers differ
    only in *what* ``owner_of`` resolves, not in how mismatches are
    aggregated into a suggested target — factored out here (task 2206
    review) so that logic can't drift out of sync between them.

    Returns ``(mismatched_items, suggested_project)``.  An item is
    mismatched when ``owner_of(item)`` returns a project_id other than
    *project_id*; items with no registered owner (``owner_of`` returns
    ``None``) are dropped from the result — an unclassifiable item is never
    grounds for rejection, keeping the guard conservative.
    ``suggested_project`` is the single shared owner of every mismatch, or
    ``None`` when mismatches span more than one project.
    """
    mismatched: list[str] = []
    owners: set[str] = set()
    first_owner: str | None = None
    for item in items:
        owner = owner_of(item)
        if owner is None or owner == project_id:
            continue
        mismatched.append(item)
        if first_owner is None:
            first_owner = owner
        owners.add(owner)
    suggested = first_owner if len(owners) == 1 else None
    return mismatched, suggested


def _resolve_mismatches(
    matched: list[str], project_id: str, registry: ProjectPrefixRegistry,
) -> tuple[list[str], str | None]:
    """Filter *matched* prefixes to those owned by some other project.

    Returns ``(mismatched_prefixes, suggested_project)``.  ``suggested_project``
    is the owner of the first mismatch when all mismatches share an owner;
    ``None`` when they span multiple projects.  Prefixes with no registered
    owner are dropped from the rejection (the registry could not classify
    them — ignoring them keeps the guard conservative).

    Thin wrapper around :func:`_aggregate_owner_mismatches` bound to the
    prefix-string lookup :meth:`ProjectPrefixRegistry.project_for_prefix`.
    """
    return _aggregate_owner_mismatches(matched, project_id, registry.project_for_prefix)


def check_candidate_for_scope(
    candidate: CandidateTask,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject *candidate* when it cites paths owned by another project.

    Scans ``title``, ``description``, ``details``, and
    ``files_to_modify``.  Returns ``ok`` when the registry is empty, when
    no prefixes match, or when every match is owned by ``project_id``.
    """
    if not registry:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    parts: list[str] = [
        candidate.title or '',
        candidate.description or '',
        candidate.details or '',
    ]
    parts.extend(candidate.files_to_modify or [])
    combined = '\n'.join(parts)

    matched = find_paths(combined, registry.all_prefixes())
    mismatched, suggested = _resolve_mismatches(matched, project_id, registry)
    if mismatched:
        return PathGuardVerdict(
            outcome='rejection',
            project_id=project_id,
            matched_paths=tuple(mismatched),
            suggested_project=suggested,
        )
    return PathGuardVerdict(outcome='ok', project_id=project_id)


def check_files_for_scope(
    files: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject when any of *files* is owned by a project other than *project_id*.

    CERTAIN counterpart to :func:`check_candidate_for_scope` /
    :func:`check_text_for_scope`: classifies each concrete file path via
    :meth:`ProjectPrefixRegistry.project_for_path` (exact leading-path-
    component match) instead of the heuristic regex-over-prose
    :func:`find_paths`.  Used by the interceptor's FILES-certain check
    (task 2206) to hard-reject a submission whose ``metadata.files`` name a
    path under a KNOWN other project's tree — no LLM adjudication, since an
    exact owner lookup leaves nothing to adjudicate.

    Returns ``ok`` when the registry is empty/falsy, *files* is empty/falsy,
    or every file is either unowned or owned by *project_id*.  On a
    mismatch, ``matched_paths`` carries the offending file paths (not
    prefixes) and ``suggested_project`` is the single owner when all
    mismatches share one, else ``None``.
    """
    if not registry or not files:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    mismatched: list[str] = []
    owners: set[str] = set()
    first_owner: str | None = None
    for f in files:
        owner = registry.project_for_path(f)
        if owner is None or owner == project_id:
            continue
        mismatched.append(f)
        if first_owner is None:
            first_owner = owner
        owners.add(owner)

    if not mismatched:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    suggested = first_owner if len(owners) == 1 else None
    return PathGuardVerdict(
        outcome='rejection',
        project_id=project_id,
        matched_paths=tuple(mismatched),
        suggested_project=suggested,
    )


def all_files_foreign_owner(
    files: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> str | None:
    """Return the single foreign owner iff EVERY owned file belongs to ONE other project.

    Recognises the cross-repo deliverable shape (task 3004 / reify-task 5308):
    a task filed under *project_id* whose ``metadata.files`` are ALL owned by
    one OTHER registered project, so its own branch is legitimately empty (the
    deliverable lands on that project's branch).  The interceptor consumes the
    returned owner to TAG the submission (``metadata.cross_repo`` +
    ``cross_repo_project``) and ALLOW it, instead of hard-rejecting via
    :func:`check_files_for_scope`.

    FILER MUST BE REGISTERED (task 3004 / esc-3004 NARROW decision): the
    allow-and-tag path fires ONLY when *project_id* is itself a registered
    project (``registry.is_known(project_id)``).  A cross-repo deliverable is a
    relationship between two KNOWN projects (e.g. reify → dark_factory), not an
    open door for any namespace to declare another project's files.  An
    UNREGISTERED filer whose files are all foreign therefore returns ``None``
    here and falls through to the :func:`check_files_for_scope` hard reject,
    preserving task-2206's anti-bypass guard.

    Distinct from :func:`check_files_for_scope`, which rejects on ANY foreign
    file: here the submission must be ENTIRELY foreign under a SINGLE owner,
    with NO locally-owned file, before it qualifies.  Reuses
    :meth:`ProjectPrefixRegistry.project_for_path` exactly as
    ``check_files_for_scope`` does; files with no registered owner
    (``project_for_path`` returns ``None``) are neutral — they neither block
    nor establish cross-repo (conservative, matching
    :func:`_aggregate_owner_mismatches`).

    Returns ``None`` when the registry/*files* are empty, when the FILER
    (*project_id*) is not a registered project, when ANY file is owned by
    *project_id* (a genuine same-repo or mixed scope, left to
    ``check_files_for_scope``'s hard-reject), when no file is foreign, or when
    foreign files span more than one owner.
    """
    if not registry or not files:
        return None
    if not registry.is_known(project_id):
        # Unregistered filer → task-2206 anti-bypass preserved; a cross-repo
        # deliverable is a relationship between two REGISTERED projects, not a
        # blanket allowance for any namespace to declare foreign files.
        return None
    foreign_owners: set[str] = set()
    first_owner: str | None = None
    for f in files:
        owner = registry.project_for_path(f)
        if owner is None:
            continue  # unowned → neutral, never grounds for classification
        if owner == project_id:
            return None  # a locally-owned file → not a pure cross-repo deliverable
        if first_owner is None:
            first_owner = owner
        foreign_owners.add(owner)
    # len 0 (no foreign file) or >1 (spans multiple owners) → not cross-repo.
    if len(foreign_owners) != 1:
        return None
    return first_owner


def is_routing_override(routing_override_reason: str | None) -> bool:
    """Return True when *routing_override_reason* constitutes a deliberate bypass.

    A reason is considered present (and the path guards should be skipped) when
    it is a non-empty string after stripping leading/trailing whitespace.
    Empty strings, None, and whitespace-only strings all return False —
    preserving the default no-override semantics.

    Use only when sure the task belongs to the submitting project.
    If unsure, escalate rather than risking a mis-filed task.
    """
    return bool((routing_override_reason or '').strip())


def check_text_for_scope(
    text: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject *text* when it cites paths owned by another project.

    Prompt-only counterpart to :func:`check_candidate_for_scope` for the
    ``prompt``-only ``submit_task`` branch where no ``title`` is supplied
    (and therefore no ``CandidateTask`` is built).
    """
    if not registry:
        return PathGuardVerdict(outcome='ok', project_id=project_id)
    matched = find_paths(text or '', registry.all_prefixes())
    mismatched, suggested = _resolve_mismatches(matched, project_id, registry)
    if mismatched:
        return PathGuardVerdict(
            outcome='rejection',
            project_id=project_id,
            matched_paths=tuple(mismatched),
            suggested_project=suggested,
        )
    return PathGuardVerdict(outcome='ok', project_id=project_id)
