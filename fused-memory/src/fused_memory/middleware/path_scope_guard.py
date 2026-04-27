"""Multi-project path-scope guard.

Generalises the dark-factory-only guard from task 1088 (now a re-export
shim at :mod:`fused_memory.middleware.dark_factory_path_guard`) to a
per-project prefix registry built from configured ``known_project_roots``.

A candidate is rejected when its title / description / details / files
mention a path prefix owned by a project other than the one being filed
into.  The verdict's ``suggested_project`` field carries the owning
project_id of the first mismatched prefix so the caller can resubmit (or
the LLM can re-route under the multi-project Stage 2 prompt).

Wired into :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`
at the ``submit_task`` and ``add_subtask`` entry points.  Path-guard
rejections also fire a ``scope_violation`` escalation via
:class:`fused_memory.middleware.scope_violation_escalator.ScopeViolationEscalator`
when one is configured.
"""

from __future__ import annotations

import re
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
    a character that is NOT ``[A-Za-z0-9_-]``.  Stops e.g.
    ``foo-orchestrator/`` from matching ``orchestrator/`` while still
    matching bare ``orchestrator/`` after whitespace or punctuation.
    """
    if prefixes not in _PATTERN_CACHE:
        alts = '|'.join(re.escape(p) for p in prefixes)
        pattern = re.compile(rf'(?:^|(?<=[^A-Za-z0-9_\-]))({alts})')
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


def _resolve_mismatches(
    matched: list[str], project_id: str, registry: ProjectPrefixRegistry,
) -> tuple[list[str], str | None]:
    """Filter *matched* prefixes to those owned by some other project.

    Returns ``(mismatched_prefixes, suggested_project)``.  ``suggested_project``
    is the owner of the first mismatch when all mismatches share an owner;
    ``None`` when they span multiple projects.  Prefixes with no registered
    owner are dropped from the rejection (the registry could not classify
    them — ignoring them keeps the guard conservative).
    """
    mismatched: list[str] = []
    owners: set[str] = set()
    first_owner: str | None = None
    for prefix in matched:
        owner = registry.project_for_prefix(prefix)
        if owner is None or owner == project_id:
            continue
        mismatched.append(prefix)
        if first_owner is None:
            first_owner = owner
        owners.add(owner)
    suggested = first_owner if len(owners) == 1 else None
    return mismatched, suggested


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
