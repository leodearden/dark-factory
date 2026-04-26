"""Dark-factory path-scope guardrail.

Rejects tasks that reference dark-factory module paths when filed into a
non-dark-factory project tree.  This is a pre-filing guard wired into
:class:`~fused_memory.middleware.task_interceptor.TaskInterceptor` at the
``submit_task`` and ``add_subtask`` entry points.

Motivation — four scope_violation escalations filed against the *reify*
project on 2026-04-26:

  * esc-2240-37  (targeted ``orchestrator/harness.py``)
  * esc-2246-47  (targeted ``fused-memory/.../reconciliation/``)
  * esc-2249-48  (targeted ``fused-memory/.../reconciliation/``)
  * esc-2254-58  (targeted ``fused-memory/.../middleware/task_curator.py``)

Each required an agent-hour plus reconciliation cycles to cancel.  The guard
short-circuits the filing and returns a structured
``DarkFactoryPathScopeViolation`` error so the caller can resubmit to the
correct project.

Design decisions — see plan.json design_decisions for rationale:
  - Reject (not auto-route or flag)
  - Word-boundary anchoring for prefix matching
  - Hard-coded prefix list (not config-driven) with ``prefixes`` kwarg for tests
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DARK_FACTORY_PROJECT_ID: str = 'dark_factory'
"""The project_id derived by ``resolve_project_id`` for ``/home/leo/src/dark-factory``."""

DARK_FACTORY_PATH_PREFIXES: tuple[str, ...] = (
    'orchestrator/',
    'fused-memory/',
    'fused_memory/',
    'mem0/',
    'graphiti/',
    'taskmaster-ai/',
    'taskmaster_ai/',
    'dashboard/',
)
"""Directory prefixes that uniquely identify dark-factory subprojects.

The leading word-boundary check in :func:`find_dark_factory_paths` prevents
siblings like ``foo-orchestrator/`` from matching ``orchestrator/``.

Intentionally **excludes** generic directory names that commonly appear in
non-dark-factory projects: ``shared/``, ``skills/``, ``review/``,
``hooks/``, ``escalation/``.  A task referencing ``hooks/pre-commit.sh``
or ``shared/utils.py`` should not be rejected as a scope violation.
"""

# ---------------------------------------------------------------------------
# find_dark_factory_paths
# ---------------------------------------------------------------------------

# Pre-compiled regex cache: maps a tuple of prefixes to the compiled pattern.
_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _build_pattern(prefixes: tuple[str, ...]) -> re.Pattern[str]:
    """Build (and cache) a compiled regex for the given prefix tuple."""
    if prefixes not in _PATTERN_CACHE:
        # Alternation of re-escaped prefixes, each anchored by a leading
        # word boundary: start-of-string OR a character that is NOT
        # [A-Za-z0-9_-].  This stops ``foo-orchestrator/`` from matching
        # ``orchestrator/`` while still matching bare ``orchestrator/`` at the
        # start of a string or after whitespace/punctuation.
        alts = '|'.join(re.escape(p) for p in prefixes)
        pattern = re.compile(
            rf'(?:^|(?<=[^A-Za-z0-9_\-]))({alts})',
        )
        _PATTERN_CACHE[prefixes] = pattern
    return _PATTERN_CACHE[prefixes]


def find_dark_factory_paths(
    text: str,
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
) -> list[str]:
    """Scan *text* for dark-factory path prefixes and return ordered, deduplicated matches.

    Args:
        text: Any string — task title, description, details, file path, etc.
        prefixes: Tuple of path prefixes to scan for.  Defaults to
            :data:`DARK_FACTORY_PATH_PREFIXES`.

    Returns:
        A list of the matched prefixes in the order of their first appearance
        in *text*, with duplicates removed.  Empty when no match is found.
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
# PathGuardVerdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathGuardVerdict:
    """Outcome of a path-scope guard check.

    Mirrors the :class:`~fused_memory.reconciliation.backlog_policy.BacklogVerdict`
    pattern: a frozen dataclass with an ``is_rejection`` property and a
    ``to_error_dict()`` method that returns ``{}`` on ok and a structured
    error payload on rejection.

    Fields:
        outcome: ``'ok'`` or ``'rejection'``.
        project_id: The project_id that was checked (the wrong one on rejection).
        matched_paths: Tuple of dark-factory path prefixes found in the candidate.
        error_type: Stable error-type string for callers to branch on.
    """

    outcome: Literal['ok', 'rejection']
    project_id: str = ''
    matched_paths: tuple[str, ...] = field(default_factory=tuple)
    error_type: str = 'DarkFactoryPathScopeViolation'

    @property
    def is_rejection(self) -> bool:
        """True iff ``outcome == 'rejection'``."""
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Return a structured MCP error dict, or ``{}`` on ok.

        The error message names the matched paths, the wrong project_id, and
        tells the caller to resubmit to ``dark_factory`` instead.
        """
        if not self.is_rejection:
            return {}
        paths_str = ', '.join(self.matched_paths)
        return {
            'error': (
                f'{self.error_type}: task references dark-factory paths '
                f'({paths_str}) but was filed under project {self.project_id!r}. '
                f'Resubmit to the dark_factory project instead.'
            ),
            'error_type': self.error_type,
            'project_id': self.project_id,
            'matched_paths': list(self.matched_paths),
        }


# ---------------------------------------------------------------------------
# check_candidate_for_dark_factory_paths
# ---------------------------------------------------------------------------

def check_candidate_for_dark_factory_paths(
    candidate: CandidateTask,
    project_id: str,
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
    dark_factory_project_id: str = DARK_FACTORY_PROJECT_ID,
) -> PathGuardVerdict:
    """Check whether *candidate* references dark-factory paths in a wrong project.

    Args:
        candidate: The task candidate to check (title, description, details,
            files_to_modify are scanned).
        project_id: The project_id the task is being filed into.
        prefixes: Path prefixes to check against.  Defaults to
            :data:`DARK_FACTORY_PATH_PREFIXES`.
        dark_factory_project_id: The project_id considered "correct" for
            dark-factory paths.  Defaults to :data:`DARK_FACTORY_PROJECT_ID`.

    Returns:
        A :class:`PathGuardVerdict` with ``outcome='ok'`` when the task is
        correctly filed (or has no dark-factory paths), and
        ``outcome='rejection'`` when dark-factory paths are found in a
        non-dark-factory project.
    """
    # Correctly-filed tasks are always ok — no need to scan.
    if project_id == dark_factory_project_id:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    # Concatenate all text fields the guard cares about into one blob.
    parts: list[str] = [
        candidate.title or '',
        candidate.description or '',
        candidate.details or '',
    ]
    parts.extend(candidate.files_to_modify or [])
    combined = '\n'.join(parts)

    matched = find_dark_factory_paths(combined, prefixes)
    if matched:
        return PathGuardVerdict(
            outcome='rejection',
            project_id=project_id,
            matched_paths=tuple(matched),
        )
    return PathGuardVerdict(outcome='ok', project_id=project_id)
