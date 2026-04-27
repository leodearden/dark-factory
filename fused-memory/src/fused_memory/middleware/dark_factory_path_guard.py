"""Dark-factory-specific path-scope guard — back-compat shim.

The original dark-factory-only guard introduced in task 1088 has been
generalised into the multi-project
:mod:`fused_memory.middleware.path_scope_guard`.  This module remains as a
thin re-export so callers and tests that still import the old names keep
working for one merge cycle; new code should import from the new module
directly.

The shim builds a one-project registry containing dark_factory and its
hard-coded prefix list, then delegates to the new check functions.
Behaviour is identical to the original guard for the dark_factory →
non-dark_factory direction (which is all the original guard did).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fused_memory.middleware.path_scope_guard import (
    PathGuardVerdict,
    check_candidate_for_scope,
    check_text_for_scope,
    find_paths,
)
from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask

# ---------------------------------------------------------------------------
# Constants — preserved for back-compat
# ---------------------------------------------------------------------------

DARK_FACTORY_PROJECT_ID: str = 'dark_factory'

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


def _dark_factory_only_registry(
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
    dark_factory_project_id: str = DARK_FACTORY_PROJECT_ID,
) -> ProjectPrefixRegistry:
    """Build a single-project registry equivalent to the original guard.

    Used internally by the back-compat check functions so they can call the
    new generalised guard without touching the filesystem.
    """
    return ProjectPrefixRegistry(
        project_to_root={dark_factory_project_id: '/home/leo/src/dark-factory'},
        project_to_prefixes={dark_factory_project_id: prefixes},
        prefix_to_project={p: dark_factory_project_id for p in prefixes},
    )


# ---------------------------------------------------------------------------
# Back-compat re-exports
# ---------------------------------------------------------------------------


def find_dark_factory_paths(
    text: str,
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
) -> list[str]:
    """Scan *text* for dark-factory path prefixes (back-compat re-export).

    Forwards to :func:`fused_memory.middleware.path_scope_guard.find_paths`
    with the dark-factory prefix list pinned as the default so existing
    tests that omit *prefixes* keep matching the same patterns.
    """
    return find_paths(text, prefixes)


def check_candidate_for_dark_factory_paths(
    candidate: CandidateTask,
    project_id: str,
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
    dark_factory_project_id: str = DARK_FACTORY_PROJECT_ID,
) -> PathGuardVerdict:
    """Back-compat: dark-factory-only candidate check.

    Builds a single-project registry from *prefixes* and delegates to
    :func:`check_candidate_for_scope`.  Preserves the original
    ``project_id == dark_factory_project_id → ok`` short-circuit because
    a dark-factory candidate cannot reference its own prefixes as
    "another project's" paths (the registry attributes them to dark_factory).
    """
    registry = _dark_factory_only_registry(prefixes, dark_factory_project_id)
    return check_candidate_for_scope(candidate, project_id, registry)


def check_text_for_dark_factory_paths(
    text: str | None,
    project_id: str,
    prefixes: tuple[str, ...] = DARK_FACTORY_PATH_PREFIXES,
    dark_factory_project_id: str = DARK_FACTORY_PROJECT_ID,
) -> PathGuardVerdict:
    """Back-compat: dark-factory-only text check.

    Builds a single-project registry from *prefixes* and delegates to
    :func:`check_text_for_scope`.
    """
    registry = _dark_factory_only_registry(prefixes, dark_factory_project_id)
    return check_text_for_scope(text, project_id, registry)


__all__ = [
    'DARK_FACTORY_PATH_PREFIXES',
    'DARK_FACTORY_PROJECT_ID',
    'PathGuardVerdict',
    'check_candidate_for_dark_factory_paths',
    'check_text_for_dark_factory_paths',
    'find_dark_factory_paths',
]
