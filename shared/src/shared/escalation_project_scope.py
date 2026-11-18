"""Which project a call on an escalation server is scoped to (task 6910).

A project store has exactly one project. The reconciliation store (port 8103)
serves every project, and ``ProjectRootGuard`` refuses every ``project_root``
on it, so a caller names its project per call with the
:data:`ESCALATION_PROJECT_HEADER` request header instead. The server honours
that header only for a project the store serves.

Both halves of the protocol live here, the header a client sends and the
:data:`~shared.uuid_prefix_guard.ProjectFor` a server answers with, so the two
cannot drift apart. Why the scope is never a union of projects and never a
guess: ``plans/uuid-prefix-resolution-prd.md`` D6.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

from fastmcp.server.dependencies import get_http_headers

from shared.uuid_prefix_guard import ProjectFor

__all__ = [
    'ESCALATION_PROJECT_HEADER',
    'escalation_project_for',
    'escalation_project_headers',
]

logger = logging.getLogger(__name__)

ESCALATION_PROJECT_HEADER = 'X-Escalation-Project'

#: get_http_headers() lowercases every key.
_HEADER_KEY = ESCALATION_PROJECT_HEADER.lower()


def escalation_project_headers(project_id: str) -> dict[str, str]:
    """The request headers that scope a caller's escalation calls to *project_id*."""
    if not isinstance(project_id, str) or not project_id:
        raise ValueError(
            f'an escalation project scope needs a non-empty project id, got {project_id!r}'
        )
    return {ESCALATION_PROJECT_HEADER: project_id}


def escalation_project_for(
    store_project: str | None, per_call_projects: Collection[str]
) -> ProjectFor:
    """The scope of each call on a store whose own project is *store_project*.

    A store with a project of its own always answers it. A store without one
    answers the project its caller's header names, when that project is in
    *per_call_projects*, and ``None`` otherwise.
    """
    if isinstance(per_call_projects, str):
        raise TypeError(
            'per_call_projects must be a collection of project ids, '
            f'not the single string {per_call_projects!r}'
        )
    served = frozenset(per_call_projects)
    if store_project is not None:
        if served:
            raise ValueError(
                f'a store scoped to {store_project!r} cannot also take per-call '
                f'projects {sorted(served)!r}; per-call projects only scope a '
                'store that has no project of its own'
            )
        return lambda _arguments: store_project

    def project_for(_arguments: Mapping[str, Any]) -> str | None:
        asserted = get_http_headers().get(_HEADER_KEY)
        if asserted is None:
            return None
        if asserted in served:
            return asserted
        logger.warning(
            'escalation project header names %r, which this store does not serve '
            '(%s); the call is left unscoped',
            asserted,
            ', '.join(sorted(served)),
        )
        return None

    return project_for
