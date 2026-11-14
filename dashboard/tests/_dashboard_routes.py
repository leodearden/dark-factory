"""The dashboard's GET routes under ``/api/v2/dashboard/``, read off the live app.

One definition of "the dashboard's GET routes" for every guard that must cover
them all: the hung-MCP probe sweeps the literal ones, and the payload budget
table must budget each literal route and budget or exempt each templated one.
Deriving the set from ``app.routes`` means a new endpoint is picked up by both
without anyone remembering to list it.
"""

from __future__ import annotations

from dataclasses import dataclass

DASHBOARD_API_PREFIX = '/api/v2/dashboard/'


@dataclass(frozen=True)
class DashboardGetPaths:
    """The GET paths, split by whether a request can name them verbatim.

    ``templated`` paths carry path parameters (``/task/{project:path}/T-{task_id}``),
    so requesting one needs real parameter values bound in; ``literal`` paths
    can be requested exactly as written. Both are sorted.
    """

    literal: tuple[str, ...]
    templated: tuple[str, ...]


def dashboard_get_paths() -> DashboardGetPaths:
    from dashboard.app import app

    literal: set[str] = set()
    templated: set[str] = set()
    # ``app.routes`` is typed ``list[BaseRoute]``, which declares no ``path``
    # — bind it via getattr so the duck-typed sweep still covers every route
    # class, Mount included.
    for route in app.routes:
        path = getattr(route, 'path', '')
        if not path.startswith(DASHBOARD_API_PREFIX):
            continue
        if 'GET' not in (getattr(route, 'methods', None) or set()):
            continue
        has_parameters = bool(getattr(route, 'param_convertors', None))
        (templated if has_parameters else literal).add(path)
    return DashboardGetPaths(literal=tuple(sorted(literal)), templated=tuple(sorted(templated)))
