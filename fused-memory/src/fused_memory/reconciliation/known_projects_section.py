"""The ``### Known Projects`` payload section Stages 2 and 3 render for cross-project routing.

Its listing is the project registry recon-report's ``cite_task`` validates
project_ids against, so a project is listed exactly when a citation into it
resolves.
"""

from __future__ import annotations

from collections.abc import Mapping

KNOWN_PROJECTS_HEADER = '### Known Projects (for cross-project routing)'


def render_known_projects_section(
    known_projects: Mapping[str, str], *, current_project_id: str
) -> str:
    """Render every registered project and its root, current project first, else ``''``.

    Fewer than two projects leaves no cross-project dimension to surface, so
    nothing is rendered.
    """
    if len(known_projects) < 2:
        return ''
    ordered = (
        [(current_project_id, known_projects[current_project_id])]
        if current_project_id in known_projects
        else []
    )
    ordered += sorted(
        (pid, root) for pid, root in known_projects.items() if pid != current_project_id
    )
    width = max(len(pid) for pid, _ in ordered)
    rows = [
        f'- {pid:<{width}}  → {root}{"  (current)" if pid == current_project_id else ""}'
        for pid, root in ordered
    ]
    return '\n' + KNOWN_PROJECTS_HEADER + '\n' + '\n'.join(rows) + '\n'
