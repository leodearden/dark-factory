"""Shared project-selection helper for fused-memory maintenance scripts.

Several ``scripts/*.py`` maintenance tools (e.g. ``prune_recon_cycle_summaries.py``,
``cleanup_count_snapshots.py``, ``tag_cgl_eta_rehome_scope.py``) accept an
optional ``--project-id`` filter over a ``{project_id: project_root}`` known-
projects map. :func:`select_projects` is the one canonical implementation of
that selection rule -- import it rather than re-defining a per-script copy.
"""

from __future__ import annotations


def select_projects(
    known_map: dict[str, str],
    project_id_filter: str | None,
) -> list[str]:
    """Return the sorted list of project_ids to process.

    Parameters
    ----------
    known_map:
        ``{project_id: project_root}`` from ``build_known_projects_map``.
    project_id_filter:
        When given, restrict to this single project_id. Raises ValueError
        with the list of known ids if the filter is not recognised.

    Returns
    -------
    Sorted list of project_ids.
    """
    if project_id_filter is None:
        return sorted(known_map.keys())
    if project_id_filter not in known_map:
        known_ids = sorted(known_map.keys())
        raise ValueError(
            f'Unknown project_id {project_id_filter!r}. '
            f'Known project ids: {known_ids}'
        )
    return [project_id_filter]
