"""Backend data layer for the Escalations dashboard section.

Provides three public functions:

- ``load_queue_escalations`` — root-only *.json reader for a single escalation
  queue directory (no archive traversal).
- ``resolve_owning_project`` — maps a reconciliation escalation back to its
  owning project via worktree-prefix matching or task-map probe.
- ``build_escalation_queues`` — enumerates per-project escalation dirs plus the
  fused-memory reconciliation queue, returning a structured ``{subsections, summary}``
  dict for the API layer to serve.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_queue_escalations(esc_dir: Path) -> list[dict]:
    """Load all escalation JSON files from *esc_dir* (root only, no archive).

    Only ``*.json`` files directly inside *esc_dir* are read — subdirectories
    (including ``archive/YYYY-MM-DD/``) are **not** traversed.  This is an
    intentional divergence from :func:`escalation.queue.iter_all_escalation_paths`,
    which also walks the archive subtree.

    A missing or non-directory *esc_dir* returns ``[]`` without raising.
    A file that cannot be parsed as JSON is skipped with a ``WARNING`` log.

    Args:
        esc_dir: Path to the escalation queue root directory.

    Returns:
        List of escalation dicts (fields passed through unchanged).
    """
    if not esc_dir.is_dir():
        return []

    results: list[dict] = []
    for path in esc_dir.glob('*.json'):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Failed to load escalation %s: %s', path, exc)
            continue
    return results


def resolve_owning_project(
    esc: dict,
    roots: list[tuple[Path, list[dict]]],
) -> str | None:
    """Resolve the owning project for a reconciliation escalation.

    Two-pass probe (first match wins in each pass):

    1. **Worktree prefix** — if ``esc['worktree']`` starts with ``str(root)``
       or ``str(root / '.worktrees')``, return ``root.name``.
    2. **Task-map probe** — if ``str(esc['task_id'])`` matches ``str(t['id'])``
       for any task *t* in a root's task map, return ``root.name``.

    Args:
        esc: Escalation dict (fields ``worktree`` and ``task_id`` are probed).
        roots: Ordered list of ``(root_path, task_map)`` tuples.  The first
            matching root wins, so callers should put the primary root first.

    Returns:
        The matching root's basename (``root.name``), or ``None`` if neither
        probe finds a match.
    """
    worktree = esc.get('worktree') or ''

    # Pass 1: worktree prefix matching
    if worktree:
        for root, _task_map in roots:
            if worktree.startswith(str(root)) or worktree.startswith(str(root / '.worktrees')):
                return root.name

    # Pass 2: task-map probe
    task_id = str(esc.get('task_id', ''))
    if task_id:
        for root, task_map in roots:
            if any(str(t.get('id')) == task_id for t in task_map):
                return root.name

    return None


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_LEVEL_KEYS = (0, 1, 2)
_STATUS_KEYS = ('pending', 'resolved', 'dismissed')


def _bucket(escalations: list[dict]) -> dict:
    """Compute per-subsection summary counts from a list of escalation dicts.

    Only ``level`` values in ``{0, 1, 2}`` and ``status`` values in
    ``{pending, resolved, dismissed}`` are counted.  Unknown values are
    silently ignored (the escalation is still present in the subsection's
    ``escalations`` list).

    Returns:
        ``{"by_level": {0: int, 1: int, 2: int},
           "by_status": {"pending": int, "resolved": int, "dismissed": int}}``
    """
    by_level: dict = {k: 0 for k in _LEVEL_KEYS}
    by_status: dict = {k: 0 for k in _STATUS_KEYS}
    for esc in escalations:
        lvl = esc.get('level')
        if lvl in by_level:
            by_level[lvl] += 1
        st = esc.get('status')
        if st in by_status:
            by_status[st] += 1
    return {'by_level': by_level, 'by_status': by_status}


def _merge_summaries(summaries: list[dict]) -> dict:
    """Merge a list of per-subsection summary dicts into one aggregate."""
    by_level: dict = {k: 0 for k in _LEVEL_KEYS}
    by_status: dict = {k: 0 for k in _STATUS_KEYS}
    for s in summaries:
        for k in _LEVEL_KEYS:
            by_level[k] += s['by_level'].get(k, 0)
        for k in _STATUS_KEYS:
            by_status[k] += s['by_status'].get(k, 0)
    return {'by_level': by_level, 'by_status': by_status}


def build_escalation_queues(config) -> dict:
    """Enumerate escalation queues across all project roots and reconciliation.

    Produces one ``orchestrator`` subsection per project root (primary first,
    then :attr:`~dashboard.config.DashboardConfig.known_project_roots`, de-duped)
    plus a single ``reconciliation`` subsection from
    :attr:`~dashboard.config.DashboardConfig.reconciliation_escalations_dir`.

    Each subsection has the shape::

        {
            "id":         str,        # str(root) for orchestrators; "reconciliation"
            "label":      str,        # root.name for orchestrators; "fused-memory"
            "kind":       str,        # "orchestrator" | "reconciliation"
            "escalations": list[dict],
            "summary":    {"by_level": {0,1,2}, "by_status": {pending,resolved,dismissed}},
        }

    The top-level return value also carries an aggregated ``summary`` block.

    Args:
        config: :class:`~dashboard.config.DashboardConfig` instance.

    Returns:
        ``{"subsections": [...], "summary": {...}}``
    """
    subsections: list[dict] = []

    # De-duped root iteration: primary root first, then known_project_roots.
    seen: set[Path] = {config.project_root}
    roots_to_visit: list[Path] = [config.project_root]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            roots_to_visit.append(root)

    for root in roots_to_visit:
        escs = load_queue_escalations(root / 'data' / 'escalations')
        subsections.append({
            'id': str(root),
            'label': root.name,
            'kind': 'orchestrator',
            'escalations': escs,
            'summary': _bucket(escs),
        })

    # Reconciliation subsection (fused-memory queue)
    recon_escs = load_queue_escalations(config.reconciliation_escalations_dir)
    subsections.append({
        'id': 'reconciliation',
        'label': 'fused-memory',
        'kind': 'reconciliation',
        'escalations': recon_escs,
        'summary': _bucket(recon_escs),
    })

    top_summary = _merge_summaries([s['summary'] for s in subsections])
    return {'subsections': subsections, 'summary': top_summary}
