"""Single-owner candidate_key computation (fm-task-dedup W8 task A1).

Deliberately dependency-free (stdlib ``hashlib`` only): the SQLite task
backend imports this leaf directly from every INSERT chokepoint, and it
must not drag in ``task_curator`` (which imports ``qdrant_client`` /
``graphiti_core``) or the backend itself — either would recreate the
import cycle this leaf exists to avoid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _normalize_title(title: str | None) -> str:
    """Lowercase + collapse whitespace for forgiving title comparison.

    Mirror site: this is a byte-for-byte copy of
    ``fused_memory.middleware.task_curator.normalize_title``, duplicated
    here (rather than imported) to keep this module stdlib-only. Any
    change to the algorithm there must be mirrored here.
    """
    return ' '.join((title or '').strip().lower().split())


def compute_candidate_key(
    title: str | None, files: Iterable[object] | None,
) -> str | None:
    """Stable 16-char hash over normalised (title, files) — the single
    definition of task "sameness" used for dedup at the store level.

    Case + whitespace insensitive on the title, file-order insensitive.
    Returns ``None`` when the normalised title is empty (``None``, ``''``,
    or whitespace-only) — such rows are uncomputable and excluded from
    the future partial UNIQUE index (PRD fm-task-dedup §C-A).

    ``files`` entries that are not ``str`` (e.g. a caller passed a raw
    metadata list containing dicts or ints) are silently dropped rather
    than raising: every caller documents that key computation must never
    be the reason an insert/update fails, and ``sorted``/``join`` would
    otherwise raise ``TypeError`` on a mixed-type list.
    """
    normalized_title = _normalize_title(title)
    if not normalized_title:
        return None
    normalized_files = '\n'.join(sorted(f for f in (files or []) if isinstance(f, str)))
    h = hashlib.sha256()
    h.update(normalized_title.encode())
    h.update(b'|')
    h.update(normalized_files.encode())
    # 16-char sha256-hex shape — canonical owner: orchestrator.agents.triage.sha256_16.
    # Any change to length or algorithm must be mirrored there and at the
    # task_curator.py mirror sites (payload_hash, _intra_batch_key, _normalize_key).
    return h.hexdigest()[:16]
