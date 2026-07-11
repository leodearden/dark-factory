"""cockpit.registry_reader — pure-consumer read path over the C1 session registry.

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Imports the frozen
orchestrator.session_registry contract (PRD §6 G5: consumers import, never
re-derive the record shape). This module is read-only: it never calls
write_record/write_decision/update_decision_state/set_manual_boost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator import session_registry

logger = logging.getLogger(__name__)

# Substantive SessionRecord fields a snapshot is keyed on. Deliberately
# excludes start_ts/age-derived values so a purely-time-passing poll tick
# (nothing on disk actually changed) diffs as a no-op.
_SNAPSHOT_FIELDS = (
    'status',
    'title',
    'role',
    'project',
    'task_id',
    'escalation_id',
    'parent_session_id',
)


def scan_sessions(root: Path | str | None = None) -> list[session_registry.SessionRecord]:
    """Return every readable SessionRecord under ``sessions_dir(root)``.

    Mirrors reap_stale_records' identity-from-path iterdir loop: an absent
    sessions/ dir returns [] (not an error), and a single corrupt or
    vanished record.json is logged and skipped rather than aborting the
    scan of the remaining slugs (fail-soft, PRD §2).
    """
    base = session_registry.sessions_dir(root)
    if not base.is_dir():
        return []

    records: list[session_registry.SessionRecord] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        try:
            record = session_registry.read_record(slug, root=root)
        except (FileNotFoundError, session_registry.CorruptSessionRecord):
            logger.warning('scan_sessions: skipping unreadable record for %s', slug, exc_info=True)
            continue
        records.append(record)
    return records


def build_snapshot(
    records: list[session_registry.SessionRecord],
) -> dict[str, tuple]:
    """Map session_slug -> a hashable tuple of substantive display fields.

    Deliberately excludes start_ts/age so a quiet (nothing-changed) tick
    produces an identical snapshot -- this is what lets the app's poll
    rebuild the table only when something real changed (PRD §5).
    """
    snapshot: dict[str, tuple] = {}
    for record in records:
        values = tuple(getattr(record, field) for field in _SNAPSHOT_FIELDS)
        question_text = record.question.text if record.question is not None else None
        snapshot[record.session_slug] = (*values, question_text)
    return snapshot


def snapshot_changed(old: dict[str, tuple], new: dict[str, tuple]) -> bool:
    """True iff a record was added, removed, or had a substantive field change."""
    return old != new
