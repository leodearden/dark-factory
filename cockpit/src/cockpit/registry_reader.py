"""cockpit.registry_reader — pure-consumer read path over the C1 session registry.

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Imports the frozen
orchestrator.session_registry contract (PRD §6 G5: consumers import, never
re-derive the record shape). This module is read-only: it never calls
write_record/write_decision/update_decision_state/set_manual_boost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

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


def _read_record_soft(
    slug: str, root: Path | str | None, *, caller: str
) -> session_registry.SessionRecord | None:
    """Read *slug*'s record.json, fail-soft: return None instead of raising.

    Shared by scan_sessions and SessionScanner.scan (below) so the fail-soft
    policy -- which exceptions are swallowed, what gets logged -- lives in
    exactly one place: both loops build their per-slug read step on this one
    helper, so they stay behaviorally identical by construction rather than
    by convention (PRD §2; see the parity test
    test_scan_matches_scan_sessions_for_seeded_dir). *caller* is folded into
    the warning message only, preserving each call site's original
    log-message prefix for anyone grepping logs.
    """
    try:
        return session_registry.read_record(slug, root=root)
    except (FileNotFoundError, session_registry.CorruptSessionRecord):
        logger.warning('%s: skipping unreadable record for %s', caller, slug, exc_info=True)
        return None


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
        record = _read_record_soft(slug_dir.name, root, caller='scan_sessions')
        if record is not None:
            records.append(record)
    return records


class SessionScanner:
    """Stateful, cache-aware counterpart to scan_sessions (above).

    A poll loop calling scan_sessions() every tick pays a full read_text() +
    JSON parse for every session slug, every tick, even when nothing on disk
    changed -- at 10k+ sessions this is the multi-second cost this class
    exists to eliminate (see registry_reader/app.py module docstrings). A
    SessionScanner owns a per-slug cache keyed on record.json's
    st_mtime_ns, so a later scan() call reuses a cached SessionRecord for
    any slug whose record.json mtime is unchanged since the last scan --
    iterdir() is cheap and always runs (still needed to detect added/removed
    slugs), but the parse-heavy read_record() call is skipped for an
    unchanged slug. A stateful, constructed object (rather than a free
    function) gives the cache an obvious owner/lifecycle: each CockpitApp
    instance (or test) gets its own SessionScanner with its own independent
    cache; scan_sessions itself is left untouched as the stateless reader.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = root
        self._cache: dict[str, tuple[int, session_registry.SessionRecord]] = {}

    def scan(self) -> list[session_registry.SessionRecord]:
        """Return every readable SessionRecord under sessions_dir(root).

        Mirrors scan_sessions' fail-soft iterdir + read_record loop (see its
        docstring): an absent sessions/ dir returns [], and a corrupt or
        vanished record.json is logged and skipped rather than aborting the
        scan of the remaining slugs. Additionally: a slug whose record.json
        st_mtime_ns is unchanged since the last scan() call reuses the
        cached SessionRecord instead of paying read_record's read_text() +
        JSON parse again.

        The cache is rebuilt from scratch every call from exactly the slugs
        seen THIS scan, and only assigned to self._cache at the end -- a
        slug that vanished since the last scan is dropped from the cache
        even if it never resurfaces, so a later, unrelated record written at
        the same slug is always read fresh rather than risking a match
        against a stale leftover cache entry (e.g. a coincidentally-repeated
        mtime). A stat() failure (record.json vanished between iterdir() and
        stat(), or any other OSError) is treated as a cache miss -- it falls
        back to read_record(), which raises FileNotFoundError/
        CorruptSessionRecord in turn and is fail-soft skipped exactly like
        scan_sessions, rather than propagating out of scan() uncaught.
        """
        base = session_registry.sessions_dir(self._root)
        if not base.is_dir():
            self._cache = {}
            return []

        records: list[session_registry.SessionRecord] = []
        new_cache: dict[str, tuple[int, session_registry.SessionRecord]] = {}
        for slug_dir in sorted(base.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            record_path = session_registry.record_path_for_slug(slug, root=self._root)
            try:
                mtime_ns: int | None = record_path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None

            cached = self._cache.get(slug)
            if mtime_ns is not None and cached is not None and cached[0] == mtime_ns:
                new_cache[slug] = cached
                records.append(cached[1])
                continue

            record = _read_record_soft(slug, self._root, caller='SessionScanner.scan')
            if record is None:
                continue
            if mtime_ns is not None:
                new_cache[slug] = (mtime_ns, record)
            records.append(record)
        self._cache = new_cache
        return records


@runtime_checkable
class SessionScannerProtocol(Protocol):
    """Structural shape of SessionScanner's scan() -- CockpitApp's `scanner` DI seam.

    Mirrors cockpit.backends.base.FocusArrangeBackend's Protocol-for-DI
    pattern: typing the seam structurally (rather than as the concrete
    SessionScanner class) lets a test's fake scanner satisfy it by
    implementing scan() directly, with no need to subclass the real
    SessionScanner -- which would drag in its mtime cache and disk-scanning
    internals for no reason.
    """

    def scan(self) -> list[session_registry.SessionRecord]:
        """Return the current readable SessionRecord set."""
        ...


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
