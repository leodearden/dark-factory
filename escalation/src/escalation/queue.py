"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from shared.timestamps import parse_timestamp_or_warn

from escalation import archive
from escalation.models import Escalation

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def escalation_id_lock(queue_dir: Path, escalation_id: str) -> Iterator[None]:
    """Per-escalation-id exclusive advisory lock using a stable sidecar file.

    WHY A SIDECAR (PRD D3 rationale):
    The queue's writers are atomic tmp+rename (_atomic_write_path).  After a
    tmp+rename replace, the data file ``{escalation_id}.json`` is a NEW inode.
    A second writer that flock()s the (new) data-file path binds to a different
    inode and races anyway — the lock is defeated.  The fix is a STABLE lock
    target: ``{escalation_id}.json.lock``, created once via os.open(O_CREAT)
    and NEVER renamed or replaced, so all writers flock the same inode and
    actually serialize.

    EXPORT CONTRACT:
    This helper is module-level and exported so task ε (server-start sweep /
    reaper) can import it as::

        from escalation.queue import escalation_id_lock

    and take the same lock around its root↔archive relocations without
    instantiating an EscalationQueue.

    GLOB INVISIBILITY:
    The sidecar ends in ``.lock`` and does NOT match the ``esc-*.json`` glob
    used by get_pending / get_by_task / make_id / iter_all_escalation_paths.
    Lock files are intentionally never deleted or renamed (stable inode);
    accumulation at current queue volumes is acceptable.

    Usage::

        with escalation_id_lock(queue_dir, escalation_id):
            data = read(); data = mutate(data); atomic_write(data)
    """
    lock_path = Path(queue_dir) / f'{escalation_id}.json.lock'
    # Defensively create queue_dir so standalone callers (e.g. task ε sweep/reaper)
    # can take the lock without having first instantiated an EscalationQueue.
    Path(queue_dir).mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

# Severity rank map for promotion logic.  Alphabetical comparison is wrong
# ('blocking' < 'info'), so we use an explicit rank.  Unknown severities
# default to rank 0 (treated as info-level) so malformed input never causes
# unexpected promotion.
_SEVERITY_RANK: dict[str, int] = {'info': 0, 'blocking': 1}


def _max_severity(a: str, b: str) -> str:
    """Return the higher-urgency severity string between *a* and *b*."""
    for val in (a, b):
        if val not in _SEVERITY_RANK:
            logger.warning(
                '_max_severity: unrecognised severity %r — treating as info-level '
                '(rank 0). Known values: %s',
                val,
                ', '.join(_SEVERITY_RANK),
            )
    return a if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0) else b


def iter_all_escalation_paths(escalations_dir: Path) -> Iterator[Path]:
    """Yield all ``esc-*.json`` paths from *escalations_dir* and its archive subtree.

    Two-tier scan with root precedence (mirrors the convention in
    ``EscalationQueue.get_by_task``):

    1. Queue root — ``escalations_dir.glob('esc-*.json')`` is iterated first.
       Each yielded path's stem is added to ``seen``.
    2. Archive subtree — ``escalations_dir/archive/`` (``archive.ARCHIVE_SUBDIR``)
       is scanned with ``rglob('esc-*.json')``.  Paths whose stem is already in
       ``seen`` (i.e. a root copy was found) are skipped so the root copy wins
       on id collisions.  Stems are added to ``seen`` as they are yielded so
       archive-only multi-date duplicates (same id under two dated subdirs) are
       also deduplicated.

    If *escalations_dir* does not exist or is not a directory, the iterator
    yields nothing without raising an exception.  This matches the
    dashboard's read-only usage: a missing or misconfigured path produces an
    empty result rather than a hard failure.

    Primary caller: ``dashboard.data.performance._load_escalations`` — bulk
    read of all escalation files for performance aggregation.
    """
    escalations_dir = Path(escalations_dir)
    if not escalations_dir.is_dir():
        return

    seen: set[str] = set()

    # Tier 1: queue root
    for path in escalations_dir.glob('esc-*.json'):
        seen.add(path.stem)
        yield path

    # Tier 2: archive subtree (root wins on id collisions)
    archive_root = escalations_dir / archive.ARCHIVE_SUBDIR
    if archive_root.exists():
        for path in archive_root.rglob('esc-*.json'):
            if path.stem not in seen:
                seen.add(path.stem)
                yield path


class EscalationQueue:
    """Filesystem queue for escalations.

    Each escalation is a JSON file named {id}.json in the queue directory.
    Writes are atomic (tmp file + rename). Reads tolerate partial writes.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._notify_callback: Callable[[Escalation], None] | None = None
        self._resolve_callback: Callable[[Escalation], None] | None = None
        # Cache 1 — archive listing (serves get()/_locate_path).
        # None = not yet built; dict = {dated-subdir-name: set-of-esc-id-stems}.
        # Lazy-built on first archive lookup; incrementally updated in _archive_resolved.
        self._archive_listing: dict[str, set[str]] | None = None
        # Cache 2 — max_seq per task_id (serves make_id()).
        # {task_id: max archived seq} — absent key means "not yet scanned".
        # Upward-only updates in _archive_resolved guarantee no under-estimate.
        self._archive_max_seq_cache: dict[str, int] = {}

    def set_notify_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._notify_callback = callback

    def set_resolve_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._resolve_callback = callback

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _scan_archive_listing(self) -> dict[str, set[str]]:
        """Build the archive listing from a single rglob scan.

        Returns a dict mapping dated-subdir-name (e.g. '2025-06-15') to a set
        of esc-id stems (e.g. {'esc-1-1', 'esc-1-2'}) for all files under the
        archive root.  Returns an empty dict when the archive root does not exist.
        """
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        listing: dict[str, set[str]] = {}
        if archive_root.exists():
            for path in archive_root.rglob('esc-*.json'):
                listing.setdefault(path.parent.name, set()).add(path.stem)
        return listing

    def _get_archive_listing(self) -> dict[str, set[str]]:
        """Return the memoised archive listing, building it on first access."""
        if self._archive_listing is None:
            self._archive_listing = self._scan_archive_listing()
        return self._archive_listing

    def _scan_archive_max_seq(self, prefix: str) -> int:
        """Scan the archive for the maximum sequence number for *prefix*.

        Iterates _iter_archive_paths(f'{prefix}*.json'), parses the trailing
        integer from each matching stem, and returns the max found (0 if none).
        Used by make_id() on a per-task_id cache miss to seed the cache.
        """
        max_seq = 0
        for path in self._iter_archive_paths(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        return max_seq

    def _iter_archive_paths(self, pattern: str) -> Iterator[Path]:
        """Yield escalation JSON files from the archive subtree matching *pattern*.

        Returns an empty iterator when the archive root does not exist, avoiding
        a spurious ``rglob`` call on a missing directory.  Centralising the
        existence-check + rglob here means get(), get_by_task(), and make_id()
        all benefit from any future caching or indexing added in one place.
        """
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        if archive_root.exists():
            yield from archive_root.rglob(pattern)

    def submit(self, escalation: Escalation) -> str:
        """Atomic write: {id}.tmp -> rename to {id}.json.

        The write is serialized per-id by ``escalation_id_lock`` to prevent a
        concurrent same-id submit from clobbering an in-progress RMW mutator
        (add_members_to_l2, attach_dedupe_child, resolve).  The notify callback
        fires OUTSIDE the lock so a callback that re-enters the queue for the
        same id does not deadlock.
        """
        with escalation_id_lock(self.queue_dir, escalation.id):
            self._atomic_write(escalation.id, escalation.to_json())
        logger.info(f'Escalation submitted: {escalation.id} [{escalation.severity}]')

        if self._notify_callback:
            try:
                self._notify_callback(escalation)
            except Exception as e:
                logger.warning(f'Notify callback failed for {escalation.id}: {e}')

        return escalation.id

    def _locate_path(self, escalation_id: str) -> Path | None:
        """Return the on-disk path for *escalation_id*, or None if not found.

        Lookup order:
        1. ``queue_dir/{escalation_id}.json`` (queue root).
        2. Archive subtree (``archive/YYYY-MM-DD/{escalation_id}.json``):
           - If exactly one candidate: return it.
           - If multiple candidates (multi-date duplicates): emit a WARNING naming the id
             and return the newest by ``parent.name`` lexicographic order
             (YYYY-MM-DD sorts lexicographically == chronologically; non-date names fall back
             to ``''`` and are treated as oldest).

        Shared by ``get()`` and ``patch_resolution_metadata()`` so that the archive
        fallback and newest-by-date selection logic live in exactly one place.
        """
        path = self.queue_dir / f'{escalation_id}.json'
        if path.exists():
            return path
        # Fall back to archive using the memoised per-subdir listing.
        # Reconstruct candidate paths from the listing without re-rglob-ing.
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        listing = self._get_archive_listing()
        candidates = [
            archive_root / subdir / f'{escalation_id}.json'
            for subdir, stems in listing.items()
            if escalation_id in stems
            and (archive_root / subdir / f'{escalation_id}.json').exists()
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                f'Multiple archive files for {escalation_id}: '
                f'{[str(p) for p in candidates]}; selecting newest by parent dir date'
            )
            # YYYY-MM-DD sorts lexicographically == chronologically.
            # Non-YYYY-MM-DD parent names fall back to '' (treated as oldest),
            # matching the comment and ensuring valid date dirs always win.
            return max(
                candidates,
                key=lambda p: (
                    p.parent.name
                    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.parent.name)
                    else ''
                ),
            )
        return candidates[0]

    def get(self, escalation_id: str) -> Escalation | None:
        """Read a single escalation by ID.

        Falls back to the archive directory when the file is not in the
        queue root (i.e. the escalation has been resolved and archived).

        The archive fallback uses a memoised per-dated-subdir listing
        (_archive_listing) so that repeated get() calls for archived ids
        scan the archive at most once per instance lifetime.  The listing
        is lazily built on the first archive miss and incrementally updated
        by _archive_resolved() when this instance archives a new escalation.

        Per-instance freshness: escalations resolved by another
        EscalationQueue instance (e.g. a concurrent process) after this
        instance's listing was built are not visible until this instance
        next calls _archive_resolved().  get() returns None in that window,
        degrading gracefully rather than raising.  This is consistent with
        the single-writer production model (one harness process per queue).
        """
        path = self._locate_path(escalation_id)
        if path is None:
            return None
        try:
            return Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
    ) -> list[Escalation]:
        """Scan dir for escalations matching a task ID.

        Two-tier scan:
        - status == 'pending': scan queue root only (fast path, skips archive).
        - status is None or another value: scan queue root PLUS archive/**/ to
          include resolved/dismissed escalations that have been moved out.

        Deduplication: if the same escalation id appears in both the queue root
        and the archive (e.g. crash mid-resolve), only the first occurrence that
        passes all filters is returned and a DEBUG message is logged for any
        later same-id path that also passes.  Filters (task_id / status / level)
        are applied BEFORE adding to ``seen``, so a copy that fails its own
        filter never shadows a matching copy in the other tier.  Iteration order
        is queue root first, archive second, so the queue_dir copy wins when
        both copies pass the filter.

        Pre-scan warning: before applying any filter, if the same escalation id
        appears in BOTH the queue_dir root AND the archive subtree, a single
        WARNING is logged regardless of which (if any) copies pass the caller's
        filter.  Note: the pre-scan covers only the paths that are actually
        scanned — when ``status == 'pending'`` the archive is skipped entirely,
        so a cross-tier duplicate is invisible to the pre-scan in that mode.
        """
        # Build the candidate path list.
        paths: list[Path] = list(self.queue_dir.glob('esc-*.json'))
        if status != 'pending':
            paths.extend(self._iter_archive_paths('esc-*.json'))

        # Pre-scan: detect ids that span both queue_dir root and archive subtree.
        # Uses filename stems only (no JSON parsing) to avoid extra I/O.
        #
        # Tier-membership: classify each path by comparing its parent directory
        # to self.queue_dir.  Files written by queue_dir.glob('esc-*.json') are
        # non-recursive, so their parent IS self.queue_dir (root tier).  Files
        # from _iter_archive_paths come from archive_root.rglob(...) and land
        # inside a dated subdir (e.g. archive/2025-06-15/); their parent is that
        # date dir, NOT self.queue_dir (archive tier).
        #
        # A single id can legitimately appear multiple times within the archive
        # tier (the same stem in two different dated subdirs after a requeue or
        # backup-restore).  That situation is handled by get()'s newest-by-date
        # selection (queue.py:102-117) and is NOT a cross-tier inconsistency.
        # We warn only when an id has members in BOTH tiers — that signals a
        # crash-mid-resolve or backup-restore scenario requiring reconciliation.
        id_to_paths: dict[str, list[Path]] = {}
        for path in paths:
            id_to_paths.setdefault(path.stem, []).append(path)
        for esc_id, esc_paths in id_to_paths.items():
            has_root = any(p.parent == self.queue_dir for p in esc_paths)
            has_archive = any(p.parent != self.queue_dir for p in esc_paths)
            if has_root and has_archive:
                logger.warning(
                    f'Escalation {esc_id!r} exists in both queue_dir and archive: '
                    f'{[str(p) for p in esc_paths]}; reconciliation may be needed'
                )

        seen: set[str] = set()
        results = []
        for path in paths:
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.task_id != task_id:
                    continue
                if status is not None and esc.status != status:
                    continue
                if level is not None and esc.level != level:
                    continue
                if esc.id in seen:
                    logger.debug(
                        f'Duplicate escalation id {esc.id!r} at {path}; '
                        'skipping (pre-scan WARNING already logged; queue_dir copy takes precedence)'
                    )
                    continue
                seen.add(esc.id)
                results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse {path}: {e}')
                continue
        return results

    def has_open_l1(self, task_id: str) -> bool:
        """Return True when the task has at least one pending level-1 escalation.

        Level-1 is the handed-to-human tier: the presence of one signals that
        the workflow must not auto-requeue the task — a human will unblock.
        """
        return bool(self.get_by_task(task_id, status='pending', level=1))

    def get_pending(self) -> list[Escalation]:
        """Get all pending escalations."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.status == 'pending':
                    results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse {path}: {e}')
                continue
        return results

    def resolve(
        self, escalation_id: str, resolution: str, dismiss: bool = False,
        *, resolved_by: str | None = None, resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Update an escalation's status to resolved or dismissed.

        Concurrency: the get → status-check → mutate → _atomic_write →
        _archive_resolved critical section is serialized under
        ``escalation_id_lock``.  Moving the idempotency status-check INSIDE
        the lock ensures that two concurrent resolves for the same id
        serialize and produce exactly one archive copy.

        Callbacks and cascade run AFTER releasing the lock:
        - ``_resolve_callback`` fires after the lock is released, preventing
          re-entrant callback → resolve deadlocks.
        - The member cascade calls ``self.resolve(member_id, …)`` for each L1
          member; each cascaded call takes its OWN distinct sidecar lock
          (different file → different inode → no same-fd re-entrancy/deadlock).

        Cascade: if the escalation is an L2 with a non-empty ``members`` list,
        after archiving the L2 each member id is resolved recursively with the
        same *resolution* text.  The cascade uses
        ``resolved_by='l2-cascade:{escalation_id}'`` so the audit trail
        distinguishes direct resolves from cascades.  *dismiss* is propagated
        so a dismissed L2 dismisses its members too.

        Cascade contract:
        - Recursion terminates because L1 members carry empty ``members`` lists.
        - Per-member exceptions are caught and logged; a single bad member id
          never blocks the remaining cascade.
        - ``resolve()`` is idempotent, so cascading to an already-resolved member
          is a safe no-op.

        Ordering note (for ``_resolve_callback`` consumers):
        ``_resolve_callback`` fires for the L2 **before** the cascade to member
        L1s begins.  Consumers of the callback that inspect member state will see
        members still pending at callback time; they should re-query member state
        rather than assuming terminality.  This ordering is stable — do not rely
        on members being resolved at the moment the L2 callback fires.
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            esc = self.get(escalation_id)
            if esc is None:
                return None

            if esc.status != 'pending':
                logger.info(
                    f'Escalation {escalation_id} already {esc.status}; resolve() is a no-op'
                )
                return esc

            esc.status = 'dismissed' if dismiss else 'resolved'
            esc.resolution = resolution
            esc.resolved_at = datetime.now(UTC).isoformat()
            if resolved_by is not None:
                esc.resolved_by = resolved_by
            if resolution_turns is not None:
                esc.resolution_turns = resolution_turns

            self._atomic_write(escalation_id, esc.to_json())
            self._archive_resolved(escalation_id, esc.resolved_at)

        logger.info(f'Escalation {escalation_id} {esc.status}: {resolution[:100]}')

        if self._resolve_callback:
            try:
                self._resolve_callback(esc)
            except Exception as e:
                logger.warning(f'Resolve callback failed for {escalation_id}: {e}')

        # Cascade to member L1s (L2 clusters only — L0/L1 have empty members).
        # Recursion is bounded: L1 members carry empty members[], so no deeper nesting.
        if esc.members:
            cascade_resolved_by = f'l2-cascade:{escalation_id}'
            for member_id in esc.members:
                try:
                    self.resolve(
                        member_id,
                        resolution,
                        dismiss=dismiss,
                        resolved_by=cascade_resolved_by,
                    )
                except Exception as e:
                    logger.warning(
                        'cascade: failed to resolve member %s of L2 %s: %s',
                        member_id, escalation_id, e,
                    )

        return esc

    def park(
        self,
        escalation_id: str,
        resolution: str,
        *,
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Park an escalation: promote to level=2, stamp resolution_action='park',
        persist resolution text, keep status='pending' (NOT archived).

        Differs from resolve() in three ways:
        - status stays 'pending' (escalation remains OPEN — sweep-quiescence mechanism)
        - level is promoted to 2 (L2 = human-only consumer)
        - _archive_resolved is NOT called (record stays live in queue_dir)
        - resolved_at is NOT set (preserving the resolved_at=None ↔ status='pending' invariant)

        After releasing the per-id lock, fires _resolve_callback(esc) to trigger the harness
        teardown (kill workflow + set task 'blocked') — same callback, different escalation state.

        For cluster L2s: fires _resolve_callback per member L1 with an in-memory
        resolved_by='l2-cascade:<L2-id>' stamp (does NOT persist/archive member records —
        members stay pending L1s covering their tasks).

        Idempotent: returns existing record unchanged if either (a) status != 'pending'
        (already resolved/dismissed) or (b) resolution_action == 'park' AND resolution
        is not None (already parked — resolution text was written by a prior park() call).
        The combined second guard prevents a re-park from re-firing teardown callbacks
        without falsely short-circuiting a first-time park() on an L2 whose
        resolution_action was pre-set to 'park' but resolution is still None.
        The first guard matches the contract of resolve().
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            esc = self.get(escalation_id)
            if esc is None:
                return None

            if esc.status != 'pending' or (
                esc.resolution_action == 'park' and esc.resolution is not None
            ):
                logger.info(
                    'Escalation %s already %s (resolution_action=%r); park() is a no-op',
                    escalation_id, esc.status, esc.resolution_action,
                )
                return esc

            # Promote to L2, stamp park metadata, keep status='pending'.
            esc.level = 2
            esc.resolution_action = 'park'
            esc.resolution = resolution
            if resolved_by is not None:
                esc.resolved_by = resolved_by
            if resolution_turns is not None:
                esc.resolution_turns = resolution_turns
            # resolved_at stays None — parked escalation is still open/pending.

            # In-place rewrite (no archive — record stays live in queue_dir).
            self._atomic_write(escalation_id, esc.to_json())

        logger.info('Escalation %s parked (kept open at L2): %s', escalation_id, resolution[:100])

        # Fire resolve callback for the L2 itself to trigger harness teardown.
        if self._resolve_callback:
            try:
                self._resolve_callback(esc)
            except Exception as e:
                logger.warning('Park callback failed for %s: %s', escalation_id, e)

        # Cluster L2: fire per-member callback with in-memory l2-cascade stamp.
        # Do NOT persist or resolve member records — members stay pending L1s.
        if esc.members:
            for member_id in esc.members:
                try:
                    member = self.get(member_id)
                    if member is None:
                        logger.warning(
                            'park cascade: member %s of L2 %s not found — skipping',
                            member_id, escalation_id,
                        )
                        continue
                    if member.status != 'pending':
                        # Mirror resolve()'s idempotent-member contract: skip members
                        # that are already resolved/dismissed so that a terminal member
                        # does not have its task driven to 'blocked' after closure.
                        logger.debug(
                            'park cascade: member %s of L2 %s already %s — skipping',
                            member_id, escalation_id, member.status,
                        )
                        continue
                    # In-memory cascade signal only; no disk write.
                    member.resolved_by = f'l2-cascade:{escalation_id}'
                    if self._resolve_callback:
                        try:
                            self._resolve_callback(member)
                        except Exception as e:
                            logger.warning(
                                'Park cascade callback failed for member %s of L2 %s: %s',
                                member_id, escalation_id, e,
                            )
                except Exception as e:
                    logger.warning(
                        'park cascade: error processing member %s of L2 %s: %s',
                        member_id, escalation_id, e,
                    )

        return esc

    def submit_resolved(
        self,
        escalation: Escalation,
        resolution: str,
        *,
        resolved_by: str | None = None,
    ) -> Escalation:
        """Atomically write *escalation* directly in resolved state.

        Unlike the two-call ``submit()`` + ``resolve()`` pattern, this helper
        performs a single file write (no transient pending intermediate) and
        fires only ``_resolve_callback`` — never ``_notify_callback``.  This
        eliminates the spurious "pending escalation" wake that the two-call
        path emits before the resolve callback fires.

        Contract:
        - Mutates *escalation* in-place (status/resolution/resolved_at/resolved_by)
          **before** the disk write.  If the write raises (e.g. ENOSPC), the in-memory
          object is already in resolved state — callers must not reuse it after catching
          the exception.  Treat the input as consumed on call.
        - Writes the resolved JSON atomically to ``queue_dir/{id}.json`` using
          the same tmp+rename pattern as ``submit()``.
        - Best-effort archives via ``os.replace`` into the dated archive subdir
          (same two-step design as ``resolve()``): if the archive move fails,
          the resolved file stays in ``queue_dir`` where ``get()`` can find it.
        - Fires ``_resolve_callback`` once (if set); never fires ``_notify_callback``.
        - Returns *escalation* (non-Optional — disk failures raise, no None case).
        """
        # Step 1: mutate in memory
        escalation.status = 'resolved'
        escalation.resolution = resolution
        escalation.resolved_at = datetime.now(UTC).isoformat()
        if resolved_by is not None:
            escalation.resolved_by = resolved_by

        # Step 2: atomic write + best-effort archive under the per-id lock.
        # The lock prevents a concurrent same-id RMW from clobbering this write.
        # The resolve callback fires OUTSIDE the lock (see lock-scope design decision).
        with escalation_id_lock(self.queue_dir, escalation.id):
            self._atomic_write(escalation.id, escalation.to_json())
            # Step 3: best-effort archive move
            self._archive_resolved(escalation.id, escalation.resolved_at)

        # Step 4: log
        logger.info(
            'Escalation submit_resolved: %s [%s]: %s',
            escalation.id, escalation.severity, resolution[:100],
        )

        # Step 5: fire resolve callback only — never notify callback
        if self._resolve_callback:
            try:
                self._resolve_callback(escalation)
            except Exception as e:
                logger.warning(f'Resolve callback failed for {escalation.id}: {e}')

        return escalation

    def find_pending_l2_by_root_cause(self, root_cause: str) -> str | None:
        """Return the id of the oldest pending L2 escalation whose root_cause matches.

        Algorithm:
        1. Strip *root_cause*; return None immediately for empty/whitespace-only input
           (the falsy-key guard — mirrors ``dedupe.find_dedupe_parent``'s convention).
        2. Iterate ``self.get_pending()``, filtering to ``level == 2`` and
           ``esc.root_cause.strip() == candidate``.
        3. Among matches, return the id of the entry with the OLDEST timestamp
           (ISO 8601 string comparison; malformed timestamps fall back to
           ``datetime.min`` so they sort first — never silently lost).

        Cost: O(N) over pending escalations, where N is the current queue depth.
        Acceptable at current escalation rates; mirrors the existing
        ``find_dedupe_parent`` O(N) pattern in dedupe.py.
        """
        candidate = root_cause.strip()
        if not candidate:
            return None

        oldest_id: str | None = None
        oldest_ts: datetime = datetime.max.replace(tzinfo=UTC)

        for esc in self.get_pending():
            if esc.level != 2:
                continue
            if esc.root_cause.strip() != candidate:
                continue
            # Parse timestamp; fall back to datetime.min on bad input so malformed
            # entries are treated as oldest (never silently dropped). Emits a WARNING
            # (loud-over-silent) via parse_timestamp_or_warn when the timestamp is bad.
            ts, _ = parse_timestamp_or_warn(
                esc.timestamp,
                context='queue.find_pending_l2_by_root_cause',
            )
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_id = esc.id

        return oldest_id

    def add_members_to_l2(
        self, escalation_id: str, new_member_ids: list[str],
    ) -> Escalation | None:
        """Append *new_member_ids* to a pending L2 escalation's ``members`` list.

        **Concurrency contract (sidecar flock).**  The entire read-modify-write
        of ``members`` is serialized per-id by ``escalation_id_lock``, so
        concurrent appends from multiple processes are union-preserving — no
        write clobbers another's additions.  The lock target is the stable
        sidecar ``{escalation_id}.json.lock`` (see ``escalation_id_lock`` for
        the PRD-D3 rationale).

        Loads the L2 directly from ``queue_dir/{escalation_id}.json`` (queue root
        only).  This refuses archived L2s — they were already adjudicated by a
        human; re-opening them via member append would resurrect a closed decision
        (the same Defect-2 class of bug that motivated task 1498).  For the same
        reason this method never falls back to the archive.

        Set-union semantics: member ids already present in ``esc.members`` are
        not added again.  *new_member_ids* is also deduplicated internally via
        ``dict.fromkeys`` so passing ``['a', 'a', 'b']`` adds 'a' exactly once.
        New ids are appended in the order they first appear in *new_member_ids*.

        Only ``members`` is modified.  ``root_cause``, ``options``, ``summary``,
        ``detail``, and ``timestamp`` are preserved so the human-facing decision
        context remains the L2's original framing across repeated auto-watcher
        triage passes.

        Returns the updated ``Escalation`` (or the unchanged escalation when
        *new_member_ids* is empty or all ids are already present).  Returns
        ``None`` when *escalation_id* is not found in the queue root (unknown id
        or archived).
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            path = self.queue_dir / f'{escalation_id}.json'
            if not path.exists():
                return None
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse L2 escalation {escalation_id}: {e}')
                return None

            if not new_member_ids:
                return esc  # no-op

            existing = set(esc.members)
            appended = [m for m in dict.fromkeys(new_member_ids) if m not in existing]
            if appended:
                esc.members.extend(appended)
                self._rewrite(escalation_id, esc)
                logger.info(
                    'add_members_to_l2: added %d new member(s) to %s (total=%d)',
                    len(appended), escalation_id, len(esc.members),
                )
            return esc

    def attach_dedupe_child(
        self, parent_id: str, child_id: str, *, child_severity: str = 'info',
    ) -> Escalation | None:
        """Append *child_id* to the pending parent's dedupe_children list.

        **Concurrency contract (sidecar flock).**  All three mutations —
        ``dedupe_count``, ``dedupe_children``, and ``severity`` — are serialized
        per-id by ``escalation_id_lock``.  Concurrent attaches from multiple
        processes are therefore safe: no child is lost, no count is reverted,
        and no severity promotion is undone.  The lock target is the stable
        sidecar ``{parent_id}.json.lock`` (see ``escalation_id_lock`` for the
        PRD-D3 rationale; different parent ids take different sidecars —
        no cross-id contention).

        Loads the parent directly from ``queue_dir/{parent_id}.json`` — it does
        NOT fall back to the archive.  This ensures that resolved / dismissed
        parents (which have been moved to the archive) are treated as
        not-eligible and return ``None`` without any mutation.

        On a successful attach:
        - ``parent.dedupe_children`` gains *child_id* (appended).
        - ``parent.dedupe_count`` is incremented by 1.
        - ``parent.severity`` is promoted via
          ``_max_severity(parent.severity, child_severity)``; never demoted.
        - The updated parent is written back to disk via ``_rewrite()``.  Only
          this final file-replace step is atomic (``tempfile.mkstemp`` +
          ``os.rename``); the preceding in-memory mutations are not.
        - The updated ``Escalation`` object is returned.

        Returns ``None`` when:
        - The parent file does not exist in the queue root (unknown id or
          already archived / resolved — refusing to touch archived parents).

        **Recon reuse contract (A7a/A7b):**  ``dedupe.submit_or_dedupe`` calls
        this function for *both* infra and recon dedup.  Two invariants hold:

        1. ``dedupe_count`` doubles as the recon *recurrence count* — each
           successful fold into a recon parent increments it by 1, so the
           steward can read ``dedupe_count`` as "this finding recurred N times".

        2. ``parent.dedupe_fingerprint`` is **preserved** across each
           ``_rewrite()`` call — this function does not touch it.  The
           invariant is that ``submit_or_dedupe`` can match successive folds
           to the same canonical parent by fingerprint without the parent ever
           losing its fingerprint identity after the first write.
        """
        with escalation_id_lock(self.queue_dir, parent_id):
            path = self.queue_dir / f'{parent_id}.json'
            if not path.exists():
                return None
            try:
                parent = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse parent escalation {parent_id}: {e}')
                return None
            parent.dedupe_children.append(child_id)
            parent.dedupe_count += 1
            parent.severity = _max_severity(parent.severity, child_severity)
            self._rewrite(parent_id, parent)
        logger.info(
            f'Dedupe: folded {child_id} into parent {parent_id} '
            f'(dedupe_count={parent.dedupe_count}, severity={parent.severity})'
        )
        return parent

    def patch_resolution_metadata(
        self,
        escalation_id: str,
        *,
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Patch resolved_by and/or resolution_turns on an already-resolved escalation.

        Updates the existing file in place wherever it lives (root or archive); never
        resurrects an archived file into the queue root.  The tmp file used for the
        atomic write is created in the target file's parent directory, so the rename
        stays within the same filesystem subtree.

        Multi-date duplicates: when multiple archive copies exist for the same id, only
        the newest copy (newest parent directory name in YYYY-MM-DD lexicographic order)
        is patched; staler copies remain untouched.  Callers should clean up duplicate
        archive copies independently.

        Returns None when the escalation is missing OR still pending — caller must have
        already resolved/dismissed it.

        Returns the updated Escalation on success, or the unmodified Escalation when
        called with no patch arguments (both resolved_by and resolution_turns are None).
        """
        # Locate the file (shared helper — same root-first, archive-fallback,
        # newest-by-date logic as get()).
        path = self._locate_path(escalation_id)
        if path is None:
            return None

        # Parse
        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

        # Guard: only patch resolved/dismissed escalations
        if esc.status not in ('resolved', 'dismissed'):
            return None

        # No-op early return: nothing to patch — avoid disk I/O when both args are None.
        if resolved_by is None and resolution_turns is None:
            return esc

        # Apply patches
        if resolved_by is not None:
            esc.resolved_by = resolved_by
        if resolution_turns is not None:
            esc.resolution_turns = resolution_turns

        # Atomically rewrite AT THE SAME PATH (tmp file in path.parent, not queue root).
        self._atomic_write_path(path, esc.to_json())
        return esc

    def _rewrite(self, escalation_id: str, escalation: Escalation) -> None:
        """Atomically rewrite an escalation's JSON file."""
        self._atomic_write(escalation_id, escalation.to_json())

    def _atomic_write(self, escalation_id: str, json_text: str) -> None:
        """Write *json_text* atomically to ``queue_dir/{escalation_id}.json``.

        Thin wrapper around ``_atomic_write_path`` that constructs the target
        path from *escalation_id* and ``queue_dir``.  Always writes to the queue
        root — not the archive.

        Callers: submit(), resolve(), submit_resolved(), _rewrite().
        """
        self._atomic_write_path(self.queue_dir / f'{escalation_id}.json', json_text)

    def _atomic_write_path(self, path: Path, json_text: str) -> None:
        """Write *json_text* atomically to *path*.

        Uses the tmp+rename pattern: the tmp file is created in ``path.parent``
        (not hard-coded to ``queue_dir``) so that the ``os.rename`` stays within
        the same directory — required for archive targets where *path* lives in
        a dated subdir.  On failure the tmp file is cleaned up and the exception
        propagates unchanged.

        Callers: _atomic_write() (root targets), patch_resolution_metadata()
        (root or archive targets).
        """
        fd, tmp_path_str = tempfile.mkstemp(
            suffix='.tmp', prefix=path.stem, dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(json_text)
            os.rename(tmp_path_str, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            raise

    def _archive_resolved(self, escalation_id: str, resolved_at: str) -> None:
        """Best-effort move ``queue_dir/{escalation_id}.json`` into the dated archive subdir.

        Two-step design (write-to-root then os.replace to archive) is intentional:
        if the archive move fails, the resolved file remains in ``queue_dir`` so
        ``get()`` can still return it — no data is lost.  Failure logs a warning
        but does not abort the resolution.

        Callers: resolve(), submit_resolved().
        """
        path = self.queue_dir / f'{escalation_id}.json'
        try:
            archive_dir = archive.archive_dir_for_date(self.queue_dir, resolved_at)
            archive_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(path), str(archive_dir / f'{escalation_id}.json'))
            # Incrementally update the listing cache so get() finds the newly-archived
            # id without a full rescan.  Only update if the cache has already been built.
            if self._archive_listing is not None:
                self._archive_listing.setdefault(archive_dir.name, set()).add(escalation_id)
            # Upward-bump the max_seq cache for the archived task_id so make_id()
            # sees the newly-archived seq without a rescan.  Upward-only guarantee:
            # never lower the cached value (external pruning can only reduce the
            # true on-disk max, so an over-estimate is always safe; an under-estimate
            # would cause id reuse/collision).
            try:
                # escalation_id is 'esc-<task_id>-<seq>'; strip the 'esc-' prefix.
                head, tail = escalation_id.rsplit('-', 1)
                archived_task_id = head[len('esc-'):]
                archived_seq = int(tail)
            except (ValueError, IndexError):
                # Non-conforming id: cannot identify which task_id to update,
                # so leave all other cached entries intact.  Log a warning so a
                # genuinely malformed escalation_id is observable.
                logger.warning(
                    f'Could not parse task_id/seq from escalation_id {escalation_id!r}; '
                    'max_seq cache not updated for this archive write'
                )
            else:
                if archived_task_id in self._archive_max_seq_cache:
                    self._archive_max_seq_cache[archived_task_id] = max(
                        self._archive_max_seq_cache[archived_task_id], archived_seq
                    )
        except OSError as exc:
            logger.warning(
                f'Failed to archive escalation {escalation_id}: {exc}; '
                'file remains in queue_dir'
            )

    def dismiss_all_pending(self, resolution: str) -> int:
        """Dismiss all pending L0 escalations with the given resolution message.

        Level-1 escalations are PRESERVED — they represent steward→human work
        that must survive orchestrator restart and require human attention.
        Only level-0 (agent→steward) escalations are dismissed.

        Returns the number of escalations where resolve() returned non-None.
        In the common single-writer case this equals the number actually dismissed.
        In a concurrent-write race (another process resolves an escalation between
        get_pending() and resolve()), resolve() is a no-op but still returns the
        existing escalation, so the count may include those no-ops.  The counter
        is best read as "attempted dismissals, including no-ops for concurrent
        resolutions".  This function is single-writer in practice.
        """
        pending = self.get_pending()
        count = 0
        for esc in (e for e in pending if e.level == 0):
            try:
                if self.resolve(esc.id, resolution, dismiss=True, resolved_by='auto-dismissed') is not None:
                    count += 1
            except Exception as e:
                logger.warning(f'Failed to dismiss escalation {esc.id}: {e}')
        if count:
            logger.info(f'Dismissed {count} stale L0 escalation(s): {resolution[:100]}')
        return count

    def make_id(self, task_id: str) -> str:
        """Generate a unique escalation ID.

        Scans existing files in both the queue root and the archive
        subdirectory to avoid reusing sequence numbers from escalations
        that have been archived after resolution (the in-memory counter
        resets on process restart, so we must re-derive max_seq from disk).

        The queue root is always scanned fresh each call (cheap; reflects
        just-submitted pending ids and cross-process root writes).  The
        archive contribution is cached per task_id in _archive_max_seq_cache:
        on a cache miss, _scan_archive_max_seq() scans the archive once and
        stores the result; subsequent calls for the same task_id are O(1).
        _archive_resolved() bumps the cache upward on each archive write.

        Per-instance freshness: the cache is correct only for the
        single-writer model (one EscalationQueue instance owns a given
        task_id's escalations).  If a concurrent process archives an
        escalation for a task_id whose entry is already seeded in this
        instance's cache, the cache will not reflect that higher seq until
        this instance's _archive_resolved() runs for that task_id.  In
        production, harness.py owns a single long-lived EscalationQueue
        per process and sentinel task_ids are not shared across processes,
        so this invariant holds.
        """
        prefix = f'esc-{task_id}-'
        max_seq = 0
        for path in self.queue_dir.glob(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        # Obtain the archive contribution from the per-task_id cache.
        # Scan the archive exactly once per task_id per instance (cache miss only).
        if task_id not in self._archive_max_seq_cache:
            self._archive_max_seq_cache[task_id] = self._scan_archive_max_seq(prefix)
        archive_max = self._archive_max_seq_cache[task_id]
        max_seq = max(max_seq, archive_max)
        seq = max(max_seq + 1, self._next_seq())
        return f'{prefix}{seq}'
