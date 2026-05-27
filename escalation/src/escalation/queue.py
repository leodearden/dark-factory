"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timezone
from pathlib import Path

from escalation import archive
from escalation.models import Escalation

logger = logging.getLogger(__name__)

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

    def set_notify_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._notify_callback = callback

    def set_resolve_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._resolve_callback = callback

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

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
        """Atomic write: {id}.tmp -> rename to {id}.json."""
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
        # Fall back to archive: search all dated subdirs.
        candidates = list(self._iter_archive_paths(f'{escalation_id}.json'))
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

        Note: the archive fallback performs an O(|archive|) rglob on every miss.
        For a 30-day retention window this is bounded, but callers on hot paths
        should avoid repeated get() calls for ids known to be archived.
        TODO: memoise the archive listing per dated subdir to reduce repeated scans.
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

        Idempotent: if the escalation is already resolved or dismissed, this
        method returns the existing escalation unchanged without re-archiving
        or re-firing the _resolve_callback.
        """
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

        # Step 2: atomic write then best-effort archive (delegates to shared helpers)
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
        oldest_ts: datetime = datetime.max.replace(tzinfo=timezone.utc)

        for esc in self.get_pending():
            if esc.level != 2:
                continue
            if esc.root_cause.strip() != candidate:
                continue
            # Parse timestamp; fall back to datetime.min on bad input so malformed
            # entries are treated as oldest (never silently dropped).
            try:
                ts = datetime.fromisoformat(esc.timestamp)
                # Ensure tz-aware for comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                ts = datetime.min.replace(tzinfo=timezone.utc)
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_id = esc.id

        return oldest_id

    def attach_dedupe_child(
        self, parent_id: str, child_id: str, *, child_severity: str = 'info',
    ) -> Escalation | None:
        """Append *child_id* to the pending parent's dedupe_children list.

        **Not concurrency-safe.**  The read-modify-write of ``dedupe_count``,
        ``dedupe_children``, and ``severity`` is *not* atomic: two concurrent
        callers for the same parent each read the same pre-mutation snapshot,
        both append once and both write with the same incremented count, and
        the second rewrite silently clobbers the first — losing a child and
        potentially reverting a severity promotion.  The caller must serialize
        concurrent attaches against the same parent.  Today this invariant
        holds because the MCP server is single-writer; any multi-writer
        migration must add explicit serialization for all three fields before
        calling this function.

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

        Note: the archive scan is O(|archive files for task_id|) on every call.
        make_id() is a slow path (invoked only at submission), so this is
        acceptable within a 30-day retention window.
        TODO: cache max_seq per task_id to eliminate repeated archive scans.
        """
        prefix = f'esc-{task_id}-'
        max_seq = 0
        for path in self.queue_dir.glob(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        # Also scan the archive so post-restart calls don't return IDs
        # already used by archived escalations for the same task.
        for path in self._iter_archive_paths(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        seq = max(max_seq + 1, self._next_seq())
        return f'{prefix}{seq}'
