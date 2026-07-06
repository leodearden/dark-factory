"""Durable outbox for landed (main-advanced) merge requests (task 2153 / W1 α).

Journals the SpeculativeMergeWorker's CAS-advance results — the deliberate
CONTRAST to merge_queue_store.MergeQueueStore's accept-side journal, which is
``remove()``'d on terminal (merge_queue_store.py:131). A landed row must
SURVIVE until :meth:`LandedOutbox.consume` confirms the task is done (WA-3):
the startup reconciler and the scheduler consult-before-dispatch gate (both
separate in-batch tasks) need to observe a landed row that has not yet been
consumed.

Design highlights
-----------------
* Keyed by ``task_id`` — last-write-wins (WA-2), mirroring
  ``MergeQueueStore``'s ``{request_id: value}`` shape.
* Every mutation (``record``/``consume``) flushes through a single
  ``_save_raw`` chokepoint that fsyncs the temp file BEFORE ``os.replace``
  and the parent directory AFTER (WA-1) — durable across a crash immediately
  following ``record()``. No prior ``os.fsync``-on-file precedent exists in
  this repo (only SQLite ``PRAGMA synchronous=FULL``), so this mirrors
  ``merge_queue_store._save_raw``'s atomic tmp+``os.replace`` and adds the
  two fsyncs.
* Fail-open reads via ``shared.safe_io.load_json_or_warn`` (same as
  ``MergeQueueStore``): a missing file is a silent empty store; a corrupt
  file is a warned empty store.
* ``MergeProvenance`` is a thin process-global façade over ONE bound
  ``LandedOutbox`` (the worker's), so non-worker callers (e.g. the scheduler
  gate) can do ``MergeProvenance.lookup(task_id)`` without holding a
  reference to the worker itself. ``lookup`` returns ``None`` when unbound
  (fail-safe for bare-worker/no-worker contexts).

This task (α) delivers the store + façade + worker-held instance ONLY.
``record()`` is not yet called at any CAS-advance site — that is a separate
in-batch task (β).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.safe_io import load_json_or_warn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LandedRow:
    """A single landed (main-advanced) merge record.

    Frozen — landed rows are immutable value objects; a re-record for the
    same ``task_id`` replaces the LandedOutbox entry wholesale (WA-2) rather
    than mutating a field in place.
    """

    task_id: str
    branch_tip_sha: str
    advanced_sha: str
    landed_at: float


class LandedOutbox:
    """Durable, fsync'd journal of landed merge requests, keyed by task_id.

    Backed by an atomic JSON file at *path*. All reads (``lookup``/``all``)
    serve an in-memory cache warmed at construction time; all mutations
    (``record``/``consume``) update the cache then flush it atomically and
    durably before returning (WA-1).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        # In-memory mirror of the on-disk journal, warmed at construction
        # time. All mutations go through this dict so lookup()/all() never
        # re-read the file — mirrors MergeQueueStore's _cache pattern.
        self._cache: dict[str, dict[str, Any]] = self._load_raw()
        # Observability counter: bumped each time _save_raw fail-opens on an
        # OSError, i.e. record()/consume() returned normally but the row was
        # NOT durably flushed (still only in self._cache). record()/consume()
        # deliberately never raise for this (never block the merge
        # pipeline), so this counter is the only in-process signal that
        # durability was lost — callers/metrics may poll it.
        self.save_failures: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, row: LandedRow) -> None:
        """Persist *row*, overwriting any existing entry for ``row.task_id`` (WA-2).

        Flushes through :meth:`_save_raw` before returning — fsync'd durable
        (WA-1).
        """
        self._cache[row.task_id] = {
            'branch_tip_sha': row.branch_tip_sha,
            'advanced_sha': row.advanced_sha,
            'landed_at': row.landed_at,
        }
        self._save_raw(self._cache)

    def lookup(self, task_id: str) -> LandedRow | None:
        """Return the landed row for *task_id*, or None if not present."""
        value = self._cache.get(task_id)
        if value is None:
            return None
        return LandedRow(task_id=task_id, **value)

    def all(self) -> list[LandedRow]:
        """Return every landed row currently in the outbox (for the startup reconciler)."""
        return [LandedRow(task_id=k, **v) for k, v in self._cache.items()]

    def consume(self, task_id: str) -> None:
        """Idempotently prune *task_id* from the outbox (WA-3).

        No-op if *task_id* is not present (repeated consume / unknown id).
        Flushes through :meth:`_save_raw` so the prune is durable — a
        consumed row does not resurrect after a restart.
        """
        if task_id not in self._cache:
            return
        del self._cache[task_id]
        self._save_raw(self._cache)

    # ------------------------------------------------------------------
    # Internal I/O (mirrors merge_queue_store._load_raw/_save_raw)
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict[str, dict[str, Any]]:
        """Load the JSON map from disk; return {} on any error (fail-open).

        A row that is valid JSON but schema-drifted (not a dict, or missing/
        extra keys relative to :class:`LandedRow`'s fields) is dropped
        individually with a WARNING rather than left in the cache — without
        this guard, ``lookup()``/``all()`` would raise ``TypeError`` the
        first time they reconstructed that row via
        ``LandedRow(task_id=k, **v)``, defeating the fail-open contract for
        callers like the scheduler's consult-before-dispatch gate.
        """
        data, ok = load_json_or_warn(self._path, default={}, on_corrupt='warn')
        if not ok or not isinstance(data, dict):
            return {}
        clean: dict[str, dict[str, Any]] = {}
        for task_id, value in data.items():
            if not isinstance(value, dict):
                logger.warning(
                    'landed_outbox: dropping non-dict row for task_id=%r', task_id,
                )
                continue
            try:
                LandedRow(task_id=task_id, **value)
            except TypeError as exc:
                logger.warning(
                    'landed_outbox: dropping schema-drifted row for task_id=%r: %s',
                    task_id, exc,
                )
                continue
            clean[task_id] = value
        return clean

    def _save_raw(self, state: dict[str, Any]) -> None:
        """Atomically AND durably write *state* to disk (WA-1).

        fsyncs the temp file's fd BEFORE ``os.replace``, then fsyncs the
        parent directory's fd AFTER — a rename without a directory fsync can
        be lost on crash (the directory-entry update is a separate
        durability domain from the file's own contents on most
        filesystems). Fail-open: any OSError is logged at ERROR (this is a
        lost-durability event, not a benign condition) and counted in
        ``self.save_failures``, but never raised — a transient disk failure
        must never block the merge pipeline. On failure, best-effort cleans
        up the sibling ``.json.tmp`` so a partial-write failure doesn't
        leave an orphan file behind (the tmp is otherwise consumed by
        ``os.replace`` on the success path).
        """
        tmp = self._path.with_suffix('.json.tmp')
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(json.dumps(state))
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self._path))
            dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            self.save_failures += 1
            logger.error(
                'landed_outbox: failed to durably save outbox (row may only'
                ' be held in memory): %s', exc,
            )
            # Best-effort cleanup; the save failure is already reported above.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


class MergeProvenance:
    """Process-global façade over the worker's bound LandedOutbox.

    The contract fixes ``lookup`` as a ``@staticmethod`` (PRD §8.2), so
    binding is done via a class-level reference set by :meth:`bind` rather
    than the façade holding its own outbox instance. Single-worker-per-
    process means the last bind wins. ``lookup`` is fail-safe: returns
    ``None`` when nothing is bound (e.g. a bare-worker/no-worker test
    context) instead of raising.
    """

    _outbox: LandedOutbox | None = None

    @classmethod
    def bind(cls, outbox: LandedOutbox) -> None:
        """Bind *outbox* as the process-global provenance source."""
        cls._outbox = outbox

    @staticmethod
    def lookup(task_id: str) -> LandedRow | None:
        """Return the landed row for *task_id*, or None if unbound/absent."""
        if MergeProvenance._outbox is None:
            return None
        return MergeProvenance._outbox.lookup(task_id)
