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
  ``_save_raw`` chokepoint which calls ``shared.safe_io.atomic_write_text``
  with ``fsync=True`` — the temp file is fsynced BEFORE ``os.replace`` and the
  parent directory AFTER (WA-1), so a row is durable across a crash
  immediately following ``record()``. This is the only site in the repo that
  asks for the fsyncs, which is why they are opt-in there.
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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared import safe_io
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

        ``fsync=True`` is what makes this durable rather than merely atomic:
        :func:`shared.safe_io.atomic_write_text` fsyncs the temp file's fd
        BEFORE ``os.replace`` and the parent directory's fd AFTER — a rename
        without a directory fsync can be lost on crash (the directory-entry
        update is a separate durability domain from the file's own contents on
        most filesystems). This is the only site in the repo that needs it,
        which is why the helper's default is off.

        ``mode`` is deliberately left at the helper's umask default rather than
        narrowed: this file is read by other processes.

        Fail-open, and that policy stays HERE rather than in the helper (which
        always propagates): any OSError is logged at ERROR (this is a
        lost-durability event, not a benign condition) and counted in
        ``self.save_failures``, but never raised — a transient disk failure
        must never block the merge pipeline. Temp cleanup on failure is now the
        helper's own guarantee, so this block no longer does it.
        """
        try:
            safe_io.atomic_write_text(
                self._path,
                json.dumps(state),
                encoding='utf-8',
                fsync=True,
                mkdir=True,
            )
        except OSError as exc:
            self.save_failures += 1
            logger.error(
                'landed_outbox: failed to durably save outbox (row may only'
                ' be held in memory): %s', exc,
            )


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

    @staticmethod
    def consume(task_id: str) -> None:
        """Idempotently prune the landed row for task_id from the bound outbox; no-op when unbound (fail-safe) or absent."""
        if MergeProvenance._outbox is None:
            return
        MergeProvenance._outbox.consume(task_id)
