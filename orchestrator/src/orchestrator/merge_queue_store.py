"""Durable journal for the SpeculativeMergeWorker's owned merge requests (task 1772).

Persists a worker-owned subset of in-flight MergeRequests so they can be
recovered after an orchestrator restart (crash / redeploy / watchdog kill).

Design highlights
-----------------
* Only ``MergeRequest`` (not ``GroupMergeRequest``) is journaled — train merges
  carry live scheduler callbacks that cannot be serialized.
* Atomic file I/O (tmp + os.replace) mirrors b3_gate.py ``_save_state``.
* Fail-open reads: missing file → empty list (benign, silent); empty / corrupt
  file → empty list **plus** ``journal_corrupt=True`` and a deduped WARNING
  (empty files are treated as corrupt because _save_raw never writes an empty
  string, so an empty file is an anomaly, not a legitimate fresh state).
* Keyed by ``request_id`` so ``record()`` is idempotent on redispatch (updates
  in place) and ``remove()`` is O(1) on terminal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.safe_io import load_json_or_warn

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.merge_queue import MergeRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persisted data model
# ---------------------------------------------------------------------------


@dataclass
class PersistedMergeRequest:
    """Serializable identity subset of a MergeRequest.

    All fields are JSON-safe primitives or None.  Non-serializable fields
    (result Future, config, module_configs) are excluded — they are
    re-injected at recovery time.
    """

    request_id: str
    task_id: str
    branch: str
    worktree: str          # stored as str; Path on reconstruction
    pre_rebased: bool
    task_files: list[str] | None
    snapshot_tip: str | None
    generation: int
    lane: str
    enqueued_at: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MergeQueueStore:
    """Durable journal for owned merge requests.

    Backed by an atomic JSON file at *path*, keyed by ``request_id``.

    All writes are fail-open for the caller: if a write fails the journal may
    be stale, but the merge pipeline is never blocked.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        # Set journal_corrupt BEFORE _load_raw so the attribute always exists
        # even if _load_raw raises unexpectedly (it shouldn't, but defensive).
        self.journal_corrupt: bool = False
        # In-memory mirror of the on-disk journal, warmed at construction time.
        # All mutations go through this dict so record()/remove() never re-read
        # the file — they just update the cache and write it out atomically.
        # This keeps blocking I/O off the hot merge-drain path (suggestion 3).
        self._cache: dict[str, Any] = self._load_raw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, req: MergeRequest) -> None:  # type: ignore[type-arg]
        """Persist *req*'s serializable identity.

        Silently skips ``GroupMergeRequest`` (train merges are not journaled).
        Re-recording the same ``request_id`` overwrites the previous entry
        (idempotent on redispatch).
        """
        # Import here to keep the dependency one-directional:
        # merge_queue_store -> merge_queue (not the reverse).
        from orchestrator.merge_queue import GroupMergeRequest

        if isinstance(req, GroupMergeRequest):
            return

        persisted = PersistedMergeRequest(
            request_id=req.request_id,
            task_id=req.task_id,
            branch=req.branch,
            worktree=str(req.worktree),
            pre_rebased=req.pre_rebased,
            task_files=req.task_files,
            snapshot_tip=req.snapshot_tip,
            generation=req.generation,
            lane=req.lane,
            enqueued_at=req.enqueued_at,
        )

        # Update in-memory mirror first; then flush atomically without re-reading.
        self._cache[req.request_id] = asdict(persisted)
        self._save_raw(self._cache)

    def remove(self, request_id: str) -> None:
        """Remove *request_id* from the journal.

        No-op if the id is not present (idempotent terminal cleanup).
        """
        if request_id in self._cache:
            del self._cache[request_id]
            self._save_raw(self._cache)

    def load(self) -> list[PersistedMergeRequest]:
        """Return all journaled records as ``PersistedMergeRequest`` objects.

        Reads from the in-memory mirror (warmed at construction; kept in sync
        by record/remove).  Returns ``[]`` if the journal is empty or was
        corrupt at startup (fail-open).
        """
        result: list[PersistedMergeRequest] = []
        for entry in self._cache.values():
            try:
                result.append(PersistedMergeRequest(**entry))
            except (TypeError, KeyError) as exc:
                logger.warning('merge_queue_store: skipping malformed entry: %s', exc)
        return result

    # ------------------------------------------------------------------
    # Internal I/O (mirrors b3_gate._save_state / _load_state)
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict[str, Any]:
        """Load the JSON map from disk; return {} on any error.

        Missing file → {} (benign, silent, journal_corrupt stays False).
        Empty / corrupt file → {} + journal_corrupt=True + deduped WARNING.
        Non-dict but valid JSON → {} + journal_corrupt=True + WARNING.
        Non-FileNotFoundError OSErrors propagate (unexpected environment failure).
        """
        data, ok = load_json_or_warn(self._path, default={}, on_corrupt='warn')
        if not ok:
            self.journal_corrupt = True
            return {}
        if not isinstance(data, dict):
            self.journal_corrupt = True
            logger.warning(
                'merge_queue_store: journal at %s is not a JSON object (got %s);'
                ' treating as corrupt',
                self._path,
                type(data).__name__,
            )
            return {}
        return data

    def _save_raw(self, state: dict[str, Any]) -> None:
        """Atomically write *state* to disk (tmp + os.replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(state), encoding='utf-8')
            os.replace(str(tmp), str(self._path))
        except OSError as exc:
            logger.warning('merge_queue_store: failed to save journal: %s', exc)


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------


def reconstruct_merge_request(
    persisted: PersistedMergeRequest,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a live ``MergeRequest`` from a persisted record.

    The returned request has:
    * A fresh unresolved ``asyncio.Future`` (the original died with the crash).
    * Preserved ``request_id`` and all identity fields.
    * ``module_configs=[]`` — re-injecting the full scope is not possible
      post-restart; [] causes the worker to run the default verification scope.
    * ``pre_rebased=False`` — ensures the worker rebases before merging.
    * ``config`` from the live harness (current config is always correct).
    """
    from orchestrator.merge_queue import MergeRequest  # local import; avoid circularity

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    # NOTE — module_configs divergence (suggestion 4):
    # The original request may have been scoped to a narrower set of verification
    # modules.  That information is not persisted (it holds live objects).  The
    # recovered request therefore runs the *default* verification scope ([] means
    # "all configured modules").  This is intentionally conservative — it errs on
    # the side of verifying more than the original, never less.  Callers that care
    # about scope parity should persist a scope hint in task_files (which IS
    # preserved) and filter inside the verifier.
    return MergeRequest(
        task_id=persisted.task_id,
        branch=persisted.branch,
        worktree=Path(persisted.worktree),
        pre_rebased=False,
        task_files=persisted.task_files,
        module_configs=[],
        config=config,
        result=future,
        request_id=persisted.request_id,
        snapshot_tip=persisted.snapshot_tip,
        generation=persisted.generation,
        lane=persisted.lane,  # type: ignore[arg-type]
        enqueued_at=persisted.enqueued_at,
    )


# ---------------------------------------------------------------------------
# Recovery entry-point
# ---------------------------------------------------------------------------


async def recover_pending_merges(
    store: MergeQueueStore,
    queue: asyncio.Queue,  # type: ignore[type-arg]
    git_ops: Any,
    config: OrchestratorConfig,
    event_store: Any = None,
    *,
    main_branch: str,
    branch_prefix: str,
    retention: Any = None,
) -> dict[str, Any]:
    """Re-enqueue surviving merge requests from the durable journal.

    For each journaled record:
    * Builds ``full_branch = f'{branch_prefix}{record.branch}'``.
    * Drops the record (and removes it from the store) if:
        - ``git_ops.resolve_branch_sha(full_branch)`` is ``None``
          (branch was deleted after the crash), OR
        - ``git_ops.is_ancestor(full_branch, main_branch)`` is ``True``
          (branch already landed on main — idempotency guard), OR
        - The persisted worktree path no longer exists on disk
          (pruned by crash cleanup / redeploy — drop so the scheduler can
          rediscover and re-dispatch through normal channels).
    * Otherwise reconstructs a fresh ``MergeRequest`` and enqueues it via
      ``enqueue_merge_request`` so ``_on_finalized`` re-arms the durable
      ``merge_finalized`` event for any polling caller.

    Errors on individual records are logged and skipped (fail-open) so one
    bad entry never aborts the whole recovery pass.

    Returns a dict with ``recovered`` and ``dropped`` counts.
    """
    from orchestrator.merge_queue import enqueue_merge_request  # avoid circular

    records = store.load()
    recovered = 0
    dropped = 0
    recovered_requests: list[MergeRequest] = []

    for record in records:
        full_branch = f'{branch_prefix}{record.branch}'
        try:
            sha = await git_ops.resolve_branch_sha(full_branch)
            if sha is None:
                # Branch gone — drop without re-enqueuing.
                store.remove(record.request_id)
                dropped += 1
                logger.info(
                    'merge_queue_store: dropping %s (branch %s gone)',
                    record.request_id,
                    full_branch,
                )
                continue

            already_merged = await git_ops.is_ancestor(full_branch, main_branch)
            if already_merged:
                store.remove(record.request_id)
                dropped += 1
                logger.info(
                    'merge_queue_store: dropping %s (branch %s already on %s)',
                    record.request_id,
                    full_branch,
                    main_branch,
                )
                continue

            # Worktree-existence check (suggestion 5):
            # A crash or redeploy may have pruned the worktree directory.  If
            # the worktree is gone the worker cannot rebase/merge and would
            # produce an 'error' outcome that (without this guard) would be
            # re-enqueued on every subsequent restart — an indefinite retry loop.
            # Drop the record so the next scheduler pass can rediscover and
            # re-dispatch the task through normal channels.
            worktree_path = Path(record.worktree)
            if not worktree_path.exists():
                store.remove(record.request_id)
                dropped += 1
                logger.info(
                    'merge_queue_store: dropping %s (worktree %s gone)',
                    record.request_id,
                    record.worktree,
                )
                continue

            # Branch exists, is not yet merged, and worktree is present —
            # reconstruct and re-enqueue.
            req = reconstruct_merge_request(record, config)
            await enqueue_merge_request(queue, req, event_store, retention=retention)
            recovered_requests.append(req)
            recovered += 1
            logger.info(
                'merge_queue_store: recovered %s (branch %s)',
                record.request_id,
                full_branch,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'merge_queue_store: error recovering %s: %s',
                record.request_id,
                exc,
                exc_info=True,
            )

    return {'recovered': recovered, 'dropped': dropped, 'requests': recovered_requests}
