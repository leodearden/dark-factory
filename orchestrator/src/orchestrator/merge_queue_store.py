"""Durable journal for the SpeculativeMergeWorker's owned merge requests (task 1772).

Persists a worker-owned subset of in-flight MergeRequests so they can be
recovered after an orchestrator restart (crash / redeploy / watchdog kill).

Design highlights
-----------------
* Only ``MergeRequest`` (not ``GroupMergeRequest``) is journaled — train merges
  carry live scheduler callbacks that cannot be serialized.
* Atomic file I/O (tmp + os.replace) mirrors b3_gate.py ``_save_state``.
* Fail-open reads: missing / empty / corrupt file → empty list; startup never
  crashes because of a damaged journal.
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

        state = self._load_raw()
        state[req.request_id] = asdict(persisted)
        self._save_raw(state)

    def remove(self, request_id: str) -> None:
        """Remove *request_id* from the journal.

        No-op if the id is not present (idempotent terminal cleanup).
        """
        state = self._load_raw()
        if request_id in state:
            del state[request_id]
            self._save_raw(state)

    def load(self) -> list[PersistedMergeRequest]:
        """Return all journaled records as ``PersistedMergeRequest`` objects.

        Returns ``[]`` if the file is missing, empty, or corrupt (fail-open).
        """
        state = self._load_raw()
        result: list[PersistedMergeRequest] = []
        for entry in state.values():
            try:
                result.append(PersistedMergeRequest(**entry))
            except (TypeError, KeyError) as exc:
                logger.warning('merge_queue_store: skipping malformed entry: %s', exc)
        return result

    # ------------------------------------------------------------------
    # Internal I/O (mirrors b3_gate._save_state / _load_state)
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict[str, Any]:
        """Load the JSON map from disk; return {} on any error (fail-open)."""
        try:
            text = self._path.read_text(encoding='utf-8')
            if not text.strip():
                return {}
            return json.loads(text)
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

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
          (branch already landed on main — idempotency guard).
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

            # Branch exists and is not yet merged — reconstruct and re-enqueue.
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
