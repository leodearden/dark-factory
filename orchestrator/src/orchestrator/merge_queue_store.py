"""Durable journal for the SpeculativeMergeWorker's owned merge requests (task 1772).

Persists a worker-owned subset of in-flight MergeRequests so they can be
recovered after an orchestrator restart (crash / redeploy / watchdog kill).

Design highlights
-----------------
* Only ``MergeRequest`` (not ``GroupMergeRequest``) is journaled — train merges
  carry live scheduler callbacks that cannot be serialized.
* Atomic file I/O via ``shared.safe_io.atomic_write_text`` (task 3223).
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared import safe_io
from shared.safe_io import load_json_or_warn

from orchestrator.merge_types import (
    InFlightMergeRegistry,
    QueuedBranch,
    WaiterRecord,
)

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

        # Persist the bare canonical shape (task 2037 fix 2 / task ν): the
        # journal only ever holds ONE canonical shape (the bare id), regardless
        # of what shape the caller submitted.  branch is now a typed
        # QueuedBranch, so .bare_id is the byte-identical successor to the old
        # removeprefix(branch_prefix) strip.
        persisted = PersistedMergeRequest(
            request_id=req.request_id,
            task_id=req.task_id,
            branch=req.branch.bare_id,
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
        """Atomically write *state* to disk.

        ``mode`` is deliberately left at the helper's umask default rather than
        narrowed: this journal is read by other processes.

        Fail-open, and that policy stays HERE rather than in the helper (which
        always propagates): a failure to journal must never take down the merge
        worker, so any OSError is logged at WARNING and swallowed.
        """
        try:
            safe_io.atomic_write_text(
                self._path,
                json.dumps(state),
                encoding='utf-8',
                mkdir=True,
            )
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
        branch=QueuedBranch.parse(persisted.branch, config.git.branch_prefix),
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
    registry: InFlightMergeRegistry | None = None,
) -> dict[str, Any]:
    """Re-enqueue surviving merge requests from the durable journal.

    For each journaled record:
    * Builds ``full_branch = QueuedBranch.parse(record.branch,
      branch_prefix).full_name`` — shape-tolerant so a legacy journal entry that was
      already persisted in the prefixed shape (e.g. ``'task/4959'``, from a
      journal written before the step-4 enqueue normalization) is not
      double-prefixed into an unresolvable ``'task/task/4959'``.
    * Drops the record (and removes it from the store) if:
        - ``git_ops.resolve_branch_sha(full_branch)`` is ``None``
          (branch was deleted after the crash), OR
        - ``git_ops.is_ancestor(full_branch, main_branch)`` is ``True``
          (branch already landed on main — idempotency guard), OR
        - The persisted worktree path no longer exists on disk
          (pruned by crash cleanup / redeploy — drop so the scheduler can
          rediscover and re-dispatch through normal channels).
    * Otherwise the record SURVIVES (Phase 1).  Surviving records are then
      reconstructed into live ``MergeRequest``s and enqueued via
      ``enqueue_merge_request`` so ``_on_finalized`` re-arms the durable
      ``merge_finalized`` event for any polling caller (Phase 2).

    Registry-gated per-branch collapse (task 2926, C3 γ)
    ----------------------------------------------------
    When *registry* (the shared :class:`InFlightMergeRegistry`) is supplied,
    Phase 2 collapses per-branch duplicates through it BEFORE enqueue so a
    branch with two surviving journal entries (e.g. the 2026-07-22 task/5326
    double-rehydrate) never dispatches two concurrent verifies of one work
    item.  Survivors are grouped by ``QueuedBranch.bare_id`` (first-seen order);
    for each group the descendant-most snapshot tip is chosen as the WINNER via
    :func:`orchestrator.merge_queue.select_recovery_winner`, which routes the
    pairwise ancestry decision through δ's shared
    ``decide_c3_submit_action(verifying=False)`` (SAME/SUBSET keep the winner,
    SUPERSET replaces it, a DIVERGENT force-push folds through
    ``resolve_divergent`` and logs a D2 WARNING).  The winner acquires the
    registry slot (``source='recovery'``) and is enqueued exactly once; every
    loser is attached as a peer :class:`WaiterRecord` so BOTH requesters resolve
    via the registry's primary→peer future-mirror (the coalesce attach is
    observed, not inferred), then dropped from the journal.  Losers are absent
    from ``report['requests']`` so their worktrees are cleaned by the existing
    ``_reap_orphaned_merge_worktrees`` pass (γ does not call the C1 primitive).

    When *registry* is None (default) Phase 2 preserves the pre-γ behaviour
    exactly — every survivor is reconstructed and enqueued directly — so every
    existing caller/test is unaffected.

    Errors on individual records / branch groups are logged and skipped
    (fail-open) so one bad entry never aborts the whole recovery pass.

    Returns a dict with ``recovered``, ``dropped`` and ``coalesced`` counts,
    the enqueued ``requests``, and the ``journal_corrupt`` flag.
    """
    from orchestrator.merge_queue import (  # avoid circular
        enqueue_merge_request,
        select_recovery_winner,
    )

    # Detect and loudly surface a corrupt journal so pending merges are NOT
    # silently dropped — operators must see the distinction between a corrupt
    # journal (possible data loss) and a fresh/empty one (nothing to recover).
    if store.journal_corrupt:
        logger.warning(
            'merge_queue_store: journal was corrupt at startup — pending merges'
            ' may have been lost and cannot be recovered from the journal;'
            ' scheduler will rediscover/redispatch affected tasks'
        )

    records = store.load()
    recovered = 0
    dropped = 0
    coalesced = 0
    recovered_requests: list[MergeRequest] = []

    # ── Phase 1: per-record survival checks (unchanged drop semantics) ──────
    # Collect survivors as (record, QueuedBranch) in journal order.  Errors on
    # a single record are logged and that record skipped (fail-open) so one bad
    # entry never aborts the whole pass.
    survivors: list[tuple[PersistedMergeRequest, QueuedBranch]] = []
    for record in records:
        # Shape-tolerant: QueuedBranch.parse prepends branch_prefix unless
        # record.branch is already prefixed (legacy on-disk journals may
        # predate the enqueue normalization).  .full_name is the exact git ref
        # the existence checks resolve against (byte-identical to the old
        # canonical_queued_branch_name result).
        qb = QueuedBranch.parse(record.branch, branch_prefix)
        full_branch = qb.full_name
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

            # Branch exists, is not yet merged, and worktree is present.
            survivors.append((record, qb))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'merge_queue_store: error recovering %s: %s',
                record.request_id,
                exc,
                exc_info=True,
            )

    # ── Phase 2a: no registry → back-compat direct enqueue per survivor ─────
    # Preserves the pre-γ behaviour exactly so every existing caller/test that
    # does not pass a registry is unaffected.
    if registry is None:
        for record, qb in survivors:
            try:
                req = reconstruct_merge_request(record, config)
                await enqueue_merge_request(
                    queue, req, event_store, retention=retention,
                )
                recovered_requests.append(req)
                recovered += 1
                logger.info(
                    'merge_queue_store: recovered %s (branch %s)',
                    record.request_id,
                    qb.full_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'merge_queue_store: error recovering %s: %s',
                    record.request_id,
                    exc,
                    exc_info=True,
                )
        return {
            'recovered': recovered,
            'dropped': dropped,
            'coalesced': coalesced,
            'requests': recovered_requests,
            'journal_corrupt': store.journal_corrupt,
        }

    # ── Phase 2b: registry-gated per-branch collapse (task 2926, C3 γ) ──────
    # Group survivors by bare branch id (dict preserves first-seen order), then
    # enqueue exactly ONE winner per branch and attach the rest as peers.
    groups: dict[str, list[tuple[PersistedMergeRequest, QueuedBranch]]] = {}
    for record, qb in survivors:
        groups.setdefault(qb.bare_id, []).append((record, qb))

    for bare_id, group in groups.items():
        try:
            # Pick the descendant-most snapshot tip as the single enqueued
            # winner (order-independent) via the SHARED C3 disposition table;
            # nothing is verifying at recovery so REJECT is unreachable.
            winner_idx, divergence_seen = await select_recovery_winner(
                [rec.snapshot_tip for rec, _ in group], git_ops,
            )
            winner_record, winner_qb = group[winner_idx]
            winner_req = reconstruct_merge_request(winner_record, config)

            # Acquire the winner's slot ONCE, then enqueue behind a slot-leak
            # release-on-fail guard (mirrors coalesce_or_enqueue_merge_request).
            if registry.acquire(
                bare_id,
                winner_req.task_id,
                winner_req.result,
                request_id=winner_req.request_id,
                source='recovery',
                snapshot_tip=winner_record.snapshot_tip,
            ):
                try:
                    await enqueue_merge_request(
                        queue, winner_req, event_store, retention=retention,
                    )
                except BaseException:
                    # Slot-leak guard: release the freshly-claimed slot if the
                    # enqueue fails before the worker can ever resolve the future.
                    registry.release(bare_id)
                    raise
                recovered_requests.append(winner_req)
                recovered += 1
                # The enqueued winner IS the in-flight primary; every loser
                # aliases onto its request_id (below).
                primary_request_id: str | None = winner_req.request_id
                logger.info(
                    'merge_queue_store: recovered %s (branch %s)',
                    winner_record.request_id,
                    winner_qb.full_name,
                )
            else:
                # The registry is EMPTY at startup so acquire almost always
                # succeeds; a concurrent live merge_request could theoretically
                # hold the slot first.  Attach the recovered winner as a peer so
                # its future still resolves with the in-flight outcome — never a
                # second work item — and count it as coalesced.  Here the winner
                # is NOT the primary: the pre-existing in-flight entry is.
                existing = registry.entry(bare_id)
                primary_request_id = (
                    existing.request_id if existing is not None else None
                )
                registry.attach(bare_id, WaiterRecord(
                    request_id=winner_req.request_id,
                    future=winner_req.result,
                    source='recovery',
                    submitted_tip=winner_record.snapshot_tip,
                ))
                # The winner itself coalesced onto the pre-existing primary, so
                # alias IT onto that primary too (mirrors
                # coalesce_or_enqueue_merge_request) — otherwise a durable poll
                # on the winner's id would dangle.  Losers below alias onto the
                # SAME real primary, never onto the non-primary winner.  In
                # production retention is None on the recovery path (the harness
                # does not thread it — see the module docstring), so this is
                # correct-by-construction rather than currently exercised.
                if retention is not None and primary_request_id:
                    retention.record_alias(
                        winner_req.request_id, primary_request_id,
                    )
                store.remove(winner_record.request_id)
                coalesced += 1
                logger.warning(
                    'merge_queue_store: branch %s already in-flight at recovery; '
                    'coalescing recovered winner %s onto the existing entry',
                    winner_qb.full_name,
                    winner_record.request_id,
                )

            # Attach every loser as a peer waiter so both requesters resolve via
            # the primary→peer future-mirror; register the durable alias when
            # retention is supplied, then drop the loser from the journal.
            for j, (loser_record, _loser_qb) in enumerate(group):
                if j == winner_idx:
                    continue
                loser_req = reconstruct_merge_request(loser_record, config)
                registry.attach(bare_id, WaiterRecord(
                    request_id=loser_req.request_id,
                    future=loser_req.result,
                    source='recovery',
                    submitted_tip=loser_record.snapshot_tip,
                ))
                # Alias onto the ACTUAL in-flight primary (the enqueued winner
                # when acquire succeeded, else the pre-existing entry) so a
                # durable poll on a loser id resolves to the real primary
                # outcome — matching coalesce_or_enqueue_merge_request.
                if retention is not None and primary_request_id:
                    retention.record_alias(
                        loser_req.request_id, primary_request_id,
                    )
                store.remove(loser_record.request_id)
                coalesced += 1

            if divergence_seen:
                logger.warning(
                    'merge_queue_store: recovery collapsed branch %s across a '
                    'DIVERGENT force-push (D2) — enqueued the descendant-most '
                    'snapshot as the single winner and coalesced %d duplicate(s)',
                    winner_qb.full_name,
                    len(group) - 1,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'merge_queue_store: error collapsing branch group %s: %s',
                bare_id,
                exc,
                exc_info=True,
            )

    return {
        'recovered': recovered,
        'dropped': dropped,
        'coalesced': coalesced,
        'requests': recovered_requests,
        'journal_corrupt': store.journal_corrupt,
    }
