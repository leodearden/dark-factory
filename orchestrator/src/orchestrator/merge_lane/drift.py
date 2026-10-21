"""Lever-C drift-check detective (MQ-refactor task γ).

The drift-check runner (runs a local and a remote leg concurrently through
:class:`~orchestrator.verify_runner.DriftDetector`, the local leg in a
throwaway worktree), the remote-runner builder, and the land-hook cadence
gate.  A due check that cannot get both host slots defers, and the gate
re-arms it for the next landing.  The local trust-anchor verifies through the
worker's injected :class:`~orchestrator.merge_lane.ports.VerifyPort`; this
module never imports the worker.

Note: despite both being off-serial-lane detective controls spawned from the
same 'done'-land hook, :func:`_run_drift_check` does NOT call
``_run_cold_shadow_verify`` or ``_run_shadow_compare`` (in
:mod:`orchestrator.merge_lane.shadow`) — drift-check and shadow-compare are
independent sibling detectives, not caller/callee.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.deployed_sha import DeployedSha, resolve_code_checkout
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import MergeRequest
from orchestrator.verify_runner import (
    DriftDetector,
    HostAllocator,
    HostLease,
    LocalRunner,
    RemoteRunner,
    RunnerUnavailableSink,
    VerifyRunner,
    VerifyRunnerPool,
    build_merge_verify_spec,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.merge_lane.ports import VerifyPort
    from orchestrator.merge_lane.worker import SpeculativeMergeWorker

logger = logging.getLogger('orchestrator.merge_lane')

# How long a due drift check waits for both host slots before it defers.  It
# must outlast the landing merge's own host lease, which
# merge_lane/worker.py::SpeculativeMergeWorker._finalize_inflight releases only
# in its finally, after this land hook, refresh_warm_base and the CAS walk.
_DRIFT_HOST_SETTLE_SECS = 600.0
_DRIFT_HOST_POLL_SECS = 5.0


def _build_remote_runners(
    config: OrchestratorConfig,
    cwd: str | Path,
    *,
    deployed_sha: DeployedSha,
    quarantine: set[str] | None = None,
) -> list[RemoteRunner]:
    """Build the list of RemoteRunner instances from operator config (Lever C).

    Returns REMOTES ONLY — the LocalRunner trust anchor is prepended by callers
    since it needs call-specific arguments (worktree path, module configs, etc.).

    Filters out disabled runners (enabled=False) and any runner whose name is in
    the quarantine set (in-memory worker-level quarantine from DriftDetector).
    quarantine=None is treated as an empty set (no quarantine).

    _build_verify_runners passes main_branch=config.git.main_branch so the
    remote host receives a freshness push before the merge-sha transport.

    Each runner gates its verifies on *deployed_sha* (task 6565, the commit
    this process's wire path was imported from) through the runner's
    :meth:`~orchestrator.config.VerifyRunnerConfig.gate_paths`, publishing the
    sha from this process's own code checkout, resolved once here.
    """
    code_checkout = resolve_code_checkout()
    runners = []
    for r in config.enabled_verify_runners:
        if quarantine is not None and r.name in quarantine:
            continue
        gate = r.gate_paths()
        runners.append(RemoteRunner(
            name=r.name,
            ssh_host=r.ssh_host,
            git_remote=r.git_remote,
            cwd=cwd,
            config_path=r.config_path,
            main_branch=config.git.main_branch,
            df_remote_checkout=gate.source,
            gate_checkouts_path=gate.root,
            deployed_sha=deployed_sha,
            code_checkout=code_checkout,
        ))
    return runners


# ---------------------------------------------------------------------------
# Lever-C drift-check cadence persistence (task 2886 fix 1a)
#
# Modelled on merge_lane/shadow.py::ShadowCompareState and its
# _load_shadow_compare_state / _save_shadow_compare_state.  The drift-check
# land counter was a worker-level in-memory int (``worker._drift_land_count``)
# that the ~8h fleet redeploy resets to 0 before it ever reaches
# ``verify_drift_check_every_n_lands`` (default 20) — which is exactly why the
# drift check has NEVER fired in any runs.db.  Persisting the counter to
# ``project_root/data/orchestrator/drift_check_state.json`` makes the cadence
# survive restarts.  The runner quarantine set is deliberately NOT persisted:
# the dedup'd open L1 verify_drift_divergence escalation is the cross-restart
# guard (see merge_queue.py SpeculativeMergeWorker.__init__ RESTART RE-TRUST
# WINDOW note).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftCheckState:
    """Persisted cadence state for the Lever-C drift check.

    Stored as JSON at
    ``config.project_root/data/orchestrator/drift_check_state.json`` so the
    drift-check cadence survives orchestrator restarts (the ~8h fleet redeploy
    otherwise resets the in-memory ``worker._drift_land_count`` before it ever
    reaches ``verify_drift_check_every_n_lands``).

    Fields:
        land_count: Count of successful ('done') lands observed so far.
            Additive across the worker lifetime (never reset or decremented),
            mirroring the in-memory ``worker._drift_land_count`` it supersedes.
        consecutive_deferrals: Consecutive due checks that could not acquire
            both host slots.  Greater than 0 means a check is overdue and
            fires at the next land; reset when a check acquires its slots.
    """

    land_count: int = 0
    consecutive_deferrals: int = 0

    def is_due(self, every_n: int) -> bool:
        """The firing rule: a cadence multiple, or a deferred check still owed."""
        return self.consecutive_deferrals > 0 or self.land_count % every_n == 0

    def after_land(self) -> DriftCheckState:
        return dataclasses.replace(self, land_count=self.land_count + 1)

    def after_deferral(self) -> DriftCheckState:
        return dataclasses.replace(
            self, consecutive_deferrals=self.consecutive_deferrals + 1,
        )

    def after_start(self) -> DriftCheckState:
        return dataclasses.replace(self, consecutive_deferrals=0)


def _load_drift_check_state(path: Path) -> DriftCheckState:
    """Load the drift-check cadence state from a JSON file.

    Fail-safe: returns a default ``DriftCheckState()`` on any error (file not
    found, unreadable, unparseable JSON, or missing/wrong-typed keys) so the
    orchestrator never fails to start due to a corrupt state file.  A file
    written before ``consecutive_deferrals`` existed loads with 0 deferrals.
    Modelled on merge_lane/shadow.py::_load_shadow_compare_state.

    Args:
        path: Path to the JSON state file (typically
            ``config.project_root/data/orchestrator/drift_check_state.json``).

    Returns:
        The persisted state, or ``DriftCheckState(0)`` on any failure.
    """
    try:
        data = json.loads(path.read_text())
        return DriftCheckState(
            land_count=int(data['land_count']),
            consecutive_deferrals=int(data.get('consecutive_deferrals', 0)),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return DriftCheckState()


def _save_drift_check_state(path: Path, state: DriftCheckState) -> None:
    """Persist the drift-check cadence state to a JSON file.

    Creates parent directories as needed.  Modelled on
    merge_lane/shadow.py::_save_shadow_compare_state.

    Args:
        path: Destination path for the JSON state file.
        state: The :class:`DriftCheckState` to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(state)))


def _record_drift_deferral(path: Path | None) -> int:
    """Persist one more consecutive deferral; the new count (1 with no state file)."""
    if path is None:
        return 1
    state = _load_drift_check_state(path).after_deferral()
    _save_drift_check_state(path, state)
    return state.consecutive_deferrals


def _record_drift_start(path: Path | None) -> int:
    """Clear the deferral count once a check holds its slots; the count it cleared."""
    if path is None:
        return 0
    state = _load_drift_check_state(path)
    if state.consecutive_deferrals > 0:
        _save_drift_check_state(path, state.after_start())
    return state.consecutive_deferrals


class _HeldHostLeases:
    """The host-slot leases one drift check still holds; each is released once.

    ``HostAllocator.release`` keys on slot state, not lease identity, so a
    second release of a lease whose slot was since re-acquired would free the
    new occupant.  A lease is forgotten before it is released.
    """

    def __init__(self, allocator: HostAllocator) -> None:
        self._allocator = allocator
        self._held: list[HostLease] = []

    def hold(self, *leases: HostLease | None) -> None:
        self._held.extend(lease for lease in leases if lease is not None)

    def seat(self, lease: HostLease, runner: VerifyRunner) -> None:
        """Put *runner* on held *lease*'s slot, so the end of its leg releases it."""
        self._held[self._held.index(lease)] = dataclasses.replace(lease, runner=runner)

    async def release_leg(self, runner: VerifyRunner) -> None:
        """Release the lease whose runner just ended its leg; idempotent."""
        lease = next((held for held in self._held if held.runner is runner), None)
        if lease is not None:
            self._held.remove(lease)
            await self._allocator.release(lease)

    async def release_remaining(self) -> None:
        while self._held:
            await self._allocator.release(self._held.pop())


async def _try_acquire_drift_hosts(
    allocator: HostAllocator,
) -> tuple[HostLease, HostLease] | None:
    """One all-or-nothing attempt: (local, remote) leases, or None holding neither.

    The local lease claims the slot with no runner yet: the local leg's runner
    needs its worktree, which is built only once both slots are held (see
    :meth:`_HeldHostLeases.seat`), so a deferral costs no git work.
    """
    remote = allocator.acquire_remote()
    if remote is None:
        return None
    local = allocator.acquire_local(lambda: None)
    if local is None:
        await allocator.release(remote)
        return None
    return local, remote


async def _acquire_drift_hosts(
    allocator: HostAllocator,
    *,
    settle_secs: float,
    poll_secs: float,
) -> tuple[HostLease, HostLease] | None:
    """Both host slots, retried for *settle_secs*; None if they never came free.

    A waiting check holds no slot.  ``settle_secs=0`` makes exactly one attempt.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settle_secs
    while True:
        leases = await _try_acquire_drift_hosts(allocator)
        if leases is not None or loop.time() >= deadline:
            return leases
        await asyncio.sleep(poll_secs)


def _report_drift_deferral(
    event_store: EventStore | None,
    req: MergeRequest,
    merge_commit: str,
    allocator: HostAllocator,
    consecutive: int,
) -> None:
    """Log the deferral (INFO for the first of a run, DEBUG after) and emit its event."""
    hosts = allocator.host_states()
    logger.log(
        logging.INFO if consecutive == 1 else logging.DEBUG,
        'Drift check deferred for task %s / commit %s: host slots stayed busy '
        '(%d consecutive deferral(s)); re-armed for the next landing. Hosts: %s',
        req.task_id, merge_commit, consecutive, hosts,
    )
    if event_store is not None:
        event_store.emit(
            EventType.drift_check_deferred,
            task_id=req.task_id,
            data={
                'merge_sha': merge_commit,
                'consecutive_deferrals': consecutive,
                'hosts': hosts,
            },
        )


def _report_drift_start(req: MergeRequest, merge_commit: str, cleared: int) -> None:
    if cleared:
        logger.info(
            'Drift check for task %s / commit %s running after %d deferred landing(s)',
            req.task_id, merge_commit, cleared,
        )


async def _run_drift_check(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    escalation_queue: Any,
    event_store: EventStore | None,
    quarantine_set: set[str],
    *,
    verifier: VerifyPort,
    allocator: HostAllocator,
    on_unavailable: RunnerUnavailableSink | None = None,
    state_path: Path | None = None,
    host_settle_secs: float = _DRIFT_HOST_SETTLE_SECS,
    host_poll_secs: float = _DRIFT_HOST_POLL_SECS,
) -> None:
    """Drift detective control: run DriftDetector.check in a throwaway worktree.

    Mirrors ``_run_cold_shadow_verify`` / ``_run_shadow_compare`` as an
    off-serial-lane detective control (spawned via :func:`_maybe_run_drift_check`).
    Waits up to *host_settle_secs* for both host slots, then creates a
    throwaway verify worktree for the local leg, builds a 2-host pool (local
    trust-anchor + eligible remote), runs ``DriftDetector.check_env_safely`` and
    then ``check``, and propagates any quarantined remote name into
    *quarantine_set* (the worker-level shared set consulted by
    ``_run_post_merge_verify``).

    Each host slot is released the moment its own leg ends; the ``finally``
    releases only what is still held, then ``cleanup_merge_worktree``.  If the
    slots never come free the check DEFERS: it builds no worktree, runs no leg,
    records the deferral in *state_path* and emits ``drift_check_deferred``.
    Exceptions are caught and logged so this detective control never crashes
    the worker.

    The verify body runs under ``merge_verify_lease(lane_dir=wt)`` so a
    concurrent periodic worktree reap (task 3018) cannot delete the checkout
    mid-verify; the lease is released before the ``finally`` cleanup — see the
    inline comment for why that scope is load-bearing.  It is distinct from,
    and orthogonal to, the allocator's host-slot leases.

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed (config, task_files, …).
        merge_commit: The just-landed merge commit SHA.
        escalation_queue: Live escalation queue (None-safe: passed to DriftDetector).
        event_store: Optional event store (None-safe: passed to DriftDetector).
        quarantine_set: Worker-level mutable set; quarantined remote names are
            added here so subsequent dispatches skip the diverged remote.
        allocator: Worker-lifetime :class:`~orchestrator.verify_runner.HostAllocator`
            (β decision 5), from which both host slots are acquired.
        verifier: The worker's :class:`~orchestrator.merge_lane.ports.VerifyPort`;
            the local trust-anchor runs its scoped verify and unscoped type-checks.
        on_unavailable: Where the remote leg's RunnerUnavailable is reported
            (the worker's bench); the check itself stays INCONCLUSIVE.
        state_path: The persisted :class:`DriftCheckState`, where a deferral is
            recorded and cleared; None counts nothing.
        host_settle_secs: How long to wait for both host slots before deferring.
        host_poll_secs: How often to retry the slots while waiting.
    """
    # wt is initialised before the try so the finally guard (`if wt is not None`)
    # is safe even when create_ephemeral_verify_worktree itself raises.  Moving the
    # creation inside the try ensures the docstring contract ("Exceptions are caught
    # and logged") holds for all failure modes — disk-full / git errors included.
    wt = None
    held = _HeldHostLeases(allocator)
    try:
        # β decision 5: both hosts come through the allocator so slot
        # accounting is respected (≤1 verify-merge per host at any time).
        leases = await _acquire_drift_hosts(
            allocator, settle_secs=host_settle_secs, poll_secs=host_poll_secs,
        )
        if leases is None:
            _report_drift_deferral(
                event_store, req, merge_commit, allocator,
                _record_drift_deferral(state_path),
            )
            return
        held.hold(*leases)
        _report_drift_start(req, merge_commit, _record_drift_start(state_path))
        local_slot, remote_lease = leases
        wt = await git_ops.create_ephemeral_verify_worktree(merge_commit)
        # Hold THIS throwaway lane's flock for the duration of the drift
        # verify (task 3018).  Three things to know:
        #
        # (i) WHY IT IS NEEDED.  Task 3018 promoted
        #     `reap_orphaned_merge_worktrees` from a startup-only sweep to a
        #     steady-state one fired from the merge worker's heartbeat, which
        #     turned RESOURCE_AUDIT_WORKTREE_GRACE_SECS from a *detection*
        #     threshold into a *destruction* deadline.  This throwaway tree is
        #     neither registered in `_owned_merge_worktrees` (so the reap's
        #     owned-ledger skip misses it) nor touched by
        #     `_touch_owned_merge_worktrees` (so its measured age is real
        #     elapsed time) — the lane flock consulted by
        #     `remove_merge_worktree_guarded` is its ONLY liveness protection.
        #     Holding it makes that protection independent of how long the
        #     drift verify runs, rather than resting on an age heuristic: a
        #     concurrent sweep gets 'skipped_lease_held' instead of deleting
        #     the checkout out from under a running verify.
        #
        # (ii) PRECEDENT.  Passing `lane_dir=` an ephemeral `_merge-*`
        #     worktree (rather than the persistent lane) is established — the
        #     DF-2822 per-land REMOTE-green cross-check does exactly this
        #     (merge_queue.py, `lane_dir=merge_wt`).
        #
        # (iii) CONTENTION IS NOT A PRACTICAL FAILURE MODE here, so no
        #     MergeVerifyLeaseContended fail-safe branch is warranted: the
        #     `_merge-<uuid>` path was just minted by
        #     create_ephemeral_verify_worktree and is uncontended by
        #     construction — nobody else knows it.  (Were one ever raised it
        #     would land in the except-Exception fail-open below, the same
        #     detective-control degradation as any other drift-check error.)
        #
        # (iv) FLOCK ONLY, NO GLOBAL RENDEZVOUS (task 4189).  Because `wt` is
        #     ephemeral, this lease takes the lane flock and does NOT record
        #     the single fixed-key global holder-pgid — see
        #     `merge_verify_lease` for the full argument (it neither defers
        #     `_run_warm_lane_gc_reclaim` over a lane reclaim would never
        #     reset, nor strips a concurrent persistent-lane verify's live
        #     rendezvous on exit).  The `skipped_lease_held` protection in (i)
        #     is unaffected: it comes from the FLOCK, which is untouched.
        #
        # Entered only AFTER the assignment above succeeds — `wt` is still
        # None if creation raises, and the finally's `if wt is not None` guard
        # depends on that.  SCOPE IS LOAD-BEARING: the lease wraps only the
        # verify body and closes before the finally.  If cleanup ran while we
        # still held the lease, `remove_merge_worktree_guarded`'s NON-BLOCKING
        # acquire would fail against OURSELVES, return 'skipped_lease_held',
        # and leak the very tree the finally exists to remove.  The host-SLOT
        # leases (`held`) are orthogonal to this lane flock: each is released
        # as its own leg ends, with the finally as the safety net.
        async with git_ops.merge_verify_lease(lane_dir=wt):
            # FULL-GATE drift spec (task 2886 fix 1b, PRD §8δ): the drift check
            # re-dispatches this spec to BOTH the local trust-anchor and the
            # eligible remote and alarms on divergence.  Re-dispatching the SAME
            # scoped/no-source spec that produced a trivial pass would trivially
            # pass on both hosts and structurally cannot catch the trivial-pass
            # divergence class.  Forcing task_files=None runs the complete
            # workspace gate (verify.py task_files=None path) on both hosts so a
            # genuine divergence is observable.  This deliberately does NOT
            # derive/scope task_files (unlike _run_post_merge_verify's serial-lane
            # dispatching-host derivation) — the derivation gate is intentionally
            # dropped here.  task_files_tuple stays None so BOTH the spec and the
            # LocalRunner below run the whole suite.
            task_files_tuple = None
            spec = build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)
            local_runner = LocalRunner(
                wt, req.config, req.module_configs, task_files_tuple,
                run_scoped=verifier.run_scoped,
                run_unscoped=verifier.run_unscoped_typechecks,
                task_id=req.task_id,
            )
            held.seat(local_slot, local_runner)
            pool = VerifyRunnerPool(
                [local_runner, remote_lease.runner],
                event_store=event_store,
                task_id=req.task_id,
                on_unavailable=on_unavailable,
            )

            # Cadence is already enforced upstream in _maybe_run_drift_check
            # (DriftCheckState.is_due).  DriftDetector.should_sample is
            # never called on this path, so omitting every_n_lands here keeps a
            # single source of truth and avoids a dead duplicate cadence value.
            detector = DriftDetector(
                pool,
                event_store=event_store,
                escalation_queue=escalation_queue,
                task_id=req.task_id,
            )
            await detector.check_env_safely(req.config.verify_env_probes)
            # Capture the eligible remote BEFORE check() so we can propagate its
            # name even after the pool quarantines it (eligible_remote() returns
            # None after quarantine).
            eligible_before = pool.eligible_remote()
            result = await detector.check(
                merge_commit, spec, on_leg_done=held.release_leg,
            )
            if result.quarantined and eligible_before is not None:
                quarantine_set.add(eligible_before.name)
    except Exception:
        logger.warning(
            'Drift check failed for task %s / commit %s; ignoring (detective control)',
            req.task_id, merge_commit,
            exc_info=True,
        )
    finally:
        await held.release_remaining()
        if wt is not None:
            await git_ops.cleanup_merge_worktree(wt)


def _advance_land_count(worker: SpeculativeMergeWorker) -> DriftCheckState:
    """Count this landing, persisted when the worker has a state path.

    The persisted DriftCheckState (task 2886 fix 1a) survives the ~8h fleet
    redeploy that reset the in-memory counter before it ever reached every_n.
    A bare-harness worker (no ``_drift_state_path``) counts in memory only, so
    it cannot carry a deferral.  ``worker._drift_land_count`` is kept in
    lockstep either way.
    """
    path = worker._drift_state_path
    if path is None:
        state = DriftCheckState(land_count=worker._drift_land_count).after_land()
    else:
        state = _load_drift_check_state(path).after_land()
        _save_drift_check_state(path, state)
    worker._drift_land_count = state.land_count
    return state


def _spawn_drift_task(
    worker: SpeculativeMergeWorker, coro: Coroutine[Any, Any, None],
) -> asyncio.Task[None] | None:
    """Schedule *coro* with a strong reference held until it finishes."""
    task = asyncio.create_task(coro)
    if not isinstance(task, asyncio.Task):
        # create_task was intercepted (e.g. by a test mock) and did not
        # schedule coro.  Close it here to prevent
        # "coroutine was never awaited" RuntimeWarning from leaking into
        # unrelated tests (pyproject.toml converts those warnings to errors).
        coro.close()
        return None
    # asyncio keeps only a weak ref to Tasks, so without this the drift
    # detective could be GC'd mid-run.  Mirrors _shadow_compare_tasks.
    worker._drift_check_tasks.add(task)
    task.add_done_callback(lambda t: worker._drift_check_tasks.discard(t))
    return task


async def _maybe_run_drift_check(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    *,
    verifier: VerifyPort,
    host_settle_secs: float = _DRIFT_HOST_SETTLE_SECS,
) -> asyncio.Task[None] | None:
    """Cadence gate + off-serial-lane spawn for the Lever C drift detective.

    Called from merge_lane/worker.py::SpeculativeMergeWorker._finalize_inflight
    on every successful ('done') land, immediately after
    ``_maybe_schedule_shadow_compare``.  Returns immediately: the spawned
    drift-check Task when one is due, otherwise None.

    Firing rule: :meth:`DriftCheckState.is_due` — every
    ``config.verify_drift_check_every_n_lands`` lands, and at every land after a
    check deferred for want of host slots, until one runs.  A due land while a
    drift check is already in flight (running, or waiting for its slots)
    spawns nothing: that check serves it.  No-op when
    ``config.enabled_verify_runners`` is empty (Lever C off) or when
    ``verify_drift_check_every_n_lands`` is 0 or negative (guards against
    ``ZeroDivisionError`` for a mocked/hand-built config that bypasses
    ``OrchestratorConfig``'s ``ge=1`` field validation).

    Args:
        worker: Live :class:`SpeculativeMergeWorker` instance.
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed.
        merge_commit: The just-landed merge commit SHA.
        verifier: The worker's :class:`~orchestrator.merge_lane.ports.VerifyPort`,
            handed on to :func:`_run_drift_check`.
        host_settle_secs: How long the spawned check waits for both host slots.
    """
    if not req.config.enabled_verify_runners:
        return None
    every_n = req.config.verify_drift_check_every_n_lands
    if every_n <= 0:
        # OrchestratorConfig enforces ge=1 at construction, but req.config is
        # frequently a MagicMock/hand-built stand-in in tests (and could in
        # principle be mutated post-construction) — guard here too so a
        # misconfigured/mocked 0-or-negative value degrades to "disabled"
        # instead of raising ZeroDivisionError on every land.  Mirrors the
        # every_n > 0 guard in _safety_valve_due (merge_liveness.py).
        return None

    if not _advance_land_count(worker).is_due(every_n):
        return None
    if worker._drift_check_tasks:
        logger.debug(
            'Drift check due at commit %s is served by the one already in flight',
            merge_commit,
        )
        return None
    return _spawn_drift_task(worker, _run_drift_check(
        git_ops, req, merge_commit,
        worker._escalation_queue, worker._event_store,
        worker._runner_quarantine,
        verifier=verifier,
        allocator=worker._ensure_host_allocator(req.config),
        on_unavailable=worker.bench_unavailable_runner,
        state_path=worker._drift_state_path,
        host_settle_secs=host_settle_secs,
    ))
