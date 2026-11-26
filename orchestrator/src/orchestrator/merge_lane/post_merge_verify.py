"""The post-merge verify: run one merge's verify and decide whether it may land.

:func:`run_post_merge_verify` binds its call once into a
:class:`_PostMergeVerify` and runs its phases in order: the merge-request
module boundary, the pre-verify disk guard, the merge gate decision (narrow
or full, docs/narrow-merge-gate.md), task-file derivation and the wire spec,
the runner pool, dispatch with the ENOSPC and infra-transient retry
budgets, flake-observation recording, and then exactly one of the land-time
cross-check of a remote green
(``orchestrator/src/orchestrator/merge_lane/verify_cross_check.py``), the
red-verify outcome
(``orchestrator/src/orchestrator/merge_lane/verify_failure_outcome.py``), or
the pass tail (the known-red gate for a verdict that is no whole-tree
evidence, and baseline seeding).
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.event_store import EventStore, EventType
from orchestrator.flake_recorder import record_merge_flake_suppression
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.disposition import MergeFailureDisposition
from orchestrator.merge_lane.history_leg import HistoryLegRunner
from orchestrator.merge_lane.main_health_probe import MainHealthProbeHandles
from orchestrator.merge_lane.ports import VerifyPort
from orchestrator.merge_lane.retry_verify_env import (
    assemble_retry_verify_env,
    build_attempt0_payload,
)
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest
from orchestrator.merge_lane.verify_cross_check import CrossCheck, cross_check_remote_green
from orchestrator.merge_lane.verify_failure_outcome import (
    DryRunInvestigationHandles,
    RedVerifyContext,
    red_verify_outcome,
    verify_hit_enospc,
)
from orchestrator.merge_verify_facts import MergeVerifyDispatchFacts
from orchestrator.narrow_gate_plan import NarrowGateDecision, executed_commands
from orchestrator.verify import (
    VerifyResult,
    _derive_task_files_from_git,
    known_failing_ids_on_main,
    narrow_merge_gate_decision,
    seed_main_baseline,
)
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_plan import effective_merge_module_configs
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
    VerifyRunner,
    VerifyRunnerPool,
    build_merge_verify_spec,
)

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger('orchestrator.merge_lane')


TRIVIAL_PASS_MAIN_RED_REASON_PREFIX = (
    'Trivial-pass withheld: config-only merge would advance main over a '
    'known-red state'
)
"""Prefix of the ``MergeOutcome.reason`` string emitted when a config-only
merge that short-circuited to :func:`verify._trivial_pass` (no ``.py``/``.rs``
in the diff — ``VerifyResult.trivial is True``) is refused advancement over a
main tip whose cached failing-id baseline is known-red (task 2823, gate-hole
#3/3 of the reify 2026-07-19 red-main incident).

A trivial pass ran NO test suite, so it is no evidence the red cleared; letting
it CAS-advance main would persist the red. The gate keys on
``VerifyResult.is_whole_tree_evidence``, so a narrow gate's pass (task 5411) is
withheld the same way and its reason names the resolution after this prefix. A
pass of the full merge suite on the merged tree legitimately heals a red main
and is never blocked.
It is a CACHE-ONLY peek (:func:`verify.known_failing_ids_on_main`) that
never triggers a probe on the critical path (G4, task 2564) and fails OPEN on a
cold/unknown baseline, matching the task's "known-red" scope. Paired with
``MergeFailureDisposition.MAIN_RED``: the break is pre-existing, not this task's
fault, so the blocked task re-dispatches and passes once main is green again.

Caveat (task 2823 amendment): that self-heal assumes some OTHER, code-touching
merge greens main first. A config-only change that is *itself* the fix for the
red main is deliberately NOT trusted here — a trivial pass runs no suite, so it
can never prove the red cleared, and it will oscillate re-dispatch/block (loud,
via MAIN_RED) until a non-trivial merge or an operator greens main. The gate
logs a distinct WITHHELD warning so this edge is recognisable rather than read
as routine main-red churn."""


def merge_boundary_module_configs(
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
) -> list[ModuleConfig]:
    """The effective merge module set for a merge REQUEST — γ's widening, with
    the zero-module task deliberately left exactly where it already was.

    A thin wrapper over :func:`verify_plan.effective_merge_module_configs`
    (the ONE implementation of "which modules does a merge-role verify
    cover?", INV-5) carrying the single policy that belongs to the merge
    boundary rather than to the breadth knob: an EMPTY passed set is never
    widened.

    WHY (reviewer amendment, task 3787 γ). A task with no assigned modules —
    ``WorkflowRunner._resolve_module_configs`` returns ``[]`` whenever
    ``self.modules`` is empty, i.e. docs-only and root-file tasks — is
    verified today, even at breadth='full', through the GLOBAL fallback on
    BOTH legs: ``build_merge_verify_spec`` sources ``global_verify_command``
    precisely because the projected set is empty (INV-1, task 2883), and
    ``run_scoped_verification``'s own widening sits INSIDE its ``if
    module_configs:`` guard, so an empty set falls through to
    ``_derive_fallback_runs``. Widening ``[]`` to the whole registry here
    would silently re-point both of those legs at a per-module full-suite
    fan-out (and ``_run_unscoped_typechecks`` at every registered module) — a
    materially different, untested gate for the lightest merges, on a live
    config that sets breadth='full' with 9 modules registered. γ's mandate is
    to make the suppression gate map node-ids against the set that ACTUALLY
    RAN; it is not to change what runs.

    So this guard MIRRORS ``run_scoped_verification``'s ``if module_configs:``
    guard: the boundary widens exactly where that site would have, and is the
    identity otherwise. Whether breadth='full' OUGHT to cover the whole
    registry for a zero-module task is a real question — but a pre-existing
    one, and answering it is not γ's job.

    The shared helper's own contract is deliberately NOT changed: its other
    caller, ``run_scoped_verification``'s force_workspace branch (verify.py
    site A), widened an empty passed set before γ and must keep doing so.
    """
    if not module_configs:
        return module_configs
    return effective_merge_module_configs(config, module_configs)


def placeholder_local_runner(
    merge_wt: Path, req: MergeRequest, verifier: VerifyPort,
) -> LocalRunner:
    """The LocalRunner a local host slot is acquired with.

    Its modules are the merge boundary's (PRD §8.2), through the SAME wrapper
    as the runner that actually drives the verify, so the two can never
    disagree about which modules a merge covers.
    """
    return LocalRunner(
        merge_wt,
        req.config,
        merge_boundary_module_configs(req.config, req.module_configs),
        None,   # task_files — derived inside run_post_merge_verify
        run_scoped=verifier.run_scoped,
        run_unscoped=verifier.run_unscoped_typechecks,
        task_id=req.task_id,
    )


async def run_post_merge_verify(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    *,
    timeouts: dict[str, int],
    enospc_retries: dict[str, int],
    max_timeouts: int,
    max_enospc: int,
    verifier: VerifyPort,
    max_narrowed: int = 1,
    narrowed_retries: dict[str, int] | None = None,
    event_store: EventStore | None = None,
    merge_sha: str = '',
    on_result: Callable[[VerifyResult], None] | None = None,
    shadow_baseline_sink: dict[str, str] | None = None,
    quarantine_divergent: Callable[[str], None] | None = None,
    keep_worktrees: Collection[Path] | None = None,
    runner: VerifyRunner | None = None,
    escalation_queue: Any = None,
    halt_hook: Callable[[str], None] | None = None,
    dry_run_handles: DryRunInvestigationHandles | None = None,
    main_health_probe_handles: MainHealthProbeHandles | None = None,
    depth: int | None = None,
    speculative: bool | None = None,
    chain_items: int = 1,
    chain_build_ms: int | None = None,
    merge_base_sha: str | None = None,
    main_sha: str | None = None,
    history_leg_command: str | None = None,
    dispatch_facts: MergeVerifyDispatchFacts | None = None,
    narrow_gate_base: str | None = None,
) -> MergeOutcome | None:
    """Run post-merge verification for a single task.

    Returns ``None`` when verification passes; returns a ``MergeOutcome``
    (and cleans up *merge_wt*) when it fails via a controlled path (disk
    guard, verify-not-passed).  Does **not** contain a ``try/except`` — any
    exception from the verify propagates to the caller;
    ``SpeculativeMergeWorker`` wraps the call in a ``try/except`` that maps a
    raised verify to a ``'Verification error: ...'`` outcome.

    Args:
        verifier: The ``VerifyPort`` every verify of this call runs through.
            Required: the worker passes its own (``PRODUCTION_VERIFIER`` in
            production), so a verifier's cross-check and dry-run always run
            through that same verifier.
        max_narrowed: Budget for the classified-infra-transient retry loop
            (task 2835) when this call's failed-only retry was actually
            NARROWED — the D2 producer built a payload from attempt-0's own
            result and merged ``retry_env``.  A non-narrowed retry (flag off,
            a payload that could not be built or corroborated, or a remote
            dispatch) always uses ``max_enospc`` instead.
        narrowed_retries: Per-task counter dict for the narrowed budget
            above, mirroring ``enospc_retries`` but DECOUPLED from it (an
            earlier ENOSPC event never starves a narrowed retry, and vice
            versa).  ``None`` (default) is normalized to a fresh, throwaway
            ``{}`` local to this call.
        on_result: Optional callback invoked with the final
            :class:`~orchestrator.verify.VerifyResult` BEFORE the pass/fail
            branch.  Used by :class:`SpeculativeMergeWorker` to capture warm
            per-test results for PRD §10 invariant 6(b) shadow compare.
        shadow_baseline_sink: Optional mutable out-param (PRD
            verify-retry-failed-only D4, §5.4).  When supplied AND this call
            actually narrowed the retry, the attempt-0 ``debug_verdicts`` ∪
            ``release_verdicts`` map is copied into it, so the worker can union
            it with the PARTIAL narrowed-retry warm output before storing the
            warm shadow baseline (no phantom ``only_cold`` divergence).
        quarantine_divergent: The divergence-quarantine sink for the land-time
            cross-check (task 2822): on a remote PASS / local trust-anchor FAIL
            split it is handed the remote's name — the lane passes
            ``host_quarantine.py::HostQuarantine.quarantine_divergent``.
            ``None`` records no host name; the cross-check still fails closed.
        keep_worktrees: Additional worktrees to protect during any disk-pressure
            prune triggered inside this verify run (pre-verify guard and ENOSPC
            retry); *merge_wt* is always protected.  The worker passes its
            owned speculative worktrees (PRD §5 decision 5).

            **Snapshot timing:** The caller's keep-set is a one-shot snapshot
            taken at dispatch.  Worktrees registered into the ledger *after*
            this snapshot is taken are not protected by prunes triggered inside
            this call — they rely on the heartbeat mtime / grace-period
            mechanism to avoid premature removal in the residual window between
            snapshot capture and prune execution.  The gate re-verify passes
            the live ledger instead (task 6155): dispatch continues while it
            runs, so each prune reads the ledger as it stands at that moment.
        runner: The verify host.  ``None`` (default) verifies on a LOCAL-only
            pool — the recovery paths stay on the trust anchor and out of slot
            accounting.  A remote runner verifies on ``VerifyRunnerPool([runner])``
            with NO LocalRunner, so a ``RunnerUnavailable`` propagates to the
            caller, which owns the bench-and-redispatch decision.
        escalation_queue: Live escalation queue or ``None`` (default).  Files
            the born-at-L2 flock-worktree-contention escalation (task 2307 β),
            the cross-check divergence escalation and the flake recorder's
            INV-4 storm escalation.
        halt_hook: Optional synchronous all-lane merge-queue halt (task 2886
            fix 3, PRD §3.4 / decision 4).  Invoked with a human-readable
            reason the instant a GENUINE concluded cross-check divergence
            (remote PASS / local trust-anchor FAIL with a non-infra-transient
            local category) is detected — BEFORE the local FAIL is adopted —
            so no FURTHER adoption can occur before a human looks.  An
            infra-transient local FAIL withholds the land and quarantines the
            remote but does NOT halt (task 2886 amendment).  Only the worker's
            lane verify and train verify pass the worker's ``operator_halt``.
        dry_run_handles: Opaque bundle of scheduler/mcp/usage_gate/cost_store
            (task η).  ``None`` (default) makes the dry-run spawn a no-op.
            Only ``VerifyDispatcher.post_merge_verify`` passes a
            live bundle, at which point a MERGE_VERIFY_RED outcome fires a
            fire-and-forget dry-run investigation.
        main_health_probe_handles: Opaque bundle carrying the worker's
            ``background_tasks`` set (task 2564, reify 5067 merge-slot-stall
            fix).  ``None`` (default) runs the main-health probe
            SYNCHRONOUSLY.  A live bundle (only the worker's lane verify)
            makes a main-health-eligible failure SKIP the synchronous probe,
            return the provisional task-fault outcome immediately, and spawn
            the classification as a detached background task — so the
            caller's verify task (and the merge slot / host lease it holds) is
            freed within seconds of the verdict.
        depth: Verify-frontier stack height (task 2340, ε=1890) at dispatch
            time; threaded straight into ``pool.dispatch`` for the
            ``merge_verify`` event.
        speculative: Mirrors ``item.speculative``; threaded straight into
            ``pool.dispatch`` alongside *depth*.
        chain_items: Count, in CHAIN-ITEM units, of the items contained in the
            tree this verify exercises (task 3185, PRD γ) — the dispatching
            item is chain item #1, and each chained successor actually BUILT
            adds one.  Frontier-INDEPENDENT, so ``chain_items >= 2`` is a sound
            deep-verify discriminator.  Threaded into every dispatch, retries
            included — a retry re-verifies the same tree.
        chain_build_ms: Wall-clock milliseconds the deep chain build that
            produced this verify's tree cost, or ``None`` when no chain was
            built.  Forwarded verbatim to every dispatch so a retry re-reports
            the SAME build cost: the build happened once.
        merge_base_sha: Dispatch-time merge-base SHA (task 2383 β, 2357:
            caller-supplied, never re-derived here).  Paired with *main_sha*
            to classify a non-preexisting task-fault failure's merge-skew
            disposition.  ``None`` (default) skips classification entirely.
        main_sha: Dispatch-time main SHA (``item.base_sha``) paired with
            *merge_base_sha*; both must be non-``None`` for classification.
        history_leg_command: The project's declared same-tree history leg
            (task 7248), passed only by ``gates.py::_reverify_rebased_tree``.
            When set, the LOCAL pool runs only this command, through
            ``history_leg.py::HistoryLegRunner``, instead of the full gate.
        dispatch_facts: The ``runner_reason``/``base_relation`` attempt 0's
            ``merge_verify`` row carries; every retry reports it as
            ``retry_held_host`` (``orchestrator/merge_verify_facts.py``).
        narrow_gate_base: The main sha the merge was built on.  When set and
            ``narrow_merge_gate`` is enabled, the merge's real tree delta
            decides its gate (``verify.py::narrow_merge_gate_decision``): a
            narrow one runs on the local host whatever *runner* is, and the
            decision is recorded as a ``merge_gate_resolution`` event.
            ``None`` (default) keeps the full gate and records nothing.
    """
    # THE merge-request boundary resolution (flake-ledger PRD §8.2, task 3787
    # γ), resolved ONCE here so the local runner, the wire spec (and hence the
    # remote's reconstruction of it) and the merge-flake suppression gate all
    # reason against the SAME widened set by construction.  req.module_configs
    # is NOT mutated, and the main-health probe keeps the task-scoped set.
    effective_module_configs = merge_boundary_module_configs(
        req.config, req.module_configs,
    )
    return await _PostMergeVerify(
        git_ops=git_ops, req=req, merge_wt=merge_wt,
        effective_module_configs=effective_module_configs,
        timeouts=timeouts, enospc_retries=enospc_retries,
        max_timeouts=max_timeouts, max_enospc=max_enospc, verifier=verifier,
        max_narrowed=max_narrowed,
        narrowed_retries={} if narrowed_retries is None else narrowed_retries,
        event_store=event_store, merge_sha=merge_sha, on_result=on_result,
        shadow_baseline_sink=shadow_baseline_sink,
        quarantine_divergent=quarantine_divergent, keep_worktrees=keep_worktrees,
        runner=runner, escalation_queue=escalation_queue, halt_hook=halt_hook,
        dry_run_handles=dry_run_handles,
        main_health_probe_handles=main_health_probe_handles,
        depth=depth, speculative=speculative, chain_items=chain_items,
        chain_build_ms=chain_build_ms, merge_base_sha=merge_base_sha,
        main_sha=main_sha, history_leg_command=history_leg_command,
        dispatch_facts=dispatch_facts, narrow_gate_base=narrow_gate_base,
    ).run()


@dataclasses.dataclass(frozen=True)
class _PostMergeVerify:
    """One post-merge verify call, bound once; :meth:`run` runs its phases in order."""

    git_ops: GitOps
    req: MergeRequest
    merge_wt: Path
    effective_module_configs: list[ModuleConfig]
    timeouts: dict[str, int]
    enospc_retries: dict[str, int]
    max_timeouts: int
    max_enospc: int
    verifier: VerifyPort
    max_narrowed: int
    narrowed_retries: dict[str, int]
    event_store: EventStore | None
    merge_sha: str
    on_result: Callable[[VerifyResult], None] | None
    shadow_baseline_sink: dict[str, str] | None
    quarantine_divergent: Callable[[str], None] | None
    keep_worktrees: Collection[Path] | None
    runner: VerifyRunner | None
    escalation_queue: Any
    halt_hook: Callable[[str], None] | None
    dry_run_handles: DryRunInvestigationHandles | None
    main_health_probe_handles: MainHealthProbeHandles | None
    depth: int | None
    speculative: bool | None
    chain_items: int
    chain_build_ms: int | None
    merge_base_sha: str | None
    main_sha: str | None
    history_leg_command: str | None
    dispatch_facts: MergeVerifyDispatchFacts | None
    narrow_gate_base: str | None
    gate_decision: NarrowGateDecision | None = None

    @property
    def task_id(self) -> str:
        return self.req.task_id

    @property
    def archive_root(self) -> Path:
        return self.req.config.project_root / 'data' / 'verify-logs'

    async def run(self) -> MergeOutcome | None:
        # Pre-verify disk guard: if free space is low, prune stale merge
        # worktrees; if still low, skip the build and escalate as transient
        # infra rather than entering a doomed multi-minute ENOSPC build.
        disk = await self.verifier.ensure_disk_space(
            self.git_ops, self.merge_wt,
            self.req.config.merge_verify_min_free_disk_bytes, self.task_id,
            keep_worktrees=self.keep_worktrees,
        )
        if disk.reason is not None:
            await self.git_ops.cleanup_merge_worktree(self.merge_wt)
            return MergeOutcome('blocked', reason=disk.reason, verify_skipped=True)
        return await self._under_gate(await self._merge_gate_decision())._verify()

    async def _merge_gate_decision(self) -> NarrowGateDecision | None:
        """This merge's gate, decided once; None when the lane asked for none or the gate is off.

        A history-leg re-verify runs only its declared leg, so it is never narrowed.
        """
        if (
            self.narrow_gate_base is None
            or self.history_leg_command is not None
            or not self.req.config.narrow_merge_gate.enabled
        ):
            return None
        return await narrow_merge_gate_decision(
            self.merge_wt, self.req.config, base_sha=self.narrow_gate_base, merge_sha=self.merge_sha,
        )

    def _under_gate(self, decision: NarrowGateDecision | None) -> _PostMergeVerify:
        """This verify under *decision*: a narrow gate verifies on the local host, never an injected remote.

        The wire spec carries no narrow plan, so a remote could only run the full gate; and every
        host-keyed step after dispatch (the lane lease, the retry producer, the cross-check, the red
        outcome's origin) then sees the verify where it actually ran.
        """
        runner = None if _narrow(decision) is not None else self.runner
        return dataclasses.replace(self, gate_decision=decision, runner=runner)

    async def _verify(self) -> MergeOutcome | None:
        task_files = await self._task_files()
        # Projected from the SAME effective set the LocalRunner receives
        # (INV-5), so the remote's reconstruction of it yields the identical
        # module set.  build_merge_verify_spec sources global_verify_command
        # ONLY when the projected set is empty, and the boundary never widens
        # an empty set, so a zero-module task keeps its global fallback
        # (test_merge_boundary_effective_module_configs.py::
        # TestMergeBoundaryLeavesTheZeroModuleTaskAlone).
        spec = build_merge_verify_spec(self.req.config, self.effective_module_configs, task_files)
        attempts = await self._dispatch_attempts(self._pool(task_files), spec)
        verify = attempts[-1]
        await self._record_observations(attempts)
        # Called with the FINAL VerifyResult (after any retry) so the warm
        # per-test results are always the last-observed verify for this commit.
        # A verdict that is no whole-tree evidence leaves the shadow map-less (INV-5).
        if self.on_result is not None and verify.is_whole_tree_evidence:
            self.on_result(verify)
        remote = self._cross_check_remote(verify)
        if remote is not None:
            verify = await cross_check_remote_green(
                verify, self._cross_check(spec, task_files, remote),
            )
        self._record_gate_resolution(verify)
        if not verify.passed:
            await self.git_ops.cleanup_merge_worktree(self.merge_wt)
            return await red_verify_outcome(verify, self._red_verify_context())
        return await self._pass_outcome(verify)

    async def _task_files(self) -> tuple[str, ...] | None:
        """The verify's task files: the request's own, else derived on THIS host.

        Mechanism 5b: when Lever C is on and task_files was not supplied, the
        scope is derived on the DISPATCHING host (which has a fresh main), so
        the spec ships a faithful non-None task_files and the remote never
        derives it against its own possibly-stale main.  Lever C off keeps the
        local-only path byte-identical.
        """
        if self.req.task_files is not None:
            return tuple(self.req.task_files)
        if not self.req.config.enabled_verify_runners:
            return None
        derived = await _derive_task_files_from_git(self.merge_wt, self.req.config)
        return tuple(derived) if derived else None

    def _pool(self, task_files: tuple[str, ...] | None) -> VerifyRunnerPool:
        """The verify host's pool: the injected remote alone, or the local merge-verify runner.

        The remote path gets NO LocalRunner (γ decision 4): its
        RunnerUnavailable reaches the caller, which benches the host and
        re-dispatches.  Warm-swap runs only for LOCAL leases (the caller's
        job).  Both archive under data/verify-logs/<task_id>/ beside the
        task-verify logs for operator triage (task 1920).

        Both join the verify-result cache through the access the verifier
        holds: the remote ships a view of its registry under the config's
        modes, and the pool records either's greens (task 9123).
        """
        access = self.verifier.verify_cache
        if self.runner is not None:
            return VerifyRunnerPool(
                [self.runner],
                event_store=self.event_store,
                task_id=self.task_id,
                archive_root=self.archive_root,
                verify_cache_registry=None if access is None else access.registry,
                verify_cache_modes=None if access is None else self.req.config.verify_cache,
            )
        return VerifyRunnerPool(
            [self._local_merge_verify_runner(task_files, access)],
            event_store=self.event_store,
            task_id=self.task_id,
            verify_cache_registry=None if access is None else access.registry,
        )

    def _local_merge_verify_runner(
        self, task_files: tuple[str, ...] | None, access: CacheAccess | None,
    ) -> VerifyRunner:
        """The full gate's scoped + unscoped bundle, or only the declared history leg (task 7248)."""
        if self.history_leg_command is not None:
            return HistoryLegRunner(
                self.merge_wt, self.req.config, self.history_leg_command,
                task_id=self.task_id, archive_root=self.archive_root,
            )
        # INV-1 (task 2883): the dispatching store lets run_scoped's merge gate
        # emit trivial_pass_escalated.  The merge-flake gate's side-effects are
        # recorded by this call (task ε), off the FlakeSuppression the returned
        # VerifyResult carries.
        return self._local_runner(task_files, verify_cache=access, narrow_gate=_narrow(self.gate_decision))

    def _local_runner(
        self,
        task_files: tuple[str, ...] | None,
        *,
        verify_cache: CacheAccess | None = None,
        narrow_gate: NarrowGateDecision | None = None,
    ) -> LocalRunner:
        """The local scoped + unscoped bundle; outside the cache unless given *verify_cache*,
        so the cross-check's trust anchor always runs, and on the full gate unless given *narrow_gate*."""
        return LocalRunner(
            self.merge_wt, self.req.config, self.effective_module_configs, task_files,
            run_scoped=self.verifier.run_scoped,
            run_unscoped=self.verifier.run_unscoped_typechecks,
            task_id=self.task_id,
            archive_root=self.archive_root,
            event_store=self.event_store,
            verify_cache=verify_cache,
            narrow_gate=narrow_gate,
        )

    def _holds_merge_lane_lease(self) -> bool:
        """Whether this verify holds the persistent warm lane's ``<lane_dir>.lock``.

        Merge-verify lease (task 2315; lock path converged in task 2685): ONLY
        a LOCAL in-process verify on the persistent warm lane records the
        holder-pgid lease and holds the SAME lock reify's warm-lane scripts and
        DF's ``_seed_warm_lane`` take, so a reset or GC reclaim detects the
        in-flight verify instead of clobbering its worktree.
        """
        return (
            self.runner is None
            and self.req.config.git.persistent_merge_worktree
            and self.git_ops.is_persistent_merge_lane(self.merge_wt)
        )

    async def _dispatch_attempts(
        self, pool: VerifyRunnerPool, spec: MergeVerifySpec,
    ) -> list[VerifyResult]:
        """Every attempt's result, oldest first; the last one is the verdict.

        max_retries=0 inside the pool: post-merge verify hangs are usually
        deterministic, so only the two classified retries below re-dispatch.
        An ENOSPC red takes the prune-and-retry; any other infra-transient
        red the bounded (possibly narrowed) retry loop — never both.
        """
        async with contextlib.AsyncExitStack() as stack:
            if self._holds_merge_lane_lease():
                await stack.enter_async_context(self.git_ops.merge_verify_lease())
            first = await pool.dispatch(
                self.merge_sha, spec, depth=self.depth, speculative=self.speculative,
                chain_items=self.chain_items, chain_build_ms=self.chain_build_ms,
                dispatch_facts=self.dispatch_facts,
            )
            if not first.passed and verify_hit_enospc(first):
                return [first, *await self._retry_after_enospc(pool, spec)]
            if _infra_transient_red(first):
                return [first, *await self._retry_infra_transient(pool, spec, first)]
            return [first]

    async def _redispatch(
        self, pool: VerifyRunnerPool, spec: MergeVerifySpec, attempt: int,
    ) -> VerifyResult:
        return await pool.dispatch(
            self.merge_sha, spec, attempt=attempt, depth=self.depth,
            speculative=self.speculative, chain_items=self.chain_items,
            chain_build_ms=self.chain_build_ms,
            dispatch_facts=(self.dispatch_facts or MergeVerifyDispatchFacts()).as_retry(),
        )

    async def _retry_after_enospc(
        self, pool: VerifyRunnerPool, spec: MergeVerifySpec,
    ) -> list[VerifyResult]:
        """Prune stale _merge-* worktrees (never task worktrees) and retry once, within budget.

        ENOSPC is often a self-healing host condition, and always comes from
        the scoped phase, before the unscoped gate runs.
        """
        prior_enospc = self.enospc_retries.get(self.task_id, 0)
        if prior_enospc >= self.max_enospc:
            return []
        self.enospc_retries[self.task_id] = prior_enospc + 1
        enospc_keep = {self.merge_wt, *(self.keep_worktrees or ())}
        pruned = await self.git_ops.prune_stale_merge_worktrees(keep=enospc_keep)
        logger.warning(
            'Task %s: post-merge verify hit ENOSPC; pruned %d '
            'stale merge worktree(s), retrying verify once',
            self.task_id, len(pruned),
        )
        return [await self._redispatch(pool, spec, 1)]

    async def _retry_infra_transient(
        self, pool: VerifyRunnerPool, spec: MergeVerifySpec, first: VerifyResult,
    ) -> list[VerifyResult]:
        """The bounded retry of a classified infra-transient red (task ν, verify-scope-inversion-prd.md).

        A NARROWED (failed-only) retry earns its own budget
        (max_narrowed/narrowed_retries, task 2835), decoupled from the legacy
        enospc_retries/max_enospc budget so an unrelated earlier ENOSPC can
        never starve it; a non-narrowed retry shares the legacy budget.
        reverify_member_solo never sets retry_failed_only; its budget is
        pinned by
        orchestrator/tests/test_merge_queue_train_attribution.py::TestReverifyMemberSoloContract::test_persistent_infra_transient_red_earns_one_full_retry

        The loop RE-CHECKS infra-category membership every iteration (INV-4
        storm-escape): a retry whose fresh category flips to a deterministic
        red exits at once and falls through to the generic block.
        """
        narrowed_spec = await self._narrowed_retry_spec(first, spec)
        narrowed = narrowed_spec is not None
        if narrowed_spec is not None:
            spec = narrowed_spec
        retries, budget = (
            (self.narrowed_retries, self.max_narrowed) if narrowed
            else (self.enospc_retries, self.max_enospc)
        )
        attempts: list[VerifyResult] = []
        verify = first
        while _infra_transient_red(verify) and retries.get(self.task_id, 0) < budget:
            retries[self.task_id] = retries.get(self.task_id, 0) + 1
            logger.warning(
                'Task %s: post-merge verify classified infra-transient '
                '(category=%s); retrying verify (attempt=%d, budget=%d, narrowed=%s)',
                self.task_id, verify.category, retries[self.task_id], budget, narrowed,
            )
            verify = await self._redispatch(pool, spec, retries[self.task_id])
            attempts.append(verify)
        return attempts

    async def _narrowed_retry_spec(
        self, attempt0_result: VerifyResult, spec: MergeVerifySpec,
    ) -> MergeVerifySpec | None:
        """The retry's spec narrowed to attempt-0's did-not-pass subset, or None to retry FULL.

        Failed-only merge-verify retry PRODUCER (PRD verify-retry-failed-only
        D2, re-wired in task 3059), built from THIS call's own attempt-0
        result: build_attempt0_payload, then the INV-3 tree-OID corroboration
        in assemble_retry_verify_env.  Guarded by req.retry_failed_only (D1,
        task 2833).  Every fail-safe route returns None, so the retry runs
        FULL on the legacy budget; a REMOTE dispatch skips the producer
        altogether (verify_runner.py::MergeVerifySpec).  On a narrowed retry
        the attempt-0 verdict map is copied into the shadow-baseline sink
        (D4, §5.4) — only here, so an uncorroborated tree never seeds a stale
        baseline.
        """
        if not self.req.retry_failed_only or self.runner is not None:
            return None
        # spec.verify_env is attempt-0's OWN build environment; threading it
        # makes the `cargo nextest list` probe model the same build the verify
        # performs — without it a warm lane's probe compiles cold.
        attempt0 = await build_attempt0_payload(
            self.merge_wt, attempt0_result, verify_env=spec.verify_env
        )
        if attempt0 is None:
            return None
        retry_env = await assemble_retry_verify_env(
            self.git_ops, self.req, self.merge_wt, attempt0
        )
        if retry_env is None:
            return None
        if self.shadow_baseline_sink is not None:
            self.shadow_baseline_sink.update(
                {**attempt0.debug_verdicts, **attempt0.release_verdicts}
            )
        return dataclasses.replace(spec, verify_env={**spec.verify_env, **retry_env})

    async def _record_observations(self, attempts: list[VerifyResult]) -> None:
        """Record EVERY attempt's flake observation, oldest first (task ε).

        A superseded attempt's VERDICT is dead, but its OBSERVATION is not
        (§5.5 — record the observation, not the remedy): an attempt whose
        SCOPED red was suppressed can still come back failing on the unscoped
        gate or on ENOSPC and be retried.  §8.3's dedup keeps a replay
        idempotent, and each attempt stamps its own ``observed_at``.
        """
        for superseded in attempts[:-1]:
            if getattr(superseded, 'flake_suppression', None) is not None:
                await self._record_observation(superseded)
        await self._record_observation(attempts[-1])

    async def _record_observation(self, result: VerifyResult) -> None:
        """Record whatever flake observation *result* carries (PRD task ε, §5.8).

        THE topology rule ε encodes: the DISCRIMINATOR runs where the WORKTREE
        is (local host or remote runner); the RECORDER runs HERE, on the
        dispatcher, the only scope where a local OR remote ``VerifyResult``
        sits alongside an ``EventStore``, an escalation queue,
        ``project_root``, ``merge_sha`` and ``task_id`` at once — which makes
        the ``flake_occurrence`` row, the ``merge_flake_suppressed`` fact and
        the INV-4 storm-streak bump unconditional BY CONSTRUCTION.  Never
        raises (the recorder owns that guarantee).
        """
        await record_merge_flake_suppression(
            result,
            project_root=self.req.config.project_root,
            project_id=self.req.config.fused_memory.project_id,
            merge_sha=self.merge_sha,
            task_id=self.task_id,
            event_store=self.event_store,
            escalation_queue=self.escalation_queue,
        )

    def _cross_check_remote(self, verify: VerifyResult) -> VerifyRunner | None:
        """The remote whose green the local trust anchor must re-verify before it lands, if any.

        A REMOTE dispatch's REAL-suite green, with
        verify_cross_check_remote_green on.  A verdict that is no whole-tree
        evidence (a trivial config-only pass, or a narrow gate's) has no full
        suite to cross-check; it is left to the known-red gate.
        """
        if (
            self.runner is not None
            and verify.passed
            and verify.is_whole_tree_evidence
            and self.req.config.verify_cross_check_remote_green
        ):
            return self.runner
        return None

    def _cross_check(
        self, spec: MergeVerifySpec, task_files: tuple[str, ...] | None, remote: VerifyRunner,
    ) -> CrossCheck:
        return CrossCheck(
            git_ops=self.git_ops, req=self.req, merge_wt=self.merge_wt,
            merge_sha=self.merge_sha, spec=spec, remote=remote,
            trust_anchor=self._local_runner(task_files),
            event_store=self.event_store, escalation_queue=self.escalation_queue,
            halt_hook=self.halt_hook, quarantine_divergent=self.quarantine_divergent,
            record_observation=self._record_observation,
        )

    def _record_gate_resolution(self, verify: VerifyResult) -> None:
        """Record which merge gate this verify took and what it ran, once its verdict is final."""
        decision = self.gate_decision
        if decision is None or self.event_store is None:
            return
        miss = decision.miss
        self.event_store.emit(EventType.merge_gate_resolution, task_id=self.task_id, data={
            'resolution': decision.resolution.value,
            'miss_reason': None if miss is None else miss.reason.value,
            'miss_path': None if miss is None else miss.path,
            'changed_paths': list(decision.changed_paths),
            'commands': None if decision.plan is None else list(executed_commands(decision.plan)),
            'passed': verify.passed,
            'base_sha': self.narrow_gate_base,
            'merge_sha': self.merge_sha,
        })

    def _red_verify_context(self) -> RedVerifyContext:
        # HOST-AFFINITY (task 2565): recorded, never steered by.  getattr's
        # True default covers a runner-like object that omits is_local.
        return RedVerifyContext(
            git_ops=self.git_ops, req=self.req, verifier=self.verifier,
            timeouts=self.timeouts, max_timeouts=self.max_timeouts,
            event_store=self.event_store, escalation_queue=self.escalation_queue,
            dry_run_handles=self.dry_run_handles,
            main_health_probe_handles=self.main_health_probe_handles,
            merge_base_sha=self.merge_base_sha, main_sha=self.main_sha,
            origin_is_local=self.runner is None or getattr(self.runner, 'is_local', True),
        )

    async def _pass_outcome(self, verify: VerifyResult) -> MergeOutcome | None:
        withheld = await self._withheld_trivial_pass(verify)
        if withheld is not None:
            return withheld
        self._seed_main_baseline(verify)
        return None

    async def _withheld_trivial_pass(self, verify: VerifyResult) -> MergeOutcome | None:
        """Refuse to CAS-advance a pass that is no whole-tree evidence over a KNOWN-red main tip (task 2823).

        A config-only merge short-circuits to verify._trivial_pass WITHOUT
        running any suite, and a narrow gate (task 5411) runs only its command
        set, so neither is evidence main is green (the reify 2026-07-19
        incident).  A whole-tree pass legitimately heals a red main.
        CACHE-ONLY peek, failing OPEN when nothing is known red.
        """
        if verify.is_whole_tree_evidence:
            return None
        resolution = verify.narrow_gate or 'trivial'
        try:
            _pass_main_sha = await self.git_ops.get_main_sha()
        except Exception:
            _pass_main_sha = ''
        _known_red_ids = known_failing_ids_on_main(_pass_main_sha) if _pass_main_sha else frozenset()
        if not _known_red_ids:
            return None
        # A distinct WITHHELD warning, not the ordinary main-red-churn line, so
        # the config-only-fixes-red edge is recognisable (task 2823 amendment).
        logger.warning(
            'Task %s: main-red gate WITHHELD a %s pass from advancing main %s '
            'over a known-red baseline (%d failing test(s): %s). It ran no '
            'whole-tree suite, so if this change is itself the fix it cannot '
            'self-heal by re-dispatch — main must be greened by a code-touching '
            'merge or operator action first.',
            self.task_id, resolution, _pass_main_sha, len(_known_red_ids),
            ', '.join(sorted(_known_red_ids)[:10]),
        )
        # Any non-None MergeOutcome return owns merge_wt cleanup (the caller
        # only reuses merge_wt on a None pass return).
        await self.git_ops.cleanup_merge_worktree(self.merge_wt)
        return MergeOutcome(
            'blocked',
            reason=(
                f'{TRIVIAL_PASS_MAIN_RED_REASON_PREFIX} [{resolution}] '
                f'(main {_pass_main_sha:.8}, {len(_known_red_ids)} '
                f'failing test(s)): '
                + ', '.join(sorted(_known_red_ids)[:10])
            ),
            disposition=MergeFailureDisposition.MAIN_RED,
        )

    def _seed_main_baseline(self, verify: VerifyResult) -> None:
        """Seed merge_sha's failing-id baseline — the EMPTY set — on a full-gate pass (task μ, B2).

        merge_sha IS the next main tip, so the next gate's baseline lookup is
        a cache hit.  Only merge+full breadth collects a junit report
        (failing_test_ids not None).  The EMPTY set, not the parsed ids
        (task 2590): a passing verify has no live failures by definition, and
        a rerun plugin's fail-then-pass would otherwise mask a later genuine
        regression of that test — so that anomaly is logged loudly.
        """
        if not self.merge_sha or verify.failing_test_ids is None or not verify.is_whole_tree_evidence:
            return
        if verify.failing_test_ids:
            logger.warning(
                'Task %s: post-merge verify passed but junit reported %d '
                'failing test id(s) on a pass (%s) — seeding the EMPTY set '
                'as the baseline for %s instead of trusting the parsed ids '
                '(likely a rerun-plugin retry-to-pass; a passing verify has '
                'no live failures by definition)',
                self.task_id, len(verify.failing_test_ids),
                ', '.join(sorted(verify.failing_test_ids)), self.merge_sha,
            )
        seed_main_baseline(self.merge_sha, frozenset(), skipped=verify.skipped_test_ids)


def _narrow(decision: NarrowGateDecision | None) -> NarrowGateDecision | None:
    return decision if decision is not None and decision.resolution.is_narrow else None


def _infra_transient_red(verify: VerifyResult) -> bool:
    return not verify.passed and (verify.category or '') in INFRA_TRANSIENT_CATEGORIES
