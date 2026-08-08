"""Tests for wiring the ItemLifecycle registry onto SpeculativeMergeWorker
(merge-queue-reliability PRD scope-4 kappa / task 2169).

Task iota (task 2164) delivered the UNWIRED substrate — ``ItemLifecycle``
(register/current/transition), ``ItemLifecycleState``, ``_LEGAL_TRANSITIONS``,
``IllegalLifecycleTransition`` — in merge_queue.py/merge_types.py, but nothing
in production calls ``register()``/``transition()`` yet. This task (kappa)
wires it in.

Steps covered:
  step-1  RED   — worker._lifecycle / worker._live_items presence +
                  worker._note_transition best-effort-loud contract
  step-2  GREEN — SpeculativeMergeWorker.__init__ wires _lifecycle/_live_items
                  + _register_item/_note_transition/_retire_item chokepoint
                  helpers

This module imports orchestrator.merge_queue LOCALLY inside each test method
(not at module scope) so a not-yet-implemented symbol (e.g. _note_transition,
before step-2) never breaks collection of the rest of the file during the RED
steps — mirrors test_merge_queue_request_liveness.py's / test_item_lifecycle.py's
convention.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeRequest, QueuedBranch

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_request_liveness.py / test_merge_queue_invariant_integration_gate.py;
# there is no shared worker conftest).
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


class _FakeEscalationQueue:
    """Minimal fake escalation queue (copied from
    test_merge_queue_request_liveness.py:119 — per-file duplication
    convention).
    """

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True


def _make_worker(git_ops: GitOps, *, escalation_queue: Any = None):
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_invariant_integration_gate.py:212's _make_worker.
    """
    from orchestrator.merge_queue import SpeculativeMergeWorker

    return SpeculativeMergeWorker(git_ops, asyncio.Queue(), escalation_queue=escalation_queue)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop.

    Duplicated from test_merge_queue_request_liveness.py (per-file
    duplication convention — see this file's module docstring).
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path.

    Duplicated from test_merge_queue.py (per-file duplication convention).
    """
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _mock_verify_pass() -> AsyncMock:
    """Return a mock that makes run_scoped_verification always pass.

    Duplicated from test_merge_queue.py (per-file duplication convention).
    """
    return AsyncMock(return_value=MagicMock(passed=True, summary=''))


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: registry + live-items substrate
# ---------------------------------------------------------------------------


class TestLifecycleRegistryPresence:
    """A freshly-built worker exposes an ItemLifecycle registry and an empty
    live-items object index (task 2169 step-1).

    RED until step-2 GREEN wires ``self._lifecycle``/``self._live_items``
    into ``SpeculativeMergeWorker.__init__``.
    """

    def test_lifecycle_is_an_item_lifecycle_instance(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import ItemLifecycle

        worker = _make_worker(git_ops)

        assert isinstance(worker._lifecycle, ItemLifecycle)

    def test_live_items_starts_empty(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)

        assert worker._live_items == {}


class TestNoteTransitionBestEffort:
    """``worker._note_transition(rid, from_state, to_state)`` NEVER raises on
    the hot path (PRD design-decision 4: invariants escalate loudly, degrade
    never) — it logs a WARNING and fires a best-effort escalation instead of
    letting ``IllegalLifecycleTransition`` propagate (task 2169 step-1).

    RED until step-2 GREEN adds ``_note_transition``.
    """

    def test_unregistered_rid_does_not_raise(
        self,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._note_transition(
                'mr-unregistered0', ItemLifecycleState.QUEUED, ItemLifecycleState.MERGING,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert 'mr-unregistered0' in warnings[0].message
        assert len(fake_eq.submitted) == 1

    def test_illegal_edge_does_not_raise_and_leaves_state_unchanged(
        self,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        rid = 'mr-aaaaaaaa'
        worker._lifecycle.register(rid)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            # Skip-stage move (mirrors test_item_lifecycle.py's
            # test_skip_stage_move_raises) — an illegal EDGE, not an
            # unregistered rid.
            worker._note_transition(
                rid, ItemLifecycleState.QUEUED, ItemLifecycleState.FINALIZING,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert rid in warnings[0].message
        assert len(fake_eq.submitted) == 1
        # The underlying registry state is untouched by the rejected move —
        # matches ItemLifecycle.transition()'s own "leaves state UNCHANGED"
        # contract (test_item_lifecycle.py::test_skip_stage_move_raises).
        assert worker._lifecycle.current(rid) == ItemLifecycleState.QUEUED

    def test_no_escalation_queue_still_does_not_raise(
        self,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """None-safe: a bare worker with no escalation_queue wired must not
        raise either (mirrors the None-safety convention used throughout
        this file — e.g. _submit_loop_escalation / _alarm_resource_audit)."""
        from orchestrator.merge_queue import ItemLifecycleState

        worker = _make_worker(git_ops, escalation_queue=None)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._note_transition(
                'mr-unregistered1', ItemLifecycleState.QUEUED, ItemLifecycleState.MERGING,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: merger-drain wiring — register QUEUED at the
# `_queue` drain chokepoint, transition to LANE_BUFFERED there, then to
# MERGING at the `_pop_next_pickable` pop feeding `_inflight_req`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergerDrainRegistersAndTransitions:
    """Driving one request through the merger drain (Style-A:
    ``worker.run()`` + ``q.put(req)``) must register it at QUEUED, observe
    LANE_BUFFERED, then MERGING via ``worker._lifecycle.current(...)`` — in
    step with the still-present transient fields (task 2169 step-3).

    RED until step-4 GREEN wires ``_register_item``/``_note_transition``
    into ``_buffer_owned_request`` (QUEUED -> LANE_BUFFERED at the drain)
    and ``_pop_next_pickable`` (-> MERGING at the pop).
    """

    async def test_item_in_external_queue_before_drain_is_pre_registry(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """An item still sitting in the external ``_queue`` — never yet
        drained by the worker — has no registry entry at all.

        Documents the producer-boundary asymmetry recorded in this task's
        design decisions: register() happens at the worker's first sighting
        of a request (the `_queue` DRAIN chokepoint), NOT at the external
        producer's put() — the module-level enqueue helpers have no
        reference to the worker/registry.
        """
        worker = _make_worker(git_ops)
        req = _make_request('pre-registry', 'pre-registry', git_ops.project_root, config)

        assert worker._lifecycle.current(req.request_id) is None

        await worker._queue.put(req)

        # Still sitting in _queue, undrained — still pre-registry.
        assert worker._lifecycle.current(req.request_id) is None
        assert req.request_id not in worker._live_items

    async def test_request_progresses_queued_lane_buffered_merging(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-seq', 'file_kappa_seq.py', 'x = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-seq', 'kappa-seq', wt, config)

        # Producer-boundary asymmetry (see test above): unseen by the
        # worker before the first drain, so pre-registry.
        assert worker._lifecycle.current(req.request_id) is None

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)

            # worker.run() has been scheduled but the loop has not yet
            # switched to it: asyncio.Queue.put() on a non-full unbounded
            # queue never actually suspends (mirrors test_merge_queue.py's
            # test_speculative_basic_throughput "Submit ... before the
            # worker processes them" comment) — still pre-registry.
            assert worker._lifecycle.current(req.request_id) is None

            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert observed[:2] == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'

        await worker.stop()
        await worker_task

    async def test_live_items_holds_request_during_merging_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """``_live_items`` must hold the MergeRequest, and the registry must
        read MERGING, throughout the dequeue-time merge window (task 2169
        step-3; task 2435 kappa-b dropped the former field-vs-registry
        agreement check now that the transient ``_inflight_req`` field is
        gone — the registry is the sole source of truth).
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-merging', 'file_kappa_merging.py', 'y = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-merging', 'kappa-merging', wt, config)

        captured: list[tuple[Any, Any]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            # Gate on the registry (task 2435 kappa-b: formerly
            # worker._inflight_req is not None — mirrors
            # test_merge_queue.py::test_merger_exception_resolves_inflight_future's
            # fault-injection rationale): recompute_suffix_conflict_graph()
            # also calls get_main_sha() before dequeue, when the registry is
            # not yet at MERGING — skip those, and fire only once.
            if (
                worker._lifecycle.current(req.request_id) == ItemLifecycleState.MERGING
                and not fired
            ):
                fired = True
                captured.append((
                    worker._lifecycle.current(req.request_id),
                    worker._live_items.get(req.request_id),
                ))
            return await original_get_main_sha()

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'get_main_sha', new=_spying_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert fired, 'expected the get_main_sha gate to fire while the registry read MERGING'
        observed_state, observed_live_obj = captured[0]
        assert observed_state == ItemLifecycleState.MERGING
        assert observed_live_obj is req

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: merger->verifier handoff (the 4 _verifier_queue
# puts transition MERGING -> AWAITING_VERIFY) and the 3 requeue-to-QUEUED
# sites (operator-halt / cascade self-requeue).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergerVerifierHandoffRegistersAwaitingVerify:
    """Every ``_verifier_queue.put(...)`` site — the successful RealMergeItem
    put and the three DecidedItem puts (conflict/train/loop-breaker) — must
    transition the registry MERGING -> AWAITING_VERIFY the instant the item
    becomes verifier-owned (task 2169 step-5).

    RED until step-6 GREEN wires ``_note_transition`` at each
    ``_verifier_queue.put`` call site.
    """

    async def test_successful_merge_progresses_to_awaiting_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import (
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        wt = await _make_branch_with_file(
            git_ops, 'kappa-awaiting-verify', 'file_kappa_av.py', 'z = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-awaiting-verify', 'kappa-awaiting-verify', wt, config)

        observed: list[tuple[Any, Any]] = []
        live_item_at_awaiting_verify: list[Any] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            original_note_transition(rid, from_state, to_state, **kwargs)
            # Capture _live_items' object AT the AWAITING_VERIFY transition
            # itself (not post-completion): step-8 wires DISPATCHING/VERIFYING
            # onward from here, and this single-item run awaits FULL pipeline
            # completion, so by the time `req.result` resolves `_live_items`
            # has already moved on to whatever the LATEST chokepoint (e.g.
            # `_inflight_append`) set it to. `_note_transition` applies the
            # live_obj update synchronously before returning, so this capture
            # (right after the real call, still within this single-threaded
            # transition) reflects exactly the state this test cares about.
            if rid == req.request_id and to_state == ItemLifecycleState.AWAITING_VERIFY:
                live_item_at_awaiting_verify.append(worker._live_items.get(rid))

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        # NOTE: this test awaits FULL pipeline completion (`req.result`), which
        # for a successful RealMergeItem only resolves once verify+finalize
        # has run to completion — i.e. strictly AFTER the DISPATCHING/VERIFYING
        # transitions step-8 wires at the verifier's dispatch call sites and
        # `_inflight_append`, and (since step-10) the FINALIZING/TERMINAL
        # transitions wired into `_finalize_inflight`/`_resolve_or_drop_abandoned`.
        # Unlike the loop-breaker/passthrough test below (whose Future resolves
        # early via `_oob_deliver`, racing ahead of the background verifier-side
        # dispatch), this path's assertion window necessarily includes the full
        # tail.
        assert observed == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.AWAITING_VERIFY),
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.VERIFYING),
            (ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING),
            (ItemLifecycleState.FINALIZING, ItemLifecycleState.TERMINAL),
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'
        assert live_item_at_awaiting_verify and isinstance(
            live_item_at_awaiting_verify[0], RealMergeItem,
        ), (
            '_live_items must hold the RealMergeItem (not the stale MergeRequest) '
            'once the item becomes verifier-owned'
        )

        await worker.stop()
        await worker_task

    async def test_loop_breaker_decided_item_progresses_to_awaiting_verify(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """The loop-breaker short-circuit (task already timed out
        MAX_POST_MERGE_VERIFY_TIMEOUTS times) also rides the verifier queue
        as a DecidedItem passthrough — the registry must observe
        MERGING -> AWAITING_VERIFY there too, not just on the successful
        RealMergeItem path above. Fires before any git work, so no worktree
        setup is needed (mirrors the "without doing any git work" contract
        in the loop-breaker's own docstring).
        """
        from orchestrator.merge_queue import (
            DecidedItem,
            ItemLifecycleState,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request(
            'kappa-loop-breaker', 'kappa-loop-breaker', tmp_path / 'no-such-wt', config,
        )
        worker._post_merge_verify_timeouts[req.task_id] = worker.MAX_POST_MERGE_VERIFY_TIMEOUTS

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)
        assert outcome.status == 'blocked', f'{outcome}'

        assert observed == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.AWAITING_VERIFY),
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'
        assert isinstance(worker._live_items[req.request_id], DecidedItem)

        await worker.stop()
        await worker_task


@pytest.mark.asyncio
class TestRequeueToQueuedSites:
    """The 3 ``_queue.put_nowait(req)`` requeue sites (pre-dispatch
    operator-halt in ``_dispatch_item``, mid-verify operator-halt in
    ``_run_inflight_verify``, and the head-failure cascade's downstream
    self-requeue) must transition a TRACKED item's registry entry back to
    QUEUED, and the SAME request_id must be able to re-arm/re-drain through
    the normal ``_buffer_owned_request`` chokepoint afterwards WITHOUT a
    duplicate-register ``ValueError`` or a spurious escalation (task 2169
    step-5).

    Targets the pre-dispatch operator-halt branch of ``_dispatch_item``
    directly (bare worker, no real ``worker.run()`` loop) — mirrors
    test_merge_queue_request_liveness.py::TestOperatorHaltRequeueNoFalseAlarm's
    bare-item pattern, which drives the SAME branch on an UNREGISTERED item
    and asserts zero escalations; this test additionally pre-registers the
    item (as kappa's later steps will have it arrive: DISPATCHING) to prove
    the registry-side re-arm/re-drain contract.

    RED until step-6 GREEN wires ``_note_requeue`` at the 3 sites and
    teaches ``_buffer_owned_request`` to skip re-``register()``-ing an
    already-tracked (requeued) request_id.
    """

    async def test_predispatch_operator_halt_requeues_and_redrains_cleanly(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import (
            InflightStatus,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        req = _make_request(
            'kappa-predispatch-halt', 'kappa-predispatch-halt', tmp_path, config,
        )
        item = RealMergeItem(
            request=req,
            merge_result=MagicMock(),
            merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef',
            speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.DISPATCHING)

        worker._operator_halt.set()
        entry = await worker._dispatch_item(item)

        assert entry is not None and entry.status == InflightStatus.REQUEUED_PREDISPATCH
        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.QUEUED
        assert worker._live_items[req.request_id] is req
        assert not queue.empty(), 'req must be put back on _queue by the halt branch'
        assert len(fake_eq.submitted) == 0, (
            f'requeue must not escalate: {fake_eq.submitted!r}'
        )

        # Re-arm and re-drain: the SAME request_id flows back through the
        # normal drain chokepoint without a duplicate-register ValueError.
        drained = await queue.get()
        worker._buffer_owned_request(drained)

        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.LANE_BUFFERED
        assert worker._live_items[req.request_id] is drained
        assert len(fake_eq.submitted) == 0, (
            f're-drain must not escalate either: {fake_eq.submitted!r}'
        )

    async def test_predispatch_operator_halt_on_unregistered_item_stays_silent(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """REGRESSION GUARD: an item whose request was never registered
        (mirrors TestOperatorHaltRequeueNoFalseAlarm's bare-item pattern —
        several existing unit tests construct items directly, bypassing the
        drain chokepoint) must NOT escalate either.  ``_note_requeue`` is a
        deliberate no-op on an unregistered rid — unlike
        ``_note_transition``'s other callers, this is not itself treated as
        a wiring-bug signal at these 3 sites.
        """
        from orchestrator.merge_queue import (
            InflightStatus,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        req = _make_request(
            'kappa-predispatch-halt-bare', 'kappa-predispatch-halt-bare', tmp_path, config,
        )
        item = RealMergeItem(
            request=req,
            merge_result=MagicMock(),
            merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef',
            speculative=False,
        )

        worker._operator_halt.set()
        entry = await worker._dispatch_item(item)

        assert entry is not None and entry.status == InflightStatus.REQUEUED_PREDISPATCH
        assert worker._lifecycle.current(req.request_id) is None
        assert len(fake_eq.submitted) == 0, (
            f'unregistered requeue must stay silent: {fake_eq.submitted!r}'
        )


# ---------------------------------------------------------------------------
# task 3082 step-5 PART 2 RED / step-6 GREEN: the on_requeued/_note_requeue
# PAIRING INVARIANT.
#
# The canonical requeue recipe — named verbatim by the contended-lease
# branch's own comment — is
#   release_or_cleanup + put_nowait + on_requeued + _note_requeue + REQUEUED.
# Four of the five requeue sites follow it; one (the dead-verify no-progress
# abort) omits `_note_requeue`, and that omission is this task's defect. The
# ledger half and the registry half must move together at EVERY site, so this
# is pinned two ways: a RUNTIME spy-parity check over the drivable paths, and
# an AST STRUCTURAL check over all five sites (including the two a unit test
# cannot easily drive).
#
# Sibling task 3204 owns extracting the five-site recipe into one helper; this
# task lands only the invariant.
# ---------------------------------------------------------------------------


async def _make_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
):
    """Create a merged ``RealMergeItem`` on a fresh branch (real git, no mocks).

    Duplicated from test_merge_queue_request_liveness.py:621 (per-file
    duplication convention — see this file's module docstring).
    """
    from orchestrator.merge_queue import RealMergeItem

    wt = await _make_branch_with_file(git_ops, branch, filename, content)
    req = _make_request(branch, branch, wt, config)
    merge_result = await git_ops.merge_to_main(wt, branch)
    assert merge_result.success and merge_result.merge_commit, f'{merge_result!r}'
    assert merge_result.merge_worktree is not None
    base_sha = await git_ops.get_main_sha()
    item = RealMergeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=False,
    )
    return req, item


async def _dead_gate_never_returns(*args: object, **kwargs: object) -> MagicMock:
    """Simulates a dead/hung LOCAL verify: never returns, never writes.

    Duplicated from test_merge_queue_request_liveness.py:1090 (per-file
    duplication convention).
    """
    await asyncio.Event().wait()
    raise AssertionError('unreachable — this Event is never set')  # pragma: no cover


def _attr_calls(node: ast.AST, attr: str) -> list[ast.Call]:
    """Every ``<something>.attr(...)`` call in *node*'s subtree."""
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def _first_arg_src(call: ast.Call, aliases: dict[str, str] | None = None) -> str:
    """The source text of *call*'s first positional argument ('' if none).

    A bare ``Name`` argument is resolved through *aliases* (local
    ``name = <expr>`` bindings) so a cosmetic ``rid = req.request_id`` beside
    ``self._note_requeue(req.request_id, ...)`` still reads as the same
    request_id rather than a spurious pairing failure (amendment review,
    task 3082).
    """
    if not call.args:
        return ''
    src = ast.unparse(call.args[0])
    if aliases and isinstance(call.args[0], ast.Name):
        return aliases.get(src, src)
    return src


@pytest.mark.asyncio
class TestOnRequeuedIsAlwaysPairedWithNoteRequeue:
    """Every requeue site must move the ledger AND the lifecycle registry
    together: an ``on_requeued(rid)`` call is only correct when a
    ``_note_requeue(rid)`` call sits beside it (task 3082 step-5 RED / step-6
    GREEN, requirement 3).

    RED until step-6 adds the missing ``_note_requeue`` to the dead-verify
    no-progress abort.
    """

    async def test_every_driven_requeue_moves_ledger_and_registry_together(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """RUNTIME spy parity: drive three requeue paths on ONE worker and
        assert the recorded request_id MULTISETS are equal.

        Two are known-good controls (the contended-lease defer and the
        pre-dispatch operator halt) and one is the defect (the dead-verify
        no-progress abort). Asserting equality against controls in the SAME
        test is what makes this a parity check rather than a restatement of the
        site-level test in test_merge_queue_request_liveness.py.
        """
        from orchestrator.git_ops import MergeVerifyLeaseContended
        from orchestrator.merge_queue import (
            InflightStatus,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_runner import HostLease

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        ledger_rids: list[str] = []
        registry_rids: list[str] = []
        real_on_requeued = worker._request_ledger.on_requeued
        real_note_requeue = worker._note_requeue

        def _spy_on_requeued(rid: str, *a: Any, **k: Any) -> Any:
            ledger_rids.append(rid)
            return real_on_requeued(rid, *a, **k)

        def _spy_note_requeue(rid: str, *a: Any, **k: Any) -> Any:
            registry_rids.append(rid)
            return real_note_requeue(rid, *a, **k)

        worker._request_ledger.on_requeued = _spy_on_requeued  # type: ignore[method-assign]
        worker._note_requeue = _spy_note_requeue  # type: ignore[method-assign]

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        async def _lease_contended(*_a: object, **_k: object) -> object:
            raise MergeVerifyLeaseContended(Path('/x/_merge-verify.lock'), 300.0)

        # ── CONTROL 1: contended-lease defer (merge_queue.py :13839/:13840) ──
        req_cl, item_cl = await _make_merged_item(
            git_ops, config, 'df3082-pair-contended', 'pc.py', 'c=1\n',
        )
        worker._register_owned_merge_worktree(item_cl.merge_wt)
        worker._register_item(item_cl, initial=ItemLifecycleState.VERIFYING)
        with patch('orchestrator.merge_queue._run_post_merge_verify', _lease_contended):
            vr_cl = await asyncio.wait_for(
                worker._run_inflight_verify(item_cl, lease), timeout=15.0,
            )
        assert vr_cl.status == InflightStatus.REQUEUED, f'{vr_cl!r}'

        # ── THE DEFECT: dead-verify no-progress abort (:13775/:13776) ────────
        req_dv, item_dv = await _make_merged_item(
            git_ops, config, 'df3082-pair-deadverify', 'pd.py', 'd=1\n',
        )
        worker._register_owned_merge_worktree(item_dv.merge_wt)
        worker._register_item(item_dv, initial=ItemLifecycleState.VERIFYING)
        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns,
        ):
            vr_dv = await asyncio.wait_for(
                worker._run_inflight_verify(item_dv, lease), timeout=15.0,
            )
        assert vr_dv.status == InflightStatus.REQUEUED, f'{vr_dv!r}'

        # ── CONTROL 2: pre-dispatch operator halt (:14727/:14730) ────────────
        # LAST, because setting _operator_halt would otherwise route the two
        # _run_inflight_verify drives above into the operator-halt abort
        # (trigger 2) instead of the branches under test.
        req_ph = _make_request('df3082-pair-halt', 'df3082-pair-halt', tmp_path, config)
        item_ph = RealMergeItem(
            request=req_ph, merge_result=MagicMock(),
            merge_wt=tmp_path / 'merge_wt_ph', base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item_ph, initial=ItemLifecycleState.DISPATCHING)
        worker._operator_halt.set()
        entry_ph = await worker._dispatch_item(item_ph)
        assert entry_ph is not None
        assert entry_ph.status == InflightStatus.REQUEUED_PREDISPATCH, f'{entry_ph!r}'

        expected = {req_cl.request_id, req_dv.request_id, req_ph.request_id}
        assert set(ledger_rids) == expected, (
            f'all three drives must have hit the ledger requeue: {ledger_rids!r}'
        )
        assert sorted(registry_rids) == sorted(ledger_rids), (
            f'every on_requeued must be paired with a _note_requeue at the same '
            f'site — ledger={sorted(ledger_rids)!r} registry={sorted(registry_rids)!r}; '
            f'unpaired={sorted(set(ledger_rids) - set(registry_rids))!r}'
        )
        for rid in expected:
            assert worker._lifecycle.current(rid) == ItemLifecycleState.QUEUED, (
                f'{rid} must be left at QUEUED; reads {worker._lifecycle.current(rid)!r}'
            )

    async def test_every_on_requeued_call_site_has_a_sibling_note_requeue(self) -> None:
        """STRUCTURAL parity over EVERY ``on_requeued`` site, via ``ast``.

        Covers the site a unit test cannot easily drive (the head-failure
        cascade's downstream self-requeue) and fails loudly if a future requeue
        site forgets the call.

        Keys on the enclosing function name and the argument EXPRESSION, never
        on line numbers (they drift) and deliberately NOT on the site COUNT
        (amendment review, task 3082: a count is a number, not an invariant —
        task 3204's pending extraction of the recipe into one helper will
        legitimately collapse it). Bare ``Name`` arguments are resolved through
        their local binding so a cosmetic ``rid = req.request_id`` is not a
        spurious failure. The pairing must be a sibling in the same statement
        block — function-level scoping would be too weak, since
        ``_run_inflight_verify`` contains both a correctly-paired requeue
        branch and (before step-6) an unpaired one, both using
        ``req.request_id``.
        """
        import orchestrator.merge_queue as _mq  # module-local import convention

        source = Path(inspect.getfile(_mq)).read_text()
        tree = ast.parse(source)

        # id(node) -> parent node, and id(stmt) -> the statement list holding it.
        parent: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[id(child)] = node
        owner: dict[int, list] = {}
        for node in ast.walk(tree):
            for field in ('body', 'orelse', 'finalbody'):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    for st in block:
                        if isinstance(st, ast.stmt):
                            owner[id(st)] = block

        def _enclosing(node: ast.AST, kinds: tuple[type, ...]) -> ast.AST | None:
            cur: ast.AST | None = node
            while cur is not None and not isinstance(cur, kinds):
                cur = parent.get(id(cur))
            return cur

        # Local `name = <expr>` bindings, so a bare Name argument at either call
        # resolves to the same expression text (amendment review, task 3082).
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    aliases[target.id] = ast.unparse(node.value)

        sites = _attr_calls(tree, 'on_requeued')
        assert sites, 'no on_requeued call sites found — the AST walk is mis-wired'

        unpaired: list[str] = []
        for call in sites:
            stmt = _enclosing(call, (ast.stmt,))
            assert stmt is not None
            block = owner.get(id(stmt))
            assert block is not None, f'on_requeued at line {call.lineno} has no block'
            arg = _first_arg_src(call, aliases)
            siblings = {
                _first_arg_src(c, aliases)
                for sib in block for c in _attr_calls(sib, '_note_requeue')
            }
            if arg not in siblings:
                fn = _enclosing(call, (ast.FunctionDef, ast.AsyncFunctionDef))
                unpaired.append(
                    f'{getattr(fn, "name", "<module>")}:{call.lineno} '
                    f'on_requeued({arg}) has no sibling _note_requeue({arg})'
                )

        assert unpaired == [], (
            'every on_requeued call site must carry a sibling _note_requeue with the '
            'SAME request_id expression (the canonical requeue recipe: '
            'release_or_cleanup + put_nowait + on_requeued + _note_requeue + '
            f'InflightStatus.REQUEUED). Unpaired: {unpaired!r}'
        )


# ---------------------------------------------------------------------------
# task 3204 step-1 RED / step-2 GREEN: the requeue recipe has a single
# CHOKEPOINT — `SpeculativeMergeWorker._requeue_request`.
#
# This is the strictly stronger, "preferred" form of the pairing invariant
# that TestOnRequeuedIsAlwaysPairedWithNoteRequeue (above) asserts in its
# weaker sibling form. Both are kept deliberately: after the extraction the
# sibling test still holds (the helper body keeps the two calls as siblings in
# one statement block), while sole-callership adds the property that makes
# re-introducing task 3082's defect impossible to EXPRESS — an unpaired
# requeue can no longer be written outside the helper at all.
# ---------------------------------------------------------------------------

_MQ_CHOKEPOINT_CLASS = 'SpeculativeMergeWorker'
_MQ_REQUEUE_CHOKEPOINT = '_requeue_request'


def _chokepoint_ranges(tree: ast.Module, cls_name: str, method_name: str) -> list[tuple[int, int]]:
    """Line ranges of every *method_name* defined DIRECTLY on ``class cls_name``.

    Copied from test_lock_release_single_writer_guard.py:134 (and the same
    recursion in test_prune_chokepoint_guard.py) per the per-file duplication
    convention for orchestrator test helpers — there is deliberately no shared
    guard conftest.

    Class-qualification matters: a plain name-only scan would also sanction a
    same-named method on ANOTHER class, or a module-level function. The walk
    tracks the nearest *directly enclosing* ClassDef and resets it to None
    inside a function body, so a closure nested within a method does not
    inherit the class qualification either.
    """
    ranges: list[tuple[int, int]] = []

    def visit(node: ast.AST, enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == method_name and enclosing_class == cls_name:
                    ranges.append((child.lineno, getattr(child, 'end_lineno', child.lineno)))
                visit(child, None)
            else:
                visit(child, enclosing_class)

    visit(tree, None)
    return ranges


def _requeue_chokepoint_offenders(source: str) -> list[str]:
    """Every requeue-recipe call site in *source* OUTSIDE the chokepoint.

    A "requeue-recipe call site" is a ``self._queue.put_nowait(...)`` (the
    queue half) or any ``<something>.on_requeued(...)`` (the ledger half).
    Both must live inside ``SpeculativeMergeWorker._requeue_request``, which is
    what makes the two halves — plus the ``_note_requeue`` registry mirror the
    helper also fires — impossible to separate.

    Deliberately does NOT scan ``_note_requeue`` call sites: the invariant is
    ONE-DIRECTIONAL (every ``on_requeued`` needs a ``_note_requeue``, not the
    converse), and ``_finalize_inflight``'s sentinel chokepoint repair is a
    legitimate unpaired ``_note_requeue``. Deliberately does NOT assert a site
    COUNT either — see the :888 docstring above: a count is a number, not an
    invariant.

    Pure (takes source text, not a path) so the scanner self-tests below can
    drive it on synthetic modules; :func:`_scan_merge_queue_requeue_sites` is
    the thin read_text() wrapper. Returns ``[]`` rather than raising on a
    source that cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    ranges = _chokepoint_ranges(tree, _MQ_CHOKEPOINT_CLASS, _MQ_REQUEUE_CHOKEPOINT)

    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def _enclosing_fn(node: ast.AST) -> str:
        cur: ast.AST | None = node
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cur = parent.get(id(cur))
        return getattr(cur, 'name', '<module>')

    # `put_nowait` is anchored on the `_queue` receiver: an unrelated
    # `.put_nowait()` on some other queue is not a requeue. `on_requeued` needs
    # no anchor — it exists solely for this recipe.
    sites: list[ast.Call] = [
        call for call in _attr_calls(tree, 'put_nowait')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == '_queue'
    ]
    sites += _attr_calls(tree, 'on_requeued')

    offenders = [
        f'{_enclosing_fn(call)}:{call.lineno} {ast.unparse(call)}'
        for call in sites
        if not any(start <= call.lineno <= end for start, end in ranges)
    ]
    # ast.walk is breadth-first, so hits come out in tree-depth order rather
    # than source order. Sort so the offender list reads top-to-bottom.
    offenders.sort(key=lambda entry: int(entry.split(':', 1)[1].split(' ', 1)[0]))
    return offenders


def _scan_merge_queue_requeue_sites() -> list[str]:
    """Thin read_text() wrapper around :func:`_requeue_chokepoint_offenders`."""
    import orchestrator.merge_queue as _mq  # module-local import convention

    return _requeue_chokepoint_offenders(Path(inspect.getfile(_mq)).read_text())


class TestRequeueRecipeHasASingleChokepoint:
    """``_requeue_request`` is the SOLE caller of both ``_queue.put_nowait``
    and ``on_requeued`` (task 3204 requirement 3, "preferred" form).

    RED until step-2 adds the helper and routes all five requeue sites
    through it.
    """

    # ── scanner self-tests: the ratchet must never pass vacuously ──────────

    def test_scanner_accepts_a_compliant_chokepoint(self) -> None:
        """POSITIVE control: both halves inside the helper -> no offenders."""
        src = (
            'class SpeculativeMergeWorker:\n'
            '    def _requeue_request(self, req):\n'
            '        self._queue.put_nowait(req)\n'
            '        self._request_ledger.on_requeued(req.request_id)\n'
            '        self._note_requeue(req.request_id, live_obj=req)\n'
        )
        assert _requeue_chokepoint_offenders(src) == []

    def test_scanner_flags_a_bypass_outside_the_helper(self) -> None:
        """SYNTHETIC BYPASS: a second method re-queuing by hand is flagged.

        This is the whole point of the guard — without it the scanner could be
        mis-wired and still report a green ratchet.
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    def _requeue_request(self, req):\n'
            '        self._queue.put_nowait(req)\n'
            '        self._request_ledger.on_requeued(req.request_id)\n'
            '\n'
            '    def _sneaky(self, req):\n'
            '        self._queue.put_nowait(req)\n'
            '        self._request_ledger.on_requeued(req.request_id)\n'
        )
        offenders = _requeue_chokepoint_offenders(src)
        assert len(offenders) == 2, offenders
        assert all(entry.startswith('_sneaky:') for entry in offenders), offenders

    def test_scanner_flags_a_same_named_method_on_another_class(self) -> None:
        """CLASS-QUALIFICATION: a ``_requeue_request`` on some OTHER class does
        not sanction its body — that is why the copied ``_chokepoint_ranges``
        recursion tracks the nearest directly-enclosing ClassDef.
        """
        src = (
            'class SomethingElse:\n'
            '    def _requeue_request(self, req):\n'
            '        self._queue.put_nowait(req)\n'
        )
        offenders = _requeue_chokepoint_offenders(src)
        assert len(offenders) == 1, offenders
        assert offenders[0].startswith('_requeue_request:'), offenders

    def test_scanner_ignores_docstring_and_comment_mentions(self) -> None:
        """AST-not-regex: prose naming the guarded symbols is not a call site.

        merge_queue.py genuinely contains several such mentions (the
        ``_queue.put_nowait(req)`` re-arm narrative and the REQUEUED_PREDISPATCH
        comment), so a grep-based guard would report them as offenders forever.
        """
        src = (
            '"""A module whose docstring names on_requeued and\n'
            'self._queue.put_nowait(req) purely as prose."""\n'
            '\n'
            '# self._queue.put_nowait(req) in a comment is not a call either.\n'
            'X = 1\n'
        )
        assert _requeue_chokepoint_offenders(src) == []

    def test_scanner_returns_empty_on_unparseable_source(self) -> None:
        """SYNTAX ERROR: return [] rather than raising."""
        assert _requeue_chokepoint_offenders('def broken(:\n') == []

    # ── the ratchet ────────────────────────────────────────────────────────

    def test_requeue_recipe_has_exactly_one_chokepoint(self) -> None:
        """STRUCTURAL: no ``_queue.put_nowait`` / ``on_requeued`` call site in
        merge_queue.py lives outside ``_requeue_request``.
        """
        offenders = _scan_merge_queue_requeue_sites()
        assert offenders == [], (
            'the requeue recipe must have exactly ONE chokepoint: every '
            '`self._queue.put_nowait(...)` and `.on_requeued(...)` call site in '
            'merge_queue.py must live inside '
            f'`{_MQ_CHOKEPOINT_CLASS}.{_MQ_REQUEUE_CHOKEPOINT}`, which fires the '
            'queue, the ledger and the lifecycle-registry re-arm together and '
            'atomically. Route these sites through `self._requeue_request(req)` '
            f'instead. Offenders: {offenders!r}'
        )


@pytest.mark.asyncio
class TestRequeueRequestHelperContract:
    """``_requeue_request`` moves the queue, the ledger and the registry
    together, and is deliberately SYNCHRONOUS (task 3204 step-1 RED).
    """

    async def test_requeue_request_moves_queue_ledger_and_registry_together(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        worker = _make_worker(git_ops)
        req = _make_request('df3204-helper', 'df3204-helper', tmp_path, config)
        rid = worker._register_item(req, initial=ItemLifecycleState.VERIFYING)
        worker._request_ledger.on_dequeue(req, now=0.0)
        assert rid in worker._request_ledger.open_request_ids()

        worker._requeue_request(req)

        assert worker._queue.qsize() == 1, 'the request must be back on _queue'
        assert worker._queue.get_nowait() is req
        assert rid not in worker._request_ledger.open_request_ids(), (
            'the ledger entry must be removed so this parked request never ages out'
        )
        assert worker._lifecycle.current(rid) == ItemLifecycleState.QUEUED, (
            f'registry must read QUEUED; reads {worker._lifecycle.current(rid)!r}'
        )
        # The OBJECT SWAP is what actually prevents a phantom finalize head:
        # `_finalizing_head_entry()` only counts `isinstance(obj, InflightEntry)`.
        assert worker._live_items[rid] is req, (
            '_live_items must hold the MergeRequest, not the InflightEntry'
        )

    async def test_requeue_request_is_synchronous(self, git_ops: GitOps) -> None:
        """The helper must be a plain ``def``, never ``async def``.

        That is the property which makes the documented atomicity — no `await`
        between the ``put_nowait`` and the registry re-arm, so the merger
        loop's drain can never observe the request on ``_queue`` while the
        registry still reads VERIFYING — STRUCTURAL rather than conventional.
        """
        worker = _make_worker(git_ops)
        assert not inspect.iscoroutinefunction(worker._requeue_request), (
            '_requeue_request must stay sync: an await between the put_nowait and '
            'the _note_requeue re-opens exactly the window task 3082 closed'
        )
        src = inspect.getsource(type(worker)._requeue_request)
        assert 'await ' not in src, f'_requeue_request must contain no await:\n{src}'


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: verifier DISPATCH-FILL + blocking-get wiring —
# AWAITING_VERIFY/REDISPATCH_PARKED -> DISPATCHING at BOTH _dispatch_item call
# sites (DISPATCH-FILL's get_nowait/redispatch-popleft branch and the
# blocking-get branch), DISPATCHING -> VERIFYING at _inflight_append,
# DISPATCHING -> REDISPATCH_PARKED on the no-host appendleft (both call
# sites), `_live_items` holding the SpeculativeItem during the DISPATCHING
# window, and the dispatch-time staleness remerge window observed at MERGING.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifierDispatchFillRegistersDispatching:
    """Every path through the verifier's DISPATCH-FILL / blocking-get dispatch
    call sites must transition the registry into DISPATCHING before calling
    ``_dispatch_item``, then onward to VERIFYING (real dispatch),
    REDISPATCH_PARKED (no host free), or MERGING (dispatch-time staleness
    remerge) — not just the legacy ``_dispatching_item``/``_remerging_item``
    census fields, which cover only a subset of these paths (task 2169
    step-7).

    RED until step-8 GREEN wires ``_note_transition`` at both
    ``_dispatch_item`` call sites in ``_verifier_loop``, at
    ``_inflight_append``, at the two no-host ``_redispatch.appendleft``
    sites, and around the dispatch-time remerge inside ``_dispatch_item``.
    """

    async def test_single_item_progresses_through_dispatching_to_verifying(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Happy path (single fresh request, idle worker): the item is
        fetched by the BLOCKING-GET call site (the verifier loop is idle —
        DISPATCH-FILL finds nothing and falls through to the blocking
        ``_verifier_queue.get()`` — see this task's design notes), proving
        that call site's DISPATCHING wiring as well as the normal
        DISPATCHING -> VERIFYING transition at ``_inflight_append``. This
        test awaits FULL pipeline completion (`req.result`), so (since
        step-10) the tail also covers the FINALIZING/TERMINAL transitions.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-dispatching-verifying', 'file_kappa_dv.py', 'v = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request(
            'kappa-dispatching-verifying', 'kappa-dispatching-verifying', wt, config,
        )

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert observed == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.AWAITING_VERIFY),
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.VERIFYING),
            (ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING),
            (ItemLifecycleState.FINALIZING, ItemLifecycleState.TERMINAL),
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'

        await worker.stop()
        await worker_task

    async def test_live_items_holds_item_during_dispatching_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """``_live_items`` must hold the SpeculativeItem, and the registry
        must read DISPATCHING, throughout the dispatch window (task 2169
        step-7; task 2435 kappa-b dropped the former field-vs-registry
        agreement check now that the transient ``_dispatching_item`` field
        is gone — the registry is the sole source of truth).
        """
        from orchestrator.merge_queue import (
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        wt = await _make_branch_with_file(
            git_ops, 'kappa-dispatching-window', 'file_kappa_dw.py', 'd = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request(
            'kappa-dispatching-window', 'kappa-dispatching-window', wt, config,
        )

        captured: list[tuple[Any, Any]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            # Gate on the registry (task 2435 kappa-b: formerly
            # worker._dispatching_item is not None).
            if (
                worker._lifecycle.current(req.request_id) == ItemLifecycleState.DISPATCHING
                and not fired
            ):
                fired = True
                captured.append((
                    worker._lifecycle.current(req.request_id),
                    worker._live_items.get(req.request_id),
                ))
            return await original_get_main_sha()

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'get_main_sha', new=_spying_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert fired, 'expected the get_main_sha gate to fire while the registry read DISPATCHING'
        observed_state, observed_live_obj = captured[0]
        assert observed_state == ItemLifecycleState.DISPATCHING
        assert isinstance(observed_live_obj, RealMergeItem)

        await worker.stop()
        await worker_task

    async def test_dispatch_fill_call_site_no_host_bounces_to_redispatch_parked(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """An item already sitting in ``_verifier_queue`` when the loop is
        entered is fetched by DISPATCH-FILL's ``get_nowait()`` branch (the
        FIRST ``_dispatch_item`` call site). With zero free hosts forced,
        ``_dispatch_item`` returns None and the item bounces onto
        ``_redispatch`` — this must transition
        AWAITING_VERIFY -> DISPATCHING -> REDISPATCH_PARKED, not just set
        the legacy ``_dispatching_item`` field.
        """
        from orchestrator.merge_queue import (
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        fake_allocator = MagicMock()
        fake_allocator.free_host_count.return_value = 0
        worker._host_allocator = fake_allocator

        req = _make_request(
            'kappa-dispatch-fill-no-host', 'kappa-dispatch-fill-no-host', tmp_path, config,
        )
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.AWAITING_VERIFY)

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        await worker._verifier_queue.put(item)
        await worker._verifier_queue.put(None)  # type: ignore[arg-type]

        await worker._verifier_loop()

        assert observed == [
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.REDISPATCH_PARKED),
        ], f'unexpected transition sequence: {observed!r}'
        assert len(worker._redispatch) == 1
        # _live_items must track whatever object actually ended up in
        # _redispatch (may be a dataclasses.replace()'d copy of `item` with
        # cap_permit cleared, not `item` itself) — the lockstep invariant is
        # membership-consistency, not object identity with the original.
        assert worker._live_items[req.request_id] is worker._redispatch[0]

    async def test_blocking_get_call_site_no_host_bounces_to_redispatch_parked(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """An item that arrives AFTER the verifier loop is already blocked on
        ``_verifier_queue.get()`` (DISPATCH-FILL found nothing and fell
        through) is dispatched via the SECOND ``_dispatch_item`` call site.
        With zero free hosts forced, this must also transition
        AWAITING_VERIFY -> DISPATCHING -> REDISPATCH_PARKED — this call site
        has no legacy ``_dispatching_item`` coverage at all today.
        """
        from orchestrator.merge_queue import (
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        fake_allocator = MagicMock()
        fake_allocator.free_host_count.return_value = 0
        worker._host_allocator = fake_allocator

        req = _make_request(
            'kappa-blocking-get-no-host', 'kappa-blocking-get-no-host', tmp_path, config,
        )
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.AWAITING_VERIFY)

        observed: list[tuple[Any, Any]] = []
        reached = asyncio.Event()
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
                if len(observed) >= 2:
                    reached.set()
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        loop_task = asyncio.create_task(worker._verifier_loop())
        try:
            # Let the loop find _redispatch/_verifier_queue empty and fall
            # through to the blocking get() BEFORE the item is put — proving
            # this is genuinely the blocking-get call site, not DISPATCH-FILL's.
            for _ in range(5):
                await asyncio.sleep(0)
            await worker._verifier_queue.put(item)
            await asyncio.wait_for(reached.wait(), timeout=5.0)
        finally:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task

        assert observed[:2] == [
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.REDISPATCH_PARKED),
        ], f'unexpected transition sequence: {observed!r}'
        assert worker._live_items[req.request_id] is worker._redispatch[0]

    async def test_dispatch_time_staleness_remerge_observed_at_merging(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The dispatch-time staleness remerge window (Mechanism 2, inside
        ``_dispatch_item``) must be observed at MERGING, then transition
        back to DISPATCHING before the (now fresh) item proceeds to a normal
        dispatch — mirrors test_merge_queue.py's
        test_verify_pickup_rebases_when_main_advanced's OOB-advance
        interposition to force the ``main_advanced`` remerge deterministically.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-stale-remerge', 'file_kappa_stale.py', 's = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-stale-remerge', 'kappa-stale-remerge', wt, config)

        original_dispatch = worker._dispatch_item
        advanced = {'done': False}

        async def _dispatch_with_oob_advance(item: Any) -> Any:
            if not advanced['done']:
                advanced['done'] = True
                (git_ops.project_root / 'oob_kappa.py').write_text('extra = 1\n')
                await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
                await _run(
                    ['git', 'commit', '-m', 'oob advance for kappa remerge test'],
                    cwd=git_ops.project_root,
                )
            return await original_dispatch(item)

        worker._dispatch_item = _dispatch_with_oob_advance  # type: ignore[method-assign]

        captured: list[Any] = []
        original_remerge = worker._remerge

        async def _spying_remerge(req_: Any, started: Any) -> Any:
            captured.append(worker._lifecycle.current(req.request_id))
            return await original_remerge(req_, started)

        worker._remerge = _spying_remerge  # type: ignore[method-assign]

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert captured == [ItemLifecycleState.MERGING], (
            f'expected the registry to read MERGING while _remerge() ran: {captured!r}'
        )
        assert (ItemLifecycleState.DISPATCHING, ItemLifecycleState.MERGING) in observed, (
            f'missing DISPATCHING -> MERGING on remerge entry: {observed!r}'
        )
        assert (ItemLifecycleState.MERGING, ItemLifecycleState.DISPATCHING) in observed, (
            f'missing MERGING -> DISPATCHING on remerge exit ("then back"): {observed!r}'
        )

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: verify -> finalize wiring — VERIFYING ->
# FINALIZING at the _finalizing_head-set (excluding passthrough / pre-dispatch
# sentinel entries, which never reach real finalize work), entry.phase
# FINALIZING <-> GATE_REVERIFY sub-transitions, FINALIZING -> TERMINAL on
# land, passthrough entries retiring directly (DISPATCHING -> TERMINAL, no
# FINALIZING hop), the RUNNER_UNAVAILABLE FINALIZING -> MERGING ->
# REDISPATCH_PARKED cascade, `_live_items` holding the InflightEntry during
# the FINALIZING window (parallel to `_finalizing_head`), and rid retirement
# (current()==TERMINAL, dropped from `_live_items`) at land.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightRegistersFinalizingAndTerminal:
    """Driving verify -> finalize must observe VERIFYING -> FINALIZING ->
    TERMINAL via ``worker._lifecycle``, hold the InflightEntry in
    ``worker._live_items`` during the FINALIZING window (parallel to the
    still-present ``_finalizing_head``), and retire the request_id
    (``current()==TERMINAL``, dropped from ``_live_items``) once the request
    lands (task 2169 step-9).

    RED until step-10 GREEN wires ``_note_transition``/``_retire_item`` into
    ``_finalize_inflight`` and ``_resolve_or_drop_abandoned``.
    """

    async def test_full_pipeline_land_observes_finalizing_then_terminal(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Happy-path full-pipeline drive: the registry must show
        VERIFYING -> FINALIZING -> TERMINAL as the tail of the transition
        sequence, and the rid must be retired once the request lands.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-finalize-land', 'file_kappa_fl.py', 'f = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-finalize-land', 'kappa-finalize-land', wt, config)

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert observed == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.AWAITING_VERIFY),
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.VERIFYING),
            (ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING),
            (ItemLifecycleState.FINALIZING, ItemLifecycleState.TERMINAL),
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'

        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items

        await worker.stop()
        await worker_task

    async def test_live_items_holds_entry_during_finalizing_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """``_live_items`` must hold the InflightEntry, and the registry must
        read FINALIZING, throughout the finalize window (task 2169 step-9;
        task 2435 kappa-b dropped the former field-vs-registry agreement
        check now that the transient ``_finalizing_head`` field is gone —
        the registry is the sole source of truth).

        Mirrors test_merge_queue.py::TestEntryPhaseDuringFinalize's direct
        ``_finalize_inflight()`` drive (compat-shim entry, verify_task=None,
        pre-established PASS) to gate on ``git_ops.advance_main``.
        """
        from orchestrator.merge_queue import (
            InflightEntry,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_runner import HostLease

        branch = 'kappa-finalize-window'
        wt = await _make_branch_with_file(git_ops, branch, 'kappa_fw.py', 'x = 1\n')
        req = _make_request(branch, branch, wt, config)
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        assert merge_result.merge_worktree is not None

        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req, merge_result=merge_result, merge_wt=merge_result.merge_worktree,
            base_sha=base_sha, speculative=False,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        mock_allocator = MagicMock()
        mock_allocator.release = AsyncMock()
        mock_allocator.cancel_and_release = AsyncMock()
        worker._host_allocator = mock_allocator
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = InflightEntry(
            item=item, lease=lease, verify_task=None,
            merge_wt=item.merge_wt, was_speculative=False,
        )

        captured: list[tuple[Any, Any]] = []
        original_advance = git_ops.advance_main

        async def _capturing_advance(*args: Any, **kwargs: Any) -> Any:
            captured.append((
                worker._lifecycle.current(req.request_id),
                worker._live_items.get(req.request_id),
            ))
            return await original_advance(*args, **kwargs)

        git_ops.advance_main = _capturing_advance  # type: ignore[method-assign]
        try:
            with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
                advanced = await worker._finalize_inflight(entry)
        finally:
            git_ops.advance_main = original_advance  # type: ignore[method-assign]

        assert advanced is True, f'Expected True (advanced); got: {advanced}'
        assert len(captured) >= 1, 'advance_main must have been called at least once'
        observed_state, observed_live_obj = captured[0]
        assert observed_state == ItemLifecycleState.FINALIZING, (
            f'expected the registry to read FINALIZING while advance_main ran: {observed_state!r}'
        )
        assert observed_live_obj is entry

        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items

    async def test_passthrough_entry_retires_directly_without_finalizing(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """A passthrough (immediate_outcome) InflightEntry never awaits a
        verify_task and never runs the CAS loop — it must retire straight
        from DISPATCHING to TERMINAL, WITHOUT an intermediate FINALIZING hop
        (DISPATCHING -> FINALIZING is not a legal edge; only
        VERIFYING -> FINALIZING is).
        """
        from orchestrator.merge_queue import (
            DecidedItem,
            InflightEntry,
            ItemLifecycleState,
            MergeOutcome,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        req = _make_request('kappa-passthrough', 'kappa-passthrough', tmp_path, config)
        outcome = MergeOutcome('conflict', reason='merge conflict')
        decided = DecidedItem(
            request=req, base_sha='deadbeef', speculative=False, immediate_outcome=outcome,
        )
        worker._register_item(decided, initial=ItemLifecycleState.DISPATCHING)

        entry = InflightEntry(
            item=decided, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False, passthrough_outcome=outcome,
        )

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        result = await worker._finalize_inflight(entry)

        assert result is False  # conflict -> not advanced
        assert observed == [
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.TERMINAL),
        ], f'unexpected transition sequence: {observed!r}'
        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items

    async def test_runner_unavailable_cascade_finalizing_merging_redispatch_parked(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """RUNNER_UNAVAILABLE re-merges the entry and front-queues it on
        ``_redispatch`` — the registry must observe
        FINALIZING -> MERGING (for the duration of the ``_remerge()`` call)
        -> REDISPATCH_PARKED (once landed on ``_redispatch``), mirroring the
        dispatch-time staleness remerge window's "then back" shape.
        """
        from orchestrator.merge_queue import (
            InflightEntry,
            InflightStatus,
            InflightVerifyResult,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeItem,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_runner import HostLease

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        fake_allocator = MagicMock()
        fake_allocator.quarantine_and_release = AsyncMock()
        worker._host_allocator = fake_allocator

        req = _make_request('kappa-ru-cascade', 'kappa-ru-cascade', tmp_path, config)
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        fake_runner = MagicMock()
        fake_runner.name = 'laptop'
        fake_runner.is_local = False
        lease = HostLease(name='laptop', runner=fake_runner, is_local=False)

        async def _fake_ru_verify() -> InflightVerifyResult:
            return InflightVerifyResult(
                outcome=None, merge_wt=item.merge_wt,
                status=InflightStatus.RUNNER_UNAVAILABLE, reason='ssh timeout',
            )

        entry = InflightEntry(
            item=item, lease=lease, verify_task=asyncio.ensure_future(_fake_ru_verify()),
            merge_wt=item.merge_wt, was_speculative=False,
        )

        remerged = MagicMock(spec=SpeculativeItem)
        captured_mid_remerge: list[Any] = []

        async def _spying_remerge(req_: Any, started: Any) -> Any:
            captured_mid_remerge.append(worker._lifecycle.current(req.request_id))
            return remerged

        worker._remerge = _spying_remerge  # type: ignore[method-assign]

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        result = await worker._finalize_inflight(entry)

        assert result is False
        assert captured_mid_remerge == [ItemLifecycleState.MERGING], (
            f'expected the registry to read MERGING while _remerge() ran: {captured_mid_remerge!r}'
        )
        assert observed == [
            (ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING),
            (ItemLifecycleState.FINALIZING, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.REDISPATCH_PARKED),
        ], f'unexpected transition sequence: {observed!r}'
        assert worker._live_items[req.request_id] is remerged
        assert worker._redispatch[0] is remerged

    async def test_gate_reverify_sub_transitions_finalizing_gate_reverify_finalizing(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """A ``rebased_pending_reverify`` retry must be observed at
        GATE_REVERIFY, then transition back to FINALIZING once the gate
        clears and the CAS loop continues — mirrors test_merge_queue.py::
        TestSpeculativeGateReverifyConsumesAdvanceOutcome's ``advance_main``
        side_effect technique to force the branch deterministically.
        """
        from orchestrator.git_ops import AdvanceOutcome
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-gate-reverify', 'file_kappa_gr.py', 'g = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-gate-reverify', 'kappa-gate-reverify', wt, config)

        advance_calls: list[Any] = []

        async def _advance_side_effect(*args: Any, **kwargs: Any) -> Any:
            advance_calls.append((args, kwargs))
            if len(advance_calls) == 1:
                return AdvanceOutcome(
                    'rebased_pending_reverify',
                    advanced_sha='ab' * 20, rebased_from='ba' * 20, rebased_onto='cd' * 20,
                )
            # Second call (post-rebuild): terminal failure — sidesteps the
            # success-path gate machinery, which is not this test's concern.
            return AdvanceOutcome('not_descendant')

        async def _fake_reverify_rebased_tree(*args: Any, **kwargs: Any) -> Any:
            return None  # gate clears — disjoint/green re-verify

        observed: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            if rid == req.request_id:
                observed.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'advance_main', side_effect=_advance_side_effect),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
            patch(
                'orchestrator.merge_queue._reverify_rebased_tree',
                new=_fake_reverify_rebased_tree,
            ),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked', f'expected blocked; got {outcome!r}'
        assert (ItemLifecycleState.FINALIZING, ItemLifecycleState.GATE_REVERIFY) in observed, (
            f'missing FINALIZING -> GATE_REVERIFY: {observed!r}'
        )
        assert (ItemLifecycleState.GATE_REVERIFY, ItemLifecycleState.FINALIZING) in observed, (
            f'missing GATE_REVERIFY -> FINALIZING (gate cleared, loop continues): {observed!r}'
        )

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# task 3082 step-1 RED / step-2 GREEN: a DROPPED/REQUEUED sentinel entry must
# never be recorded as FINALIZING.
#
# ``_finalize_inflight``'s VERIFYING -> FINALIZING hop is guarded ONLY against
# ABANDONED_PREDISPATCH / REQUEUED_PREDISPATCH, and it sits BEFORE the
# DROPPED/REQUEUED sentinel check — so a requeued item reaches it and BOTH
# dispositions are wrong:
#   (B) registry still at VERIFYING (a requeue site that skipped
#       ``_note_requeue``): the hop fires SILENTLY, records the requeued item
#       as FINALIZING and swaps ``_live_items[rid]`` to the InflightEntry, so
#       ``_finalizing_head_entry()`` returns a phantom head forever.
#   (A) registry already bounced to QUEUED (the shape all four working
#       requeue siblings produce): ``_advance_if_at`` neither fires nor
#       tolerates QUEUED, falls through to ``_note_transition`` and fires a
#       dedup'd ``merge_lifecycle_transition_rejected`` L1 on EVERY dead-verify
#       abort — a legitimate, expected, recurring event.
# Moving the hop below the sentinel check makes a DROPPED/REQUEUED item never
# reach it at all: correct by construction in both directions.
# ---------------------------------------------------------------------------


async def _make_verifying_entry_with_sentinel(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    escalation_queue: Any,
    status: Any,
) -> tuple[Any, MergeRequest, Any]:
    """Build ``(worker, req, entry)`` for a REAL merged item registered at
    VERIFYING whose already-completed ``verify_task`` carries an
    ``InflightVerifyResult`` with sentinel *status* (task 3082).

    The shape mirrors
    ``TestFinalizeInflightRegistersFinalizingAndTerminal::
    test_live_items_holds_entry_during_finalizing_window`` (:1097) — a real
    ``git_ops.merge_to_main`` + ``RealMergeItem`` + mocked host allocator —
    differing only in that ``verify_task`` is a completed future carrying the
    sentinel rather than ``None``.  Both task-3082 classes below need exactly
    this fixture chain.

    NOTE ``_register_item`` leaves ``_live_items[rid]`` holding the
    ``RealMergeItem``, NOT the ``InflightEntry`` — production swaps it at
    ``_inflight_append``.  That is deliberate here: it makes the ``live_obj``
    swap performed by whichever registry hop actually runs directly
    observable, which is the mechanism under test.
    """
    from orchestrator.merge_queue import (
        InflightEntry,
        InflightVerifyResult,
        ItemLifecycleState,
        RealMergeItem,
        SpeculativeMergeWorker,
    )
    from orchestrator.verify_runner import HostLease

    wt = await _make_branch_with_file(git_ops, branch, filename, content)
    req = _make_request(branch, branch, wt, config)
    merge_result = await git_ops.merge_to_main(wt, branch)
    assert merge_result.success and merge_result.merge_commit, f'{merge_result!r}'
    assert merge_result.merge_worktree is not None
    base_sha = await git_ops.get_main_sha()
    item = RealMergeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=False,
    )

    worker = SpeculativeMergeWorker(
        git_ops, asyncio.Queue(), escalation_queue=escalation_queue,
    )
    mock_allocator = MagicMock()
    mock_allocator.release = AsyncMock()
    mock_allocator.cancel_and_release = AsyncMock()
    worker._host_allocator = mock_allocator
    worker._register_owned_merge_worktree(item.merge_wt)
    worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

    async def _sentinel_result() -> InflightVerifyResult:
        return InflightVerifyResult(outcome=None, merge_wt=None, status=status)

    entry = InflightEntry(
        item=item,
        lease=HostLease(name='local', runner=MagicMock(), is_local=True),
        verify_task=cast(Any, asyncio.ensure_future(_sentinel_result())),
        merge_wt=item.merge_wt,
        was_speculative=False,
    )
    return worker, req, entry


def _rejected_transition_escalations(fake_eq: _FakeEscalationQueue) -> list:
    """The ``merge_lifecycle_transition_rejected`` subset of *fake_eq*.

    Category-specific assertion idiom, copied from
    test_merge_queue_duplicate_submission.py:316/:341 (per-file duplication
    convention).
    """
    return [e for e in fake_eq.submitted if e.category == 'merge_lifecycle_transition_rejected']


@pytest.mark.asyncio
class TestRequeuedSentinelNeverRecordedAsFinalizing:
    """A ``_finalize_inflight`` drive whose verify returned the REQUEUED
    sentinel must never leave the registry reading FINALIZING, and must never
    fire a ``merge_lifecycle_transition_rejected`` escalation — regardless of
    whether the requeue site already bounced the registry to QUEUED
    (task 3082 step-1 RED / step-2 GREEN).

    RED until step-2 relocates the VERIFYING -> FINALIZING hop below the
    DROPPED/REQUEUED sentinel check.
    """

    async def test_requeued_sentinel_is_never_recorded_as_finalizing(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Today's trigger-3 shape: the requeue site put the request back on
        ``_queue`` but never called ``_note_requeue``, so the registry still
        reads VERIFYING when finalize runs.  The hop must NOT fire — a
        requeued item is not finalizing.
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker, req, entry = await _make_verifying_entry_with_sentinel(
            git_ops, config, 'df3082-requeued-verifying', 'df3082_rv.py', 'x = 1\n',
            escalation_queue=fake_eq, status=InflightStatus.REQUEUED,
        )
        rid = req.request_id

        advanced = await worker._finalize_inflight(entry)

        assert advanced is False, f'a REQUEUED sentinel must not advance main: {advanced!r}'
        current = worker._lifecycle.current(rid)
        assert current != ItemLifecycleState.FINALIZING, (
            f'a requeued item was recorded as FINALIZING (phantom finalize head); '
            f'registry reads {current!r}'
        )
        assert worker._finalizing_head_entry() is None, (
            f'requeued entry left as the finalize head: '
            f'{worker._finalizing_head_entry()!r} (registry={current!r})'
        )
        assert not req.result.done(), (
            'a requeued request must be left PENDING for its re-dispatch, not resolved'
        )
        assert _rejected_transition_escalations(fake_eq) == [], (
            f'a requeue must not escalate: {fake_eq.submitted!r}'
        )

    async def test_note_requeued_shaped_requeue_fires_no_rejected_transition_escalation(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The shape all four working requeue siblings produce (and the one
        trigger 3 acquires at step-6): the site already bounced the registry
        to QUEUED via ``_note_requeue`` before finalize ran.  The hop must not
        fire a rejected-transition L1 for that legitimate, recurring event.
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker, req, entry = await _make_verifying_entry_with_sentinel(
            git_ops, config, 'df3082-requeued-queued', 'df3082_rq.py', 'y = 2\n',
            escalation_queue=fake_eq, status=InflightStatus.REQUEUED,
        )
        rid = req.request_id

        # What the requeue site itself does (merge_queue.py:13840 template).
        worker._note_requeue(rid, live_obj=req)
        assert worker._lifecycle.current(rid) == ItemLifecycleState.QUEUED

        advanced = await worker._finalize_inflight(entry)

        assert advanced is False, f'a REQUEUED sentinel must not advance main: {advanced!r}'
        assert _rejected_transition_escalations(fake_eq) == [], (
            f'a properly-requeued item must not fire a rejected-transition L1 on '
            f'every dead-verify abort: {fake_eq.submitted!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.QUEUED, (
            f'finalize must leave an already-requeued item at QUEUED so it can '
            f're-enter through the drain; registry reads {current!r}'
        )
        assert worker._live_items[rid] is req, (
            f'_live_items must hold the MergeRequest (the object swap is what '
            f'clears the finalize head): {worker._live_items.get(rid)!r}'
        )
        assert worker._finalizing_head_entry() is None, (
            f'requeued entry left as the finalize head: {worker._finalizing_head_entry()!r}'
        )
        assert not req.result.done(), (
            'a requeued request must be left PENDING for its re-dispatch, not resolved'
        )


# ---------------------------------------------------------------------------
# task 3082 step-3 RED / step-4 GREEN: neither sentinel exit may leave
# `_live_items` residue.
#
# The DROPPED/REQUEUED sentinel branch returns with no registry disposition at
# all, so an InflightEntry that reached it stays in `_live_items` and
# `_finalizing_head_entry()` keeps returning it — the phantom head, plus one
# accreted 'extra InflightEntry object(s)' WARNING per occurrence. The two
# sentinels need DIFFERENT dispositions: a DROPPED item's sole waiter is gone
# and nothing will ever re-enter (retire), while a REQUEUED item is sitting on
# `_queue` and MUST re-enter through the drain (bounce to QUEUED, never
# retire).
# ---------------------------------------------------------------------------


def _accretion_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The ``_finalizing_head_entry`` at-most-one-mid-finalize accretion
    WARNINGs in *caplog* (merge_queue.py:8005-8014) — the signal the field
    journal emitted repeatedly while a phantom head sat in ``_live_items``.
    """
    return [
        r.getMessage() for r in caplog.records
        if 'Invariant violation:' in r.getMessage()
        or 'extra InflightEntry object(s)' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestSentinelExitLeavesNoLiveItemsResidue:
    """``_finalize_inflight``'s DROPPED and REQUEUED sentinel exits must each
    dispose of the entry explicitly — no ``_live_items`` residue, no phantom
    finalize head, and no accretion WARNING (task 3082 step-3 RED / step-4
    GREEN).

    RED until step-4 gives the branch its two dispositions: DROPPED ->
    ``_retire_item``, REQUEUED -> idempotent ``_note_requeue``.
    """

    async def test_dropped_sentinel_retires_the_entry(
        self, git_ops: GitOps, config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The sole-waiter-gave-up exit: nothing will ever re-enter the
        pipeline for this request_id, so TERMINAL + a ``_live_items`` pop is
        the only correct end state.
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker, req, entry = await _make_verifying_entry_with_sentinel(
            git_ops, config, 'df3082-dropped', 'df3082_dr.py', 'd = 1\n',
            escalation_queue=fake_eq, status=InflightStatus.DROPPED,
        )
        rid = req.request_id

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            advanced = await worker._finalize_inflight(entry)
            head = worker._finalizing_head_entry()

        assert advanced is False, f'a DROPPED sentinel must not advance main: {advanced!r}'
        assert head is None, f'dropped entry left as the finalize head: {head!r}'
        assert rid not in worker._live_items, (
            f'dropped entry left in _live_items: {worker._live_items.get(rid)!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.TERMINAL, (
            f'a dropped request must be retired to TERMINAL; registry reads {current!r}'
        )
        assert _accretion_warnings(caplog) == [], (
            f'the at-most-one-mid-finalize accretion WARNING must stay silent: '
            f'{_accretion_warnings(caplog)!r}'
        )
        assert _rejected_transition_escalations(fake_eq) == [], (
            f'retiring a dropped request must not escalate: {fake_eq.submitted!r}'
        )

    async def test_requeued_sentinel_leaves_no_inflight_entry_residue(
        self, git_ops: GitOps, config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A requeue site that skipped ``_note_requeue`` (today's trigger 3):
        the chokepoint must repair it to QUEUED with the MergeRequest back in
        ``_live_items`` — the object swap, not the state change, is what clears
        the finalize head.  It must NOT retire: the request is on ``_queue``
        and has to re-enter through the drain.
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker, req, entry = await _make_verifying_entry_with_sentinel(
            git_ops, config, 'df3082-requeued-residue', 'df3082_rr.py', 'r = 1\n',
            escalation_queue=fake_eq, status=InflightStatus.REQUEUED,
        )
        rid = req.request_id

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            advanced = await worker._finalize_inflight(entry)
            head = worker._finalizing_head_entry()

        assert advanced is False, f'a REQUEUED sentinel must not advance main: {advanced!r}'
        assert head is None, f'requeued entry left as the finalize head: {head!r}'
        assert worker._live_items[rid] is req, (
            f'_live_items must hold the MergeRequest, not the InflightEntry: '
            f'{worker._live_items.get(rid)!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.QUEUED, (
            f'a requeued request must be left at QUEUED so _buffer_owned_request '
            f'buffers it normally; registry reads {current!r}'
        )
        assert _accretion_warnings(caplog) == [], (
            f'the at-most-one-mid-finalize accretion WARNING must stay silent: '
            f'{_accretion_warnings(caplog)!r}'
        )
        assert _rejected_transition_escalations(fake_eq) == [], (
            f'the chokepoint repair must not escalate: {fake_eq.submitted!r}'
        )
        assert not req.result.done(), (
            'a requeued request must be left PENDING for its re-dispatch, not resolved'
        )

    async def test_correctly_wired_requeue_raced_by_the_drain_is_left_alone(
        self, git_ops: GitOps, config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The chokepoint repair must NOT fire for a requeue that was already
        wired correctly and then legitimately advanced past QUEUED while
        ``_finalize_inflight`` was suspended at ``await entry.verify_task``.

        ``_merger_loop``'s speculative look-ahead calls
        ``_buffer_owned_request`` + ``_drain_queue_into_lanes`` while a
        predecessor verify is in flight, so the requeued request can already
        read LANE_BUFFERED by the time finalize resumes.  A merely-not-QUEUED
        guard would then attempt LANE_BUFFERED -> QUEUED — not a legal edge —
        and fire the exact ``merge_lifecycle_transition_rejected`` L1 this task
        retires, on a legitimate, successful re-drain (amendment review,
        task 3082).
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker, req, entry = await _make_verifying_entry_with_sentinel(
            git_ops, config, 'df3082-requeue-raced', 'df3082_rq.py', 'q = 1\n',
            escalation_queue=fake_eq, status=InflightStatus.REQUEUED,
        )
        rid = req.request_id

        # What the requeue SITE does (correctly wired, post step-6), then what
        # the merger loop's look-ahead drain does before finalize resumes.
        worker._queue.put_nowait(req)
        worker._note_requeue(rid, live_obj=req)
        assert worker._lifecycle.current(rid) == ItemLifecycleState.QUEUED
        worker._drain_queue_into_lanes()
        assert worker._lifecycle.current(rid) == ItemLifecycleState.LANE_BUFFERED, (
            f'the drain must have buffered the requeued request; registry reads '
            f'{worker._lifecycle.current(rid)!r}'
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            advanced = await worker._finalize_inflight(entry)

        assert advanced is False, f'a REQUEUED sentinel must not advance main: {advanced!r}'
        assert _rejected_transition_escalations(fake_eq) == [], (
            f'a correctly-wired requeue raced past QUEUED by the drain must not '
            f'fire a rejected-transition escalation: {fake_eq.submitted!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.LANE_BUFFERED, (
            f'the drain\'s legitimate progress must be left untouched; registry '
            f'reads {current!r}'
        )
        assert worker._live_items[rid] is req, (
            f'_live_items must still hold the MergeRequest the drain buffered: '
            f'{worker._live_items.get(rid)!r}'
        )
        assert worker._finalizing_head_entry() is None, (
            f'no phantom finalize head may survive: {worker._finalizing_head_entry()!r}'
        )
        assert _accretion_warnings(caplog) == [], (
            f'the at-most-one-mid-finalize accretion WARNING must stay silent: '
            f'{_accretion_warnings(caplog)!r}'
        )
        assert not req.result.done(), (
            'the buffered request must stay PENDING for its re-dispatch'
        )


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: wiring-completeness proof (BEFORE repoint) —
# every dequeued request_id must reach TERMINAL with no leaks across the
# stop() drains, the two abandon/detach-drop sites, coalesce-superseded, and
# the auto-chain parent-superseded path; the registry's non-terminal set must
# agree with snapshot()['entries'] at several sampling points.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStopMidFlightRetiresEveryContainer:
    """``worker.stop()`` must retire the registry entry (best-effort
    TERMINAL + drop from ``_live_items``) for an item parked in EVERY
    drainable container — not just resolve its Future — so a
    stop()-mid-flight shutdown never leaves the registry's non-terminal set
    disagreeing with reality (task 2169 step-11).

    Seeds one item into each of: the lane buffer, the raw external queue (a
    previously-requeued, still-tracked item), the persistent verifier
    getter's already-harvested Future, the verifier queue, ``_inflight``,
    and ``_redispatch`` — then calls ``stop()`` directly on a bare (never
    ``run()``) worker, which exercises the drain logic deterministically
    without needing the merger/verifier loops actually running.

    RED until step-12 GREEN wires ``_retire_item`` calls into each of
    ``stop()``'s drain sites.
    """

    async def test_stop_retires_items_in_every_container(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import (
            InflightEntry,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # 1. Lane-buffered request.
        req_lb = _make_request('kappa-stop-lb', 'kappa-stop-lb', tmp_path, config)
        worker._register_item(req_lb, initial=ItemLifecycleState.LANE_BUFFERED)
        worker._lane_buffers['normal'].append(req_lb)

        # 2. Raw-queue request — simulates a previously-requeued, still-
        # tracked item sitting in the external queue at shutdown time (NOT
        # the pre-registry producer-boundary case; this one IS registered).
        req_q = _make_request('kappa-stop-q', 'kappa-stop-q', tmp_path, config)
        worker._register_item(req_q, initial=ItemLifecycleState.QUEUED)
        worker._queue.put_nowait(req_q)

        # 3. Harvested-but-unprocessed item sitting in the persistent
        # verifier getter's already-done Future.
        req_pvg = _make_request('kappa-stop-pvg', 'kappa-stop-pvg', tmp_path, config)
        item_pvg = RealMergeItem(
            request=req_pvg, merge_result=MagicMock(), merge_wt=tmp_path / 'wt_pvg',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item_pvg, initial=ItemLifecycleState.AWAITING_VERIFY)
        pvg_future: asyncio.Future = asyncio.get_running_loop().create_future()
        pvg_future.set_result(item_pvg)
        # The production attr is typed asyncio.Task | None; this test seeds a
        # done Future (only the .done()/.result() Future API is exercised).
        worker._pending_verifier_get = cast(asyncio.Task, pvg_future)

        # 4. Verifier-queue item.
        req_vq = _make_request('kappa-stop-vq', 'kappa-stop-vq', tmp_path, config)
        item_vq = RealMergeItem(
            request=req_vq, merge_result=MagicMock(), merge_wt=tmp_path / 'wt_vq',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item_vq, initial=ItemLifecycleState.AWAITING_VERIFY)
        worker._verifier_queue.put_nowait(item_vq)

        # 5. In-flight (verifying) entry — seeded directly (not via
        # _inflight_append, which itself forces a DISPATCHING -> VERIFYING
        # move unconditionally; this item is already AT VERIFYING).
        req_infl = _make_request('kappa-stop-infl', 'kappa-stop-infl', tmp_path, config)
        item_infl = RealMergeItem(
            request=req_infl, merge_result=MagicMock(), merge_wt=tmp_path / 'wt_infl',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item_infl, initial=ItemLifecycleState.VERIFYING)
        entry_infl = InflightEntry(
            item=item_infl, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
        )
        worker._inflight.append(entry_infl)

        # 6. Redispatch-parked item.
        req_rd = _make_request('kappa-stop-rd', 'kappa-stop-rd', tmp_path, config)
        item_rd = RealMergeItem(
            request=req_rd, merge_result=MagicMock(), merge_wt=tmp_path / 'wt_rd',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item_rd, initial=ItemLifecycleState.REDISPATCH_PARKED)
        worker._redispatch.append(item_rd)

        all_reqs = [req_lb, req_q, req_pvg, req_vq, req_infl, req_rd]

        await worker.stop()

        for req in all_reqs:
            assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL, (
                f'{req.task_id} ({req.request_id}) did not retire to TERMINAL: '
                f'{worker._lifecycle.current(req.request_id)!r}'
            )
            assert req.request_id not in worker._live_items, (
                f'{req.task_id} ({req.request_id}) still present in _live_items after stop()'
            )
            assert req.result.done(), f'{req.task_id}: Future must be resolved by stop()'


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN (continued): the two abandon/detach-drop sites
# — the merger's own pre-merge drop-on-detect, and _dispatch_item's own
# pre-dispatch abandon branch — must retire the registry even though neither
# ever resolves a Future (the waiter already cancelled it).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAbandonPredispatchRetiresRegistry:
    """Both abandon/detach-drop sites drop a request whose waiter already
    cancelled its own Future — no outcome is ever delivered — so the ONLY
    observable sign the item is gone is the registry retiring it (task 2169
    step-11).

    RED until step-12 GREEN adds a ``_retire_item`` call at each site.
    """

    async def test_merger_side_predispatch_abandon_retires_registry(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The merger's own drop-on-detect (right after dequeue, before any
        git work) — a request whose Future was cancelled by the waiter
        before the merger reached it — is currently MERGING and is silently
        dropped (`continue`) without ever resolving a Future.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-merger-abandon', 'file_kappa_ma.py', 'a = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-merger-abandon', 'kappa-merger-abandon', wt, config)
        req.result.cancel()

        retired = asyncio.Event()
        original_note_transition = worker._note_transition

        def _spy_note_transition(rid: str, from_state: Any, to_state: Any, **kwargs: Any) -> None:
            original_note_transition(rid, from_state, to_state, **kwargs)
            if rid == req.request_id and to_state == ItemLifecycleState.TERMINAL:
                retired.set()

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())
        await queue.put(req)
        await asyncio.wait_for(retired.wait(), timeout=10)

        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items
        assert req.result.cancelled(), 'the abandon drop must never overwrite the cancellation'

        await worker.stop()
        await worker_task

    async def test_dispatch_item_predispatch_abandon_retires_registry(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """``_dispatch_item``'s own pre-dispatch abandon branch — the waiter
        cancelled the Future before the item reached a host — is currently
        DISPATCHING and returns an ABANDONED_PREDISPATCH passthrough entry
        with the Future already done (cancelled).
        """
        from orchestrator.merge_queue import (
            InflightStatus,
            ItemLifecycleState,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        req = _make_request(
            'kappa-dispatch-abandon', 'kappa-dispatch-abandon', tmp_path, config,
        )
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        worker._register_item(item, initial=ItemLifecycleState.DISPATCHING)
        req.result.cancel()

        entry = await worker._dispatch_item(item)

        assert entry is not None and entry.status == InflightStatus.ABANDONED_PREDISPATCH
        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN (continued): coalesce-superseded — every
# absorbed single's Future resolves 'superseded', and each one's OWN
# request_id (never the new train's) must retire.
# ---------------------------------------------------------------------------


def _stub_train_callback_factory(train_id: str):  # noqa: ARG001
    """Minimal TrainCallbacks stub (duplicated from
    test_merge_queue_coalesce.py's _stub_factory — per-file duplication
    convention, see this file's module docstring)."""
    from orchestrator.merge_queue import TrainCallbacks

    return TrainCallbacks(
        status_check=AsyncMock(return_value={}),
        mark_member_done=AsyncMock(),
    )


@pytest.mark.asyncio
class TestCoalesceSupersededRetiresRegistry:
    """``_maybe_coalesce_waiting_singles``'s "RESOLVE absorbed futures" loop
    resolves each absorbed single's Future with a 'superseded' outcome — a
    genuine terminal exit for that request_id (the new GroupMergeRequest is
    a BRAND NEW request_id, already registered separately at LANE_BUFFERED
    by step-4) — so each absorbed member must retire too (task 2169
    step-11).

    RED until step-12 GREEN adds a ``_retire_item`` call in that loop.
    """

    async def test_coalesce_superseded_members_retire_registry(
        self, git_repo: Path, git_config: GitConfig, git_ops: GitOps,
    ) -> None:
        from orchestrator.config import OrchestratorConfig
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            ItemLifecycleState,
            SpeculativeMergeWorker,
        )

        coalesce_cfg = OrchestratorConfig(
            project_root=git_repo, git=git_config, merge_train_coalesce_enabled=True,
        )

        wt1 = await _make_branch_with_file(git_ops, 'kc1', 'file_kc1.py', 'a = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'kc2', 'file_kc2.py', 'b = 1\n')
        wt3 = await _make_branch_with_file(git_ops, 'kc3', 'file_kc3.py', 'c = 1\n')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, train_callback_factory=_stub_train_callback_factory,
        )

        req1 = _make_request('kc1', 'kc1', wt1, coalesce_cfg)
        req2 = _make_request('kc2', 'kc2', wt2, coalesce_cfg)
        req3 = _make_request('kc3', 'kc3', wt3, coalesce_cfg)
        for req in (req1, req2, req3):
            worker._register_item(req, initial=ItemLifecycleState.LANE_BUFFERED)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'expected the 3 disjoint singles to coalesce into one train'
        buf = list(worker._lane_buffers['normal'])
        assert len(buf) == 1 and isinstance(buf[0], GroupMergeRequest)
        group_req = buf[0]

        for req in (req1, req2, req3):
            assert req.result.done() and req.result.result().status == 'superseded'
            assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL, (
                f'{req.task_id} did not retire after being absorbed into the train'
            )
            assert req.request_id not in worker._live_items

        # The new train's OWN request_id is a DIFFERENT registry entry —
        # already registered at LANE_BUFFERED (step-4), untouched by the
        # absorbed members' retirement.
        assert worker._lifecycle.current(group_req.request_id) == ItemLifecycleState.LANE_BUFFERED


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN (continued): auto-chain parent-superseded —
# ALREADY covered by the existing _resolve_or_drop_abandoned chokepoint
# (step-10); this test is a completeness proof, not new wiring.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAutoChainParentSupersededRetiresRegistry:
    """The auto-chain generation path (``_maybe_auto_chain_generation``,
    reached via ``_finalize_advanced_merge``'s equivalence-gate
    ``on_blocked`` hook) resolves the ORIGINAL (parent) request's Future
    with a 'superseded' outcome while a gen-(n+1) successor is enqueued
    under a NEW request_id. The parent's OWN request_id must retire —
    proven here by forcing ``_finalize_advanced_merge`` to return
    'superseded' directly rather than driving the full git-tip-advance
    auto-chain machinery (task 2169 step-11).

    Unlike the other tests in this module, this one is a REGRESSION proof,
    not a RED-until-step-12 test: the parent's retirement already flows
    through the SAME ``_resolve_or_drop_abandoned`` chokepoint step-10
    wired for the FAIL/'done' branches (merge_queue.py's CAS loop calls it
    unconditionally on every ``result == 'advanced'`` outcome, regardless
    of the finalize outcome's status) — this test documents and locks that
    in.
    """

    async def test_auto_chain_parent_superseded_retires_registry(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import (
            InflightEntry,
            ItemLifecycleState,
            MergeOutcome,
            RealMergeItem,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_runner import HostLease

        branch = 'kappa-auto-chain-parent'
        wt = await _make_branch_with_file(git_ops, branch, 'kappa_acp.py', 'x = 1\n')
        req = _make_request(branch, branch, wt, config)
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        assert merge_result.merge_worktree is not None

        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req, merge_result=merge_result, merge_wt=merge_result.merge_worktree,
            base_sha=base_sha, speculative=False,
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        mock_allocator = MagicMock()
        mock_allocator.release = AsyncMock()
        mock_allocator.cancel_and_release = AsyncMock()
        worker._host_allocator = mock_allocator
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._register_item(item, initial=ItemLifecycleState.VERIFYING)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = InflightEntry(
            item=item, lease=lease, verify_task=None,
            merge_wt=item.merge_wt, was_speculative=False,
        )

        superseded_outcome = MergeOutcome('superseded', superseded_by='mr-fake-gen2')
        finalize_mock = AsyncMock(return_value=superseded_outcome)

        with (
            patch('orchestrator.merge_queue._finalize_advanced_merge', finalize_mock),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            advanced = await worker._finalize_inflight(entry)

        assert advanced is True, (
            "main WAS advanced (the CAS succeeded); 'superseded' is the "
            "PARENT's own outcome, not a failure to advance"
        )
        finalize_mock.assert_awaited_once()
        assert req.result.done()
        assert req.result.result() is superseded_outcome
        assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL
        assert req.request_id not in worker._live_items


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN (continued): the registry's non-terminal set
# must agree with snapshot()['entries'] at several sampling points — proving
# the wiring is exhaustive before steps 13+ repoint snapshot() onto the
# registry directly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRegistrySnapshotAgreementAtSamplingPoints:
    """At each of several deliberate mid-pipeline sampling points, the SET
    of request_ids the registry considers non-terminal must exactly equal
    the SET of request_ids visible in ``snapshot()['entries']`` — proving
    kappa's wiring gives snapshot()/the registry a single, agreeing census
    (task 2169 step-11). Reuses the exact gates already proven reliable by
    the step-3/step-7/step-9 ``_live_items``-agreement tests above (gate on
    ``get_main_sha``/``advance_main``), extended here to compare full SETS
    rather than a single item's identity. Single-item drives (not
    concurrent multi-item) so there is no ambiguity from the producer-
    boundary asymmetry (an undrained raw-``_queue`` item would legitimately
    show up in snapshot() before it is registered).

    The DISPATCHING window WAS the one exception: ``_dispatching_item`` was
    documented as census-only, deliberately absent from
    ``snapshot()['entries']`` (task 2068) — a pre-existing gap outside kappa
    step-11/12's scope. Task 2435 (kappa-b) closes it by repointing
    snapshot()'s registry-sourced entry loop to surface DISPATCHING, so the
    DISPATCHING sample below now asserts equality too, exactly like every
    other sampling point in this class.
    """

    async def test_agreement_at_merging_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-sample-merging', 'file_ksm.py', 'm = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-sample-merging', 'kappa-sample-merging', wt, config)

        captured: list[tuple[set, set]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            # Gate on the registry (task 2435 kappa-b: formerly
            # worker._inflight_req is not None).
            if (
                worker._lifecycle.current(req.request_id) == ItemLifecycleState.MERGING
                and not fired
            ):
                fired = True
                snap_ids = {e['request_id'] for e in worker.snapshot()['entries']}
                registry_ids = {
                    rid for rid, st in worker._lifecycle._states.items()
                    if st != ItemLifecycleState.TERMINAL
                }
                captured.append((snap_ids, registry_ids))
            return await original_get_main_sha()

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'get_main_sha', new=_spying_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert fired, 'expected the get_main_sha gate to fire while MERGING'
        snap_ids, registry_ids = captured[0]
        assert snap_ids == registry_ids, (
            f'registry/snapshot disagree at MERGING sample: '
            f'snapshot only={snap_ids - registry_ids}, registry only={registry_ids - snap_ids}'
        )
        assert req.request_id in registry_ids

        await worker.stop()
        await worker_task

    async def test_dispatching_window_registry_equals_snapshot(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The DISPATCHING window now agrees exactly between the registry and
        snapshot() (task 2435 kappa-b): snapshot()'s registry-sourced entry
        loop now surfaces DISPATCHING directly, closing the task-2068 census
        gap that used to make this sampling point a strict-superset
        exception. Mirrors the equality idiom already used by
        ``test_agreement_at_merging_window``/``test_agreement_at_finalizing_window``
        above.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-sample-dispatching', 'file_ksd.py', 'd = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request(
            'kappa-sample-dispatching', 'kappa-sample-dispatching', wt, config,
        )

        captured: list[tuple[set, set]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            # Gate on the registry (task 2435 kappa-b: formerly
            # worker._dispatching_item is not None).
            if (
                worker._lifecycle.current(req.request_id) == ItemLifecycleState.DISPATCHING
                and not fired
            ):
                fired = True
                snap_ids = {e['request_id'] for e in worker.snapshot()['entries']}
                registry_ids = {
                    rid for rid, st in worker._lifecycle._states.items()
                    if st != ItemLifecycleState.TERMINAL
                }
                captured.append((snap_ids, registry_ids))
            return await original_get_main_sha()

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'get_main_sha', new=_spying_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert fired, 'expected the get_main_sha gate to fire while DISPATCHING'
        snap_ids, registry_ids = captured[0]
        assert req.request_id in registry_ids
        assert snap_ids == registry_ids, (
            f'registry/snapshot disagree at DISPATCHING sample: '
            f'snapshot only={snap_ids - registry_ids}, registry only={registry_ids - snap_ids}'
        )

        await worker.stop()
        await worker_task

    async def test_agreement_at_finalizing_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-sample-finalizing', 'file_ksf.py', 'f = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request(
            'kappa-sample-finalizing', 'kappa-sample-finalizing', wt, config,
        )

        captured: list[tuple[set, set]] = []
        original_advance = git_ops.advance_main

        async def _capturing_advance(*args: Any, **kwargs: Any) -> Any:
            snap_ids = {e['request_id'] for e in worker.snapshot()['entries']}
            registry_ids = {
                rid for rid, st in worker._lifecycle._states.items()
                if st != ItemLifecycleState.TERMINAL
            }
            captured.append((snap_ids, registry_ids))
            return await original_advance(*args, **kwargs)

        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'advance_main', new=_capturing_advance),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)
            assert outcome.status == 'done', f'{outcome}'

        assert len(captured) >= 1, 'advance_main must have been called at least once'
        snap_ids, registry_ids = captured[0]
        assert snap_ids == registry_ids, (
            f'registry/snapshot disagree at FINALIZING sample: '
            f'snapshot only={snap_ids - registry_ids}, registry only={registry_ids - snap_ids}'
        )
        assert req.request_id in registry_ids

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN (continued): a full multi-item pipeline run to
# quiescence — every dequeued request_id ends at TERMINAL, none leaked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMultiItemPipelineToQuiescenceNoLeaks:
    """Driving several concurrent requests through the full happy-path
    pipeline to completion must retire EVERY one of them — no leaked
    registry entries once the pipeline goes quiet (task 2169 step-11).
    """

    async def test_three_items_all_reach_terminal_no_leaks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wts = [
            await _make_branch_with_file(
                git_ops, f'kappa-quiesce-{i}', f'file_kq{i}.py', f'{chr(97 + i)} = {i}\n',
            )
            for i in range(3)
        ]
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        reqs = [
            _make_request(f'kappa-quiesce-{i}', f'kappa-quiesce-{i}', wts[i], config)
            for i in range(3)
        ]

        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            for req in reqs:
                await queue.put(req)
            outcomes = await asyncio.gather(
                *(asyncio.wait_for(req.result, timeout=60) for req in reqs)
            )

        assert all(o.status == 'done' for o in outcomes), f'{outcomes}'

        for req in reqs:
            assert worker._lifecycle.current(req.request_id) == ItemLifecycleState.TERMINAL, (
                f'{req.task_id} leaked at '
                f'{worker._lifecycle.current(req.request_id)!r} instead of TERMINAL'
            )
            assert req.request_id not in worker._live_items

        snap_ids_after = {e['request_id'] for e in worker.snapshot()['entries']}
        assert snap_ids_after.isdisjoint({r.request_id for r in reqs}), (
            'a completed request must not still appear in snapshot() entries'
        )

        await worker.stop()
        await worker_task
