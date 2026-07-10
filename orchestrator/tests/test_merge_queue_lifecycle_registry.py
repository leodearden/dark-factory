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

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeRequest

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
        branch=branch,
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
        read MERGING, WHILE the still-present ``_inflight_req`` transient
        field is set to that same request — the two must agree during the
        additive (fields-kept) phase of this task.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

        wt = await _make_branch_with_file(
            git_ops, 'kappa-merging', 'file_kappa_merging.py', 'y = 1\n',
        )
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('kappa-merging', 'kappa-merging', wt, config)

        captured: list[tuple[Any, Any, bool]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            # Gate on worker._inflight_req (mirrors
            # test_merge_queue.py::test_merger_exception_resolves_inflight_future's
            # fault-injection rationale): recompute_suffix_conflict_graph()
            # also calls get_main_sha() before dequeue, when _inflight_req
            # is still None — skip those, and fire only once.
            if worker._inflight_req is not None and not fired:
                fired = True
                captured.append((
                    worker._lifecycle.current(req.request_id),
                    worker._live_items.get(req.request_id),
                    worker._inflight_req is req,
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

        assert fired, 'expected the get_main_sha gate to fire while _inflight_req was set'
        observed_state, observed_live_obj, inflight_matches = captured[0]
        assert observed_state == ItemLifecycleState.MERGING
        assert observed_live_obj is req
        assert inflight_matches is True

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
        # `_inflight_append`. Unlike the loop-breaker/passthrough test below
        # (whose Future resolves early via `_oob_deliver`, racing ahead of the
        # background verifier-side dispatch), this path's assertion window
        # necessarily includes those two additional entries.
        assert observed == [
            (ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED),
            (ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING),
            (ItemLifecycleState.MERGING, ItemLifecycleState.AWAITING_VERIFY),
            (ItemLifecycleState.AWAITING_VERIFY, ItemLifecycleState.DISPATCHING),
            (ItemLifecycleState.DISPATCHING, ItemLifecycleState.VERIFYING),
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
        DISPATCHING -> VERIFYING transition at ``_inflight_append``.
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
        ], f'unexpected transition sequence for {req.request_id}: {observed!r}'

        await worker.stop()
        await worker_task

    async def test_live_items_holds_item_during_dispatching_window(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """``_live_items`` must hold the SpeculativeItem, and the registry
        must read DISPATCHING, WHILE the still-present ``_dispatching_item``
        transient field is set to that same item — the two must agree
        during the additive (fields-kept) phase of this task.
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

        captured: list[tuple[Any, Any, bool]] = []
        original_get_main_sha = git_ops.get_main_sha
        fired = False

        async def _spying_get_main_sha() -> str:
            nonlocal fired
            if worker._dispatching_item is not None and not fired:
                fired = True
                captured.append((
                    worker._lifecycle.current(req.request_id),
                    worker._live_items.get(req.request_id),
                    worker._dispatching_item is worker._live_items.get(req.request_id),
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

        assert fired, 'expected the get_main_sha gate to fire while _dispatching_item was set'
        observed_state, observed_live_obj, matches = captured[0]
        assert observed_state == ItemLifecycleState.DISPATCHING
        assert isinstance(observed_live_obj, RealMergeItem)
        assert matches is True

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
