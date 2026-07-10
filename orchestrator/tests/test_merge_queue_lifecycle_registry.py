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
