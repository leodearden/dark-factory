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

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run

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
