"""Tests making the two hardcoded-from_state ItemLifecycle hops
(``_buffer_owned_request``'s QUEUED->LANE_BUFFERED and
``_finalize_inflight``'s VERIFYING->FINALIZING) re-entry-tolerant, so a
duplicate/re-entrant submission coalesces as a benign no-op instead of
firing a ``merge_lifecycle_transition_rejected`` escalation (task 2852;
SHARPER RCA, reify cluster mr-8cafbaf0 / mr-99585bb8 / original incident
mr-c69ca480).

Both offender sites hardcode the transition's ``from_state`` instead of
reading the :class:`~orchestrator.merge_queue.ItemLifecycle` registry's
CURRENT state first (the pattern already used by ``_note_requeue``) — so a
duplicate/re-entrant driver that touches a request_id already deep in the
pipeline gets an illegal-edge rejection (non-corrupting, but persistently
noisy) instead of being recognized and coalesced.

Steps covered:
  step-1  RED   — SpeculativeMergeWorker._advance_if_at fire + tolerant
                  no-op contract
  step-2  GREEN — _advance_if_at added
  step-3  impl  — _finalize_inflight wired through _advance_if_at (no new
                  test here — proven by step-1(b) at the exact seam; normal-
                  path firing regression-covered by test_merge_speculation.py)
  step-4  RED   — _buffer_owned_request twin-drain repro (headline incident)
  step-5  GREEN — _buffer_owned_request restructured + _coalesce_reentrant_drain
                  (drop-only)
  step-6  RED   — fresh/requeue regressions + twin-future-resolved +
                  observability
  step-7  GREEN — _coalesce_reentrant_drain resolves the twin's future +
                  emits a merge_coalesced event

This module imports orchestrator.merge_queue LOCALLY inside each test
method (not at module scope) so a not-yet-implemented symbol (e.g.
``_advance_if_at``, before step-2) never breaks collection of the rest of
the file during the RED steps — mirrors
test_merge_queue_lifecycle_registry.py's convention.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeRequest

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — copied from
# test_merge_queue_lifecycle_registry.py; there is no shared worker conftest).
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
    test_merge_queue_lifecycle_registry.py — per-file duplication
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


def _make_worker(git_ops: GitOps, *, escalation_queue: Any = None, event_store: Any = None):
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_lifecycle_registry.py's ``_make_worker``,
    extended with an optional ``event_store`` passthrough for the
    observability test (step-6(d) / step-7).
    """
    from orchestrator.merge_queue import SpeculativeMergeWorker

    return SpeculativeMergeWorker(
        git_ops, asyncio.Queue(), event_store=event_store, escalation_queue=escalation_queue,
    )


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop.

    Duplicated from test_merge_queue_lifecycle_registry.py (per-file
    duplication convention — see this file's module docstring), extended
    with an optional ``request_id`` override so twin-submission tests can
    force two DISTINCT ``MergeRequest`` objects to share one request_id —
    mirroring the journal-recovery ``reconstruct_merge_request`` path that
    produced the original incident's twin.
    """
    kwargs: dict[str, Any] = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
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
        **kwargs,
    )


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: _advance_if_at shared re-entry-tolerant primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAdvanceIfAt:
    """``SpeculativeMergeWorker._advance_if_at(request_id, from_state,
    to_state, *, live_obj=None) -> bool`` — the shared re-entry-tolerant
    forward-hop primitive (task 2852 step-1/step-2), mirroring
    ``_note_requeue``'s read-``current()``-first discipline: fires the
    transition (via ``_note_transition``) iff the registry's current state
    equals *from_state*; otherwise logs a WARNING and no-ops — NOT an
    escalation.

    RED until step-2 GREEN adds ``_advance_if_at`` — AttributeError on
    current main.
    """

    async def test_fires_when_current_matches_from_state(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        item = _make_request('advance-fire', 'advance-fire', tmp_path, config)
        rid = item.request_id
        worker._register_item(item, initial=ItemLifecycleState.QUEUED)

        result = worker._advance_if_at(
            rid, ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED, live_obj=item,
        )

        assert result is True
        assert worker._lifecycle.current(rid) == ItemLifecycleState.LANE_BUFFERED
        assert worker._live_items[rid] is item
        assert fake_eq.submitted == []

    async def test_tolerant_noop_at_finalize_seam_when_already_advanced(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The exact double-finalize condition (SHARPER RCA mr-99585bb8):
        the registry is already at FINALIZING when ``_finalize_inflight``'s
        hop fires a second time for the same request_id.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        entry = _make_request('advance-noop', 'advance-noop', tmp_path, config)
        rid = entry.request_id
        worker._register_item(entry, initial=ItemLifecycleState.FINALIZING)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._advance_if_at(
                rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING, live_obj=entry,
            )

        assert result is False
        assert worker._lifecycle.current(rid) == ItemLifecycleState.FINALIZING
        assert fake_eq.submitted == [], (
            f'tolerant no-op must NOT escalate: {fake_eq.submitted!r}'
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert rid in warnings[0].message
