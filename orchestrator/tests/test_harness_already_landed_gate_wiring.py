"""Tests for the harness's already-landed pre-dispatch gate (task 2313).

Architecturally parallel to task 2156's landed-outbox dispatch gate
(test_harness_landed_dispatch_gate_wiring.py), but this gate consults LIVE
GIT STATE (ancestry + content-equivalence) rather than the durable
LandedOutbox, so it also catches OUT-OF-BAND landings that never went
through this orchestrator's own merge queue: a sibling direct-merge, a prior
orchestrator run, or a squash/rebase/manual landing.

Covers:
  step-3  (RED)  Ancestry happy-path: is_ancestor True + citation present +
                 not degenerate -> flips to done, returns True.

Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``
bare-harness construction helper exactly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryHappyPath:
    """Ancestry happy-path: is_ancestor True + citation present + not degenerate."""

    async def test_ancestry_with_citation_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """RED until step-4 adds Harness._already_landed_dispatch_gate.

        is_ancestor True, a citation commit is present on main, and the
        branch is not degenerate -> the gate drives the task to done via
        _mark_in_progress_done (anchored on the citation sha) and returns
        True so the scheduler withholds dispatch this tick.
        """
        h = _build_harness(mock_orch_config)

        citation_sha = 'a' * 40
        h.git_ops = MagicMock()
        h.git_ops.is_ancestor = AsyncMock(return_value=True)
        h.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
        h.git_ops.config.branch_prefix = 'task/'
        h.git_ops.config.main_branch = 'main'

        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {}},
        )
        h._branch_is_degenerate = AsyncMock(return_value=False)
        h._mark_in_progress_done = AsyncMock()
        h._escalation_queue = None

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        h._mark_in_progress_done.assert_awaited_once()
        call_args = h._mark_in_progress_done.await_args
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-already-on-main'


def _wired_ancestry_harness(mock_orch_config) -> Harness:
    """Bare harness pre-wired so the ancestry path would flip on its own —
    each guard test overrides exactly one attribute to trip its veto.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.is_ancestor = AsyncMock(return_value=True)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value='a' * 40)
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryGuards:
    """Ancestry-path false-positive guards must veto the flip (RED until step-6).

    Each sub-case starts from an otherwise-flipping ancestry setup
    (is_ancestor True, citation present, not degenerate, no open L1) and
    trips exactly one guard — the gate must return False and must NOT
    call _mark_in_progress_done.
    """

    async def test_open_l1_vetoes_flip(self, mock_orch_config) -> None:
        """An open L1 escalation is a deliberate human handoff — never
        second-guessed, even though is_ancestor is True.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=True)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        h._mark_in_progress_done.assert_not_awaited()

    async def test_degenerate_branch_vetoes_flip(self, mock_orch_config) -> None:
        """A degenerate branch (tip == branch_base_sha) carries zero task
        work — is_ancestor==True is a false 'already on main' signal.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._branch_is_degenerate = AsyncMock(return_value=True)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        h._mark_in_progress_done.assert_not_awaited()

    async def test_missing_citation_vetoes_flip(self, mock_orch_config) -> None:
        """No commit on main cites this task — reject the zero-commit-branch
        shape where is_ancestor returns True trivially but no work landed.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        h._mark_in_progress_done.assert_not_awaited()
