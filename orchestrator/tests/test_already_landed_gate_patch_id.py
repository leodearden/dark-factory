"""Real-git consumer suite: the already-landed dispatch gate decides by patch-id (task 4648).

``Harness._already_landed_dispatch_gate`` re-derives a pending task's landing
from live git.  Its LIVE-branch arms decide through
``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::branch_work_landed``,
the non-decaying producer, so the boundary rows that producer is defined
against must hold one layer up, at the gate:

* **B1** a landing churned by later same-path commits on main is still stamped.
* **B2** a sync-merge tip is stamped without ever computing a main-history diff.
* **B4** a no-op landing and **B6** a degenerate branch are refused, and
  recorded as ``landing_evidence_rejected`` rows at the ancestry site.

Every repository is a genuine throwaway one built by ``_landing_git_scenarios``;
the gate runs over the REAL ``GitOps`` ``Harness.__init__`` builds for it.  Only
the task read, the done-stamp and the escalation queue are stand-ins, so what
the gate decides is what git says.  Each row carries ``method == 'patch_id'``,
the per-site pin of which landing policy decided (the site-to-policy map is
``docs/prds/landed-not-done-recovery.effect-present-caller-enumeration.md`` §7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _landing_evidence_helpers import landing_reject_rows
from _landing_git_scenarios import (
    TASK_ID,
    WORK_PATH,
    LandingRepo,
    Scenario,
    advance_main_touching_same_paths,
    build_degenerate_branch,
    build_merged_landing,
    build_no_op_landing,
    build_sync_merge_branch,
    numbered_lines,
)

from orchestrator.config import DeliveredChecksConfig, RecoveryEmissionConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness

pytestmark = pytest.mark.asyncio

#: The main-history predicates that decay under later commits on main.  A
#: live-branch arm deciding by patch-id never awaits any of them.
DECAYING_PREDICATES = (
    'branch_content_in_main',
    'commit_effect_present_in_main',
    'describe_commit_effect_in_main',
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A temporary git repository with an initial commit on ``main``."""
    root = tmp_path / 'repo'
    root.mkdir()
    repo = LandingRepo.init(root)
    repo.git('config', 'user.email', 'test@test.com')
    repo.git('config', 'user.name', 'Test')
    repo.commit('chore: initial commit', {'README.md': '# Test\n'})
    return root


@pytest.fixture
def repo(git_repo: Path) -> LandingRepo:
    return LandingRepo(git_repo)


class WiredGate:
    """The dispatch gate over a real ``GitOps``, its effects recorded.

    ``Harness.__init__`` builds the ``GitOps`` from ``config.project_root``, so
    pointing that at the scenario's repository is what makes every git call the
    gate makes a real one.  The done-stamp is an ``AsyncMock`` held here, the
    escalation queue a recording ``MagicMock`` stating "no pending records",
    and the event store a real one, read back through
    :func:`landing_reject_rows`.
    """

    def __init__(self, mock_orch_config: Any, sc: Scenario, tmp_path: Path) -> None:
        mock_orch_config.project_root = sc.repo.root
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'
        mock_orch_config.recovery_emission = RecoveryEmissionConfig()
        mock_orch_config.delivered_checks = DeliveredChecksConfig()
        with patch('orchestrator.harness.McpLifecycle'), \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            self.harness = Harness(mock_orch_config)
        h: Any = self.harness
        h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': TASK_ID, 'metadata': sc.metadata},
        )
        self.mark = AsyncMock()
        h._mark_in_progress_done = self.mark
        self.queue = MagicMock()
        self.queue.get_by_task = MagicMock(return_value=[])
        self.queue.has_open_l1 = MagicMock(return_value=False)
        self.queue.find_terminal_by_citation = MagicMock(return_value=None)
        self.queue.make_id = MagicMock(return_value=f'esc-{TASK_ID}-1')
        h._escalation_queue = self.queue

    async def run(self) -> bool:
        return await self.harness._already_landed_dispatch_gate(TASK_ID)

    def rejects(self) -> list[dict[str, Any]]:
        """Every ``landing_evidence_rejected`` row's ``data``, oldest first."""
        return [row['data'] for row in landing_reject_rows(self.harness.event_store)]

    def spy(self, monkeypatch: pytest.MonkeyPatch, name: str) -> AsyncMock:
        """Wrap the real ``GitOps.<name>`` so its awaits are counted."""
        git_ops = self.harness.git_ops
        recorder = AsyncMock(wraps=getattr(git_ops, name))
        monkeypatch.setattr(git_ops, name, recorder)
        return recorder


class TestAncestryArmDecidesByPatchId:
    """The arm taken when the live branch tip is an ancestor of main."""

    async def test_b1_a_churned_landing_is_still_stamped(
        self, mock_orch_config: Any, repo: LandingRepo, tmp_path: Path,
    ) -> None:
        """Three same-path churn commits, the last rewriting the task's lines.

        That churn drops line survival below the effect-present thresholds, so
        a decaying predicate would refuse; patch-id containment still holds.
        """
        sc = build_merged_landing(repo)
        advance_main_touching_same_paths(sc, 3, rewrite=True)
        gate = WiredGate(mock_orch_config, sc, tmp_path)

        assert await gate.run() is True

        gate.mark.assert_awaited_once_with(
            TASK_ID, sc.landed_sha, ANY, 'dispatch-gate-already-on-main',
        )
        assert gate.rejects() == []
        gate.queue.submit.assert_not_called()

    async def test_b2_a_sync_merge_tip_is_stamped_without_a_main_history_diff(
        self, mock_orch_config: Any, repo: LandingRepo, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sc = build_sync_merge_branch(repo)
        repo.commit(
            'chore: main touches the path it merged into the branch',
            {WORK_PATH: numbered_lines('feature', 40) + numbered_lines('mainside', 6)
             + numbered_lines('later', 3)},
        )
        sc.refresh()
        gate = WiredGate(mock_orch_config, sc, tmp_path)
        spies = {name: gate.spy(monkeypatch, name) for name in DECAYING_PREDICATES}

        assert await gate.run() is True

        gate.mark.assert_awaited_once_with(
            TASK_ID, sc.landed_sha, ANY, 'dispatch-gate-already-on-main',
        )
        assert {name: spy.await_count for name, spy in spies.items()} == dict.fromkeys(
            DECAYING_PREDICATES, 0,
        )

    async def test_b4_a_no_op_landing_is_refused_and_recorded(
        self, mock_orch_config: Any, repo: LandingRepo, tmp_path: Path,
    ) -> None:
        sc = build_no_op_landing(repo)
        gate = WiredGate(mock_orch_config, sc, tmp_path)

        assert await gate.run() is False

        gate.mark.assert_not_awaited()
        [row] = gate.rejects()
        assert (row['site'], row['reason'], row['method']) == (
            'already_landed_gate_ancestry', 'no_op_landing', 'patch_id',
        )
        gate.queue.submit.assert_not_called()

    async def test_b6_a_degenerate_branch_is_refused_and_recorded(
        self, mock_orch_config: Any, repo: LandingRepo, tmp_path: Path,
    ) -> None:
        sc = build_degenerate_branch(repo)
        gate = WiredGate(mock_orch_config, sc, tmp_path)

        assert await gate.run() is False

        gate.mark.assert_not_awaited()
        [row] = gate.rejects()
        assert (row['site'], row['reason'], row['method']) == (
            'already_landed_gate_ancestry', 'degenerate_branch', 'patch_id',
        )
        gate.queue.submit.assert_not_called()

    @pytest.mark.parametrize('build', [build_no_op_landing, build_degenerate_branch])
    async def test_every_row_this_arm_writes_names_the_patch_id_policy(
        self, mock_orch_config: Any, repo: LandingRepo, tmp_path: Path, build: Any,
    ) -> None:
        """The per-site policy pin: the ancestry arm's verdicts come from ``branch_work_landed``."""
        gate = WiredGate(mock_orch_config, build(repo), tmp_path)

        await gate.run()

        rows = gate.rejects()
        assert rows
        assert {row['method'] for row in rows} == {'patch_id'}
