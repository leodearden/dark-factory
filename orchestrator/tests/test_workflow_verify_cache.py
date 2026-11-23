"""The task role records its verify's greens where its workflow_verify row is emitted, into the cache access its workflow was given (task 5653)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _task_workflow_scene import (
    SCENE_TASK_ID,
    forked_task_workflow,
    scene_git,
    task_scene_config,
    task_workflow,
)
from _verify_cache_fixtures import TEST_CODE
from _verify_cache_units import (
    clean_leg,
    ran_unit,
    recorded_entries,
    unit_decision,
    unit_key,
    verify_result,
)

from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import OrchestratorConfig, VerifyCacheConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.verify import VerifyResult
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import LegRun
from orchestrator.verify_cache.registry import Entry, Registry, registry_db_path
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan
from orchestrator.verify_evidence import Leg, TipUnresolved, UnitTier
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_TASK_LOG = VerifyCacheConfig(task_mode=CacheMode.LOG)
_TASK_UNIT = 'module:alpha:test'


def _task_green() -> tuple[VerifyResult, UnitKey]:
    """A passing task-role verdict whose one test leg ran clean under a keyed, admitted unit."""
    key = unit_key(_TASK_UNIT)
    replay = ReplayPlan(
        role='task', modes=_TASK_LOG,
        tree=unit_decision(Decision.MISS, key=unit_key('tree', tier=UnitTier.TREE)),
        modules={_TASK_UNIT: unit_decision(Decision.MISS, key=key)},
    )
    units, record = replay.settle((ran_unit(_TASK_UNIT),), (LegRun(_TASK_UNIT, clean_leg(Leg.TEST)),))
    return verify_result(units=units, cache_record=record, duration_secs=1.0), key


class TestWorkflowRecordsTaskGreens:
    @staticmethod
    def _access(tmp_path: Path) -> CacheAccess:
        return CacheAccess(registry=Registry(tmp_path / 'verify_cache.db'), orchestrator_code=TEST_CODE)

    @staticmethod
    async def _scene(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, forked: bool = False,
        verify_cache: CacheAccess | None,
    ) -> tuple[OrchestratorConfig, EventStore, TaskWorkflow]:
        config = task_scene_config(tmp_path)
        config.pre_merge_reverify = forked
        store = EventStore(tmp_path / 'events.db', run_id='r')
        scene = forked_task_workflow if forked else task_workflow
        workflow = await scene(
            config, GitOps(config.git, config.project_root), store, monkeypatch, verify_cache=verify_cache,
        )
        monkeypatch.setattr(workflow, '_review', AsyncMock(return_value=ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={'analyst': {}},
        )))
        return config, store, workflow

    @staticmethod
    def _verifies(monkeypatch: pytest.MonkeyPatch, result: VerifyResult) -> AsyncMock:
        verify = AsyncMock(return_value=result)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)
        return verify

    @staticmethod
    async def _verify_at_site(workflow: TaskWorkflow, *, forked: bool = False) -> None:
        """Drive the site the scene was built for: the post-rebase re-verify when *forked*, else the implement loop."""
        if forked:
            assert await workflow._rebase_before_enqueue() is True
        else:
            assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

    @staticmethod
    async def _recorded_in(access: CacheAccess, key: UnitKey) -> Entry:
        assert access.registry is not None
        return (await recorded_entries(access.registry, key))[_TASK_UNIT]

    @pytest.mark.asyncio
    async def test_the_implement_site_records_its_green_against_the_review_edge_row(self, tmp_path, monkeypatch):
        access = self._access(tmp_path)
        _config, store, workflow = await self._scene(tmp_path, monkeypatch, verify_cache=access)
        result, key = _task_green()
        self._verifies(monkeypatch, result)

        await self._verify_at_site(workflow)

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        entry = await self._recorded_in(access, key)
        assert entry.evidence_event_id == row['id']
        assert (entry.run_id, entry.sha) == ('r', row['data']['tip_sha'])

    @pytest.mark.asyncio
    async def test_the_pre_merge_site_records_its_green_and_verifies_through_the_access_it_was_given(
        self, tmp_path, monkeypatch,
    ):
        access = self._access(tmp_path)
        _config, store, workflow = await self._scene(tmp_path, monkeypatch, forked=True, verify_cache=access)
        result, key = _task_green()
        verify = self._verifies(monkeypatch, result)

        await self._verify_at_site(workflow, forked=True)

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        entry = await self._recorded_in(access, key)
        assert entry.evidence_event_id == row['id']
        assert (entry.run_id, entry.sha) == ('r', row['data']['tip_sha'])
        assert verify.await_args is not None
        assert verify.await_args.kwargs['verify_cache'] is access

    @pytest.mark.asyncio
    @pytest.mark.parametrize('forked', [False, True], ids=['implement-site', 'pre-merge-site'])
    async def test_a_workflow_outside_the_cache_records_nothing(self, tmp_path, monkeypatch, forked):
        config, store, workflow = await self._scene(tmp_path, monkeypatch, forked=forked, verify_cache=None)
        self._verifies(monkeypatch, _task_green()[0])

        await self._verify_at_site(workflow, forked=forked)

        assert len(store.fetch_events_by_type(EventType.workflow_verify)) == 1
        assert not registry_db_path(config.project_root).exists()
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_a_verdict_without_a_cache_record_records_nothing(self, tmp_path, monkeypatch):
        _config, store, workflow = await self._scene(tmp_path, monkeypatch, verify_cache=self._access(tmp_path))
        self._verifies(monkeypatch, verify_result(duration_secs=1.0))

        await self._verify_at_site(workflow)

        assert len(store.fetch_events_by_type(EventType.workflow_verify)) == 1
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_an_unresolved_tip_records_nothing(self, tmp_path, monkeypatch):
        _config, store, workflow = await self._scene(tmp_path, monkeypatch, verify_cache=self._access(tmp_path))
        self._verifies(monkeypatch, _task_green()[0])
        monkeypatch.setattr(workflow, '_verified_tip', AsyncMock(return_value=TipUnresolved.HEAD_UNREADABLE))

        await self._verify_at_site(workflow)

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        assert row['data']['tip_sha_unresolved'] is True
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_a_checkpoint_hit_records_nothing(self, tmp_path, monkeypatch):
        _config, store, workflow = await self._scene(tmp_path, monkeypatch, verify_cache=self._access(tmp_path))
        assert workflow.worktree is not None
        head = scene_git(workflow.worktree, 'rev-parse', 'HEAD')
        EventStore(tmp_path / 'events.db', run_id='prior').emit(
            EventType.workflow_verify, task_id=SCENE_TASK_ID, data={'passed': True, 'tip_sha': head},
        )
        verify = self._verifies(monkeypatch, _task_green()[0])

        await self._verify_at_site(workflow)

        verify.assert_not_awaited()
        assert store.fetch_events_by_type(EventType.workflow_verify) == []
        assert not (tmp_path / 'verify_cache.db').exists()
