"""A task's live tier and pin as a never-raising lookup (task 9908).

``Scheduler.task_priority`` answers from the scheduler's own memory;
``read_task_priority`` is the one guard every consumer reads it through.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_verify_facts import ChainMembers, MergeVerifyDispatchFacts, RunnerReason
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import Scheduler
from orchestrator.task_priority import UNKNOWN_PRIORITY, TaskPriority, read_task_priority


def _pending(task_id: str, priority: str, *, deps: tuple[str, ...] = ()) -> dict:
    return {
        'id': task_id, 'title': f'Task {task_id}', 'status': 'pending',
        'dependencies': [{'id': d} for d in deps],
        'metadata': {'files': [f'{task_id}/src']}, 'priority': priority,
    }


def _scheduler(tmp_path: Path, *, with_store: bool) -> tuple[Scheduler, OverrideStore | None]:
    config = OrchestratorConfig(max_per_module=1, project_root=tmp_path)
    store = OverrideStore(tmp_path / 'overrides.db') if with_store else None
    scheduler = Scheduler(
        config, override_store=store, state_snapshot_path=tmp_path / 'scheduler_state.json',
    )
    scheduler.finish_startup()
    return scheduler, store


class TestSchedulerTaskPriority:

    def test_everything_is_unknown_before_the_first_tick(self, tmp_path: Path) -> None:
        scheduler, store = _scheduler(tmp_path, with_store=True)
        assert store is not None
        store.set_override(str(tmp_path), 'P', pinned=True)

        assert scheduler.task_priority('P') == TaskPriority(tier=None, pinned=None)

    @pytest.mark.asyncio
    async def test_a_tick_supplies_the_effective_tier_and_the_pin(self, tmp_path: Path) -> None:
        scheduler, store = _scheduler(tmp_path, with_store=True)
        assert store is not None
        store.set_override(str(tmp_path), 'P', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[
            _pending('A', 'medium'), _pending('B', 'high', deps=('A',)), _pending('P', 'low'),
        ])

        await scheduler.acquire_next()

        assert scheduler.task_priority('A') == TaskPriority(tier='high', pinned=False)
        assert scheduler.task_priority('P') == TaskPriority(tier='low', pinned=True)
        assert scheduler.task_priority('unranked') == TaskPriority(tier=None, pinned=False)
        assert scheduler.get_state_snapshot().effective_priorities['A'] == 'high'

    @pytest.mark.asyncio
    async def test_no_override_store_leaves_the_pin_unknown(self, tmp_path: Path) -> None:
        scheduler, _ = _scheduler(tmp_path, with_store=False)
        scheduler.get_tasks = AsyncMock(return_value=[_pending('A', 'medium')])

        await scheduler.acquire_next()

        assert scheduler.task_priority('A') == TaskPriority(tier='medium', pinned=None)


class TestReadTaskPriority:

    def test_no_lookup_is_unknown(self) -> None:
        assert read_task_priority(None, '7') is UNKNOWN_PRIORITY

    def test_a_raising_lookup_is_unknown(self) -> None:
        def broken(task_id: str) -> TaskPriority:
            raise KeyError(task_id)

        assert read_task_priority(broken, '7') is UNKNOWN_PRIORITY

    def test_an_answer_of_the_wrong_type_is_unknown(self) -> None:
        assert read_task_priority(MagicMock(), '7') is UNKNOWN_PRIORITY

    def test_a_field_of_the_wrong_type_is_unknown(self) -> None:
        def odd(task_id: str) -> TaskPriority:
            return TaskPriority(tier=MagicMock(), pinned='yes')  # type: ignore[arg-type]

        assert read_task_priority(odd, '7') == UNKNOWN_PRIORITY

    def test_the_task_id_reaches_the_lookup_as_a_string(self) -> None:
        asked: list[object] = []

        def recording(task_id: str) -> TaskPriority:
            asked.append(task_id)
            return TaskPriority(tier='high', pinned=True)

        assert read_task_priority(recording, 42) == TaskPriority(tier='high', pinned=True)
        assert asked == ['42']


class TestChainMembersOnTheVerifyRow:

    def test_members_add_both_lists_in_order(self) -> None:
        facts = MergeVerifyDispatchFacts(runner_reason=RunnerReason.REMOTE_BUSY).covering(
            ChainMembers(('mr-a', 'mr-b'), ('1', '2')),
        )

        assert facts.event_fields() == {
            'runner_reason': 'remote_busy', 'base_relation': None,
            'chain_request_ids': ['mr-a', 'mr-b'], 'chain_task_ids': ['1', '2'],
        }

    def test_a_retry_covers_the_same_members(self) -> None:
        members = ChainMembers(('mr-a',), ('1',))

        assert MergeVerifyDispatchFacts().covering(members).as_retry().members == members
