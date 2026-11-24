"""Shared fixtures for driving the real ``Harness.run`` startup table in a test.

Imported by bare module name (``from _startup_harness import ...``), matching
the flat-test-helper convention of ``_orch_helpers.py`` (orchestrator/tests/ has
no ``__init__.py``).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from _git_fixtures import seed_repo
from _merge_lane_fakes import branch_merge_request, lane_scene_config
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler

#: The stubbed table steps in run() order, as (step name, Harness attribute).
STUBBED_STEPS: tuple[tuple[str, str], ...] = (
    ('1c', '_dismiss_stale_escalations'),
    ('1c0-dirty', '_file_dirty_tree_escalation'),
    ('1c0-config-keys', '_file_config_unknown_keys_escalation'),
    ('1c0', '_rehydrate_merge_halt'),
    ('1c0a', '_recover_pending_merges'),
    ('1c0a2', '_reap_orphaned_merge_worktrees'),
    ('1c0a3', '_reconcile_landed_outbox'),
    ('1c0b', '_file_restored_pause_escalation'),
    ('1c1g', '_run_interactive_worktree_reaper_pass'),
    ('1c1h', '_run_leftover_verify_scope_reaper_pass'),
    ('1c1i', '_ensure_project_slice_weight'),
    ('2c', '_recover_crashed_tasks'),
    ('2c2', '_reconcile_lane_checkouts'),
    ('2c3-streak-alarms', '_inherit_pending_streak_alarms'),
    ('2d', '_reconcile_stranded_in_progress'),
    ('2e', '_reap_orphan_worktrees'),
    ('diff7', '_reconcile_terminal_lanes'),
    ('2g', '_run_scheduled_deploy_settle_pass'),
)
SYNC_STEPS = frozenset({
    '_file_restored_pause_escalation', '_inherit_pending_streak_alarms',
})
#: Table steps the fixtures run for real against their mocked services.
UNSTUBBED_STEPS = frozenset({'1b', '1b-crash-loop', '2f'})

_SCENE_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
#: The startup-table steps the merge-lane harness runs for real: the merge
#: worker's construction and the recovery that journals into its queue.
_REAL_STARTUP_STEPS = UNSTUBBED_STEPS | {'1c0a'}


def merge_lane_harness_factory(
    tmp_path: Path, *, real_steps: frozenset[str] = frozenset(),
) -> Callable[[], Harness]:
    """Build Harnesses over ONE seeded repo, each running the real 1b and 1c0a.

    *real_steps* names further ``STUBBED_STEPS`` steps each Harness keeps real.
    A later Harness sees the journal and runs.db an earlier one left.
    """
    unknown = real_steps - {step for step, _attr in STUBBED_STEPS}
    if unknown:
        raise ValueError(f'not stubbed startup steps: {sorted(unknown)}')
    kept_real = _REAL_STARTUP_STEPS | real_steps
    repo = seed_repo(tmp_path / 'repo')

    def make() -> Harness:
        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(lane_scene_config(repo, _SCENE_GIT))
        mock_mcp_cls.return_value.start = AsyncMock()
        mock_mcp_cls.return_value.stop = AsyncMock()

        async def start_escalation_server() -> None:
            h._escalation_queue = EscalationQueue(tmp_path / 'escalations')

        h._start_escalation_server = AsyncMock(side_effect=start_escalation_server)
        h._stop_escalation_server = AsyncMock()
        for step, attr in STUBBED_STEPS:
            if step not in kept_real:
                setattr(h, attr, MagicMock() if attr in SYNC_STEPS else AsyncMock())
        h._tag_task_modules = AsyncMock()
        h._tag_prd_metadata = AsyncMock()

        h.scheduler = MagicMock()
        h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
        h.scheduler.set_task_status = AsyncMock()
        return h

    return make


async def harness_with_real_merge_worker(
    make: Callable[[], Harness], scheduler: Scheduler,
) -> Harness:
    """A Harness over the factory's shared escalation dir, holding a real merge worker."""
    return await start_real_merge_worker(make(), scheduler)


async def start_real_merge_worker(h: Harness, scheduler: Scheduler) -> Harness:
    """Start *h*'s escalation server and a real merge worker over *scheduler*; *h*."""
    await h._start_escalation_server()
    h.scheduler = scheduler
    await h._start_merge_worker()
    return h


async def journal_merge_request(
    h: Harness, task_id: str, *, task_files: list[str] | None = None,
) -> str:
    """Journal a real merge of task *task_id*'s new branch, as the live enqueue does; its id.

    *task_files* are the files the journalled request records for the task.
    """
    request = await branch_merge_request(h.git_ops, h.config, task_id, filename=f'{task_id}.txt')
    if task_files is not None:
        request = dataclasses.replace(request, task_files=task_files)
    h._merge_store.record(request)
    return request.request_id


#: The merge worker's shutdown outcome. Spelled out: importing the lane's
#: constant would make this file, and so every file importing it, lane-importing.
MERGE_WORKER_SHUTDOWN_OUTCOME = SimpleNamespace(
    status='blocked', reason='Merge worker shutting down',
    block_category=SimpleNamespace(value='merge_error'),
    failure_category=None, failure_cause_hint=None,
)


def recovered_request(task_id: str, *, task_files: list[str] | None = None) -> SimpleNamespace:
    """A merge request of task *task_id* as step 1c0a recovers it, its outcome unset."""
    return SimpleNamespace(
        task_id=task_id, request_id=f'mr-recovered-{task_id}',
        branch=SimpleNamespace(bare_id=task_id), task_files=task_files,
        result=asyncio.get_running_loop().create_future(),
    )


async def recover_requests(h: Harness, *requests: SimpleNamespace) -> dict:
    """Run step 1c0a with the journal recovering exactly *requests*; its report."""
    report = {
        'recovered': len(requests), 'dropped': 0, 'coalesced': 0,
        'requests': list(requests), 'journal_corrupt': False,
    }
    with patch('orchestrator.harness.recover_pending_merges', AsyncMock(return_value=report)):
        return await h._recover_pending_merges()


async def drain_background_tasks(h: Harness) -> None:
    """Await the background work *h* has scheduled, such as a recovered outcome's settle."""
    await asyncio.sleep(0)
    await asyncio.gather(*h._background_tasks)
