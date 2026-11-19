"""The routed filer files onto an orchestrator queue only while a process holds that root's flock.

``ReconciliationHarness._file_finding_task_escalation`` asks whether an
orchestrator is draining ``<project_root>/data/escalations`` before it files
there. The answer comes from the orchestrator's flock, not the PID line in its
lock file: a PID line can outlive its orchestrator, and can be truncated by a
second start that failed to take the lock. A probe that raises files nothing.
"""

from __future__ import annotations

import errno
import logging
import os
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

import fused_memory.reconciliation.harness as harness_module
from reconciliation.harness_fixtures import open_harness
from reconciliation.orchestrator_lock_fixtures import (
    held_orchestrator_lock,
    write_orchestrator_lock,
)

_HARNESS_LOGGER = 'fused_memory.reconciliation.harness'
_TASK_ID = '4458'


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    return root


@pytest_asyncio.fixture
async def harness(tmp_path, project_root):
    async with open_harness(
        tmp_path, taskmaster=AsyncMock(), known_projects={'test-project': str(project_root)},
    ) as h:
        yield h


def _finding() -> dict:
    return {
        'finding_id': 'f-4458',
        'severity': 'serious',
        'category': 'memory_contradiction',
        'description': 'Operator ruled Option A on esc-4458-87; the commit did the opposite',
        'suggested_action': 'Re-read the operator ruling before closing',
        'actionable': True,
        'task_id': _TASK_ID,
        'affected_ids': [_TASK_ID],
    }


def _file(harness) -> str | None:
    return harness._file_finding_task_escalation('test-project', 'run-1', _finding(), 4)


@pytest.mark.asyncio
async def test_a_lock_naming_a_live_pid_that_nobody_holds_files_nothing(harness, project_root):
    write_orchestrator_lock(project_root, f'PID {os.getpid()} started 2026-11-14T00:00:00+00:00')

    assert _file(harness) is None
    assert not escalation_queue_dir(project_root).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'first_line',
    [
        pytest.param('', id='pid-line-truncated-by-a-failed-second-start'),
        pytest.param(
            'PID 100000000000000000000 started 2026-11-14T00:00:00+00:00',
            id='pid-line-naming-no-live-process',
        ),
    ],
)
async def test_a_held_lock_files_whatever_its_pid_line_says(harness, project_root, first_line):
    with held_orchestrator_lock(project_root, first_line):
        esc_id = _file(harness)

    assert esc_id is not None
    pending = EscalationQueue(escalation_queue_dir(project_root)).get_by_task(
        _TASK_ID, status='pending',
    )
    assert [e.id for e in pending] == [esc_id]


@pytest.mark.asyncio
async def test_a_raising_lock_probe_files_nothing_and_warns(
    harness, project_root, monkeypatch, caplog,
):
    def _no_locks(_root):
        raise OSError(errno.ENOLCK, 'No locks available')

    monkeypatch.setattr(harness_module, 'is_orchestrator_lock_held', _no_locks)

    with caplog.at_level(logging.WARNING, logger=_HARNESS_LOGGER):
        esc_id = _file(harness)

    assert esc_id is None
    assert not escalation_queue_dir(project_root).exists()
    [warning] = [
        r for r in caplog.records
        if r.name == _HARNESS_LOGGER and r.levelno == logging.WARNING
        and r.getMessage() == 'reconciliation.finding_task_escalation_orchestrator_probe_failed'
    ]
    assert warning.task_id == _TASK_ID
