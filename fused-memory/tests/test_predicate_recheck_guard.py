"""``before_done.recheck_secs`` at the submit guard and the write boundary.

PRD plans/completion-driven-triggers-prd.md C1. The submit guard
(``deterministic_task_error``, called by ``tools.py::submit_task``) delegates
the shape and kind half to ``shared.task_metadata.BeforeDone`` and the
dated-milestone half to ``recheck_secs_milestone_conflict``. The milestone half
also lives on ``TaskMetadata``, so an enforced ``update_task`` cannot strip the
dated milestone from a re-arming predicate either.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.deterministic_task_guard import deterministic_task_error

DATED = {'mode': 'dated', 'at': '2026-11-01T00:00:00+00:00'}


def _predicate(tmp_path, *, milestone: dict | None = DATED, **before_done_fields) -> dict:
    script = tmp_path / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    os.chmod(script, 0o755)
    meta: dict = {
        'before_done': {
            'kind': 'predicate', 'script': 'run.sh', 'timeout_secs': 60, **before_done_fields,
        },
    }
    if milestone is not None:
        meta['milestone'] = milestone
    return meta


def _assert_refused(result: dict | None, *, naming: tuple[str, ...]) -> None:
    assert result is not None
    assert result['error_type'] == 'ValidationError'
    for fragment in naming:
        assert fragment in result['error'], (fragment, result['error'])
    assert isinstance(result['hint'], str)
    assert result['hint']


class TestSubmitGuardAccepts:
    def test_declared_recheck_on_a_dated_milestone(self, tmp_path) -> None:
        meta = _predicate(tmp_path, recheck_secs=86400)
        assert deterministic_task_error('deterministic', meta, str(tmp_path)) is None


class TestSubmitGuardRefuses:
    @pytest.mark.parametrize('bad', [299, True], ids=['below-min', 'bool'])
    def test_malformed_recheck_secs(self, tmp_path, bad: object) -> None:
        meta = _predicate(tmp_path, recheck_secs=bad)
        _assert_refused(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            naming=('recheck_secs',),
        )

    def test_recheck_secs_on_a_deploy(self, tmp_path) -> None:
        meta = _predicate(tmp_path, recheck_secs=86400)
        meta['before_done']['kind'] = 'deploy'
        _assert_refused(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            naming=('recheck_secs',),
        )

    @pytest.mark.parametrize(
        'milestone', [{'mode': 'delayed', 'after_secs': 259200}, None], ids=['delayed', 'absent'],
    )
    def test_recheck_secs_without_a_dated_milestone(self, tmp_path, milestone) -> None:
        meta = _predicate(tmp_path, milestone=milestone, recheck_secs=86400)
        _assert_refused(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            naming=('recheck_secs', "mode='dated'"),
        )


class TestRecheckFreeSubmissionsUnchanged:
    @pytest.mark.parametrize(
        'milestone', [{'mode': 'delayed', 'after_secs': 259200}, None], ids=['delayed', 'absent'],
    )
    def test_predicate_without_recheck_secs(self, tmp_path, milestone) -> None:
        meta = _predicate(tmp_path, milestone=milestone)
        assert deterministic_task_error('deterministic', meta, str(tmp_path)) is None

    def test_before_done_model_is_consulted_only_for_recheck_secs(self, tmp_path) -> None:
        meta = _predicate(tmp_path, args=[1])
        assert deterministic_task_error('deterministic', meta, str(tmp_path)) is None


class TestWriteBoundaryHoldsAtUpdateTask:
    @pytest.mark.asyncio
    async def test_update_cannot_strip_the_dated_milestone(self, tmp_path) -> None:
        project_root = str(tmp_path / 'proj')
        backend = SqliteTaskBackend(
            TaskmasterConfig(project_root=str(tmp_path)), task_metadata_enforce=True,
        )
        await backend.start()
        try:
            stored = {
                'task_kind': 'deterministic',
                'before_done': {
                    'kind': 'predicate', 'script': 'run.sh', 'timeout_secs': 60,
                    'recheck_secs': 86400,
                },
                'milestone': DATED,
            }
            dto = await backend.add_task(
                project_root=project_root, title='gate', metadata=json.dumps(stored),
            )
            task_id = dto['id']

            with pytest.raises(ValidationError):
                await backend.update_task(
                    task_id, project_root=project_root,
                    metadata=json.dumps({'milestone': {'mode': 'delayed', 'after_secs': 60}}),
                    metadata_mode='merge',
                )
            task = await backend.get_task(task_id, project_root=project_root)
            assert task['metadata']['milestone'] == DATED

            rearm = {
                'milestone': {'mode': 'dated', 'at': '2026-11-02T00:00:00+00:00'},
                'predicate_recheck': {
                    'consecutive': 1,
                    'first_rearmed_at': '2026-11-01T00:00:00+00:00',
                    'last_checked_at': '2026-11-01T00:00:00+00:00',
                    'next_check_at': '2026-11-02T00:00:00+00:00',
                },
                'predicate_verdict': {'verdict': 'not_yet'},
            }
            await backend.update_task(
                task_id, project_root=project_root,
                metadata=json.dumps(rearm), metadata_mode='merge',
            )
            task = await backend.get_task(task_id, project_root=project_root)
            assert task['metadata']['milestone'] == rearm['milestone']
            assert task['metadata']['predicate_recheck'] == rearm['predicate_recheck']
        finally:
            await backend.close()
