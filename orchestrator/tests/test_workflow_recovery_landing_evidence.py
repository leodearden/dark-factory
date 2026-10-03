"""The recovery guards' fallback arms stamp only ATTRIBUTABLE landings (task 4704).

``TaskWorkflow``'s three already-merged guards — pre-PLAN
(``_recover_if_already_merged``), pre-EXECUTE (``_recover_before_execute``)
and pre-MERGE (``_recover_before_merge``) — each have a journal-miss FALLBACK
arm that used to stamp ``found_on_main`` with main's CURRENT TIP once its
has-work heuristic passed.  Main's tip is whatever landed most recently, from
any task, so that stamp could name a commit carrying none of this task's work
(measured: task 3269's pre-MERGE guard stamped unrelated tip 2496ea9dcd).

Each fallback arm now runs ``validate_landing_evidence`` in DISCOVERY mode
first and stamps the task's OWN citation; with no attributable, surviving
landing it stamps nothing and lets the phase proceed.

Driven through ``_workflow_helpers._make``'s git stubs: ``landing_citation``
is the commit on main citing the task and ``landing_effect_present`` whether
its effect survives.  This file imports no merge-lane module on purpose.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import _Fixture, _make

from orchestrator.workflow import WorkflowOutcome

_CITATION = 'citationsha123'
_MAIN_TIP = 'mainsha123'
_VALIDATE_TARGET = 'orchestrator.workflow.validate_landing_evidence'


def _record_prior_work(f: _Fixture) -> None:
    """A work-classified iteration entry for the REAL has-work classifier."""
    f.artifacts.append_iteration_log({
        'agent': 'implementer', 'source': 'orchestrator',
        'steps_attempted': ['s1'], 'steps_completed': ['s1'],
        'commit': 'newhead',
    })


async def _pre_plan(f: _Fixture) -> WorkflowOutcome | None:
    f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
        return_value=('wthead123', _MAIN_TIP),
    )
    _record_prior_work(f)
    return await f.wf._recover_if_already_merged()


async def _pre_execute(f: _Fixture) -> WorkflowOutcome | None:
    f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
        return_value=('wthead123', _MAIN_TIP),
    )
    f.wf.git_ops.get_merge_diff_files = AsyncMock(return_value=(['pkg/a.py'], None))
    return await f.wf._recover_before_execute()


async def _pre_merge(f: _Fixture) -> WorkflowOutcome | None:
    _record_prior_work(f)
    return await f.wf._recover_before_merge('branchhead123', _MAIN_TIP)


_Guard = Callable[[_Fixture], Awaitable['WorkflowOutcome | None']]

#: (driver, the guard's unchanged provenance note, the branch tip it validates)
GUARDS = [
    pytest.param(
        _pre_plan,
        'branch already on main at workflow start (pre-PLAN recovery)',
        'wthead123', id='pre-PLAN',
    ),
    pytest.param(
        _pre_execute,
        'branch already on main at workflow start (pre-EXECUTE recovery)',
        'wthead123', id='pre-EXECUTE',
    ),
    pytest.param(
        _pre_merge,
        'branch already on main at merge phase (pre-MERGE recovery)',
        'branchhead123', id='pre-MERGE',
    ),
]


def _fixture(tmp_path: Path, **overrides: Any) -> _Fixture:
    return _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', **overrides)


@pytest.mark.asyncio
@pytest.mark.parametrize(('guard', 'note', 'tip'), GUARDS)
class TestFallbackStampsTheTasksOwnCitation:

    async def test_attributable_landing_stamps_the_citation_not_mains_tip(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
    ) -> None:
        f = _fixture(tmp_path)

        outcome = await guard(f)

        assert outcome == WorkflowOutcome.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha=_CITATION, note=note,
        )
        stamped = [c.kwargs['sha'] for c in f.mark_done.await_args_list]
        assert _MAIN_TIP not in stamped, "main's tip is never this task's provenance"

    async def test_unattributed_landing_stamps_nothing_and_proceeds(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
    ) -> None:
        """The task-3269 shape: the branch reads as merged and work-shaped,
        but no commit on main cites the task — so nothing is stamped."""
        f = _fixture(tmp_path, landing_citation=None)

        outcome = await guard(f)

        assert outcome is None
        f.mark_done.assert_not_awaited()

    async def test_landing_whose_effect_is_gone_stamps_nothing(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
    ) -> None:
        f = _fixture(tmp_path, landing_effect_present=False)

        outcome = await guard(f)

        assert outcome is None
        f.mark_done.assert_not_awaited()

    async def test_a_reject_is_logged_with_its_reason(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = _fixture(tmp_path, landing_citation=None)

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            await guard(f)

        assert 'no_citation' in caplog.text
        assert tip[:8] in caplog.text

    async def test_validates_the_tasks_own_branch_tip_checks_and_pattern(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
    ) -> None:
        """DISCOVERY mode, shaped like the harness dispatch gate's ancestry arm."""
        f = _fixture(tmp_path)
        checks = [{'name': 'cap-x', 'kind': 'grep', 'pattern': 'y', 'expect': 'present'}]
        f.wf.task['metadata'] = {'delivered_checks': checks}
        validate = AsyncMock(return_value=SimpleNamespace(
            accepted=False, evidence_sha=None, reason='no_citation',
        ))

        with patch(_VALIDATE_TARGET, validate):
            outcome = await guard(f)

        assert outcome is None
        validate.assert_awaited_once()
        assert validate.await_args is not None
        args, kwargs = validate.await_args
        assert args == (f.wf.git_ops, f.wf.task_id, f'task/{f.wf.task_id}')
        assert kwargs['branch_tip_sha'] == tip
        assert kwargs['pattern_template'] == f.wf.git_ops.config.commit_citation_pattern
        assert kwargs['delivered_checks'] == checks
        assert 'candidate_sha' not in kwargs

    async def test_the_real_validation_asks_git_about_the_tasks_branch(
        self, tmp_path: Path, guard: _Guard, note: str, tip: str,
    ) -> None:
        f = _fixture(tmp_path)

        await guard(f)

        cast(AsyncMock, f.wf.git_ops.find_task_citation_commit).assert_awaited_once_with(
            f.wf.task_id,
            pattern_template=f.wf.git_ops.config.commit_citation_pattern,
        )
        f.is_ancestor.assert_any_await(_CITATION, f'task/{f.wf.task_id}')
