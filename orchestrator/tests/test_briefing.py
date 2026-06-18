"""Regression tests for ``BriefingAssembler._format_task``.

Locks in the post-normalization invariant that ``task['metadata']`` is
always a dict by the time ``_format_task`` reads it. The boundary
normalizer in :meth:`Scheduler.get_tasks` is responsible for the coerce;
this test confirms ``_format_task`` builds the prompt without raising
``AttributeError`` when handed the wire shape downstream consumers
should now see.

Also covers :meth:`BriefingAssembler.build_plan_tightening_prompt`,
introduced for the plan-files-not-touched architect-narrowing retry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


class TestFormatTaskMetadataInvariant:
    def test_dict_metadata_with_files(self, briefing: BriefingAssembler):
        task = {
            'id': '424',
            'title': 'Trivial DRY consolidation',
            'description': 'Move three helpers',
            'metadata': {'files': ['orchestrator', 'shared']},
        }
        out = briefing._format_task(task)
        assert '**ID:** 424' in out
        assert '**Files:** orchestrator, shared' in out

    def test_empty_dict_metadata(self, briefing: BriefingAssembler):
        task = {
            'id': '1',
            'title': 'No files',
            'metadata': {},
        }
        out = briefing._format_task(task)
        assert '**ID:** 1' in out
        assert 'Files' not in out

    def test_metadata_absent(self, briefing: BriefingAssembler):
        task = {'id': '2', 'title': 'No metadata key'}
        out = briefing._format_task(task)
        assert '**ID:** 2' in out
        assert 'Modules' not in out

    def test_metadata_dict_without_modules_key(self, briefing: BriefingAssembler):
        task = {
            'id': '3',
            'title': 'Other metadata only',
            'metadata': {'files': ['a.py']},
        }
        out = briefing._format_task(task)
        assert '**ID:** 3' in out
        assert 'Modules' not in out


class TestFormatTaskIncludeFiles:
    """Unit tests for the include_files parameter on _format_task.

    Locks in the anti-anchor mechanic introduced for C-A1: when
    include_files=False the Files line must be absent so the first architect
    derivation cannot rubber-stamp the queue-time metadata.files guess.
    """

    def _task_with_files(self) -> dict:
        return {
            'id': '1835',
            'title': 'Anti-anchor task',
            'description': 'Derive files independently',
            'metadata': {'files': ['orchestrator', 'shared']},
        }

    def test_include_files_false_omits_files_line(
        self, briefing: BriefingAssembler,
    ):
        """include_files=False must hide the Files line."""
        out = briefing._format_task(self._task_with_files(), include_files=False)
        assert '**Files:**' not in out

    def test_include_files_false_keeps_description(
        self, briefing: BriefingAssembler,
    ):
        """include_files=False must not suppress Description or other fields."""
        out = briefing._format_task(self._task_with_files(), include_files=False)
        assert '**Description:**' in out
        assert '**Title:**' in out

    def test_include_files_default_still_shows_files(
        self, briefing: BriefingAssembler,
    ):
        """Default call (no flag) must still render the Files line."""
        out = briefing._format_task(self._task_with_files())
        assert '**Files:** orchestrator, shared' in out


@pytest.mark.asyncio
class TestBuildPlanTighteningPrompt:
    """Architect narrowing pass after the plan-files-not-touched gate.

    Prohibition assertions use positive-directive form per
    feedback_test_assert_negative_directives.md — the prompt itself
    contains the token names it forbids, so a bare ``not in`` would
    self-conflict.
    """

    async def _build(
        self,
        briefing: BriefingAssembler,
        files: list[str] | None = None,
        not_touched: list[str] | None = None,
    ) -> str:
        task = {
            'id': '2656',
            'title': 'Test task',
            'description': 'Demo',
        }
        plan = {'files': files if files is not None else ['a.py', 'b.py', 'c.py']}
        return await briefing.build_plan_tightening_prompt(
            task,
            plan,
            not_touched if not_touched is not None else ['a.py', 'b.py'],
            worktree=None,
            context='',
        )

    async def test_mentions_both_valid_actions(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'update_plan_metadata' in prompt
        assert 'confirm_plan' in prompt

    async def test_lists_not_touched_entries(self, briefing: BriefingAssembler):
        prompt = await self._build(
            briefing,
            files=['x.py', 'y.py', 'z.py'],
            not_touched=['x.py', 'y.py'],
        )
        assert 'x.py' in prompt
        assert 'y.py' in prompt
        # Current plan files should also be visible.
        assert 'z.py' in prompt

    async def test_forbids_creation_and_step_edits(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        # Positive-directive assertions covering the forbidden tools.
        # The prompt mentions create_plan/add_plan_step/replace_plan_step
        # while listing them as off-limits — assert by section header
        # rather than 'not in prompt' to avoid the self-conflict pattern.
        assert 'Forbidden' in prompt or 'Do not call' in prompt
        assert 'create_plan' in prompt
        assert 'add_plan_step' in prompt
        assert 'replace_plan_step' in prompt

    async def test_states_new_file_addition_will_be_rejected(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        assert 'must NOT add new files' in prompt

    async def test_includes_agent_identity(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'claude-task-2656-architect' in prompt


@pytest.mark.asyncio
class TestBuildPlanCompletionPrompt:
    """Completion pass that resumes a partial plan left by an interrupted run."""

    async def _build(self, briefing: BriefingAssembler) -> str:
        task = {'id': '3822', 'title': 'Add recon closure', 'description': 'Demo'}
        partial = {
            'task_id': '3822',
            'title': 'Add recon closure',
            'files': ['recon.py'],
            'analysis': 'partial analysis',
            'prerequisites': [],
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'test a',
                 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'type': 'impl', 'description': 'impl a',
                 'status': 'pending', 'commit': None},
            ],
        }
        return await briefing.build_plan_completion_prompt(
            task, partial, worktree=None, context='',
        )

    async def test_shows_partial_and_requires_finalize(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        # The existing steps are shown so the architect can build on them.
        assert 'step-1' in prompt
        assert 'step-2' in prompt
        # Must instruct finalizing the completed plan.
        assert 'confirm_plan' in prompt
        # Frames the work as finishing, not starting over.
        assert 'Completion' in prompt or 'finish' in prompt.lower()

    async def test_offers_restart_and_rejection_escape_hatches(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        assert 'create_plan' in prompt
        assert 'report_unactionable_task' in prompt

    async def test_includes_agent_identity(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'claude-task-3822-architect' in prompt
