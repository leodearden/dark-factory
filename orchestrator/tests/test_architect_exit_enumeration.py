"""Architect/SIMPLE_TASK prompt enumeration of the α/β exit surfaces (task 3034).

PRD: ``plans/architect-already-complete-exits.md`` §ζ.

After the step-9 review fix, covers: the ``_PLAN_CREATOR_TOOLS`` grant of
``mcp__plan-tools__report_ready_to_merge`` to ARCHITECT/SIMPLE_TASK (enforced
by ``agents/invoke.py:979-985``, which builds the backend allowlist from
``allowed_tools`` — an ungranted tool is unreachable regardless of prompt
text); the dead-grant check; and bare tool-name-presence anchors for
``report_ready_to_merge`` / ``mark_step_committed`` on the two roles.py
system prompts and the two ``BriefingAssembler`` dispatch prompts.

Prose-window/keyword assertions were deliberately removed in review (design
decision 4) — do not reintroduce them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts

# allowed_tools is enforced (agents/invoke.py:979-985 builds the backend allowlist from it).

_REPORT_READY_TO_MERGE = 'mcp__plan-tools__report_ready_to_merge'


class TestArchitectCanReachReportReadyToMerge:
    def test_architect_is_granted_report_ready_to_merge(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert _REPORT_READY_TO_MERGE in ARCHITECT.allowed_tools, (
            f'ARCHITECT.allowed_tools omits {_REPORT_READY_TO_MERGE!r} — the '
            f'backend allowlist built at agents/invoke.py:979-985 would strip '
            f'the call, making the ζ signal structurally unreachable. Got: '
            f'{ARCHITECT.allowed_tools!r}'
        )

    def test_simple_task_is_granted_report_ready_to_merge(self):
        """workflow.py:4465 dispatches the ready-to-merge artifact on the
        SIMPLE_TASK path too (the same handler as the architect path), so a
        simple-task agent hitting a merge-landing desync needs the same
        reachable exit."""
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert _REPORT_READY_TO_MERGE in SIMPLE_TASK.allowed_tools, (
            f'SIMPLE_TASK.allowed_tools omits {_REPORT_READY_TO_MERGE!r} — '
            f'workflow.py:4465 dispatches .task/ready_to_merge.json on this '
            f'path too, so the tool must be reachable here as well. Got: '
            f'{SIMPLE_TASK.allowed_tools!r}'
        )

    def test_implementer_is_not_granted_report_ready_to_merge(self):
        """Reporting merge-readiness is an AUTHORING-time authority, not an
        execution one — same reasoning as mark_step_committed's role split."""
        from orchestrator.agents.roles import IMPLEMENTER  # noqa: PLC0415

        assert _REPORT_READY_TO_MERGE not in IMPLEMENTER.allowed_tools

    def test_debugger_is_not_granted_report_ready_to_merge(self):
        from orchestrator.agents.roles import DEBUGGER  # noqa: PLC0415

        assert _REPORT_READY_TO_MERGE not in DEBUGGER.allowed_tools

    def test_architect_declares_plan_tools_family(self):
        """Keeps the inverse-capability invariant satisfied by the grant.

        test_agent_capability_wiring.py::TestInverseCapabilityInvariant
        asserts every role allowing an ``mcp__plan-tools__`` prefix declares
        the ``plan_tools`` family; asserted here too so a grant added without
        the family declaration fails in this suite as well.
        """
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'plan_tools' in ARCHITECT.mcp_families

    def test_simple_task_declares_plan_tools_family(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert 'plan_tools' in SIMPLE_TASK.mcp_families


@pytest.mark.asyncio
class TestGrantedReadyToMergeToolExists:
    async def test_granted_name_resolves_to_a_registered_tool(self, tmp_path: Path):
        """The grant must never name a tool the server does not register.

        Mirrors test_architect_all_committed_plan.py::TestGrantedToolActuallyExists
        — an allowlist entry for a non-existent tool is a silent dead grant.
        """
        from orchestrator.mcp import plan_tools  # noqa: PLC0415

        artifacts = TaskArtifacts(tmp_path)
        artifacts.init('42', 'X', 'desc', base_commit='base')
        server = plan_tools.create_server(artifacts)

        tool = await server.get_tool('report_ready_to_merge')

        assert tool is not None
        # The granted allowlist name is the mcp__<server>__<tool> spelling of
        # exactly this registered tool.
        assert _REPORT_READY_TO_MERGE.endswith('__report_ready_to_merge')


class TestArchitectPromptEnumeratesNewExits:
    def test_prompt_names_report_ready_to_merge(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'report_ready_to_merge' in ARCHITECT.system_prompt

    def test_prompt_names_mark_step_committed(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'mark_step_committed' in ARCHITECT.system_prompt


# workflow.py:4465 routes .task/ready_to_merge.json on the simple-task path too (same handler).


class TestSimpleTaskPromptEnumeratesNewExits:
    def test_rejection_artifacts_list_names_report_ready_to_merge(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert 'mcp__plan-tools__report_ready_to_merge' in SIMPLE_TASK.system_prompt

    def test_prompt_names_mark_step_committed(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert 'mark_step_committed' in SIMPLE_TASK.system_prompt


# Dispatch-time briefings, separate from roles.py system_prompt; fixture mirrors test_briefing.py.


@pytest.fixture
def briefing(tmp_path: Path):
    from orchestrator.agents.briefing import BriefingAssembler  # noqa: PLC0415
    from orchestrator.config import GitConfig, OrchestratorConfig  # noqa: PLC0415

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


def _partial_plan_with_pending_steps() -> dict:
    return {
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


@pytest.mark.asyncio
class TestBriefingPromptsEnumerateNewExits:
    async def test_plan_completion_prompt_names_report_ready_to_merge(self, briefing):
        """Path-c rejection-exit list (briefing.py, build_plan_completion_prompt)
        must offer report_ready_to_merge alongside the pre-existing three."""
        task = {'id': '3822', 'title': 'Add recon closure', 'description': 'Demo'}
        prompt = await briefing.build_plan_completion_prompt(
            task, _partial_plan_with_pending_steps(), worktree=None, context='',
        )
        assert 'report_ready_to_merge' in prompt

    async def test_plan_completion_prompt_names_mark_step_committed_for_partial_resume(
        self, briefing,
    ):
        """Finishing a partial plan must also teach pre-satisfying steps the
        branch already carries green commits for, instead of leaving them
        pending — the same guidance build_architect_prompt gives on fresh
        derivation."""
        task = {'id': '3822', 'title': 'Add recon closure', 'description': 'Demo'}
        prompt = await briefing.build_plan_completion_prompt(
            task, _partial_plan_with_pending_steps(), worktree=None, context='',
        )
        assert 'mark_step_committed' in prompt

    async def test_simple_task_prompt_names_report_ready_to_merge(self, briefing):
        """The SIMPLE_TASK dispatch briefing's only exit sentence currently
        names just report_unactionable_task; it must also teach
        report_ready_to_merge for the branch-complete-but-unmerged case."""
        task = {'id': '77', 'title': 'Small fix', 'description': 'Demo'}
        prompt = await briefing.build_simple_task_prompt(
            task, worktree=None, context='',
        )
        assert 'report_ready_to_merge' in prompt
