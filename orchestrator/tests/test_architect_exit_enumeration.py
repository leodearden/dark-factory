"""Architect/SIMPLE_TASK prompt enumeration of the α/β exit surfaces (task 3034).

PRD: ``plans/architect-already-complete-exits.md`` §ζ ("Update the architect
prompt/skill exit enumeration to teach: use ``mark_step_committed`` for
already-committed-green steps (skip implement); use ``report_ready_to_merge``
for the merge-landing desync (not ``report_unactionable``)").

α (task 3030) landed ``mark_step_committed`` end-to-end, INCLUDING the
``_PLAN_CREATOR_TOOLS`` grant (see
``test_architect_all_committed_plan.py::TestArchitectCanReachMarkStepCommitted``).
β (task 3031) landed ``report_ready_to_merge`` end-to-end EXCEPT the allowlist
grant — the tool existed on the plan-tools MCP server but no role could reach
it. ζ's job is twofold:

  1. Grant ``mcp__plan-tools__report_ready_to_merge`` to ARCHITECT and
     SIMPLE_TASK via ``_PLAN_CREATOR_TOOLS`` (roles.py) — otherwise the prose
     below would teach an exit the agent cannot structurally call.
  2. Teach BOTH new surfaces in the prompt bodies that enumerate exits:
     ``ARCHITECT.system_prompt`` (rule-2 exit list + the pre-satisfy note),
     ``SIMPLE_TASK.system_prompt`` (the "## Rejection artifacts" list), and
     the two ``BriefingAssembler`` dispatch-time prompts that carry their own
     rejection-exit enumerations (``build_plan_completion_prompt``,
     ``build_simple_task_prompt``).

Covered surfaces:

  - ``ARCHITECT.allowed_tools`` / ``SIMPLE_TASK.allowed_tools`` grant
    ``mcp__plan-tools__report_ready_to_merge`` (and IMPLEMENTER/DEBUGGER
    deliberately do NOT) — roles.py
  - The granted name resolves to an actually-registered plan-tools MCP tool
    (the dead-grant check) — mcp/plan_tools.py
  - ``ARCHITECT.system_prompt`` enumerates both ``report_ready_to_merge`` and
    ``mark_step_committed`` with a discriminating when-to-use keyword each —
    roles.py
  - ``SIMPLE_TASK.system_prompt`` enumerates both — roles.py
  - ``BriefingAssembler.build_plan_completion_prompt`` and
    ``build_simple_task_prompt`` enumerate ``report_ready_to_merge`` (and the
    former also teaches ``mark_step_committed`` for the partial-plan-resume
    case) — briefing.py

Assertions here are deliberately COMPACT — tool-name presence plus one
discriminating keyword per surface, not sentence-wording pins — mirroring the
existing prompt-anchor convention (``test_roles_staging_command.py``,
``test_roles_background_warning.py``) and this task's own design decision to
avoid freezing prose a later prompt-tuning pass would need to touch anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts

# ---------------------------------------------------------------------------
# step-1 RED: allowlist reachability for report_ready_to_merge
# ---------------------------------------------------------------------------
#
# allowed_tools is the ENFORCED tool surface — agents/invoke.py:979-985 builds
# the backend `--tools` allowlist from it (and loudly drops specs it cannot
# express), so a tool absent from allowed_tools is structurally unreachable no
# matter what the prompt says. See the identical rationale in
# test_architect_all_committed_plan.py::TestArchitectCanReachMarkStepCommitted.

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


# ---------------------------------------------------------------------------
# step-3 RED: ARCHITECT.system_prompt enumerates both new exits
# ---------------------------------------------------------------------------
#
# Compact anchors only — tool-name presence plus one discriminating keyword
# per surface, per this task's design decision to avoid pinning sentence
# wording a later prompt-tuning pass would need to touch anyway (see
# test_roles_background_warning.py's identical "one-line existence check, not
# a prose pin" posture).


def _window_around(text: str, needle: str, before: int = 200, after: int = 900) -> str:
    """Substring of *text* surrounding the first occurrence of *needle*.

    Lets a test confirm two related tokens sit in the same guidance block
    without pinning the exact sentence connecting them.
    """
    idx = text.index(needle)
    return text[max(0, idx - before): idx + after]


class TestArchitectPromptEnumeratesNewExits:
    def test_prompt_names_report_ready_to_merge(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'report_ready_to_merge' in ARCHITECT.system_prompt

    def test_report_ready_to_merge_bullet_has_discriminating_keyword(self):
        """The bullet must name the clean-fast-forward predicate, not just
        the tool — otherwise an architect can't tell this exit apart from
        report_task_already_done."""
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        window = _window_around(ARCHITECT.system_prompt, 'report_ready_to_merge(')
        assert 'fast-forward' in window, (
            f'Expected the report_ready_to_merge guidance to name the '
            f'clean-fast-forward predicate. Window: {window!r}'
        )

    def test_report_ready_to_merge_bullet_steers_away_from_report_unactionable_task(self):
        """Must explicitly name report_unactionable_task as the wrong exit —
        that L1 cost a human ~100k tokens and vetoed the verified-green
        auto-merge reaper (workflow.py:5854-5859)."""
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        window = _window_around(ARCHITECT.system_prompt, 'report_ready_to_merge(')
        assert 'report_unactionable_task' in window, (
            f'Expected the report_ready_to_merge guidance to steer away from '
            f'report_unactionable_task. Window: {window!r}'
        )

    def test_prompt_names_mark_step_committed(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'mark_step_committed' in ARCHITECT.system_prompt

    def test_mark_step_committed_note_has_discriminating_keyword(self):
        """The pre-satisfy note must contrast against leaving a step pending
        (or explicitly say "committed"), not just name the tool."""
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        window = _window_around(
            ARCHITECT.system_prompt, 'mark_step_committed(',
        ).lower()
        assert 'pending' in window or 'committed' in window, (
            f'Expected the mark_step_committed note to mention "pending" or '
            f'"committed". Window: {window!r}'
        )

    def test_escalate_blocker_fallthrough_lists_report_ready_to_merge(self):
        """The rule-2 escalate_blocker fall-through sentence (roles.py:353)
        must include report_ready_to_merge among the exits that route to a
        clean L1 instead of a retry."""
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        prompt = ARCHITECT.system_prompt
        idx = prompt.index('clean L1')
        window = prompt[max(0, idx - 400): idx + 100]
        assert 'report_ready_to_merge' in window, (
            f'Expected the escalate_blocker fall-through sentence to list '
            f'report_ready_to_merge. Window: {window!r}'
        )


# ---------------------------------------------------------------------------
# step-5 RED: SIMPLE_TASK.system_prompt enumerates both new exits
# ---------------------------------------------------------------------------
#
# workflow.py:4465 routes .task/ready_to_merge.json on the simple-task path
# with the identical handler as the architect path, so a simple-task agent
# hitting a merge-landing desync has the same correct exit available and must
# be taught it.


class TestSimpleTaskPromptEnumeratesNewExits:
    def test_rejection_artifacts_list_names_report_ready_to_merge(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert 'mcp__plan-tools__report_ready_to_merge' in SIMPLE_TASK.system_prompt

    def test_report_ready_to_merge_entry_has_discriminating_keyword(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        window = _window_around(
            SIMPLE_TASK.system_prompt, 'mcp__plan-tools__report_ready_to_merge',
        )
        assert 'fast-forward' in window, (
            f'Expected the report_ready_to_merge entry to name the '
            f'clean-fast-forward predicate. Window: {window!r}'
        )

    def test_prompt_names_mark_step_committed_with_pre_satisfy_guidance(self):
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert 'mark_step_committed' in SIMPLE_TASK.system_prompt
        window = _window_around(
            SIMPLE_TASK.system_prompt, 'mark_step_committed(',
        ).lower()
        assert 'pending' in window or 'committed' in window, (
            f'Expected the mark_step_committed guidance to mention "pending" '
            f'or "committed". Window: {window!r}'
        )


# ---------------------------------------------------------------------------
# step-7 RED: BriefingAssembler dispatch-time prompts enumerate both new exits
# ---------------------------------------------------------------------------
#
# Separate from the roles.py system_prompt strings covered above: these are
# the dispatch-time briefings TaskWorkflow actually sends for the
# partial-plan-resume path (build_plan_completion_prompt) and the SIMPLE_TASK
# path (build_simple_task_prompt). Fixture shape mirrors test_briefing.py's
# `briefing` fixture (BriefingAssembler over a tmp_path project_root);
# `context=''` bypasses the real fused-memory search call, matching the
# convention used throughout test_briefing.py for methods that accept a
# `context` override.


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
