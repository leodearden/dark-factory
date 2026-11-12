"""Unit tests for build_steward_initial_prompt.

Scoped to the prompt builder — no MCP, no workflow. Patches
``BriefingAssembler._get_memory_context`` so the tests don't hit the
fused-memory HTTP endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.landed_sha_searches import LandedShaSearches, already_merged_step
from orchestrator.config import GitConfig, OrchestratorConfig

_TASK = {
    'id': 'df_task_42',
    'title': 'Add widget',
    'description': 'Simple widget addition.',
    'status': 'blocked',
    'dependencies': [],
    'priority': 'medium',
    'metadata': {},
}
_ESCALATION = {
    'id': 'esc-42-1',
    'category': 'task_failure',
    'summary': 'Execution iterations exhausted',
    'detail': 'Agent ran out of iterations.',
    'severity': 'blocking',
}


def _fixture_git() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


def _assembler(tmp_path: Path, git: GitConfig) -> BriefingAssembler:
    return BriefingAssembler(OrchestratorConfig(project_root=tmp_path, git=git))


async def _initial_prompt(briefing: BriefingAssembler, worktree: Path) -> str:
    with patch.object(
        BriefingAssembler,
        '_get_memory_context',
        return_value='# Context\n\n_stub_',
    ):
        return await briefing.build_steward_initial_prompt(
            task=_TASK,
            escalation=_ESCALATION,
            pending_escalations=[],
            worktree=worktree,
        )


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    return _assembler(tmp_path, _fixture_git())


@pytest.mark.asyncio
class TestBuildStewardInitialPrompt:
    async def test_steward_initial_prompt_instructs_done_provenance(
        self, briefing, tmp_path
    ):
        """Steward's initial prompt must instruct it to pass done_provenance.

        Phase 2 enforcement: when the steward marks a task done (already-on-main
        case), it must supply done_provenance={'commit': <sha>} or
        {'note': <text>}. Minimal substring check per design decision 6 in the
        plan — prose wording may evolve freely as long as the token is present.
        """
        prompt = await _initial_prompt(briefing, tmp_path)

        assert 'done_provenance' in prompt, (
            'Steward prompt must reference done_provenance so the LLM agent '
            'knows to pass it when calling set_task_status(done) after the '
            'enforcement flag is flipped.'
        )

    @pytest.mark.parametrize(
        'git',
        [
            pytest.param(_fixture_git(), id='fixture-git'),
            pytest.param(GitConfig(main_branch='trunk', branch_prefix='feat/'), id='trunk-feat'),
        ],
    )
    async def test_already_merged_step_is_rendered_for_this_task_and_project(
        self, tmp_path: Path, git: GitConfig,
    ) -> None:
        briefing = _assembler(tmp_path, git)

        prompt = await _initial_prompt(briefing, tmp_path)

        expected = already_merged_step(LandedShaSearches.for_task(str(_TASK['id']), git))
        assert f'\n2. {expected}\n3. ' in prompt
