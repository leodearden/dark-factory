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
        task = {
            'id': 'df_task_42',
            'title': 'Add widget',
            'description': 'Simple widget addition.',
            'status': 'blocked',
            'dependencies': [],
            'priority': 'medium',
            'metadata': {},
        }
        escalation = {
            'id': 'esc-42-1',
            'category': 'task_failure',
            'summary': 'Execution iterations exhausted',
            'detail': 'Agent ran out of iterations.',
            'severity': 'blocking',
        }

        with patch.object(
            BriefingAssembler,
            '_get_memory_context',
            return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_steward_initial_prompt(
                task=task,
                escalation=escalation,
                pending_escalations=[],
                worktree=tmp_path,
            )

        assert 'done_provenance' in prompt, (
            'Steward prompt must reference done_provenance so the LLM agent '
            'knows to pass it when calling set_task_status(done) after the '
            'enforcement flag is flipped.'
        )
