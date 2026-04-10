"""Unit tests for build_completion_judge_prompt (ζ).

Scoped to the prompt builder — no MCP, no workflow. Patches
``BriefingAssembler._get_memory_context`` so the tests don't hit the
fused-memory HTTP endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import (
    COMPLETION_JUDGE_SCHEMA,
    BriefingAssembler,
    CompletionJudgeVerdict,
)
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


@pytest.fixture
def plan() -> dict:
    return {
        'task_id': 'df_task_42',
        'title': 'Add widget',
        'analysis': 'Simple widget addition via TDD.',
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'Write failing widget test', 'status': 'pending'},
            {'id': 'step-2', 'type': 'impl', 'description': 'Implement widget', 'status': 'pending'},
        ],
    }


@pytest.fixture
def small_diff() -> str:
    return (
        'diff --git a/widget.py b/widget.py\n'
        'index 0000000..1111111 100644\n'
        '--- a/widget.py\n'
        '+++ b/widget.py\n'
        '@@ -0,0 +1,3 @@\n'
        '+def widget():\n'
        '+    return 42\n'
    )


@pytest.mark.asyncio
class TestBuildCompletionJudgePrompt:
    async def test_includes_plan(self, briefing, plan, small_diff):
        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_completion_judge_prompt(
                plan=plan,
                iteration_log=[],
                diff=small_diff,
                task_id='df_task_42',
            )

        assert '# Plan' in prompt
        assert '# Code Diff' in prompt
        assert 'step-1' in prompt
        assert 'step-2' in prompt
        assert 'Follow the safety rules' in prompt
        # Should include the diff body
        assert 'def widget()' in prompt

    async def test_truncates_large_diff(self, briefing, plan):
        # 60k characters — above the 50k cap
        huge_diff = 'x' * 60000
        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_completion_judge_prompt(
                plan=plan,
                iteration_log=[],
                diff=huge_diff,
                task_id='df_task_42',
            )

        assert '[diff truncated]' in prompt

    async def test_recent_iterations_capped_at_5(self, briefing, plan, small_diff):
        iteration_log = [
            {'iteration': i, 'agent': 'implementer', 'summary': f'iter {i} work'}
            for i in range(1, 11)
        ]
        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_completion_judge_prompt(
                plan=plan,
                iteration_log=iteration_log,
                diff=small_diff,
                task_id='df_task_42',
            )

        # Only the last 5 entries (iterations 6..10) should appear
        assert 'iter 10 work' in prompt
        assert 'iter 6 work' in prompt
        # The earlier ones must not be present
        assert 'iter 1 work' not in prompt
        assert 'iter 5 work' not in prompt

    async def test_empty_iteration_log_omits_section(self, briefing, plan, small_diff):
        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_completion_judge_prompt(
                plan=plan,
                iteration_log=[],
                diff=small_diff,
                task_id='df_task_42',
            )

        assert '## Recent Iterations' not in prompt


class TestCompletionJudgeSchema:
    """Sanity checks on the shared schema constant + dataclass alignment."""

    def test_schema_required_fields_match_dataclass(self):
        # All four dataclass fields must appear in schema.required
        required = set(COMPLETION_JUDGE_SCHEMA['required'])
        assert required == {
            'complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work',
        }

    def test_dataclass_accepts_full_verdict(self):
        v = CompletionJudgeVerdict(
            complete=True,
            reasoning='diff covers all steps',
            uncovered_plan_steps=[],
            substantive_work=True,
        )
        assert v.complete is True
        assert v.substantive_work is True
