"""BriefingAssembler.build_debugger_prompt's verify-attempt section (task 5357, sweep F S5-14).

A debugger dispatched on the third failed verify attempt used to receive the
same prompt shape as one dispatched on the first, with no sign that earlier
fixes had already been tried. These tests pin the optional `# Verify Attempt`
section that carries the attempt number and the earlier causes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orchestrator.agents.briefing import BriefingAssembler, VerifyAttemptContext
from orchestrator.config import GitConfig, OrchestratorConfig

_FAILURES = '## Test Failures\n\n```\nFAILED tests/test_x.py::test_a\n```'
_PLAN = {'task_id': '42', 'title': 'T', 'analysis': 'A'}


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


async def _prompt(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_debugger_prompt(_FAILURES, _PLAN, context='CTX', **kwargs)


@pytest.mark.asyncio
async def test_no_attempt_context_leaves_the_prompt_unchanged(briefing):
    omitted = await _prompt(briefing)
    explicit_none = await _prompt(briefing, attempt_context=None)

    assert omitted == explicit_none
    assert '# Verify Attempt' not in omitted


@pytest.mark.asyncio
async def test_first_attempt_is_numbered_and_called_first(briefing):
    prompt = await _prompt(
        briefing, attempt_context=VerifyAttemptContext(attempt=1, max_attempts=5),
    )

    assert '# Verify Attempt' in prompt
    assert 'Verify attempt 1 of 5 failed' in prompt
    assert 'first failed attempt in this verify/debug loop' in prompt
    section_at = prompt.index('# Verify Attempt')
    assert prompt.index('# Task Context') < section_at < prompt.index('# Failures')


@pytest.mark.asyncio
async def test_later_attempt_lists_prior_causes_oldest_first(briefing):
    prompt = await _prompt(briefing, attempt_context=VerifyAttemptContext(
        attempt=3, max_attempts=5,
        prior_causes=('test_failure: FAILED a', 'lint_failure: F401 b'),
    ))

    assert 'Verify attempt 3 of 5 failed' in prompt
    first = prompt.index('- attempt 1: test_failure: FAILED a')
    second = prompt.index('- attempt 2: lint_failure: F401 b')
    assert first < second
    assert 'did not work' in prompt
    assert 'first failed attempt' not in prompt


@pytest.mark.asyncio
async def test_first_and_third_attempt_prompts_differ(briefing):
    first = await _prompt(
        briefing, attempt_context=VerifyAttemptContext(attempt=1, max_attempts=5),
    )
    third = await _prompt(briefing, attempt_context=VerifyAttemptContext(
        attempt=3, max_attempts=5,
        prior_causes=('test_failure: FAILED a', 'test_failure: FAILED a'),
    ))

    assert first != third


def test_verify_attempt_context_is_immutable():
    ctx = VerifyAttemptContext(attempt=1, max_attempts=5)

    pytest.raises(dataclasses.FrozenInstanceError, setattr, ctx, 'attempt', 2)
