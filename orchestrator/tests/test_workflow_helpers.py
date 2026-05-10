"""Contract + smoke tests for the _workflow_helpers module.

Verifies that the 5 shared workflow test-helpers are importable from
_workflow_helpers and behave as expected after extraction from
test_workflow_e2e.py (task 1195).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_scheduler_smoke() -> None:
    """FakeScheduler tracks status changes correctly."""
    from _workflow_helpers import FakeScheduler  # noqa: PLC0415

    sched = FakeScheduler()
    await sched.set_task_status('x', 'pending')
    await sched.set_task_status('x', 'done')
    assert await sched.get_status('x') == 'done'
    assert sched.statuses == {'x': ['pending', 'done']}


def test_fake_mcp_smoke() -> None:
    """FakeMcp returns the expected URL and empty MCP config."""
    from _workflow_helpers import FakeMcp  # noqa: PLC0415

    mcp = FakeMcp()
    assert mcp.url == 'http://localhost:9999'
    assert mcp.mcp_config_json() == {'mcpServers': {}}


@pytest.mark.asyncio
async def test_fake_briefing_smoke() -> None:
    """FakeBriefing returns non-empty strings for all 7 prompt-builder methods."""
    from _workflow_helpers import FakeBriefing  # noqa: PLC0415

    briefing = FakeBriefing()

    impl_prompt = await briefing.build_implementer_prompt({'title': 't'}, [])
    assert isinstance(impl_prompt, str) and impl_prompt, 'implementer prompt must be non-empty'

    arch_prompt = await briefing.build_architect_prompt({'title': 'my task'})
    assert isinstance(arch_prompt, str) and arch_prompt, 'architect prompt must be non-empty'
    assert 'my task' in arch_prompt, 'architect prompt must interpolate title'

    debug_prompt = await briefing.build_debugger_prompt('test failure msg', {})
    assert isinstance(debug_prompt, str) and debug_prompt, 'debugger prompt must be non-empty'
    assert 'test failure msg' in debug_prompt, 'debugger prompt must interpolate failures'

    review_prompt = await briefing.build_reviewer_prompt('comprehensive', 'some diff')
    assert isinstance(review_prompt, str) and review_prompt, 'reviewer prompt must be non-empty'
    assert 'comprehensive' in review_prompt, 'reviewer prompt must interpolate reviewer_type'

    judge_prompt = await briefing.build_completion_judge_prompt(
        {'steps': [1, 2]}, [], 'some diff', task_id='42'
    )
    assert isinstance(judge_prompt, str) and judge_prompt, 'completion-judge prompt must be non-empty'
    assert '42' in judge_prompt, 'completion-judge prompt must interpolate task_id'

    merge_prompt = await briefing.build_merger_prompt('conflict text', 'intent text')
    assert isinstance(merge_prompt, str) and merge_prompt, 'merger prompt must be non-empty'
    assert 'conflict text' in merge_prompt, 'merger prompt must interpolate conflicts'

    resume_prompt = await briefing.build_resume_prompt({}, {}, 'the summary', 'the resolution')
    assert isinstance(resume_prompt, str) and resume_prompt, 'resume prompt must be non-empty'
    assert 'the resolution' in resume_prompt, 'resume prompt must interpolate resolution'
