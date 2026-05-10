"""Smoke tests for the _workflow_helpers module.

Verifies that FakeScheduler, FakeMcp, and FakeBriefing behave as expected.
Import-contract verification for _make_resolving_steward and
_make_status_setting_steward relies on indirect coverage from
test_workflow_e2e.py and test_workflow_status_on_resume.py.
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
    """FakeBriefing returns a non-empty string for each of the 7 prompt-builder methods.

    Asserts shape (non-empty str) and that each method threads its key input
    argument into the output.  Literal wording tokens (e.g. 'Plan task:') are
    not checked — only the caller-supplied argument value must appear in the
    result, guarding against a builder silently dropping its argument.
    """
    from _workflow_helpers import FakeBriefing  # noqa: PLC0415

    briefing = FakeBriefing()

    # build_implementer_prompt returns a fixed string ('Implement the plan')
    # without interpolating its arguments, so only a shape check applies here.
    impl_prompt = await briefing.build_implementer_prompt({'title': 't'}, [])
    assert isinstance(impl_prompt, str) and impl_prompt, 'implementer prompt must be non-empty'

    arch_prompt = await briefing.build_architect_prompt({'title': 'my task'})
    assert isinstance(arch_prompt, str) and arch_prompt, 'architect prompt must be non-empty'
    assert 'my task' in arch_prompt, 'architect prompt must include the task title'

    debug_prompt = await briefing.build_debugger_prompt('test failure msg', {})
    assert isinstance(debug_prompt, str) and debug_prompt, 'debugger prompt must be non-empty'
    assert 'test failure msg' in debug_prompt, 'debugger prompt must include the failure text'

    review_prompt = await briefing.build_reviewer_prompt('comprehensive', 'some diff')
    assert isinstance(review_prompt, str) and review_prompt, 'reviewer prompt must be non-empty'
    assert 'comprehensive' in review_prompt, 'reviewer prompt must include the reviewer_type'

    judge_prompt = await briefing.build_completion_judge_prompt(
        {'steps': [1, 2]}, [], 'some diff', task_id='42'
    )
    assert isinstance(judge_prompt, str) and judge_prompt, 'completion-judge prompt must be non-empty'
    assert '42' in judge_prompt, 'completion-judge prompt must include the task_id'

    merge_prompt = await briefing.build_merger_prompt('conflict text', 'intent text')
    assert isinstance(merge_prompt, str) and merge_prompt, 'merger prompt must be non-empty'
    assert 'conflict text' in merge_prompt, 'merger prompt must include the conflict text'

    resume_prompt = await briefing.build_resume_prompt({}, {}, 'the summary', 'the resolution')
    assert isinstance(resume_prompt, str) and resume_prompt, 'resume prompt must be non-empty'
    assert 'the resolution' in resume_prompt, 'resume prompt must include the resolution'
