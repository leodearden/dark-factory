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
    """FakeBriefing returns a non-empty string for each of the 7 prompt-builder methods.

    Asserts shape only (non-empty str).  Substring/interpolation checks are
    omitted deliberately — they pin internal f-string wording that no caller
    observes and break on harmless refactors.  The real workflow tests in
    test_workflow_e2e.py and test_workflow_status_on_resume.py exercise
    FakeBriefing in its actual call path and form the true contract.
    """
    from _workflow_helpers import FakeBriefing  # noqa: PLC0415

    briefing = FakeBriefing()

    impl_prompt = await briefing.build_implementer_prompt({'title': 't'}, [])
    assert isinstance(impl_prompt, str) and impl_prompt, 'implementer prompt must be non-empty'

    arch_prompt = await briefing.build_architect_prompt({'title': 'my task'})
    assert isinstance(arch_prompt, str) and arch_prompt, 'architect prompt must be non-empty'

    debug_prompt = await briefing.build_debugger_prompt('test failure msg', {})
    assert isinstance(debug_prompt, str) and debug_prompt, 'debugger prompt must be non-empty'

    review_prompt = await briefing.build_reviewer_prompt('comprehensive', 'some diff')
    assert isinstance(review_prompt, str) and review_prompt, 'reviewer prompt must be non-empty'

    judge_prompt = await briefing.build_completion_judge_prompt(
        {'steps': [1, 2]}, [], 'some diff', task_id='42'
    )
    assert isinstance(judge_prompt, str) and judge_prompt, 'completion-judge prompt must be non-empty'

    merge_prompt = await briefing.build_merger_prompt('conflict text', 'intent text')
    assert isinstance(merge_prompt, str) and merge_prompt, 'merger prompt must be non-empty'

    resume_prompt = await briefing.build_resume_prompt({}, {}, 'the summary', 'the resolution')
    assert isinstance(resume_prompt, str) and resume_prompt, 'resume prompt must be non-empty'
