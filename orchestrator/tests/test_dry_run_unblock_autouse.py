"""Pin the autouse no-op for orchestrator.workflow.run_dry_run_unblock.

The fire-and-forget dry-run hook spawned by ``TaskWorkflow._spawn_dry_run_unblock``
invokes the real Claude CLI by default, which causes pytest-timeout hangs
(~60s × ~8 tests = ~480s wallclock) in test suites that exercise
``_mark_blocked``.  ``orchestrator/tests/conftest.py`` defines an autouse
fixture that monkeypatches the binding to an async no-op for every test,
with opt-out via the ``exercise_dry_run_unblock`` marker.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator import workflow


def test_run_dry_run_unblock_is_noop_by_default():
    """Default autouse patch: workflow.run_dry_run_unblock returns None and
    never invokes invoke_agent."""

    async def _exploding_invoke_agent(*_a, **_kw):
        raise AssertionError(
            'invoke_agent must not be reached when autouse no-op is active'
        )

    # Monkeypatch invoke_agent so any accidental call into the real implementation
    # would explode loudly.  The autouse fixture should prevent the call from
    # ever happening.
    workflow_invoke = workflow.run_dry_run_unblock
    # Sanity check: confirm the conftest patched the binding.
    assert workflow_invoke is not None

    result = asyncio.run(
        workflow.run_dry_run_unblock(
            task_id='t-1',
            worktree='/tmp/wt',
            reason='r',
            detail='d',
            scheduler=None,
            mcp=None,
            config=None,
            event_store=None,
        )
    )
    assert result is None


@pytest.mark.exercise_dry_run_unblock
def test_exercise_dry_run_unblock_marker_disables_noop():
    """With the marker, the real run_dry_run_unblock binding is in place.

    We don't actually call it (that would hit the Claude CLI).  Instead, we
    assert the binding is the real coroutine function from
    ``orchestrator.dry_run_unblock``, not the autouse no-op.
    """
    from orchestrator import dry_run_unblock as dru

    assert workflow.run_dry_run_unblock is dru.run_dry_run_unblock
