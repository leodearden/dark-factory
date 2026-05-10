"""Contract + smoke tests for the _workflow_helpers module.

Verifies that the 5 shared workflow test-helpers are importable from
_workflow_helpers and behave as expected after extraction from
test_workflow_e2e.py (task 1195).
"""

from __future__ import annotations

import inspect

import pytest
from escalation.queue import EscalationQueue


# ---------------------------------------------------------------------------
# Import-contract test
# ---------------------------------------------------------------------------


def test_imports_resolve() -> None:
    """All 5 helpers are importable from _workflow_helpers without ImportError."""
    from _workflow_helpers import (  # noqa: PLC0415
        FakeBriefing,
        FakeMcp,
        FakeScheduler,
        _make_resolving_steward,
        _make_status_setting_steward,
    )

    assert inspect.isclass(FakeBriefing), 'FakeBriefing must be a class'
    assert inspect.isclass(FakeMcp), 'FakeMcp must be a class'
    assert inspect.isclass(FakeScheduler), 'FakeScheduler must be a class'
    assert callable(_make_resolving_steward), '_make_resolving_steward must be callable'
    assert callable(_make_status_setting_steward), '_make_status_setting_steward must be callable'


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
    """FakeBriefing returns non-empty strings for implementer and architect prompts."""
    from _workflow_helpers import FakeBriefing  # noqa: PLC0415

    briefing = FakeBriefing()
    impl_prompt = await briefing.build_implementer_prompt({'title': 't'}, [])
    arch_prompt = await briefing.build_architect_prompt({'title': 't'})
    assert isinstance(impl_prompt, str) and impl_prompt, 'implementer prompt must be non-empty'
    assert isinstance(arch_prompt, str) and arch_prompt, 'architect prompt must be non-empty'


def test_make_resolving_steward_returns_class(tmp_path: pytest.TempPathFactory) -> None:
    """_make_resolving_steward returns a class with async start/stop methods."""
    from _workflow_helpers import _make_resolving_steward  # noqa: PLC0415

    queue = EscalationQueue(tmp_path / 'q')
    steward_cls = _make_resolving_steward(queue, '42')
    assert inspect.isclass(steward_cls), '_make_resolving_steward must return a class'
    assert inspect.iscoroutinefunction(steward_cls.start), 'start must be an async method'
    assert inspect.iscoroutinefunction(steward_cls.stop), 'stop must be an async method'


def test_make_status_setting_steward_returns_class(tmp_path: pytest.TempPathFactory) -> None:
    """_make_status_setting_steward returns a class with async start/stop methods."""
    from _workflow_helpers import (  # noqa: PLC0415
        FakeScheduler,
        _make_status_setting_steward,
    )

    queue = EscalationQueue(tmp_path / 'q')
    sched = FakeScheduler()
    steward_cls = _make_status_setting_steward(queue, sched, '42', 'deferred')
    assert inspect.isclass(steward_cls), '_make_status_setting_steward must return a class'
    assert inspect.iscoroutinefunction(steward_cls.start), 'start must be an async method'
    assert inspect.iscoroutinefunction(steward_cls.stop), 'stop must be an async method'
