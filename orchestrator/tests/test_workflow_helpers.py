"""Contract + smoke tests for the _workflow_helpers module.

Verifies that the 5 shared workflow test-helpers are importable from
_workflow_helpers and behave as expected after extraction from
test_workflow_e2e.py (task 1195).
"""

from __future__ import annotations

import inspect
from pathlib import Path

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


def test_make_resolving_steward_returns_class(tmp_path: Path) -> None:
    """_make_resolving_steward returns a class with async start/stop methods."""
    from _workflow_helpers import _make_resolving_steward  # noqa: PLC0415

    queue = EscalationQueue(tmp_path / 'q')
    steward_cls = _make_resolving_steward(queue, '42')
    assert inspect.isclass(steward_cls), '_make_resolving_steward must return a class'
    assert inspect.iscoroutinefunction(steward_cls.start), 'start must be an async method'
    assert inspect.iscoroutinefunction(steward_cls.stop), 'stop must be an async method'


def test_make_status_setting_steward_returns_class(tmp_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# Identity test — test_workflow_e2e must re-export from _workflow_helpers
# ---------------------------------------------------------------------------


def test_test_workflow_e2e_uses_canonical_helpers() -> None:
    """Each helper on test_workflow_e2e is the same object as on _workflow_helpers.

    Fails if test_workflow_e2e re-creates a local definition instead of importing
    from _workflow_helpers — duplicate definitions diverge silently.
    """
    import _workflow_helpers  # noqa: PLC0415
    import test_workflow_e2e  # noqa: PLC0415

    assert test_workflow_e2e.FakeMcp is _workflow_helpers.FakeMcp, (
        'test_workflow_e2e.FakeMcp must be re-exported from _workflow_helpers, '
        'not redefined locally — duplicate definitions diverge silently'
    )
    assert test_workflow_e2e.FakeScheduler is _workflow_helpers.FakeScheduler, (
        'test_workflow_e2e.FakeScheduler must be re-exported from _workflow_helpers, '
        'not redefined locally — duplicate definitions diverge silently'
    )
    assert test_workflow_e2e.FakeBriefing is _workflow_helpers.FakeBriefing, (
        'test_workflow_e2e.FakeBriefing must be re-exported from _workflow_helpers, '
        'not redefined locally — duplicate definitions diverge silently'
    )
    assert test_workflow_e2e._make_resolving_steward is _workflow_helpers._make_resolving_steward, (
        'test_workflow_e2e._make_resolving_steward must be re-exported from _workflow_helpers, '
        'not redefined locally — duplicate definitions diverge silently'
    )
    assert test_workflow_e2e._make_status_setting_steward is _workflow_helpers._make_status_setting_steward, (
        'test_workflow_e2e._make_status_setting_steward must be re-exported from _workflow_helpers, '
        'not redefined locally — duplicate definitions diverge silently'
    )


# ---------------------------------------------------------------------------
# Source-content test — test_workflow_status_on_resume must not cross-import
# ---------------------------------------------------------------------------


def test_status_on_resume_uses_workflow_helpers_not_e2e() -> None:
    """test_workflow_status_on_resume.py must import from _workflow_helpers, not test_workflow_e2e.

    Uses the same re.compile(...).search(...) pattern as
    tests/scripts/test_pytest_workspace_collection.py to enforce module decoupling.
    Asserts:
      (a) 'from _workflow_helpers import' is present
      (b) 'from test_workflow_e2e import' is absent
      (c) orchestrator imports come after _workflow_helpers import (first-party-last)
    """
    import re  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    source = (Path(__file__).parent / 'test_workflow_status_on_resume.py').read_text()

    # Positive: must import from _workflow_helpers
    assert re.search(r'^from _workflow_helpers import\b', source, re.MULTILINE), (
        'test_workflow_status_on_resume.py must import workflow fakes from '
        '_workflow_helpers (the canonical source). Cross-test-file imports are '
        'fragile — see task 1195.'
    )

    # Negative: must not cross-import from sibling test module
    assert not re.search(r'^from test_workflow_e2e import\b', source, re.MULTILINE), (
        'test_workflow_status_on_resume.py must NOT import from sibling test module '
        'test_workflow_e2e. Use _workflow_helpers instead. Task 1195 extracted the '
        'shared helpers to break this coupling.'
    )

    # Ordering: orchestrator imports must come after _workflow_helpers import
    lines = source.splitlines()
    helpers_line = next(
        (i for i, ln in enumerate(lines) if re.match(r'^from _workflow_helpers import\b', ln)),
        None,
    )
    orchestrator_line = next(
        (i for i, ln in enumerate(lines) if re.match(r'^from orchestrator\b', ln)),
        None,
    )
    if helpers_line is not None and orchestrator_line is not None:
        assert orchestrator_line > helpers_line, (
            'orchestrator imports must come AFTER _workflow_helpers import in '
            'test_workflow_status_on_resume.py (first-party-orchestrator-last grouping). '
            f'_workflow_helpers at line {helpers_line + 1}, orchestrator at line {orchestrator_line + 1}.'
        )
