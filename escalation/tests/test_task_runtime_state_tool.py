"""Tests for the get_task_runtime_state escalation MCP tool.

PRD ``plans/dashboard-task-runtime-endpoint-prd.md`` task beta. The tool is
a thin, sync, duck-typed projection of ``Harness.task_runtime_snapshot()``
(task alpha, task 2634) into the shared ``shared.task_runtime_state`` wire
model — see that module's docstring for the honest-failure contract this
test suite exercises.

Uses the sync-tool FastMCP unit-test pattern already established for
``get_merge_queue``/``get_merge_halt_status`` (test_server.py:2629-2630 /
82-84): ``tool = await server.get_tool(name); tool.fn()`` — no ``await`` on
``.fn()`` itself since the tool body is a sync ``def``. The fake harness is
a ``types.SimpleNamespace`` whose ``task_runtime_snapshot`` returns
``SimpleNamespace`` stand-ins for ``orchestrator.task_runtime.TaskRuntimeState``
— no orchestrator import needed, matching the escalation package's
reverse-dep discipline.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server


async def _call_get_task_runtime_state(server: Any) -> dict[str, Any]:
    """Invoke the get_task_runtime_state MCP tool directly (sync tool)."""
    tool = await server.get_tool('get_task_runtime_state')
    return tool.fn()


# A pooled (lane-assigned) success entry — all 9 TaskRuntimeState attributes.
_POOLED_ENTRY = types.SimpleNamespace(
    task_id=42,
    has_worktree=True,
    loops=3,
    attempts=1,
    started='2026-07-16T00:00:00+00:00',
    lane='_lane-3',
    phase='EXECUTE',
    lane_state='assigned',
    error=None,
)
_POOLED_EXPECTED = {
    'task_id': 42,
    'has_worktree': True,
    'loops': 3,
    'attempts': 1,
    'started': '2026-07-16T00:00:00+00:00',
    'lane': '_lane-3',
    'phase': 'EXECUTE',
    'lane_state': 'assigned',
    'error': None,
}

# A non-pooled success entry — lane/lane_state are None, distinct from a
# lane-assigned task.
_NON_POOLED_ENTRY = types.SimpleNamespace(
    task_id=7,
    has_worktree=True,
    loops=2,
    attempts=0,
    started='2026-07-16T00:00:00+00:00',
    lane=None,
    phase='PLAN',
    lane_state=None,
    error=None,
)
_NON_POOLED_EXPECTED = {
    'task_id': 7,
    'has_worktree': True,
    'loops': 2,
    'attempts': 0,
    'started': '2026-07-16T00:00:00+00:00',
    'lane': None,
    'phase': 'PLAN',
    'lane_state': None,
    'error': None,
}

# A per-task artifact READ FAILURE entry — loops/attempts/phase/started are
# None (never fabricated to 0/'PLAN'); lane/lane_state/has_worktree stay
# populated since they are sourced independently of the failed artifact read
# (mirrors orchestrator.task_runtime.TaskRuntimeState's docstring).
_HONEST_FAILURE_ENTRY = types.SimpleNamespace(
    task_id=13,
    has_worktree=True,
    loops=None,
    attempts=None,
    started=None,
    lane='_lane-9',
    phase=None,
    lane_state='quarantined',
    error='RuntimeError: boom',
)
_HONEST_FAILURE_EXPECTED = {
    'task_id': 13,
    'has_worktree': True,
    'loops': None,
    'attempts': None,
    'started': None,
    'lane': '_lane-9',
    'phase': None,
    'lane_state': 'quarantined',
    'error': 'RuntimeError: boom',
}

# A stand-in that omits `error` entirely (every other fixture in this suite
# sets it explicitly) -- exercises the tool's getattr(s, 'error', None)
# fallback, the only reason getattr is used instead of a direct s.error read
# like the other 8 fields.
_NO_ERROR_ATTR_ENTRY = types.SimpleNamespace(
    task_id=21,
    has_worktree=False,
    loops=0,
    attempts=0,
    started=None,
    lane=None,
    phase='PLAN',
    lane_state=None,
)

# An out-of-vocab `phase` -- today's orchestrator source (_derive_phase)
# only ever emits 'PLAN'/'EXECUTE'/'DONE', but the source types phase as
# free-form str while the wire model narrows it to a Literal. Pins the
# tool's behavior at that boundary: isolated per-task degradation, not a
# whole-snapshot failure.
_BOGUS_PHASE_ENTRY = types.SimpleNamespace(
    task_id=99,
    has_worktree=True,
    loops=5,
    attempts=2,
    started='2026-07-16T00:00:00+00:00',
    lane='_lane-1',
    phase='VERIFY',
    lane_state='assigned',
    error=None,
)

# An out-of-vocab `lane_state` -- today's _LANE_STATE_MAP only ever emits
# 'assigned'/'quarantined'/'released', same boundary concern as phase above.
_BOGUS_LANE_STATE_ENTRY = types.SimpleNamespace(
    task_id=100,
    has_worktree=True,
    loops=1,
    attempts=0,
    started='2026-07-16T00:00:00+00:00',
    lane='_lane-2',
    phase='PLAN',
    lane_state='weird',
    error=None,
)


@pytest.mark.asyncio
class TestGetTaskRuntimeStateTool:
    """get_task_runtime_state — registration, standalone guard, projection."""

    async def test_tool_is_registered(self, tmp_path: Path):
        server = create_server(EscalationQueue(tmp_path / 'esc'))
        tool = await server.get_tool('get_task_runtime_state')
        assert tool is not None

    async def test_harness_none_returns_legible_empty_shape(self, tmp_path: Path):
        """Standalone server (no harness wired) never crashes — it returns
        the model's default-constructed empty envelope."""
        server = create_server(EscalationQueue(tmp_path / 'esc'))
        result = await _call_get_task_runtime_state(server)
        # offline_reason is dashboard-synthesized (task 3517): the server
        # itself always emits None, exactly like offline=False.
        assert result == {'offline': False, 'offline_reason': None, 'tasks': []}

    async def test_projects_pooled_and_non_pooled_entries(self, tmp_path: Path):
        harness = types.SimpleNamespace(
            task_runtime_snapshot=lambda: [_POOLED_ENTRY, _NON_POOLED_ENTRY],
        )
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
        result = await _call_get_task_runtime_state(server)
        assert result['offline'] is False
        assert result['tasks'] == [_POOLED_EXPECTED, _NON_POOLED_EXPECTED]

    async def test_projects_honest_failure_entry_without_coercion(self, tmp_path: Path):
        harness = types.SimpleNamespace(
            task_runtime_snapshot=lambda: [_HONEST_FAILURE_ENTRY],
        )
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
        result = await _call_get_task_runtime_state(server)
        assert result['tasks'] == [_HONEST_FAILURE_EXPECTED]
        entry = result['tasks'][0]
        # Explicit not-coerced assertions (belt-and-braces on top of the
        # dict-equality check above): a read failure must never be silently
        # fabricated into an honest-looking 0 / 'PLAN'.
        assert entry['loops'] is None
        assert entry['attempts'] is None
        assert entry['phase'] is None
        assert entry['started'] is None
        assert entry['error'] == 'RuntimeError: boom'

    async def test_missing_error_attribute_defaults_to_none(self, tmp_path: Path):
        """A snapshot stand-in that omits `error` entirely exercises the
        tool's getattr(s, 'error', None) fallback (the other 8 fields are
        read directly as `s.<field>`, so this is the one path where a
        missing attribute must not raise AttributeError)."""
        harness = types.SimpleNamespace(
            task_runtime_snapshot=lambda: [_NO_ERROR_ATTR_ENTRY],
        )
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
        result = await _call_get_task_runtime_state(server)
        assert result['tasks'][0]['error'] is None

    async def test_out_of_vocab_phase_degrades_to_isolated_error_entry(self, tmp_path: Path):
        """A `phase` outside the wire model's Literal vocabulary (the
        orchestrator source types it as free-form str -- see
        `_project_task_runtime_entry`'s docstring) must not raise and take
        down the whole snapshot. The offending task degrades to an honest
        per-task error entry; a healthy task in the same call is unaffected."""
        harness = types.SimpleNamespace(
            task_runtime_snapshot=lambda: [_BOGUS_PHASE_ENTRY, _POOLED_ENTRY],
        )
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
        result = await _call_get_task_runtime_state(server)
        assert result['offline'] is False
        bogus, healthy = result['tasks']
        assert bogus['task_id'] == 99
        assert bogus['has_worktree'] is True
        assert bogus['loops'] is None
        assert bogus['attempts'] is None
        assert bogus['started'] is None
        assert bogus['lane'] is None
        assert bogus['phase'] is None
        assert bogus['lane_state'] is None
        assert bogus['error'] is not None
        assert 'wire-contract violation' in bogus['error']
        # Isolation: the next task in the same snapshot still renders normally.
        assert healthy == _POOLED_EXPECTED

    async def test_out_of_vocab_lane_state_degrades_to_isolated_error_entry(self, tmp_path: Path):
        """Same boundary as phase above, for `lane_state`."""
        harness = types.SimpleNamespace(
            task_runtime_snapshot=lambda: [_BOGUS_LANE_STATE_ENTRY],
        )
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
        result = await _call_get_task_runtime_state(server)
        bogus = result['tasks'][0]
        assert bogus['task_id'] == 100
        assert bogus['has_worktree'] is True
        assert bogus['loops'] is None
        assert bogus['attempts'] is None
        assert bogus['started'] is None
        assert bogus['lane'] is None
        assert bogus['phase'] is None
        assert bogus['lane_state'] is None
        assert bogus['error'] is not None
        assert 'wire-contract violation' in bogus['error']
