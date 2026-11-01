"""The escalation-watcher-auto rotation's cap-hit policy (task 4837).

A rotation is a stateless triage loop over a queue it re-reads every cycle,
so a capped rotation must retry FRESH: its own prompt and charter, never a
``--resume`` carrying only a continuation prompt.  The retry loop is also
count-bounded, so a pathological cap storm surfaces as
``AllAccountsCappedException`` to the supervisor rather than spinning
invisibly inside one call.  Driven through ``Harness._run_watcher_rotation``
with the real ``invoke_with_cap_retry``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.testing import make_gate_mock

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import _WATCHER_MAX_CAP_RETRIES, Harness

_INVOKE_PATCH = 'orchestrator.harness.invoke_agent'
_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'


def _rotation_harness(tmp_path: Path, gate: MagicMock) -> Harness:
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig(project_root=tmp_path)
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits = deque()
    h.usage_gate = gate
    h.cost_store = MagicMock(save_invocation=AsyncMock())
    h._run_id = 'run-cap-policy'
    h.mcp = MagicMock()
    h.mcp.mcp_config_json.return_value = {'mcp': 'sentinel'}
    return h


def _capped(session_id: str) -> AgentResult:
    return AgentResult(success=False, output='usage limit', cost_usd=0.5, session_id=session_id)


class TestWatcherRotationCapPolicy:
    @pytest.mark.asyncio
    async def test_capped_rotation_retries_fresh_not_resumed(self, tmp_path: Path) -> None:
        gate = make_gate_mock(
            account_count=2,
            detect_cap_hit=MagicMock(side_effect=[True, False]),
        )
        h = _rotation_harness(tmp_path, gate)
        digest = AgentResult(
            success=True, output='digest', cost_usd=0.5, turns=3, duration_ms=9000,
        )

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[_capped('sess-w'), digest]) as invoke_agent,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await h._run_watcher_rotation()

        first, retry = (c.kwargs for c in invoke_agent.call_args_list)
        assert 'ROTATION_ESCALATIONS=' in first['prompt']
        assert 'resume_session_id' not in retry
        assert retry['prompt'] == first['prompt']
        assert retry['system_prompt'] == first['system_prompt']

    @pytest.mark.asyncio
    async def test_cap_retry_loop_is_bounded(self, tmp_path: Path) -> None:
        gate = make_gate_mock(
            account_count=2,
            detect_cap_hit=MagicMock(return_value=True),
        )
        h = _rotation_harness(tmp_path, gate)

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[_capped(f'sess-{i}') for i in range(100)]) as invoke_agent,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(AllAccountsCappedException) as ei,
        ):
            await h._run_watcher_rotation()

        assert ei.value.retries == invoke_agent.await_count
        assert ei.value.retries == _WATCHER_MAX_CAP_RETRIES
