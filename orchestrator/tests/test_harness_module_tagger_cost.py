"""Tests for module_tagger cost-store threading (task 2534 δ step-9/10).

``Harness._tag_task_modules``'s ``invoke_with_cap_retry`` call (harness.py
~2036, labeled 'Module tagging') does not currently pass
cost_store/run_id/task_id/project_id/role, so its invocations are never
recorded in the ``invocations`` table — making module_tagger invisible to
``digest.model_role_rollup`` / the dashboard's per-(model×role) rollup
(unlike unblock_auto, which dry_run_unblock.py already threads this way).

This drives threading ``cost_store=self.cost_store``, ``run_id=self._run_id
or ''``, ``task_id=None``, ``project_id=self.config.fused_memory.project_id``,
``role='module_tagger'`` onto that call, mirroring dry_run_unblock.py's
unblock_auto site (shared/src/shared/cli_invoke.py's
``invoke_with_cap_retry`` records via ``cost_store.save_invocation`` whenever
``cost_store`` is supplied).

Reuses the ``_cost_store_factory`` fixture from test_harness_park_stop.py
(step-19/20) for a real temp CostStore, and the SAMPLE_TASKS /
AGENT_MODULE_RESPONSE / invoke_agent-patch pattern from
test_harness_module_tagging.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from test_harness_park_stop import _cost_store_factory  # noqa: F401 -- reused cross-module fixture

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness

SAMPLE_TASKS = [
    {
        'id': '1',
        'title': 'Add health endpoint',
        'description': 'Add GET /health to the server',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
    {
        'id': '2',
        'title': 'Refactor config schema',
        'description': 'Split config into sub-models',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
]

AGENT_MODULE_RESPONSE = {
    'tasks': [
        {'id': '1', 'files': ['src/server/app.py']},
        {'id': '2', 'files': ['src/config/schema.py']},
    ],
}


def _fetch_invocations(db_path: Path, role: str) -> list[dict]:
    """Read invocations rows for *role* directly (CostStore has no query API)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT * FROM invocations WHERE role = ?', (role,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


@pytest.fixture
def harness(config: OrchestratorConfig) -> Harness:
    return Harness(config)


class TestModuleTaggerCostThreading:
    """_tag_task_modules must record its invocation in the CostStore (step-9/10)."""

    @pytest.mark.asyncio
    async def test_module_tagger_records_invocation_with_cost(
        self, harness, config, _cost_store_factory,
    ):
        """A successful module-tagging invocation lands in `invocations`.

        Before step-10's threading, no cost_store is passed to
        invoke_with_cap_retry, so this query returns zero rows.
        """
        harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
        harness.scheduler.update_task = AsyncMock()

        # cost_store starts as None (Harness.__init__) — open a real temp one,
        # as production's run() does against runs.db (harness.py:1431-1436).
        harness.cost_store = await _cost_store_factory('runs.db')
        harness._run_id = 'run-test-0001'

        agent_result = AgentResult(
            success=True,
            output=json.dumps(AGENT_MODULE_RESPONSE),
            structured_output=AGENT_MODULE_RESPONSE,
            cost_usd=0.0321, duration_ms=6000, turns=2,
        )

        with patch(
            'orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)
        ):
            await harness._tag_task_modules()

        rows = _fetch_invocations(harness.cost_store.db_path, 'module_tagger')
        assert len(rows) == 1, f'Expected exactly 1 module_tagger invocation row; got {rows}'
        row = rows[0]
        assert row['role'] == 'module_tagger'
        assert row['project_id'] == config.fused_memory.project_id, row
        assert row['run_id'] == 'run-test-0001', row
        assert row['cost_usd'] == pytest.approx(0.0321), row
        # No single task owns a module-tagging invocation (it tags a whole
        # batch) — invoke_with_cap_retry's `task_id or None` normalization
        # turns the omitted/default task_id into a NULL row, matching the
        # module_tagger fixture in test_digest.py's rollup tests (step-1).
        assert row['task_id'] is None, row
