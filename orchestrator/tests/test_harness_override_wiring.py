"""Integration tests that Harness wires OverrideStore into Scheduler correctly.

These tests verify *behaviour* through the public scheduler API: a pin
pre-written to config.overrides_db_path drives Harness scheduler's dispatch
order, proving the OverrideStore reaches the right SQLite file AND that
Scheduler.acquire_next() consults overrides on its selection path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.overrides import OverrideStore


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


class TestHarnessOverrideStoreIntegration:
    @pytest.mark.asyncio
    async def test_scheduler_dispatches_pinned_override_through_canonical_db_path(
        self, config: OrchestratorConfig, forbid_live_mcp
    ):
        """A pin pre-written to config.overrides_db_path drives Harness
        scheduler's dispatch order — proves the OverrideStore reaches the
        right SQLite file AND that Scheduler.acquire_next() consults overrides
        on its selection path.

        Without override wiring: task-critical (critical priority) would score
        higher and be dispatched first.
        With correct wiring: task-pin (pinned, low priority) dispatches first
        because the pin-dispatch loop in acquire_next() runs before scoring.
        """
        project_root_str = str(config.project_root)

        # Pre-populate a pin at the canonical DB path using a separate store
        # instance that the Harness cannot share.
        seed_store = OverrideStore(config.overrides_db_path)
        seed_store.set_override(project_root_str, "task-pin", pinned=True)

        # Harness constructs its own OverrideStore at the same path.
        harness = Harness(config)
        injected_session = HermeticMcpSession()
        harness.scheduler._mcp_session = injected_session
        # This test instance-mocks get_tasks and seeds empty-dependency tasks
        # below, so it is data-conditionally hermetic regardless of the
        # injected session — forbid_live_mcp's network spy alone can't prove
        # the injection above actually took effect. This identity check only
        # guards against the attribute assignment above being silently lost
        # (e.g. a future refactor of Harness construction clobbering it); it
        # does NOT prove dispatch_tool actually routes through the injected
        # session for this test. That routing guarantee is established by
        # test_hermetic_mcp_session.py, backed by forbid_live_mcp.
        assert harness.scheduler._mcp_session is injected_session
        # This test drives acquire_next() directly (bypassing Harness.run()'s
        # startup sweeps), so it must call finish_startup() itself (task 2235).
        harness.scheduler.finish_startup()

        # Two competing tasks: pinned-low vs unpinned-critical.
        # Without overrides, task-critical would win on score alone.
        # Only the pin-dispatch path can produce the asserted outcome.
        pinned_task = {
            "id": "task-pin",
            "title": "Task pin (pinned, low)",
            "status": "pending",
            "dependencies": [],
            "metadata": {"files": ["pin/src"]},
            "priority": "low",
        }
        critical_task = {
            "id": "task-critical",
            "title": "Task critical (critical, not pinned)",
            "status": "pending",
            "dependencies": [],
            "metadata": {"files": ["crit/src"]},
            "priority": "critical",
        }
        harness.scheduler.get_tasks = AsyncMock(
            return_value=[pinned_task, critical_task]
        )

        assignment = await harness.scheduler.acquire_next()

        assert assignment is not None, (
            "acquire_next() returned None — possible scheduling or wiring bug"
            " unrelated to overrides"
        )
        assert assignment.task_id == "task-pin", (
            f"Expected pinned task 'task-pin' to dispatch first, got"
            f" {assignment.task_id!r}. Failure modes: (1) OverrideStore not"
            " wired into Scheduler's selection path, or (2) canonical DB path"
            " mismatch between Harness construction and the seed write."
        )
