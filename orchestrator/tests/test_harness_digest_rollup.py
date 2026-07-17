"""Tests for Harness._maybe_write_digest's per-(model×role) rollup wiring.

Task 2534 δ (Phase-1 substrate of plans/adaptive-model-routing-prd.md),
boundary test 12: the digest written by ``_maybe_write_digest`` must include
the '## Per-(model×role) rollup' section produced by
``digest.model_role_rollup``, sourced from the SAME runs.db file backing
``self.event_store`` — in production EventStore/CostStore/RunStore all share
one runs.db file (harness.py:1373-1398), so the rollup query needs no
separate DB handle.

Reuses the ``_make_harness_with_mocks`` / ``_cost_store_factory`` fixture
pattern from test_harness_park_stop.py's ``TestHarnessMaybeWriteDigest``
(step-19/20), and the raw-SQLite seeding helpers from test_digest.py's
model×role rollup tests (step-1).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_digest import _insert_invocation, _insert_task_result
from test_harness_park_stop import (
    _cost_store_factory,  # noqa: F401 -- reused cross-module fixture
    _make_harness_with_mocks,
)

from orchestrator.config import OrchestratorConfig
from orchestrator.run_store import RunStore


class TestHarnessDigestRollupWiring:
    """_maybe_write_digest includes the per-(model×role) rollup (step-7/8 impl)."""

    @pytest.mark.asyncio
    async def test_digest_includes_model_role_rollup_section(
        self, tmp_path: Path, _cost_store_factory,
    ) -> None:
        """A seeded (sonnet, implementer) done invocation shows up in the written digest."""
        harness, _, event_store = _make_harness_with_mocks(tmp_path)
        harness.config = OrchestratorConfig(
            project_root=tmp_path,
            digest_every_n_escalations=3,
            digest_ewa_threshold=999.0,  # high threshold — no trip
        )

        # cost_store MUST share event_store's db file — this is what makes
        # invocations/task_results/events co-located, matching production
        # (harness.py:1373-1398: EventStore/CostStore/RunStore all open the
        # same runs.db). _make_harness_with_mocks names it 'events.db'.
        runs_db = event_store.db_path
        harness.cost_store = await _cost_store_factory(runs_db.name)
        assert harness.cost_store is not None
        assert harness.cost_store.db_path == runs_db, (
            'cost_store and event_store must share one DB file for the '
            'rollup to see both invocations and task_results/events'
        )

        # RunStore's schema (task_results) isn't created by EventStore —
        # applying it here (construct-and-discard, schema is a __init__
        # side effect) mirrors _seed_rollup_runs_db in test_digest.py.
        RunStore(runs_db)

        assert harness._run_id is not None
        run_id = harness._run_id
        _insert_task_result(runs_db, run_id=run_id, task_id='t1', outcome='done')
        _insert_invocation(
            runs_db, run_id=run_id, task_id='t1', model='sonnet', role='implementer',
            cost_usd=2.0, capped=False,
            completed_at=datetime.now(UTC).isoformat(),
        )

        harness._escalation_event_count = 5
        harness._last_digest_event_count = 0

        await harness._maybe_write_digest()

        digest_dir = tmp_path / 'data' / 'digests'
        files = list(digest_dir.glob('digest-*.md'))
        assert len(files) == 1, f'Expected 1 digest file; got {files}'
        content = files[0].read_text()
        assert '## Per-(model×role) rollup' in content, content
        assert re.search(r'\| sonnet \| implementer \|', content), content
