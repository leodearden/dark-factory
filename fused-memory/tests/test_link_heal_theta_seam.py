"""Gate θ reads the summary a real link-heal sweep leaves (task 6188).

One report-mode sweep over the in-process server in ``_link_heal_harness`` writes
``summary.json``. θ's predicate, ``scripts/check_write_triage_readiness_gate.py``,
then runs over it with θ's own summary bounds, so a key θ requires that the
producer stops writing fails here. The predicate is reached through the
repository, which is why this module is apart from ``test_link_heal_metrics``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    DF,
    FakeAdjudicator,
    RecordingFiler,
    build_harness,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.maintenance.link_heal import HealAction, Verdict
from fused_memory.maintenance.link_heal_executor import RunLimits
from fused_memory.maintenance.link_heal_ledger import LinkHealLedger
from fused_memory.maintenance.link_heal_metrics import SweepMetricsTarget, summary_path
from fused_memory.maintenance.link_heal_sweep import SweepSettings, run_sweep
from fused_memory.server.grouped_read import SIGHTING_KIND

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

NOW = datetime(2026, 11, 15, 2, 15, tzinfo=UTC)
GATE_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'check_write_triage_readiness_gate.py'
#: Gate θ's three summary bounds, copied verbatim from task 6191's metadata.before_done.args.
THETA_SUMMARY_REQUIRES = (
    ('s', 'runs_7d', '>=', '6'),
    ('s', 'would_escape_7d', '<=', '0'),
    ('s', 'adjudication_failure_rate_7d', '<=', '0.05'),
)


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()


@pytest.fixture
def ledger(tmp_path: Path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


async def _report_mode_sweep(harness, ledger, tmp_path: Path) -> Path:
    """Sweep one in-window sighting without writes; the summary it leaves."""
    harness.seed_link(
        kind=SIGHTING_KIND, child_text='a sighting', parent_text='its parent',
        meta={'created_at': (NOW - timedelta(hours=12)).isoformat()},
    )
    root = tmp_path / 'memory-evals'
    await run_sweep(
        adjudicate=FakeAdjudicator(lambda _text: Verdict.SAME),
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=RunLimits(
            max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3,
            misfile_share_ceiling=0.25, corrects_share_ceiling=0.60,
        ),
        settings=SweepSettings(
            projects=(DF,), writes=False, dry_run=False,
            actions=frozenset({HealAction.DETACH, HealAction.RELABEL}), window_hours=24,
            max_adjudications=200,
        ),
        filer=RecordingFiler(),
        reports_dir=tmp_path / 'reports',
        now=NOW,
        metrics=SweepMetricsTarget(root, DF),
    )
    return summary_path(root)


class TestThetaReadsTheSummary:
    @pytest.mark.asyncio
    async def test_every_theta_summary_bound_resolves_to_a_number(
        self, harness, ledger, tmp_path,
    ):
        summary = await _report_mode_sweep(harness, ledger, tmp_path)
        argv = [sys.executable, str(GATE_SCRIPT), '--gate', 'theta', '--report', 's', str(summary)]
        for require in THETA_SUMMARY_REQUIRES:
            argv += ['--require', *require]

        completed = subprocess.run(argv, capture_output=True, text=True, timeout=60)

        verdict = json.loads(completed.stdout.strip().splitlines()[-2])
        assert len(verdict['checks']) == 3
        for check in verdict['checks']:
            assert check['note'] is None, check
            assert isinstance(check['actual'], int | float), check
            assert not isinstance(check['actual'], bool), check
        assert [check['check'] for check in verdict['checks'] if not check['ok']] == [
            's:runs_7d >= 6',
        ]
