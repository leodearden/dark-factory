"""Tests that Harness._maybe_write_digest renders deep merge-ahead chain telemetry from its live runs.db (task 3188 ε)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_digest_chain_telemetry import _seed_chain_events_db
from test_harness_park_stop import _make_harness_with_mocks

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


def _arm_digest(harness: Harness, tmp_path: Path) -> None:
    harness.config = OrchestratorConfig(
        project_root=tmp_path,
        digest_every_n_escalations=3,
        digest_ewa_threshold=999.0,
    )
    harness._escalation_event_count = 5
    harness._last_digest_event_count = 0


def _written_chain_section(tmp_path: Path) -> str:
    files = list((tmp_path / 'data' / 'digests').glob('digest-*.md'))
    assert len(files) == 1, f'Expected 1 digest file; got {files}'
    content = files[0].read_text()
    assert '## Deep merge-ahead chains' in content, content
    return content.split('## Deep merge-ahead chains', 1)[1].split('\n## ', 1)[0]


class TestHarnessDigestChainTelemetryWiring:
    @pytest.mark.asyncio
    async def test_digest_renders_the_seeded_chain_numbers(self, tmp_path: Path) -> None:
        harness, _, event_store = _make_harness_with_mocks(tmp_path)
        _arm_digest(harness, tmp_path)
        recent = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        _seed_chain_events_db(event_store.db_path, [
            (recent, 'run-other', 'merge_verify', {'chain_items': 4, 'passed': True}),
            (recent, 'run-other', 'merge_verify', {'chain_items': 1, 'passed': False}),
            *[
                (recent, 'run-other', 'merge_finalized', {'state': 'done', 'landed_via_chain': 1})
                for _ in range(4)
            ],
        ])

        await harness._maybe_write_digest()

        section = _written_chain_section(tmp_path)
        assert re.search(r'^\| 4 \| 1 \| 1 \| 100\.0% \|$', section, re.MULTILINE), section
        assert re.search(r'^\| 1 \| 1 \| 0 \| 0\.0% \|$', section, re.MULTILINE), section
        assert re.search(r'^- Items landed per deep verify: 4\.00$', section, re.MULTILINE), section
        assert re.search(r'^- Deep-fail rate: 0\.0% \(0/1\)$', section, re.MULTILINE), section

    @pytest.mark.asyncio
    async def test_no_event_store_renders_placeholders(self, tmp_path: Path) -> None:
        harness, _, _ = _make_harness_with_mocks(tmp_path)
        _arm_digest(harness, tmp_path)
        harness.event_store = None

        await harness._maybe_write_digest()

        section = _written_chain_section(tmp_path)
        assert re.search(r'^_none_$', section, re.MULTILINE), section
        assert re.search(r'^- Items landed per deep verify: n/a$', section, re.MULTILINE), section
        assert re.search(r'^- Deep-fail rate: n/a$', section, re.MULTILINE), section
