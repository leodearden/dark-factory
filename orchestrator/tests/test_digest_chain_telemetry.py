"""Tests for the digest's deep merge-ahead chain telemetry: the windowed reader and its rendered section (task 3188 ε)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest
from test_digest import _make_digest_inputs

import orchestrator.digest as digest
from orchestrator.chain_telemetry import ChainBucket, ChainTelemetry
from orchestrator.event_store import EventStore

_WINDOW_START = '2026-05-10T00:00:00+00:00'
_WINDOW_END = '2026-05-10T23:59:59+00:00'
_SECTION = '## Deep merge-ahead chains'


def _seed_chain_events_db(
    db_path: Path, rows: list[tuple[str, str, str, dict | str]],
) -> None:
    """Seed ``(timestamp, run_id, event_type, data)`` rows into a real-schema events DB.

    A str ``data`` is written verbatim, so a caller can store corrupt JSON.
    """
    EventStore(db_path, 'run-test')
    conn = sqlite3.connect(str(db_path))
    try:
        for ts, run_id, event_type, data in rows:
            raw = data if isinstance(data, str) else json.dumps(data)
            conn.execute(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (ts, run_id, 't', event_type, raw),
            )
        conn.commit()
    finally:
        conn.close()


def _in_window(hour: int) -> str:
    return f'2026-05-10T{hour:02d}:00:00+00:00'


def _read(db_path: Path) -> ChainTelemetry:
    return digest.chain_telemetry_in_window(db_path, _WINDOW_START, _WINDOW_END)


class TestChainTelemetryInWindow:
    def test_combines_verify_and_finalized_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _seed_chain_events_db(db_path, [
            (_in_window(1), 'run-a', 'merge_verify', {'chain_items': 1, 'passed': True}),
            (_in_window(2), 'run-a', 'merge_verify', {'chain_items': 3, 'passed': True}),
            (_in_window(3), 'run-a', 'merge_verify', {'chain_items': 3, 'passed': False}),
            (_in_window(4), 'run-a', 'merge_finalized', {'state': 'done', 'landed_via_chain': 1}),
            (_in_window(5), 'run-a', 'merge_finalized', {'state': 'done', 'landed_via_chain': 1}),
            (_in_window(6), 'run-a', 'merge_finalized', {'state': 'done', 'landed_via_chain': None}),
        ])
        telemetry = _read(db_path)
        assert telemetry.chain_items_histogram == {1: ChainBucket(1, 1), 3: ChainBucket(1, 2)}, (
            f'{telemetry=}'
        )
        assert (telemetry.deep_verifies, telemetry.deep_failures) == (2, 1), f'{telemetry=}'
        assert telemetry.items_landed_via_chain == 2, f'{telemetry=}'
        assert telemetry.items_landed_per_deep_verify == 1.0, f'{telemetry=}'

    def test_numerator_and_denominator_share_the_window(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _seed_chain_events_db(db_path, [
            ('2026-05-09T23:59:59+00:00', 'run-a', 'merge_verify',
             {'chain_items': 2, 'passed': True}),
            (_in_window(1), 'run-a', 'merge_verify', {'chain_items': 2, 'passed': True}),
            (_in_window(2), 'run-a', 'merge_finalized', {'state': 'done', 'landed_via_chain': 1}),
            ('2026-05-11T00:00:01+00:00', 'run-a', 'merge_finalized',
             {'state': 'done', 'landed_via_chain': 1}),
        ])
        telemetry = _read(db_path)
        assert telemetry.deep_verifies == 1, f'{telemetry=}'
        assert telemetry.items_landed_via_chain == 1, f'{telemetry=}'

    def test_other_event_types_are_ignored(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _seed_chain_events_db(db_path, [
            (_in_window(1), 'run-a', 'merge_attempt', {'chain_items': 2, 'passed': True}),
            (_in_window(2), 'run-a', 'speculative_merge', {'landed_via_chain': 1}),
        ])
        assert _read(db_path) == ChainTelemetry(), f'{_read(db_path)=}'

    def test_rows_from_every_run_are_counted(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _seed_chain_events_db(db_path, [
            (_in_window(1), 'run-a', 'merge_verify', {'chain_items': 2, 'passed': True}),
            (_in_window(2), 'run-b', 'merge_verify', {'chain_items': 2, 'passed': True}),
        ])
        telemetry = _read(db_path)
        assert telemetry.deep_verifies == 2, f'{telemetry=}'

    def test_missing_db_returns_neutral_value_and_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'no-such.db'
        assert _read(db_path) == ChainTelemetry()
        assert not db_path.exists(), 'chain_telemetry_in_window must not create a stub DB file'

    def test_schema_less_db_returns_neutral_value_with_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db_path = tmp_path / 'schema-less.db'
        sqlite3.connect(str(db_path)).close()
        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            telemetry = _read(db_path)
        assert telemetry == ChainTelemetry(), f'{telemetry=}'
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest'
        ]
        assert len(warnings) == 1, f'{[r.message for r in warnings]}'

    @pytest.mark.parametrize('corrupt', ['{not json', '[2, 3]'], ids=['unparseable', 'not-an-object'])
    def test_corrupt_payload_is_skipped_without_aborting(self, tmp_path: Path, corrupt: str) -> None:
        db_path = tmp_path / 'events.db'
        _seed_chain_events_db(db_path, [
            (_in_window(1), 'run-a', 'merge_verify', corrupt),
            (_in_window(2), 'run-a', 'merge_verify', {'chain_items': 2, 'passed': True}),
            (_in_window(3), 'run-a', 'merge_finalized', '[1]'),
            (_in_window(4), 'run-a', 'merge_finalized', {'state': 'done', 'landed_via_chain': 1}),
        ])
        telemetry = _read(db_path)
        assert telemetry.deep_verifies == 1, f'{telemetry=}'
        assert telemetry.items_landed_via_chain == 1, f'{telemetry=}'


def _section(md: str) -> str:
    return md.split(_SECTION, 1)[1].split('\n## ', 1)[0]


def _render(telemetry: ChainTelemetry) -> str:
    inputs = _make_digest_inputs()
    inputs.chain_telemetry = telemetry
    return digest.render_digest_markdown(inputs)


class TestRenderDigestChainTelemetrySection:
    def test_digest_inputs_default_to_the_neutral_value(self) -> None:
        assert _make_digest_inputs().chain_telemetry == ChainTelemetry()

    def test_histogram_rows_render_in_ascending_order(self) -> None:
        section = _section(_render(ChainTelemetry(
            chain_items_histogram={4: ChainBucket(2, 2), 1: ChainBucket(9, 12)},
            deep_verifies=2,
            deep_fail_rate=0.0,
            items_landed_via_chain=8,
            items_landed_per_deep_verify=4.0,
        )))
        one = re.search(r'^\| 1 \| 12 \| 9 \| 75\.0% \|$', section, re.MULTILINE)
        four = re.search(r'^\| 4 \| 2 \| 2 \| 100\.0% \|$', section, re.MULTILINE)
        assert one and four, section
        assert one.start() < four.start(), section

    def test_rates_render_from_populated_telemetry(self) -> None:
        section = _section(_render(ChainTelemetry(
            chain_items_histogram={4: ChainBucket(1, 2)},
            deep_verifies=2,
            deep_failures=1,
            deep_fail_rate=0.5,
            items_landed_via_chain=8,
            items_landed_per_deep_verify=4.0,
            excluded_legacy_depth=7,
        )))
        assert re.search(r'^- Items landed per deep verify: 4\.00$', section, re.MULTILINE), section
        assert re.search(r'^- Deep-fail rate: 50\.0% \(1/2\)$', section, re.MULTILINE), section
        assert re.search(r'^- Excluded legacy depth>=2 events: 7$', section, re.MULTILINE), section

    def test_no_deep_verify_renders_rates_as_not_available(self) -> None:
        section = _section(_render(ChainTelemetry(
            chain_items_histogram={1: ChainBucket(3, 3)}, items_landed_via_chain=2,
        )))
        assert re.search(r'^- Items landed per deep verify: n/a$', section, re.MULTILINE), section
        assert re.search(r'^- Deep-fail rate: n/a$', section, re.MULTILINE), section

    def test_neutral_value_renders_placeholders(self) -> None:
        section = _section(_render(ChainTelemetry()))
        assert re.search(r'^_none_$', section, re.MULTILINE), section
        assert re.search(r'^- Items landed per deep verify: n/a$', section, re.MULTILINE), section
        assert re.search(r'^- Deep-fail rate: n/a$', section, re.MULTILINE), section
        assert re.search(r'^- Excluded legacy depth>=2 events: 0$', section, re.MULTILINE), section

    def test_section_sits_between_rollup_and_stale_lanes(self) -> None:
        md = _render(ChainTelemetry())
        rollup = md.index('## Per-(model×role) rollup')
        chains = md.index(_SECTION)
        stale = md.index('## Stale lane assignments')
        assert rollup < chains < stale, md
