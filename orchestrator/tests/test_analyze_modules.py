"""Tests for the analyze_modules per-module conflict helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.analyze_modules import (
    ModuleStats,
    _first_component,
    _parse_since,
    aggregate,
    render_json,
    render_table,
    suggest_max_per_module,
)
from orchestrator.event_store import EventStore, EventType


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='test')


def test_first_component_strips_and_splits():
    assert _first_component('autopilot/analyze/asr') == 'autopilot'
    assert _first_component('/crates/foo/src') == 'crates'
    assert _first_component('bare') == 'bare'
    assert _first_component('') == ''


def test_parse_since_duration_shorthand():
    # '7d' shouldn't raise and must yield a past timestamp.
    from datetime import UTC, datetime
    cut = _parse_since('7d')
    assert cut < datetime.now(UTC)


def test_parse_since_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_since('nonsense')


def test_aggregate_counts_dispatches_and_skips(tmp_path: Path, event_store: EventStore):
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.lock_released,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    from datetime import UTC, datetime, timedelta
    stats = aggregate(event_store.db_path, datetime.now(UTC) - timedelta(days=1))
    assert 'crates' in stats
    assert stats['crates'].dispatches == 1
    assert stats['crates'].skipped_waiting == 2
    assert stats['crates'].conflict_rate() == pytest.approx(2.0)


def test_aggregate_ignores_events_before_since(tmp_path: Path, event_store: EventStore):
    from datetime import UTC, datetime, timedelta
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    # Pretend these happened in the past: cut is "now + 1 hour" so nothing qualifies.
    stats = aggregate(event_store.db_path, datetime.now(UTC) + timedelta(hours=1))
    assert stats == {}


def test_suggest_max_per_module_tiers():
    assert suggest_max_per_module(ModuleStats(dispatches=1, skipped_waiting=2)) == 1
    assert suggest_max_per_module(ModuleStats(dispatches=2, skipped_waiting=1)) == 2
    assert suggest_max_per_module(ModuleStats(dispatches=10, skipped_waiting=1)) == 3
    assert suggest_max_per_module(ModuleStats(dispatches=100, skipped_waiting=0)) == 4


def test_render_table_orders_by_conflict_desc():
    stats = {
        'low': ModuleStats(dispatches=10, skipped_waiting=0),
        'hot': ModuleStats(dispatches=5, skipped_waiting=20),
    }
    table = render_table(stats)
    lines = table.splitlines()
    # Header + two rows, hot first.
    assert lines[1].startswith('hot')
    assert lines[2].startswith('low')


def test_render_json_is_machine_readable():
    # Asymmetric values so a dispatches<->skipped swap would be caught:
    # swapped input would be (7, 4) → ratio 4/7 ≈ 0.57 → still suggest 2,
    # so the conflict_rate assertion does the distinguishing.
    stats = {
        'crates': ModuleStats(dispatches=4, skipped_waiting=7),
    }
    payload = json.loads(render_json(stats))
    assert payload['crates']['dispatches'] == 4
    assert payload['crates']['skipped_waiting'] == 7
    # conflict = 7/4 = 1.75 (>= 0.5 but < 2.0) → suggest 2.
    assert payload['crates']['conflict_rate'] == 1.75
    assert payload['crates']['suggested_max_per_module'] == 2
