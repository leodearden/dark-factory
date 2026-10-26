"""Shadow-eval event vocabulary (plans/live-shadow-eval-prd.md C2)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from enum import StrEnum
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType, ShadowSkipReason

_SHADOW_EVENT_NAMES = ('shadow_cell_opened', 'shadow_cell_skipped', 'shadow_cell_settled')

_SKIP_REASON_VALUES = {
    'sampled_out',
    'no_credentials',
    'isolation_unavailable',
    'budget_exhausted',
    'credits_exhausted',
    'concurrency_full',
    'host_saturated',
    'shape_disabled',
    'already_open',
    'paused',
}


class TestShadowEventTypes:
    @pytest.mark.parametrize('name', _SHADOW_EVENT_NAMES)
    def test_member_value_equals_name(self, name: str) -> None:
        member = EventType[name]
        assert member.value == member.name == name

    def test_skipped_round_trips_from_raw(self) -> None:
        assert EventType('shadow_cell_skipped') is EventType.shadow_cell_skipped


class TestShadowSkipReason:
    def test_value_set_is_exactly_the_closed_vocabulary(self) -> None:
        assert {m.value for m in ShadowSkipReason} == _SKIP_REASON_VALUES

    def test_is_a_str_enum(self) -> None:
        assert issubclass(ShadowSkipReason, StrEnum)


def test_skip_reason_persists_as_bare_value(tmp_path: Path) -> None:
    db_path = tmp_path / 'events.db'
    EventStore(db_path, 'run-shadow').emit(
        EventType.shadow_cell_skipped,
        task_id='42',
        data={
            'reason': ShadowSkipReason.SAMPLED_OUT,
            'shape': 'implementer',
            'candidate': 'c',
        },
    )

    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT task_id, json_extract(data, '$.reason') FROM events "
            'WHERE event_type = ?',
            (EventType.shadow_cell_skipped.value,),
        ).fetchall()

    assert rows == [('42', 'sampled_out')]
    assert ShadowSkipReason(rows[0][1]) is ShadowSkipReason.SAMPLED_OUT
