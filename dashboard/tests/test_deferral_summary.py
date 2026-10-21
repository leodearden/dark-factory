"""How the Tasks tab words a deferred row's hold — ``dashboard.data.deferral_summary``.

Every row is acquired through the real snapshot unit, with the canned
fused-memory substrate patched at the public ``dashboard.data.tasks.mcp_tool_call``.
The hold is therefore read from exactly the row shape production hands the
collector. The rows come from ``_deferral_fixture.py``. Every expected verdict
comes from ``shared.task_deferral.needs_human``, never from a copy of it.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import timedelta
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP
from _deferral_fixture import (
    CARRIED_LIVE,
    CARRIED_REASON,
    CARRIER_GONE,
    CLOSABLE,
    CONDITION,
    CONDITION_EXPIRING,
    CONDITION_STALE,
    FORGED_BOOT_ID,
    HELD_DEAD,
    HELD_LIVE,
    HELD_OTHER_HOST,
    INVALID,
    LEGACY,
    NOW,
    ORPHANED,
    OTHER_HOST,
    PLAIN,
    PLANNING_FRESH,
    PLANNING_STALE,
    STALE_CONDITION,
    STAMPED_BY,
    STRAY,
    UNRECORDED,
    FixtureRow,
    deferral_fixture,
    expected_needs_human,
    raw_rows,
    status_map,
)
from shared.process_identity import Liveness, ProcessIdentity, capture
from shared.task_deferral import HeldBySessionDeferral, record_to_json
from shared.task_statuses import TaskStatus

from dashboard.data.deferral_summary import (
    UNREAD,
    HoldReads,
    HoldState,
    deferral_row_fields,
    read_liveness,
    session_holders,
)
from dashboard.data.task_snapshot import acquire_snapshot

STRAY_HELD = 191
"""A pending row carrying a held_by_session record: a stray, never a holder to read."""


@pytest.fixture()
def live_holder() -> ProcessIdentity:
    holder = capture(os.getpid())
    assert holder is not None, 'the test process must be able to identify itself'
    return holder


@pytest.fixture()
def fixture_rows(live_holder) -> tuple[FixtureRow, ...]:
    stray_held = HeldBySessionDeferral(
        holder=live_holder, deferred_at=NOW, stamped_at=NOW, stamped_by=STAMPED_BY
    )
    return (
        *deferral_fixture(now=NOW, live_holder=live_holder),
        FixtureRow(STRAY_HELD, TaskStatus.PENDING, 'pending with a stray held record',
                   record_to_json(stray_held)),
    )


@pytest.fixture()
async def rows(fixture_rows, dashboard_config, dummy_client, tmp_path) -> dict[int, dict]:
    """The fixture's rows as the snapshot unit holds them, by id."""
    canned = CannedMCP(rows=raw_rows(fixture_rows), status_map=status_map(fixture_rows))
    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        snapshot = await acquire_snapshot(
            dummy_client, dashboard_config, tmp_path / 'deferrals', now=NOW
        )
    assert snapshot.rows.value is not None
    return {row['id']: row for row in snapshot.rows.value}


@pytest.fixture()
def measured(fixture_rows, rows) -> HoldReads:
    """Both referents read: the fixture's status map and every holder's real liveness."""
    return HoldReads(
        carrier_statuses=status_map(fixture_rows),
        holder_liveness=read_liveness(session_holders(rows.values())),
    )


def _summary(row: dict, reads: HoldReads) -> dict:
    fields = deferral_row_fields(row, now=NOW, reads=reads)
    assert fields.keys() == {'deferral'}, fields
    summary = fields['deferral']
    assert summary is not None
    return summary


class TestWhichRowsCarryAHold:
    def test_a_row_that_is_not_deferred_carries_no_deferral_key(self, rows, measured):
        for task_id in (STRAY, STRAY_HELD, PLAIN):
            assert deferral_row_fields(rows[task_id], now=NOW, reads=measured) == {}, task_id

    def test_a_deferred_row_without_a_record_carries_an_explicit_null(self, rows, measured):
        assert deferral_row_fields(rows[UNRECORDED], now=NOW, reads=measured) == {
            'deferral': None
        }

    def test_the_summary_carries_no_kind_for_the_front_end_to_switch_on(self, rows, measured):
        for task_id in (CLOSABLE, HELD_LIVE, LEGACY, INVALID):
            assert _summary(rows[task_id], measured).keys() == {
                'label', 'state', 'age_seconds', 'needs_human', 'detail',
            }


class TestNeedsHumanIsTheSharedVerdict:
    def test_every_recorded_deferred_row_reads_the_shared_verdict(
        self, fixture_rows, rows, measured
    ):
        expected = {
            task_id: None if reason is None else reason.value
            for task_id, reason in expected_needs_human(fixture_rows, now=NOW).items()
            if task_id != UNRECORDED
        }
        shown = {task_id: _summary(rows[task_id], measured)['needs_human'] for task_id in expected}
        assert shown == expected


class TestLabels:
    @pytest.mark.parametrize(
        ('task_id', 'label'),
        [
            (CLOSABLE, 'parked: carried by #900 (closable)'),
            (ORPHANED, 'parked: carried by #901 (orphaned)'),
            (CARRIED_LIVE, 'parked: carried by #902'),
            (CARRIER_GONE, 'parked: carried by #903 (carrier missing)'),
            (CONDITION_EXPIRING, f'parked until {CONDITION} (lapses 2026-10-24 12:00 UTC)'),
            (CONDITION_STALE, f'parked until {STALE_CONDITION}'),
            (PLANNING_FRESH, 'planning batch'),
            (PLANNING_STALE, 'planning batch'),
            (LEGACY, 'legacy: no recorded reason'),
            (INVALID, 'invalid deferral record'),
        ],
    )
    def test_each_hold_is_worded_as_decision_10_words_it(self, rows, measured, task_id, label):
        assert _summary(rows[task_id], measured)['label'] == label

    @pytest.mark.parametrize(
        ('task_id', 'qualifier'),
        [(HELD_LIVE, 'live'), (HELD_DEAD, 'lost, releasing'), (HELD_OTHER_HOST, 'on another host')],
    )
    def test_a_session_hold_names_its_holder_and_its_liveness(
        self, rows, measured, live_holder, task_id, qualifier
    ):
        assert _summary(rows[task_id], measured)['label'] == (
            f'held by session {live_holder.pid} ({qualifier})'
        )


class TestState:
    @pytest.mark.parametrize(
        ('task_id', 'state'),
        [
            (HELD_LIVE, HoldState.HELD),
            (CARRIED_LIVE, HoldState.HELD),
            (PLANNING_FRESH, HoldState.HELD),
            (CONDITION_EXPIRING, HoldState.HELD),
            (CLOSABLE, HoldState.NEEDS_HUMAN),
            (ORPHANED, HoldState.NEEDS_HUMAN),
            (CARRIER_GONE, HoldState.NEEDS_HUMAN),
            (HELD_DEAD, HoldState.NEEDS_HUMAN),
            (HELD_OTHER_HOST, HoldState.NEEDS_HUMAN),
            (CONDITION_STALE, HoldState.NEEDS_HUMAN),
            (PLANNING_STALE, HoldState.NEEDS_HUMAN),
            (LEGACY, HoldState.NEEDS_HUMAN),
            (INVALID, HoldState.NEEDS_HUMAN),
        ],
    )
    def test_state_follows_the_verdict_when_every_referent_was_read(
        self, rows, measured, task_id, state
    ):
        assert _summary(rows[task_id], measured)['state'] == state


class TestUnreadReferents:
    @pytest.mark.parametrize(
        'holder_liveness', [{}, {HELD_LIVE: None}], ids=['absent', 'unreadable']
    )
    def test_an_unread_holder_is_unknown_and_never_lost(
        self, fixture_rows, rows, live_holder, holder_liveness
    ):
        reads = HoldReads(carrier_statuses=status_map(fixture_rows),
                          holder_liveness=holder_liveness)
        summary = _summary(rows[HELD_LIVE], reads)
        assert summary['label'] == f'held by session {live_holder.pid} (liveness unknown)'
        assert summary['state'] == HoldState.UNKNOWN
        assert summary['needs_human'] == 'liveness_unknown'

    @pytest.mark.parametrize('task_id', [CLOSABLE, ORPHANED, CARRIED_LIVE, CARRIER_GONE])
    def test_an_unmeasured_status_map_leaves_every_carrier_unknown(
        self, rows, measured, task_id
    ):
        reads = dataclasses.replace(measured, carrier_statuses=None)
        summary = _summary(rows[task_id], reads)
        assert summary['label'].endswith(' (carrier status unknown)')
        assert summary['label'].startswith('parked: carried by #')
        assert summary['state'] == HoldState.UNKNOWN
        assert summary['needs_human'] is None

    def test_only_an_unread_holder_needs_a_human(self, rows):
        """The shared model has LIVENESS_UNKNOWN and no reason for an unread carrier."""
        carrier = _summary(rows[CLOSABLE], UNREAD)
        holder = _summary(rows[HELD_LIVE], UNREAD)
        assert carrier['state'] == holder['state'] == HoldState.UNKNOWN
        assert (carrier['needs_human'], holder['needs_human']) == (None, 'liveness_unknown')

    def test_unread_reads_neither_referent(self, rows, live_holder):
        assert _summary(rows[CLOSABLE], UNREAD)['label'] == (
            'parked: carried by #900 (carrier status unknown)'
        )
        assert _summary(rows[HELD_LIVE], UNREAD)['label'] == (
            f'held by session {live_holder.pid} (liveness unknown)'
        )


class TestAgeAndDetail:
    @pytest.mark.parametrize(
        ('task_id', 'age'),
        [
            (CLOSABLE, timedelta(hours=2)),
            (CONDITION_EXPIRING, timedelta(days=1)),
            (CONDITION_STALE, timedelta(days=31)),
            (HELD_LIVE, timedelta(minutes=10)),
            (PLANNING_FRESH, timedelta(hours=1)),
            (PLANNING_STALE, timedelta(hours=25)),
        ],
    )
    def test_age_is_whole_seconds_since_the_row_was_deferred(
        self, rows, measured, task_id, age
    ):
        assert _summary(rows[task_id], measured)['age_seconds'] == int(age.total_seconds())

    @pytest.mark.parametrize('task_id', [LEGACY, INVALID])
    def test_age_is_unknown_without_a_deferred_at(self, rows, measured, task_id):
        assert _summary(rows[task_id], measured)['age_seconds'] is None

    def test_detail_names_who_stamped_the_hold_and_its_reason(self, rows, measured):
        detail = _summary(rows[CLOSABLE], measured)['detail']
        assert f'stamped by {STAMPED_BY}' in detail
        assert f'reason: {CARRIED_REASON}' in detail

    def test_detail_calls_an_unnamed_stamper_unattributed(self, rows, measured):
        detail = _summary(rows[CONDITION_STALE], measured)['detail']
        assert 'stamped by unattributed' in detail
        assert 'reason:' not in detail

    def test_an_invalid_record_details_why_it_did_not_parse(self, rows, measured):
        assert 'carrier_task_id' in _summary(rows[INVALID], measured)['detail']


class TestHolderReads:
    def test_only_deferred_rows_holding_a_session_name_a_holder(self, rows, live_holder):
        assert session_holders(rows.values()) == {
            HELD_LIVE: live_holder,
            HELD_DEAD: dataclasses.replace(live_holder, boot_id=FORGED_BOOT_ID),
            HELD_OTHER_HOST: dataclasses.replace(live_holder, host=OTHER_HOST),
        }

    def test_each_holder_is_read_from_proc(self, rows):
        assert read_liveness(session_holders(rows.values())) == {
            HELD_LIVE: Liveness.ALIVE,
            HELD_DEAD: Liveness.DEAD,
            HELD_OTHER_HOST: Liveness.OTHER_HOST,
        }

    def test_an_unreadable_proc_is_never_read_as_death(self, rows, tmp_path):
        empty_proc = tmp_path / 'proc'
        empty_proc.mkdir()
        assert read_liveness(session_holders(rows.values()), proc_root=empty_proc) == {
            HELD_LIVE: None,
            HELD_DEAD: None,
            HELD_OTHER_HOST: Liveness.OTHER_HOST,
        }
