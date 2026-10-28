"""The offline lane's passes reach the flake ledger: ``LaneEngine`` over the real ledger.

PRD ``plans/offline-lane-red-path-prd.md`` D6, section 5.8 and the ledger scenarios of
section 6: the engine's recorder calls over a real ledger in a tmp ``runs.db``, the fake
task client, a real escalation queue, and the ledger's own drain on its wall clock.  The
pure rules are ``test_offline_lane_ledger.py``'s subject.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD0,
    HEAD1,
    LANE_COMMAND,
    T0,
    A,
    B,
    C,
    Lane,
    _run,
)
from shared.lane_probe import OUTAGE_EXIT_STATUS
from shared.psi import PsiSample

from orchestrator import flake_recorder
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    FlakeCallSite,
    FlakeOccurrenceRow,
    FlakeSuppression,
    FlakeVerdict,
    read_occurrences,
)
from orchestrator.flake_report import build_report
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import PassEffect, unconfirmable_node
from orchestrator.offline_lane_ledger import NUMERIC_UNNAMED_CLEAR
from orchestrator.offline_lane_records import (
    LEGACY_NUMERIC_COMMAND,
    FailureRecord,
    LanePass,
    ProbeKind,
    ProbeResult,
)

PROBED = LANE_COMMAND.model_copy(update={'outage_probe': 'python probe.py'})
OUTAGE = ProbeResult(ProbeKind.OUTAGE, OUTAGE_EXIT_STATUS, 'qdrant: down', 'qdrant: down', 1.0)


def _rows(lane: Lane, test_id: str | None = None) -> list[FlakeOccurrenceRow]:
    return read_occurrences(lane.ledger, test_id=test_id)


def _detail(row: FlakeOccurrenceRow) -> dict:
    return json.loads(row.detail)


def _current(lane: Lane, command: str = COMMAND) -> LanePass:
    lane_pass = lane.records.current_pass(command)
    assert lane_pass is not None
    return lane_pass


async def _numeric(lane: Lane, outcome: Confirmed | Unconfirmable, failing: tuple[str, ...] = ()) -> PassEffect:
    detect = _run('detect', HEAD1, lane.clock(), failing=failing, returncode=1)
    effect = await lane.pass_with(HEAD1, outcome, detect, None, command=LEGACY_NUMERIC_COMMAND, leg='numeric')
    lane.clock.advance(minutes=30)
    return effect


# --- scenario 25: every non-outage red pass is recorded at call site offline_lane ---------


@pytest.mark.asyncio
async def test_scenario_25_each_reproducing_pass_writes_a_fails_in_isolation_row(tmp_path: Path):
    lane = Lane(tmp_path)
    passes = []
    await lane.red()
    passes.append(_current(lane))
    await lane.spaced_red()
    passes.append(_current(lane))

    rows = _rows(lane, A)

    assert [(r.verdict, r.call_site, r.merge_sha, r.task_id, r.runner, r.psi_cpu_some10, r.observed_at) for r in rows] == [
        ('fails_in_isolation', 'offline_lane', HEAD1, None, 'local', 7.5, lane_pass.started_at.isoformat())
        for lane_pass in passes
    ]
    assert [_detail(r)['evidence'] for r in rows] == [
        {
            'kind': 'failed',
            'first_line': f'AssertionError in {A}',
            'record_dir': str(lane.records.pass_directory(COMMAND, lane_pass.pass_id)),
        }
        for lane_pass in passes
    ]


@pytest.mark.asyncio
async def test_scenario_25_a_clearing_pass_writes_a_passes_in_isolation_row(tmp_path: Path):
    lane = Lane(tmp_path)

    await lane.red(reproduced=(), cleared=(A,))

    assert [(r.test_id, r.verdict) for r in _rows(lane)] == [(A, 'passes_in_isolation')]


@pytest.mark.asyncio
async def test_scenario_25_an_unconfirmable_pass_is_counted_under_the_sentinel(tmp_path: Path):
    lane = Lane(tmp_path)

    await lane.unconfirmable()

    (row,) = _rows(lane)
    assert (row.test_id, row.verdict) == (UNKNOWN_TEST_ID, 'unconfirmable')
    assert _detail(row)['unconfirmable_reason'] == 'confirm_exited_nonzero_naming_no_test'
    assert _rows(lane, unconfirmable_node(COMMAND)) == []


@pytest.mark.asyncio
async def test_scenario_25_inconclusive_tests_are_recorded_one_reason_each(tmp_path: Path):
    lane = Lane(tmp_path)
    at = lane.clock()
    detect = _run('detect', HEAD1, at, failing=(A, B))
    confirm = replace(
        _run('confirm', HEAD1, at),
        failures=(FailureRecord(A, 'error', f'error: {A}'),),
        returncode=1,
        outcomes=None,
        session_complete=False,
    )

    await lane.pass_with(HEAD1, Confirmed(ConfirmMode.NARROWED, (), (), (A, B)), detect, confirm)

    assert {r.test_id: (r.verdict, _detail(r)['unconfirmable_reason']) for r in _rows(lane)} == {
        A: ('unconfirmable', 'confirm_kind_differs'),
        B: ('unconfirmable', 'confirm_session_incomplete'),
    }


@pytest.mark.asyncio
async def test_scenario_25_the_legacy_numeric_leg_is_recorded(tmp_path: Path):
    lane = Lane(tmp_path)

    await _numeric(lane, Confirmed(ConfirmMode.LEG_NATIVE, (A,), (), ()), failing=(A,))
    await _numeric(lane, Confirmed(ConfirmMode.LEG_NATIVE, (), (), ()))
    await _numeric(lane, Unconfirmable(UnconfirmableReason.NUMERIC_EXIT, '64'))

    assert [(r.test_id, r.verdict, _detail(r).get('unconfirmable_reason')) for r in _rows(lane)] == [
        (A, 'fails_in_isolation', None),
        (UNKNOWN_TEST_ID, 'unconfirmable', NUMERIC_UNNAMED_CLEAR),
        (UNKNOWN_TEST_ID, 'unconfirmable', 'numeric_exit'),
    ]


@pytest.mark.asyncio
async def test_a_green_pass_and_an_outage_pass_write_no_row(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.green()
    lane.clock.advance(minutes=30)
    observation = lane.written_pass(HEAD1, None, _run('detect', HEAD1, lane.clock(), failing=(A,)), None)

    await lane.engine.observe(replace(observation, lane_command=PROBED, outcome=Outage(OUTAGE), probe=OUTAGE))

    assert _rows(lane) == []


@pytest.mark.asyncio
async def test_the_same_observation_twice_lands_one_row_per_test(tmp_path: Path):
    lane = Lane(tmp_path)
    at = lane.clock()
    observation = lane.written_pass(
        HEAD1,
        Confirmed(ConfirmMode.NARROWED, (A,), (B,), ()),
        _run('detect', HEAD1, at, failing=(A, B)),
        _run('confirm', HEAD1, at, failing=(A,)),
    )

    await lane.engine.observe(observation)
    lane.clock.advance(minutes=5)
    await lane.engine.observe(observation)

    assert sorted((r.test_id, r.verdict) for r in _rows(lane)) == [(A, 'fails_in_isolation'), (B, 'passes_in_isolation')]


def _raising_reader() -> PsiSample:
    raise OSError('pressure unreadable')


@pytest.mark.parametrize(
    'psi',
    [_raising_reader, PsiSample(cpu_some10=55.0, mem_some10=0, mem_full10=0, io_some10=0, read_ok=False)],
    ids=['reader_raises', 'read_failed'],
)
@pytest.mark.asyncio
async def test_an_unknowable_host_pressure_is_recorded_as_none_and_changes_no_effect(tmp_path: Path, psi):
    healthy = Lane(tmp_path / 'healthy')
    degraded = Lane(tmp_path / 'degraded', psi=psi)

    expected = await healthy.red(reproduced=(A,), cleared=(B,))
    effect = await degraded.red(reproduced=(A,), cleared=(B,))

    assert effect == expected
    assert [r.psi_cpu_some10 for r in _rows(degraded)] == [None, None]
    assert [r.psi_cpu_some10 for r in _rows(healthy)] == [7.5, 7.5]


@pytest.mark.asyncio
async def test_the_lanes_rows_leave_the_merge_gates_report_counters_unchanged(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(flake_recorder, '_merge_flake_suppression_streak', 0)
    lane = Lane(tmp_path)
    merge_gate = FlakeSuppression(
        verdict=FlakeVerdict.passes_in_isolation,
        test_ids=(C,),
        observed_at=(T0 - timedelta(minutes=1)).isoformat(),
        call_site=FlakeCallSite.merge_gate,
        runner='local',
        psi_cpu_some10=1.0,
        unconfirmable_reason=None,
    )
    await flake_recorder.record_flake_observation(
        merge_gate, project_root=tmp_path, project_id=lane.project_id, merge_sha=HEAD0, task_id='42',
    )
    now = T0 + timedelta(hours=2)
    before = build_report(lane.ledger, now=now)

    await lane.red(reproduced=(A,), cleared=(B,))
    lane.clock.advance(minutes=30)
    await lane.unconfirmable()
    after = build_report(lane.ledger, now=now)

    assert after.gate_blind_by_call_site['merge_gate'] == before.gate_blind_by_call_site['merge_gate']
    assert after.systemic_by_call_site['merge_gate'] == before.systemic_by_call_site['merge_gate']
    assert 'offline_lane' not in before.call_sites_written
    assert 'offline_lane' in after.call_sites_written
    assert (before.gate_blind_by_call_site['offline_lane'].total, after.gate_blind_by_call_site['offline_lane'].total) == (0, 3)
    assert after.systemic_by_call_site['offline_lane'].peak_distinct_tests == 1
