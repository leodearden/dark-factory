"""The offline lane's passes reach the flake ledger: ``LaneEngine`` over the real ledger.

PRD ``plans/offline-lane-red-path-prd.md`` D6, section 5.8 and the ledger scenarios of
section 6: the engine's recorder calls over a real ledger in a tmp ``runs.db``, the fake
task client, a real escalation queue, and the ledger's own drain on its wall clock.  The
pure rules are ``test_offline_lane_ledger.py``'s subject.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _lane_episode_fakes import (
    COMMAND,
    FIX_COMMIT,
    HEAD0,
    HEAD1,
    HEAD2,
    HEAD3,
    LANE_COMMAND,
    T0,
    A,
    B,
    C,
    Lane,
    _run,
)
from escalation.categories import EscalationCategory
from shared.lane_probe import OUTAGE_EXIT_STATUS
from shared.psi import PsiSample

from orchestrator import flake_recorder
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtTrigger,
    DueRow,
    FlakeCallSite,
    FlakeOccurrenceRow,
    FlakeSuppression,
    FlakeVerdict,
    due_item_context,
    read_debt,
    read_due_items,
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
from orchestrator.offline_lane_episode_state import (
    EpisodeStore,
    OfferedDebt,
    PassEffect,
    unconfirmable_node,
)
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


def _due(lane: Lane) -> list[DueRow]:
    """Every item on the due list, backed off or not: the list is stamped with the wall clock."""
    items = read_due_items(lane.ledger, now=datetime.now(UTC) + timedelta(days=1), limit=50)
    assert items is not None
    return items


def _pending(lane: Lane, category: EscalationCategory) -> list:
    assert lane.queue is not None
    return [e for e in lane.queue.get_pending() if e.category == category]


def _merge_gate_clear(index: int) -> FlakeSuppression:
    return FlakeSuppression(
        verdict=FlakeVerdict.passes_in_isolation,
        test_ids=(f'tests/test_m.py::test_{index}',),
        observed_at=(T0 + timedelta(minutes=index)).isoformat(),
        call_site=FlakeCallSite.merge_gate,
        runner='local',
        psi_cpu_some10=1.0,
        unconfirmable_reason=None,
    )


def _done(lane: Lane, task_id: str) -> None:
    lane.client.set_status(task_id, 'done', done_provenance={'kind': 'merged', 'commit': FIX_COMMIT})


async def _fix_filed(lane: Lane) -> str:
    """A reproduced on two spaced passes, which files its fix task; the task's id."""
    await lane.red()
    effect = await lane.spaced_red()
    (task_id,) = effect.tasks_filed
    return task_id


async def _clear(lane: Lane, node: str = A, *, head: str = HEAD1) -> PassEffect:
    lane.clock.advance(minutes=30)
    return await lane.red(reproduced=(), cleared=(node,), head=head)


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
    assert before.call_sites_written is not None
    assert after.call_sites_written is not None
    assert 'offline_lane' not in before.call_sites_written
    assert 'offline_lane' in after.call_sites_written
    assert (before.gate_blind_by_call_site['offline_lane'].total, after.gate_blind_by_call_site['offline_lane'].total) == (0, 3)
    assert after.systemic_by_call_site['offline_lane'].peak_distinct_tests == 1


# --- a cleared test owes debt unless an open fix task covers it ---------------------------


@pytest.mark.asyncio
async def test_scenario_26_a_first_clear_owes_debt_and_the_drain_files_its_owner(tmp_path: Path):
    lane = Lane(tmp_path)

    effect = await lane.red(reproduced=(), cleared=(A,))

    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.cleared, HEAD1),)
    (item,) = _due(lane)
    assert (item.test_id, item.call_site, item.head) == (A, 'offline_lane', HEAD1)
    context = due_item_context(item)
    assert context is not None
    assert (context.trigger, context.command, context.working_directory, context.heads) == (
        DebtTrigger.cleared, COMMAND, 'integration', (HEAD1,),
    )
    assert context.reproduce is not None and A in context.reproduce
    assert [entry.first_line for entry in context.evidence] == [f'AssertionError in {A}']

    await lane.drain()

    debt = read_debt(lane.ledger, A)
    assert debt is not None and debt.owner_task_id is not None
    (filed,) = lane.client.submitted
    assert 'cleared on the offline lane (integration)' in filed['title']
    assert 'cycle 1' in filed['title']
    assert context.reproduce in filed['description']
    assert f'AssertionError in {A}' in filed['description']
    assert (filed['metadata']['flake_debt_call_site'], filed['metadata']['flake_debt_trigger']) == (
        'offline_lane', 'cleared',
    )

    await _clear(lane)
    await lane.drain()

    assert len(lane.client.submitted) == 1


@pytest.mark.asyncio
async def test_scenario_29_nine_clears_in_one_pass_make_no_task_client_call_and_nine_owners(tmp_path: Path):
    lane = Lane(tmp_path)
    nodes = tuple(f'tests/test_n.py::test_{index}' for index in range(9))

    effect = await lane.red(reproduced=(), cleared=nodes)

    assert lane.log == []
    assert sorted(item.test_id for item in _due(lane)) == sorted(nodes)
    assert effect.debts_offered == tuple(OfferedDebt(node, DebtTrigger.cleared, HEAD1) for node in nodes)

    await lane.drain()

    assert len(lane.client.submitted) == 9


@pytest.mark.asyncio
async def test_scenario_30_lane_clears_never_count_toward_the_merge_gates_storm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(flake_recorder, '_merge_flake_suppression_streak', 0)
    lane = Lane(tmp_path)
    for index in range(6):
        await _clear(lane, f'tests/test_s.py::test_{index}')

    for index in range(4):
        await flake_recorder.record_flake_observation(
            _merge_gate_clear(index), project_root=tmp_path, project_id=lane.project_id,
            merge_sha=HEAD0, task_id=str(4200 + index), escalation_queue=lane.queue,
        )
    assert _pending(lane, EscalationCategory.MERGE_FLAKE_SUPPRESSION_STORM) == []

    await flake_recorder.record_flake_observation(
        _merge_gate_clear(4), project_root=tmp_path, project_id=lane.project_id,
        merge_sha=HEAD0, task_id='4204', escalation_queue=lane.queue,
    )
    assert len(_pending(lane, EscalationCategory.MERGE_FLAKE_SUPPRESSION_STORM)) == 1


@pytest.mark.asyncio
async def test_scenario_33_an_open_fix_task_covers_a_clear_until_it_is_done(tmp_path: Path):
    lane = Lane(tmp_path)
    task_id = await _fix_filed(lane)

    covered = await _clear(lane)

    assert _due(lane) == []
    assert (covered.debts_offered, covered.debts_undecided) == ((), ())

    _done(lane, task_id)
    offered = await _clear(lane)

    assert [item.test_id for item in _due(lane)] == [A]
    assert offered.debts_offered == (OfferedDebt(A, DebtTrigger.cleared, HEAD1),)


@pytest.mark.asyncio
async def test_an_unreadable_covering_status_leaves_the_clear_undecided_and_visible(tmp_path: Path, caplog):
    lane = Lane(tmp_path)
    task_id = await _fix_filed(lane)
    _done(lane, task_id)
    lane.client.statuses_error = ConnectionError('task store unreachable')

    with caplog.at_level(logging.ERROR):
        undecided = await _clear(lane)

    assert _due(lane) == []
    assert (undecided.debts_offered, undecided.debts_undecided) == ((), (A,))
    assert any(
        r.levelno >= logging.ERROR and A in r.getMessage() and task_id in r.getMessage() for r in caplog.records
    )

    lane.client.statuses_error = None
    offered = await _clear(lane)

    assert offered.debts_offered == (OfferedDebt(A, DebtTrigger.cleared, HEAD1),)
    assert [item.test_id for item in _due(lane)] == [A]


@pytest.mark.asyncio
async def test_the_ledger_is_written_before_the_episodes_new_state(tmp_path: Path, monkeypatch):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=30)

    def unwritable(self, record):
        raise OSError('episode file unwritable')

    monkeypatch.setattr(EpisodeStore, 'save', unwritable)
    with pytest.raises(OSError):
        await lane.red(reproduced=(), cleared=(A,))
    monkeypatch.undo()

    assert [r.verdict for r in _rows(lane, A)] == ['fails_in_isolation', 'passes_in_isolation']
    assert [item.test_id for item in _due(lane)] == [A]
    assert lane.episode().live


@pytest.mark.asyncio
async def test_scenarios_31_and_40_a_clear_re_opens_debt_only_at_a_head_holding_the_fix(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red(reproduced=(), cleared=(A,))
    await lane.drain()
    first = read_debt(lane.ledger, A)
    assert first is not None and first.owner_task_id is not None
    _done(lane, first.owner_task_id)
    lane.ancestry.answers[(FIX_COMMIT, HEAD2)] = False

    await _clear(lane, head=HEAD2)
    await lane.drain()

    assert len(_rows(lane, A)) == 2
    assert _pending(lane, EscalationCategory.REGRESSED_AFTER_RESOLUTION) == []
    stale = read_debt(lane.ledger, A)
    assert stale is not None and stale.open_count == 1
    assert len(lane.client.submitted) == 1

    await _clear(lane, head=HEAD3)
    await lane.drain()

    reopened = read_debt(lane.ledger, A)
    assert reopened is not None and reopened.open_count == 2
    (regression,) = _pending(lane, EscalationCategory.REGRESSED_AFTER_RESOLUTION)
    assert 'offline-lane' in f'{regression.summary}\n{regression.detail}'
    cycle_one, cycle_two = lane.client.submitted
    assert 'cycle 2' in cycle_two['title']
    assert cycle_two['title'] != cycle_one['title']
    assert reopened.owner_task_id != first.owner_task_id


@pytest.mark.asyncio
async def test_scenario_40_a_clear_enqueued_before_the_fix_is_dropped_once_its_owner_is_done(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red(reproduced=(), cleared=(A,))
    await lane.drain()
    owner = read_debt(lane.ledger, A)
    assert owner is not None and owner.owner_task_id is not None
    lane.ancestry.answers[(FIX_COMMIT, HEAD2)] = False

    offered = await _clear(lane, head=HEAD2)
    _done(lane, owner.owner_task_id)
    await lane.drain()

    assert offered.debts_offered == (OfferedDebt(A, DebtTrigger.cleared, HEAD2),)
    assert _due(lane) == []
    assert _pending(lane, EscalationCategory.REGRESSED_AFTER_RESOLUTION) == []
    after = read_debt(lane.ledger, A)
    assert after is not None and after.open_count == 1
    assert len(lane.client.submitted) == 1


# --- a recovered test owes debt at the head of its evidence ------------------------------


async def _outage(lane: Lane) -> PassEffect:
    observation = lane.written_pass(HEAD1, None, _run('detect', HEAD1, lane.clock(), failing=(A,)), None)
    return await lane.engine.observe(replace(observation, lane_command=PROBED, outcome=Outage(OUTAGE), probe=OUTAGE))


@pytest.mark.asyncio
async def test_scenario_27_a_red_then_green_at_one_head_owes_debt_at_that_head(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=10)

    effect = await lane.green()

    green = _current(lane)
    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.recovered, HEAD1),)
    (item,) = _due(lane)
    assert (item.test_id, item.head, item.observed_at) == (A, HEAD1, green.started_at.isoformat())
    context = due_item_context(item)
    assert context is not None
    assert (context.trigger, context.heads) == (DebtTrigger.recovered, (HEAD1,))
    assert context.reproduce is not None and A in context.reproduce
    assert [entry.first_line for entry in context.evidence] == [f'AssertionError in {A}']
    assert [r.verdict for r in _rows(lane, A)] == ['fails_in_isolation']

    await lane.drain()

    (filed,) = lane.client.submitted
    assert 'recovered on the offline lane (integration)' in filed['title']


@pytest.mark.asyncio
async def test_scenario_27_a_pass_at_another_head_owes_debt_at_the_earlier_green_head(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.green()
    lane.clock.advance(minutes=10)
    await lane.red()
    assert lane.episode().last_green_head == HEAD1
    lane.clock.advance(minutes=10)

    effect = await lane.red(reproduced=(B,), outcomes={A: 'passed', B: 'failed'}, head=HEAD2)

    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.recovered, HEAD1),)
    assert [(item.test_id, item.head) for item in _due(lane)] == [(A, HEAD1)]
    assert [r.merge_sha for r in _rows(lane, B)] == [HEAD2]


@pytest.mark.asyncio
async def test_two_recoveries_at_different_evidence_heads_make_two_items(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.green(head=HEAD1)
    lane.clock.advance(minutes=10)
    await lane.red((A,), head=HEAD1)
    lane.clock.advance(minutes=10)
    await lane.red((C,), head=HEAD2)
    lane.clock.advance(minutes=10)

    effect = await lane.green(head=HEAD2)

    assert effect.debts_offered == (
        OfferedDebt(A, DebtTrigger.recovered, HEAD1), OfferedDebt(C, DebtTrigger.recovered, HEAD2),
    )
    assert sorted((item.test_id, item.head) for item in _due(lane)) == [(A, HEAD1), (C, HEAD2)]


@pytest.mark.asyncio
async def test_an_uncounted_reproduction_head_is_the_recoverys_evidence_head(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red(head=HEAD1)
    lane.clock.advance(minutes=3)
    await lane.red(head=HEAD2)
    lane.clock.advance(minutes=3)

    effect = await lane.green(head=HEAD2)

    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.recovered, HEAD2),)
    assert [(item.test_id, item.head) for item in _due(lane)] == [(A, HEAD2)]


@pytest.mark.asyncio
async def test_scenario_28_a_test_that_never_passed_at_a_reproduced_head_owes_nothing(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red(head=HEAD1)
    lane.clock.advance(minutes=10)

    effect = await lane.green(head=HEAD2)

    assert effect.debts_offered == ()
    assert _due(lane) == []


@pytest.mark.asyncio
async def test_scenario_28_an_expired_episode_owes_nothing(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(hours=25)

    effect = await lane.green()

    assert [ended.reason for ended in effect.episodes_ended] == ['expired']
    assert (effect.debts_offered, _due(lane)) == ((), [])


@pytest.mark.asyncio
async def test_scenario_28_an_episode_that_saw_an_outage_pass_owes_nothing(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=10)
    await _outage(lane)
    lane.clock.advance(minutes=10)

    effect = await lane.green()

    assert [ended.reason for ended in effect.episodes_ended] == ['passed']
    assert (effect.debts_offered, _due(lane)) == ((), [])


@pytest.mark.asyncio
async def test_scenario_28_an_episode_whose_fix_task_was_filed_owes_nothing(tmp_path: Path):
    lane = Lane(tmp_path)
    await _fix_filed(lane)
    lane.clock.advance(minutes=10)

    effect = await lane.green()

    assert [ended.reason for ended in effect.episodes_ended] == ['passed']
    assert (effect.debts_offered, _due(lane)) == ((), [])


@pytest.mark.asyncio
async def test_scenario_28_a_cleared_episode_owes_debt_as_cleared_never_recovered(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.red()

    effect = await _clear(lane)

    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.cleared, HEAD1),)
    (item,) = _due(lane)
    context = due_item_context(item)
    assert context is not None and context.trigger is DebtTrigger.cleared


@pytest.mark.asyncio
async def test_scenario_34_a_restart_between_the_recorder_and_the_store_files_one_owner(tmp_path: Path, monkeypatch):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=10)

    def unwritable(self, record):
        raise OSError('episode file unwritable')

    monkeypatch.setattr(EpisodeStore, 'save', unwritable)
    with pytest.raises(OSError):
        await lane.green()
    monkeypatch.undo()

    assert [item.test_id for item in _due(lane)] == [A]
    assert lane.episode().live

    lane.restarted()
    await lane.drain()
    debt = read_debt(lane.ledger, A)
    assert debt is not None and debt.owner_task_id is not None

    lane.clock.advance(minutes=10)
    again = await lane.green()
    assert again.debts_offered == (OfferedDebt(A, DebtTrigger.recovered, HEAD1),)
    await lane.drain()

    assert len(lane.client.submitted) == 1
    assert _due(lane) == []


@pytest.mark.asyncio
async def test_scenario_40_a_recovery_is_judged_at_its_evidence_head_not_the_ending_passes(tmp_path: Path):
    lane = Lane(tmp_path)
    await lane.green(head=HEAD1)
    await _clear(lane, head=HEAD1)
    await lane.drain()
    owner = read_debt(lane.ledger, A)
    assert owner is not None and owner.owner_task_id is not None
    _done(lane, owner.owner_task_id)
    lane.ancestry.answers[(FIX_COMMIT, HEAD1)] = False
    lane.clock.advance(minutes=10)
    await lane.red((A,), head=HEAD1)
    lane.clock.advance(minutes=10)

    effect = await lane.green(head=HEAD2)
    await lane.drain()

    assert effect.debts_offered == (OfferedDebt(A, DebtTrigger.recovered, HEAD1),)
    assert (FIX_COMMIT, HEAD1) in lane.ancestry.calls
    assert (FIX_COMMIT, HEAD2) not in lane.ancestry.calls
    assert _due(lane) == []
    assert _pending(lane, EscalationCategory.REGRESSED_AFTER_RESOLUTION) == []
    after = read_debt(lane.ledger, A)
    assert after is not None and after.open_count == 1
    assert len(lane.client.submitted) == 1


# --- the fix digest names the live de-flake owner ---------------------------------------


async def _owed(lane: Lane, node: str = A) -> str:
    """*node* cleared once with no episode and drained: the id of its open de-flake owner."""
    await lane.red(reproduced=(), cleared=(node,))
    await lane.drain()
    debt = read_debt(lane.ledger, node)
    assert debt is not None and debt.owner_task_id is not None
    lane.clock.advance(minutes=30)
    return debt.owner_task_id


@pytest.mark.parametrize(('status', 'named'), [('pending', True), ('deferred', True), ('done', False)])
@pytest.mark.asyncio
async def test_the_fix_digest_names_a_live_de_flake_owner(tmp_path: Path, status: str, named: bool):
    lane = Lane(tmp_path)
    owner = await _owed(lane)
    lane.client.tasks[owner]['status'] = status

    await _fix_filed(lane)

    details = lane.client.submitted[-1]['details']
    assert ('Live de-flake owners:' in details) is named
    assert (f'{A}: task {owner}' in details) is named


@pytest.mark.asyncio
async def test_an_unreadable_owner_status_is_named_as_unreadable_and_the_fix_is_still_filed(tmp_path: Path):
    lane = Lane(tmp_path)
    owner = await _owed(lane)
    await lane.red()
    lane.client.statuses_error = ConnectionError('task store unreachable')

    effect = await lane.spaced_red()

    assert len(effect.tasks_filed) == 1
    assert f'{A}: task {owner} (status unreadable' in lane.client.submitted[-1]['details']


@pytest.mark.asyncio
async def test_only_the_tests_owing_debt_are_listed(tmp_path: Path):
    lane = Lane(tmp_path)
    owner = await _owed(lane, B)
    await lane.red((A, B))

    await lane.spaced_red(A, B)

    details = lane.client.submitted[-1]['details']
    assert f'{B}: task {owner}' in details
    assert f'{A}: task' not in details
