"""The offline lane's episode records and the pure rules over them: the store, the D5 and D10 tables.

PRD ``plans/offline-lane-red-path-prd.md`` D5, D10, D12 and section 5.3, driven through
``EpisodeStore`` and ``apply_pass``.  The engine around them is
``test_offline_lane_episodes.py``'s subject.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD0,
    HEAD1,
    HEAD2,
    POLICY,
    SPACED,
    T0,
    A,
    B,
    C,
    Clock,
    StoreWrites,
    _apply,
    _covered,
    _episode,
    _episodes_path,
    _observation,
    _record,
    _red,
    _run,
)
from shared.lane_probe import OUTAGE_EXIT_STATUS

from orchestrator.guard_state import PersistentMap
from orchestrator.offline_lane_confirm import (
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import (
    EPISODE_RETENTION,
    CommandRecord,
    CoveringVerdict,
    Episode,
    EpisodeStore,
    LanePolicy,
    PassObservation,
    apply_outage,
    recheck_due,
    unconfirmable_node,
)
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import ProbeKind, ProbeResult

OUTAGE_PROBE = ProbeResult(ProbeKind.OUTAGE, OUTAGE_EXIT_STATUS, 'qdrant: get collections: ConnectError',
                           'qdrant: get collections: ConnectError', 1.5)
PI_KEYS = ('probe_broken_escalation_id', 'last_probe_at', 'probe_record_behind')

# --- records and store -------------------------------------------------------------


def _full_record() -> CommandRecord:
    note = EvidenceNote(source='hook', node=A, text='cache hit on a stale key')
    live = _covered(
        A,
        notes=(note,),
        prior_task_ids=('8001',),
        filing_due=True,
        filing_due_since=T0,
        reproduce='cd integration && pytest tests/test_a.py::test_one',
        filing_failure='server busy',
        escalation_id='esc-1',
    )
    ended = _episode(B, ended_at=T0 + SPACED, binding_swallowed=True, covering_task_id='9002')
    return _record(
        live,
        ended,
        rechecks={HEAD1: 1},
        outage_escalation_id='esc-2',
        last_probe_result=OUTAGE_PROBE,
        last_pass_started_at=T0,
        last_pass_head=HEAD1,
        last_pass_id='p0',
        last_green_head=HEAD0,
        stuck_filing_escalation_id='esc-3',
        probe_broken_escalation_id='esc-4',
        last_probe_at=T0 + SPACED,
        probe_record_behind=True,
    )


def test_a_command_record_survives_a_json_round_trip():
    record = _full_record()

    assert CommandRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_a_record_written_before_the_probe_fields_loads_whole_with_their_defaults(tmp_path: Path):
    payload = _full_record().to_json()
    for key in PI_KEYS:
        del payload[key]
    _stored_raw(tmp_path, COMMAND, payload)

    record = EpisodeStore(tmp_path, clock=Clock()).load(COMMAND)

    assert record == replace(
        _full_record(), probe_broken_escalation_id=None, last_probe_at=None, probe_record_behind=False,
    )
    assert record.episodes[A].covering_task_id == '9001'
    assert (record.episodes[A].escalation_id, record.outage_escalation_id) == ('esc-1', 'esc-2')


@pytest.mark.parametrize('key', ['last_pass_id', 'last_probe_result', 'outage_escalation_id'])
def test_a_record_missing_a_key_older_than_the_probe_fields_is_refused(key: str):
    payload = _full_record().to_json()
    del payload[key]

    with pytest.raises(ValueError, match=key):
        CommandRecord.from_json(payload)


def test_an_outage_observation_carries_the_probe_that_answered_it():
    with pytest.raises(ValueError):
        replace(_observation(), outcome=Outage(OUTAGE_PROBE))


def test_a_legacy_observation_carries_no_probe():
    with pytest.raises(ValueError):
        replace(_observation(), leg='numeric', lane_command=None, probe=OUTAGE_PROBE)


def test_a_due_episode_without_its_reproduce_line_is_refused():
    with pytest.raises(ValueError):
        _episode(filing_due=True, filing_due_since=T0)


def test_a_generic_observation_needs_its_lane_command():
    with pytest.raises(ValueError):
        replace(_observation(), lane_command=None)


def test_an_unknown_command_loads_an_empty_record(tmp_path: Path):
    store = EpisodeStore(tmp_path, clock=Clock())

    assert store.load(COMMAND) == CommandRecord.empty(COMMAND)


def _stored_raw(tmp_path: Path, command: str, payload: object) -> None:
    """Store *payload* under *command* as the episode store's own map would, bypassing its checks."""
    PersistentMap(_episodes_path(tmp_path), ttl=EPISODE_RETENTION, now=Clock())[command] = payload


def test_a_malformed_record_is_logged_loudly_and_loads_empty(tmp_path: Path, caplog):
    payload = _full_record().to_json()
    payload['episodes'][0]['counted_passes'] = 'p0'
    _stored_raw(tmp_path, COMMAND, payload)

    with caplog.at_level(logging.ERROR):
        record = EpisodeStore(tmp_path, clock=Clock()).load(COMMAND)

    assert record == CommandRecord.empty(COMMAND)
    assert any(r.levelno == logging.ERROR and 'malformed' in r.getMessage() for r in caplog.records)


def test_a_record_stored_under_another_command_loads_empty(tmp_path: Path, caplog):
    _stored_raw(tmp_path, 'other', _full_record().to_json())

    with caplog.at_level(logging.ERROR):
        record = EpisodeStore(tmp_path, clock=Clock()).load('other')

    assert record == CommandRecord.empty('other')
    assert any(r.levelno == logging.ERROR and repr(COMMAND) in r.getMessage() for r in caplog.records)


def test_one_save_is_one_file_write(tmp_path: Path):
    clock = Clock()
    writes = StoreWrites(tmp_path, clock)
    clock.observers.append(writes.sample)
    store = EpisodeStore(tmp_path, clock=clock)

    store.save(_full_record())
    writes.sample()

    assert writes.records == [_full_record()]


def test_pinned_pass_ids_cover_live_and_ended_episodes_across_commands(tmp_path: Path):
    store = EpisodeStore(tmp_path, clock=Clock())
    store.save(_record(_episode(A, counted_passes=('p1', 'p2')), _episode(B, counted_passes=('p3',), ended_at=T0)))
    other = replace(CommandRecord.empty('other'), episodes={C: _episode(C, counted_passes=('p9',))})
    store.save(other)

    assert store.pinned_pass_ids() == frozenset({'p1', 'p2', 'p3', 'p9'})


def test_the_pass_effect_is_json(tmp_path: Path):
    applied = _apply(_record(_episode()), _red(cleared=(A,), reproduced=(B,)))

    assert json.loads(json.dumps(applied.effect.to_json()))['episodes_ended'] == [
        {'node': A, 'reason': 'cleared', 'counted_passes': 1, 'reproduced_heads': [HEAD0]},
    ]


# --- D5: what one pass does to a held test's episode -------------------------------

D5_ROWS = {
    'reproduced, at least recheck_secs after the last counted start: extends': (
        _red(at=T0 + SPACED), True, None,
    ),
    'reproduced, sooner than that: none': (
        _red(at=T0 + SPACED - timedelta(seconds=1)), False, None,
    ),
    'cleared: ends': (
        _red(reproduced=(B,), cleared=(A,)), False, 'cleared',
    ),
    'passed, per-test evidence: ends': (
        _red(reproduced=(B,), outcomes={A: 'passed', B: 'failed'}), False, 'passed',
    ),
    'passed, no per-test evidence and detection exited 0: ends': (
        _observation(), False, 'passed',
    ),
    'inconclusive: none': (
        _red(reproduced=(B,), inconclusive=(A,)), False, None,
    ),
    'skipped per the per-test evidence: none': (
        _red(reproduced=(B,), outcomes={A: 'skipped', B: 'failed'}), False, None,
    ),
    'absent from the failing set, no per-test evidence, detection non-zero: none': (
        _red(reproduced=(B,)), False, None,
    ),
    'the pass is unconfirmable: none for the test': (
        _observation(
            outcome=Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '1'),
            detect=_run('detect', HEAD1, T0 + SPACED, failing=(A,)),
        ),
        False,
        None,
    ),
}


@pytest.mark.parametrize(('observation', 'counted', 'ended'), D5_ROWS.values(), ids=list(D5_ROWS))
def test_d5_row(observation: PassObservation, counted: bool, ended: str | None):
    applied = _apply(_record(_episode()), observation)

    episode = applied.record.episodes[A]
    assert (A in applied.effect.tests_counted) is counted
    assert len(episode.counted_passes) == (2 if counted else 1)
    assert [e.reason for e in applied.effect.episodes_ended if e.node == A] == ([ended] if ended else [])
    assert episode.live is (ended is None)


def test_d5_row_the_covering_task_is_done_and_the_head_lacks_its_commit_is_a_stale_head():
    applied = _apply(
        _record(_covered()), _red(), verdicts={A: CoveringVerdict.DONE_STALE_HEAD},
    )

    assert applied.effect.stale_heads == (A,)
    assert applied.effect.tests_counted == ()
    assert applied.record.episodes[A] == replace(_covered(), last_refreshed_at=T0 + SPACED)


def test_an_unconfirmable_pass_is_a_reproduced_pass_of_the_command_pseudo_test():
    reason = UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST
    observation = _observation(outcome=Unconfirmable(reason, '1'), detect=_run('detect', HEAD1, T0, failing=(A,)))

    applied = _apply(_record(_episode()), observation)

    pseudo = applied.record.episodes[unconfirmable_node(COMMAND)]
    assert applied.effect.episodes_started == (unconfirmable_node(COMMAND),)
    assert pseudo.kind == reason.value
    assert applied.record.episodes[A] == replace(_episode(), last_refreshed_at=T0 + SPACED)


def test_a_reproduction_sooner_than_recheck_still_records_its_head():
    applied = _apply(_record(_episode()), _red(at=T0 + timedelta(minutes=3), head=HEAD2))

    assert applied.record.episodes[A].reproduced_heads == (HEAD0, HEAD2)


def test_a_stale_head_records_no_reproduced_head():
    applied = _apply(_record(_covered()), _red(head=HEAD2), verdicts={A: CoveringVerdict.DONE_STALE_HEAD})

    assert applied.record.episodes[A].reproduced_heads == (HEAD0,)


# --- D10: what the covering task's status means when the test is red again ----------

D10_ROWS = {
    'open: nothing; the record accumulates passes': (
        CoveringVerdict.OPEN, _covered(), 2, '9001', (), False,
    ),
    'deferred, commit_planning never confirmed: finishes the filing': (
        CoveringVerdict.DEFERRED_UNFINISHED, _covered(planning_committed=False), 2, '9001', (), True,
    ),
    'deferred otherwise: nothing': (
        CoveringVerdict.DEFERRED, _covered(), 2, '9001', (), False,
    ),
    'done, contained: binding dropped, a new episode at 1 names the prior task': (
        CoveringVerdict.DONE_CONTAINED, _covered(), 1, None, ('9001',), False,
    ),
    'done, head lacks the fix: a stale head': (
        CoveringVerdict.DONE_STALE_HEAD, _covered(), 1, '9001', (), False,
    ),
    'cancelled or absent, episode live: no refile while it lasts': (
        CoveringVerdict.CLOSED, _covered(), 2, '9001', (), False,
    ),
    'cancelled or absent, episode ended: binding dropped, a new episode at 1': (
        CoveringVerdict.CLOSED, _covered(ended_at=T0), 1, None, ('9001',), False,
    ),
    'cancelled or absent, its escalation resolved: binding dropped, a new episode at 1': (
        CoveringVerdict.CLOSED_RELEASED, _covered(escalation_id='esc-1'), 1, None, ('9001',), False,
    ),
    'unreadable: no decision this pass': (
        CoveringVerdict.UNREADABLE, _covered(), 1, '9001', (), False,
    ),
}


@pytest.mark.parametrize(
    ('verdict', 'before', 'counted', 'covering', 'prior', 'due'), D10_ROWS.values(), ids=list(D10_ROWS),
)
def test_d10_row(
    verdict: CoveringVerdict, before: Episode, counted: int, covering: str | None,
    prior: tuple[str, ...], due: bool,
):
    applied = _apply(_record(before), _red(), verdicts={A: verdict})

    episode = applied.record.episodes[A]
    assert len(episode.counted_passes) == counted
    assert episode.covering_task_id == covering
    assert episode.prior_task_ids == prior
    assert episode.filing_due is due
    assert episode.live
    assert applied.to_file == ()
    assert (A in applied.effect.no_decision) is (verdict is CoveringVerdict.UNREADABLE)
    assert (A in applied.effect.stale_heads) is (verdict is CoveringVerdict.DONE_STALE_HEAD)


def test_a_done_contained_task_ends_the_live_episode_it_covered():
    applied = _apply(_record(_covered()), _red(), verdicts={A: CoveringVerdict.DONE_CONTAINED})

    assert [(e.node, e.reason) for e in applied.effect.episodes_ended] == [(A, 'replaced')]
    assert applied.effect.episodes_started == (A,)


# --- persistence, grouping, expiry, notes, refresh ----------------------------------


def test_a_first_reproduction_holds_one_counted_pass_and_files_nothing():
    applied = _apply(_record(), _red())

    assert applied.record.episodes[A].counted_passes == ('p1',)
    assert applied.to_file == ()


def test_persist_passes_of_one_files_on_the_first_reproduced_pass():
    applied = _apply(_record(), _red(), policy=LanePolicy(persist_passes=1))

    assert applied.to_file == ((A,),)


def test_tests_crossing_on_one_pass_share_a_group():
    applied = _apply(_record(_episode(A), _episode(B)), _red(reproduced=(A, B)))

    assert applied.to_file == ((A, B),)
    assert applied.effect.threshold_reached == (A, B)


def test_a_test_crossing_on_a_later_pass_gets_its_own_group():
    first = _apply(_record(_episode(A), _episode(B), _episode(C, last_counted_started_at=T0 + SPACED)),
                   _red(reproduced=(A, B, C)))
    later = _apply(first.record, _red(reproduced=(A, B, C), at=T0 + 2 * SPACED, pass_id='p2'))

    assert first.to_file == ((A, B),)
    assert later.to_file == ((C,),)


def test_a_held_episode_no_pass_extends_expires_after_red_unlanded_hours():
    late = T0 + POLICY.red_unlanded
    applied = _apply(_record(_episode()), _red(reproduced=(B,), at=late))

    assert [(e.node, e.reason) for e in applied.effect.episodes_ended] == [(A, 'expired')]
    assert not applied.record.episodes[A].live


def test_expiry_is_measured_from_the_pass_start_not_from_the_clock():
    started = T0 + POLICY.red_unlanded - timedelta(seconds=1)

    applied = _apply(_record(_episode()), _red(reproduced=(B,), at=started), now=started + timedelta(hours=1))

    assert applied.record.episodes[A].live


def test_persist_passes_of_one_files_a_restarted_episode_on_its_first_counted_pass():
    applied = _apply(
        _record(_covered()), _red(), verdicts={A: CoveringVerdict.DONE_CONTAINED}, policy=LanePolicy(persist_passes=1),
    )

    assert applied.effect.episodes_started == (A,)
    assert applied.to_file == ((A,),)


def test_a_crossing_test_whose_filing_is_already_due_is_not_filed_again():
    due = _episode(filing_due=True, filing_due_since=T0, reproduce='cd integration && pytest -m integration tests')

    applied = _apply(_record(due), _red())

    assert applied.effect.threshold_reached == (A,)
    assert applied.to_file == ()


def test_a_held_episode_just_inside_red_unlanded_hours_does_not_expire():
    applied = _apply(_record(_episode()), _red(reproduced=(B,), at=T0 + POLICY.red_unlanded - timedelta(seconds=1)))

    assert applied.record.episodes[A].live


def test_a_new_episode_takes_the_previous_pass_head_when_that_pass_showed_the_test_passed():
    previous = _run('detect', HEAD0, T0, failing=(B,), outcomes={A: 'passed', B: 'failed'})

    applied = _apply(_record(last_green_head='e' * 40), _red(), previous=previous)

    assert applied.record.episodes[A].last_green_head == HEAD0


def test_a_new_episode_falls_back_to_the_command_last_green_head():
    previous = _run('detect', HEAD0, T0, failing=(A,))

    applied = _apply(_record(last_green_head='e' * 40), _red(), previous=previous)

    assert applied.record.episodes[A].last_green_head == 'e' * 40


def test_an_unnamed_note_reaches_a_reproduced_test_whose_pass_is_not_counted():
    unnamed = EvidenceNote('hook', None, 'unnamed')

    applied = _apply(_record(_episode()), _red(at=T0 + timedelta(minutes=3), notes=(unnamed,)))

    assert applied.effect.tests_counted == ()
    assert applied.record.episodes[A].notes == (unnamed,)


def test_named_notes_go_to_their_test_and_unnamed_ones_to_every_reproduced_test():
    named = EvidenceNote('hook', B, 'named')
    unnamed = EvidenceNote('hook', None, 'unnamed')

    applied = _apply(_record(), _red(reproduced=(A, B), notes=(named, unnamed)))

    assert applied.record.episodes[A].notes == (unnamed,)
    assert applied.record.episodes[B].notes == (named, unnamed)


def test_the_command_fields_follow_the_pass():
    record = _record(rechecks={HEAD0: 1, HEAD1: 1}, last_green_head=HEAD0)

    red = _apply(record, _red(head=HEAD1)).record
    green = _apply(red, _observation(head=HEAD2, pass_id='p2', at=T0 + 2 * SPACED)).record

    assert (red.last_pass_id, red.last_pass_head, red.last_pass_started_at) == ('p1', HEAD1, T0 + SPACED)
    assert red.last_green_head == HEAD0
    assert red.rechecks == {HEAD1: 1}
    assert green.last_green_head == HEAD2
    assert (green.leg, green.fix_task_priority) == ('generic', 'medium')


def test_live_and_open_bound_episodes_are_refreshed():
    now = T0 + timedelta(days=20)
    open_bound = _covered(B, ended_at=T0)
    applied = _apply(_record(_episode(last_counted_started_at=now), open_bound), _red(reproduced=(C,), at=now))

    assert applied.record.episodes[A].last_refreshed_at == now
    assert applied.record.episodes[B].last_refreshed_at == now


def test_an_ended_episode_whose_task_is_terminal_is_dropped_fourteen_days_after_its_last_refresh():
    ended = _covered(B, ended_at=T0)
    record = _record(ended)

    kept = _apply(record, _observation(at=T0 + timedelta(days=14) - timedelta(seconds=1)), terminal=frozenset({'9001'}))
    dropped = _apply(record, _observation(at=T0 + timedelta(days=14)), terminal=frozenset({'9001'}))

    assert kept.record.episodes[B] == ended
    assert B not in dropped.record.episodes


def test_an_unlanded_covering_task_is_named_on_a_counted_pass_at_red_unlanded_hours():
    at = T0 + POLICY.red_unlanded
    record = _record(_covered(last_counted_started_at=at - SPACED))

    applied = _apply(record, _red(at=at), verdicts={A: CoveringVerdict.OPEN})
    early = _apply(record, _red(at=at - timedelta(seconds=1)), verdicts={A: CoveringVerdict.OPEN})
    raised = _apply(_record(_covered(last_counted_started_at=at - SPACED, escalation_id='esc-1')), _red(at=at),
                    verdicts={A: CoveringVerdict.OPEN})

    assert applied.unlanded == ('9001',)
    assert early.unlanded == ()
    assert raised.unlanded == ()


def test_an_unlanded_covering_task_is_named_only_on_a_counted_pass():
    at = T0 + POLICY.red_unlanded
    record = _record(_covered(last_counted_started_at=at - timedelta(minutes=3)))

    applied = _apply(record, _red(at=at), verdicts={A: CoveringVerdict.OPEN})

    assert applied.effect.tests_counted == ()
    assert applied.unlanded == ()


@pytest.mark.parametrize(
    'verdict',
    [CoveringVerdict.UNREADABLE, CoveringVerdict.DONE_STALE_HEAD, CoveringVerdict.DONE_CONTAINED,
     CoveringVerdict.CLOSED_RELEASED],
)
def test_no_unlanded_escalation_for_a_contained_released_stale_or_unread_task(verdict: CoveringVerdict):
    at = T0 + POLICY.red_unlanded
    applied = _apply(_record(_covered(last_counted_started_at=at - SPACED)), _red(at=at), verdicts={A: verdict})

    assert applied.unlanded == ()


# --- D5's outage row and D12's outage hold -----------------------------------------


def _outage_pass(**fields) -> PassObservation:
    return replace(_observation(**fields), outcome=Outage(OUTAGE_PROBE), probe=OUTAGE_PROBE)


def test_apply_pass_refuses_an_outage_pass():
    with pytest.raises(ValueError, match='apply_outage'):
        _apply(_record(_episode()), _outage_pass())


def test_an_outage_pass_marks_every_live_episode_and_changes_nothing_else():
    live = _episode(A)
    covered = _covered(B)
    ended = _episode(C, ended_at=T0)
    record = replace(
        _record(live, covered, ended, rechecks={HEAD0: 1}, last_pass_started_at=T0, last_pass_head=HEAD0,
                last_pass_id='p0', last_green_head=HEAD0),
        leg=None, fix_task_priority=None,
    )

    after = apply_outage(record, _outage_pass(pass_id='p1', head=HEAD1, at=T0 + 2 * SPACED))

    assert after.episodes == {
        A: replace(live, had_outage_pass=True),
        B: replace(covered, had_outage_pass=True),
        C: ended,
    }
    assert after == replace(record, episodes=after.episodes, leg='generic', fix_task_priority='medium')


def test_apply_outage_takes_an_outage_pass_only():
    with pytest.raises(ValueError):
        apply_outage(_record(_episode()), _red())


def test_no_recheck_falls_due_while_the_command_is_in_an_outage():
    record = _record(_episode(last_counted_started_at=T0), last_pass_started_at=T0)
    now = T0 + POLICY.recheck

    assert recheck_due(record, HEAD1, POLICY, now) is not None
    assert recheck_due(replace(record, last_probe_result=OUTAGE_PROBE), HEAD1, POLICY, now) is None
