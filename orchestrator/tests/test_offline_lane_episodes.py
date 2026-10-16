"""The offline lane's red-path engine: episode records, the D5 and D10 tables, and the engine's scenarios.

PRD ``plans/offline-lane-red-path-prd.md`` D5, D10-D12, section 5.3-5.4 and the engine-level
halves of section 6's scenarios.  The pure rules are driven through ``apply_pass``; filing
and escalation through ``LaneEngine.observe`` and ``LaneEngine.tick`` over a fake task
client, a fake ancestry read, an injected clock and a real escalation queue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from escalation.queue import EscalationQueue
from shared import safe_io
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask, SubmittedTicket

from orchestrator.config import LaneCommand, OrchestratorConfig
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import (
    EPISODES_FILE,
    Applied,
    CommandRecord,
    CoveringVerdict,
    Episode,
    EpisodeStore,
    LanePolicy,
    PassEffect,
    PassObservation,
    PassReads,
    apply_pass,
    unconfirmable_node,
)
from orchestrator.offline_lane_episodes import LaneEngine, TickResult
from orchestrator.offline_lane_escalations import (
    STUCK_FILING_CATEGORY,
    UNLANDED_RED_CATEGORY,
    escalation_state,
    sentinel_task_id,
)
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import (
    LEGACY_NUMERIC_COMMAND,
    FailureRecord,
    RunRecord,
    RunRecordStore,
)

logger = logging.getLogger(__name__)

COMMAND = 'integration'
A = 'tests/test_a.py::test_one'
B = 'tests/test_b.py::test_two'
C = 'tests/test_c.py::test_three'
HEAD0 = '0' * 40
HEAD1 = 'a1' * 20
HEAD2 = 'b2' * 20
HEAD3 = 'c3' * 20
FIX_COMMIT = 'f' * 40
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
LANE_COMMAND = LaneCommand(name=COMMAND, command='pytest -m integration tests', cwd='integration')
SPACED = timedelta(minutes=30)
POLICY = LanePolicy()


class Clock:
    def __init__(self, start: datetime = T0) -> None:
        self.at = start

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **delta: float) -> None:
        self.at += timedelta(**delta)


def _run(
    phase: str,
    head: str,
    at: datetime,
    *,
    failing: tuple[str, ...] = (),
    returncode: int | None = None,
    outcomes: dict[str, str] | None = None,
) -> RunRecord:
    return RunRecord(
        command=COMMAND,
        leg='generic',
        head=head,
        phase=phase,
        started_at=at,
        ended_at=at,
        returncode=(1 if failing else 0) if returncode is None else returncode,
        ended_by='exit',
        output_path=f'/records/{phase}.log',
        junit_path=None,
        failures=tuple(FailureRecord(node, 'failed', f'AssertionError in {node}') for node in failing),
        ids_available=True,
        outcomes=outcomes,
        session_complete=True,
        probe=None,
    )


def _observation(
    *,
    pass_id: str = 'p1',
    at: datetime = T0 + SPACED,
    head: str = HEAD1,
    outcome: Confirmed | Unconfirmable | None = None,
    detect: RunRecord | None = None,
    notes: tuple[EvidenceNote, ...] = (),
) -> PassObservation:
    return PassObservation(
        command=COMMAND,
        leg='generic',
        head=head,
        pass_id=pass_id,
        started_at=at,
        detect=detect if detect is not None else _run('detect', head, at),
        confirm=None,
        outcome=outcome,
        fix_task_priority='medium',
        notes=notes,
        lane_command=LANE_COMMAND,
    )


def _red(
    reproduced: tuple[str, ...] = (A,),
    cleared: tuple[str, ...] = (),
    inconclusive: tuple[str, ...] = (),
    *,
    outcomes: dict[str, str] | None = None,
    **fields: Any,
) -> PassObservation:
    at = fields.get('at', T0 + SPACED)
    head = fields.get('head', HEAD1)
    detect = _run('detect', head, at, failing=(*reproduced, *cleared, *inconclusive), outcomes=outcomes)
    outcome = Confirmed(ConfirmMode.NARROWED, reproduced, cleared, inconclusive)
    return _observation(outcome=outcome, detect=detect, **fields)


def _episode(node: str = A, **fields: Any) -> Episode:
    values: dict[str, Any] = {
        'node': node,
        'kind': 'failed',
        'counted_passes': ('p0',),
        'last_counted_started_at': T0,
        'first_red_head': HEAD0,
        'reproduced_heads': (HEAD0,),
        'last_green_head': None,
        'had_outage_pass': False,
        'covering_task_id': None,
        'planning_committed': False,
        'filed_at': None,
        'filing_due': False,
        'filing_due_since': None,
        'escalation_id': None,
        'notes': (),
        'prior_task_ids': (),
        'ended_at': None,
        'last_refreshed_at': T0,
    }
    values.update(fields)
    return Episode(**values)


def _covered(node: str = A, task_id: str = '9001', **fields: Any) -> Episode:
    return _episode(node, **{'covering_task_id': task_id, 'planning_committed': True, 'filed_at': T0, **fields})


def _record(*episodes: Episode, **fields: Any) -> CommandRecord:
    base = replace(CommandRecord.empty(COMMAND), leg='generic', fix_task_priority='medium')
    return replace(base, episodes={episode.node: episode for episode in episodes}, **fields)


def _apply(
    record: CommandRecord,
    observation: PassObservation,
    *,
    verdicts: dict[str, CoveringVerdict] | None = None,
    terminal: frozenset[str] = frozenset(),
    previous: RunRecord | None = None,
    policy: LanePolicy = POLICY,
    now: datetime | None = None,
) -> Applied:
    reads = PassReads(verdicts or {}, terminal, previous)
    return apply_pass(record, observation, reads, policy, now or observation.started_at)


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
        last_probe_result={'result': 'up'},
        last_pass_started_at=T0,
        last_pass_head=HEAD1,
        last_pass_id='p0',
        last_green_head=HEAD0,
        stuck_filing_escalation_id='esc-3',
    )


def test_a_command_record_survives_a_json_round_trip():
    record = _full_record()

    assert CommandRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_a_due_episode_without_its_reproduce_line_is_refused():
    with pytest.raises(ValueError):
        _episode(filing_due=True, filing_due_since=T0)


def test_a_generic_observation_needs_its_lane_command():
    with pytest.raises(ValueError):
        replace(_observation(), lane_command=None)


def test_an_unknown_command_loads_an_empty_record(tmp_path: Path):
    store = EpisodeStore(tmp_path, clock=Clock())

    assert store.load(COMMAND) == CommandRecord.empty(COMMAND)


def test_a_malformed_record_is_logged_loudly_and_loads_empty(tmp_path: Path, caplog):
    clock = Clock()
    EpisodeStore(tmp_path, clock=clock).save(_full_record())
    path = tmp_path / 'data' / 'orchestrator' / 'guards' / EPISODES_FILE
    payload = json.loads(path.read_text())
    payload[COMMAND]['value']['episodes'][0]['counted_passes'] = 'p0'
    path.write_text(json.dumps(payload))

    with caplog.at_level(logging.ERROR):
        record = EpisodeStore(tmp_path, clock=clock).load(COMMAND)

    assert record == CommandRecord.empty(COMMAND)
    assert any(r.levelno == logging.ERROR and 'malformed' in r.getMessage() for r in caplog.records)


class _WriteCounter:
    def __init__(self, monkeypatch, log: list[str] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._log = log
        real = safe_io.atomic_write_text

        def counting(path, text, *args, **kwargs):
            if Path(path).name == EPISODES_FILE:
                self.payloads.append(json.loads(text))
                if self._log is not None:
                    self._log.append('write')
            return real(path, text, *args, **kwargs)

        monkeypatch.setattr(safe_io, 'atomic_write_text', counting)

    def episode(self, index: int, node: str = A) -> dict[str, Any]:
        episodes = self.payloads[index][COMMAND]['value']['episodes']
        return next(e for e in episodes if e['node'] == node)


def test_one_save_is_one_file_write(tmp_path: Path, monkeypatch):
    writes = _WriteCounter(monkeypatch)
    store = EpisodeStore(tmp_path, clock=Clock())

    store.save(_full_record())

    assert len(writes.payloads) == 1
    assert EpisodeStore(tmp_path, clock=Clock()).load(COMMAND) == _full_record()


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


@pytest.mark.parametrize(
    'verdict',
    [CoveringVerdict.UNREADABLE, CoveringVerdict.DONE_STALE_HEAD, CoveringVerdict.DONE_CONTAINED,
     CoveringVerdict.CLOSED_RELEASED],
)
def test_no_unlanded_escalation_for_a_contained_released_stale_or_unread_task(verdict: CoveringVerdict):
    at = T0 + POLICY.red_unlanded
    applied = _apply(_record(_covered(last_counted_started_at=at - SPACED)), _red(at=at), verdicts={A: verdict})

    assert applied.unlanded == ()


# --- the engine: observe and tick over fakes ----------------------------------------


class FakeLedgerClient:
    """The task seam as ``chronic_flake.py::SchedulerChronicFlakeTaskClient`` behaves: a failed
    ``commit_planning`` is only logged, so the caller learns of it from a live read."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.tasks: dict[str, dict[str, Any]] = {}
        self.outcomes: list[SubmitOutcome | Exception] = []
        self.submitted: list[dict[str, Any]] = []
        self.statuses_error: Exception | None = None
        self.unreadable: set[str] = set()
        self.commit_failures = 0
        self._next_id = 7000

    def fail_submissions(self, *reasons: str) -> None:
        self.outcomes = [SubmitFailed(reason) for reason in reasons]

    def set_status(self, task_id: str, status: str, **metadata: Any) -> None:
        self.tasks[task_id] = {'id': task_id, 'status': status, 'metadata': metadata}

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        self.log.append('submit_task')
        self.submitted.append(arguments)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self._next_id += 1
        task_id = str(self._next_id)
        self.set_status(task_id, 'deferred')
        return SubmittedTask(task_id, 'deferred')

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        self.log.append('get_statuses')
        if self.statuses_error is not None:
            return {}, self.statuses_error
        return {i: self.tasks[i]['status'] for i in ids if i in self.tasks}, None

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        self.log.append('get_task')
        if task_id in self.unreadable:
            return None, ConnectionError('task store unreachable')
        return self.tasks.get(task_id), None

    async def commit_planning(self, task_ids: list[str]) -> None:
        self.log.append('commit_planning')
        if self.commit_failures:
            self.commit_failures -= 1
            logger.warning('commit_planning failed for %s; they stay deferred', task_ids)
            return
        for task_id in task_ids:
            if self.tasks.get(task_id, {}).get('status') == 'deferred':
                self.tasks[task_id]['status'] = 'pending'


class FakeAncestry:
    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], bool | None] = {}
        self.calls: list[tuple[str, str]] = []

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None:
        self.calls.append((ancestor, descendant))
        return self.answers.get((ancestor, descendant), True)


class FailingOnceQueue(EscalationQueue):
    def __init__(self, queue_dir: Path) -> None:
        super().__init__(queue_dir)
        self.failures_left = 1

    def submit(self, escalation):
        if self.failures_left:
            self.failures_left -= 1
            raise OSError('queue directory unwritable')
        return super().submit(escalation)


class Lane:
    """A lane command's passes, written as real run records and observed by a real engine."""

    def __init__(self, tmp_path: Path, *, client: bool = True, queue: EscalationQueue | None | bool = True) -> None:
        self.tmp_path = tmp_path
        self.log: list[str] = []
        self.clock = Clock()
        self.client = FakeLedgerClient(self.log)
        self.ancestry = FakeAncestry()
        self.queue = EscalationQueue(tmp_path / 'queue') if queue is True else (queue or None)
        self.records = RunRecordStore(tmp_path, clock=self.clock)
        self._with_client = client
        self.engine = self.restarted()

    def restarted(self) -> LaneEngine:
        self.engine = LaneEngine(
            OrchestratorConfig(project_root=self.tmp_path),
            self.ancestry,
            task_client=self.client if self._with_client else None,
            escalation_queue=self.queue,
            now=self.clock,
        )
        return self.engine

    async def _observe(
        self,
        head: str,
        outcome: Confirmed | Unconfirmable | None,
        detect: RunRecord,
        confirm: RunRecord | None,
        notes: tuple[EvidenceNote, ...] = (),
        *,
        command: str = COMMAND,
        leg: str = 'generic',
    ) -> PassEffect:
        lane_pass = self.records.allocate_pass(command, leg, head)
        detect = replace(detect, command=command, leg=leg)
        self.records.write_run(lane_pass, detect)
        if confirm is not None:
            self.records.write_run(lane_pass, replace(confirm, command=command, leg=leg))
        observation = PassObservation(
            command=command,
            leg=leg,
            head=head,
            pass_id=lane_pass.pass_id,
            started_at=lane_pass.started_at,
            detect=detect,
            confirm=confirm,
            outcome=outcome,
            fix_task_priority='medium' if leg == 'generic' else 'high',
            notes=notes,
            lane_command=LANE_COMMAND if leg == 'generic' else None,
        )
        return await self.engine.observe(observation)

    async def red(
        self,
        reproduced: tuple[str, ...] = (A,),
        cleared: tuple[str, ...] = (),
        inconclusive: tuple[str, ...] = (),
        *,
        head: str = HEAD1,
        outcomes: dict[str, str] | None = None,
        command: str = COMMAND,
        leg: str = 'generic',
    ) -> PassEffect:
        at = self.clock()
        detect = _run('detect', head, at, failing=(*reproduced, *cleared, *inconclusive), outcomes=outcomes)
        confirm = _run('confirm', head, at, failing=reproduced)
        outcome = Confirmed(ConfirmMode.NARROWED, reproduced, cleared, inconclusive)
        return await self._observe(head, outcome, detect, confirm, command=command, leg=leg)

    async def green(self, *, head: str = HEAD1) -> PassEffect:
        return await self._observe(head, None, _run('detect', head, self.clock()), None)

    async def unconfirmable(self, *, head: str = HEAD1) -> PassEffect:
        at = self.clock()
        outcome = Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')
        return await self._observe(head, outcome, _run('detect', head, at, failing=(A,)), _run('confirm', head, at, returncode=2))

    async def spaced_red(self, *nodes: str, minutes: float = 30, **fields: Any) -> PassEffect:
        self.clock.advance(minutes=minutes)
        return await self.red(nodes or (A,), **fields)

    async def tick(self, *, head: str = HEAD1) -> TickResult:
        return await self.engine.tick(head=head)

    def stored(self, command: str = COMMAND) -> CommandRecord:
        return EpisodeStore(self.tmp_path, clock=self.clock).load(command)

    def episode(self, node: str = A) -> Episode:
        return self.stored().episodes[node]

    def pending(self, category: str) -> list:
        sentinel = sentinel_task_id(COMMAND)
        return [e for e in self.queue.get_pending() if e.task_id == sentinel and e.category == category] if self.queue else []

    def lane_metadata(self, index: int = -1) -> dict[str, Any]:
        return self.client.submitted[index]['metadata']['x_offline_lane']


@pytest.mark.asyncio
async def test_scenario_3_a_first_reproduced_pass_files_nothing_and_holds_one_counted_pass(tmp_path):
    lane = Lane(tmp_path)

    effect = await lane.red()

    assert effect.tasks_filed == ()
    assert lane.client.submitted == []
    assert len(lane.episode().counted_passes) == 1
    assert effect.episodes_started == (A,)


@pytest.mark.asyncio
async def test_scenario_4_a_second_reproduction_thirty_minutes_on_files_one_committed_task(tmp_path):
    lane = Lane(tmp_path)
    first = await lane.red()

    effect = await lane.spaced_red()

    episode = lane.episode()
    assert effect.tasks_filed == ('7001',)
    assert len(lane.client.submitted) == 1
    for pass_id in episode.counted_passes:
        assert f'pass {pass_id} detect' in lane.client.submitted[0]['details']
    assert (episode.covering_task_id, episode.planning_committed, episode.filing_due) == ('7001', True, False)
    assert lane.client.tasks['7001']['status'] == 'pending'
    assert first.tests_counted == effect.tests_counted == (A,)
    assert 'Reproduce: cd integration && ' in lane.client.submitted[0]['description']
    assert A in lane.client.submitted[0]['description'].split('Reproduce: ')[1]


@pytest.mark.asyncio
async def test_scenario_5_a_reproduction_three_minutes_on_is_not_counted(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()

    effect = await lane.spaced_red(minutes=3)

    assert effect.tests_counted == ()
    assert lane.client.submitted == []
    assert len(lane.episode().counted_passes) == 1


@pytest.mark.asyncio
async def test_scenario_6_a_cleared_test_ends_its_episode_and_files_nothing(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=30)

    effect = await lane.red(reproduced=(), cleared=(A,))

    assert [(e.node, e.reason) for e in effect.episodes_ended] == [(A, 'cleared')]
    assert lane.client.submitted == []
    assert not lane.episode().live


@pytest.mark.asyncio
async def test_scenario_8_two_spaced_unconfirmable_passes_file_one_command_level_task(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    held = lane.episode()
    lane.clock.advance(minutes=30)
    await lane.unconfirmable()
    lane.clock.advance(minutes=30)

    effect = await lane.unconfirmable()

    assert len(lane.client.submitted) == 1
    arguments = lane.client.submitted[0]
    assert effect.tasks_filed == ('7001',)
    assert 'exits non-zero with no reproducible failing test' in arguments['title']
    assert 'Reproduce: cd integration && pytest -m integration tests' in arguments['description']
    assert 'confirm_exited_nonzero_naming_no_test' in arguments['description']
    assert lane.episode() == replace(held, last_refreshed_at=lane.clock())


@pytest.mark.asyncio
async def test_scenario_10_skipped_then_truncated_leave_the_episode_and_a_pass_ends_it(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    held = lane.episode()
    lane.clock.advance(minutes=30)
    await lane.red(reproduced=(B,), outcomes={A: 'skipped', B: 'failed'})
    after_skip = lane.episode()
    lane.clock.advance(minutes=30)
    await lane.red(reproduced=(B,), inconclusive=(A,))
    after_truncation = lane.episode()
    lane.clock.advance(minutes=30)

    effect = await lane.green()

    assert after_skip.counted_passes == after_truncation.counted_passes == held.counted_passes
    assert after_truncation.live
    assert (A, 'passed') in [(e.node, e.reason) for e in effect.episodes_ended]


@pytest.mark.asyncio
async def test_scenario_12_same_pass_crossers_share_a_task_and_a_later_crosser_gets_its_own(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    await lane.spaced_red(A, B)
    await lane.spaced_red(A, B, C)

    effect = await lane.spaced_red(A, B, C)
    await lane.spaced_red(A, B, C)

    assert [lane.lane_metadata(i)['nodes'] for i in range(len(lane.client.submitted))] == [[A, B], [C]]
    assert effect.tasks_filed == ('7002',)
    assert lane.episode(C).covering_task_id == '7002'


async def _filed(lane: Lane) -> str:
    await lane.red()
    effect = await lane.spaced_red()
    (task_id,) = effect.tasks_filed
    return task_id


@pytest.mark.asyncio
async def test_scenario_13_a_persistent_red_under_an_open_task_escalates_once_after_a_day(tmp_path):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    filed_at = lane.episode().filed_at
    assert filed_at is not None
    pending_after: list[int] = []

    for _ in range(12):
        effect = await lane.spaced_red(minutes=150)
        pending_after.append(len(lane.pending(UNLANDED_RED_CATEGORY)))
        assert effect.tasks_filed == ()

    assert pending_after == [0] * 9 + [1, 1, 1]
    (record,) = lane.pending(UNLANDED_RED_CATEGORY)
    assert task_id in record.summary
    assert lane.episode().escalation_id == record.id
    assert len(lane.client.submitted) == 1


@pytest.mark.asyncio
async def test_scenario_14_a_covered_test_red_again_after_green_files_nothing_new(tmp_path):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    lane.clock.advance(minutes=30)
    await lane.green()

    await lane.spaced_red()
    await lane.spaced_red()

    assert len(lane.client.submitted) == 1
    assert lane.episode().covering_task_id == task_id


@pytest.mark.asyncio
async def test_scenario_15_a_cancelled_task_escalates_once_and_after_resolution_a_new_task_names_it(tmp_path):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    lane.client.set_status(task_id, 'cancelled')
    for _ in range(10):
        await lane.spaced_red(minutes=150)
    (record,) = lane.pending(UNLANDED_RED_CATEGORY)
    assert len(lane.client.submitted) == 1
    assert lane.queue is not None
    lane.queue.resolve(record.id, 'seen; refile when it persists', resolved_by='operator')

    held = await lane.spaced_red()
    refiled = await lane.spaced_red()

    assert held.episodes_started == (A,)
    assert held.tasks_filed == ()
    assert refiled.tasks_filed == ('7002',)
    assert lane.lane_metadata()['prior_task_ids'] == [task_id]
    assert len(lane.pending(UNLANDED_RED_CATEGORY)) == 0


def _done(lane: Lane, task_id: str, commit: str | None = FIX_COMMIT) -> None:
    provenance = {'kind': 'merged', 'commit': commit} if commit else {'kind': 'deterministic-gate'}
    lane.client.set_status(task_id, 'done', done_provenance=provenance)


@pytest.mark.asyncio
async def test_scenario_16_a_done_task_is_a_stale_head_until_the_fix_is_contained_then_refiled(tmp_path):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    _done(lane, task_id)
    lane.ancestry.answers[(FIX_COMMIT, HEAD2)] = False
    before = lane.episode()

    stale = await lane.spaced_red(head=HEAD2)
    after_stale = lane.episode()
    restarted = await lane.spaced_red(head=HEAD3)
    refiled = await lane.spaced_red(head=HEAD3)

    assert stale.stale_heads == (A,)
    assert replace(after_stale, last_refreshed_at=before.last_refreshed_at) == before
    assert restarted.episodes_started == (A,)
    assert refiled.tasks_filed == ('7002',)
    assert lane.lane_metadata()['prior_task_ids'] == [task_id]
    assert (FIX_COMMIT, HEAD3) in lane.ancestry.calls


@pytest.mark.asyncio
async def test_scenario_17_a_done_task_with_no_fix_commit_is_contained(tmp_path):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    _done(lane, task_id, commit=None)

    effect = await lane.spaced_red()

    assert effect.episodes_started == (A,)
    assert lane.episode().prior_task_ids == (task_id,)
    assert lane.ancestry.calls == []


@pytest.mark.asyncio
async def test_scenario_17_a_failed_ancestry_read_is_no_decision_logged_loudly(tmp_path, caplog):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    _done(lane, task_id)
    lane.ancestry.answers[(FIX_COMMIT, HEAD1)] = None
    before = lane.episode()

    with caplog.at_level(logging.ERROR):
        effect = await lane.spaced_red()

    assert effect.no_decision == (A,)
    assert lane.episode().counted_passes == before.counted_passes
    assert any(r.levelno >= logging.ERROR and task_id in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_scenario_18_a_failed_filing_stays_due_and_the_tick_files_it_without_a_rerun(tmp_path):
    lane = Lane(tmp_path)
    lane.client.fail_submissions('task store unreachable', 'task store unreachable')
    await lane.red()
    effect = await lane.spaced_red()
    due = lane.episode()
    lane.clock.advance(minutes=1)
    still_down = await lane.tick()
    lane.clock.advance(minutes=1)

    reachable = await lane.tick()

    assert effect.tasks_filed == ()
    assert (due.filing_due, due.covering_task_id, due.filing_failure) == (True, None, 'task store unreachable')
    assert still_down == TickResult((), ())
    assert reachable == TickResult((), ('7001',))
    assert len(lane.client.submitted) == 3
    assert lane.episode().filing_due is False


@pytest.mark.asyncio
async def test_scenario_19_a_restart_between_submit_and_commit_commits_without_refiling(tmp_path):
    lane = Lane(tmp_path)
    lane.client.commit_failures = 1
    await lane.red()
    await lane.spaced_red()
    stranded = lane.episode()
    lane.restarted()

    await lane.spaced_red()

    assert (stranded.covering_task_id, stranded.planning_committed, stranded.filing_due) == ('7001', False, True)
    assert 'still deferred' in (stranded.filing_failure or '')
    assert len(lane.client.submitted) == 1
    assert lane.client.tasks['7001']['status'] == 'pending'
    assert (lane.episode().planning_committed, lane.episode().filing_due) == (True, False)


@pytest.mark.asyncio
async def test_scenario_19_a_deferred_unfinished_task_read_on_a_red_pass_is_completed_not_refiled(tmp_path):
    lane = Lane(tmp_path)
    lane.client.commit_failures = 2
    await lane.red()
    await lane.spaced_red()
    del lane.log[:]

    await lane.spaced_red()

    assert lane.log.count('commit_planning') == 2
    assert 'submit_task' not in lane.log
    assert (lane.episode().planning_committed, lane.episode().filing_due) == (True, False)


@pytest.mark.asyncio
async def test_scenario_20_one_recheck_per_head_for_held_tests_counted_and_filed_when_red(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    lane.clock.advance(minutes=29)
    early = await lane.tick()
    lane.clock.advance(minutes=1)

    due = await lane.tick()
    again = await lane.tick()
    effect = await lane.red((A, B))
    lane.clock.advance(minutes=31)
    after = await lane.tick()

    assert early.commands_due == ()
    assert due.commands_due == (COMMAND,)
    assert again.commands_due == ()
    assert effect.tests_counted == (A, B)
    assert effect.tasks_filed == ('7001',)
    assert after.commands_due == ()


@pytest.mark.asyncio
async def test_scenario_20_no_recheck_when_a_pass_started_after_it_fell_due(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=31)
    await lane.red(reproduced=(B,), inconclusive=(A,))
    lane.clock.advance(minutes=1)

    assert (await lane.tick()).commands_due == ()


@pytest.mark.asyncio
async def test_scenario_20_a_never_extended_episode_expires_after_red_unlanded_hours(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    lane.clock.advance(minutes=30)
    assert (await lane.tick()).commands_due == (COMMAND,)
    await lane.red(reproduced=(B,), inconclusive=(A,))
    lane.clock.advance(hours=24)
    assert (await lane.tick(head=HEAD2)).commands_due == ()

    effect = await lane.red(reproduced=(B,), head=HEAD2)

    assert (A, 'expired') in [(e.node, e.reason) for e in effect.episodes_ended]


@pytest.mark.asyncio
async def test_scenario_20_a_legacy_leg_is_never_returned_for_a_recheck(tmp_path):
    lane = Lane(tmp_path)
    await lane.red(command=LEGACY_NUMERIC_COMMAND, leg='numeric')
    lane.clock.advance(hours=1)

    assert (await lane.tick()).commands_due == ()
    assert lane.stored(LEGACY_NUMERIC_COMMAND).episodes[A].live


@pytest.mark.asyncio
async def test_a_legacy_leg_filing_reproduces_with_the_leg_script(tmp_path):
    lane = Lane(tmp_path)
    await lane.red(command=LEGACY_NUMERIC_COMMAND, leg='numeric')
    lane.clock.advance(minutes=30)

    await lane.red(command=LEGACY_NUMERIC_COMMAND, leg='numeric')

    (arguments,) = lane.client.submitted
    assert 'Reproduce: scripts/run-offline-deep.sh' in arguments['description']
    assert arguments['priority'] == 'high'


@pytest.mark.asyncio
async def test_scenario_21_an_unreadable_store_with_an_open_task_costs_at_most_one_duplicate(tmp_path, caplog):
    lane = Lane(tmp_path)
    await _filed(lane)
    path = tmp_path / 'data' / 'orchestrator' / 'guards' / EPISODES_FILE
    path.write_text('{ not json')

    with caplog.at_level(logging.WARNING):
        lane.restarted()
    for _ in range(4):
        await lane.spaced_red()

    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert len(lane.client.submitted) == 2


@pytest.mark.asyncio
async def test_the_six_filing_writes_happen_in_order(tmp_path, monkeypatch):
    lane = Lane(tmp_path)
    await lane.red()
    writes = _WriteCounter(monkeypatch, lane.log)
    del lane.log[:]

    await lane.spaced_red()

    assert lane.log == [
        'write', 'write', 'submit_task', 'write', 'commit_planning', 'get_task', 'write', 'write',
    ]
    states = [
        (e['filing_due'], e['covering_task_id'], e['planning_committed'])
        for e in (writes.episode(i) for i in range(len(writes.payloads)))
    ]
    assert states == [
        (False, None, False),
        (True, None, False),
        (True, '7001', False),
        (True, '7001', True),
        (False, '7001', True),
    ]


@pytest.mark.asyncio
async def test_a_pass_that_files_nothing_is_one_store_write(tmp_path, monkeypatch):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    writes = _WriteCounter(monkeypatch)

    await lane.spaced_red(A, minutes=3)

    assert len(writes.payloads) == 1


@pytest.mark.asyncio
async def test_a_submission_combined_into_an_open_task_is_covered_without_commit_planning(tmp_path):
    lane = Lane(tmp_path)
    lane.client.set_status('5555', 'in-progress')
    lane.client.outcomes = [SubmittedTask('5555', 'in-progress', combined=True)]

    await lane.red()
    effect = await lane.spaced_red()

    assert effect.tasks_filed == ('5555',)
    assert 'commit_planning' not in lane.log
    episode = lane.episode()
    assert (episode.covering_task_id, episode.planning_committed, episode.filing_due) == ('5555', True, False)


@pytest.mark.asyncio
async def test_a_submission_combined_into_a_deferred_task_commits_it_like_a_fresh_one(tmp_path):
    lane = Lane(tmp_path)
    lane.client.set_status('5556', 'deferred')
    lane.client.outcomes = [SubmittedTask('5556', 'deferred', combined=True)]

    await lane.red()
    effect = await lane.spaced_red()

    assert effect.tasks_filed == ('5556',)
    assert 'commit_planning' in lane.log
    assert lane.client.tasks['5556']['status'] == 'pending'
    assert lane.episode().planning_committed


@pytest.mark.asyncio
async def test_a_submission_combined_into_a_terminal_task_is_swallowed_and_treated_as_cancelled(tmp_path, caplog):
    lane = Lane(tmp_path)
    _done(lane, '5557', commit=None)
    lane.client.outcomes = [SubmittedTask('5557', 'done', combined=True)]
    await lane.red()

    with caplog.at_level(logging.ERROR):
        effect = await lane.spaced_red()
    later = await lane.spaced_red()

    assert effect.tasks_filed == ()
    assert any('swallowed' in r.getMessage() for r in caplog.records)
    episode = lane.episode()
    assert (episode.covering_task_id, episode.binding_swallowed, episode.filing_due) == ('5557', True, False)
    assert later.episodes_started == ()
    assert len(episode.counted_passes) == 3
    assert len(lane.client.submitted) == 1


@pytest.mark.parametrize(
    ('outcome', 'reason'),
    [
        (SubmittedTicket('tkt_0001'), 'ticket'),
        (SubmitFailed('planning mode refused'), 'planning mode refused'),
        (RuntimeError('transport closed'), 'transport closed'),
    ],
    ids=['ticket', 'failed', 'exception'],
)
@pytest.mark.asyncio
async def test_a_submission_that_makes_no_task_leaves_the_filing_due(tmp_path, outcome, reason):
    lane = Lane(tmp_path)
    lane.client.outcomes = [outcome]
    await lane.red()

    effect = await lane.spaced_red()

    episode = lane.episode()
    assert effect.tasks_filed == ()
    assert (episode.filing_due, episode.covering_task_id) == (True, None)
    assert reason in (episode.filing_failure or '')


@pytest.mark.asyncio
async def test_with_no_task_client_the_engine_tracks_and_files_nothing(tmp_path):
    lane = Lane(tmp_path, client=False)
    await lane.red()

    effect = await lane.spaced_red()
    tick = await lane.tick()

    assert effect.threshold_reached == (A,)
    assert effect.tasks_filed == ()
    assert tick.tasks_filed == ()
    assert lane.log == []
    assert lane.episode().filing_due is False


@pytest.mark.asyncio
async def test_with_no_escalation_queue_nothing_is_raised_and_nothing_raises(tmp_path):
    lane = Lane(tmp_path, queue=None)
    task_id = await _filed(lane)
    lane.client.fail_submissions(*['down'] * 30)
    await lane.spaced_red(A, B)
    await lane.spaced_red(A, B)
    lane.clock.advance(hours=1)
    await lane.tick()

    for _ in range(10):
        effect = await lane.spaced_red(A, B, minutes=150)
        assert effect.escalations_raised == ()

    assert lane.episode(A).covering_task_id == task_id
    assert lane.episode(B).filing_due
    assert lane.stored().stuck_filing_escalation_id is None


@pytest.mark.asyncio
async def test_a_failed_unlanded_escalation_write_is_retried_at_the_next_qualifying_pass(tmp_path):
    lane = Lane(tmp_path, queue=FailingOnceQueue(tmp_path / 'queue'))
    await _filed(lane)
    raised: list[tuple[str, ...]] = []

    for _ in range(11):
        raised.append((await lane.spaced_red(minutes=150)).escalations_raised)

    assert [bool(r) for r in raised] == [False] * 10 + [True]
    assert len(lane.pending(UNLANDED_RED_CATEGORY)) == 1


@pytest.mark.asyncio
async def test_a_filing_due_for_recheck_secs_raises_one_stuck_record_amended_then_resolved(tmp_path):
    lane = Lane(tmp_path)
    lane.client.fail_submissions(*['server busy'] * 4, 'quota exhausted')
    await lane.red()
    await lane.spaced_red()
    lane.clock.advance(minutes=29)
    await lane.tick()
    before = lane.pending(STUCK_FILING_CATEGORY)
    lane.clock.advance(minutes=1)
    await lane.tick()
    (stuck,) = lane.pending(STUCK_FILING_CATEGORY)
    lane.clock.advance(minutes=1)
    await lane.tick()
    (unchanged,) = lane.pending(STUCK_FILING_CATEGORY)
    lane.clock.advance(minutes=1)
    await lane.tick()
    (amended,) = lane.pending(STUCK_FILING_CATEGORY)
    lane.clock.advance(minutes=1)

    filed = await lane.tick()

    assert before == []
    assert 'server busy' in stuck.summary
    assert (unchanged.id, unchanged.amendments) == (stuck.id, stuck.amendments)
    assert amended.id == stuck.id and 'quota exhausted' in str(amended.amendments)
    assert filed.tasks_filed == ('7001',)
    assert lane.pending(STUCK_FILING_CATEGORY) == []
    assert lane.queue is not None
    assert escalation_state(lane.queue, stuck.id) == 'resolved'
    assert lane.stored().stuck_filing_escalation_id is None


@pytest.mark.asyncio
async def test_an_unreadable_status_batch_refreshes_every_binding_and_decides_nothing_new(tmp_path, caplog):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    lane.client.statuses_error = ConnectionError('task store unreachable')
    lane.client.unreadable.add(task_id)

    with caplog.at_level(logging.ERROR):
        effect = await lane.spaced_red()

    assert effect.no_decision == (A,)
    assert lane.episode().last_refreshed_at == lane.clock()
    assert sum('unreadable' in r.getMessage() for r in caplog.records) >= 2


@pytest.mark.asyncio
async def test_an_ended_episode_bound_to_an_unreadable_task_is_still_refreshed(tmp_path):
    lane = Lane(tmp_path)
    await _filed(lane)
    lane.clock.advance(minutes=30)
    await lane.green()
    ended_at = lane.episode().ended_at
    lane.client.statuses_error = ConnectionError('task store unreachable')
    lane.clock.advance(days=3)

    await lane.green()

    assert lane.episode().ended_at == ended_at
    assert lane.episode().last_refreshed_at == lane.clock()
