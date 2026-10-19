"""The offline lane's red-path engine: ``LaneEngine.observe`` and ``LaneEngine.tick`` over fakes.

PRD ``plans/offline-lane-red-path-prd.md`` D10-D12, section 5.4 and the engine-level halves
of section 6's scenarios: filing and escalation over a fake task client, a fake ancestry
read, an injected clock and a real escalation queue.  The pure rules are
``test_offline_lane_episode_state.py``'s subject.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _lane_episode_fakes import (
    COMMAND,
    FIX_COMMIT,
    HEAD0,
    HEAD1,
    HEAD2,
    HEAD3,
    LANE_COMMAND,
    A,
    B,
    C,
    Clock,
    FakeAncestry,
    FakeLedgerClient,
    StoreWrites,
    _episodes_path,
    _run,
)
from escalation.queue import EscalationQueue
from shared.task_submit import SubmitFailed, SubmittedTask, SubmittedTicket

from orchestrator.config import OrchestratorConfig
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    Episode,
    EpisodeStore,
    PassEffect,
    PassObservation,
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
    RunRecord,
    RunRecordStore,
)

# --- the engine: observe and tick over fakes ----------------------------------------


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

    async def unconfirmable(self, *, head: str = HEAD1, notes: tuple[EvidenceNote, ...] = ()) -> PassEffect:
        at = self.clock()
        outcome = Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')
        detect = _run('detect', head, at, failing=(A,))
        return await self._observe(head, outcome, detect, _run('confirm', head, at, returncode=2), notes)

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

    def watch_writes(self) -> StoreWrites:
        writes = StoreWrites(self.tmp_path, self.clock, self.log)
        self.clock.observers.append(writes.sample)
        self.client.observers.append(writes.sample)
        return writes


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


@pytest.mark.parametrize('dismiss', [False, True], ids=['resolved', 'dismissed'])
@pytest.mark.asyncio
async def test_scenario_15_a_cancelled_task_escalates_once_and_after_closure_a_new_task_names_it(tmp_path, dismiss):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    lane.client.set_status(task_id, 'cancelled')
    for _ in range(10):
        await lane.spaced_red(minutes=150)
    (record,) = lane.pending(UNLANDED_RED_CATEGORY)
    assert len(lane.client.submitted) == 1
    assert lane.queue is not None
    lane.queue.resolve(record.id, 'seen; refile when it persists', dismiss=dismiss, resolved_by='operator')

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
    _episodes_path(tmp_path).write_text('{ not json')

    with caplog.at_level(logging.WARNING):
        lane.restarted()
    for _ in range(4):
        await lane.spaced_red()

    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert len(lane.client.submitted) == 2


@pytest.mark.asyncio
async def test_the_six_filing_writes_happen_in_order(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    writes = lane.watch_writes()
    del lane.log[:]

    await lane.spaced_red()
    writes.sample()

    assert lane.log == [
        'write', 'write', 'submit_task', 'write', 'commit_planning', 'get_task', 'write', 'write',
    ]
    states = [
        (e.filing_due, e.covering_task_id, e.planning_committed)
        for e in (record.episodes[A] for record in writes.records)
    ]
    assert states == [
        (False, None, False),
        (True, None, False),
        (True, '7001', False),
        (True, '7001', True),
        (False, '7001', True),
    ]


@pytest.mark.asyncio
async def test_a_pass_that_files_nothing_is_one_store_write(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    writes = lane.watch_writes()

    await lane.spaced_red(A, minutes=3)
    writes.sample()

    assert len(writes.records) == 1


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


@pytest.mark.asyncio
async def test_the_engine_reports_the_pass_ids_its_held_episodes_pin(tmp_path):
    lane = Lane(tmp_path)

    await lane.red()

    assert lane.engine.pinned_pass_ids() == frozenset(lane.episode().counted_passes)
    assert len(lane.engine.pinned_pass_ids()) == 1


async def _due_group_of_two(lane: Lane) -> None:
    lane.client.fail_submissions('task store unreachable', 'task store unreachable')
    await lane.red((A, B))
    await lane.spaced_red(A, B)
    assert lane.episode(A).filing_due and lane.episode(B).filing_due


@pytest.mark.asyncio
async def test_a_due_uncovered_filing_whose_episodes_all_ended_is_cancelled(tmp_path, caplog):
    lane = Lane(tmp_path)
    await _due_group_of_two(lane)
    lane.clock.advance(minutes=30)
    await lane.red(reproduced=(), cleared=(A, B))
    lane.clock.advance(minutes=1)

    with caplog.at_level(logging.WARNING):
        tick = await lane.tick()

    assert tick.tasks_filed == ()
    assert len(lane.client.submitted) == 2
    assert (lane.episode(A).filing_due, lane.episode(B).filing_due) == (False, False)
    assert any('filing cancelled: no live episode' in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_due_filing_with_one_live_episode_left_still_files(tmp_path):
    lane = Lane(tmp_path)
    await _due_group_of_two(lane)
    lane.clock.advance(minutes=30)
    await lane.red(reproduced=(B,), cleared=(A,))
    lane.clock.advance(minutes=1)

    tick = await lane.tick()

    assert tick.tasks_filed == ('7001',)
    assert lane.lane_metadata()['nodes'] == [A, B]
    assert lane.episode(B).covering_task_id == '7001'


@pytest.mark.asyncio
async def test_a_command_level_task_after_a_contained_one_names_it_and_carries_the_hook_note(tmp_path):
    lane = Lane(tmp_path)
    note = EvidenceNote('hook', None, 'suite-wide leak')
    await lane.unconfirmable(notes=(note,))
    lane.clock.advance(minutes=30)
    (first,) = (await lane.unconfirmable(notes=(note,))).tasks_filed
    _done(lane, first, commit=None)
    lane.clock.advance(minutes=30)
    await lane.unconfirmable(head=HEAD2, notes=(note,))
    lane.clock.advance(minutes=30)

    (second,) = (await lane.unconfirmable(head=HEAD2, notes=(note,))).tasks_filed

    pseudo = unconfirmable_node(COMMAND)
    details = lane.client.submitted[1]['details']
    assert second != first
    assert lane.lane_metadata(0)['nodes'] == lane.lane_metadata(1)['nodes'] == [pseudo]
    assert lane.lane_metadata(1)['prior_task_ids'] == [first]
    assert f'Prior lane tasks for these tests:\n  {pseudo}: {first}' in details
    assert '  [hook] (all reproduced tests): suite-wide leak' in details


@pytest.mark.asyncio
async def test_a_counted_pass_whose_record_is_unreadable_renders_an_unknown_head_start_and_mode(tmp_path):
    lane = Lane(tmp_path)
    await lane.red()
    (unread,) = lane.episode().counted_passes
    shutil.rmtree(lane.records.pass_directory(COMMAND, unread))

    await lane.spaced_red()

    assert f'  {unread} | unknown | unknown | unknown' in lane.client.submitted[0]['details']


@pytest.mark.asyncio
async def test_the_unlanded_record_names_the_tests_still_red_and_every_test_of_the_task_holds_its_id(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B, C))
    await lane.spaced_red(A, B, C)
    lane.clock.advance(minutes=30)
    await lane.red((A, B), cleared=(C,))

    for _ in range(12):
        await lane.spaced_red(A, B, minutes=150)

    (record,) = lane.pending(UNLANDED_RED_CATEGORY)
    assert {lane.episode(node).escalation_id for node in (A, B, C)} == {record.id}
    assert not lane.episode(C).live
    assert f'  - {A}' in record.detail and f'  - {B}' in record.detail
    assert C not in record.detail


@pytest.mark.asyncio
async def test_one_batched_status_read_per_pass_covers_every_bound_task(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    await lane.spaced_red(A, B)
    await lane.spaced_red(A, B, C)
    await lane.spaced_red(A, B, C)
    assert {lane.episode(node).covering_task_id for node in (A, B, C)} == {'7001', '7002'}
    del lane.log[:]

    await lane.spaced_red(A, B, C)

    assert lane.log.count('get_statuses') == 1


@pytest.mark.asyncio
async def test_tests_sharing_a_covering_task_share_one_read_of_it(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A, B))
    await lane.spaced_red(A, B)
    del lane.log[:]

    await lane.spaced_red(A, B)

    assert lane.log.count('get_task') == 1


@pytest.mark.asyncio
async def test_a_new_episode_takes_its_last_green_head_from_the_previous_pass_record(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((B,), head=HEAD0, outcomes={A: 'passed', B: 'failed'})
    lane.clock.advance(minutes=30)

    await lane.red((A, B))

    assert lane.episode(A).last_green_head == HEAD0
    assert lane.stored().last_green_head is None


@pytest.mark.asyncio
async def test_an_ancestry_read_that_raises_is_no_decision_logged_loudly(tmp_path, caplog):
    lane = Lane(tmp_path)
    task_id = await _filed(lane)
    _done(lane, task_id)
    lane.ancestry.failure = OSError('git exited 128')
    before = lane.episode()

    with caplog.at_level(logging.ERROR):
        effect = await lane.spaced_red()

    assert effect.no_decision == (A,)
    assert lane.episode().counted_passes == before.counted_passes
    assert any(r.levelno >= logging.ERROR and r.exc_info and FIX_COMMIT in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_due_filings_that_fell_due_on_different_passes_file_separate_tasks(tmp_path):
    lane = Lane(tmp_path)
    lane.client.fail_submissions('down', 'down', 'down')
    await lane.red((A,))
    await lane.spaced_red(A, C)
    await lane.spaced_red(A, C)
    lane.clock.advance(minutes=1)

    tick = await lane.tick()

    assert tick.tasks_filed == ('7001', '7002')
    assert [lane.lane_metadata(i)['nodes'] for i in (-2, -1)] == [[A], [C]]


@pytest.mark.asyncio
async def test_a_filing_retried_in_a_pass_is_reported_in_that_pass_effect(tmp_path):
    lane = Lane(tmp_path)
    lane.client.fail_submissions('down')
    await lane.red()
    failed = await lane.spaced_red()

    retried = await lane.spaced_red()

    assert failed.tasks_filed == ()
    assert retried.tasks_filed == ('7001',)


async def _command_level_filed(lane: Lane) -> tuple[str, ...]:
    await lane.unconfirmable()
    lane.clock.advance(minutes=30)
    await lane.unconfirmable()
    return lane.episode(unconfirmable_node(COMMAND)).counted_passes


@pytest.mark.asyncio
async def test_a_command_level_digest_takes_the_opening_pass_mode_from_its_outcome(tmp_path):
    lane = Lane(tmp_path)

    *_earlier, opening = await _command_level_filed(lane)

    row = f'  {opening} | {HEAD1} | {lane.clock().isoformat()} | unconfirmable'
    assert row in lane.client.submitted[0]['details']


@pytest.mark.asyncio
async def test_a_command_level_digest_carries_each_pass_failure_line_whatever_its_test(tmp_path):
    lane = Lane(tmp_path)

    counted = await _command_level_filed(lane)

    details = lane.client.submitted[0]['details']
    for pass_id in counted:
        assert f'  pass {pass_id} detect : AssertionError in {A}' in details


@pytest.mark.asyncio
async def test_the_title_counts_the_distinct_passes_of_its_tests(tmp_path):
    lane = Lane(tmp_path)
    await lane.red((A,))
    await lane.spaced_red(B)

    await lane.spaced_red(A, B)

    (arguments,) = lane.client.submitted
    assert f'offline-lane[{COMMAND}]: 2 test(s) red on 3 passes since ' in arguments['title']


@pytest.mark.asyncio
async def test_no_recheck_for_a_test_already_held_at_the_threshold(tmp_path):
    lane = Lane(tmp_path)
    await _filed(lane)
    lane.clock.advance(minutes=31)

    assert (await lane.tick()).commands_due == ()
