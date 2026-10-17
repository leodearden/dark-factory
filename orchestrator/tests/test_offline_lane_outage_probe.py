"""The offline lane's outage probe: PRD ``plans/offline-lane-red-path-prd.md`` D7, section 5.10, scenarios 35-38.

``run_pass`` and a real ``LaneEngine`` over a real escalation queue and run-record store,
with a scripted probe runner, or the real ``run_probe`` over ``sh`` probes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD1,
    LANE_COMMAND,
    A,
    Clock,
    FakeAncestry,
    FakeLedgerClient,
    _run,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp import Client
from shared.lane_probe import OUTAGE_EXIT_STATUS

from orchestrator.config import GitConfig, LaneCommand, OrchestratorConfig
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    ConfirmTarget,
    PassOutcome,
    Unconfirmable,
)
from orchestrator.offline_lane_episode_state import CommandRecord, EpisodeStore
from orchestrator.offline_lane_episodes import LaneEngine, TickResult
from orchestrator.offline_lane_escalations import (
    LANE_ESCALATION_ROLE,
    UnlandedRed,
    file_unlanded_red,
    sentinel_task_id,
)
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_pass import LaneLeg, run_pass
from orchestrator.offline_lane_records import (
    FIRST_LINE_MAX_CHARS,
    ProbeKind,
    ProbeResult,
    RunRecord,
    RunRecordStore,
)
from orchestrator.offline_lane_runs import ConfirmRunner, ProbeSeam, probe_result, run_probe

PROBED = LANE_COMMAND.model_copy(update={'outage_probe': 'python probe.py'})
OUTAGE_LINE = 'qdrant: get collections: ConnectError'
SENTINEL = sentinel_task_id(COMMAND)
WATCHER = 'orchestrator-escalation-watcher-auto'
ROOT_CAUSE = 'offline lane outage: qdrant unreachable'


def _outage(duration: float = 1.5) -> ProbeResult:
    return ProbeResult(ProbeKind.OUTAGE, OUTAGE_EXIT_STATUS, OUTAGE_LINE, OUTAGE_LINE, duration)


def _up() -> ProbeResult:
    return ProbeResult(ProbeKind.UP, 0, None, '', 0.5)


class FakeProbeRunner:
    """Answers each call with the next scripted answer, logging ``(command, worktree)``."""

    def __init__(self) -> None:
        self.answers: list[ProbeResult] = []
        self.calls: list[tuple[str, Path]] = []

    async def __call__(self, command: LaneCommand, worktree: Path) -> ProbeResult:
        self.calls.append((command.name, worktree))
        return self.answers.pop(0)


@dataclass
class FakeConfirm:
    store: RunRecordStore
    calls: list[ConfirmTarget] = field(default_factory=list)

    async def __call__(
        self, target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner,
    ) -> Confirmed | Unconfirmable:
        self.calls.append(target)
        lane_pass = self.store.current_pass(COMMAND)
        assert lane_pass is not None
        self.store.write_run(lane_pass, _run('confirm', lane_pass.head, self.store.now(), failing=(A,)))
        return Confirmed(ConfirmMode.NARROWED, (A,), (), ())


@dataclass
class RecordingHook:
    calls: list[PassOutcome] = field(default_factory=list)

    async def __call__(
        self, command: str, head: str, outcome: PassOutcome, run_records: Sequence[RunRecord],
    ) -> Sequence[EvidenceNote]:
        self.calls.append(outcome)
        return ()


class SubmitFailsOnce(EscalationQueue):
    def __init__(self, queue_dir: Path) -> None:
        super().__init__(queue_dir)
        self.failures = 1

    def submit(self, escalation: Escalation) -> str:
        if self.failures:
            self.failures -= 1
            raise OSError('queue directory unwritable')
        return super().submit(escalation)


class Lane:
    """One probe-declaring command's lane: its passes, its ticks, and what they left behind."""

    def __init__(self, tmp_path: Path, command: LaneCommand = PROBED, *, queue: EscalationQueue | None = None) -> None:
        self.tmp_path = tmp_path
        self.clock = Clock()
        self.deep = tmp_path / '_offline-deep'
        (self.deep / command.cwd).mkdir(parents=True)
        self.queue = queue or EscalationQueue(tmp_path / 'queue')
        self.tasks = FakeLedgerClient([])
        self.probes = FakeProbeRunner()
        self.store = RunRecordStore(tmp_path, clock=self.clock)
        self.confirm = FakeConfirm(self.store)
        self.hook = RecordingHook()
        self.config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(offline_lane_commands=[command]))
        self.engine = LaneEngine(
            self.config, FakeAncestry(deep_worktree=self.deep), task_client=self.tasks,
            escalation_queue=self.queue, now=self.clock, probe_runner=self.probes,
        )

    def reconfigure(self, *commands: LaneCommand) -> None:
        self.config.git = GitConfig(offline_lane_commands=list(commands))

    async def red_pass(self, *answers: ProbeResult, probe_runner: ProbeSeam | None = None) -> str:
        """One red pass whose probe answers *answers* (or *probe_runner* does); its pass id."""
        self.probes.answers.extend(answers)
        (command,) = self.config.git.offline_lane_commands

        async def detect() -> tuple[int, str]:
            lane_pass = self.store.current_pass(COMMAND)
            assert lane_pass is not None
            self.store.write_run(lane_pass, _run('detect', lane_pass.head, self.store.now(), failing=(A,)))
            return 1, ''

        await run_pass(
            LaneLeg.generic(command), worktree=self.deep, head=HEAD1, detect=detect, confirm=self.confirm,
            store=self.store, engine=self.engine, hooks=(self.hook,), probe_runner=probe_runner or self.probes,
        )
        lane_pass = self.store.current_pass(COMMAND)
        assert lane_pass is not None
        self.clock.advance(minutes=30)
        return lane_pass.pass_id

    async def tick(self, *answers: ProbeResult) -> TickResult:
        self.probes.answers.extend(answers)
        return await self.engine.tick(head=HEAD1)

    def record(self) -> CommandRecord:
        return EpisodeStore(self.tmp_path, clock=self.clock).load(COMMAND)

    def pending(self) -> list[Escalation]:
        return [e for e in self.queue.get_pending() if e.task_id == SENTINEL]

    def escalation(self, escalation_id: str | None) -> Escalation:
        assert escalation_id is not None
        found = self.queue.get(escalation_id)
        assert found is not None
        return found

    def pass_json(self, pass_id: str) -> dict:
        return json.loads((self.store.pass_directory(COMMAND, pass_id) / 'record.json').read_text())


async def _in_outage(lane: Lane) -> tuple[str, str]:
    """A held episode from a confirmed pass, then one outage pass; the two pass ids, with the probe log cleared."""
    seed = await lane.red_pass(_up())
    first = await lane.red_pass(_outage())
    lane.probes.calls.clear()
    return seed, first


async def _promoted(queue: EscalationQueue, *member_ids: str) -> dict:
    async with Client(create_server(queue, startup_sweep=False)) as client:
        result = await client.call_tool('promote_to_l2', {
            'task_id': SENTINEL,
            'agent_role': WATCHER,
            'member_ids': list(member_ids),
            'root_cause': ROOT_CAUSE,
            'evidence': 'the lane filed an infra_issue outage record',
            'options': ['wait for the lane to close it'],
            'summary': 'qdrant is down for the offline lane',
        })
    return result.data


# --- scenario 35: an outage pass --------------------------------------------------


@pytest.mark.asyncio
async def test_two_outage_passes_skip_the_confirm_mark_the_held_episode_and_keep_one_unamended_record(tmp_path):
    lane = Lane(tmp_path)
    seed = await lane.red_pass(_up())
    held = lane.record().episodes[A]

    first = await lane.red_pass(_outage(1.5))
    second = await lane.red_pass(_outage(2.5))

    record = lane.record()
    episode = record.episodes[A]
    assert (len(lane.confirm.calls), len(lane.hook.calls)) == (1, 1)
    for pass_id in (first, second):
        block = lane.pass_json(pass_id)
        assert block['outcome']['arm'] == 'outage'
        assert block['detect']['probe']['failed_dependency'] == OUTAGE_LINE
    assert episode.had_outage_pass is True
    assert (episode.counted_passes, episode.last_counted_started_at, episode.ended_at) == (
        held.counted_passes, held.last_counted_started_at, None,
    )
    assert record.last_pass_id == seed
    assert lane.tasks.submitted == []
    (outage,) = lane.pending()
    assert (outage.category, outage.level, outage.agent_role) == ('infra_issue', 1, LANE_ESCALATION_ROLE)
    assert OUTAGE_LINE in outage.detail
    assert str(lane.store.pass_directory(COMMAND, first)) in outage.detail
    assert record.outage_escalation_id == outage.id
    assert outage.amendments == []
    assert record.in_outage


@pytest.mark.asyncio
async def test_an_outage_pass_effect_names_only_the_record_it_opened(tmp_path):
    lane = Lane(tmp_path)
    _, first = await _in_outage(lane)

    effect = lane.pass_json(first)['effect']

    assert effect['escalations_raised'] == [lane.record().outage_escalation_id]
    assert [key for key, value in effect.items() if value] == ['escalations_raised']


# --- scenario 36: the probe loop on the tick ----------------------------------------


@pytest.mark.asyncio
async def test_the_tick_reprobes_alone_every_recheck_and_an_up_closes_the_record_and_its_level_2(tmp_path):
    lane = Lane(tmp_path)
    await _in_outage(lane)
    l1 = lane.record().outage_escalation_id
    assert l1 is not None
    l2 = (await _promoted(lane.queue, l1))['id']

    lane.clock.at -= timedelta(seconds=1)
    early = await lane.tick()
    lane.clock.advance(seconds=1)
    still_out = await lane.tick(_outage())
    lane.clock.advance(minutes=30)
    passed = await lane.tick(_up())

    assert early == TickResult((), ())
    assert still_out.commands_due == ()
    assert lane.probes.calls == [(COMMAND, lane.deep), (COMMAND, lane.deep)]
    assert len(lane.confirm.calls) == 1
    assert passed.commands_due == (COMMAND,)
    for escalation_id in (l1, l2):
        closed = lane.escalation(escalation_id)
        assert (closed.status, closed.resolved_by) == ('resolved', LANE_ESCALATION_ROLE)
    record = lane.record()
    assert (record.outage_escalation_id, record.in_outage) == (None, False)


@pytest.mark.asyncio
async def test_a_level_2_holding_another_member_is_amended_not_resolved_when_the_probe_passes(tmp_path):
    lane = Lane(tmp_path)
    await _in_outage(lane)
    l1 = lane.record().outage_escalation_id
    assert l1 is not None
    other = file_unlanded_red(lane.queue, UnlandedRed(COMMAND, '9001', 'pending', (A,), lane.clock(), ()))
    l2 = (await _promoted(lane.queue, l1))['id']
    folded = await _promoted(lane.queue, other)
    assert (folded['id'], folded['members']) == (l2, [l1, other])

    await lane.tick(_up())
    await lane.tick()

    holder = lane.escalation(l2)
    closing = [a for a in holder.amendments if f'closed its record {l1}' in a['detail']]
    assert holder.status == 'pending'
    assert len(closing) == 1
    assert lane.escalation(other).status == 'pending'
    assert lane.escalation(l1).status == 'resolved'


# --- scenario 37: a record resolved by hand -------------------------------------------


@pytest.mark.asyncio
async def test_a_hand_resolved_outage_record_is_replaced_by_the_next_failing_pass_not_by_the_tick(tmp_path):
    lane = Lane(tmp_path)
    await _in_outage(lane)
    old = lane.record().outage_escalation_id
    assert old is not None
    lane.queue.resolve(old, 'resolved by hand while qdrant is still down', resolved_by='leo')

    ticked = await lane.tick(_outage())

    assert ticked.commands_due == ()
    assert lane.pending() == []
    assert lane.record().outage_escalation_id == old

    await lane.red_pass(_outage())

    (new,) = lane.pending()
    assert new.id != old
    assert lane.record().outage_escalation_id == new.id


# --- scenario 38: a broken probe ------------------------------------------------------


BROKEN_PROBES = {
    'another exit code': ('exit 3', 180),
    'exit 75 printing nothing': (f'exit {OUTAGE_EXIT_STATUS}', 180),
    'exit 75 naming the dependency on stderr only': (f"echo 'x: y' >&2; exit {OUTAGE_EXIT_STATUS}", 180),
    'a crash': ('kill -SEGV $$', 180),
    'the timeout': ('sleep 30', 1),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(('probe', 'timeout'), BROKEN_PROBES.values(), ids=list(BROKEN_PROBES))
async def test_a_broken_probe_is_confirmed_as_if_none_were_declared_under_one_broken_record(tmp_path, probe, timeout):
    command = LANE_COMMAND.model_copy(update={'outage_probe': probe, 'outage_probe_timeout_secs': timeout})
    lane = Lane(tmp_path, command)

    await lane.red_pass(probe_runner=run_probe)

    record = lane.record()
    (broken,) = lane.pending()
    assert len(lane.confirm.calls) == 1
    assert record.episodes[A].live
    assert record.outage_escalation_id is None
    assert record.probe_broken_escalation_id == broken.id
    assert (broken.category, broken.level) == ('infra_issue', 1)

    await lane.red_pass(probe_runner=run_probe)

    assert [e.id for e in lane.pending()] == [broken.id]
    assert len(lane.confirm.calls) == 2

    await lane.red_pass(_up())

    assert lane.pending() == []
    assert lane.escalation(broken.id).status == 'resolved'
    assert lane.record().probe_broken_escalation_id is None


@pytest.mark.asyncio
async def test_a_command_with_no_probe_is_confirmed_and_keeps_no_probe_record(tmp_path):
    lane = Lane(tmp_path, LANE_COMMAND)

    await lane.red_pass()

    record = lane.record()
    assert len(lane.confirm.calls) == 1
    assert lane.probes.calls == []
    assert lane.pending() == []
    assert (record.last_probe_result, record.outage_escalation_id, record.probe_broken_escalation_id) == (
        None, None, None,
    )


# --- the two hold exits ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'configured',
    [
        (PROBED.model_copy(update={'outage_probe': None}),),
        (PROBED.model_copy(update={'enabled': False}),),
        (),
    ],
    ids=['probe removed', 'command disabled', 'command removed'],
)
async def test_an_outage_whose_probe_can_no_longer_run_ends_on_the_tick_and_rechecks_resume(tmp_path, configured):
    lane = Lane(tmp_path)
    await _in_outage(lane)
    outage = lane.record().outage_escalation_id
    lane.reconfigure(*configured)

    ticked = await lane.tick()

    record = lane.record()
    assert lane.probes.calls == []
    assert lane.escalation(outage).status == 'resolved'
    assert (record.last_probe_result, record.outage_escalation_id) == (None, None)
    assert ticked.commands_due == (COMMAND,)
    assert record.rechecks == {HEAD1: 1}


@pytest.mark.asyncio
async def test_an_outage_whose_worktree_is_gone_is_handed_to_a_pass_without_probing(tmp_path):
    lane = Lane(tmp_path)
    await _in_outage(lane)
    before = lane.record()
    shutil.rmtree(lane.deep)

    ticked = await lane.tick()

    after = lane.record()
    assert ticked.commands_due == (COMMAND,)
    assert lane.probes.calls == []
    assert after.in_outage
    assert after.last_probe_at == before.last_probe_at


# --- section 5.10: a failed escalation write ------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_outage_record_write_is_retried_by_the_next_tick_probe(tmp_path):
    lane = Lane(tmp_path, queue=SubmitFailsOnce(tmp_path / 'queue'))
    await lane.red_pass(_up())
    first = await lane.red_pass(_outage())

    record = lane.record()
    assert lane.pass_json(first)['outcome']['arm'] == 'outage'
    assert len(lane.confirm.calls) == 1
    assert (record.in_outage, record.probe_record_behind, record.outage_escalation_id) == (True, True, None)
    assert lane.pending() == []

    await lane.tick(_outage())

    (outage,) = lane.pending()
    record = lane.record()
    assert (record.probe_record_behind, record.outage_escalation_id) == (False, outage.id)


# --- run_probe and probe_result -----------------------------------------------------------


PROBE_EXITS = {
    'exit 0 is up': (0, '', ProbeKind.UP, None),
    'exit 0 with output is up': (0, 'all good\n', ProbeKind.UP, None),
    'exit 75 naming a dependency is an outage': (
        OUTAGE_EXIT_STATUS, f'{OUTAGE_LINE}\n', ProbeKind.OUTAGE, OUTAGE_LINE,
    ),
    'lines after the first are ignored': (
        OUTAGE_EXIT_STATUS, f'  {OUTAGE_LINE}  \nmore detail\n', ProbeKind.OUTAGE, OUTAGE_LINE,
    ),
    'exit 75 printing nothing is broken': (OUTAGE_EXIT_STATUS, '', ProbeKind.BROKEN, None),
    'exit 75 with a blank first line is broken': (OUTAGE_EXIT_STATUS, '\nqdrant: down\n', ProbeKind.BROKEN, None),
    'another exit code is broken': (3, f'{OUTAGE_LINE}\n', ProbeKind.BROKEN, None),
    'a signal is broken': (-11, '', ProbeKind.BROKEN, None),
    'no exit status is broken': (None, '', ProbeKind.BROKEN, None),
}


@pytest.mark.parametrize(('returncode', 'stdout', 'kind', 'dependency'), PROBE_EXITS.values(), ids=list(PROBE_EXITS))
def test_probe_result_maps_an_exit_and_its_first_stdout_line(returncode, stdout, kind, dependency):
    result = probe_result(returncode, stdout, 2.0)

    assert (result.result, result.returncode, result.failed_dependency, result.duration) == (
        kind, returncode, dependency, 2.0,
    )


def test_probe_result_caps_the_first_line():
    result = probe_result(OUTAGE_EXIT_STATUS, 'q' * (FIRST_LINE_MAX_CHARS + 50), 1.0)

    assert result.failed_dependency == 'q' * FIRST_LINE_MAX_CHARS
    assert result.first_line == result.failed_dependency


def _probe_command(tmp_path: Path, probe: str, timeout: int = 30) -> LaneCommand:
    (tmp_path / LANE_COMMAND.cwd).mkdir(exist_ok=True)
    return LANE_COMMAND.model_copy(update={'outage_probe': probe, 'outage_probe_timeout_secs': timeout})


@pytest.mark.asyncio
async def test_run_probe_reads_the_dependency_from_stdout_in_the_command_cwd(tmp_path):
    (tmp_path / LANE_COMMAND.cwd).mkdir()
    (tmp_path / LANE_COMMAND.cwd / 'marker').touch()
    probe = f"test -f marker && echo 'qdrant: down' && echo noise >&2; exit {OUTAGE_EXIT_STATUS}"

    result = await run_probe(_probe_command(tmp_path, probe), tmp_path)

    assert (result.result, result.failed_dependency) == (ProbeKind.OUTAGE, 'qdrant: down')


@pytest.mark.asyncio
async def test_run_probe_runs_with_the_lane_role_and_at_normal_priority(tmp_path):
    probe = f'test "$DF_VERIFY_ROLE" = offline && test "$(nice)" = "{os.nice(0)}"'

    result = await run_probe(_probe_command(tmp_path, probe), tmp_path)

    assert (result.result, result.returncode) == (ProbeKind.UP, 0)


@pytest.mark.asyncio
async def test_a_timed_out_probe_is_broken_and_its_whole_process_group_is_gone(tmp_path):
    probe = 'sleep 30 & echo $! > child.pid; sleep 30'

    result = await run_probe(_probe_command(tmp_path, probe, timeout=1), tmp_path)

    child = int((tmp_path / LANE_COMMAND.cwd / 'child.pid').read_text())
    assert (result.result, result.returncode) == (ProbeKind.BROKEN, None)
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f'background child {child} of the timed-out probe is still alive')


@pytest.mark.asyncio
async def test_a_probe_whose_cwd_is_missing_is_broken(tmp_path):
    command = LANE_COMMAND.model_copy(update={'outage_probe': 'true'})

    result = await run_probe(command, tmp_path / 'absent')

    assert (result.result, result.returncode) == (ProbeKind.BROKEN, None)


@pytest.mark.asyncio
async def test_run_probe_refuses_a_command_that_declares_none(tmp_path):
    with pytest.raises(ValueError, match='declares no outage_probe'):
        await run_probe(LANE_COMMAND, tmp_path)
