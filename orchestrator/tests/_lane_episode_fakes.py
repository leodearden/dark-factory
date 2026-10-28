"""Fakes and factories shared by the offline lane's episode tests.

The injected clock, the fake task client and ancestry read the engine runs over, the
observer of the episode file's writes, the builders of run records, observations,
episodes and command records, and :class:`Lane`, the engine harness that writes real
run records and observes them through a real ``LaneEngine``.  Imported by bare module
name (``from _lane_episode_fakes import ...``) by ``test_offline_lane_episode_state.py``
and ``test_offline_lane_episodes.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from escalation.queue import EscalationQueue
from shared.psi import PsiSample
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask

from orchestrator.config import LaneCommand, OrchestratorConfig
from orchestrator.flake_debt_drain import DrainPassReport, run_drain_pass
from orchestrator.flake_ledger import ledger_db_path
from orchestrator.guard_state import guard_path
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
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
)
from orchestrator.offline_lane_episodes import LaneEngine, TickResult
from orchestrator.offline_lane_escalations import sentinel_task_id
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import (
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
HEALTHY_PSI = PsiSample(cpu_some10=7.5, mem_some10=0, mem_full10=0, io_some10=0, read_ok=True)


class Clock:
    """An injected clock; each reader of it is first shown to every observer."""

    def __init__(self, start: datetime = T0) -> None:
        self.at = start
        self.observers: list[Callable[[], None]] = []

    def __call__(self) -> datetime:
        for observe in self.observers:
            observe()
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
    outcome: Confirmed | Unconfirmable | Outage | None = None,
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


def _episodes_path(tmp_path: Path) -> Path:
    path = guard_path(tmp_path, EPISODES_FILE)
    assert path is not None
    return path


class StoreWrites:
    """Each write of the episode file, seen as a change of the file's identity, with the record it left.

    :meth:`sample` runs whenever the code under test reads the clock or calls the task
    client, which every store write is preceded by; call it once more after the last.
    """

    def __init__(self, tmp_path: Path, clock: Clock, log: list[str] | None = None) -> None:
        self._tmp_path = tmp_path
        self._path = _episodes_path(tmp_path)
        self._clock = clock
        self._log = log
        self._seen = self._identity()
        self.records: list[CommandRecord] = []

    def _identity(self) -> tuple[int, int] | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_mtime_ns

    def sample(self) -> None:
        identity = self._identity()
        if identity == self._seen:
            return
        self._seen = identity
        self.records.append(EpisodeStore(self._tmp_path, clock=Clock(self._clock.at)).load(COMMAND))
        if self._log is not None:
            self._log.append('write')


def _id_order(task_id: str) -> tuple[bool, int, str]:
    return not task_id.isdigit(), int(task_id) if task_id.isdigit() else 0, task_id


def _holds(stored: object, value: str) -> bool:
    return stored == value or (isinstance(stored, list) and value in stored)


class FakeLedgerClient:
    """The task seam as ``chronic_flake.py::SchedulerChronicFlakeTaskClient`` behaves: a failed
    ``commit_planning`` is only logged, so the caller learns of it from a live read."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.tasks: dict[str, dict[str, Any]] = {}
        self.outcomes: list[SubmitOutcome | Exception] = []
        self.submitted: list[dict[str, Any]] = []
        self.statuses_error: Exception | None = None
        self.lookup_error: Exception | None = None
        self.unreadable: set[str] = set()
        self.commit_failures = 0
        self.observers: list[Callable[[], None]] = []
        self._next_id = 7000

    def _called(self, name: str) -> None:
        for observe in self.observers:
            observe()
        self.log.append(name)

    def fail_submissions(self, *reasons: str) -> None:
        self.outcomes = [SubmitFailed(reason) for reason in reasons]

    def set_status(self, task_id: str, status: str, **metadata: Any) -> None:
        self.tasks[task_id] = {'id': task_id, 'status': status, 'metadata': metadata}

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        self._called('submit_task')
        self.submitted.append(arguments)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self._next_id += 1
        task_id = str(self._next_id)
        self.set_status(task_id, 'deferred', **arguments.get('metadata', {}))
        return SubmittedTask(task_id, 'deferred')

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        self._called('get_statuses')
        if self.statuses_error is not None:
            return {}, self.statuses_error
        return {i: self.tasks[i]['status'] for i in ids if i in self.tasks}, None

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        self._called('get_task')
        if task_id in self.unreadable:
            return None, ConnectionError('task store unreachable')
        return self.tasks.get(task_id), None

    async def find_tasks_by_metadata(
        self, key: str, value: str, statuses: Sequence[str],
    ) -> tuple[list[dict], Exception | None]:
        """The rows the tool answers with, in its task-id order, with the wire's int ids."""
        self._called('find_tasks_by_metadata')
        if self.lookup_error is not None:
            return [], self.lookup_error
        return [
            {'task_id': int(task_id) if task_id.isdigit() else task_id, 'status': task['status']}
            for task_id, task in sorted(self.tasks.items(), key=lambda item: _id_order(item[0]))
            if task['status'] in statuses and _holds(task['metadata'].get(key), value)
        ], None

    async def commit_planning(self, task_ids: list[str]) -> None:
        self._called('commit_planning')
        if self.commit_failures:
            self.commit_failures -= 1
            logger.warning('commit_planning failed for %s; they stay deferred', task_ids)
            return
        for task_id in task_ids:
            if self.tasks.get(task_id, {}).get('status') == 'deferred':
                self.tasks[task_id]['status'] = 'pending'


class FakeAncestry:
    def __init__(self, *, deep_worktree: Path = Path('/nonexistent/_offline-deep')) -> None:
        self.persistent_offline_deep_worktree_path = deep_worktree
        self.answers: dict[tuple[str, str], bool | None] = {}
        self.calls: list[tuple[str, str]] = []
        self.failure: Exception | None = None

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None:
        self.calls.append((ancestor, descendant))
        if self.failure is not None:
            raise self.failure
        return self.answers.get((ancestor, descendant), True)


def _reader(psi: PsiSample | Callable[[], PsiSample]) -> Callable[[], PsiSample]:
    if isinstance(psi, PsiSample):
        return lambda: psi
    return psi


class Lane:
    """A lane command's passes, written as real run records and observed by a real engine.

    *psi* is the host pressure the engine reads: a sample, or a reader passed through as is.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        client: bool = True,
        queue: EscalationQueue | None | bool = True,
        psi: PsiSample | Callable[[], PsiSample] = HEALTHY_PSI,
    ) -> None:
        self.tmp_path = tmp_path
        self.log: list[str] = []
        self.clock = Clock()
        self.client = FakeLedgerClient(self.log)
        self.ancestry = FakeAncestry()
        self.queue = EscalationQueue(tmp_path / 'queue') if queue is True else (queue or None)
        self.records = RunRecordStore(tmp_path, clock=self.clock)
        self.config = OrchestratorConfig(project_root=tmp_path)
        self._with_client = client
        self._read_psi = _reader(psi)
        self.engine = self.restarted()

    def restarted(self) -> LaneEngine:
        self.engine = LaneEngine(
            self.config,
            self.ancestry,
            task_client=self.client if self._with_client else None,
            escalation_queue=self.queue,
            now=self.clock,
            read_psi=self._read_psi,
        )
        return self.engine

    @property
    def ledger(self) -> Path:
        return ledger_db_path(self.tmp_path)

    @property
    def project_id(self) -> str:
        return self.config.fused_memory.project_id

    async def drain(self) -> DrainPassReport:
        """One pass of the ledger's drain, on its own wall clock: the due list is stamped with the wall clock."""
        return await run_drain_pass(
            self.ledger, self.project_id, task_client=self.client, git=self.ancestry, escalation_queue=self.queue,
        )

    async def pass_with(
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
        observation = self.written_pass(head, outcome, detect, confirm, notes, command=command, leg=leg)
        return await self.engine.observe(observation)

    def written_pass(
        self,
        head: str,
        outcome: Confirmed | Unconfirmable | None,
        detect: RunRecord,
        confirm: RunRecord | None,
        notes: tuple[EvidenceNote, ...] = (),
        *,
        command: str = COMMAND,
        leg: str = 'generic',
    ) -> PassObservation:
        """A pass written as run records, and the observation the engine is told of it."""
        lane_pass = self.records.allocate_pass(command, leg, head)
        detect = replace(detect, command=command, leg=leg)
        self.records.write_run(lane_pass, detect)
        if confirm is not None:
            self.records.write_run(lane_pass, replace(confirm, command=command, leg=leg))
        return PassObservation(
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
        return await self.pass_with(head, outcome, detect, confirm, command=command, leg=leg)

    async def green(self, *, head: str = HEAD1) -> PassEffect:
        return await self.pass_with(head, None, _run('detect', head, self.clock()), None)

    async def unconfirmable(self, *, head: str = HEAD1, notes: tuple[EvidenceNote, ...] = ()) -> PassEffect:
        at = self.clock()
        outcome = Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')
        detect = _run('detect', head, at, failing=(A,))
        return await self.pass_with(head, outcome, detect, _run('confirm', head, at, returncode=2), notes)

    async def spaced_red(self, *nodes: str, minutes: float = 30, **fields: Any) -> PassEffect:
        self.clock.advance(minutes=minutes)
        return await self.red(nodes or (A,), **fields)

    def unobserved(
        self, head: str, *, failing: tuple[str, ...] = (), outcomes: dict[str, str] | None = None,
    ) -> None:
        """A pass as a pre-cutover worker ran it: its run records are written and the engine is never told."""
        lane_pass = self.records.allocate_pass(COMMAND, 'generic', head)
        self.records.write_run(lane_pass, _run('detect', head, self.clock(), failing=failing, outcomes=outcomes))

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
