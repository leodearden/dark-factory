"""A project root for the offline-lane soak report's tests, written by the lane's REAL writers (task 6867).

Every store is written through its owner: run records and their blocks through
``RunRecordStore``, episode records through ``EpisodeStore``, ledger rows, due items and
debts through ``flake_ledger``, escalations through the lane's filers on a real
``EscalationQueue``, fix tasks through ``build_lane_fix_task_arguments``.  A format change
therefore breaks these tests rather than the soak gate.  The task store is a fake.
"""
from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import _checkout as _checkout
from escalation.queue import EscalationQueue
from orchestrator.config import LaneCommand
from orchestrator.flake_ledger import (
    DebtContext,
    DebtRow,
    DebtTrigger,
    DueItem,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    debt_context_to_json,
    enqueue_debt,
    ensure_schema,
    ledger_db_path,
    open_debt,
    record_flake_occurrence,
)
from orchestrator.offline_lane_confirm import PassOutcome, outcome_to_json
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    EpisodeStore,
    PassEffect,
    PassObservation,
)
from orchestrator.offline_lane_filing import (
    CountedPassEvidence,
    TestEvidence,
    build_lane_fix_task_arguments,
)
from orchestrator.offline_lane_ledger import debt_context, recent_occurrences
from orchestrator.offline_lane_records import (
    FailureRecord,
    LanePass,
    Leg,
    PassRecord,
    Phase,
    ProbeResult,
    RunRecord,
    RunRecordStore,
    classify_exit,
)
from shared.project_layout import escalation_queue_dir
from shared.task_submit import SubmittedTask

PROJECT_ID = 'dark_factory'
T0 = datetime(2026, 10, 20, 9, 0, 0, tzinfo=UTC)
HEAD = 'aaaaaaaaaaaa1111'
OTHER_HEAD = 'bbbbbbbbbbbb2222'
UNIT = 'unit'
LINT = 'lint'
FAILURE_LINE = 'AssertionError: expected 3, got 4'


def failure(node: str, *, kind: str = 'failed', first_line: str = FAILURE_LINE) -> FailureRecord:
    return FailureRecord(node=node, kind=kind, first_line=first_line)


def effect(**fields: Any) -> PassEffect:
    """A pass effect with every field empty but those given."""
    empty: dict[str, Any] = {
        'tests_counted': (), 'episodes_started': (), 'episodes_extended': (), 'episodes_ended': (),
        'threshold_reached': (), 'stale_heads': (), 'no_decision': (), 'tasks_filed': (),
        'escalations_raised': (),
    }
    return PassEffect(**{**empty, **fields})


class Clock:
    def __init__(self, start: datetime = T0) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current


class FakeTaskStore:
    """The ledger's task seam (``flake_ledger.FlakeLedgerTaskClient``) and the soak report's task reader.

    An id told :meth:`answer_absent` reads as a corroborated absence, and one told
    :meth:`fail` as a failed read, by the ``(task, error)`` convention.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.absent: set[str] = set()
        self.failing: set[str] = set()
        self._next_id = 9000

    def answer_absent(self, task_id: str) -> None:
        self.absent.add(task_id)

    def fail(self, task_id: str) -> None:
        self.failing.add(task_id)

    async def submit_task(self, arguments: dict) -> SubmittedTask:
        self._next_id += 1
        task_id = str(self._next_id)
        self.tasks[task_id] = {
            'id': task_id,
            'title': arguments['title'],
            'description': arguments.get('description', ''),
            'details': arguments.get('details', ''),
            'priority': arguments.get('priority'),
            'status': 'deferred',
            'metadata': copy.deepcopy(arguments.get('metadata', {})),
        }
        return SubmittedTask(task_id, 'deferred')

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        return {i: self.tasks[i]['status'] for i in ids if i in self.tasks and i not in self.absent}, None

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        if task_id in self.failing:
            return None, ConnectionError(f'task store unreachable reading {task_id}')
        if task_id in self.absent or task_id not in self.tasks:
            return None, None
        return copy.deepcopy(self.tasks[task_id]), None

    async def commit_planning(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            if self.tasks.get(task_id, {}).get('status') == 'deferred':
                self.tasks[task_id]['status'] = 'pending'


def lane_command(command: str) -> LaneCommand:
    return LaneCommand(name=command, command='pytest tests', cwd='.')


class SoakProject:
    """One temporary project root, written only through the lane's real writers."""

    def __init__(self, root: Path, clock: Clock | None = None) -> None:
        self.root = root
        self.clock = clock or Clock()
        self.records = RunRecordStore(root, clock=self.clock)
        self.tasks = FakeTaskStore()

    @property
    def db(self) -> Path:
        return ledger_db_path(self.root)

    def _run(
        self, lane_pass: LanePass, phase: Phase, failures: Sequence[FailureRecord], probe: ProbeResult | None,
    ) -> RunRecord:
        returncode = 1 if failures else 0
        output = lane_pass.output_path(phase)
        output.write_text(''.join(f'{f.node} {f.kind}: {f.first_line}\n' for f in failures) or 'all passed\n')
        record = RunRecord(
            command=lane_pass.command,
            leg=lane_pass.leg,
            head=lane_pass.head,
            phase=phase.value,
            started_at=lane_pass.started_at,
            ended_at=lane_pass.started_at + timedelta(seconds=30),
            returncode=returncode,
            ended_by=classify_exit(returncode).value,
            output_path=str(output),
            junit_path=None,
            failures=tuple(failures),
            ids_available=True,
            outcomes=None,
            session_complete=True,
            probe=probe,
        )
        self.records.write_run(lane_pass, record)
        return record

    def add_pass(
        self,
        command: str,
        *,
        at: datetime,
        head: str = HEAD,
        detect: Sequence[FailureRecord] | None = (),
        confirm: Sequence[FailureRecord] | None = None,
        outcome: PassOutcome | None = None,
        effect: PassEffect | None = None,
        blocks: bool = True,
        probe: ProbeResult | None = None,
    ) -> LanePass:
        """One pass: *detect* None writes no sub-run (a header only), ``()`` a green one.

        *blocks* False leaves the outcome and effect blocks as the header wrote them,
        which is how an interrupted pass, and a pass written before δ2, look.
        """
        self.clock.current = at
        lane_pass = self.records.allocate_pass(command, Leg.GENERIC, head)
        if detect is not None:
            self._run(lane_pass, Phase.DETECT, detect, probe)
        if confirm is not None:
            self._run(lane_pass, Phase.CONFIRM, confirm, None)
        if blocks:
            self.set_blocks(lane_pass, outcome=outcome, effect=effect)
        return lane_pass

    def set_blocks(self, lane_pass: LanePass, *, outcome: PassOutcome | None, effect: PassEffect | None) -> None:
        """The outcome and effect blocks, as ``offline_lane_pass.run_pass`` writes them at the end of a pass."""
        self.records.set_pass_blocks(
            lane_pass,
            outcome=None if outcome is None else outcome_to_json(outcome),
            effect=None if effect is None else effect.to_json(),
        )

    def pass_record(self, lane_pass: LanePass) -> PassRecord:
        record = self.records.load_pass(lane_pass.command, lane_pass.pass_id)
        assert record is not None, f'{lane_pass.pass_id} did not read back'
        return record

    def save_command(self, record: CommandRecord) -> None:
        EpisodeStore(self.root, clock=self.clock).save(record)

    def provision_ledger(self) -> None:
        ensure_schema(self.db)

    def escalations(self) -> EscalationQueue:
        return EscalationQueue(escalation_queue_dir(self.root))

    def provision(self, *commands: str) -> None:
        """Every store the report needs, empty: the ledger, the queue and an episode record per command."""
        self.provision_ledger()
        self.escalations()
        for command in commands:
            self.save_command(CommandRecord.empty(command))

    def occurrence(
        self,
        lane_pass: LanePass,
        test_ids: Sequence[str],
        verdict: FlakeVerdict,
        *,
        call_site: FlakeCallSite = FlakeCallSite.offline_lane,
    ) -> None:
        suppression = FlakeSuppression(
            verdict=verdict,
            test_ids=tuple(test_ids),
            observed_at=lane_pass.started_at.isoformat(),
            call_site=call_site,
            runner='local',
            psi_cpu_some10=None,
            unconfirmable_reason=None,
        )
        record_flake_occurrence(self.db, PROJECT_ID, suppression, merge_sha=lane_pass.head, task_id=None)

    def _context(self, node: str, trigger: DebtTrigger, current: LanePass, earlier: Sequence[LanePass]) -> DebtContext:
        record = self.pass_record(current)
        observation = PassObservation(
            command=record.command,
            leg=record.leg,
            head=record.head,
            pass_id=record.pass_id,
            started_at=record.started_at,
            detect=record.detect,
            confirm=record.confirm,
            outcome=None,
            fix_task_priority='medium',
            notes=(),
            lane_command=lane_command(record.command),
        )
        occurrences = recent_occurrences(node, [self.pass_record(p) for p in earlier], observation)
        return debt_context(node, trigger, observation, None, occurrences)

    def due(
        self,
        node: str,
        trigger: DebtTrigger,
        current: LanePass,
        *,
        enqueued_at: datetime,
        earlier: Sequence[LanePass] = (),
    ) -> None:
        item = DueItem(
            test_id=node,
            observed_at=current.started_at.isoformat(),
            call_site=FlakeCallSite.offline_lane,
            head=current.head,
            observing_task_id=None,
            context=debt_context_to_json(self._context(node, trigger, current, earlier)),
        )
        enqueue_debt(self.db, PROJECT_ID, [item], now=enqueued_at)

    def open_lane_debt(
        self,
        node: str,
        trigger: DebtTrigger,
        current: LanePass,
        *,
        earlier: Sequence[LanePass] = (),
        now: datetime | None = None,
    ) -> DebtRow:
        """The real ``open_debt`` with a real lane context, filing the de-flake owner into :attr:`tasks`."""
        row = asyncio.run(open_debt(
            self.db,
            PROJECT_ID,
            node,
            task_client=self.tasks,
            now=now or self.clock.current,
            context=self._context(node, trigger, current, earlier),
        ))
        assert row is not None and row.owner_task_id is not None, f'{node}: open_debt filed no owner'
        return row

    def file_fix_task(self, command: str, nodes: Sequence[str], passes: Sequence[LanePass]) -> str:
        """A lane fix task covering *nodes*, rendered by ``build_lane_fix_task_arguments``."""
        counted = tuple(
            CountedPassEvidence(
                pass_id=p.pass_id,
                head=p.head,
                started_at=p.started_at,
                mode='narrowed',
                detect_first_line=FAILURE_LINE,
                confirm_first_line=FAILURE_LINE,
                record_dir=str(p.directory),
            )
            for p in passes
        )
        tests = [
            TestEvidence(
                node=node, kind='failed', passes=counted, first_red_head=passes[0].head,
                last_green_head=None, notes=(), prior_task_ids=(),
            )
            for node in nodes
        ]
        arguments = build_lane_fix_task_arguments(
            command=command, leg=Leg.GENERIC.value, priority='medium', tests=tests, reproduce='pytest tests',
        )
        return asyncio.run(self.tasks.submit_task(arguments)).task_id
