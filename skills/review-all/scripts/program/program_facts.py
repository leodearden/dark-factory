"""What a /review-all program is now: its run home read under one lock, joined with the task store.

The read model every ``program_doc.py`` mode consumes. It reads; it never writes.
"""
from __future__ import annotations

import enum
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
import questions
import run_cli
import state

LABEL_KEY = 'prd_task_label'
ACCEPTANCE_KEY = 'x_acceptance_reason'
CANCELLED = 'cancelled'


@dataclass(frozen=True)
class HomeRead:
    """The run, its rulings, and the raw bytes of the three state files, all read under one lock."""

    run: run_cli.Run
    rulings: tuple[Mapping[str, Any], ...]
    files: Mapping[str, bytes]


def _raw(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b''


def read_home(run_id: str) -> HomeRead:
    """The home of *run_id*; its parsed state and its file bytes describe the same instant."""
    home = state.run_home(run_id)
    with home_writer.locked(home) as snapshot:
        files = {path.name: _raw(path) for path in (home.program_json, home.rulings, home.ledger)}
    program = snapshot.program
    if program.run_id != run_id:
        raise state.StateError(f'{home.program_json} is the program of {program.run_id}, not {run_id}')
    views = state.stream_views(program, snapshot.ledger)
    return HomeRead(
        run=run_cli.Run(home, program, snapshot.ledger, views),
        rulings=snapshot.rulings,
        files=types.MappingProxyType(files),
    )


class ProgramState(enum.StrEnum):
    """A program task's state as the programme sees it; the orchestrator's progress never moves it."""

    FILED = 'filed'
    HELD = 'held'
    RELEASED = 'released'
    CANCELLED_AS_ACCEPTED = 'cancelled-as-accepted'
    CANCELLED_WITHOUT_REASON = 'cancelled-without-reason'


def program_state(row: state.TaskRow) -> ProgramState:
    if state.is_held(row):
        return ProgramState.HELD
    if state.SEAM_RELEASED_KEYS.key in row.metadata:
        return ProgramState.RELEASED
    if row.status == CANCELLED:
        reason = row.metadata.get(ACCEPTANCE_KEY)
        accepted = isinstance(reason, str) and reason.strip()
        return ProgramState.CANCELLED_AS_ACCEPTED if accepted else ProgramState.CANCELLED_WITHOUT_REASON
    return ProgramState.FILED


def task_label(row: state.TaskRow) -> str | None:
    """The PRD's label for the task (``α``), when the task carries one."""
    label = row.metadata.get(LABEL_KEY)
    return label if isinstance(label, str) and label else None


@dataclass(frozen=True)
class StoreFacts:
    """The program's tasks by stream (and the unstreamed bucket), its gates, and when the store was read."""

    by_stream: Mapping[str, tuple[state.TaskRow, ...]]
    gate: state.TaskRow | None
    completion_gate: state.TaskRow | None
    measured_at: str

    @property
    def rows(self) -> tuple[state.TaskRow, ...]:
        return tuple(sorted((r for rows in self.by_stream.values() for r in rows), key=_task_id))

    @property
    def seam_leaves(self) -> tuple[state.TaskRow, ...]:
        """Every program task that waits, or waited, on a seam: held and released leaves alike."""
        return tuple(row for row in self.rows if state.SEAM_WAIT_KEYS.key in row.metadata)

    @property
    def is_empty(self) -> bool:
        return not any(self.by_stream.values())

    def gate_deps(self, stream: str) -> tuple[int, ...]:
        """The ids of *stream*'s tasks the human gate depends on; empty when there is no gate."""
        if self.gate is None:
            return ()
        return tuple(row.task_id for row in self.by_stream[stream] if row.task_id in self.gate.dependencies)


def _task_id(row: state.TaskRow) -> int:
    return row.task_id


def _only(read: state.StoreRead) -> state.TaskRow | None:
    return read.rows[0] if read.rows else None


def read_store(reader: state.StoreReader, home: HomeRead) -> StoreFacts:
    run_id, run = home.run.program.run_id, home.run
    reads = (
        reader.program_tasks(run_id, run.prd_paths),
        reader.gate(run_id),
        reader.completion_gate(run_id),
    )
    tasks, gate, completion_gate = reads
    buckets: dict[str, list[state.TaskRow]] = {
        stream: [] for stream in (*run.program.stream_ids, state.UNSTREAMED)
    }
    for row in sorted(tasks.rows, key=_task_id):
        buckets[state.stream_of(row, run.views)].append(row)
    return StoreFacts(
        by_stream=types.MappingProxyType({stream: tuple(rows) for stream, rows in buckets.items()}),
        gate=_only(gate),
        completion_gate=_only(completion_gate),
        measured_at=min(read.measured_at for read in reads),
    )


@dataclass(frozen=True)
class CurrentRuling:
    """A ruling no later ruling supersedes, and the ids it replaces, transitively, newest first."""

    ruling: Mapping[str, Any]
    superseded: tuple[str, ...]


def current_rulings(rulings: Sequence[Mapping[str, Any]]) -> tuple[CurrentRuling, ...]:
    """Every ruling still in force, in file order; a reversal replaces the ruling it supersedes."""
    by_id = {ruling['id']: ruling for ruling in rulings}
    replaced = {ruling['supersedes'] for ruling in rulings if ruling['supersedes'] is not None}

    def chain(ruling: Mapping[str, Any]) -> tuple[str, ...]:
        ids: list[str] = []
        prior = ruling['supersedes']
        while prior is not None:
            ids.append(prior)
            prior = by_id[prior]['supersedes']
        return tuple(ids)

    return tuple(CurrentRuling(r, chain(r)) for r in rulings if r['id'] not in replaced)


def program_section(home: HomeRead, section: state.ProgramSection) -> tuple[Any, ...]:
    """The section's program.json values, then each ``amend_program`` value for it, in seq order."""
    amended = tuple(
        line['value'] for line in home.run.ledger
        if line['op'] == state.LedgerOp.AMEND_PROGRAM and line['section'] == section
    )
    return (*getattr(home.run.program, section.value), *amended)


@dataclass(frozen=True)
class OpenAction:
    """A lead action the ledger records and no ``action_result`` line has closed yet."""

    id: str
    lead_op: str
    task: int
    target: Any
    text: str
    stream: str


def _stream_of_returns(ledger: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each applied return's stream, from the ``stream_status`` line its application wrote."""
    return {
        line['return']: line['stream'] for line in ledger
        if line['op'] == state.LedgerOp.STREAM_STATUS and 'return' in line
    }


def open_actions(ledger: Sequence[Mapping[str, Any]]) -> tuple[OpenAction, ...]:
    """Every ``action`` line no ``action_result`` names, in seq order, with the stream it came from."""
    closed = {line['id'] for line in ledger if line['op'] == state.LedgerOp.ACTION_RESULT}
    streams = _stream_of_returns(ledger)
    actions = []
    for line in ledger:
        if line['op'] != state.LedgerOp.ACTION or line['id'] in closed:
            continue
        if line['return'] not in streams:
            raise state.StateError(
                f'ledger seq {line["seq"]}: action {line["id"]} came from {line["return"]}, '
                'which no stream_status line names'
            )
        actions.append(OpenAction(
            id=line['id'], lead_op=line['lead_op'], task=line['task'], target=line['target'],
            text=line['text'], stream=streams[line['return']],
        ))
    return tuple(actions)


def pending_questions(home: HomeRead) -> tuple[questions.RecordedQuestion, ...]:
    """The recorded questions still awaiting a ruling, in the order they were first recorded."""
    recorded = questions.recorded_questions(home.run.ledger)
    pending = set(questions.pending_items(recorded, home.rulings))
    return tuple(question for ref, question in recorded.items() if ref in pending)
