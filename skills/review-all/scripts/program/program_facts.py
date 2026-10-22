"""What a /review-all program is now: its run home read under one lock, joined with the task store.

The read model every ``program_doc.py`` mode consumes. It reads; it never writes.
"""
from __future__ import annotations

import enum
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
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
    def is_empty(self) -> bool:
        return not any(self.by_stream.values())

    def gate_deps(self, stream: str) -> tuple[int, ...]:
        """The ids of *stream*'s tasks the human gate depends on; empty when there is no gate."""
        if self.gate is None:
            return ()
        return tuple(row.task_id for row in self.by_stream[stream] if row.task_id in self.gate.dependencies)


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
    for row in sorted(tasks.rows, key=lambda row: row.task_id):
        buckets[state.stream_of(row, run.views)].append(row)
    return StoreFacts(
        by_stream=types.MappingProxyType({stream: tuple(rows) for stream, rows in buckets.items()}),
        gate=_only(gate),
        completion_gate=_only(completion_gate),
        measured_at=min(read.measured_at for read in reads),
    )
