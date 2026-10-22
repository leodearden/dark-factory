#!/usr/bin/env python3
"""Judge a /review-all run's human gate against the trigger-chain form the task store shows.

Read-only: it reads the run home and the store, prints one line per defect,
and writes nothing.
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import state


class Exit(enum.IntEnum):
    CORRECT = 0
    DEFECTIVE = 1
    UNCHECKED = 2
    NO_GATE = 3


_EXIT_MEANINGS: Mapping[Exit, str] = {
    Exit.CORRECT: 'the human gate is correct',
    Exit.DEFECTIVE: 'the human gate is defective; each defect is one line',
    Exit.UNCHECKED: 'the check could not run (usage, run home, or store)',
    Exit.NO_GATE: 'no human_gate task carries the run',
}


class ChainForm(enum.StrEnum):
    INTERIM = 'interim'


class DefectKind(enum.StrEnum):
    MISSING = 'missing dependency'
    FOREIGN = 'foreign dependency'
    ESCAPES = 'escapes the completion measure'
    EMPTY = 'empty program'


@dataclass(frozen=True)
class Defect:
    """One defect; *row* is the named task as the store holds it, None when absent or unnamed."""

    kind: DefectKind
    task_id: int | None
    reason: str
    row: state.TaskRow | None = None


@dataclass(frozen=True)
class Reads:
    """Everything one judgement reads: the gate, the program rows, and the gate's other deps."""

    run_id: str
    prd_paths: tuple[str, ...]
    gate: state.TaskRow
    program: tuple[state.TaskRow, ...]
    outsiders: Mapping[int, state.TaskRow | None]
    measured_at: str

    def in_program(self, task_id: int) -> bool:
        return any(row.task_id == task_id for row in self.program)

    def row(self, task_id: int) -> state.TaskRow | None:
        for row in self.program:
            if row.task_id == task_id:
                return row
        return self.outsiders.get(task_id)


@dataclass(frozen=True)
class Verdict:
    form: ChainForm
    gate: state.TaskRow
    defects: tuple[Defect, ...]
    measured_at: str


_GATING_PRIORITIES = frozenset({state.Priority.HIGH, state.Priority.CRITICAL})
_NOT_IN_PROGRAM = 'not a program task'


def _is_open(row: state.TaskRow) -> bool:
    return row.status not in state.TERMINAL_STATUSES


def _empty_program(reads: Reads) -> tuple[Defect, ...]:
    if reads.program:
        return ()
    return (Defect(
        DefectKind.EMPTY, None,
        f'no store task carries x_finding_run = {reads.run_id} or an authored stream prd_path '
        f'{list(reads.prd_paths)}',
    ),)


def _interim_foreign(reads: Reads, dep: int) -> Defect:
    row = reads.row(dep)
    if row is None or not reads.in_program(dep):
        return Defect(DefectKind.FOREIGN, dep, _NOT_IN_PROGRAM, row)
    return Defect(
        DefectKind.FOREIGN, dep,
        f'a {row.priority} program task: the interim form depends on high/critical program tasks',
        row,
    )


def judge_interim(reads: Reads) -> tuple[Defect, ...]:
    """§4: the gate depends on each open high/critical program task, else on every open one."""
    gating = tuple(row for row in reads.program if row.priority in _GATING_PRIORITIES)
    expected = gating or reads.program
    rule = ('the interim form depends on each open high/critical program task' if gating else
            'with no high/critical program task, the interim form depends on every open one')
    missing = tuple(
        Defect(DefectKind.MISSING, row.task_id, f'not a dependency of the gate: {rule}', row)
        for row in expected if _is_open(row) and row.task_id not in reads.gate.dependencies
    )
    expected_ids = {row.task_id for row in expected}
    foreign = tuple(
        _interim_foreign(reads, dep) for dep in reads.gate.dependencies if dep not in expected_ids
    )
    return (*_empty_program(reads), *missing, *foreign)


def _load_home(run_id: str) -> tuple[state.Program, tuple[str, ...]]:
    """The run's program, and the prd paths of its authored streams after the ledger."""
    home = state.run_home(run_id)
    try:
        data = json.loads(home.program_json.read_text())
    except ValueError as bad:
        raise state.StateError(f'{home.program_json} is not JSON: {bad}') from None
    program = state.parse_program(data)
    views = state.stream_views(program, state.read_ledger(home))
    return program, tuple(view.prd_path for view in views.values() if view.prd_path)


def _read_store(
    reader: state.StoreReader, run_id: str, prd_paths: tuple[str, ...], gate_read: state.StoreRead,
) -> Reads:
    program_read = reader.program_tasks(run_id, prd_paths)
    gate = gate_read.rows[0]
    program_ids = {row.task_id for row in program_read.rows}
    outsider_reads = {
        dep: reader.task(dep) for dep in gate.dependencies if dep not in program_ids
    }
    return Reads(
        run_id=run_id, prd_paths=prd_paths, gate=gate, program=program_read.rows,
        outsiders={dep: read.rows[0] if read.rows else None for dep, read in outsider_reads.items()},
        measured_at=min(read.measured_at for read in (
            gate_read, program_read, *outsider_reads.values())),
    )


def _subject(defect: Defect) -> str:
    if defect.row is not None:
        row = defect.row
        return f'task {row.task_id} ({row.priority}, {row.status}) {row.title} — '
    if defect.task_id is not None:
        return f'task {defect.task_id} (absent from the store) — '
    return ''


def _report(verdict: Verdict) -> None:
    gate = verdict.gate
    print(f'{verdict.form} form: human gate {gate.task_id} ({gate.status}); '
          f'store read {verdict.measured_at}')
    for defect in verdict.defects:
        print(f'{defect.kind}: {_subject(defect)}{defect.reason}')
    count = len(verdict.defects)
    if count:
        noun = 'defect' if count == 1 else 'defects'
        print(f'gate {gate.task_id}: {count} {noun}; correct them before the human commits the gate')
    else:
        print(f'gate {gate.task_id}: correct')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').partition('\n')[0],
        epilog='exit codes: ' + '; '.join(f'{int(e)} {m}' for e, m in _EXIT_MEANINGS.items()),
    )
    parser.add_argument('--run', required=True, help='the run id whose home and gate to check')
    return parser


def _check(run_id: str) -> Exit:
    program, prd_paths = _load_home(run_id)
    reader = state.open_reader(program)
    gate_read = reader.gate(run_id)
    if not gate_read.rows:
        print(f'no human_gate task carries trigger_chain.run_id = {run_id}')
        return Exit.NO_GATE
    reads = _read_store(reader, run_id, prd_paths, gate_read)
    verdict = Verdict(ChainForm.INTERIM, reads.gate, judge_interim(reads), reads.measured_at)
    _report(verdict)
    return Exit.DEFECTIVE if verdict.defects else Exit.CORRECT


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _check(args.run)
    except (state.StateError, OSError) as failed:
        print(f'gate_check: cannot check: {failed}', file=sys.stderr)
        return Exit.UNCHECKED


if __name__ == '__main__':
    raise SystemExit(main())
