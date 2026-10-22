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
from collections.abc import Callable, Mapping, Sequence
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
    Exit.DEFECTIVE: 'the human gate is defective (one line per defect)',
    Exit.UNCHECKED: 'the check could not run (usage, run home, or store)',
    Exit.NO_GATE: 'no human_gate task carries the run',
}


class ChainForm(enum.StrEnum):
    INTERIM = 'interim'
    FINAL = 'final'


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
    """Everything one judgement reads: both gates, the program rows, and the gate's other deps."""

    run_id: str
    prd_paths: tuple[str, ...]
    gate: state.TaskRow
    completion_gate: state.TaskRow | None
    program: tuple[state.TaskRow, ...]
    outsiders: Mapping[int, state.TaskRow | None]
    measured_at: str

    @property
    def form(self) -> ChainForm:
        """§3 when a completion_gate sibling is in the store, else §4."""
        return ChainForm.INTERIM if self.completion_gate is None else ChainForm.FINAL

    def in_program(self, task_id: int) -> bool:
        return any(row.task_id == task_id for row in self.program)

    def row(self, task_id: int) -> state.TaskRow | None:
        if self.completion_gate is not None and self.completion_gate.task_id == task_id:
            return self.completion_gate
        for row in self.program:
            if row.task_id == task_id:
                return row
        return self.outsiders.get(task_id)


@dataclass(frozen=True)
class Verdict:
    form: ChainForm
    gate: state.TaskRow
    completion_gate: state.TaskRow | None
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


_FINAL_RULE = "the final form's human gate depends on its completion gate only"


def _completion_gate_missing(reads: Reads) -> tuple[Defect, ...]:
    completion = reads.completion_gate
    if completion is None or completion.task_id in reads.gate.dependencies:
        return ()
    return (Defect(
        DefectKind.MISSING, completion.task_id,
        f'not a dependency of the human gate: {_FINAL_RULE}', completion,
    ),)


def _escapee(reads: Reads, row: state.TaskRow) -> Defect:
    tag = row.metadata.get('x_finding_run')
    carries = 'no x_finding_run' if tag is None else f'x_finding_run {tag}'
    return Defect(
        DefectKind.ESCAPES, row.task_id,
        f'prd_path {row.metadata.get("prd_path")}, {carries}: the completion measure counts '
        f'only x_finding_run = {reads.run_id}',
        row,
    )


def judge_final(reads: Reads) -> tuple[Defect, ...]:
    """§3: the gate depends on its completion gate only, whose measure must count every open task."""
    completion_id = reads.completion_gate.task_id if reads.completion_gate else None
    foreign = tuple(
        Defect(DefectKind.FOREIGN, dep, _FINAL_RULE, reads.row(dep))
        for dep in reads.gate.dependencies if dep != completion_id
    )
    escapes = tuple(
        _escapee(reads, row) for row in reads.program
        if _is_open(row) and row.metadata.get('x_finding_run') != reads.run_id
    )
    return (*_empty_program(reads), *_completion_gate_missing(reads), *foreign, *escapes)


_JUDGEMENTS: Mapping[ChainForm, Callable[[Reads], tuple[Defect, ...]]] = {
    ChainForm.INTERIM: judge_interim,
    ChainForm.FINAL: judge_final,
}


def judge(reads: Reads) -> Verdict:
    return Verdict(
        reads.form, reads.gate, reads.completion_gate, _JUDGEMENTS[reads.form](reads),
        reads.measured_at,
    )


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
    completion_read = reader.completion_gate(run_id)
    program_read = reader.program_tasks(run_id, prd_paths)
    gate = gate_read.rows[0]
    known_ids = {row.task_id for row in (*program_read.rows, *completion_read.rows)}
    outsider_reads = {
        dep: reader.task(dep) for dep in gate.dependencies if dep not in known_ids
    }
    return Reads(
        run_id=run_id, prd_paths=prd_paths, gate=gate,
        completion_gate=completion_read.rows[0] if completion_read.rows else None,
        program=program_read.rows,
        outsiders={dep: read.rows[0] if read.rows else None for dep, read in outsider_reads.items()},
        measured_at=min(read.measured_at for read in (
            gate_read, completion_read, program_read, *outsider_reads.values())),
    )


def _subject(defect: Defect) -> str:
    if defect.row is not None:
        row = defect.row
        return f'task {row.task_id} ({row.priority}, {row.status}) {row.title} — '
    if defect.task_id is not None:
        return f'task {defect.task_id} (absent from the store) — '
    return ''


def _form_line(verdict: Verdict) -> str:
    gates = [('human gate', verdict.gate), ('completion gate', verdict.completion_gate)]
    named = ', '.join(f'{role} {row.task_id} ({row.status})' for role, row in gates if row)
    return f'{verdict.form} form: {named}; store read {verdict.measured_at}'


def _report(verdict: Verdict) -> None:
    gate = verdict.gate
    print(_form_line(verdict))
    for defect in verdict.defects:
        print(f'{defect.kind}: {_subject(defect)}{defect.reason}')
    count = len(verdict.defects)
    if count:
        noun, pronoun = ('defect', 'it') if count == 1 else ('defects', 'them')
        print(f'gate {gate.task_id}: {count} {noun}; correct {pronoun} before the human commits '
              'the gate')
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
    verdict = judge(_read_store(reader, run_id, prd_paths, gate_read))
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
