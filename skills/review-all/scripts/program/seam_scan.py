#!/usr/bin/env python3
"""List one stream's G4 seam candidates: open tasks outside the program that name a path it touches.

Read-only: it reads the stream's applied author return and the task store, prints
one candidate row per task in the skills/prd/references/gates.md G4 columns, and
writes nothing. Direction, Owner and Status stay ``?`` for the author's G4 ruling.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import state

G4_COLUMNS = ('Other PRD', 'Direction', 'Seam mechanism', 'Owner', 'Status')
UNRULED = '?'
NONE_LINE = 'no non-terminal task outside the program declares any touched path'


class Anchor(NamedTuple):
    path: str
    check: str


@dataclass(frozen=True)
class SeamCandidate:
    task_id: int
    title: str
    status: str
    prd_path: str | None
    declared: tuple[str, ...]
    anchored: tuple[Anchor, ...]


class Scan(NamedTuple):
    candidates: tuple[SeamCandidate, ...]
    measured_at: str


@dataclass(frozen=True)
class Run:
    """The run home and what it holds: the program, the ledger, and the authored prd paths."""

    home: state.Home
    program: state.Program
    ledger: tuple[Mapping[str, Any], ...]
    prd_paths: tuple[str, ...]


def _load_run(run_id: str) -> Run:
    home = state.run_home(run_id)
    try:
        data = json.loads(home.program_json.read_text())
    except ValueError as bad:
        raise state.StateError(f'{home.program_json} is not JSON: {bad}') from None
    program = state.parse_program(data)
    ledger = state.read_ledger(home)
    views = state.stream_views(program, ledger)
    return Run(home, program, ledger, tuple(v.prd_path for v in views.values() if v.prd_path))


def _author_return(run: Run, stream: str, reader: state.StoreReader) -> state.PrdReturn:
    path = run.home.returns_file(stream, state.Pass.AUTHOR)
    check = state.validate_return(
        path.read_text(), program=run.program, ledger=run.ledger, store=reader,
    )
    assert check.value is not None
    return check.value


def _listed(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list | tuple) else ()


def _declared(row: state.TaskRow, touches: Sequence[str]) -> tuple[str, ...]:
    files = _listed(row.metadata.get('files'))
    return tuple(path for path in touches if path in files)


def _anchored(row: state.TaskRow, touches: Sequence[str]) -> tuple[Anchor, ...]:
    """Each (touched path, check name) where the check's ``paths`` names the path exactly."""
    return tuple(
        Anchor(path, str(check.get('name')))
        for check in _listed(row.metadata.get('delivered_checks')) if isinstance(check, Mapping)
        for path in touches if path in _listed(check.get('paths'))
    )


def _candidate(row: state.TaskRow, touches: Sequence[str]) -> SeamCandidate:
    prd_path = row.metadata.get('prd_path')
    return SeamCandidate(
        task_id=row.task_id, title=row.title, status=row.status,
        prd_path=prd_path if isinstance(prd_path, str) else None,
        declared=_declared(row, touches), anchored=_anchored(row, touches),
    )


def _candidates(
    rows: Iterable[state.TaskRow], program_ids: frozenset[int], touches: Sequence[str],
) -> tuple[SeamCandidate, ...]:
    outside = {row.task_id: row for row in rows if row.task_id not in program_ids}
    found = (_candidate(outside[task_id], touches) for task_id in sorted(outside))
    return tuple(c for c in found if c.declared or c.anchored)


def _scan(reader: state.StoreReader, run: Run, touches: Sequence[str]) -> Scan:
    """The candidates, and when the store was first read for them."""
    reads = (
        reader.program_tasks(run.program.run_id, run.prd_paths),
        reader.tasks_declaring_files(touches),
        reader.tasks_with_delivered_checks(),
    )
    program, declaring, checked = reads
    program_ids = frozenset(row.task_id for row in program.rows)
    candidates = _candidates((*declaring.rows, *checked.rows), program_ids, touches)
    return Scan(candidates, min(read.measured_at for read in reads))


def _cell(text: str) -> str:
    return text.replace('|', r'\|')


def _table_row(cells: Sequence[str]) -> str:
    return '| ' + ' | '.join(_cell(cell) for cell in cells) + ' |'


def _candidate_line(candidate: SeamCandidate) -> str:
    prd = '' if candidate.prd_path is None else f'; prd_path {candidate.prd_path}'
    mechanisms = [f'{path} declared in metadata.files' for path in candidate.declared]
    mechanisms += [f'{a.path} anchored by delivered check {a.check}' for a in candidate.anchored]
    return _table_row([
        f'task {candidate.task_id}: {candidate.title} ({candidate.status}{prd})',
        UNRULED, '; '.join(mechanisms), UNRULED, UNRULED,
    ])


def _report(ret: state.PrdReturn, scan: Scan) -> None:
    print(f'seam_scan: {ret.stream} ({ret.prd_path}) touches {len(ret.touches)} paths; '
          f'store read {scan.measured_at}')
    print(_table_row(G4_COLUMNS))
    print(_table_row(['---'] * len(G4_COLUMNS)))
    for candidate in scan.candidates:
        print(_candidate_line(candidate))
    if not scan.candidates:
        print(NONE_LINE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or '').partition('\n')[0])
    parser.add_argument('stream', help='the stream whose applied author return to scan')
    parser.add_argument('--run', required=True, help='the run id whose home holds the return')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run = _load_run(args.run)
    reader = state.open_reader(run.program)
    ret = _author_return(run, args.stream, reader)
    _report(ret, _scan(reader, run, ret.touches))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
