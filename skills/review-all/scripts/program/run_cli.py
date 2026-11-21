"""The command-line frame of a read-only script over one /review-all run's home.

A script declares its exit states, any arguments beyond ``--run``, and a body
that takes the loaded run. The frame parses the command line, loads the run,
and reports a StateError or OSError as one stderr line and COULD_NOT_RUN, the
code argparse also exits with on a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import state

COULD_NOT_RUN = 2

_Exit = TypeVar('_Exit', bound=int)


@dataclass(frozen=True)
class Run:
    """One run's home and what it holds: the program, the ledger, and each stream after the ledger."""

    home: state.Home
    program: state.Program
    ledger: tuple[Mapping[str, Any], ...]
    views: Mapping[str, state.StreamView]

    @property
    def prd_paths(self) -> tuple[str, ...]:
        """Each prd path the ledger records for a stream: the program's tasks carry one of them."""
        return tuple(view.prd_path for view in self.views.values() if view.prd_path)


def load_run(run_id: str) -> Run:
    home = state.run_home(run_id)
    try:
        data = json.loads(home.program_json.read_text())
    except ValueError as bad:
        raise state.StateError(f'{home.program_json} is not JSON: {bad}') from None
    program = state.parse_program(data)
    if program.run_id != run_id:
        raise state.StateError(f'{home.program_json} is the program of {program.run_id}, not {run_id}')
    ledger = state.read_ledger(home)
    return Run(home, program, ledger, state.stream_views(program, ledger))


def _parser(
    doc: str | None, exits: Mapping[_Exit, str],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(doc or '').partition('\n')[0],
        epilog='exit codes: ' + '; '.join(f'{int(code)} {says}' for code, says in exits.items()),
    )
    parser.add_argument('--run', required=True, help='the run id whose home the script reads')
    if add_arguments is not None:
        add_arguments(parser)
    return parser


def main(
    doc: str | None, exits: Mapping[_Exit, str], body: Callable[[Run, argparse.Namespace], _Exit],
    *, failure: str, add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """*body*'s exit over the run ``--run`` names; COULD_NOT_RUN when the run or the body fails.

    *doc*'s first line describes the script, *exits* lists every code it returns
    for the epilog, and *failure* opens the stderr line (``cannot check``).
    """
    parser = _parser(doc, exits, add_arguments)
    args = parser.parse_args(argv)
    try:
        return body(load_run(args.run), args)
    except (state.StateError, OSError) as failed:
        print(f'{parser.prog}: {failure}: {failed}', file=sys.stderr)
        return COULD_NOT_RUN
