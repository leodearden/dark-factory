#!/usr/bin/env python3
"""Write one /review-all PRD seat's or session's brief into the run home, built from the run's state alone.

The only writer of the home's ``briefs/``. It reads the home under program.json's
lock and regenerates the whole brief on every call, so a brief never rests on
saved arguments (``plans/review-all-program-tooling-prd.md`` decision 13). It
prints only paths: the lead never reads a brief (``SKILL.md`` §Lead context budget).
"""
from __future__ import annotations

import argparse
import enum
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_facts
import return_contract
import run_cli
import state


class Section(enum.StrEnum):
    """The brief's ``## `` headings."""

    RETURN = 'Return'
    SPAWN = 'Spawn'


class Exit(enum.IntEnum):
    WRITTEN = 0
    REFUSED = 1
    USAGE = 2


_EXIT_MEANINGS: Mapping[Exit, str] = types.MappingProxyType({
    Exit.WRITTEN: 'the brief was written',
    Exit.REFUSED: 'refused, nothing written',
    Exit.USAGE: 'usage error',
})

SEAT_TITLE = '<your seat title>'


@dataclass(frozen=True)
class _Seat:
    """What a brief is built from: the run, the stream it briefs, and the session that will read it."""

    run: run_cli.Run
    stream: str
    pass_: state.Pass
    item: str | None
    session: state.Session

    @property
    def view(self) -> state.StreamView:
        return self.run.views[self.stream]


_Sections = list[tuple[Section, list[str]]]


def _prd_path(seat: _Seat) -> str:
    """The stream's recorded PRD path; a placeholder under plans_dir until its author return is applied."""
    return seat.view.prd_path or f'{seat.run.program.plans_dir}/<your PRD file name>'


def _other_stream(seat: _Seat) -> str:
    return next((s for s in seat.run.program.stream_ids if s != seat.stream), seat.stream)


def _identity(seat: _Seat) -> return_contract.ExampleIdentity:
    return return_contract.ExampleIdentity(
        run_id=seat.run.program.run_id, stream=seat.stream, pass_=seat.pass_, item=seat.item,
        session=seat.session, prd_path=_prd_path(seat), keys=seat.view.keys,
        other_stream=_other_stream(seat),
    )


def _return_section(seat: _Seat) -> tuple[Section, list[str]]:
    return Section.RETURN, return_contract.section_lines(seat.run.home, _identity(seat))


def _author_sections(seat: _Seat) -> _Sections:
    return [_return_section(seat)]


_BUILDERS: Mapping[state.Pass, Callable[[_Seat], _Sections]] = types.MappingProxyType({
    state.Pass.AUTHOR: _author_sections,
})


def _title(seat: _Seat) -> str:
    program = seat.run.program
    item = '' if seat.item is None else f' {seat.item}'
    return f'# {seat.stream} {seat.pass_}{item} brief — {program.run_id} at {program.as_of_sha}'


def _text(seat: _Seat, sections: _Sections) -> str:
    lines = [_title(seat), '']
    for heading, body in sections:
        lines += [f'## {heading}', '', *body, '']
    return '\n'.join(lines)


def _seat(args: argparse.Namespace) -> _Seat:
    home = program_facts.read_home(args.run)
    stream = home.run.program.stream(args.stream).id
    pass_ = state.Pass(args.pass_)
    if pass_ not in _BUILDERS:
        raise state.StateError(f'no {pass_} brief yet')
    return _Seat(
        run=home.run, stream=stream, pass_=pass_, item=args.item,
        session=state.Session(state.SessionKind.AGENT, SEAT_TITLE),
    )


def _write(args: argparse.Namespace) -> Exit:
    seat = _seat(args)
    text = _text(seat, _BUILDERS[seat.pass_](seat))
    path = seat.run.home.brief_file(seat.stream, seat.pass_, seat.item)
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding='utf-8')
    print(f'wrote {path}')
    return Exit.WRITTEN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='gen_brief.py', description=(__doc__ or '').partition('\n')[0],
        epilog='exit codes: ' + '; '.join(f'{int(c)} {says}' for c, says in _EXIT_MEANINGS.items()),
    )
    parser.add_argument('--run', required=True, help='the run id whose home the brief is built from')
    parser.add_argument('stream', help='the stream the brief is for')
    parser.add_argument('pass_', metavar='pass', choices=[str(p) for p in state.Pass])
    parser.add_argument('--item', metavar='Q', help='the recorded question an investigate pass probes')
    return parser


def _usage_errors(args: argparse.Namespace) -> list[str]:
    investigate = args.pass_ == state.Pass.INVESTIGATE
    errors = []
    if args.item is not None and not investigate:
        errors.append('--item goes only with the investigate pass')
    if investigate and args.item is None:
        errors.append('the investigate pass needs --item')
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for error in _usage_errors(args):
        parser.error(error)
    try:
        return _write(args)
    except (state.StateError, OSError) as refused:
        print(f'gen_brief.py: refused: {refused}', file=sys.stderr)
        return Exit.REFUSED


if __name__ == '__main__':
    raise SystemExit(main())
