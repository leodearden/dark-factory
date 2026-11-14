#!/usr/bin/env python3
"""Create a /review-all program run's home (``init``) and record rulings in it (``rule``)."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
import questions
import state


def _parsed_program(text: str, source: Path) -> state.Program | None:
    """The validated program, or None after printing every violation."""
    try:
        data = json.loads(text)
    except ValueError as bad:
        print(f'refused: {source} is not JSON: {bad}', file=sys.stderr)
        return None
    try:
        return state.parse_program(data)
    except state.ProgramInvalid as invalid:
        print(f'refused: {source} is not a valid program:', file=sys.stderr)
        for violation in invalid.violations:
            print(f'  {violation}', file=sys.stderr)
        return None


def _init(args: argparse.Namespace) -> int:
    source: Path = args.program_json
    text = source.read_text(encoding='utf-8')
    program = _parsed_program(text, source)
    if program is None:
        return 1
    home = state.run_home(program.run_id)
    home.root.mkdir(parents=True, exist_ok=True)
    try:
        with home.program_json.open('x', encoding='utf-8') as handle:
            handle.write(text)
    except FileExistsError:
        print(f'refused: {home.program_json} exists; init never overwrites', file=sys.stderr)
        return 1
    (home.root / 'inbox').mkdir(exist_ok=True)
    print(f'home: {home.root}')
    print(f'doc: {program.doc_path}')
    return 0


def _answered_items(args: argparse.Namespace, snapshot: home_writer.Snapshot) -> tuple[state.ItemRef, ...]:
    """The ``--answers`` items, each a question the ledger records."""
    recorded = questions.recorded_questions(snapshot.ledger)
    answered = tuple(state.ItemRef.parse(text) for text in args.answers)
    for ref in answered:
        if ref not in recorded:
            raise state.StateError(f'--answers {ref}: the ledger records no question {ref}')
    return answered


def _ruling_fields(args: argparse.Namespace, program: state.Program) -> dict[str, Any]:
    if args.reask and not args.answers:
        raise state.StateError('--reask needs --answers: a re-ask re-opens the items it answers')
    if args.stream is not None and args.stream not in program.stream_ids:
        raise state.StateError(
            f'--stream {args.stream!r} is no stream of {program.run_id}: {list(program.stream_ids)}'
        )
    fields = {
        'stream': args.stream, 'item': args.item, 'question': args.question,
        'ruling': args.ruling, 'options': args.option, 'consequence': args.consequence,
        'keys': args.key, 'source': args.source, 'supersedes': args.supersedes,
    }
    if args.answers:
        fields['answers'] = args.answers
    if args.reask:
        fields['reask'] = True
    return fields


def _print_ruling(
    ruling: Mapping[str, Any], answered: Sequence[state.ItemRef], pending: Sequence[state.ItemRef],
) -> None:
    print(f'ruling {ruling["id"]} recorded ({ruling["ruled_by"]})')
    if ruling['supersedes'] is not None:
        print(f'  supersedes {ruling["supersedes"]}')
    for ref in answered:
        if ref in pending:
            print(f'  {ref}: stays pending: {ruling["id"]} is a re-ask')
        else:
            print(f'  {ref}: closed by {ruling["id"]}')


def _rule(args: argparse.Namespace) -> int:
    home = state.run_home(args.run)
    with home_writer.locked(home) as snapshot:
        fields = _ruling_fields(args, snapshot.program)
        answered = _answered_items(args, snapshot)
        lines = home_writer.stamp(snapshot, 'record.py', (), [(state.RuledBy(args.by), fields)])
        home_writer.append(home, lines)
    (ruling,) = lines.rulings
    recorded = questions.recorded_questions(snapshot.ledger)
    _print_ruling(ruling, answered, questions.pending_items(recorded, [*snapshot.rulings, ruling]))
    return 0


def _configure_rule(rule: argparse.ArgumentParser) -> None:
    rule.add_argument('--run', required=True, help='the run id whose home records the ruling')
    rule.add_argument('--by', required=True, choices=[str(r) for r in state.RuledBy])
    rule.add_argument('--question', required=True)
    rule.add_argument('--ruling', required=True)
    rule.add_argument('--source', required=True, help='where the ruling was made, as prose')
    rule.add_argument('--option', action='append', default=[], help='repeat once per option')
    rule.add_argument('--consequence', default='')
    rule.add_argument('--key', action='append', default=[], help='a finding key it rules on')
    rule.add_argument('--stream')
    rule.add_argument('--item')
    rule.add_argument('--supersedes', help='the id of the ruling this one replaces')
    rule.add_argument('--answers', nargs='+', action='extend', default=[],
                      metavar='STREAM:QUESTION', help='the recorded questions it answers')
    rule.add_argument('--reask', action='store_true', help='the answered items stay pending')
    rule.set_defaults(handler=_rule)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='record.py', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='validate program.json and create the run home')
    init.add_argument('program_json', type=Path)
    init.set_defaults(handler=_init)
    _configure_rule(commands.add_parser('rule', help='record one ruling under a freshly minted D/L id'))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (state.StateError, OSError) as refused:
        print(f'refused: {refused}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
