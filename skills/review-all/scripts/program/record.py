#!/usr/bin/env python3
"""Create a /review-all program run's home (``init``)."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='record.py', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='validate program.json and create the run home')
    init.add_argument('program_json', type=Path)
    init.set_defaults(handler=_init)
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
