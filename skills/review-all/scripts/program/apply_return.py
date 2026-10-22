#!/usr/bin/env python3
"""Apply one PRD-seat return from a run home's inbox to the home's ledger and rulings."""
from __future__ import annotations

import argparse
import enum
import functools
import hashlib
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
import questions
import state

DEFAULT_BY = 'apply_return.py'

_Entry = tuple[state.LedgerOp, dict[str, Any]]
_StoreFactory = Callable[[state.Program], state.StoreView]


class RefusalKind(enum.StrEnum):
    FORM = 'form'
    CONTRADICTION = 'contradiction'
    APPLICATION = 'application'


@dataclass(frozen=True)
class Applied:
    stem: str
    returns_file: Path
    ledger: tuple[dict[str, Any], ...]
    rulings: tuple[dict[str, Any], ...]
    digest: str
    needs_human: tuple[questions.RecordedQuestion, ...]
    measured_at: str | None


@dataclass(frozen=True)
class Refused:
    kind: RefusalKind
    lines: tuple[str, ...]


class NotInInbox(ValueError):
    """The file given is not in a run home's inbox/: a usage error, not a refusal."""


def home_of(inbox_file: Path) -> state.Home:
    """The run home an inbox file sits in: the inbox/ directory's parent."""
    resolved = inbox_file.resolve()
    if resolved.parent.name != 'inbox':
        raise NotInInbox(f'{inbox_file} is not in a run home\'s inbox/ directory')
    return state.Home(resolved.parent.parent)


def _refused(rejections: Sequence[state.Rejection]) -> Refused:
    contradicted = any(r.kind is state.RejectionClass.CONTRADICTION for r in rejections)
    kind = RefusalKind.CONTRADICTION if contradicted else RefusalKind.FORM
    return Refused(kind, tuple(str(r) for r in rejections))


def _name_rejection(home: state.Home, inbox_file: Path, ret: state.PrdReturn) -> state.Rejection | None:
    expected = home.inbox_file(ret.stream, ret.pass_, ret.item)
    if inbox_file.resolve() == expected:
        return None
    return state.Rejection(
        '<file>', state.RejectionClass.FORM,
        f'{inbox_file.name} carries the {expected.stem} return; name it {expected.name}',
    )


def _status_entry(ret: state.PrdReturn, status: state.StreamStatus, return_name: str) -> _Entry:
    return state.LedgerOp.STREAM_STATUS, {
        'stream': ret.stream, 'status': str(status), 'prd_path': ret.prd_path,
        'return': return_name,
    }


def _question_entries(ret: state.PrdReturn, return_name: str) -> list[_Entry]:
    return [
        (state.LedgerOp.QUESTION, questions.question_entry(ret.stream, question, return_name))
        for question in ret.open_questions
    ]


def _xref_entries(ret: state.PrdReturn, return_name: str) -> list[_Entry]:
    return [
        (state.LedgerOp.XREF, {'from': ret.stream, 'to': xref.to, 'text': xref.text,
                               'return': return_name})
        for xref in ret.xrefs
    ]


def _target_json(target: Any) -> Any:
    return list(target) if isinstance(target, tuple) else target


def _action_entries(ret: state.PrdReturn, return_name: str) -> list[_Entry]:
    """One unnumbered ``action`` entry per lead action; ``_numbered_actions`` gives the ids."""
    return [
        (state.LedgerOp.ACTION, {
            'lead_op': str(action.op), 'task': action.task, 'target': _target_json(action.target),
            'text': action.text, 'needs_human': action.needs_human, 'return': return_name,
        })
        for action in ret.lead_actions
    ]


def _numbered_actions(entries: Iterable[_Entry], ledger: Iterable[Mapping[str, Any]]) -> list[_Entry]:
    """Action ids ``A<n>``, dense across every action line the ledger holds."""
    number = 1 + sum(1 for line in ledger if line['op'] == state.LedgerOp.ACTION)
    numbered: list[_Entry] = []
    for op, fields in entries:
        if op is state.LedgerOp.ACTION:
            fields = {'id': f'A{number}', **fields}
            number += 1
        numbered.append((op, fields))
    return numbered


def _seat_entries(ret: state.PrdReturn, return_name: str, *, status: state.StreamStatus) -> list[_Entry]:
    """An author or decompose return's lines: its stream's new status, then its lists."""
    return [
        _status_entry(ret, status, return_name),
        *_question_entries(ret, return_name),
        *_xref_entries(ret, return_name),
        *_action_entries(ret, return_name),
    ]


_PASS_ENTRIES: Mapping[state.Pass, Callable[[state.PrdReturn, str], list[_Entry]]] = {
    state.Pass.AUTHOR: functools.partial(_seat_entries, status=state.StreamStatus.AUTHORED),
    state.Pass.DECOMPOSE: functools.partial(_seat_entries, status=state.StreamStatus.DECOMPOSED),
}


def _count(lines: Iterable[Mapping[str, Any]], op: state.LedgerOp) -> int:
    return sum(1 for line in lines if line['op'] == op)


def _digest(ret: state.PrdReturn, ledger: Sequence[Mapping[str, Any]], return_name: str) -> str:
    return (
        f'{ret.stream} {ret.pass_}: {len(ret.filed)} filed, {len(ret.held)} held, '
        f'{_count(ledger, state.LedgerOp.QUESTION)} questions, '
        f'{_count(ledger, state.LedgerOp.ACTION)} actions → {return_name}'
    )


def _home_relative(home: state.Home, path: Path) -> str:
    return path.relative_to(home.root).as_posix()


def _amendment_file(returns_file: Path, number: int) -> Path:
    return returns_file.with_name(f'{returns_file.stem}.amend-{number}.json')


def _prior_returns(returns_file: Path) -> tuple[Path, ...]:
    """The return's first applied file, then its amendments in order; empty when never applied."""
    if not returns_file.exists():
        return ()
    priors = [returns_file]
    while _amendment_file(returns_file, len(priors)).exists():
        priors.append(_amendment_file(returns_file, len(priors)))
    return tuple(priors)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _application_target(
    home: state.Home, returns_file: Path, raw: bytes, *, amend: bool,
) -> Path | Refused:
    """Where this application's file goes, or why the return may not be applied (again)."""
    stem, priors = returns_file.stem, _prior_returns(returns_file)
    same = [prior for prior in priors if _sha256(prior.read_bytes()) == _sha256(raw)]
    if same:
        return Refused(RefusalKind.APPLICATION, (
            f'{stem} is already applied: {_home_relative(home, same[0])} has the same content',))
    if priors and not amend:
        return Refused(RefusalKind.APPLICATION, (
            f'{stem} is already applied with different content '
            f'({_home_relative(home, priors[0])}); re-run with --amend to append the delta',))
    if amend and not priors:
        return Refused(RefusalKind.APPLICATION, (
            f'nothing to amend: {stem} is not applied; apply it without --amend',))
    return _amendment_file(returns_file, len(priors)) if priors else returns_file


def plan_application(
    snapshot: home_writer.Snapshot, inbox_file: Path, *, amend: bool, store: state.StoreView,
    by: str = DEFAULT_BY,
) -> Applied | Refused:
    """What applying *inbox_file* would append and where it would move it; reads, never writes."""
    home = home_of(inbox_file)
    raw = inbox_file.read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as bad:
        return _refused([state.Rejection('<file>', state.RejectionClass.FORM, f'is not UTF-8: {bad}')])
    checked = state.validate_return(
        text, program=snapshot.program, ledger=snapshot.ledger, store=store,
    )
    if checked.value is None:
        return _refused(checked.rejections)
    ret = checked.value
    misnamed = _name_rejection(home, inbox_file, ret)
    if misnamed is not None:
        return _refused([misnamed])
    entries = _PASS_ENTRIES.get(ret.pass_)
    if entries is None:
        return Refused(RefusalKind.APPLICATION, (f'{ret.pass_} returns are not applied yet',))
    returns_file = _application_target(
        home, home.returns_file(ret.stream, ret.pass_, ret.item), raw, amend=amend,
    )
    if isinstance(returns_file, Refused):
        return returns_file
    return_name = _home_relative(home, returns_file)
    lines = home_writer.stamp(
        snapshot, by, _numbered_actions(entries(ret, return_name), snapshot.ledger), (),
    )
    recorded = questions.recorded_questions(lines.ledger)
    return Applied(
        stem=returns_file.stem, returns_file=returns_file,
        ledger=lines.ledger, rulings=lines.rulings,
        digest=_digest(ret, lines.ledger, return_name),
        needs_human=tuple(q for q in recorded.values() if q.needs_human),
        measured_at=checked.measured_at,
    )


def apply(
    inbox_file: Path, *, amend: bool = False, by: str = DEFAULT_BY,
    store_factory: _StoreFactory = state.open_reader,
) -> Applied | Refused:
    """Apply *inbox_file* under the home's lock: append its lines, then move it into returns/."""
    home = home_of(inbox_file)
    with home_writer.locked(home) as snapshot:
        outcome = plan_application(
            snapshot, inbox_file, amend=amend, store=store_factory(snapshot.program), by=by,
        )
        if isinstance(outcome, Applied):
            home_writer.append(home, home_writer.Lines(outcome.ledger, outcome.rulings))
            outcome.returns_file.parent.mkdir(exist_ok=True)
            os.replace(inbox_file, outcome.returns_file)
    return outcome


def check(inbox_file: Path, *, store_factory: _StoreFactory = state.open_reader) -> Applied | Refused:
    """What ``apply`` would do with *inbox_file*, without the lock and without writing."""
    snapshot = home_writer.read(home_of(inbox_file))
    return plan_application(
        snapshot, inbox_file, amend=False, store=store_factory(snapshot.program),
    )


_CLOSING: Mapping[RefusalKind, str] = {
    RefusalKind.FORM: 'nothing written; fix {file} and re-apply',
    RefusalKind.CONTRADICTION: 'nothing written; the store contradicts the return — stop for the lead',
    RefusalKind.APPLICATION: 'nothing written; {file} is left where it is',
}


def _report(outcome: Applied | Refused, source: Path, *, prefix: str = '') -> int:
    if isinstance(outcome, Refused):
        for line in outcome.lines:
            print(line, file=sys.stderr)
        print(_CLOSING[outcome.kind].format(file=source), file=sys.stderr)
        return 1
    print(prefix + outcome.digest)
    if outcome.measured_at is not None:
        print(f'  store read {outcome.measured_at}')
    for question in outcome.needs_human:
        print(f'  needs_human {question.ref}: {question.question} '
              f'[recommend: {question.recommendation}]')
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='apply_return.py', description=__doc__)
    parser.add_argument('file', nargs='?', type=Path, help='the inbox file to apply')
    parser.add_argument('--check', type=Path, metavar='FILE',
                        help='report what applying FILE would do, writing nothing')
    parser.add_argument('--from-store', metavar='STREAM',
                        help="record STREAM's decompose from the task store")
    parser.add_argument('--run', help='the run id, with --from-store')
    parser.add_argument('--amend', action='store_true',
                        help='append what an already-applied return adds')
    return parser


def _validated(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    modes = [mode for mode in (args.file, args.check, args.from_store) if mode is not None]
    if len(modes) != 1:
        parser.error('give exactly one of FILE, --check FILE or --from-store STREAM')
    if (args.from_store is None) != (args.run is None):
        parser.error('--from-store and --run go together')
    if args.amend and args.file is None:
        parser.error('--amend applies a FILE')


def _run(args: argparse.Namespace) -> int:
    if args.check is not None:
        return _report(check(args.check), args.check, prefix='check ok: ')
    if args.file is not None:
        return _report(apply(args.file, amend=args.amend), args.file)
    print('refused: --from-store is not implemented yet', file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validated(parser, args)
    try:
        return _run(args)
    except NotInInbox as misplaced:
        parser.error(str(misplaced))
    except (state.StateError, OSError) as refused:
        print(f'refused: {refused}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
