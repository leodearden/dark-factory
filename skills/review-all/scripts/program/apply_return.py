#!/usr/bin/env python3
"""Apply one PRD-seat return from a run home's inbox to the home's ledger and rulings."""
from __future__ import annotations

import argparse
import enum
import hashlib
import json
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


_SEAT_STATUS: Mapping[state.Pass, state.StreamStatus] = {
    state.Pass.AUTHOR: state.StreamStatus.AUTHORED,
    state.Pass.DECOMPOSE: state.StreamStatus.DECOMPOSED,
}


def _list_entries(ret: state.PrdReturn, return_name: str) -> list[_Entry]:
    return [
        *_question_entries(ret, return_name),
        *_xref_entries(ret, return_name),
        *_action_entries(ret, return_name),
    ]


def _answered_item(ret: state.PrdReturn) -> state.ItemRef | None:
    """The recorded question an investigate return answers; None for any other pass."""
    return None if ret.item is None else state.ItemRef(ret.stream, ret.item)


def _session_ruling_fields(
    ret: state.PrdReturn, ruling: state.SessionRuling, return_name: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        'stream': ret.stream, 'item': ret.item, 'question': ruling.question,
        'ruling': ruling.ruling, 'options': list(ruling.options),
        'consequence': ruling.consequence, 'keys': [],
        'source': f'{return_name} session ruling {ruling.id}', 'supersedes': None,
    }
    item = _answered_item(ret)
    if item is not None:
        fields['answers'] = [str(item)]
    return fields


@dataclass(frozen=True)
class _Entries:
    """What one application appends: ledger entries, and human rulings by rulings_in_session index."""

    ledger: list[_Entry]
    rulings: list[tuple[int, dict[str, Any]]]


def _return_entries(ret: state.PrdReturn, return_name: str) -> _Entries:
    """Everything a first application of *ret* appends; an investigate return appends rulings only."""
    status = _SEAT_STATUS.get(ret.pass_)
    ledger = [] if status is None else [
        _status_entry(ret, status, return_name), *_list_entries(ret, return_name),
    ]
    rulings = [
        (index, _session_ruling_fields(ret, ruling, return_name))
        for index, ruling in enumerate(ret.rulings_in_session)
    ]
    return _Entries(ledger, rulings)


_UNCOMPARED_FIELDS = frozenset({'seq', 'at', 'by', 'op', 'return'})


def _ledger_content(op: str, fields: Mapping[str, Any]) -> str:
    """A ledger entry's claim, without when, by whom, from which file, or an action's id."""
    dropped = _UNCOMPARED_FIELDS | ({'id'} if op == state.LedgerOp.ACTION else set())
    kept = {name: value for name, value in fields.items() if name not in dropped}
    return json.dumps([str(op), kept], sort_keys=True, default=dict)


def _ruling_content(fields: Mapping[str, Any]) -> str:
    return json.dumps([fields['stream'], fields['question'], fields['ruling'],
                       list(fields['options']), fields['consequence']])


@dataclass(frozen=True)
class _Target:
    """Where an application's file goes: the first returns file, or amendment ``amendment``."""

    path: Path
    amendment: int | None
    priors: tuple[str, ...]


def _amendment_entries(
    ret: state.PrdReturn, entries: _Entries, snapshot: home_writer.Snapshot, target: _Target,
    return_name: str,
) -> _Entries:
    """Only what the ledger and rulings lack, plus the stream's current status naming the amendment."""
    applied = {
        _ledger_content(line['op'], line) for line in snapshot.ledger
        if line.get('return') in target.priors
    }
    lists = [
        (op, fields) for op, fields in entries.ledger
        if op is not state.LedgerOp.STREAM_STATUS and _ledger_content(op, fields) not in applied
    ]
    ruled = {
        _ruling_content(ruling) for ruling in snapshot.rulings
        if ruling['ruled_by'] == state.RuledBy.HUMAN and ruling['stream'] == ret.stream
    }
    rulings = [(index, fields) for index, fields in entries.rulings if _ruling_content(fields) not in ruled]
    status = _SEAT_STATUS.get(ret.pass_)
    if status is None:
        return _Entries(lists, rulings)
    current = state.stream_views(snapshot.program, snapshot.ledger)[ret.stream].status or status
    return _Entries([_status_entry(ret, current, return_name), *lists], rulings)


def _count(lines: Iterable[Mapping[str, Any]], op: state.LedgerOp) -> int:
    return sum(1 for line in lines if line['op'] == op)


def _seat_digest(ret: state.PrdReturn, lines: home_writer.Lines, return_name: str) -> str:
    return (
        f'{ret.stream} {ret.pass_}: {len(ret.filed)} filed, {len(ret.held)} held, '
        f'{_count(lines.ledger, state.LedgerOp.QUESTION)} questions, '
        f'{_count(lines.ledger, state.LedgerOp.ACTION)} actions → {return_name}'
    )


def _investigate_digest(ret: state.PrdReturn, lines: home_writer.Lines, return_name: str) -> str:
    ids = ', '.join(ruling['id'] for ruling in lines.rulings)
    return (
        f'{ret.stream} investigate-{ret.item}: {len(lines.rulings)} ruling(s) {ids} '
        f'answer {_answered_item(ret)} → {return_name}'
    )


def _amendment_digest(
    ret: state.PrdReturn, lines: home_writer.Lines, amendment: int, return_name: str,
) -> str:
    return (
        f'{ret.stream} {ret.pass_} amend {amendment}: '
        f'+{_count(lines.ledger, state.LedgerOp.QUESTION)} questions, '
        f'+{_count(lines.ledger, state.LedgerOp.XREF)} xrefs, '
        f'+{_count(lines.ledger, state.LedgerOp.ACTION)} actions, '
        f'+{len(lines.rulings)} rulings → {return_name}'
    )


def _digest(ret: state.PrdReturn, lines: home_writer.Lines, target: _Target, return_name: str) -> str:
    if target.amendment is not None:
        return _amendment_digest(ret, lines, target.amendment, return_name)
    if _answered_item(ret) is not None:
        return _investigate_digest(ret, lines, return_name)
    return _seat_digest(ret, lines, return_name)


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
) -> _Target | Refused:
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
    if not priors:
        return _Target(returns_file, None, ())
    names = tuple(_home_relative(home, prior) for prior in priors)
    return _Target(_amendment_file(returns_file, len(priors)), len(priors), names)


def _item_rejections(ret: state.PrdReturn, snapshot: home_writer.Snapshot) -> list[state.Rejection]:
    item = _answered_item(ret)
    if item is None or item in questions.recorded_questions(snapshot.ledger):
        return []
    return [state.Rejection(
        'item', state.RejectionClass.CONTRADICTION,
        f'{item} is no question the ledger records; an investigate return answers a recorded one',
    )]


def _prd_path_rejections(
    ret: state.PrdReturn, snapshot: home_writer.Snapshot, target: _Target,
) -> list[state.Rejection]:
    if target.amendment is None or ret.pass_ not in _SEAT_STATUS:
        return []
    recorded = state.stream_views(snapshot.program, snapshot.ledger)[ret.stream].prd_path
    if ret.prd_path == recorded:
        return []
    return [state.Rejection(
        'prd_path', state.RejectionClass.CONTRADICTION,
        f'{ret.prd_path!r} is not {ret.stream}\'s recorded PRD {recorded!r}; an amendment may not move it',
    )]


def _checked_return(
    home: state.Home, inbox_file: Path, raw: bytes, snapshot: home_writer.Snapshot,
    store: state.StoreView,
) -> tuple[state.PrdReturn, str | None] | Refused:
    """The validated return correctly named in the inbox, and when the store was read for it."""
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as bad:
        return _refused([state.Rejection('<file>', state.RejectionClass.FORM, f'is not UTF-8: {bad}')])
    checked = state.validate_return(
        text, program=snapshot.program, ledger=snapshot.ledger, store=store,
    )
    if checked.value is None:
        return _refused(checked.rejections)
    misnamed = _name_rejection(home, inbox_file, checked.value)
    if misnamed is not None:
        return _refused([misnamed])
    return checked.value, checked.measured_at


def _stamped(snapshot: home_writer.Snapshot, by: str, entries: _Entries) -> home_writer.Lines | Refused:
    """The entries stamped; a session ruling the rulings file would refuse is a form rejection."""
    rulings = [(state.RuledBy.HUMAN, fields) for _, fields in entries.rulings]
    try:
        return home_writer.stamp(
            snapshot, by, _numbered_actions(entries.ledger, snapshot.ledger), rulings,
        )
    except home_writer.RulingRefused as refused:
        index, _ = entries.rulings[refused.index]
        return _refused([state.Rejection(
            f'rulings_in_session[{index}]', state.RejectionClass.FORM, str(refused),
        )])


def plan_application(
    snapshot: home_writer.Snapshot, inbox_file: Path, *, amend: bool, store: state.StoreView,
    by: str = DEFAULT_BY,
) -> Applied | Refused:
    """What applying *inbox_file* would append and where it would move it; reads, never writes."""
    home = home_of(inbox_file)
    raw = inbox_file.read_bytes()
    checked = _checked_return(home, inbox_file, raw, snapshot, store)
    if isinstance(checked, Refused):
        return checked
    ret, measured_at = checked
    returns_file = home.returns_file(ret.stream, ret.pass_, ret.item)
    target = _application_target(home, returns_file, raw, amend=amend)
    if isinstance(target, Refused):
        return target
    rejections = _item_rejections(ret, snapshot) + _prd_path_rejections(ret, snapshot, target)
    if rejections:
        return _refused(rejections)
    return_name = _home_relative(home, target.path)
    entries = _return_entries(ret, return_name)
    if target.amendment is not None:
        entries = _amendment_entries(ret, entries, snapshot, target, return_name)
    lines = _stamped(snapshot, by, entries)
    if isinstance(lines, Refused):
        return lines
    recorded = questions.recorded_questions(lines.ledger)
    return Applied(
        stem=returns_file.stem, returns_file=target.path,
        ledger=lines.ledger, rulings=lines.rulings,
        digest=_digest(ret, lines, target, return_name),
        needs_human=tuple(q for q in recorded.values() if q.needs_human),
        measured_at=measured_at,
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
