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
_ReaderFactory = Callable[[state.Program], state.StoreReader]

DEGENERATE_BYTES = 100
LOST_WITH_THE_RETURN = ('open_questions', 'xrefs', 'lead_actions', 'moves')


class RefusalKind(enum.StrEnum):
    FORM = 'form'
    CONTRADICTION = 'contradiction'
    APPLICATION = 'application'


@dataclass(frozen=True)
class Applied:
    """What was (or, from ``check``, would be) appended; ``returns_file`` is None for a reconstruction."""

    stem: str
    returns_file: Path | None
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
    """One ``action`` entry per lead action; ``home_writer.stamp`` mints its id."""
    return [
        (state.LedgerOp.ACTION, {
            'lead_op': str(action.op), 'task': action.task, 'target': _target_json(action.target),
            'text': action.text, 'needs_human': action.needs_human, 'return': return_name,
        })
        for action in ret.lead_actions
    ]


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


def _session_source(return_name: str) -> str:
    """How the ``source`` of every ruling minted from *return_name*'s rulings_in_session begins."""
    return f'{return_name} session ruling '


def _session_ruling_fields(
    ret: state.PrdReturn, ruling: state.SessionRuling, return_name: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        'stream': ret.stream, 'item': ret.item, 'question': ruling.question,
        'ruling': ruling.ruling, 'options': list(ruling.options),
        'consequence': ruling.consequence, 'keys': [],
        'source': _session_source(return_name) + ruling.id, 'supersedes': None,
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


def _ledgered(snapshot: home_writer.Snapshot, return_name: str) -> bool:
    """Whether a ledger line or a minted ruling already records applying *return_name*."""
    return (any(line.get('return') == return_name for line in snapshot.ledger)
            or any(str(ruling['source']).startswith(_session_source(return_name))
                   for ruling in snapshot.rulings))


def _unmoved(
    home: state.Home, inbox_file: Path, path: Path, snapshot: home_writer.Snapshot,
) -> Refused | None:
    """An application whose lines were appended but whose file never reached returns/."""
    name = _home_relative(home, path)
    if not _ledgered(snapshot, name):
        return None
    return Refused(RefusalKind.APPLICATION, (
        f'{path.stem} is already applied per the ledger, which names {name}; '
        f'move {_home_relative(home, inbox_file.resolve())} to {name}',))


def _application_target(
    home: state.Home, inbox_file: Path, returns_file: Path, raw: bytes,
    snapshot: home_writer.Snapshot, *, amend: bool,
) -> _Target | Refused:
    """Where this application's file goes, or why the return may not be applied (again)."""
    stem, priors = returns_file.stem, _prior_returns(returns_file)
    path = _amendment_file(returns_file, len(priors)) if priors else returns_file
    unmoved = _unmoved(home, inbox_file, path, snapshot)
    if unmoved is not None:
        return unmoved
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
        return _Target(path, None, ())
    return _Target(path, len(priors), tuple(_home_relative(home, prior) for prior in priors))


def _item_rejections(ret: state.PrdReturn, snapshot: home_writer.Snapshot) -> list[state.Rejection]:
    item = _answered_item(ret)
    if item is None or item in questions.recorded_questions(snapshot.ledger):
        return []
    return [state.Rejection(
        'item', state.RejectionClass.CONTRADICTION,
        f'{item} is no question the ledger records; an investigate return answers a recorded one',
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
        return home_writer.stamp(snapshot, by, entries.ledger, rulings)
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
    target = _application_target(home, inbox_file, returns_file, raw, snapshot, amend=amend)
    if isinstance(target, Refused):
        return target
    rejections = _item_rejections(ret, snapshot)
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
            if outcome.returns_file is not None:
                outcome.returns_file.parent.mkdir(exist_ok=True)
                os.replace(inbox_file, outcome.returns_file)
    return outcome


def check(inbox_file: Path, *, store_factory: _StoreFactory = state.open_reader) -> Applied | Refused:
    """What ``apply`` would do with *inbox_file*, without the lock and without writing."""
    snapshot = home_writer.read(home_of(inbox_file))
    return plan_application(
        snapshot, inbox_file, amend=False, store=store_factory(snapshot.program),
    )


def degeneracy(inbox_file: Path) -> str | None:
    """Why *inbox_file* is no usable return (absent, under DEGENERATE_BYTES, unparsable), else None."""
    if not inbox_file.exists():
        return 'absent'
    raw = inbox_file.read_bytes()
    if len(raw) < DEGENERATE_BYTES:
        return f'{len(raw)} bytes, under {DEGENERATE_BYTES}'
    try:
        data = json.loads(raw)
    except ValueError as bad:
        return f'does not parse: {bad}'
    return None if isinstance(data, dict) else 'is not a JSON object'


def _unknown_stream(snapshot: home_writer.Snapshot, home: state.Home, stream: str) -> Refused | None:
    program = snapshot.program
    if stream in program.stream_ids:
        return None
    return _refused([state.Rejection(
        'stream', state.RejectionClass.FORM,
        f'{stream!r} is no stream of {program.run_id}: {list(program.stream_ids)}',
    )])


def _unauthored_stream(snapshot: home_writer.Snapshot, home: state.Home, stream: str) -> Refused | None:
    reason = state.unauthored_reason(state.stream_views(snapshot.program, snapshot.ledger)[stream])
    if reason is None:
        return None
    return _refused([state.Rejection('stream', state.RejectionClass.CONTRADICTION, reason)])


def _decomposed_stream(snapshot: home_writer.Snapshot, home: state.Home, stream: str) -> Refused | None:
    decomposed = [
        line['seq'] for line in snapshot.ledger
        if line['op'] == state.LedgerOp.STREAM_STATUS and line['stream'] == stream
        and line['status'] == state.StreamStatus.DECOMPOSED
    ]
    if not decomposed:
        return None
    return Refused(RefusalKind.APPLICATION, (
        f'{stream} is already decomposed per the ledger (seq {decomposed[0]})',))


def _usable_inbox(snapshot: home_writer.Snapshot, home: state.Home, stream: str) -> Refused | None:
    inbox_file = home.inbox_file(stream, state.Pass.DECOMPOSE)
    if degeneracy(inbox_file) is not None:
        return None
    return Refused(RefusalKind.APPLICATION, (
        f'{_home_relative(home, inbox_file)} holds a usable return; apply it: '
        f'apply_return.py {inbox_file}',))


_FROM_STORE_REFUSALS = (_unknown_stream, _unauthored_stream, _decomposed_stream, _usable_inbox)


@dataclass(frozen=True)
class _Reconstruction:
    """A stream's decompose as the store shows it."""

    filed: tuple[state.TaskRow, ...]
    held: tuple[state.TaskRow, ...]
    released: tuple[state.TaskRow, ...]


def _reconstructed(
    rows: Sequence[state.TaskRow], views: Mapping[str, state.StreamView], stream: str,
) -> _Reconstruction:
    own = [row for row in rows if state.stream_of(row, views) == stream]
    return _Reconstruction(
        filed=tuple(row for row in own if not state.is_held(row)),
        held=tuple(row for row in own if state.is_held(row)),
        released=tuple(row for row in rows if state.SEAM_RELEASED_KEYS.value(row, 'by') == stream),
    )


def _filed_text(rows: Sequence[state.TaskRow]) -> str:
    return ', '.join(str(row.task_id) for row in rows) or 'none'


def _held_leaf_text(row: state.TaskRow) -> str:
    wait = state.SEAM_WAIT_KEYS
    return (f'{row.task_id} on {wait.value(row, "stream")} {wait.value(row, "contract")} '
            f'(release owner {wait.value(row, "release_owner")})')


def _held_text(rows: Sequence[state.TaskRow]) -> str:
    return ', '.join(_held_leaf_text(row) for row in rows) or 'none'


def _released_text(rows: Sequence[state.TaskRow]) -> str:
    return ', '.join(
        f'{row.task_id} onto {list(state.SEAM_RELEASED_KEYS.value(row, "onto") or ())}'
        for row in rows
    ) or 'none'


def _note_text(stream: str, found: _Reconstruction, measured_at: str, leftover: str | None) -> str:
    text = (
        f'{stream} decompose reconstructed from the store read at {measured_at}: '
        f'filed {_filed_text(found.filed)}; held {_held_text(found.held)}; '
        f'released {_released_text(found.released)}; '
        f'lost with the return: {", ".join(LOST_WITH_THE_RETURN)}'
    )
    return text if leftover is None else f'{text}; {leftover}'


def _from_store_digest(stream: str, found: _Reconstruction, leftover: str | None) -> str:
    digest = (
        f'{stream} decompose from store: {len(found.filed)} filed, {len(found.held)} held, '
        f'{len(found.released)} released; lost: {", ".join(LOST_WITH_THE_RETURN)}'
    )
    return digest if leftover is None else f'{digest}; {leftover}'


def _reconstruction_applied(
    snapshot: home_writer.Snapshot, home: state.Home, stream: str, by: str,
    reader: state.StoreReader,
) -> Applied:
    views = state.stream_views(snapshot.program, snapshot.ledger)
    prd_paths = [view.prd_path for view in views.values() if view.authored and view.prd_path]
    read = reader.program_tasks(snapshot.program.run_id, prd_paths)
    found = _reconstructed(read.rows, views, stream)
    inbox_file = home.inbox_file(stream, state.Pass.DECOMPOSE)
    leftover = (f'{_home_relative(home, inbox_file)} left in place: {degeneracy(inbox_file)}'
                if inbox_file.exists() else None)
    lines = home_writer.stamp(snapshot, by, [
        (state.LedgerOp.STREAM_STATUS, {'stream': stream, 'status': str(state.StreamStatus.DECOMPOSED),
                                        'prd_path': views[stream].prd_path}),
        (state.LedgerOp.NOTE, {'text': _note_text(stream, found, read.measured_at, leftover)}),
    ], ())
    return Applied(
        stem=home.returns_file(stream, state.Pass.DECOMPOSE).stem, returns_file=None,
        ledger=lines.ledger, rulings=lines.rulings,
        digest=_from_store_digest(stream, found, leftover), needs_human=(),
        measured_at=read.measured_at,
    )


def from_store(
    home: state.Home, stream: str, *, by: str = DEFAULT_BY,
    store_factory: _ReaderFactory = state.open_reader,
) -> Applied | Refused:
    """Record *stream*'s decompose from the store when its session left no usable return; files nothing."""
    with home_writer.locked(home) as snapshot:
        for refusal_of in _FROM_STORE_REFUSALS:
            refusal = refusal_of(snapshot, home, stream)
            if refusal is not None:
                return refusal
        applied = _reconstruction_applied(snapshot, home, stream, by, store_factory(snapshot.program))
        home_writer.append(home, home_writer.Lines(applied.ledger, applied.rulings))
    return applied


_CLOSING: Mapping[RefusalKind, str] = {
    RefusalKind.FORM: 'nothing written; fix {subject} and re-apply',
    RefusalKind.CONTRADICTION: 'nothing written; the store contradicts the return — stop for the lead',
    RefusalKind.APPLICATION: 'nothing written; {subject} was not applied',
}


def _report(outcome: Applied | Refused, subject: str | Path, *, prefix: str = '') -> int:
    if isinstance(outcome, Refused):
        for line in outcome.lines:
            print(line, file=sys.stderr)
        print(_CLOSING[outcome.kind].format(subject=subject), file=sys.stderr)
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
    outcome = from_store(state.run_home(args.run), args.from_store)
    return _report(outcome, f'--from-store {args.from_store}')


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
