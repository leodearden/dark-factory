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
import json
import shlex
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_doc
import program_facts
import return_contract
import run_cli
import sessions
import state

sys.path.insert(0, str(state.CHECKOUT_ROOT / 'orchestrator' / 'src'))

from orchestrator import session_registry


class Section(enum.StrEnum):
    """The brief's ``## `` headings."""

    DECOMPOSE = 'Decompose'
    STREAM = 'Stream'
    RULINGS = 'Rulings binding the stream'
    UPSTREAM = 'Upstream program tasks'
    GATE = 'Gate'
    RELEASE_DUTIES = 'Release duties'
    FINDINGS = 'Findings'
    FILINGS = 'Phase 5 filings'
    XREFS = 'Cross-references to the stream'
    SITTING = 'Design questions go to the sitting'
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
LAUNCHING = 'skills/review-all/references/deliberation.md §Launching'
SELF_EXIT_ENV = 'CLAUDE_SPAWN_SELF_EXIT'
SPAWN_SCRIPT = state.CHECKOUT_ROOT / 'skills' / 'spawn' / 'spawn-claude.sh'


@dataclass(frozen=True)
class Measurement:
    name: str
    value: Any
    rule: str


@dataclass(frozen=True)
class Dispute:
    """A finding's ``calibration_disputes_verdict``: the verdict calibration would give, and why."""

    to: str
    note: str


@dataclass(frozen=True)
class Finding:
    key: str
    anchor: str
    severity: str
    statement: str
    measurements: tuple[Measurement, ...]
    verdict: str
    dispute: Dispute | None


@dataclass(frozen=True)
class RunRecord:
    """What a brief reads from the run record: its findings by key and its filings rows as written."""

    path: Path
    findings: Mapping[str, Finding]
    filings: tuple[Mapping[str, Any], ...]


class _RecordReader:
    """Reads the run record at the boundary into typed values; a fault names the path and the field."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def fault(self, where: str, reason: str) -> state.StateError:
        return state.StateError(f'the run record {self.path}: {where} {reason}')

    def entry(self, value: Any, where: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise self.fault(where, f'is not an object: {value!r}')
        return value

    def entries(
        self, parent: Mapping[str, Any], where: str, name: str,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        """Each object of *parent*'s list *name*, with the path that names it (``findings[2]``)."""
        path = f'{where}.{name}' if where else name
        value = parent.get(name)
        if not isinstance(value, list):
            raise self.fault(path, f'is not a list: {value!r}')
        return [(f'{path}[{i}]', self.entry(item, f'{path}[{i}]')) for i, item in enumerate(value)]

    def text(self, entry: Mapping[str, Any], where: str, name: str) -> str:
        value = entry.get(name)
        if not isinstance(value, str):
            raise self.fault(f'{where}.{name}', f'is not a string: {value!r}')
        return value

    def measurement(self, entry: Mapping[str, Any], where: str) -> Measurement:
        if 'value' not in entry:
            raise self.fault(f'{where}.value', 'is missing')
        return Measurement(self.text(entry, where, 'name'), entry['value'], self.text(entry, where, 'rule'))

    def dispute(self, finding: Mapping[str, Any], where: str) -> Dispute | None:
        raw = finding.get('calibration_disputes_verdict')
        if raw is None:
            return None
        where = f'{where}.calibration_disputes_verdict'
        entry = self.entry(raw, where)
        return Dispute(self.text(entry, where, 'to'), self.text(entry, where, 'note'))

    def finding(self, entry: Mapping[str, Any], where: str) -> Finding:
        return Finding(
            key=self.text(entry, where, 'key'), anchor=self.text(entry, where, 'anchor'),
            severity=self.text(entry, where, 'severity'),
            statement=self.text(entry, where, 'statement'),
            measurements=tuple(
                self.measurement(m, at) for at, m in self.entries(entry, where, 'measurements')
            ),
            verdict=self.text(entry, where, 'verdict'), dispute=self.dispute(entry, where),
        )

    def filing(self, entry: Mapping[str, Any], where: str) -> Mapping[str, Any]:
        """The row as written, once its ``key`` is known to be a string."""
        self.text(entry, where, 'key')
        return types.MappingProxyType(dict(entry))

    def record(self) -> RunRecord:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise state.StateError(f'the run record {self.path} is absent') from None
        except (OSError, ValueError) as bad:
            raise state.StateError(f'the run record {self.path} is unreadable: {bad}') from None
        if not isinstance(data, Mapping):
            raise state.StateError(f'the run record {self.path} is not a JSON object')
        findings = [self.finding(entry, at) for at, entry in self.entries(data, '', 'findings')]
        return RunRecord(
            path=self.path,
            findings=types.MappingProxyType({finding.key: finding for finding in findings}),
            filings=tuple(self.filing(entry, at) for at, entry in self.entries(data, '', 'filings')),
        )


def _run_record(program: state.Program) -> RunRecord:
    return _RecordReader(Path(program.project_root) / program.record_path).record()


@dataclass(frozen=True)
class _Seat:
    """What a brief is built from: the home read, the stream it briefs, and the session that will read it."""

    home: program_facts.HomeRead
    stream: str
    pass_: state.Pass
    item: str | None
    session: state.Session

    @property
    def run(self) -> run_cli.Run:
        return self.home.run

    @property
    def view(self) -> state.StreamView:
        return self.run.views[self.stream]

    @property
    def spawned(self) -> bool:
        return self.session.kind is state.SessionKind.SPAWN


_Sections = list[tuple[Section, list[str]]]


@dataclass(frozen=True)
class _Built:
    """A brief's sections, and when the store was read for them; None when it was not read."""

    sections: _Sections
    store_read: str | None = None


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


def _stream_section(seat: _Seat) -> tuple[Section, list[str]]:
    cells = program_doc.stream_cells(seat.run, seat.stream)
    return Section.STREAM, [
        f'{seat.stream} after the ledger\'s amendments, as the program doc\'s Streams table shows it:',
        '',
        *(f'- {column}: {cell}' for column, cell in cells),
    ]


def _binds(ruling: Mapping[str, Any], view: state.StreamView) -> bool:
    """A ruling binds the stream it names, every stream when it names none, and the streams whose keys
    or recorded questions it rules on; a re-ask binds nothing, since the items it answers stay pending."""
    if ruling.get('reask', False):
        return False
    answered = {state.ItemRef.parse(text).stream for text in ruling.get('answers', ())}
    return (ruling['stream'] in (None, view.id) or not set(ruling['keys']).isdisjoint(view.keys)
            or view.id in answered)


def _rulings_section(seat: _Seat) -> tuple[Section, list[str]]:
    binding = [
        program_doc.decision_line(current)
        for current in program_facts.current_rulings(seat.home.rulings)
        if _binds(current.ruling, seat.view)
    ]
    return Section.RULINGS, binding or [f'No current ruling binds {seat.stream}.']


def _filings_section(seat: _Seat, record: RunRecord) -> tuple[Section, list[str]]:
    rows = [row for row in record.filings if row['key'] in seat.view.keys]
    if not rows:
        return Section.FILINGS, [f'The run record {record.path} files no row for {seat.stream}\'s keys.']
    return Section.FILINGS, [
        f'The rows of {record.path} `filings` for {seat.stream}\'s keys, verbatim:',
        '',
        '```jsonl',
        *(json.dumps(dict(row), ensure_ascii=False) for row in rows),
        '```',
    ]


def _xrefs_section(seat: _Seat) -> tuple[Section, list[str]]:
    xrefs = [
        f'- from {line["from"]} ({line["return"]}): {line["text"]}'
        for line in seat.run.ledger
        if line['op'] == state.LedgerOp.XREF and line['to'] == seat.stream
    ]
    return Section.XREFS, xrefs or [f'No applied return addresses an xref to {seat.stream}.']


def _seat_sections(seat: _Seat, record: RunRecord) -> _Sections:
    """The sections both seat passes carry, author and decompose."""
    return [
        _stream_section(seat),
        _rulings_section(seat),
        _filings_section(seat, record),
        _xrefs_section(seat),
    ]


def _verdict_line(finding: Finding) -> str:
    line = f'- verdict: {finding.verdict}'
    if finding.dispute is not None:
        line += f' · calibration disputes it: {finding.dispute.to} — {finding.dispute.note}'
    return line


def _finding_lines(key: str, finding: Finding | None) -> list[str]:
    if finding is None:
        return ['', f'### {key}', '', 'The run record carries no finding for this key.']
    return [
        '',
        f'### {key} · {finding.anchor} · {finding.severity}',
        '',
        finding.statement,
        '',
        *(f'- {m.name} = {m.value} ({m.rule})' for m in finding.measurements),
        _verdict_line(finding),
    ]


def _findings_section(seat: _Seat, record: RunRecord) -> tuple[Section, list[str]]:
    return Section.FINDINGS, [
        f'Measured at `{seat.run.program.as_of_sha}`. Re-measure any count a leaf\'s done-criteria '
        'rest on before you cite it.',
        *(line for key in seat.view.keys for line in _finding_lines(key, record.findings.get(key))),
    ]


def _sitting_section(seat: _Seat) -> tuple[Section, list[str]]:
    run = seat.run
    draft = program_doc.draft_copy(run, seat.stream, '<file name>')
    reported = draft.destination.relative_to(run.program.project_root).as_posix()
    return Section.SITTING, [
        '- Every design-level question becomes an `open_questions` entry with `options` and a '
        '`recommendation`, for the author-pass sitting. Never answer one in the seat.',
        '- Mark `lead_rulable` true only for naming, file placement and seam ids; the human still '
        'sees those, as one block.',
        '- Mark `needs_human` true when the question reverses a ruling, cancels a live branch or '
        'moves a board figure.',
        f'- Write the PRD, its capability manifest and its sidecar as `{draft.source}`, and report '
        f'each as `{reported}`.',
        f'- The procedure is {LAUNCHING}.',
    ]


def _author_brief(seat: _Seat) -> _Built:
    record = _run_record(seat.run.program)
    stream, rulings, filings, xrefs = _seat_sections(seat, record)
    return _Built([
        stream, rulings, _findings_section(seat, record), filings, xrefs,
        _sitting_section(seat), _return_section(seat),
    ])


def _refuse_out_of_order(seat: _Seat) -> None:
    """A decompose waits for the stream's own author return and for every upstream stream's decompose."""
    reason = state.unauthored_reason(seat.view)
    if reason is not None:
        raise state.StateError(reason)
    for upstream in seat.view.deps:
        status = seat.run.views[upstream].status
        if status not in program_doc.SETTLED_STATUSES:
            raise state.StateError(
                f'upstream stream {upstream} is {status or program_doc.NO_STATUS}, not decomposed: '
                f'{seat.stream} decomposes once {upstream}\'s filed ids are in the store'
            )


def _decompose_section(seat: _Seat) -> tuple[Section, list[str]]:
    program = seat.run.program
    return Section.DECOMPOSE, [
        f'- Decompose the PRD at `{Path(program.project_root) / _prd_path(seat)}` into tasks.',
        '- Immediately before filing, re-run docs/quality-findings-contract.md §8 step 1 for every '
        'key, whatever the filings rows below say.',
        f'- Stamp every task you file with `x_finding_key` and `x_finding_run` = `{program.run_id}`.',
        '- File under `planning_mode`, run `commit_planning` on your own batch, and never commit.',
        f'- The procedure and its gotchas are {LAUNCHING} step 3.',
    ]


def _task_line(row: state.TaskRow) -> str:
    programme_state = program_facts.program_state(row)
    parts = [str(row.task_id), program_facts.task_label(row) or program_doc.ABSENT, row.priority,
             str(programme_state)]
    if programme_state is program_facts.ProgramState.HELD:
        wait = state.SEAM_WAIT_KEYS
        parts.append(f'waits on {wait.value(row, "stream")} {wait.value(row, "contract")}')
    return '- ' + ' · '.join(parts)


def _upstream_section(seat: _Seat, store: program_facts.StoreFacts) -> tuple[Section, list[str]]:
    if not seat.view.deps:
        return Section.UPSTREAM, [f'{seat.stream} has no upstream stream.']
    lines = ['The store\'s program tasks of each upstream stream: id · label · priority · programme state.']
    for upstream in seat.view.deps:
        rows = store.by_stream[upstream]
        prd = seat.run.views[upstream].prd_path or program_doc.ABSENT
        lines += ['', f'### {upstream} · PRD {prd}', '']
        lines += [_task_line(row) for row in rows] or [f'{upstream} has no program task in the store.']
    return Section.UPSTREAM, lines


def _gate_section(seat: _Seat, store: program_facts.StoreFacts) -> tuple[Section, list[str]]:
    run_id = seat.run.program.run_id
    if store.gate is None:
        return Section.GATE, [
            f'no {state.TriggerChainRole.HUMAN_GATE} task carries trigger_chain.run_id = {run_id}; '
            'file every leaf with `gate_dep: false`.',
        ]
    gate = store.gate
    deps = ', '.join(map(str, gate.dependencies)) or program_doc.NONE
    return Section.GATE, [
        f'The human gate is task {gate.task_id}; it depends on {deps}.',
        f'Add every `{state.Priority.CRITICAL}` or `{state.Priority.HIGH}` leaf you file as a '
        f'dependency of {gate.task_id}, and mark it `gate_dep: true`.',
    ]


def _release_duty_line(stream: str, row: state.TaskRow) -> str:
    wait = state.SEAM_WAIT_KEYS
    owned = ' '.join(part for part in (stream, program_facts.task_label(row)) if part)
    return (f'- {row.task_id} · {owned} · waits on {wait.value(row, "stream")} '
            f'{wait.value(row, "contract")}')


def _release_duties_section(seat: _Seat, store: program_facts.StoreFacts) -> tuple[Section, list[str]]:
    duties = [
        _release_duty_line(stream, row) for stream, row in store.held
        if state.SEAM_WAIT_KEYS.value(row, 'release_owner') == seat.stream
    ]
    if not duties:
        return Section.RELEASE_DUTIES, [f'No held leaf names {seat.stream} as its release owner.']
    return Section.RELEASE_DUTIES, [
        f'The held leaves {seat.stream} releases: id · owning stream and label · what it waits on.',
        '',
        *duties,
        '',
        f'Release each once the leaf delivering its contract is filed, per {LAUNCHING} step 3 '
        '(recipe `clerk_batch.py::CLERK_RULES` `release`), and list it under `released`.',
    ]


def _decompose_brief(seat: _Seat) -> _Built:
    _refuse_out_of_order(seat)
    record = _run_record(seat.run.program)
    store = program_facts.read_store(state.open_reader(seat.run.program), seat.home)
    stream, rulings, filings, xrefs = _seat_sections(seat, record)
    return _Built([
        _decompose_section(seat), stream, rulings, _upstream_section(seat, store),
        _gate_section(seat, store), _release_duties_section(seat, store), filings, xrefs,
        _return_section(seat),
    ], store.measured_at)


_BUILDERS: Mapping[state.Pass, Callable[[_Seat], _Built]] = types.MappingProxyType({
    state.Pass.AUTHOR: _author_brief,
    state.Pass.DECOMPOSE: _decompose_brief,
})


def _code_spans(outcomes: Sequence[session_registry.ResultOutcome]) -> str:
    return ', '.join(f'`{outcome}`' for outcome in outcomes)


def _spawn_section(seat: _Seat) -> tuple[Section, list[str]]:
    """What a spawned session owes its record, in session_registry's outcome vocabulary."""
    result = session_registry.RESULT_FILENAME
    terminal = [o for o in session_registry.RESULT_OUTCOMES if o in session_registry.TERMINAL_RESULT_OUTCOMES]
    inbox = seat.run.home.inbox_file(seat.stream, seat.pass_, seat.item)
    return Section.SPAWN, [
        f'- The inbox file `{inbox}` is your return. `{result}` names that path and carries no '
        'copy of it.',
        f'- After each pass, checkpoint `{result}` with outcome '
        f'`{session_registry.ResultOutcome.IN_PROGRESS}`; its final write carries a terminal '
        f'outcome: {_code_spans(terminal)}.',
        f'- On a context or usage limit, write HANDOFF.md beside `{result}` and end with outcome '
        f'`{session_registry.ResultOutcome.HANDED_OFF}`.',
        '- Run /reflect before you end.',
        '- End as the launch trailer says.',
    ]


def _spawn_command(project_root: str, title: str, brief: Path) -> str:
    """The one command that spawns the session on *brief*, which it ends itself (self-exit)."""
    argv = [str(SPAWN_SCRIPT), project_root, 'true', title, '--prompt-file', str(brief)]
    return f'{SELF_EXIT_ENV}=1 ' + shlex.join(argv)


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
    if args.spawn is None:
        session = state.Session(state.SessionKind.AGENT, SEAT_TITLE)
    else:
        session = state.Session(state.SessionKind.SPAWN, args.spawn)
    return _Seat(home=home, stream=stream, pass_=pass_, item=args.item, session=session)


def _spawned_session(seat: _Seat) -> sessions.SpawnedSession:
    """The sessions.json entry for the spawn, refused before any write when its title is taken."""
    sessions.refuse_taken(seat.run.home, seat.session.title)
    return sessions.SpawnedSession(
        title=seat.session.title, stream=seat.stream, pass_=seat.pass_, item=seat.item,
        spawned_after=state.now_utc(), record_dir=None,
    )


def _write(args: argparse.Namespace) -> Exit:
    seat = _seat(args)
    built = _BUILDERS[seat.pass_](seat)
    sections = [*built.sections, *([_spawn_section(seat)] if seat.spawned else [])]
    text = _text(seat, sections)
    spawned = _spawned_session(seat) if seat.spawned else None
    home = seat.run.home
    path = home.brief_file(seat.stream, seat.pass_, seat.item)
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding='utf-8')
    print(f'wrote {path}')
    if built.store_read is not None:
        print(f'store read {built.store_read}')
    if spawned is not None:
        sessions.add(home, spawned)
        print(f'recorded {spawned.title} in {home.sessions_json}')
        print(_spawn_command(seat.run.program.project_root, spawned.title, path))
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
    parser.add_argument('--spawn', metavar='TITLE',
                        help='record a spawned session under TITLE and print its spawn command')
    return parser


def _usage_errors(args: argparse.Namespace) -> list[str]:
    investigate = args.pass_ == state.Pass.INVESTIGATE
    errors = []
    if args.item is not None and not investigate:
        errors.append('--item goes only with the investigate pass')
    if investigate and args.item is None:
        errors.append('the investigate pass needs --item')
    if args.spawn is not None and args.pass_ == state.Pass.DECOMPOSE:
        errors.append(f'--spawn goes only with author or investigate: a decompose runs as an agent '
                      f'seat ({LAUNCHING})')
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
