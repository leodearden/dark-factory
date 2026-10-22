#!/usr/bin/env python3
"""Render a /review-all program's doc and its .state/ snapshot into the run home.

The only writer of the rendered doc and its snapshot. It reads the home under
program.json's lock and the task store through ``state.StoreReader``; it prints
when the store was read and never writes that stamp into the doc, so two renders
over an unchanged program are byte-identical.
"""
from __future__ import annotations

import argparse
import enum
import os
import string
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_facts
import questions
import run_cli
import state
from program_facts import HomeRead, ProgramState, StoreFacts


class Exit(enum.IntEnum):
    DONE = 0
    REFUSED = 1
    COULD_NOT_RUN = run_cli.COULD_NOT_RUN


_EXIT_MEANINGS: Mapping[Exit, str] = {
    Exit.DONE: 'done',
    Exit.REFUSED: 'refused, nothing written',
    Exit.COULD_NOT_RUN: 'could not run (usage, run home, store or template)',
}

TEMPLATE = Path(__file__).resolve().with_name('program_doc_template.md')
NONE = 'none'
ABSENT = '—'
NO_STATUS = 'no status'
NO_PROGRAM_TASK = 'no program task is in the store yet'


def _line(text: str) -> str:
    """*text* on one line: newlines and runs of whitespace folded to single spaces."""
    return ' '.join(text.split())


def _cell(text: str) -> str:
    return _line(text).replace('|', r'\|')


def _table_row(cells: Sequence[str]) -> str:
    return '| ' + ' | '.join(_cell(cell) for cell in cells) + ' |'


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    return '\n'.join([_table_row(header), _table_row(['---'] * len(header)),
                      *(_table_row(row) for row in rows)])


def _text(value: Any) -> str:
    """A JSON value as flat text: a list comma-joined, a mapping ``k: v; …``, a bool lower-case."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return 'null'
    if isinstance(value, Mapping):
        return '; '.join(f'{name}: {_text(item)}' for name, item in value.items())
    if isinstance(value, list | tuple):
        return ', '.join(_text(item) for item in value) or NONE
    return str(value)


def _listed(values: Iterable[Any]) -> str:
    return _text(tuple(values))


def _bullets(values: Iterable[Any]) -> str:
    return '\n'.join(f'- {_line(_text(value))}' for value in values)


def _bullets_or_none(values: Iterable[Any]) -> str:
    return _bullets(values) or NONE


@dataclass(frozen=True)
class _StreamRow:
    stream: state.Stream
    view: state.StreamView
    wave: int


_STREAM_COLUMNS: tuple[tuple[str, Callable[[_StreamRow], Any]], ...] = (
    ('ID', lambda row: row.stream.id),
    ('Name', lambda row: row.stream.name),
    ('Slug', lambda row: row.stream.slug),
    ('Finding keys', lambda row: _listed(row.view.keys)),
    ('Mode', lambda row: row.view.mode),
    ('Size', lambda row: row.stream.size),
    ('Risk', lambda row: row.stream.risk),
    ('Wave', lambda row: row.wave),
    ('Upstream deps', lambda row: _listed(row.view.deps)),
    ('Deletable', lambda row: row.view.deletable),
    ('Status', lambda row: row.view.status or NO_STATUS),
    ('PRD', lambda row: row.view.prd_path or ABSENT),
    ('Seams', lambda row: _listed(row.stream.seams)),
    ('Remedy', lambda row: row.view.remedy),
)


def _streams(home: HomeRead, store: StoreFacts) -> str:
    run = home.run
    waves = state.waves(run.views)
    rows = [_StreamRow(s, run.views[s.id], waves[s.id]) for s in run.program.streams]
    return _table(
        [header for header, _ in _STREAM_COLUMNS],
        [[_text(cell(row)) for _, cell in _STREAM_COLUMNS] for row in rows],
    )


def _program_section(section: state.ProgramSection) -> Callable[[HomeRead, StoreFacts], str]:
    def render(home: HomeRead, store: StoreFacts) -> str:
        return _bullets_or_none(program_facts.program_section(home, section))
    return render


def _conventions_extra(home: HomeRead, store: StoreFacts) -> str:
    return _bullets(program_facts.program_section(home, state.ProgramSection.CONVENTIONS_EXTRA))


_DECISION_FIELDS = ('stream', 'item', 'question', 'ruling', 'options', 'consequence', 'keys', 'answers')


def _decision(current: program_facts.CurrentRuling) -> str:
    ruling = current.ruling
    parts = [ruling['id'], ruling['ruled_by'], ruling['date']]
    parts += [
        f'{name}: {_text(ruling[name])}' for name in _DECISION_FIELDS
        if ruling.get(name) not in (None, '', ())
    ]
    if current.superseded:
        parts.append(f'supersedes {_listed(current.superseded)}')
    if ruling.get('reask', False):
        parts.append('re-ask: the items it answers stay pending')
    return '- ' + _line(' · '.join(parts))


def _decisions(home: HomeRead, store: StoreFacts) -> str:
    return '\n'.join(map(_decision, program_facts.current_rulings(home.rulings))) or NONE


def _task_name(row: state.TaskRow) -> str:
    """``[<label> ]<id>``: the task as its PRD names it."""
    return ' '.join(part for part in (program_facts.task_label(row), str(row.task_id)) if part)


def _task_entry(row: state.TaskRow) -> str:
    """``<task name>[ <programme state>]``; the state is left out while the task is filed."""
    programme_state = program_facts.program_state(row)
    if programme_state is ProgramState.FILED:
        return _task_name(row)
    return f'{_task_name(row)} {programme_state}'


_CHAIN_LABELS: Mapping[state.TriggerChainRole, str] = {
    state.TriggerChainRole.HUMAN_GATE: 'human gate',
    state.TriggerChainRole.COMPLETION_GATE: 'completion gate',
}


def _chain_line(run_id: str, role: state.TriggerChainRole, gate: state.TaskRow | None) -> str:
    if gate is None:
        return f'no {role} task carries trigger_chain.run_id = {run_id}'
    return f'{_CHAIN_LABELS[role]} {gate.task_id} · depends on {_listed(gate.dependencies)}'


def _trigger_chain(home: HomeRead, store: StoreFacts) -> str:
    run_id = home.run.program.run_id
    return _bullets([
        _chain_line(run_id, state.TriggerChainRole.HUMAN_GATE, store.gate),
        _chain_line(run_id, state.TriggerChainRole.COMPLETION_GATE, store.completion_gate),
    ])


FILED_COLUMNS = ('Stream', 'PRD', 'Tasks', 'Gate deps')


def _filed_row(home: HomeRead, store: StoreFacts, stream: str) -> list[str]:
    view = home.run.views.get(stream)
    prd = view.prd_path if view is not None and view.prd_path else ABSENT
    tasks = ', '.join(_task_entry(row) for row in store.by_stream[stream]) or NONE
    return [stream, prd, tasks, _listed(store.gate_deps(stream))]


def _filed(home: HomeRead, store: StoreFacts) -> str:
    if store.is_empty:
        tasks = NO_PROGRAM_TASK
    else:
        tasks = _table(FILED_COLUMNS, [_filed_row(home, store, s) for s in store.by_stream])
    return f'{tasks}\n\nTrigger chain:\n\n{_trigger_chain(home, store)}'


NOT_RELEASED = 'not released'
HELD_COLUMNS = (
    'Leaf', 'Waits on', 'Contract as specified', 'Release owner', 'Released onto',
    'Contract as delivered',
)


def _held_row(row: state.TaskRow) -> list[str]:
    wait, released = state.SEAM_WAIT_KEYS, state.SEAM_RELEASED_KEYS
    if released.record(row) is None:
        release = [NOT_RELEASED, ABSENT]
    else:
        release = [_text(released.value(row, 'onto')), _text(released.value(row, 'contract_as_delivered'))]
    return [
        _task_name(row), _text(wait.value(row, 'stream')), _text(wait.value(row, 'contract')),
        _text(wait.value(row, 'release_owner')), *release,
    ]


def _held(home: HomeRead, store: StoreFacts) -> str:
    if not store.seam_leaves:
        return f'no program task carries {state.SEAM_WAIT_KEYS.key}'
    return _table(HELD_COLUMNS, [_held_row(row) for row in store.seam_leaves])


_LEDGER_LEAD = ('seq', 'at', 'by', 'op')


def _ledger_entry(line: Mapping[str, Any]) -> str:
    """``<seq> · <at> · <by> · <op>``, then ``name: value`` for each of the line's own fields."""
    parts = [_text(line[name]) for name in _LEDGER_LEAD]
    parts += [f'{name}: {_text(value)}' for name, value in line.items()
              if name not in state.LEDGER_COMMON_FIELDS]
    return '- ' + _line(' · '.join(parts))


def _ledger(home: HomeRead, store: StoreFacts) -> str:
    return '\n'.join(map(_ledger_entry, home.run.ledger)) or NONE


_SECTIONS: Mapping[str, Callable[[HomeRead, StoreFacts], str]] = {
    'streams': _streams,
    'routed': _program_section(state.ProgramSection.ROUTED),
    'seams': _program_section(state.ProgramSection.SEAMS_EXTRA),
    'invariants': _program_section(state.ProgramSection.INVARIANTS),
    'decisions': _decisions,
    'conventions_extra': _conventions_extra,
    'triage': _program_section(state.ProgramSection.TRIAGE),
    'filed': _filed,
    'held': _held,
    'ledger': _ledger,
}


def _placeholders(home: HomeRead, store: StoreFacts) -> dict[str, str]:
    program = home.run.program
    return {
        'run_id': program.run_id, 'project_id': program.project_id,
        'as_of_sha': program.as_of_sha, 'record_path': program.record_path,
        **{name: render(home, store) for name, render in _SECTIONS.items()},
    }


def render_text(template: Path, home: HomeRead, store: StoreFacts) -> str:
    """The doc: *template* with each ``${placeholder}`` filled; an unknown one is refused, named."""
    values = _placeholders(home, store)
    try:
        return string.Template(template.read_text(encoding='utf-8')).substitute(values)
    except KeyError as unknown:
        raise state.StateError(
            f'{template}: unknown placeholder ${{{unknown.args[0]}}} (known: {", ".join(values)})'
        ) from None
    except ValueError as malformed:
        raise state.StateError(f'{template}: {malformed}') from None


def headings(text: str) -> tuple[str, ...]:
    """The doc's ``## `` heading lines, exactly as written."""
    return tuple(line for line in text.splitlines() if line.startswith('## '))


@dataclass(frozen=True)
class HeadingCheck:
    """A render's headings against the published doc's: what it drops, and what the allowances cover."""

    allowed: tuple[str, ...]
    refused: tuple[str, ...]
    unused: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.refused and not self.unused


def check_headings(published: str | None, rendered: str, allowances: Sequence[str]) -> HeadingCheck:
    """Each published heading the render loses must be named by an allowance, and each allowance used."""
    kept = set(headings(rendered))
    dropped = () if published is None else tuple(
        dict.fromkeys(heading for heading in headings(published) if heading not in kept)
    )
    return HeadingCheck(
        allowed=tuple(heading for heading in dropped if heading in allowances),
        refused=tuple(heading for heading in dropped if heading not in allowances),
        unused=tuple(a for a in dict.fromkeys(allowances) if a not in dropped),
    )


def _published_text(program: state.Program) -> str | None:
    published = state.published_doc(program)
    return published.read_text(encoding='utf-8') if published.exists() else None


def _report_refusal(check: HeadingCheck, published: Path, *, exists: bool) -> None:
    for heading in check.refused:
        print(f'refused: the render drops {heading} from {published}; '
              f'--allow-drop "{heading}" lets it go', file=sys.stderr)
    where = str(published) if exists else f'{published}, which does not exist yet'
    for allowance in check.unused:
        print(f'refused: --allow-drop "{allowance}" names a heading the render does not drop '
              f'from {where}', file=sys.stderr)
    print('nothing written', file=sys.stderr)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write *path* through a temp file in its existing directory, so a reader never sees half of it."""
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(handle, 'wb') as out:
            out.write(data)
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _write_render(home: HomeRead, text: str) -> Path:
    run_home, program = home.run.home, home.run.program
    doc = run_home.render_doc(program)
    run_home.render_state_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(doc, text.encode('utf-8'))
    for name, data in home.files.items():
        _write_atomic(run_home.render_state_dir / name, data)
    return doc


@dataclass(frozen=True)
class Render:
    """A render written into the home, and the store read it was made from."""

    home: HomeRead
    store: StoreFacts
    doc: Path


def _checked_render(args: argparse.Namespace) -> Render | None:
    """Render, check the headings against the published doc, and write; None when refused."""
    home, store = _read(args)
    program = home.run.program
    text = render_text(args.template or TEMPLATE, home, store)
    published = _published_text(program)
    check = check_headings(published, text, args.allow_drop or ())
    if not check.passed:
        _report_refusal(check, state.published_doc(program), exists=published is not None)
        return None
    doc = _write_render(home, text)
    for heading in check.allowed:
        print(f'dropped {heading} (allowed by --allow-drop)')
    print(f'rendered {doc} (snapshot {home.run.home.render_state_dir})')
    return Render(home, store, doc)


def _render(args: argparse.Namespace) -> Exit:
    render = _checked_render(args)
    if render is None:
        return Exit.REFUSED
    print(f'store read {render.store.measured_at}')
    return Exit.DONE


@dataclass(frozen=True)
class Copy:
    """One file publish copies; *origin* names what it is, for a refusal."""

    source: Path
    destination: Path
    origin: str


def draft_copy(run: run_cli.Run, stream: str, name: str) -> Copy:
    """The one naming rule for a stream's author-pass file: ``prds/<S>/<name>`` → ``<plans_dir>/<name>``."""
    program = run.program
    return Copy(
        run.home.prds_dir(stream) / name,
        Path(program.project_root) / program.plans_dir / name,
        f'stream {stream}',
    )


def _drafts(run: run_cli.Run, stream: str) -> list[Copy]:
    directory = run.home.prds_dir(stream)
    if not directory.exists():
        return []
    copies = []
    for entry in sorted(directory.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            raise state.StateError(f'{entry} is not a regular file; nothing published')
        copies.append(draft_copy(run, stream, entry.name))
    return copies


def _refuse_clashes(copies: Sequence[Copy]) -> None:
    origins: dict[Path, list[str]] = {}
    for copy in copies:
        origins.setdefault(copy.destination, []).append(copy.origin)
    for destination, named in origins.items():
        if len(named) > 1:
            raise state.StateError(
                f'{destination.name} is published by {" and ".join(named)} to {destination}; '
                'nothing published'
            )


def _refuse_missing_parents(copies: Sequence[Copy], made: Path) -> None:
    for copy in copies:
        parent = copy.destination.parent
        if parent != made and not parent.is_dir():
            raise state.StateError(f'{parent} does not exist to publish {copy.destination.name} into; '
                                   'nothing published')


def copy_plan(render: Render) -> tuple[Copy, ...]:
    """Every copy publish makes, in order, each refusal checked before any is made."""
    run, program = render.home.run, render.home.run.program
    if not Path(program.project_root).is_dir():
        raise state.StateError(f'project_root {program.project_root} is not a directory; nothing published')
    snapshot = state.snapshot_dir(program)
    copies = [
        Copy(render.doc, state.published_doc(program), 'the program doc'),
        *(Copy(run.home.render_state_dir / name, snapshot / name, 'the snapshot')
          for name in render.home.files),
    ]
    for stream in program.stream_ids:
        copies += _drafts(run, stream)
    _refuse_clashes(copies)
    _refuse_missing_parents(copies, snapshot)
    return tuple(copies)


def _publish(args: argparse.Namespace) -> Exit:
    render = _checked_render(args)
    if render is None:
        return Exit.REFUSED
    copies = copy_plan(render)
    program = render.home.run.program
    state.snapshot_dir(program).mkdir(exist_ok=True)
    for copy in copies:
        _write_atomic(copy.destination, copy.source.read_bytes())
    root = Path(program.project_root)
    print(f'published {len(copies)} paths into {root}; commit them with one git commit --only')
    for copy in copies:
        print(copy.destination.relative_to(root).as_posix())
    print(f'store read {render.store.measured_at}')
    return Exit.DONE


def _read(args: argparse.Namespace) -> tuple[HomeRead, StoreFacts]:
    home = program_facts.read_home(args.run)
    return home, program_facts.read_store(state.open_reader(home.run.program), home)


def _held_leaf_line(stream: str, row: state.TaskRow, store: StoreFacts) -> str:
    wait = state.SEAM_WAIT_KEYS
    waits_on = wait.value(row, 'stream')
    label = program_facts.task_label(row)
    named = stream if label is None else f'{stream} {label}'
    waited_on_filed = [other.task_id for other in store.by_stream.get(waits_on, ())]
    return (
        f'held {row.task_id} ({named}) waits on {waits_on} {wait.value(row, "contract")} · '
        f'release owner {wait.value(row, "release_owner")} · {waits_on} filed: {_listed(waited_on_filed)}'
    )


def _roster_lines(home: HomeRead, store: StoreFacts) -> list[str]:
    run = home.run
    waves = state.waves(run.views)
    actions = program_facts.open_actions(run.ledger)
    pending = program_facts.pending_questions(home)
    lines = []
    for stream in run.program.stream_ids:
        rows = store.by_stream[stream]
        held = [row for row in rows if program_facts.program_state(row) is ProgramState.HELD]
        lines.append(' · '.join([
            stream, f'wave {waves[stream]}', str(run.views[stream].status or NO_STATUS),
            f'filed {len(rows)}', f'held {len(held)}', f'gate deps {len(store.gate_deps(stream))}',
            f'open actions {sum(1 for a in actions if a.stream == stream)}',
            f'open questions {sum(1 for q in pending if q.ref.stream == stream)}',
        ]))
    lines.append(' · '.join([
        state.UNSTREAMED, f'filed {len(store.by_stream[state.UNSTREAMED])}',
        f'gate deps {len(store.gate_deps(state.UNSTREAMED))}',
    ]))
    lines += [
        _held_leaf_line(stream, row, store)
        for stream, rows in store.by_stream.items() for row in rows
        if program_facts.program_state(row) is ProgramState.HELD
    ]
    lines.append(_chain_line(run.program.run_id, state.TriggerChainRole.HUMAN_GATE, store.gate))
    return lines


def _roster(args: argparse.Namespace) -> Exit:
    home, store = _read(args)
    for line in _roster_lines(home, store):
        print(line)
    print(f'store read {store.measured_at}')
    return Exit.DONE


SCREEN_SIZE = 4
CLOSING_BLOCK = 'lead_rulable — one uncontested yes/no block'
NOTHING_PENDING = 'nothing pending'


def _draft(run: run_cli.Run, stream: str) -> str:
    prd_path = run.views[stream].prd_path
    if prd_path is None:
        return f'no PRD is recorded for {stream}'
    return str(draft_copy(run, stream, PurePosixPath(prd_path).name).source)


def _sitting_item(run: run_cli.Run, question: questions.RecordedQuestion) -> list[str]:
    """One item in the shape the sitting puts it: the question, its options, recommendation and draft."""
    return [
        f'{question.ref} [{question.source}] {_line(question.question)}',
        f'  options: {" · ".join(map(_line, question.options)) or NONE}',
        f'  recommendation: {_line(question.recommendation)}',
        f'  draft: {_draft(run, question.ref.stream)}',
    ]


def _wave_streams(run: run_cli.Run, wave: int) -> frozenset[str]:
    waves = state.waves(run.views)
    streams = frozenset(stream for stream, number in waves.items() if number == wave)
    if not streams:
        known = ', '.join(map(str, sorted(set(waves.values()))))
        raise state.StateError(f'wave {wave} is no wave of {run.program.run_id}; its waves are {known}')
    return streams


def _sitting_lines(home: HomeRead, wave: int) -> list[str]:
    run = home.run
    streams = _wave_streams(run, wave)
    pending = [q for q in program_facts.pending_questions(home) if q.ref.stream in streams]
    if not pending:
        return [NOTHING_PENDING]
    screened = [q for q in pending if not q.lead_rulable]
    lead_rulable = [q for q in pending if q.lead_rulable]
    screens = [screened[i:i + SCREEN_SIZE] for i in range(0, len(screened), SCREEN_SIZE)]
    lines = [f'sitting wave {wave}: {len(screened)} pending on {len(screens)} screen(s), '
             f'{len(lead_rulable)} lead_rulable']
    for number, screen in enumerate(screens, 1):
        lines.append(f'screen {number} of {len(screens)}')
        lines += [line for question in screen for line in _sitting_item(run, question)]
    if lead_rulable:
        lines.append(CLOSING_BLOCK)
        lines += [line for question in lead_rulable for line in _sitting_item(run, question)]
    return lines


def _sitting(args: argparse.Namespace) -> Exit:
    for line in _sitting_lines(program_facts.read_home(args.run), args.sitting):
        print(line)
    return Exit.DONE


@dataclass(frozen=True)
class _Mode:
    """A mode of the CLI: the flag that selects it, what it does, and the verb its failure names.

    Only a mode that renders takes ``--template`` and ``--allow-drop``.
    """

    flag: str | None
    verb: str
    body: Callable[[argparse.Namespace], Exit]
    renders: bool


_RENDER = _Mode(None, 'render', _render, renders=True)
_MODES = (
    _Mode('publish', 'publish', _publish, renders=True),
    _Mode('roster', 'print the roster', _roster, renders=False),
    _Mode('sitting', 'print the sitting', _sitting, renders=False),
)


def _given(value: object) -> bool:
    """Whether a mode flag was given: a switch is True when it was, ``--sitting 0`` is still given."""
    return value is not None and value is not False


def _selected(args: argparse.Namespace) -> _Mode:
    return next((mode for mode in _MODES if _given(getattr(args, mode.flag))), _RENDER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='program_doc.py', description=(__doc__ or '').partition('\n')[0],
        epilog='exit codes: ' + '; '.join(f'{int(c)} {says}' for c, says in _EXIT_MEANINGS.items()),
    )
    parser.add_argument('--run', required=True, help='the run id whose home the doc is rendered from')
    parser.add_argument('--template', type=Path, metavar='PATH',
                        help=f'render through PATH instead of {TEMPLATE.name}')
    parser.add_argument('--allow-drop', action='append', metavar='HEADING',
                        help='let the render lose this exact "## " heading of the published doc; '
                             'repeat once per heading')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--publish', action='store_true',
                       help='render, then copy the doc, its snapshot and every prds/<S>/ file '
                            'into the checkout and list the paths for one commit')
    modes.add_argument('--roster', action='store_true',
                       help='print one line per stream, each unreleased held leaf and the gate')
    modes.add_argument('--sitting', type=int, metavar='WAVE',
                       help="print WAVE's pending design questions as screens of "
                            f'{SCREEN_SIZE}; reads the run home only, never the store')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    mode = _selected(args)
    if not mode.renders and (args.template is not None or args.allow_drop):
        parser.error('--template and --allow-drop go only with a render or --publish')
    try:
        return mode.body(args)
    except (state.StateError, OSError) as failed:
        print(f'program_doc.py: cannot {mode.verb}: {failed}', file=sys.stderr)
        return Exit.COULD_NOT_RUN


if __name__ == '__main__':
    raise SystemExit(main())
