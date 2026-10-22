#!/usr/bin/env python3
"""Clerk batches for a /review-all Phase 6: ``plans/review-all-program-tooling-prd.md`` §4 H4."""
from __future__ import annotations

import argparse
import collections
import enum
import itertools
import json
import sys
import types
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
import state


class Anchor(enum.StrEnum):
    """How a check anchors a move: its ``paths`` name the move's ``from``, or only its
    ``pattern`` names a moved symbol."""

    PATH = 'path'
    SYMBOL = 'symbol'


@dataclass(frozen=True)
class Impact:
    task: int
    check: str
    kind: Any
    from_path: str
    to_path: str | None
    anchor: Anchor


def _checks(row: state.TaskRow) -> Iterator[tuple[int, Mapping[str, Any]]]:
    entries = row.metadata.get('delivered_checks')
    if not isinstance(entries, list | tuple):
        return
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            yield index, entry


def _check_name(entry: Mapping[str, Any], index: int) -> str:
    name = entry.get('name')
    return name if isinstance(name, str) and name else f'delivered_checks[{index}]'


def _anchor(entry: Mapping[str, Any], move: state.Move) -> Anchor | None:
    paths = entry.get('paths')
    if isinstance(paths, list | tuple) and move.from_path in [
        path for path in paths if isinstance(path, str)
    ]:
        return Anchor.PATH
    pattern = entry.get('pattern')
    if isinstance(pattern, str) and any(symbol and symbol in pattern for symbol in move.symbols):
        return Anchor.SYMBOL
    return None


def impact_scan(moves: Sequence[state.Move], rows: Sequence[state.TaskRow]) -> tuple[Impact, ...]:
    """Each (move, task, check) whose check names the move's ``from`` in ``paths`` or whose ``pattern`` contains a moved symbol."""
    ordered = sorted(rows, key=lambda row: row.task_id)
    return tuple(
        Impact(row.task_id, _check_name(entry, index), entry.get('kind'), move.from_path,
               move.to_path, anchor)
        for move in moves for row in ordered for index, entry in _checks(row)
        if (anchor := _anchor(entry, move)) is not None
    )


RESULT_LISTS: Mapping[str, state.ActionResultState] = types.MappingProxyType({
    'applied': state.ActionResultState.APPLIED,
    'skipped': state.ActionResultState.SKIPPED,
    'rejected': state.ActionResultState.REJECTED,
    'kept_for_human': state.ActionResultState.KEPT_FOR_HUMAN,
})
DIRTIED = 'dirtied'
WHY = 'why'
_RESULT_KEY = {result_state: key for key, result_state in RESULT_LISTS.items()}


def _explained(result_state: state.ActionResultState) -> bool:
    """Whether a result entry of *result_state* carries a ``why``: every state but applied does."""
    return result_state is not state.ActionResultState.APPLIED


def _require_every(members: Iterable[Any], covered: Collection[Any], what: str) -> None:
    missing = [str(member) for member in members if member not in covered]
    if missing:
        raise state.StateError(f'{what} misses {", ".join(missing)}')


_require_every(state.ActionResultState, _RESULT_KEY, 'RESULT_LISTS')

_SKIPPED = _RESULT_KEY[state.ActionResultState.SKIPPED]
_REJECTED = _RESULT_KEY[state.ActionResultState.REJECTED]
_APPLIED = _RESULT_KEY[state.ActionResultState.APPLIED]
_KEPT = _RESULT_KEY[state.ActionResultState.KEPT_FOR_HUMAN]

_STANDING_RULES = (
    '`get_task` first, for every action; act on what the store says now.',
    f'Skip, recording `{_SKIPPED}` with why, a task that is in-progress or '
    f'{" or ".join(state.TERMINAL_STATUSES)}, or whose task branch already carries commits.',
    'Metadata writes merge; never pass `append=True`.',
    'Never write a task whose metadata carries `trigger_chain` (the gate), nor its dependencies.',
    f'An action whose effect the task already shows is recorded `{_APPLIED}` with no write, so '
    'an action batched twice is applied once.',
    f'A judgement call goes to `{_KEPT}` with why; an action that cannot be applied as given '
    f'goes to `{_REJECTED}` with why.',
    'An action the batch lists under `accepted` needs a human, and the lead accepted it: apply it.',
    'Scripts never write the store, and you never commit: you write through the fused-memory MCP '
    f'tools, edit files on disk, and list every file you edited in `{DIRTIED}`.',
)

_RELEASED_VALUES: Mapping[str, str] = types.MappingProxyType({
    'by': 'who releases it',
    'at': 'the UTC time of the write',
    'onto': '`target`',
    'contract_as_delivered': '`text`',
})
_require_every(state.SEAM_RELEASED_KEYS.fields, _RELEASED_VALUES, '_RELEASED_VALUES')
_RELEASED_RECORD = ', '.join(
    f'{field}: {_RELEASED_VALUES[field]}' for field in state.SEAM_RELEASED_KEYS.fields
)

_RECIPES: Mapping[state.LeadActionOp, str] = types.MappingProxyType({
    state.LeadActionOp.CANCEL: (
        '`target` lists the subsuming leaf ids. Merge `metadata.x_acceptance_reason` = `text`, '
        'then `set_task_status(status="cancelled")`.'
    ),
    state.LeadActionOp.EDGE: (
        '`add_dependency(id=<task>, depends_on=<id>)` for each id in `target`; a non-empty '
        '`text` is a coordination note appended to the details.'
    ),
    state.LeadActionOp.NOTE: "Append `text` to the task's details.",
    state.LeadActionOp.STAMP: (
        'Merge the finding keys in `target` into `metadata.x_finding_key`, keeping those there.'
    ),
    state.LeadActionOp.REPATH: (
        '`target` is the new path; `text` names the check, its kind and its old path. Three '
        'writes: (1) `metadata.delivered_checks` as the COMPLETE list, the named check\'s old '
        'path replaced by `target` in its `paths` and every other field, `kind` included, '
        'untouched; (2) the sidecar yaml that '
        '`shared/src/shared/capability_manifest.py::sidecar_path` names for the task\'s '
        '`metadata.prd_path`, when it has one; (3) that sidecar\'s md twin. Edit the two files '
        f'on disk under the project root and list both in `{DIRTIED}`.'
    ),
    state.LeadActionOp.RELEASE: (
        '`target` lists the ids the held leaf waits on; `text` is the contract as delivered. In '
        'the order of `skills/review-all/references/deliberation.md` §Launching step 3: '
        '(1) `add_dependency` onto each id in `target`; (2) the description rewritten to '
        f'`text`; (3) `metadata.{state.SEAM_RELEASED_KEYS.key}` = {{{_RELEASED_RECORD}}}, '
        'merged; (4) `commit_planning(task_ids=<task>)`, which runs the polarity gate and stamps '
        'the sidecar; (5) only if (4) returns `not_planning_hold`, '
        '`set_task_status(status="pending")`. `set_task_status` never comes first. List the '
        f'stamped sidecar in `{DIRTIED}`.'
    ),
    state.LeadActionOp.AMEND: "Rewrite the task's description to `text`.",
})
_require_every(state.LeadActionOp, _RECIPES, '_RECIPES')

CLERK_RULES = '\n'.join([
    '## Clerk rules',
    '',
    *(f'- {rule}' for rule in _STANDING_RULES),
    '',
    '## Recipes, by `lead_op`',
    '',
    *(f'- `{op}`: {_RECIPES[op]}' for op in state.LeadActionOp),
])

BY = 'clerk_batch.py'


def _lines(ledger: Iterable[Mapping[str, Any]], op: state.LedgerOp) -> list[Mapping[str, Any]]:
    return [line for line in ledger if line['op'] == op]


def _results_by_id(ledger: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Each action id's first ``action_result`` line."""
    results: dict[str, Mapping[str, Any]] = {}
    for line in _lines(ledger, state.LedgerOp.ACTION_RESULT):
        results.setdefault(line['id'], line)
    return results


def _recorded(result: Mapping[str, Any]) -> str:
    return f'already recorded at seq {result["seq"]} ({result["state"]}, batch {result["batch"]})'


def _unapplied(ledger: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    results = _results_by_id(ledger)
    return [line for line in _lines(ledger, state.LedgerOp.ACTION) if line['id'] not in results]


def _accept_refusals(ledger: Sequence[Mapping[str, Any]], accept: Sequence[str]) -> list[str]:
    """Each ``--accept`` id that names no unapplied action needing a human, with why."""
    actions = {line['id']: line for line in _lines(ledger, state.LedgerOp.ACTION)}
    results = _results_by_id(ledger)
    refusals = []
    for action_id in accept:
        action = actions.get(action_id)
        if action is None:
            why = 'the ledger records no such action'
        elif action_id in results:
            why = _recorded(results[action_id])
        elif not action['needs_human']:
            why = 'needs no human; write batches it without --accept'
        else:
            continue
        refusals.append(f'--accept {action_id}: {why}')
    return refusals


def _batchable(action: Mapping[str, Any], accept: Collection[str]) -> bool:
    return not action['needs_human'] or action['id'] in accept


def _batched_fields(action: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in action.items() if key not in state.LEDGER_COMMON_FIELDS}


def _entry_shape(result_state: state.ActionResultState) -> Any:
    return {'id': '<id>', WHY: '<why>'} if _explained(result_state) else '<id>'


def _result_shape() -> str:
    lists = [
        f'- `{key}`: the actions whose state is {result_state}, each '
        f'{json.dumps(_entry_shape(result_state))}'
        for key, result_state in RESULT_LISTS.items()
    ]
    example = {key: [_entry_shape(result_state)] for key, result_state in RESULT_LISTS.items()}
    example[DIRTIED] = ['<path relative to the project root>']
    return '\n'.join([
        '## The result file',
        '',
        'ONE JSON object with exactly these keys; name every action of the batch once, by `id`:',
        '',
        *lists,
        f'- `{DIRTIED}`: every file you edited on disk, relative to the project root, for the '
        'wave commit',
        '',
        '```json',
        json.dumps(example, indent=2, ensure_ascii=False),
        '```',
    ])


def _prompt(program: state.Program, files: state.BatchFiles, number: int) -> str:
    header = '\n'.join([
        f'# Clerk batch {number} of {program.run_id}',
        '',
        f'Apply each action in `{files.actions}` to the task store of the project at '
        f'`{program.project_root}`, through the fused-memory MCP tools with '
        f'`project_root="{program.project_root}"`.',
        f'Then write the result file `{files.result}`; the lead ingests it with '
        f'`clerk_batch.py ingest {files.result}`.',
    ])
    return f'{header}\n\n{CLERK_RULES}\n\n{_result_shape()}\n'


DELETED_PATH = 'check anchors a deleted path'
SYMBOL_ONLY = 'check names a moved symbol in its pattern but not the old path in its paths'


def _held_for_human(impact: Impact) -> str | None:
    """Why the repath of *impact* is a judgement call, or None when the clerk applies it as given."""
    if impact.to_path is None:
        return DELETED_PATH
    if impact.anchor is Anchor.SYMBOL:
        return SYMBOL_ONLY
    return None


@dataclass(frozen=True)
class _DecomposeReturn:
    """An applied decompose return: its name as the ledger gives it, its stream and its moves."""

    name: str
    stream: str
    moves: tuple[state.Move, ...]


def _return_names(ledger: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each return a ``stream_status`` line names, in seq order, once."""
    named = (line.get('return') for line in _lines(ledger, state.LedgerOp.STREAM_STATUS))
    return list(dict.fromkeys(name for name in named if name is not None))


def _return_file(home: state.Home, name: str) -> Path:
    path = (home.root / name).resolve()
    if not path.is_relative_to(home.root.resolve()):
        raise state.StateError(f'the ledger names the return {name!r}, outside {home.root}')
    return path


def _decompose_return(home: state.Home, name: str) -> _DecomposeReturn | None:
    """The return *name* when its own ``pass`` is decompose; None for any other pass."""
    path = _return_file(home, name)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['pass'] != state.Pass.DECOMPOSE:
            return None
        moves = tuple(
            state.Move(from_path=move['from'], to_path=move['to'], symbols=tuple(move['symbols']))
            for move in data['moves']
        )
        return _DecomposeReturn(name, data['stream'], moves)
    except (OSError, KeyError, TypeError, ValueError) as bad:
        raise state.StateError(f'{path}: cannot read the applied return\'s moves: {bad!r}') from None


def _pending_scans(home: state.Home, ledger: Sequence[Mapping[str, Any]]) -> list[_DecomposeReturn]:
    """Each applied decompose return no ``scanned`` line names, with the moves no earlier return
    of its stream carried."""
    scanned = {line['return'] for line in _lines(ledger, state.LedgerOp.SCANNED)}
    carried: dict[str, set[state.Move]] = {}
    pending = []
    for name in _return_names(ledger):
        ret = _decompose_return(home, name)
        if ret is None:
            continue
        earlier = carried.setdefault(ret.stream, set())
        if name not in scanned:
            new = tuple(move for move in ret.moves if move not in earlier)
            pending.append(replace(ret, moves=new))
        earlier.update(ret.moves)
    return pending


def _checked_rows(program: state.Program, pending: Sequence[_DecomposeReturn]) -> tuple[state.TaskRow, ...]:
    """The non-terminal tasks carrying delivered checks, read once, and only when a move needs it."""
    if not any(ret.moves for ret in pending):
        return ()
    read = state.open_reader(program).tasks_with_delivered_checks()
    reread = f'; {len(read.reread)} re-read in full' if read.reread else ''
    print(f'store read at {read.measured_at}: {len(read.rows)} task(s) with delivered checks{reread}')
    return read.rows


def _repath_fields(ret: _DecomposeReturn, impact: Impact) -> dict[str, Any]:
    named = (f'anchors {impact.from_path}' if impact.anchor is Anchor.PATH
             else f'names a symbol moved from {impact.from_path}')
    moved = ', which the stream deletes' if impact.to_path is None else f', moved to {impact.to_path}'
    return {
        'lead_op': str(state.LeadActionOp.REPATH), 'task': impact.task, 'target': impact.to_path,
        'text': f'check {impact.check} (kind {impact.kind}) {named}{moved}',
        'needs_human': _held_for_human(impact) is not None, 'return': ret.name,
    }


def _kept_for_human(action_id: str, why: str) -> dict[str, Any]:
    return {'id': action_id, 'state': str(state.ActionResultState.KEPT_FOR_HUMAN), 'batch': None,
            'detail': why}


_Entry = tuple[state.LedgerOp, Mapping[str, Any]]


def _stamped(
    snapshot: home_writer.Snapshot, lines: Sequence[Mapping[str, Any]], entries: Sequence[_Entry],
) -> tuple[dict[str, Any], ...]:
    """*entries* stamped to follow the snapshot's ledger and then *lines*."""
    after = replace(snapshot, ledger=(*snapshot.ledger, *lines))
    return home_writer.stamp(after, BY, entries, ()).ledger


def _scan_lines(
    snapshot: home_writer.Snapshot, pending: Sequence[_DecomposeReturn],
    rows: Sequence[state.TaskRow],
) -> tuple[dict[str, Any], ...]:
    """Per return: one repath action per impact, a kept-for-human result for each impact held
    for a human, then the return's ``scanned`` line naming every action minted for it."""
    lines: list[dict[str, Any]] = []
    for ret in pending:
        impacts = impact_scan(ret.moves, rows)
        actions = _stamped(snapshot, lines, [
            (state.LedgerOp.ACTION, _repath_fields(ret, impact)) for impact in impacts
        ])
        held = [(action['id'], _held_for_human(impact))
                for action, impact in zip(actions, impacts, strict=True)]
        lines += actions
        lines += _stamped(snapshot, lines, [
            *((state.LedgerOp.ACTION_RESULT, _kept_for_human(action_id, why))
              for action_id, why in held if why is not None),
            (state.LedgerOp.SCANNED, {'return': ret.name, 'repaths': [a['id'] for a in actions]}),
        ])
    return tuple(lines)


def _print_scanned(lines: Sequence[Mapping[str, Any]]) -> None:
    kept = {line['id'] for line in _lines(lines, state.LedgerOp.ACTION_RESULT)}
    scanned = _lines(lines, state.LedgerOp.SCANNED)
    if not scanned:
        print('scan: every applied decompose return is already scanned')
    for line in scanned:
        repaths = [i for i in line['repaths'] if i not in kept]
        held = [i for i in line['repaths'] if i in kept]
        found = (f'{len(repaths)} repath(s) {" ".join(repaths)}; '
                 f'{len(held)} kept for human {" ".join(held)}') if line['repaths'] else 'nothing'
        print(f'{line["return"]}: {found}')


def _scan(home: state.Home) -> None:
    """Ledger the impact of every unscanned decompose return's moves, in one append."""
    with home_writer.locked(home) as snapshot:
        pending = _pending_scans(home, snapshot.ledger)
        lines = _scan_lines(snapshot, pending, _checked_rows(snapshot.program, pending))
        if lines:
            home_writer.append(home, home_writer.Lines(lines, ()))
    _print_scanned(lines)


def _free_batch(home: state.Home) -> int:
    return next(k for k in itertools.count(1) if not home.batch_files(k).actions.exists())


def _create(path: Path, text: str) -> None:
    try:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
    except FileExistsError:
        raise state.StateError(f'{path} appeared while this write ran; re-run write') from None


def _write_batch(
    home: state.Home, program: state.Program, actions: Sequence[Mapping[str, Any]],
    accepted: Sequence[str],
) -> tuple[int, state.BatchFiles]:
    number = _free_batch(home)
    files = home.batch_files(number)
    files.actions.parent.mkdir(exist_ok=True)
    batch = {
        'run_id': program.run_id, 'batch': number, 'result_file': str(files.result),
        'accepted': list(accepted), 'actions': [_batched_fields(action) for action in actions],
    }
    _create(files.actions, json.dumps(batch, indent=2, ensure_ascii=False, default=dict) + '\n')
    _create(files.prompt, _prompt(program, files, number))
    return number, files


def _print_awaiting(awaiting: Sequence[Mapping[str, Any]]) -> None:
    if awaiting:
        print('awaiting a human, not batched:')
    for action in awaiting:
        print(f'  {action["id"]} {action["lead_op"]} {action["task"]}: {action["text"]} '
              f'(--accept {action["id"]})')


def write(home: state.Home, *, accept: Sequence[str], scan: bool) -> int:
    """Scan first when asked, then batch every unapplied action that needs no human or is
    accepted; exit 1 on a refused --accept, before anything is written."""
    refusals = _accept_refusals(home_writer.read(home).ledger, accept)
    for refusal in refusals:
        print(refusal, file=sys.stderr)
    if refusals:
        return 1
    if scan:
        _scan(home)
    snapshot = home_writer.read(home)
    unapplied = _unapplied(snapshot.ledger)
    batched = [action for action in unapplied if _batchable(action, accept)]
    awaiting = [action for action in unapplied if not _batchable(action, accept)]
    if not batched:
        print('nothing to batch')
    else:
        number, files = _write_batch(home, snapshot.program, batched, list(dict.fromkeys(accept)))
        print(f'batch {number}: {len(batched)} action(s) → {files.actions}')
        print(f'prompt: {files.prompt}')
        print(f'result: {files.result}')
    _print_awaiting(awaiting)
    return 0


@dataclass(frozen=True)
class ResultEntry:
    """One action the clerk's result file reports: its id, its state, and why (None for applied)."""

    action_id: str
    state: state.ActionResultState
    why: str | None


def _batch_of(result_path: Path) -> tuple[state.Home, int]:
    """The run home and batch number whose result file *result_path* is, by state's naming rule."""
    path = result_path.resolve()
    home = state.Home(path.parent.parent)
    if home.program_json.is_file():
        for number in itertools.count(1):
            files = home.batch_files(number)
            if not files.actions.exists():
                break
            if files.result == path:
                return home, number
    raise state.StateError(f'{result_path} is the result file of no batch in a run home')


def _batch_ids(batch_file: Path) -> tuple[str, ...]:
    """The batch's action ids, in its order."""
    try:
        return tuple(action['id'] for action in json.loads(batch_file.read_text())['actions'])
    except (KeyError, TypeError, ValueError) as bad:
        raise state.StateError(f'{batch_file}: cannot read its action ids: {bad!r}') from None


def _text_fields(
    item: Mapping[str, Any], where: str, names: Sequence[str], violations: list[str],
) -> bool:
    """Whether *item* carries exactly *names*, each a non-empty string; each fault is named."""
    faults = [f'{where}.{name}: is required, a non-empty string' for name in names
              if not isinstance(item.get(name), str) or not item[name]]
    faults += [f'{where}.{name}: is no field of a result entry' for name in sorted(set(item) - set(names))]
    violations += faults
    return not faults


def _result_entry(
    item: Any, where: str, result_state: state.ActionResultState, violations: list[str],
) -> ResultEntry | None:
    if not _explained(result_state):
        if isinstance(item, str) and item:
            return ResultEntry(item, result_state, None)
        violations.append(f'{where}: must be an action id, a non-empty string; got {item!r}')
        return None
    if not isinstance(item, Mapping):
        violations.append(f'{where}: must be an object {{id, {WHY}}}; got {item!r}')
        return None
    if not _text_fields(item, where, ('id', WHY), violations):
        return None
    return ResultEntry(item['id'], result_state, item[WHY])


def _listed(data: Mapping[str, Any], key: str, violations: list[str]) -> list[tuple[str, Any]]:
    """Each item of list *key*, with its field path; a missing key is reported by the caller."""
    value = data.get(key, [])
    if not isinstance(value, list):
        violations.append(f'{key}: must be a list; got {value!r}')
        return []
    return [(f'{key}[{index}]', item) for index, item in enumerate(value)]


@dataclass(frozen=True)
class ClerkResult:
    entries: tuple[ResultEntry, ...]
    dirtied: tuple[str, ...]


def _parse_result(text: str) -> ClerkResult | list[str]:
    """The result file's entries and dirtied paths, or every violation of its form."""
    try:
        data = json.loads(text)
    except ValueError as bad:
        return [f'is not JSON: {bad}']
    if not isinstance(data, Mapping):
        return [f'must be a JSON object; got {data!r}']
    keys = {*RESULT_LISTS, DIRTIED}
    violations = [f'{key}: is required' for key in sorted(keys - data.keys())]
    violations += [f'{key}: is no key of a result file' for key in sorted(data.keys() - keys)]
    entries = [
        _result_entry(item, where, result_state, violations)
        for key, result_state in RESULT_LISTS.items() for where, item in _listed(data, key, violations)
    ]
    dirtied = []
    for where, item in _listed(data, DIRTIED, violations):
        if isinstance(item, str) and item:
            dirtied.append(item)
        else:
            violations.append(f'{where}: must be a path, a non-empty string; got {item!r}')
    if violations:
        return violations
    return ClerkResult(tuple(entry for entry in entries if entry is not None), tuple(dirtied))


def _sorted_entries(
    entries: Sequence[ResultEntry], batch_ids: Collection[str],
    results: Mapping[str, Mapping[str, Any]], number: int,
) -> tuple[list[ResultEntry], list[str]]:
    """The entries to record, and a refusal naming each id the batch lacks, the ledger already
    records, or the result names more than once."""
    by_id: dict[str, list[ResultEntry]] = {}
    for entry in entries:
        by_id.setdefault(entry.action_id, []).append(entry)
    recorded, refusals = [], []
    for action_id, named in by_id.items():
        if action_id not in batch_ids:
            why = f'not an action of batch-{number}'
        elif action_id in results:
            why = _recorded(results[action_id])
        elif len(named) > 1:
            why = 'named in ' + ' and '.join(_RESULT_KEY[entry.state] for entry in named)
        else:
            recorded += named
            continue
        refusals.append(f'{action_id}: {why}')
    return recorded, refusals


def _result_fields(entry: ResultEntry, number: int) -> dict[str, Any]:
    return {'id': entry.action_id, 'state': str(entry.state), 'batch': number, 'detail': entry.why}


def _print_ingested(
    number: int, recorded: Sequence[ResultEntry], refusals: Sequence[str],
    unmentioned: Sequence[str], dirtied: Sequence[str],
) -> None:
    counts = collections.Counter(entry.state for entry in recorded)
    print(f'batch {number}: recorded ' + ', '.join(
        f'{result_state} {counts[result_state]}' for result_state in RESULT_LISTS.values()
    ))
    for refusal in refusals:
        print(f'refused: {refusal}', file=sys.stderr)
    for action_id in unmentioned:
        print(f'{action_id}: no result; it stays unapplied and goes into the next batch')
    if dirtied:
        print('dirtied, for the wave commit:')
    for path in dirtied:
        print(f'  {path}')


def ingest(result_path: Path) -> int:
    """Record the clerk's result for its batch; exit 1 when the file or any id is refused."""
    home, number = _batch_of(result_path)
    batch_ids = _batch_ids(home.batch_files(number).actions)
    parsed = _parse_result(result_path.read_text(encoding='utf-8'))
    if not isinstance(parsed, ClerkResult):
        print(f'refused: {result_path} is malformed; nothing is recorded:', file=sys.stderr)
        for violation in parsed:
            print(f'  {violation}', file=sys.stderr)
        return 1
    with home_writer.locked(home) as snapshot:
        recorded, refusals = _sorted_entries(
            parsed.entries, batch_ids, _results_by_id(snapshot.ledger), number,
        )
        if recorded:
            home_writer.append(home, home_writer.stamp(snapshot, BY, [
                (state.LedgerOp.ACTION_RESULT, _result_fields(entry, number)) for entry in recorded
            ], ()))
    named = {entry.action_id for entry in parsed.entries}
    unmentioned = [action_id for action_id in batch_ids if action_id not in named]
    _print_ingested(number, recorded, refusals, unmentioned, parsed.dirtied)
    return 1 if refusals else 0


class Command(enum.StrEnum):
    WRITE = 'write'
    INGEST = 'ingest'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or '').partition('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    write_command = commands.add_parser(
        Command.WRITE, help="batch the ledger's unapplied actions for the clerk seat",
    )
    write_command.add_argument('--run', required=True, help='the run id whose home is read')
    write_command.add_argument(
        '--scan', action='store_true',
        help="first ledger a repath for each delivered check an unscanned return's moves anchor",
    )
    write_command.add_argument(
        '--accept', action='extend', nargs='+', default=[], metavar='ID',
        help='batch these needs-human actions too: a human accepted them',
    )
    ingest_command = commands.add_parser(
        Command.INGEST, help="record the clerk's result file for its batch in the ledger",
    )
    ingest_command.add_argument('result', type=Path, help='a batch-<n>.result.json of a run home')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == Command.INGEST:
            return ingest(args.result)
        return write(state.run_home(args.run), accept=args.accept, scan=args.scan)
    except (state.StateError, OSError) as refused:
        print(f'refused: {refused}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
