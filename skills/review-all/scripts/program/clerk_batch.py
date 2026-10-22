#!/usr/bin/env python3
"""Clerk batches for a /review-all Phase 6: ``plans/review-all-program-tooling-prd.md`` §4 H4."""
from __future__ import annotations

import argparse
import enum
import itertools
import json
import sys
import types
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_writer
import state


@dataclass(frozen=True)
class Impact:
    task: int
    check: str
    kind: Any
    from_path: str
    to_path: str | None


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


def _anchors(entry: Mapping[str, Any], move: state.Move) -> bool:
    paths = entry.get('paths')
    named = isinstance(paths, list | tuple) and move.from_path in [
        path for path in paths if isinstance(path, str)
    ]
    pattern = entry.get('pattern')
    return named or (
        isinstance(pattern, str) and any(symbol and symbol in pattern for symbol in move.symbols)
    )


def impact_scan(moves: Sequence[state.Move], rows: Sequence[state.TaskRow]) -> tuple[Impact, ...]:
    """Each (move, task, check) whose check names the move's ``from`` in ``paths`` or whose ``pattern`` contains a moved symbol."""
    ordered = sorted(rows, key=lambda row: row.task_id)
    return tuple(
        Impact(row.task_id, _check_name(entry, index), entry.get('kind'), move.from_path, move.to_path)
        for move in moves for row in ordered for index, entry in _checks(row)
        if _anchors(entry, move)
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
        f'on disk under the project root and list both in `{DIRTIED}`. A check whose `paths` '
        f'does not list the old path matched on a moved symbol only: `{_KEPT}`.'
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


def write(home: state.Home, *, accept: Sequence[str]) -> int:
    """Batch every unapplied action that needs no human or is accepted; exit 1 on a refused --accept."""
    snapshot = home_writer.read(home)
    refusals = _accept_refusals(snapshot.ledger, accept)
    for refusal in refusals:
        print(refusal, file=sys.stderr)
    if refusals:
        return 1
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


class Command(enum.StrEnum):
    WRITE = 'write'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or '').partition('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    write_command = commands.add_parser(
        Command.WRITE, help="batch the ledger's unapplied actions for the clerk seat",
    )
    write_command.add_argument('--run', required=True, help='the run id whose home is read')
    write_command.add_argument(
        '--accept', action='extend', nargs='+', default=[], metavar='ID',
        help='batch these needs-human actions too: a human accepted them',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return write(state.run_home(args.run), accept=args.accept)
    except state.StateError as refused:
        print(f'refused: {refused}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
