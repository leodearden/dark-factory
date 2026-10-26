"""Behavioural tests for ``skills/review-all/scripts/program/clerk_batch.py``.

``impact_scan`` is driven through the module loaded by path. ``write`` and
``ingest`` run as ``-I -S`` subprocesses from a temp cwd outside the repo, over
homes seeded from copies of the committed fixture, which is never edited.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import FIXTURE_RETURNS, RUN_ID, ProgramEnv
from review_all_program_fixture import (
    FIXTURE_DIR,
    PROGRAM_DIR,
    FakeStore,
    load_state,
    running_fake_store,
)

state = load_state()

CLERK_BATCH_PATH = PROGRAM_DIR / 'clerk_batch.py'
CLERK_BATCH_MODULE_NAME = 'review_all_program_clerk_batch'


def _load_clerk_batch() -> types.ModuleType:
    """Load ``clerk_batch.py`` by path, source-authoritative, as ``load_state`` loads state.py."""
    assert CLERK_BATCH_PATH.is_file(), f'module under test is missing: {CLERK_BATCH_PATH}'
    Path(importlib.util.cache_from_source(str(CLERK_BATCH_PATH))).unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(CLERK_BATCH_MODULE_NAME, CLERK_BATCH_PATH)
    assert spec is not None and spec.loader is not None, f'no spec for {CLERK_BATCH_PATH}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[CLERK_BATCH_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


clerk = _load_clerk_batch()

OLD = 'src/fixture/old_name.py'
NEW = 'src/fixture/new_name.py'
REMOVED = 'src/fixture/removed.py'


def _row(task_id: int, checks: Any, status: str = 'pending') -> Any:
    return state.TaskRow(
        task_id, f'task {task_id}', status, 'medium', '2026-10-19T00:00:00Z',
        metadata={'delivered_checks': checks}, dependencies=(),
    )


def _move(frm: str, to: str | None, *symbols: str) -> Any:
    return state.Move(from_path=frm, to_path=to, symbols=symbols)


def _check(
    name: str, *, kind: str = 'path', paths: Sequence[str] | None = (), pattern: str | None = None,
) -> dict[str, Any]:
    return {
        'name': name, 'kind': kind, 'paths': None if paths is None else list(paths),
        'pattern': pattern, 'expect': 'present',
    }


def _fixture_checks(task_id: int) -> list[Any]:
    tasks = json.loads((FIXTURE_DIR / 'store.json').read_text())['tasks']
    return next(t for t in tasks if t['id'] == task_id)['metadata']['delivered_checks']


def _impacts(moves: Sequence[Any], rows: Sequence[Any]) -> list[tuple[Any, ...]]:
    found = clerk.impact_scan(moves, rows)
    assert isinstance(found, tuple)
    return [(i.task, i.check, i.kind, i.from_path, i.to_path) for i in found]


def test_a_check_naming_the_moved_path_is_one_impact() -> None:
    row = _row(203, [_check('old-thing-defined', paths=[OLD])])
    assert _impacts([_move(OLD, NEW)], [row]) == [(203, 'old-thing-defined', 'path', OLD, NEW)]


def test_a_pattern_containing_a_moved_symbol_is_one_impact() -> None:
    check = _check('old-thing-defined', kind='grep', paths=['src/other.py'],
                   pattern=r'^class OldThing\b')
    impacts = _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, [check])])
    assert impacts == [(203, 'old-thing-defined', 'grep', OLD, NEW)]


def test_the_fixture_check_matching_by_path_and_by_symbol_is_one_impact() -> None:
    impacts = _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, _fixture_checks(203))])
    assert impacts == [(203, 'old-thing-defined', 'grep', OLD, NEW)]


def test_a_check_naming_the_path_is_anchored_by_path_and_one_naming_a_symbol_only_by_symbol() -> None:
    rows = [
        _row(203, _fixture_checks(203)),
        _row(206, [_check('symbol-only', kind='grep', paths=['src/other.py'],
                          pattern=r'^class OldThing\b')]),
    ]
    anchors = [(i.task, i.anchor) for i in clerk.impact_scan([_move(OLD, NEW, 'OldThing')], rows)]
    assert anchors == [(203, 'path'), (206, 'symbol')]


def test_a_deleted_path_is_an_impact_with_no_new_path() -> None:
    row = _row(204, _fixture_checks(204), status='in-progress')
    assert _impacts([_move(REMOVED, None)], [row]) == [
        (204, 'removed-module-present', 'path', REMOVED, None),
    ]


@pytest.mark.parametrize(('check', 'move'), [
    (_check('unrelated-path', paths=['src/other.py']), _move(OLD, NEW)),
    (_check('no-pattern', paths=['src/other.py']), _move(OLD, NEW, 'OldThing')),
    (_check('no-paths', kind='script', paths=None), _move(OLD, NEW)),
    (_check('unrelated-pattern', kind='grep', pattern=r'^def unrelated\('),
     _move(OLD, NEW, 'OldThing')),
    (_check('empty-symbol', kind='grep', pattern=r'^def seam_contract\('), _move(OLD, NEW, '')),
], ids=['unrelated-paths', 'pattern-none', 'paths-none', 'unrelated-pattern', 'empty-symbol'])
def test_a_check_that_names_neither_the_path_nor_a_symbol_is_no_impact(
    check: dict[str, Any], move: Any,
) -> None:
    assert _impacts([move], [_row(203, [check])]) == []


def test_kind_is_passed_through_and_never_filtered() -> None:
    substrate = _check('substrate-check', kind='substrate', paths=[OLD])
    kindless = {'name': 'kindless-check', 'paths': [OLD], 'pattern': None}
    assert _impacts([_move(OLD, NEW)], [_row(203, [substrate, kindless])]) == [
        (203, 'substrate-check', 'substrate', OLD, NEW),
        (203, 'kindless-check', None, OLD, NEW),
    ]


def test_each_matching_check_of_one_task_is_its_own_impact() -> None:
    checks = [
        _check('first', paths=[OLD]),
        {'kind': 'grep', 'paths': [], 'pattern': 'OldThing()'},
        {'name': '', 'kind': 'path', 'paths': [OLD]},
    ]
    assert _impacts([_move(OLD, NEW, 'OldThing')], [_row(203, checks)]) == [
        (203, 'first', 'path', OLD, NEW),
        (203, 'delivered_checks[1]', 'grep', OLD, NEW),
        (203, 'delivered_checks[2]', 'path', OLD, NEW),
    ]


def test_impacts_are_in_move_order_then_task_id_then_check_index() -> None:
    rows = [
        _row(209, [_check('d0', paths=[REMOVED, OLD])]),
        _row(203, [_check('c0', paths=[OLD]), _check('c1', paths=[OLD, REMOVED])]),
    ]
    assert _impacts([_move(REMOVED, None), _move(OLD, NEW)], rows) == [
        (203, 'c1', 'path', REMOVED, None),
        (209, 'd0', 'path', REMOVED, None),
        (203, 'c0', 'path', OLD, NEW),
        (203, 'c1', 'path', OLD, NEW),
        (209, 'd0', 'path', OLD, NEW),
    ]


def test_a_check_entry_that_is_not_an_object_is_skipped_not_raised() -> None:
    rows = [
        _row(203, ['old-thing-defined', None, _check('kept', paths=[OLD])]),
        _row(204, None),
    ]
    assert _impacts([_move(OLD, NEW)], rows) == [(203, 'kept', 'path', OLD, NEW)]


UNREACHABLE = 'http://127.0.0.1:9/mcp'
PROJECT_ROOT = '/review-all-fixture/project'
S1_PRD = 'plans/review-all-fixture-s1-prd.md'
S1_AUTHOR = 'returns/S1.author.json'
S1_DECOMPOSE = 'returns/S1.decompose.json'


def _status(
    status: str, ret: str | None, stream: str = 'S1', prd_path: str = S1_PRD,
) -> dict[str, Any]:
    fields: dict[str, Any] = {'op': 'stream_status', 'stream': stream, 'status': status,
                              'prd_path': prd_path}
    if ret is not None:
        fields['return'] = ret
    return fields


def _action(
    action_id: str, lead_op: str, task: int, target: Any, text: str, needs_human: bool,
    ret: str = S1_DECOMPOSE,
) -> dict[str, Any]:
    return {'op': 'action', 'id': action_id, 'lead_op': lead_op, 'task': task, 'target': target,
            'text': text, 'needs_human': needs_human, 'return': ret}


def _action_result(action_id: str, result_state: str, batch: int | None = 1,
                   detail: str | None = None) -> dict[str, Any]:
    return {'op': 'action_result', 'id': action_id, 'state': result_state, 'batch': batch,
            'detail': detail}


A1 = _action('A1', 'note', 102, None, 'Route callers only after 101 lands.', False)
A2 = _action('A2', 'edge', 102, [106], '102 consumes C2 from S2 α.', False)
A3 = _action('A3', 'cancel', 104, [101], 'subsumed by 101', True)
BASE_LEDGER = (_status('authored', S1_AUTHOR), _status('decomposed', S1_DECOMPOSE), A1, A2, A3)


def _append(home: Any, entries: Sequence[Mapping[str, Any]]) -> None:
    prior = state.read_ledger(home)
    state.append_ledger(home, [
        {'seq': state.next_seq(prior) + offset, 'at': state.now_utc(), 'by': 'test', **fields}
        for offset, fields in enumerate(entries)
    ])


def _seed(
    env: ProgramEnv, ledger: Sequence[Mapping[str, Any]] = BASE_LEDGER,
    returns: Sequence[str] = ('S1.author.json', 'S1.decompose.json'),
) -> Any:
    """A home holding the fixture program, copies of *returns* under returns/, and *ledger*."""
    home = state.Home(env.home)
    (home.root / 'returns').mkdir(parents=True)
    shutil.copyfile(FIXTURE_DIR / 'program.json', home.program_json)
    for name in returns:
        shutil.copyfile(FIXTURE_RETURNS / name, home.root / 'returns' / name)
    _append(home, ledger)
    return home


@pytest.fixture
def env(tmp_path: Path) -> ProgramEnv:
    """Every script run sees an unreachable store: a test that needs one replaces the URL."""
    return ProgramEnv.under(tmp_path, UNREACHABLE)


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def _write(env: ProgramEnv, *args: str) -> subprocess.CompletedProcess[str]:
    return env.run('clerk_batch.py', 'write', '--run', RUN_ID, *args)


def _batch(home: Any, number: int = 1) -> dict[str, Any]:
    return json.loads(home.batch_files(number).actions.read_text())


def _batched(*actions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each seeded action as a batch carries it: its own fields, without the ledger op."""
    return [{key: value for key, value in action.items() if key != 'op'} for action in actions]


def _actions_dir(home: Any) -> list[str]:
    actions = home.root / 'actions'
    return sorted(path.name for path in actions.iterdir()) if actions.exists() else []


def test_write_batches_every_unapplied_action_that_needs_no_human(env: ProgramEnv) -> None:
    home = _seed(env)
    ledger_before = home.ledger.read_bytes()
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    files = home.batch_files(1)
    assert _batch(home) == {
        'run_id': RUN_ID, 'batch': 1, 'result_file': str(files.result), 'accepted': [],
        'actions': _batched(A1, A2),
    }
    assert home.ledger.read_bytes() == ledger_before
    awaiting = [line for line in proc.stdout.splitlines() if line.lstrip().startswith('A3 ')]
    assert len(awaiting) == 1 and '--accept A3' in awaiting[0], proc.stdout
    assert str(files.actions) in proc.stdout and str(files.prompt) in proc.stdout


def test_the_prompt_carries_the_rules_the_paths_and_every_result_key(env: ProgramEnv) -> None:
    home = _seed(env)
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    files = home.batch_files(1)
    prompt = files.prompt.read_text()
    assert clerk.CLERK_RULES in prompt
    header, _, shape = prompt.partition(clerk.CLERK_RULES)
    assert str(files.actions) in header and str(files.result) in header
    assert PROJECT_ROOT in header
    for key in [*clerk.RESULT_LISTS, 'dirtied']:
        assert key in shape, f'the result shape after CLERK_RULES does not name {key!r}'


def test_an_action_with_a_result_is_not_batched_again(env: ProgramEnv) -> None:
    home = _seed(env, [*BASE_LEDGER, _action_result('A1', 'applied')])
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    assert _batch(home)['actions'] == _batched(A2)


def test_accept_batches_a_needs_human_action_and_records_it(env: ProgramEnv) -> None:
    home = _seed(env)
    proc = _write(env, '--accept', 'A3')
    assert proc.returncode == 0, _output(proc)
    batch = _batch(home)
    assert batch['actions'] == _batched(A1, A2, A3)
    assert batch['accepted'] == ['A3']


@pytest.mark.parametrize(('ledger', 'accepted'), [
    (BASE_LEDGER, 'A9'),
    (BASE_LEDGER, 'A1'),
    ((*BASE_LEDGER, _action_result('A3', 'applied')), 'A3'),
], ids=['unknown', 'needs-no-human', 'already-recorded'])
def test_accept_refuses_an_id_that_is_no_unapplied_needs_human_action(
    env: ProgramEnv, ledger: Sequence[Mapping[str, Any]], accepted: str,
) -> None:
    home = _seed(env, ledger)
    proc = _write(env, '--accept', accepted)
    assert proc.returncode == 1, _output(proc)
    assert accepted in proc.stderr
    assert _actions_dir(home) == []


def test_a_second_write_is_a_new_batch_and_leaves_the_first_untouched(env: ProgramEnv) -> None:
    home = _seed(env)
    assert _write(env).returncode == 0
    first = home.batch_files(1).actions.read_bytes()
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    assert _batch(home, 2)['batch'] == 2
    assert _batch(home, 2)['actions'] == _batched(A1, A2)
    assert home.batch_files(1).actions.read_bytes() == first


def test_nothing_unapplied_writes_no_batch(env: ProgramEnv) -> None:
    home = _seed(env, [
        *BASE_LEDGER, _action_result('A1', 'applied'), _action_result('A2', 'skipped', detail='x'),
        _action_result('A3', 'rejected', detail='y'),
    ])
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    assert 'nothing to batch' in proc.stdout
    assert _actions_dir(home) == []


def test_a_home_with_no_program_is_refused_naming_the_home(env: ProgramEnv) -> None:
    proc = _write(env)
    assert proc.returncode == 1, _output(proc)
    assert str(env.home) in proc.stderr
    assert 'record.py init' in proc.stderr


SEAM = 'src/fixture/seam.py'
SEAM_V2 = 'src/fixture/seam_v2.py'
S1_AMEND = 'returns/S1.decompose.amend-1.json'
S2_DECOMPOSE = 'returns/S2.decompose.json'
UTC_STAMP = re.compile(r'\b\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\b')


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


def _tool_calls(store: FakeStore, since: int) -> list[dict[str, Any]]:
    """Each tools/call the store received after its first *since* requests: name and arguments."""
    return [
        entry['body']['params'] for entry in store.requests()[since:]
        if entry['body'].get('method') == 'tools/call'
    ]


def _appended(home: Any, after: int) -> list[dict[str, Any]]:
    """The ledger lines after seq *after*, without their clock stamp."""
    return [
        {key: value for key, value in line.items() if key != 'at'}
        for line in state.read_ledger(home)[after:]
    ]


def _without_text(line: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in line.items() if key != 'text'}


def _repath(
    action_id: str, seq: int, task: int, target: str | None, ret: str, *, needs_human: bool = False,
) -> dict[str, Any]:
    return {'seq': seq, 'by': 'clerk_batch.py', 'op': 'action', 'id': action_id,
            'lead_op': 'repath', 'task': task, 'target': target, 'needs_human': needs_human,
            'return': ret}


def _kept(action_id: str, seq: int, why: str) -> dict[str, Any]:
    return {'seq': seq, 'by': 'clerk_batch.py', **_action_result(action_id, 'kept-for-human', None, why)}


def _scanned_home(env: ProgramEnv, store: FakeStore) -> tuple[Any, subprocess.CompletedProcess[str]]:
    """The base home, scanned once against the fixture store."""
    home = _seed(env)
    proc = _write(dataclasses.replace(env, store_url=store.url), '--scan')
    return home, proc


def test_b4_the_scan_ledgers_one_repath_per_anchored_check_and_a_deleted_path_for_a_human(
    env: ProgramEnv, store: FakeStore,
) -> None:
    since = len(store.requests())
    home, proc = _scanned_home(env, store)
    assert proc.returncode == 0, _output(proc)
    appended = _appended(home, len(BASE_LEDGER))
    assert [_without_text(line) for line in appended] == [
        _repath('A4', 6, 203, NEW, S1_DECOMPOSE),
        _repath('A5', 7, 204, None, S1_DECOMPOSE, needs_human=True),
        _kept('A5', 8, 'check anchors a deleted path'),
        {'seq': 9, 'by': 'clerk_batch.py', 'op': 'scanned', 'return': S1_DECOMPOSE,
         'repaths': ('A4', 'A5')},
    ]
    a4_text, a5_text = appended[0]['text'], appended[1]['text']
    assert all(part in a4_text for part in ('old-thing-defined', OLD, NEW)), a4_text
    assert all(part in a5_text for part in ('removed-module-present', REMOVED)), a5_text
    assert [action['id'] for action in _batch(home)['actions']] == ['A1', 'A2', 'A4']
    calls = _tool_calls(store, since)
    assert calls and all(
        call['name'] == 'find_tasks_by_metadata' and call['arguments']['key'] == 'delivered_checks'
        for call in calls
    ), calls
    assert UTC_STAMP.search(proc.stdout), proc.stdout


def test_a_scanned_return_is_never_scanned_again(env: ProgramEnv, store: FakeStore) -> None:
    home, first = _scanned_home(env, store)
    assert first.returncode == 0, _output(first)
    scanned = len(state.read_ledger(home))
    proc = _write(env, '--scan')
    assert proc.returncode == 0, _output(proc)
    assert _appended(home, scanned) == []


def test_a_return_with_no_moves_is_scanned_with_nothing_and_no_store_read(env: ProgramEnv) -> None:
    home = _seed(env, [
        _status('authored', S1_AUTHOR),
        _status('decomposed', None),
        _status('decomposed', S2_DECOMPOSE, stream='S2', prd_path='plans/review-all-fixture-s2-prd.md'),
    ], returns=('S1.author.json', 'S2.decompose.json'))
    proc = _write(env, '--scan')
    assert proc.returncode == 0, _output(proc)
    assert _appended(home, 3) == [
        {'seq': 4, 'by': 'clerk_batch.py', 'op': 'scanned', 'return': S2_DECOMPOSE, 'repaths': ()},
    ]


def _scanned_amendment(
    env: ProgramEnv, store: FakeStore, move: Mapping[str, Any],
) -> tuple[Any, int, subprocess.CompletedProcess[str]]:
    """The scanned base home, its S1 decompose amended to add *move*, then scanned again; the
    ledger's length between the two scans."""
    home, first = _scanned_home(env, store)
    assert first.returncode == 0, _output(first)
    amended = json.loads((FIXTURE_RETURNS / 'S1.decompose.json').read_text())
    amended['moves'].append(dict(move))
    (home.root / S1_AMEND).write_text(json.dumps(amended))
    _append(home, [_status('decomposed', S1_AMEND)])
    scanned = len(state.read_ledger(home))
    return home, scanned, _write(dataclasses.replace(env, store_url=store.url), '--scan')


def test_an_amended_return_is_scanned_for_its_new_moves_only(
    env: ProgramEnv, store: FakeStore,
) -> None:
    home, scanned, proc = _scanned_amendment(
        env, store, {'from': SEAM, 'to': SEAM_V2, 'symbols': []},
    )
    assert proc.returncode == 0, _output(proc)
    appended = _appended(home, scanned)
    assert [_without_text(line) for line in appended] == [
        _repath('A6', scanned + 1, 202, SEAM_V2, S1_AMEND),
        {'seq': scanned + 2, 'by': 'clerk_batch.py', 'op': 'scanned', 'return': S1_AMEND,
         'repaths': ('A6',)},
    ]
    assert 'seam-contract-defined' in appended[0]['text']


def test_a_check_matching_a_moved_symbol_only_is_kept_for_a_human_and_never_batched(
    env: ProgramEnv, store: FakeStore,
) -> None:
    home, scanned, proc = _scanned_amendment(
        env, store,
        {'from': 'src/fixture/seam_impl.py', 'to': SEAM_V2, 'symbols': ['seam_contract']},
    )
    assert proc.returncode == 0, _output(proc)
    appended = _appended(home, scanned)
    assert [_without_text(line) for line in appended] == [
        _repath('A6', scanned + 1, 202, SEAM_V2, S1_AMEND, needs_human=True),
        _kept('A6', scanned + 2, clerk.SYMBOL_ONLY),
        {'seq': scanned + 3, 'by': 'clerk_batch.py', 'op': 'scanned', 'return': S1_AMEND,
         'repaths': ('A6',)},
    ]
    assert 'seam-contract-defined' in appended[0]['text']
    assert [action['id'] for action in _batch(home, 2)['actions']] == ['A1', 'A2', 'A4']


def test_a_move_to_scan_with_no_store_writes_nothing(env: ProgramEnv) -> None:
    home = _seed(env)
    ledger_before = home.ledger.read_bytes()
    proc = _write(env, '--scan')
    assert proc.returncode == 1, _output(proc)
    assert UNREACHABLE in proc.stderr
    assert home.ledger.read_bytes() == ledger_before
    assert _actions_dir(home) == []


def _refused_before_any_store_read(env: ProgramEnv, home: Any, named: str) -> None:
    """``write --scan`` exits 1 naming *named*, reads no store, and writes nothing."""
    ledger_before = home.ledger.read_bytes()
    proc = _write(env, '--scan')
    assert proc.returncode == 1, _output(proc)
    assert named in proc.stderr and UNREACHABLE not in proc.stderr, proc.stderr
    assert home.ledger.read_bytes() == ledger_before
    assert _actions_dir(home) == []


@pytest.mark.parametrize('move', [
    {'from': OLD, 'to': NEW, 'symbols': 'OldThing'},
    {'from': 7, 'to': NEW, 'symbols': []},
    {'from': OLD, 'to': NEW, 'symbols': ['OldThing', 3]},
], ids=['symbols-a-string', 'from-not-a-string', 'a-symbol-not-a-string'])
def test_a_return_with_a_malformed_move_is_refused_naming_it(
    env: ProgramEnv, move: dict[str, Any],
) -> None:
    home = _seed(env)
    returned = home.root / S1_DECOMPOSE
    data = json.loads(returned.read_text())
    data['moves'] = [move]
    returned.write_text(json.dumps(data))
    _refused_before_any_store_read(env, home, str(returned.resolve()))


def test_a_ledgered_return_outside_the_home_is_refused_naming_it(env: ProgramEnv) -> None:
    escape = '../escape.json'
    home = _seed(env, [_status('authored', S1_AUTHOR), _status('decomposed', escape)])
    shutil.copyfile(FIXTURE_RETURNS / 'S1.decompose.json', home.root / escape)
    _refused_before_any_store_read(env, home, escape)


SIDECAR = 'plans/review-all-fixture-s1-prd.capability-manifest.yaml'
SIDECAR_MD = 'plans/review-all-fixture-s1-prd.capability-manifest.md'


def _ingest(env: ProgramEnv, path: Path) -> subprocess.CompletedProcess[str]:
    return env.run('clerk_batch.py', 'ingest', path)


def _clerk_result(**lists: Any) -> dict[str, Any]:
    return {'applied': [], 'skipped': [], 'rejected': [], 'kept_for_human': [], 'dirtied': [],
            **lists}


def _batched_home(env: ProgramEnv) -> Any:
    """The base home after one write: batch-1 holds A1 and A2."""
    home = _seed(env)
    proc = _write(env)
    assert proc.returncode == 0, _output(proc)
    return home


def _put_result(home: Any, content: Any, number: int = 1) -> Path:
    path = home.batch_files(number).result
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def _recorded(action_id: str, seq: int, result_state: str, detail: str | None = None) -> dict[str, Any]:
    return {'seq': seq, 'by': 'clerk_batch.py', **_action_result(action_id, result_state, 1, detail)}


def test_b13_ingest_records_each_known_id_and_refuses_an_unknown_one(env: ProgramEnv) -> None:
    home = _batched_home(env)
    path = _put_result(home, _clerk_result(
        applied=['A1', 'A99'], skipped=[{'id': 'A2', 'why': 'task 102 is in-progress'}],
        dirtied=[SIDECAR, SIDECAR_MD],
    ))
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert _appended(home, len(BASE_LEDGER)) == [
        _recorded('A1', 6, 'applied'),
        _recorded('A2', 7, 'skipped', 'task 102 is in-progress'),
    ]
    assert 'A99: not an action of batch-1' in proc.stderr
    assert SIDECAR in proc.stdout and SIDECAR_MD in proc.stdout


def test_ingest_refuses_a_ledger_action_the_batch_did_not_carry(env: ProgramEnv) -> None:
    home = _batched_home(env)
    path = _put_result(home, _clerk_result(
        applied=['A1'], kept_for_human=[{'id': 'A3', 'why': 'a cancel needs a ruling'}],
    ))
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert 'A3' in proc.stderr
    assert _appended(home, len(BASE_LEDGER)) == [_recorded('A1', 6, 'applied')]


def test_a_clean_result_is_recorded_and_its_actions_leave_the_next_batch(env: ProgramEnv) -> None:
    home = _batched_home(env)
    path = _put_result(home, _clerk_result(
        applied=['A1'], rejected=[{'id': 'A2', 'why': '106 does not exist'}],
    ))
    proc = _ingest(env, path)
    assert proc.returncode == 0, _output(proc)
    assert _appended(home, len(BASE_LEDGER)) == [
        _recorded('A1', 6, 'applied'), _recorded('A2', 7, 'rejected', '106 does not exist'),
    ]
    again = _write(env)
    assert again.returncode == 0, _output(again)
    assert 'nothing to batch' in again.stdout
    assert not home.batch_files(2).actions.exists()


def test_re_ingesting_a_result_names_each_id_as_already_recorded(env: ProgramEnv) -> None:
    home = _batched_home(env)
    path = _put_result(home, _clerk_result(
        applied=['A1'], skipped=[{'id': 'A2', 'why': 'task 102 is in-progress'}],
    ))
    assert _ingest(env, path).returncode == 0
    ledger_before = home.ledger.read_bytes()
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert 'A1: already recorded' in proc.stderr and 'A2: already recorded' in proc.stderr
    assert home.ledger.read_bytes() == ledger_before


def test_an_id_named_in_two_lists_is_refused_and_the_rest_recorded(env: ProgramEnv) -> None:
    home = _batched_home(env)
    path = _put_result(home, _clerk_result(
        applied=['A1'], rejected=[{'id': 'A1', 'why': 'second thoughts'}],
        skipped=[{'id': 'A2', 'why': 'task 102 is in-progress'}],
    ))
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert 'A1' in proc.stderr
    assert _appended(home, len(BASE_LEDGER)) == [
        _recorded('A2', 6, 'skipped', 'task 102 is in-progress'),
    ]


@pytest.mark.parametrize(('content', 'named'), [
    ('{not json', 'JSON'),
    ({key: value for key, value in _clerk_result().items() if key != 'dirtied'}, 'dirtied'),
    (_clerk_result(notes='extra'), 'notes'),
    (_clerk_result(skipped=[{'id': 'A2'}]), 'skipped[0].why'),
    (_clerk_result(applied=[{'id': 'A1'}]), 'applied[0]'),
], ids=['not-json', 'dirtied-missing', 'extra-key', 'skipped-without-why', 'applied-not-a-string'])
def test_a_malformed_result_is_refused_whole(env: ProgramEnv, content: Any, named: str) -> None:
    home = _batched_home(env)
    path = _put_result(home, content)
    ledger_before = home.ledger.read_bytes()
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert named in proc.stderr
    assert home.ledger.read_bytes() == ledger_before


def test_a_path_that_is_no_batch_result_is_refused_naming_it(env: ProgramEnv) -> None:
    home = _batched_home(env)
    ledger_before = home.ledger.read_bytes()
    for path in (home.batch_files(7).result, home.root / 'result.json'):
        path.write_text(json.dumps(_clerk_result(applied=['A1'])))
        proc = _ingest(env, path)
        assert proc.returncode == 1, _output(proc)
        assert str(path) in proc.stderr
    assert home.ledger.read_bytes() == ledger_before


def test_ingesting_a_result_file_not_yet_written_is_refused_naming_it(env: ProgramEnv) -> None:
    home = _batched_home(env)
    ledger_before = home.ledger.read_bytes()
    path = home.batch_files(1).result
    proc = _ingest(env, path)
    assert proc.returncode == 1, _output(proc)
    assert str(path) in proc.stderr
    assert home.ledger.read_bytes() == ledger_before
