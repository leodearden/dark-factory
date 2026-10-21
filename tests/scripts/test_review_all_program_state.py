"""Behavioural tests for ``skills/review-all/scripts/program/state.py``.

The library is loaded by path (``review_all_program_fixture.load_state``) and
driven through its public functions over the committed fixture program; no
private name is patched.
"""
from __future__ import annotations

import copy
import dataclasses
import fcntl
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from orchestrator.session_registry import SessionRecord
from review_all_program_fixture import (
    FIXTURE_DIR,
    PROGRAM_DIR,
    REPO_ROOT,
    FakeStore,
    load_state,
    running_fake_store,
)
from sitting.ownership import parse_result_outcome

from shared import task_statuses

state = load_state()

_HOST_PYTHON = Path('/usr/bin/python3')
_INTERPRETERS = [sys.executable] + (
    [str(_HOST_PYTHON)]
    if _HOST_PYTHON.exists()
    and os.path.realpath(_HOST_PYTHON) != os.path.realpath(sys.executable)
    else []
)
_ALLOWED_FIRST_PARTY = {
    '__main__',
    'state',
    'shared',
    'shared.task_statuses',
    'shared.mcp_post',
    'shared.mcp_envelope',
}
_LOADED_NON_STDLIB = """
import json, sys
sys.path.insert(0, {program_dir!r})
import state
print(json.dumps(sorted(
    name for name in sys.modules
    if name.partition('.')[0] not in sys.stdlib_module_names
)))
"""


@pytest.mark.parametrize('python', _INTERPRETERS)
def test_state_imports_under_isolated_interpreter_from_outside_the_repo(
    python: str, tmp_path: Path,
) -> None:
    code = _LOADED_NON_STDLIB.format(program_dir=str(PROGRAM_DIR))
    proc = subprocess.run(
        [python, '-I', '-S', '-c', code],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(json.loads(proc.stdout))
    assert loaded <= _ALLOWED_FIRST_PARTY, sorted(loaded - _ALLOWED_FIRST_PARTY)


def test_wire_constants() -> None:
    assert state.RETURN_SCHEMA == 'review-all-prd-return/1'
    assert state.PROGRAM_SCHEMA == 'review-all-program/1'
    assert state.SEAM_WAIT_KEYS.key == 'x_seam_wait'
    assert state.SEAM_WAIT_KEYS.fields == ('stream', 'contract', 'release_owner')
    assert state.SEAM_RELEASED_KEYS.key == 'x_seam_released'
    assert state.SEAM_RELEASED_KEYS.fields == (
        'by', 'at', 'onto', 'contract_as_delivered',
    )
    assert set(state.LEDGER_OPS) == {
        'stream_status', 'amend_stream', 'amend_program', 'xref', 'action',
        'action_result', 'scanned', 'note', 'commit',
    }
    active = tuple(sorted(str(s) for s in task_statuses.ACTIVE))
    assert active == state.NON_TERMINAL_STATUSES


def _fixture_program_data() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / 'program.json').read_text())


def _program_violation(mutate: Callable[[dict[str, Any]], None]) -> str:
    data = _fixture_program_data()
    mutate(data)
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    return str(caught.value)


def test_parse_program_reads_the_fixture_into_frozen_streams() -> None:
    program = state.parse_program(_fixture_program_data())
    assert program.run_id == 'review-all-fixture-20261019'
    assert tuple(s.id for s in program.streams) == ('S1', 'S2', 'S3')
    assert program.stream_ids == ('S1', 'S2', 'S3')
    assert program.stream('S2').deps == ('S1',)
    assert program.stream('S1').keys == ('fk-0000000000a1', 'fk-0000000000a2')
    assert program.stream('S1').mode is state.StreamMode.AGENT
    with pytest.raises(dataclasses.FrozenInstanceError):
        program.run_id = 'other'


def test_parse_program_refuses_a_key_in_two_streams_naming_both() -> None:
    data = json.loads((FIXTURE_DIR / 'program-dupkey.json').read_text())
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    message = str(caught.value)
    assert 'fk-0000000000a2' in message
    assert 'S1' in message and 'S3' in message


def _set_schema(data: dict[str, Any]) -> None:
    data['schema'] = 'review-all-program/9'


def _dep_on_missing_stream(data: dict[str, Any]) -> None:
    data['streams'][1]['deps'] = ['S9']


def _duplicate_stream_id(data: dict[str, Any]) -> None:
    data['streams'][2]['id'] = 'S1'


def _drop_doc_path(data: dict[str, Any]) -> None:
    del data['doc_path']


@pytest.mark.parametrize(
    ('mutate', 'named'),
    [
        (_set_schema, 'review-all-program/9'),
        (_dep_on_missing_stream, 'streams[1].deps[0]'),
        (_duplicate_stream_id, 'streams[2].id'),
        (_drop_doc_path, 'doc_path'),
    ],
    ids=['schema', 'dep-names-no-stream', 'duplicate-stream-id', 'missing-key'],
)
def test_parse_program_refuses_a_malformed_program_naming_the_value(
    mutate: Callable[[dict[str, Any]], None], named: str,
) -> None:
    assert named in _program_violation(mutate)


def test_parse_program_refuses_a_dependency_cycle_naming_its_streams() -> None:
    def cycle(data: dict[str, Any]) -> None:
        data['streams'][0]['deps'] = ['S2']

    message = _program_violation(cycle)
    assert 'cycle' in message
    assert 'S1' in message and 'S2' in message


@pytest.mark.parametrize('field', ['mode', 'size', 'risk'])
def test_parse_program_refuses_an_unknown_vocabulary_value(field: str) -> None:
    def unknown(data: dict[str, Any]) -> None:
        data['streams'][0][field] = 'bogus'

    message = _program_violation(unknown)
    assert f'streams[0].{field}' in message
    assert 'bogus' in message


def test_parse_program_reports_every_violation_together() -> None:
    data = copy.deepcopy(json.loads((FIXTURE_DIR / 'program-dupkey.json').read_text()))
    data['streams'][1]['risk'] = 'extreme'
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    violations = caught.value.violations
    assert any('fk-0000000000a2' in v for v in violations)
    assert any('streams[1].risk' in v and 'extreme' in v for v in violations)


PRD_PATHS = {
    'S1': 'plans/review-all-fixture-s1-prd.md',
    'S2': 'plans/review-all-fixture-s2-prd.md',
    'S3': 'plans/review-all-fixture-s3-prd.md',
}


def _program() -> Any:
    return state.parse_program(_fixture_program_data())


def _line(seq: int, op: str, **fields: Any) -> dict[str, Any]:
    return {'seq': seq, 'at': '2026-10-19T12:00:00Z', 'by': 'lead', 'op': op, **fields}


def _status(seq: int, stream: str, status: str, **fields: Any) -> dict[str, Any]:
    return _line(seq, 'stream_status', stream=stream, status=status, **fields)


def _amend(seq: int, stream: str, field: str, change: str, value: Any) -> dict[str, Any]:
    return _line(
        seq, 'amend_stream', stream=stream, field=field, change=change, value=value,
        ruling='D1',
    )


def _authored(seq: int, stream: str) -> dict[str, Any]:
    return _status(seq, stream, 'authored', prd_path=PRD_PATHS[stream])


def _row(task_id: int, **metadata: Any) -> Any:
    return state.TaskRow(
        task_id=task_id, title=f'task {task_id}', status='pending', priority='medium',
        updated_at='2026-10-19T12:00:00Z', metadata=metadata, dependencies=(),
    )


def test_stream_views_without_a_ledger_are_the_program_rows() -> None:
    views = state.stream_views(_program(), ())
    assert set(views) == {'S1', 'S2', 'S3'}
    assert views['S1'].keys == ('fk-0000000000a1', 'fk-0000000000a2')
    assert views['S2'].deps == ('S1',)
    assert views['S1'].status is None
    assert views['S1'].prd_path is None
    assert views['S1'].authored is False


def test_stream_status_lines_fold_status_and_keep_the_authored_prd_path() -> None:
    authored = state.stream_views(_program(), [_authored(1, 'S1')])['S1']
    assert authored.status is state.StreamStatus.AUTHORED
    assert authored.prd_path == PRD_PATHS['S1']
    assert authored.authored is True

    committed = state.stream_views(
        _program(), [_authored(1, 'S1'), _status(2, 'S1', 'committed')],
    )['S1']
    assert committed.status is state.StreamStatus.COMMITTED
    assert committed.prd_path == PRD_PATHS['S1']
    assert committed.authored is True


def test_amend_stream_lines_change_the_stream_row() -> None:
    views = state.stream_views(_program(), [
        _amend(1, 'S1', 'keys', 'add', 'fk-0000000000e1'),
        _amend(2, 'S1', 'keys', 'remove', 'fk-0000000000a1'),
        _amend(3, 'S2', 'deps', 'add', 'S3'),
        _amend(4, 'S3', 'mode', 'set', 'spawn'),
    ])
    assert views['S1'].keys == ('fk-0000000000a2', 'fk-0000000000e1')
    assert views['S2'].deps == ('S1', 'S3')
    assert views['S3'].mode is state.StreamMode.SPAWN


@pytest.mark.parametrize(
    ('field', 'change', 'value'),
    [('keys', 'set', ['fk-0000000000e1']), ('mode', 'add', 'spawn')],
)
def test_amend_stream_refuses_a_change_its_field_does_not_take(
    field: str, change: str, value: Any,
) -> None:
    with pytest.raises(state.StateError) as caught:
        state.stream_views(_program(), [_authored(1, 'S1'), _amend(2, 'S1', field, change, value)])
    message = str(caught.value)
    assert 'seq 2' in message
    assert field in message and change in message


def test_waves_are_one_plus_the_latest_upstream_wave() -> None:
    assert dict(state.waves(state.stream_views(_program(), ()))) == {'S1': 1, 'S2': 2, 'S3': 1}
    amended = state.stream_views(_program(), [_amend(1, 'S3', 'deps', 'add', 'S2')])
    assert dict(state.waves(amended)) == {'S1': 1, 'S2': 2, 'S3': 3}


def test_waves_refuse_an_amendment_that_makes_a_cycle() -> None:
    cyclic = state.stream_views(_program(), [_amend(1, 'S1', 'deps', 'add', 'S2')])
    with pytest.raises(state.StateError) as caught:
        state.waves(cyclic)
    assert 'S1' in str(caught.value) and 'S2' in str(caught.value)


def test_stream_of_matches_prd_path_before_finding_keys() -> None:
    views = state.stream_views(_program(), [_authored(1, 'S1'), _authored(2, 'S2')])
    row = _row(104, prd_path=PRD_PATHS['S1'], x_finding_key=['fk-0000000000b1'])
    assert state.stream_of(row, views) == 'S1'


@pytest.mark.parametrize('keys', [['fk-0000000000c1'], 'fk-0000000000c1'])
def test_stream_of_falls_back_to_the_finding_key(keys: Any) -> None:
    views = state.stream_views(_program(), [_authored(1, 'S1')])
    assert state.stream_of(_row(108, x_finding_key=keys), views) == 'S3'


@pytest.mark.parametrize(
    'metadata',
    [{}, {'x_finding_key': ['fk-00000000ffff']}],
    ids=['keyless', 'keys-in-no-stream'],
)
def test_stream_of_puts_an_unmatched_program_task_in_the_unstreamed_bucket(
    metadata: dict[str, Any],
) -> None:
    views = state.stream_views(_program(), ())
    assert state.UNSTREAMED == 'unstreamed'
    assert state.stream_of(_row(107, **metadata), views) == state.UNSTREAMED


def test_stream_of_refuses_keys_that_hit_two_streams() -> None:
    views = state.stream_views(_program(), ())
    row = _row(150, x_finding_key=['fk-0000000000a1', 'fk-0000000000b1'])
    with pytest.raises(state.AmbiguousStream) as caught:
        state.stream_of(row, views)
    message = str(caught.value)
    assert '150' in message
    assert 'S1' in message and 'S2' in message


RUN_ID = 'review-all-fixture-20261019'


def test_run_home_is_the_run_id_under_review_all_home(tmp_path: Path) -> None:
    home = state.run_home(RUN_ID, environ={'REVIEW_ALL_HOME': str(tmp_path)})
    assert home.root == tmp_path / RUN_ID


def test_run_home_defaults_under_the_users_claude_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    assert state.run_home(RUN_ID, environ={}).root == tmp_path / '.claude' / 'review-all' / RUN_ID


def test_run_home_refuses_a_relative_review_all_home() -> None:
    with pytest.raises(state.StateError, match='REVIEW_ALL_HOME'):
        state.run_home(RUN_ID, environ={'REVIEW_ALL_HOME': 'relative/homes'})


@pytest.mark.parametrize('run_id', ['', '.', '..', 'a/b'])
def test_run_home_refuses_a_run_id_that_is_not_one_path_segment(
    run_id: str, tmp_path: Path,
) -> None:
    with pytest.raises(state.StateError) as caught:
        state.run_home(run_id, environ={'REVIEW_ALL_HOME': str(tmp_path)})
    assert repr(run_id) in str(caught.value)


def test_home_names_the_state_layout_files(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    assert home.program_json == tmp_path / 'program.json'
    assert home.rulings == tmp_path / 'rulings.jsonl'
    assert home.ledger == tmp_path / 'ledger.jsonl'
    assert home.sessions_json == tmp_path / 'sessions.json'
    assert home.handoff == tmp_path / 'HANDOFF.md'
    assert home.prds_dir('S1') == tmp_path / 'prds' / 'S1'
    assert home.brief_file('S1', state.Pass.AUTHOR) == tmp_path / 'briefs' / 'S1.author.md'
    assert tuple(home.batch_files(1)) == (
        tmp_path / 'actions' / 'batch-1.json',
        tmp_path / 'actions' / 'batch-1.prompt.md',
        tmp_path / 'actions' / 'batch-1.result.json',
    )


@pytest.mark.parametrize(('method', 'folder'), [('inbox_file', 'inbox'), ('returns_file', 'returns')])
def test_return_files_are_named_by_stream_pass_and_item(
    method: str, folder: str, tmp_path: Path,
) -> None:
    locate = getattr(state.Home(tmp_path), method)
    assert locate('S1', state.Pass.AUTHOR) == tmp_path / folder / 'S1.author.json'
    assert locate('S1', state.Pass.INVESTIGATE, item='Q2') == (
        tmp_path / folder / 'S1.investigate-Q2.json'
    )
    with pytest.raises(state.StateError, match='investigate'):
        locate('S1', state.Pass.INVESTIGATE)
    with pytest.raises(state.StateError, match='Q2'):
        locate('S1', state.Pass.DECOMPOSE, item='Q2')


def test_handback_files_carry_the_checkpoint_number(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    assert home.handback_file('S1', state.Pass.AUTHOR, checkpoint=2) == (
        tmp_path / 'handbacks' / 'S1.author.checkpoint-2.md'
    )
    assert home.handback_file('S1', state.Pass.AUTHOR) == tmp_path / 'handbacks' / 'S1.author.md'


def test_render_paths_and_the_published_snapshot(tmp_path: Path) -> None:
    program = _program()
    home = state.Home(tmp_path)
    assert home.render_doc(program) == tmp_path / 'render' / 'review-all-program-fixture-20261019.md'
    assert home.render_state_dir == tmp_path / 'render' / '.state'
    published = Path('/review-all-fixture/project/plans/review-all-program-fixture-20261019.md')
    assert state.published_doc(program) == published
    assert state.snapshot_dir(program) == published.with_suffix('.state')


def test_fused_memory_url_prefers_the_environment(tmp_path: Path) -> None:
    url = state.fused_memory_url(environ={'FUSED_MEMORY_URL': 'http://x/mcp'}, checkout=tmp_path)
    assert url == 'http://x/mcp'


def test_fused_memory_url_falls_back_to_the_checkouts_mcp_json(tmp_path: Path) -> None:
    (tmp_path / '.mcp.json').write_text(json.dumps(
        {'mcpServers': {'fused-memory': {'type': 'http', 'url': 'http://127.0.0.1:9/mcp'}}}
    ))
    assert state.fused_memory_url(environ={}, checkout=tmp_path) == 'http://127.0.0.1:9/mcp'


@pytest.mark.parametrize('mcp_json', [None, {'mcpServers': {'escalation': {'url': 'http://e/mcp'}}}])
def test_fused_memory_url_without_a_source_names_both(
    mcp_json: dict[str, Any] | None, tmp_path: Path,
) -> None:
    if mcp_json is not None:
        (tmp_path / '.mcp.json').write_text(json.dumps(mcp_json))
    with pytest.raises(state.StateError) as caught:
        state.fused_memory_url(environ={}, checkout=tmp_path)
    assert 'FUSED_MEMORY_URL' in str(caught.value)
    assert str(tmp_path / '.mcp.json') in str(caught.value)


def test_fused_memory_url_defaults_to_this_checkouts_mcp_json() -> None:
    expected = json.loads((REPO_ROOT / '.mcp.json').read_text())['mcpServers']['fused-memory']['url']
    assert state.fused_memory_url(environ={}) == expected


def _ledger_bytes(home: Any) -> bytes:
    return home.ledger.read_bytes()


def test_append_ledger_creates_the_ledger_and_reads_it_back(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    state.append_ledger(home, [_authored(1, 'S1')])
    assert [line['seq'] for line in state.read_ledger(home)] == [1]


@pytest.mark.parametrize(('seq', 'expected', 'got'), [(1, 2, 1), (3, 2, 3)])
def test_append_ledger_refuses_a_seq_that_is_not_the_next(
    seq: int, expected: int, got: int, tmp_path: Path,
) -> None:
    home = state.Home(tmp_path)
    state.append_ledger(home, [_authored(1, 'S1')])
    with pytest.raises(state.LedgerRefused) as caught:
        state.append_ledger(home, [_status(seq, 'S1', 'committed')])
    assert f'expected seq {expected}' in str(caught.value)
    assert f'got {got}' in str(caught.value)


def test_append_ledger_is_all_or_nothing_and_never_rewrites(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    state.append_ledger(home, [_authored(1, 'S1')])
    first = _ledger_bytes(home)
    state.append_ledger(home, [_status(2, 'S1', 'committed'), _authored(3, 'S2')])
    before = _ledger_bytes(home)
    with pytest.raises(state.LedgerRefused, match='expected seq 5'):
        state.append_ledger(home, [_status(4, 'S2', 'committed'), _status(6, 'S2', 'decomposed')])
    assert _ledger_bytes(home) == before
    state.append_ledger(home, [_line(4, 'note', text='wave 1 closed')])
    assert _ledger_bytes(home).startswith(first)
    assert [line['seq'] for line in state.read_ledger(home)] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ('line', 'named'),
    [
        (_line(1, 'rename_stream', stream='S1'), 'rename_stream'),
        (_line(1, 'stream_status', stream='S1'), 'status'),
        (_status(1, 'S1', 'bogus'), 'bogus'),
        (_status(1, 'S1', 'authored', colour='blue'), 'colour'),
    ],
    ids=['unknown-op', 'missing-field', 'outside-enum', 'unknown-field'],
)
def test_append_ledger_refuses_a_line_its_op_does_not_describe(
    line: dict[str, Any], named: str, tmp_path: Path,
) -> None:
    home = state.Home(tmp_path)
    with pytest.raises(state.LedgerRefused, match=named):
        state.append_ledger(home, [line])
    assert not home.ledger.exists() or home.ledger.read_bytes() == b''


def test_read_ledger_refuses_a_hand_edited_gap(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    home.ledger.write_text(
        json.dumps(_authored(1, 'S1')) + '\n' + json.dumps(_status(3, 'S1', 'committed')) + '\n'
    )
    with pytest.raises(state.LedgerRefused, match='line 2'):
        state.read_ledger(home)


def test_append_ledger_waits_for_the_lock(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    state.append_ledger(home, [_authored(1, 'S1')])
    before = _ledger_bytes(home)
    failures: list[BaseException] = []

    def append() -> None:
        try:
            state.append_ledger(home, [_status(2, 'S1', 'committed')])
        except BaseException as exc:
            failures.append(exc)

    with home.ledger.open('a') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        writer = threading.Thread(target=append)
        writer.start()
        writer.join(timeout=0.5)
        assert writer.is_alive()
        assert _ledger_bytes(home) == before
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    writer.join(timeout=10)
    assert not writer.is_alive()
    assert failures == []
    assert [line['seq'] for line in state.read_ledger(home)] == [1, 2]


def _ruling(ruling_id: str, ruled_by: str = 'human', **overrides: Any) -> dict[str, Any]:
    return {
        'id': ruling_id, 'ruled_by': ruled_by, 'date': '2026-10-19', 'stream': 'S1',
        'item': 'Q1', 'question': 'Which module owns the seam?', 'ruling': 'alpha owns it',
        'options': ['alpha', 'beta'], 'consequence': 'beta imports the contract',
        'keys': ['fk-0000000000a1'], 'source': 'sitting 1', 'supersedes': None,
        **overrides,
    }


@pytest.mark.parametrize(
    ('existing', 'ruled_by', 'expected'),
    [
        ([], 'human', 'D1'),
        (['D1'], 'lead', 'L2'),
        (['D1', 'L2'], 'lead (--yes)', 'L3'),
        (['D1', 'L2'], 'human', 'D3'),
    ],
)
def test_next_ruling_id_shares_one_counter_across_prefixes(
    existing: list[str], ruled_by: str, expected: str,
) -> None:
    rulings = [_ruling(rid, 'human' if rid.startswith('D') else 'lead') for rid in existing]
    assert state.next_ruling_id(rulings, state.RuledBy(ruled_by)) == expected


def test_append_rulings_lands_the_next_id_and_reads_it_back(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    state.append_rulings(home, [_ruling('D1'), _ruling('L2', 'lead', options=[], consequence='')])
    state.append_rulings(home, [_ruling('D3', supersedes='D1', answers=['S1:Q1'], reask=True)])
    assert [r['id'] for r in state.read_rulings(home)] == ['D1', 'L2', 'D3']


def test_append_rulings_refuses_an_id_out_of_sequence(tmp_path: Path) -> None:
    home = state.Home(tmp_path)
    with pytest.raises(state.LedgerRefused) as caught:
        state.append_rulings(home, [_ruling('D1'), _ruling('D5')])
    assert "expected id 'D2'" in str(caught.value)
    assert "got 'D5'" in str(caught.value)


@pytest.mark.parametrize(('field', 'empty'), [('options', []), ('consequence', '')])
def test_a_human_ruling_needs_options_and_consequence(
    field: str, empty: Any, tmp_path: Path,
) -> None:
    with pytest.raises(state.LedgerRefused, match=field):
        state.append_rulings(state.Home(tmp_path), [_ruling('D1', **{field: empty})])


def test_append_rulings_refuses_a_supersedes_naming_no_ruling(tmp_path: Path) -> None:
    with pytest.raises(state.LedgerRefused, match='D7'):
        state.append_rulings(state.Home(tmp_path), [_ruling('D1', supersedes='D7')])


def test_append_rulings_refuses_an_answers_entry_that_is_no_item_ref(tmp_path: Path) -> None:
    with pytest.raises(state.LedgerRefused, match='S1Q1'):
        state.append_rulings(state.Home(tmp_path), [_ruling('D1', answers=['S1Q1'])])


def test_item_ref_parses_and_formats_the_stream_question_form() -> None:
    ref = state.ItemRef.parse('S1:Q2')
    assert ref == state.ItemRef('S1', 'Q2')
    assert str(ref) == 'S1:Q2'


@pytest.mark.parametrize('text', ['', 'S1:', ':Q2', 'a:b:c'])
def test_item_ref_refuses_malformed_text(text: str) -> None:
    with pytest.raises(state.StateError) as caught:
        state.ItemRef.parse(text)
    assert repr(text) in str(caught.value)


# The fixture inventory: each B-row's preconditions (plans/review-all-program-tooling-prd.md
# §Boundary-test sketch), read through state.py and the fake store. B1 is pinned by the
# parse_program tests above, B3 by test_review_all_program_return.py's notes-over-cap and
# absent-filed tests, B11 by test_review_all_program_store_reader.py's refusal tests.

PROJECT_ROOT = '/review-all-fixture/project'
RETURNS = FIXTURE_DIR / 'returns'
SESSIONS = FIXTURE_DIR / 'sessions'
AUTHORED = (_authored(1, 'S1'), _authored(2, 'S2'))


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture(scope='module')
def reader(store: FakeStore) -> Any:
    return state.StoreReader(store.url, PROJECT_ROOT)


def _return(name: str, reader: Any, ledger: Any = AUTHORED) -> Any:
    check = state.validate_return(
        (RETURNS / name).read_text(), program=_program(), ledger=ledger, store=reader,
    )
    assert check.ok, check.rejections
    return check.value


def _program_rows(reader: Any) -> dict[int, Any]:
    read = reader.program_tasks(RUN_ID, tuple(PRD_PATHS.values()))
    return {row.task_id: row for row in read.rows}


def _check_paths(row: Any) -> set[str]:
    return {path for check in row.metadata.get('delivered_checks', ()) for path in check['paths']}


def _record() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / 'record.json').read_text())


def test_b2_author_touches_paths_other_tasks_declare_and_anchor(reader: Any) -> None:
    touches = set(_return('S1.author.json', reader).touches)
    program_ids = set(_program_rows(reader))
    declaring = {
        row.task_id for row in reader.tasks_declaring_files(sorted(touches)).rows
        if row.task_id not in program_ids
    }
    anchoring = {
        row.task_id for row in reader.tasks_with_delivered_checks().rows
        if row.task_id not in program_ids and _check_paths(row) & touches
    }
    assert declaring
    assert anchoring - declaring


def test_b4_s1_decompose_holds_gates_and_moves_anchored_paths(reader: Any) -> None:
    decompose = _return('S1.decompose.json', reader)
    [held] = decompose.held
    assert state.SEAM_WAIT_KEYS.key in reader.task(held.task_id).rows[0].metadata
    assert [f.task_id for f in decompose.filed if f.gate_dep] == [101]
    checked = reader.tasks_with_delivered_checks().rows
    [renamed] = [m for m in decompose.moves if m.to_path is not None]
    assert any(
        renamed.from_path in check['paths']
        and any(symbol in (check['pattern'] or '') for symbol in renamed.symbols)
        for row in checked for check in row.metadata['delivered_checks']
    )
    [deleted] = [m for m in decompose.moves if m.to_path is None]
    assert any(deleted.from_path in _check_paths(row) for row in checked)


def test_b5_a_program_task_is_unstreamed(reader: Any) -> None:
    views = state.stream_views(_program(), AUTHORED)
    streamed = {i: state.stream_of(row, views) for i, row in _program_rows(reader).items()}
    assert streamed[107] == state.UNSTREAMED
    assert streamed[108] == 'S3'
    assert streamed[104] == 'S1'


def _gate_facts(reader: Any) -> tuple[Any, dict[int, Any]]:
    return reader.gate(RUN_ID).rows[0], _program_rows(reader)


def test_b7_the_interim_gate_misses_one_high_leaf_and_carries_one_foreign_dep(
    reader: Any,
) -> None:
    gate, program = _gate_facts(reader)
    high = {i for i, row in program.items() if row.priority in {'high', 'critical'}}
    assert high - set(gate.dependencies) == {106}
    assert set(gate.dependencies) - set(program) == {201}
    assert any('prd_path' in r.metadata and 'x_finding_run' not in r.metadata
               for r in program.values())


def test_b7_the_final_gate_depends_on_the_completion_gate_only() -> None:
    with running_fake_store(store=FIXTURE_DIR / 'store-final.json') as running:
        reader = state.StoreReader(running.url, PROJECT_ROOT)
        [completion] = reader.completion_gate(RUN_ID).rows
        gate, program = _gate_facts(reader)
    assert gate.dependencies == (completion.task_id,)
    assert any('prd_path' in r.metadata and 'x_finding_run' not in r.metadata
               for r in program.values())


def test_b7_the_final_store_differs_only_by_the_completion_gate() -> None:
    interim = {t['id']: t for t in json.loads((FIXTURE_DIR / 'store.json').read_text())['tasks']}
    final = {t['id']: t for t in json.loads((FIXTURE_DIR / 'store-final.json').read_text())['tasks']}
    [added] = set(final) - set(interim)
    assert set(interim) <= set(final)
    assert final[added]['metadata']['trigger_chain']['role'] == 'completion_gate'
    changed = {i for i in interim if interim[i] != final[i]}
    assert changed == {110}
    assert {**interim[110], 'dependencies': [added]} == final[110]


def test_b8_handoff_inputs(reader: Any) -> None:
    assert {'launched_from', 'fable_seats', 'yes'} <= set(_record()['method']['extra'])
    held = reader.held_leaves(RUN_ID, tuple(PRD_PATHS.values())).rows
    assert any(row.metadata[state.SEAM_WAIT_KEYS.key]['release_owner'] in _program().stream_ids
               for row in held)
    assert not (RETURNS / 'S3.decompose.json').exists()


def test_b9_brief_inputs(reader: Any) -> None:
    record = _record()
    finding_keys = {finding['key'] for finding in record['findings']}
    assert set(_program().stream('S1').keys) <= finding_keys
    disputed = [f['calibration_disputes_verdict'] for f in record['findings']
                if 'calibration_disputes_verdict' in f]
    assert disputed and all({'to', 'note'} <= set(d) for d in disputed)
    assert record['filings']
    assert any(x.to == 'S2' for x in _return('S1.author.json', reader).xrefs)
    _return('S2.author.json', reader)
    held = reader.held_leaves(RUN_ID, tuple(PRD_PATHS.values())).rows
    assert any(row.metadata[state.SEAM_WAIT_KEYS.key]['release_owner'] == 'S2' for row in held)


SESSION_SCENARIOS = {
    'checkpoint-running': ('running', 'in-progress'),
    'final-exited': ('exited', 'done'),
    'finished-idle': ('idle', 'done'),
    'checkpoint-exited': ('exited', 'in-progress'),
}


def _header_lines(result_md: Path) -> list[str]:
    lines = result_md.read_text().splitlines()
    assert lines[0] == '---'
    return lines[1:lines.index('---', 1)]


@pytest.mark.parametrize('scenario', sorted(SESSION_SCENARIOS))
def test_b10_session_records_and_result_headers(scenario: str) -> None:
    status, outcome = SESSION_SCENARIOS[scenario]
    record = SessionRecord.from_json((SESSIONS / scenario / 'record.json').read_text())
    assert record.status == status
    assert record.title.partition(':')[0] == record.role
    result_md = SESSIONS / scenario / 'result.md'
    if outcome == 'in-progress':
        assert 'outcome: in-progress' in _header_lines(result_md)
    else:
        assert parse_result_outcome(result_md.read_text()) == outcome


def test_b10_the_final_session_is_the_checkpoints_and_its_inbox_return_validates(
    reader: Any,
) -> None:
    def record(scenario: str) -> Any:
        return SessionRecord.from_json((SESSIONS / scenario / 'record.json').read_text())

    running, final = record('checkpoint-running'), record('final-exited')
    assert running.session_slug == final.session_slug
    assert running.title == final.title == 'prd:fixture S3'
    assert 'inbox/S3.author.json' in (SESSIONS / 'final-exited' / 'result.md').read_text()
    _return('S3.author.json', reader)
    with pytest.raises(json.JSONDecodeError):
        SessionRecord.from_json((SESSIONS / 'unreadable' / 'record.json').read_text())


def test_b16_the_sitting_inputs(reader: Any) -> None:
    author = json.loads((RETURNS / 'S1.author.json').read_text())
    amend = json.loads((RETURNS / 'S1.author.amend.json').read_text())
    added = [q for q in amend['open_questions'] if q not in author['open_questions']]
    assert len(added) == 1 and added[0]['source'] == 'critic'
    assert {k: v for k, v in amend.items() if k != 'open_questions'} == {
        k: v for k, v in author.items() if k != 'open_questions'
    }
    assert any(q.lead_rulable for q in _return('S1.author.json', reader).open_questions)
    assert _return('S1.investigate-Q2.json', reader).item == added[0]['id']
