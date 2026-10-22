"""``skills/review-all/scripts/program/apply_return.py``, run as a ``-I -S`` subprocess outside the repo.

The store is the fixture fake store; each test's home is initialised from the
fixture program, and fixture returns are copied into its inbox.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import FIXTURE_RETURNS, ProgramEnv, load_program_module
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

S1_PRD = 'plans/review-all-fixture-s1-prd.md'
S1_AUTHOR_DIGEST = 'S1 author: 0 filed, 0 held, 2 questions, 0 actions → returns/S1.author.json'


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture
def env(tmp_path: Path, store: FakeStore) -> ProgramEnv:
    env = ProgramEnv.under(tmp_path, store.url)
    proc = env.run('record.py', 'init', FIXTURE_DIR / 'program.json')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return env


def _output(proc: Any) -> str:
    return proc.stdout + proc.stderr


def _ledger(env: ProgramEnv) -> list[dict[str, Any]]:
    """The ledger lines with their clock stamp dropped."""
    return [
        {key: value for key, value in line.items() if key != 'at'}
        for line in state.read_ledger(state.Home(env.home))
    ]


def _inbox(env: ProgramEnv) -> list[str]:
    return sorted(path.name for path in (env.home / 'inbox').iterdir())


def _applied(env: ProgramEnv, fixture_return: str) -> None:
    proc = env.run('apply_return.py', env.put_inbox(fixture_return))
    assert proc.returncode == 0, _output(proc)


def _question(qid: str, **fields: Any) -> dict[str, Any]:
    return {'op': 'question', 'stream': 'S1', 'id': qid, 'return': 'returns/S1.author.json', **fields}


def test_b2_an_author_return_is_moved_and_recorded(env: ProgramEnv) -> None:
    proc = env.run('apply_return.py', env.put_inbox('S1.author.json'))
    assert proc.returncode == 0, _output(proc)
    applied = env.home / 'returns' / 'S1.author.json'
    assert applied.read_bytes() == (FIXTURE_RETURNS / 'S1.author.json').read_bytes()
    assert _inbox(env) == []
    by = {'by': 'apply_return.py'}
    assert _ledger(env) == [
        {'seq': 1, **by, 'op': 'stream_status', 'stream': 'S1', 'status': 'authored',
         'prd_path': S1_PRD, 'return': 'returns/S1.author.json'},
        {'seq': 2, **by, **_question(
            'Q1', question='Does alpha or beta own seam contract C1?',
            options=('alpha owns C1', 'beta owns C1'), recommendation='alpha owns C1',
            needs_human=True, lead_rulable=False, source='author')},
        {'seq': 3, **by, **_question(
            'Q3', question='Name the new module new_name.py or renamed.py?',
            options=('new_name.py', 'renamed.py'), recommendation='new_name.py',
            needs_human=False, lead_rulable=True, source='author')},
        {'seq': 4, **by, 'op': 'xref', 'from': 'S1', 'to': 'S2',
         'text': "S2 consumes C2 once S1's alpha task 101 lands.",
         'return': 'returns/S1.author.json'},
    ]
    assert not (env.home / 'rulings.jsonl').exists()
    digest, *rest = proc.stdout.splitlines()
    assert digest == S1_AUTHOR_DIGEST
    (needs_human,) = rest
    assert 'S1:Q1' in needs_human
    assert 'Does alpha or beta own seam contract C1?' in needs_human
    assert 'alpha owns C1' in needs_human


def test_b3_a_form_rejection_names_the_field_and_writes_nothing(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    env.put_inbox('S1.decompose.notes-over-cap.json')
    before = env.snapshot()
    from_home = dataclasses.replace(env, cwd=env.home)
    proc = from_home.run('apply_return.py', 'inbox/S1.decompose.json')
    assert proc.returncode == 1
    assert 'form: notes:' in _output(proc)
    assert '601' in _output(proc)
    assert env.snapshot() == before


def test_a_return_under_another_streams_name_is_a_form_rejection(env: ProgramEnv) -> None:
    misnamed = env.put_inbox('S1.author.json', as_stem='S2.author')
    before = env.snapshot()
    proc = env.run('apply_return.py', misnamed)
    assert proc.returncode == 1
    form_lines = [line for line in _output(proc).splitlines() if line.startswith('form: <file>:')]
    assert form_lines, _output(proc)
    assert 'S1.author' in form_lines[0] and 'S2.author' in form_lines[0]
    assert env.snapshot() == before


def test_check_reports_the_digest_and_writes_nothing(env: ProgramEnv) -> None:
    inbox_file = env.put_inbox('S1.author.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', '--check', inbox_file)
    assert proc.returncode == 0, _output(proc)
    assert proc.stdout.splitlines()[0] == f'check ok: {S1_AUTHOR_DIGEST}'
    assert env.snapshot() == before


def _form_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith('form: ')]


def test_check_reports_the_same_form_rejection_as_apply(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = env.put_inbox('S1.decompose.notes-over-cap.json')
    before = env.snapshot()
    checked = env.run('apply_return.py', '--check', inbox_file)
    assert checked.returncode == 1
    applied = env.run('apply_return.py', inbox_file)
    assert applied.returncode == 1
    assert _form_lines(_output(checked)) == _form_lines(_output(applied))
    assert any(line.startswith('form: notes:') for line in _form_lines(_output(checked)))
    assert env.snapshot() == before


def test_a_file_outside_the_inbox_is_a_usage_error(env: ProgramEnv) -> None:
    stray = env.cwd / 'S1.author.json'
    shutil.copyfile(FIXTURE_RETURNS / 'S1.author.json', stray)
    proc = env.run('apply_return.py', stray)
    assert proc.returncode == 2
    assert 'inbox' in proc.stderr


def test_an_uninitialised_home_names_record_py_init(env: ProgramEnv) -> None:
    orphan = env.homes_root / 'review-all-other-20261019' / 'inbox' / 'S1.author.json'
    orphan.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_RETURNS / 'S1.author.json', orphan)
    proc = env.run('apply_return.py', orphan)
    assert proc.returncode == 1
    assert 'record.py init' in _output(proc)


def _contradictions(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith('contradiction: ')]


def test_b3_a_store_contradiction_stops_for_the_lead_and_writes_nothing(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = env.put_inbox('S1.decompose.absent-filed.json')
    before = env.snapshot()
    applied = env.run('apply_return.py', inbox_file)
    assert applied.returncode == 1
    (absent,) = [line for line in _contradictions(_output(applied)) if 'filed[0].task_id:' in line]
    assert '999' in absent
    assert 'stop for the lead' in _output(applied).splitlines()[-1]
    assert env.snapshot() == before
    checked = env.run('apply_return.py', '--check', inbox_file)
    assert checked.returncode == 1
    assert _contradictions(_output(checked)) == _contradictions(_output(applied))
    assert env.snapshot() == before


def test_a_decompose_before_its_author_return_is_a_contradiction(env: ProgramEnv) -> None:
    inbox_file = env.put_inbox('S1.decompose.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', inbox_file)
    assert proc.returncode == 1
    assert any(line.startswith('contradiction: pass:') for line in _output(proc).splitlines())
    assert env.snapshot() == before


def _tools_calls(store: FakeStore) -> list[str]:
    return [
        entry['body']['params']['name'] for entry in store.requests()
        if entry['body'].get('method') == 'tools/call'
    ]


def test_b4_a_decompose_return_records_its_status_and_numbered_actions(
    env: ProgramEnv, store: FakeStore,
) -> None:
    _applied(env, 'S1.author.json')
    calls_before = len(_tools_calls(store))
    proc = env.run('apply_return.py', env.put_inbox('S1.decompose.json'))
    assert proc.returncode == 0, _output(proc)
    by = {'by': 'apply_return.py', 'return': 'returns/S1.decompose.json'}
    assert _ledger(env)[4:] == [
        {'seq': 5, 'op': 'stream_status', 'stream': 'S1', 'status': 'decomposed',
         'prd_path': S1_PRD, **by},
        {'seq': 6, 'op': 'action', 'id': 'A1', 'lead_op': 'note', 'task': 102, 'target': None,
         'text': 'Route callers only after 101 lands.', 'needs_human': False, **by},
        {'seq': 7, 'op': 'action', 'id': 'A2', 'lead_op': 'edge', 'task': 102, 'target': (106,),
         'text': '102 consumes C2 from S2 α.', 'needs_human': False, **by},
    ]
    assert proc.stdout.splitlines()[0] == (
        'S1 decompose: 2 filed, 1 held, 0 questions, 2 actions → returns/S1.decompose.json'
    )
    assert set(_tools_calls(store)[calls_before:]) <= {'get_task', 'find_tasks_by_metadata'}
    assert len(_tools_calls(store)) > calls_before


def _inbox_variant(env: ProgramEnv, fixture_return: str, mutate: Callable[[dict[str, Any]], None]) -> Path:
    data = json.loads((FIXTURE_RETURNS / fixture_return).read_text())
    mutate(data)
    target = env.put_inbox(fixture_return)
    target.write_text(json.dumps(data, indent=2))
    return target


def _with_lead_note(data: dict[str, Any]) -> None:
    data['lead_actions'] = [
        {'op': 'note', 'task': 105, 'target': None, 'text': 'Watch C1.', 'needs_human': False},
    ]


def test_action_ids_stay_dense_across_returns(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    _applied(env, 'S1.decompose.json')
    proc = env.run('apply_return.py', _inbox_variant(env, 'S2.author.json', _with_lead_note))
    assert proc.returncode == 0, _output(proc)
    actions = [line for line in _ledger(env) if line['op'] == 'action']
    assert [line['id'] for line in actions] == ['A1', 'A2', 'A3']
    assert actions[-1]['return'] == 'returns/S2.author.json'


def test_a_second_application_of_the_same_return_is_refused(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    _applied(env, 'S1.decompose.json')
    again = env.put_inbox('S1.decompose.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', again)
    assert proc.returncode == 1
    assert 'already applied' in _output(proc)
    assert 'returns/S1.decompose.json' in _output(proc)
    assert env.snapshot() == before
    amended = env.run('apply_return.py', again, '--amend')
    assert amended.returncode == 1
    assert 'already applied' in _output(amended)
    assert env.snapshot() == before


def test_different_content_for_an_applied_return_needs_amend(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    changed = _inbox_variant(env, 'S1.author.json', lambda data: data.update(notes='Revised.'))
    before = env.snapshot()
    proc = env.run('apply_return.py', changed)
    assert proc.returncode == 1
    assert '--amend' in _output(proc)
    assert env.snapshot() == before


def _fixture_store_factory(store: FakeStore) -> Callable[[Any], Any]:
    program_state = load_program_module('state')
    return lambda program: program_state.StoreReader(store.url, program.project_root)


def test_apply_returns_typed_outcomes_for_eta(env: ProgramEnv, store: FakeStore) -> None:
    apply_return = load_program_module('apply_return')
    factory = _fixture_store_factory(store)
    applied = apply_return.apply(env.put_inbox('S1.author.json'), store_factory=factory)
    assert isinstance(applied, apply_return.Applied)
    assert applied.returns_file.exists()
    assert applied.digest == S1_AUTHOR_DIGEST
    refused = apply_return.apply(env.put_inbox('S1.author.json'), store_factory=factory)
    assert isinstance(refused, apply_return.Refused)
    assert refused.kind is apply_return.RefusalKind.APPLICATION
