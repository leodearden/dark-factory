"""``skills/review-all/scripts/program/apply_return.py``, run as a ``-I -S`` subprocess outside the repo.

The store is the fixture fake store; each test's home is initialised from the
fixture program, and fixture returns are copied into its inbox.
"""
from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import FIXTURE_RETURNS, ProgramEnv
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

S1_PRD = 'plans/review-all-fixture-s1-prd.md'


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
    assert digest == 'S1 author: 0 filed, 0 held, 2 questions, 0 actions → returns/S1.author.json'
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
    assert proc.stdout.splitlines()[0] == (
        'check ok: S1 author: 0 filed, 0 held, 2 questions, 0 actions → returns/S1.author.json'
    )
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
