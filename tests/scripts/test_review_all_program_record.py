"""``skills/review-all/scripts/program/record.py``, run as a ``-I -S`` subprocess outside the repo."""
from __future__ import annotations

import fcntl
import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import RUN_ID, ProgramEnv, load_program_module, wait_until_blocked_on
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

PROGRAM_JSON = FIXTURE_DIR / 'program.json'


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture
def env(tmp_path: Path, store: FakeStore) -> ProgramEnv:
    return ProgramEnv.under(tmp_path, store.url)


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def _program_copy(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    data = json.loads(PROGRAM_JSON.read_text())
    mutate(data)
    path = tmp_path / 'program-variant.json'
    path.write_text(json.dumps(data, indent=2))
    return path


def test_init_copies_the_program_into_a_new_home(env: ProgramEnv) -> None:
    proc = env.run('record.py', 'init', PROGRAM_JSON)
    assert proc.returncode == 0, _output(proc)
    assert (env.home / 'program.json').read_bytes() == PROGRAM_JSON.read_bytes()
    assert (env.home / 'inbox').is_dir()
    assert str(env.home) in proc.stdout


def test_init_refuses_a_key_in_two_streams_naming_both(env: ProgramEnv) -> None:
    proc = env.run('record.py', 'init', FIXTURE_DIR / 'program-dupkey.json')
    assert proc.returncode == 1
    output = _output(proc)
    assert 'fk-0000000000a2' in output
    assert 'S1' in output and 'S3' in output
    assert not (env.home / 'program.json').exists()


def _stream(index: int, field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        data['streams'][index][field] = value
    return mutate


@pytest.mark.parametrize(
    ('mutate', 'named'),
    [
        (_stream(1, 'deps', ['S1', 'S9']), 'S9'),
        (_stream(0, 'deps', ['S2']), 'cycle'),
        (_stream(0, 'mode', 'hybrid'), 'hybrid'),
        (_stream(0, 'size', 'XL'), 'XL'),
        (_stream(0, 'risk', 'extreme'), 'extreme'),
    ],
    ids=['unknown-dep', 'cycle', 'mode', 'size', 'risk'],
)
def test_init_refuses_an_invalid_program_naming_the_violation(
    mutate: Callable[[dict[str, Any]], None], named: str, env: ProgramEnv, tmp_path: Path,
) -> None:
    proc = env.run('record.py', 'init', _program_copy(tmp_path, mutate))
    assert proc.returncode == 1
    assert named in _output(proc)
    assert env.snapshot() == {}


def test_init_never_overwrites_an_existing_home(env: ProgramEnv, tmp_path: Path) -> None:
    assert env.run('record.py', 'init', PROGRAM_JSON).returncode == 0
    before = (env.home / 'program.json').read_bytes()
    variant = _program_copy(tmp_path, _stream(0, 'remedy', 'A different remedy.'))
    proc = env.run('record.py', 'init', variant)
    assert proc.returncode == 1
    assert 'never overwrites' in _output(proc)
    assert (env.home / 'program.json').read_bytes() == before


@pytest.fixture
def home(env: ProgramEnv) -> ProgramEnv:
    """An initialised home."""
    proc = env.run('record.py', 'init', PROGRAM_JSON)
    assert proc.returncode == 0, _output(proc)
    return env


_Q1 = {
    'id': 'Q1', 'question': 'Does alpha or beta own seam contract C1?',
    'options': ['alpha owns C1', 'beta owns C1'], 'recommendation': 'alpha owns C1',
    'needs_human': True, 'lead_rulable': False, 'source': 'author',
}
_Q2 = {
    'id': 'Q2', 'question': 'Should the alpha shim be retired before S2 releases C1?',
    'options': ['retire now', 'hold until S2 releases C1'],
    'recommendation': 'hold until S2 releases C1',
    'needs_human': True, 'lead_rulable': False, 'source': 'critic',
}
_Q3 = {
    'id': 'Q3', 'question': 'Name the new module new_name.py or renamed.py?',
    'options': ['new_name.py', 'renamed.py'], 'recommendation': 'new_name.py',
    'needs_human': False, 'lead_rulable': True, 'source': 'author',
}


def _seed_questions(env: ProgramEnv, *entries: dict[str, Any]) -> None:
    home = state.Home(env.home)
    first = state.next_seq(state.read_ledger(home))
    state.append_ledger(home, [
        {'seq': first + n, 'at': '2026-10-19T12:00:00Z', 'by': 'apply_return.py',
         'op': 'question', 'stream': 'S1', 'return': 'returns/S1.author.json', **entry}
        for n, entry in enumerate(entries)
    ])


@pytest.fixture
def asked(home: ProgramEnv) -> ProgramEnv:
    """An initialised home whose ledger records S1:Q1 (needs_human), S1:Q2 (critic), S1:Q3 (lead_rulable)."""
    _seed_questions(home, _Q1, _Q2, _Q3)
    return home


def _rule(env: ProgramEnv, *args: str, by: str = 'human') -> subprocess.CompletedProcess[str]:
    return env.run(
        'record.py', 'rule', '--run', RUN_ID, '--by', by,
        '--question', 'Which module owns the seam?', '--ruling', 'alpha owns it',
        '--source', 'sitting 1', *args,
    )


_HUMAN = ('--option', 'alpha', '--option', 'beta', '--consequence', 'beta imports the contract')


def _recorded_bytes(env: ProgramEnv) -> dict[str, bytes]:
    """The home's non-empty files: a refused append may leave an empty JSONL, which records nothing."""
    return {path: data for path, data in env.snapshot().items() if data}


def _rulings(env: ProgramEnv) -> tuple[Any, ...]:
    return state.read_rulings(state.Home(env.home))


def _pending(env: ProgramEnv) -> list[str]:
    questions = load_program_module('questions')
    home = state.Home(env.home)
    recorded = questions.recorded_questions(state.read_ledger(home))
    return [str(ref) for ref in questions.pending_items(recorded, state.read_rulings(home))]


def test_rule_mints_one_id_namespace_across_human_and_lead(home: ProgramEnv) -> None:
    human = _rule(home, *_HUMAN, '--key', 'fk-0000000000a1', '--stream', 'S1')
    assert human.returncode == 0, _output(human)
    assert 'D1' in human.stdout
    lead = _rule(home, by='lead')
    assert lead.returncode == 0, _output(lead)
    assert 'L2' in lead.stdout
    first, second = _rulings(home)
    assert dict(first) | {'date': None} == {
        'id': 'D1', 'ruled_by': 'human', 'date': None, 'stream': 'S1', 'item': None,
        'question': 'Which module owns the seam?', 'ruling': 'alpha owns it',
        'options': ('alpha', 'beta'), 'consequence': 'beta imports the contract',
        'keys': ('fk-0000000000a1',), 'source': 'sitting 1', 'supersedes': None,
    }
    assert (second['id'], second['ruled_by'], second['supersedes']) == ('L2', 'lead', None)


def test_b12_a_superseding_ruling_leaves_the_superseded_line_untouched(home: ProgramEnv) -> None:
    assert _rule(home, *_HUMAN).returncode == 0
    assert _rule(home, by='lead').returncode == 0
    d1_line = (home.home / 'rulings.jsonl').read_bytes().splitlines()[0]
    proc = _rule(home, *_HUMAN, '--supersedes', 'D1')
    assert proc.returncode == 0, _output(proc)
    assert 'D3' in proc.stdout
    assert (home.home / 'rulings.jsonl').read_bytes().splitlines()[0] == d1_line
    assert _rulings(home)[2]['supersedes'] == 'D1'


@pytest.mark.parametrize(
    ('args', 'named'),
    [
        (('--consequence', 'beta imports the contract'), 'options'),
        (('--option', 'alpha', '--option', 'beta', '--consequence', ''), 'consequence'),
        ((*_HUMAN, '--supersedes', 'D9'), 'D9'),
        ((*_HUMAN, '--answers', 'S1:Q7'), 'S1:Q7'),
        ((*_HUMAN, '--answers', 'S1Q1'), 'S1Q1'),
        ((*_HUMAN, '--reask'), '--reask'),
        ((*_HUMAN, '--stream', 'S9'), 'S9'),
    ],
    ids=['no-options', 'empty-consequence', 'supersedes-unknown', 'answers-unrecorded',
         'answers-malformed', 'reask-without-answers', 'unknown-stream'],
)
def test_rule_refuses_and_writes_nothing(
    args: tuple[str, ...], named: str, asked: ProgramEnv,
) -> None:
    before = _recorded_bytes(asked)
    proc = _rule(asked, *args)
    assert proc.returncode == 1
    assert named in _output(proc)
    assert _recorded_bytes(asked) == before


def _line_naming(text: str, item: str) -> str:
    (line,) = [line for line in text.splitlines() if item in line]
    return line


def test_b16_answers_close_an_item_unless_the_ruling_reasks(asked: ProgramEnv) -> None:
    assert _pending(asked) == ['S1:Q1', 'S1:Q2', 'S1:Q3']
    closed = _rule(asked, *_HUMAN, '--answers', 'S1:Q1')
    assert closed.returncode == 0, _output(closed)
    assert 'closed' in _line_naming(closed.stdout, 'S1:Q1')
    reasked = _rule(asked, *_HUMAN, '--answers', 'S1:Q2', '--reask')
    assert reasked.returncode == 0, _output(reasked)
    assert 'stays pending' in _line_naming(reasked.stdout, 'S1:Q2')
    lead = _rule(asked, '--answers', 'S1:Q3', by='lead')
    assert lead.returncode == 0, _output(lead)
    assert 'closed' in _line_naming(lead.stdout, 'S1:Q3')
    assert _pending(asked) == ['S1:Q2']
    assert _rule(asked, *_HUMAN, '--answers', 'S1:Q2').returncode == 0
    assert _pending(asked) == []


def test_recorded_questions_keep_the_latest_line_per_item(home: ProgramEnv) -> None:
    _seed_questions(home, _Q1, _Q3, {**_Q1, 'recommendation': 'beta owns C1'})
    recorded = load_program_module('questions').recorded_questions(state.read_ledger(state.Home(home.home)))
    assert [str(ref) for ref in recorded] == ['S1:Q1', 'S1:Q3']
    assert [q.recommendation for q in recorded.values()] == ['beta owns C1', 'new_name.py']


def test_rule_waits_for_the_home_writer_lock(home: ProgramEnv) -> None:
    program_json = home.home / 'program.json'
    with program_json.open() as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        writer = home.start(
            'record.py', 'rule', '--run', RUN_ID, '--by', 'human', '--question', 'q?',
            '--ruling', 'r', '--source', 'sitting 1', *_HUMAN,
        )
        try:
            wait_until_blocked_on(program_json, writer)
            assert not (home.home / 'rulings.jsonl').exists()
        finally:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    stdout, stderr = writer.communicate(timeout=60)
    assert writer.returncode == 0, stdout + stderr
    assert [r['id'] for r in _rulings(home)] == ['D1']
