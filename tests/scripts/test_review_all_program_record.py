"""``skills/review-all/scripts/program/record.py``, run as a ``-I -S`` subprocess outside the repo."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import ProgramEnv
from review_all_program_fixture import FIXTURE_DIR, FakeStore, running_fake_store

PROGRAM_JSON = FIXTURE_DIR / 'program.json'


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture
def env(tmp_path: Path, store: FakeStore) -> ProgramEnv:
    return ProgramEnv.under(tmp_path, store.url)


def _output(proc: Any) -> str:
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
