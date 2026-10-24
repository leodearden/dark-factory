"""``watch_sessions.py`` and the ``sessions.json`` fields it owns, over a home initialised from the fixture program.

The in-process cases drive ``sessions.py``'s public API. Every other case runs
the watcher as an ``-I -S`` subprocess from a temp cwd whose ``HOME`` is that
cwd, so the registry it reads is ``<cwd>/.claude/fleet/sessions``. Fixture
record dirs and returns are copied there, never edited.
"""
from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import ProgramEnv, load_program_module
from review_all_program_fixture import FIXTURE_DIR, FakeStore, running_fake_store

SPAWNED_AFTER = '2026-10-19T09:00:00Z'
S3_TITLE = 'prd:fixture S3'
Q2_TITLE = 'team:fixture S1 Q2'


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


def _state() -> types.ModuleType:
    return load_program_module('state')


def _sessions() -> types.ModuleType:
    return load_program_module('sessions')


def _home(env: ProgramEnv) -> Any:
    return _state().Home(env.home)


def _s3_author() -> Any:
    return _sessions().SpawnedSession(
        title=S3_TITLE, stream='S3', pass_=_state().Pass.AUTHOR, item=None,
        spawned_after=SPAWNED_AFTER, record_dir=None,
    )


def _q2_investigate() -> Any:
    return _sessions().SpawnedSession(
        title=Q2_TITLE, stream='S1', pass_=_state().Pass.INVESTIGATE, item='Q2',
        spawned_after=SPAWNED_AFTER, record_dir=None,
    )


def _sessions_json(env: ProgramEnv) -> dict[str, Any]:
    return json.loads((env.home / 'sessions.json').read_text(encoding='utf-8'))


def test_an_added_session_reads_back_unreported(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)

    sessions.add(home, _s3_author())

    assert sessions.read(home)[S3_TITLE].reported is None
    assert _sessions_json(env)[S3_TITLE]['reported'] is None


def test_update_watched_persists_the_record_dir_and_the_typed_standing(
    env: ProgramEnv, tmp_path: Path,
) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    record_dir = str(tmp_path / 'fleet' / 'sessions' / 'fixture-s3-author')

    sessions.update_watched(home, [dataclasses.replace(
        sessions.read(home)[S3_TITLE], record_dir=record_dir,
        reported=sessions.Standing.CHECKPOINT,
    )])

    watched = sessions.read(home)[S3_TITLE]
    assert watched.record_dir == record_dir
    assert watched.reported is sessions.Standing.CHECKPOINT
    assert _sessions_json(env)[S3_TITLE]['reported'] == 'checkpoint'


def test_a_record_dir_is_set_once(env: ProgramEnv, tmp_path: Path) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    first = dataclasses.replace(
        sessions.read(home)[S3_TITLE], record_dir=str(tmp_path / 'first'),
    )
    sessions.update_watched(home, [first])
    before = home.sessions_json.read_bytes()

    with pytest.raises(_state().StateError) as refused:
        sessions.update_watched(home, [dataclasses.replace(first, record_dir=str(tmp_path / 'other'))])

    assert S3_TITLE in str(refused.value) and 'record_dir' in str(refused.value)
    assert home.sessions_json.read_bytes() == before


def test_an_update_of_an_unrecorded_title_is_refused(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    before = home.sessions_json.read_bytes()

    with pytest.raises(_state().StateError) as refused:
        sessions.update_watched(home, [_q2_investigate()])

    assert Q2_TITLE in str(refused.value)
    assert home.sessions_json.read_bytes() == before


@pytest.mark.parametrize(('field', 'changes'), [
    ('stream', {'stream': 'S2'}),
    ('item', {'item': 'Q3'}),
    ('spawned_after', {'spawned_after': '2026-10-20T09:00:00Z'}),
])
def test_an_update_may_not_change_a_field_the_spawn_recorded(
    env: ProgramEnv, field: str, changes: dict[str, Any],
) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _q2_investigate())
    before = home.sessions_json.read_bytes()

    with pytest.raises(_state().StateError) as refused:
        sessions.update_watched(home, [dataclasses.replace(_q2_investigate(), **changes)])

    assert Q2_TITLE in str(refused.value) and field in str(refused.value)
    assert home.sessions_json.read_bytes() == before


def test_an_update_may_not_change_the_pass(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())

    with pytest.raises(_state().StateError) as refused:
        sessions.update_watched(home, [
            dataclasses.replace(_s3_author(), pass_=_state().Pass.DECOMPOSE),
        ])

    assert S3_TITLE in str(refused.value) and 'pass' in str(refused.value)


def test_a_session_added_after_the_watchers_read_survives_its_update(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    seen = sessions.read(home)[S3_TITLE]
    sessions.add(home, _q2_investigate())

    sessions.update_watched(home, [dataclasses.replace(seen, reported=sessions.Standing.RUNNING)])

    recorded = sessions.read(home)
    assert set(recorded) == {S3_TITLE, Q2_TITLE}
    assert recorded[S3_TITLE].reported is sessions.Standing.RUNNING
    assert recorded[Q2_TITLE] == _q2_investigate()


def test_a_reported_value_outside_the_standings_is_refused_on_read(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    corrupted = _sessions_json(env)
    corrupted[S3_TITLE]['reported'] = 'bogus'
    home.sessions_json.write_text(json.dumps(corrupted), encoding='utf-8')

    with pytest.raises(_state().StateError) as refused:
        sessions.read(home)

    assert f'{S3_TITLE}.reported' in str(refused.value)
    assert all(standing in str(refused.value) for standing in sessions.Standing)


@pytest.mark.parametrize('stamp', ['2026-10-19 10:00', 'yesterday'])
def test_a_spawned_after_that_is_not_an_aware_iso_stamp_is_refused(stamp: str) -> None:
    with pytest.raises(_state().StateError) as refused:
        dataclasses.replace(_s3_author(), spawned_after=stamp)

    assert 'spawned_after' in str(refused.value) and stamp in str(refused.value)
