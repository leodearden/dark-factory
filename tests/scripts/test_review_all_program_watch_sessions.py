"""``watch_sessions.py`` and the ``sessions.json`` fields it owns, over a home initialised from the fixture program.

The in-process cases drive ``sessions.py``'s public API. Every other case runs
the watcher as an ``-I -S`` subprocess from a temp cwd whose ``HOME`` is that
cwd, so the registry it reads is ``<cwd>/.claude/fleet/sessions``. Fixture
record dirs and returns are copied there, never edited.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import types
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import FIXTURE_RETURNS, RUN_ID, ProgramEnv, load_program_module
from review_all_program_fixture import FIXTURE_DIR, FakeStore, running_fake_store

SCRIPT = 'watch_sessions.py'
SPAWNED_AFTER = '2026-10-19T09:00:00Z'
S3_TITLE = 'prd:fixture S3'
S2_TITLE = 'prd:fixture S2'
S1_TITLE = 'prd:fixture S1'
Q2_TITLE = 'team:fixture S1 Q2'
FIXTURE_SESSIONS = FIXTURE_DIR / 'sessions'
FIXTURE_SLUGS = types.MappingProxyType({
    'checkpoint-running': 'fixture-s3-author',
    'final-exited': 'fixture-s3-author',
    'finished-idle': 'fixture-s2-author',
    'checkpoint-exited': 'fixture-s1-q2-team',
    'unreadable': 'fixture-s1-author',
})
"""Each fixture record dir's slug, named here because the unreadable record cannot be parsed for its own."""


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


def registry(env: ProgramEnv) -> Path:
    """The registry the watcher reads: its HOME is the cwd, so ``<cwd>/.claude/fleet/sessions``."""
    return env.cwd / '.claude' / 'fleet' / 'sessions'


def put_record(env: ProgramEnv, label: str, slug: str | None = None) -> Path:
    """Copy fixture record dir *label* into the registry, its files stamped now."""
    target = registry(env) / (slug if slug is not None else FIXTURE_SLUGS[label])
    shutil.copytree(
        FIXTURE_SESSIONS / label, target, copy_function=shutil.copyfile, dirs_exist_ok=True,
    )
    return target


def spawned(env: ProgramEnv, title: str, stream: str, pass_: str, item: str | None = None) -> None:
    sessions = _sessions()
    sessions.add(_home(env), sessions.SpawnedSession(
        title=title, stream=stream, pass_=_state().Pass(pass_), item=item,
        spawned_after=SPAWNED_AFTER, record_dir=None,
    ))


def watch(env: ProgramEnv) -> subprocess.CompletedProcess[str]:
    return env.run(SCRIPT, '--run', RUN_ID)


def standing_line(proc: subprocess.CompletedProcess[str], title: str) -> str:
    """The one line of *proc*'s stdout reporting where *title* stands (not its record line)."""
    (line,) = [
        line for line in proc.stdout.splitlines()
        if line.startswith(f'{title}: ') and not line.startswith(f'{title}: record ')
    ]
    return line


def _record_dirs(env: ProgramEnv) -> dict[str, str | None]:
    return {title: entry['record_dir'] for title, entry in _sessions_json(env).items()}


B10_SESSIONS = (
    (S3_TITLE, 'S3', 'author', None, 'checkpoint-running'),
    (S2_TITLE, 'S2', 'author', None, 'finished-idle'),
    (Q2_TITLE, 'S1', 'investigate', 'Q2', 'checkpoint-exited'),
    (S1_TITLE, 'S1', 'author', None, 'unreadable'),
)
B10_INBOX = ('S3.author.json', 'S2.author.json', 'S1.investigate-Q2.json')


@pytest.fixture
def b10(env: ProgramEnv) -> ProgramEnv:
    """Four spawned sessions over the fixture record dirs, with their fixture returns in the inbox."""
    for title, stream, pass_, item, label in B10_SESSIONS:
        spawned(env, title, stream, pass_, item)
        put_record(env, label)
    for name in B10_INBOX:
        env.put_inbox(name)
    return env


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def test_b10_a_pass_resolves_each_readable_record_and_fills_its_record_dir(b10: ProgramEnv) -> None:
    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    resolved = {
        title: str(registry(b10) / FIXTURE_SLUGS[label])
        for title, _, _, _, label in B10_SESSIONS if label != 'unreadable'
    }
    assert _record_dirs(b10) == {**resolved, S1_TITLE: None}
    for title, record_dir in resolved.items():
        assert f'{title}: record {record_dir}' in proc.stdout.splitlines()


def test_b10_a_running_checkpoint_is_archived_and_nothing_is_applied(b10: ProgramEnv) -> None:
    before = b10.snapshot()

    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    archived = b10.home / 'handbacks' / 'S3.author.checkpoint-1.md'
    assert archived.read_bytes() == (FIXTURE_SESSIONS / 'checkpoint-running' / 'result.md').read_bytes()
    assert (b10.home / 'inbox' / 'S3.author.json').read_bytes() == (
        FIXTURE_RETURNS / 'S3.author.json').read_bytes()
    assert not (b10.home / 'returns' / 'S3.author.json').exists()
    assert b10.snapshot().get('ledger.jsonl') == before.get('ledger.jsonl')
    line = standing_line(proc, S3_TITLE)
    assert line.startswith(f'{S3_TITLE}: checkpoint')
    assert 'handbacks/S3.author.checkpoint-1.md' in line


def test_b10_a_terminal_outcome_on_an_idle_record_is_finished_not_exited(b10: ProgramEnv) -> None:
    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: finished, not exited')
    assert (b10.home / 'handbacks' / 'S2.author.checkpoint-1.md').is_file()
    assert (b10.home / 'inbox' / 'S2.author.json').read_bytes() == (
        FIXTURE_RETURNS / 'S2.author.json').read_bytes()


def test_b10_a_checkpoint_on_an_exited_record_ended_at_checkpoint(b10: ProgramEnv) -> None:
    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, Q2_TITLE).startswith(
        f'{Q2_TITLE}: ended at checkpoint — relaunch or resume')
    assert (b10.home / 'handbacks' / 'S1.investigate-Q2.checkpoint-1.md').is_file()
    assert (b10.home / 'inbox' / 'S1.investigate-Q2.json').read_bytes() == (
        FIXTURE_RETURNS / 'S1.investigate-Q2.json').read_bytes()


def test_b10_an_unreadable_record_is_printed_not_swallowed(b10: ProgramEnv) -> None:
    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    assert str(registry(b10) / 'fixture-s1-author' / 'record.json') in standing_line(proc, S1_TITLE)


def test_a_record_older_than_the_spawn_is_not_resolved(env: ProgramEnv) -> None:
    spawned(env, S2_TITLE, 'S2', 'author')
    record_json = put_record(env, 'finished-idle') / 'record.json'
    earlier = datetime(2026, 10, 19, 8, 0, tzinfo=UTC).timestamp()
    os.utime(record_json, (earlier, earlier))

    proc = watch(env)

    assert proc.returncode == 0, _output(proc)
    assert _record_dirs(env) == {S2_TITLE: None}
    assert standing_line(proc, S2_TITLE) == f'{S2_TITLE}: no registry record yet'


def test_a_record_of_another_role_is_not_resolved(env: ProgramEnv) -> None:
    spawned(env, S2_TITLE, 'S2', 'author')
    record_json = put_record(env, 'finished-idle') / 'record.json'
    record = json.loads(record_json.read_text(encoding='utf-8'))
    record_json.write_text(json.dumps({**record, 'role': 'team'}), encoding='utf-8')

    proc = watch(env)

    assert proc.returncode == 0, _output(proc)
    assert _record_dirs(env) == {S2_TITLE: None}
    assert standing_line(proc, S2_TITLE) == f'{S2_TITLE}: no registry record yet'


def test_two_records_of_one_title_leave_it_unresolved_and_name_both(env: ProgramEnv) -> None:
    spawned(env, S2_TITLE, 'S2', 'author')
    put_record(env, 'finished-idle')
    put_record(env, 'finished-idle', slug='fixture-s2-twin')

    proc = watch(env)

    assert proc.returncode == 0, _output(proc)
    assert _record_dirs(env) == {S2_TITLE: None}
    line = standing_line(proc, S2_TITLE)
    assert 'fixture-s2-author' in line and 'fixture-s2-twin' in line
