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
from review_all_program_fixture import (
    FIXTURE_DIR,
    FakeStore,
    load_session_registry,
    running_fake_store,
)

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


def test_an_entry_written_without_reported_reads_as_unreported(env: ProgramEnv) -> None:
    sessions = _sessions()
    home = _home(env)
    sessions.add(home, _s3_author())
    unwatched = _sessions_json(env)
    del unwatched[S3_TITLE]['reported']
    home.sessions_json.write_text(json.dumps(unwatched), encoding='utf-8')

    assert sessions.read(home)[S3_TITLE] == _s3_author()


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


UNCHECKED = 'no registry record yet; records that could not be checked: '


def test_b10_an_unreadable_record_is_printed_not_swallowed(b10: ProgramEnv) -> None:
    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    line = standing_line(proc, S1_TITLE)
    assert line.startswith(f'{S1_TITLE}: {UNCHECKED}')
    assert str(registry(b10) / 'fixture-s1-author' / 'record.json') in line


def test_an_unreadable_record_leaves_an_unmatched_session_awaiting_its_record(env: ProgramEnv) -> None:
    spawned(env, S2_TITLE, 'S2', 'author')
    put_record(env, 'unreadable')

    proc = watch(env)

    assert proc.returncode == 0, _output(proc)
    line = standing_line(proc, S2_TITLE)
    assert line.startswith(f'{S2_TITLE}: {UNCHECKED}')
    assert str(registry(env) / 'fixture-s1-author' / 'record.json') in line
    assert _sessions_json(env)[S2_TITLE]['reported'] == 'awaiting-record'


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


def lines_about(proc: subprocess.CompletedProcess[str], title: str) -> list[str]:
    return [line for line in proc.stdout.splitlines() if line.startswith(f'{title}: ')]


def _ledger(env: ProgramEnv) -> list[Any]:
    return list(_state().read_ledger(_home(env)))


def _watched_once(env: ProgramEnv) -> None:
    proc = watch(env)
    assert proc.returncode == 0, _output(proc)


def _s3_final_after_one_pass(env: ProgramEnv) -> subprocess.CompletedProcess[str]:
    """Pass 1 over the B10 sessions, then S3's record read again once it ended, then pass 2."""
    _watched_once(env)
    put_record(env, 'final-exited')
    return watch(env)


def test_b10_a_final_hand_back_is_archived_and_its_return_applied(b10: ProgramEnv) -> None:
    _watched_once(b10)
    before = _ledger(b10)
    put_record(b10, 'final-exited')

    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    assert (b10.home / 'handbacks' / 'S3.author.md').read_bytes() == (
        FIXTURE_SESSIONS / 'final-exited' / 'result.md').read_bytes()
    assert not (b10.home / 'inbox' / 'S3.author.json').exists()
    assert (b10.home / 'returns' / 'S3.author.json').read_bytes() == (
        FIXTURE_RETURNS / 'S3.author.json').read_bytes()
    after = _ledger(b10)
    assert after[:len(before)] == before
    (added,) = after[len(before):]
    assert {key: added[key] for key in ('op', 'stream', 'status', 'by')} == {
        'op': 'stream_status', 'stream': 'S3', 'status': 'authored', 'by': 'watch_sessions.py',
    }
    assert standing_line(proc, S3_TITLE).startswith(
        f'{S3_TITLE}: final, archived handbacks/S3.author.md')
    (digest,) = [line for line in proc.stdout.splitlines() if line.startswith('S3 author: ')]
    assert digest.endswith('→ returns/S3.author.json')
    assert _sessions_json(b10)[S3_TITLE]['reported'] == 'final'


def test_b10_a_pass_after_final_reports_nothing_new_and_applies_nothing(b10: ProgramEnv) -> None:
    second = _s3_final_after_one_pass(b10)
    assert second.returncode == 0, _output(second)
    after_final = b10.snapshot()

    third = watch(b10)

    assert third.returncode == 0, _output(third)
    assert third.stdout == 'nothing new\n'
    assert b10.snapshot() == after_final


def test_b10_a_second_pass_over_unchanged_sessions_reports_nothing_new(b10: ProgramEnv) -> None:
    _watched_once(b10)
    before = b10.snapshot()

    second = watch(b10)

    assert second.returncode == 0, _output(second)
    assert second.stdout == 'nothing new\n'
    assert b10.snapshot() == before


def test_new_checkpoint_content_is_archived_as_the_next_checkpoint(b10: ProgramEnv) -> None:
    _watched_once(b10)
    result = registry(b10) / 'fixture-s3-author' / 'result.md'
    text = result.read_text(encoding='utf-8') + 'The gamma importer inventory is done.\n'
    result.write_text(text, encoding='utf-8')

    second = watch(b10)

    assert second.returncode == 0, _output(second)
    (line,) = second.stdout.splitlines()
    assert line.startswith(f'{S3_TITLE}: checkpoint, archived handbacks/S3.author.checkpoint-2.md')
    assert (b10.home / 'handbacks' / 'S3.author.checkpoint-2.md').read_text(encoding='utf-8') == text
    assert watch(b10).stdout == 'nothing new\n'


def test_a_standing_change_without_new_content_is_reported_once(b10: ProgramEnv) -> None:
    _watched_once(b10)
    record_json = registry(b10) / 'fixture-s3-author' / 'record.json'
    record = json.loads(record_json.read_text(encoding='utf-8'))
    record_json.write_text(json.dumps({**record, 'status': 'exited', 'exit_code': 0}), encoding='utf-8')

    second = watch(b10)

    assert second.returncode == 0, _output(second)
    (line,) = second.stdout.splitlines()
    assert line.startswith(f'{S3_TITLE}: ended at checkpoint — relaunch or resume')
    assert not (b10.home / 'handbacks' / 'S3.author.checkpoint-2.md').exists()
    assert watch(b10).stdout == 'nothing new\n'


def test_a_final_return_that_fails_validation_is_left_for_the_lead(b10: ProgramEnv) -> None:
    _watched_once(b10)
    inbox = b10.home / 'inbox' / 'S3.author.json'
    fixture_return = json.loads((FIXTURE_RETURNS / 'S3.author.json').read_text(encoding='utf-8'))
    over_cap = 'n' * (_state().NOTES_CAP + 1)
    inbox.write_text(json.dumps({**fixture_return, 'notes': over_cap}), encoding='utf-8')
    refused_return = inbox.read_bytes()
    put_record(b10, 'final-exited')

    proc = watch(b10)

    assert proc.returncode == 1, _output(proc)
    assert 'notes' in proc.stderr
    assert inbox.read_bytes() == refused_return
    assert not (b10.home / 'returns' / 'S3.author.json').exists()
    assert (b10.home / 'handbacks' / 'S3.author.md').is_file()
    after = watch(b10)
    assert after.returncode == 0, _output(after)
    assert after.stdout == 'nothing new\n'


def test_a_final_session_without_a_return_names_its_inbox_path(b10: ProgramEnv) -> None:
    _watched_once(b10)
    inbox = b10.home / 'inbox' / 'S3.author.json'
    inbox.unlink()
    put_record(b10, 'final-exited')

    proc = watch(b10)

    assert proc.returncode == 1, _output(proc)
    (unusable,) = [line for line in proc.stderr.splitlines() if line.startswith(f'{S3_TITLE}: ')]
    assert str(inbox) in unusable and 'absent' in unusable


def test_a_final_return_the_lead_already_applied_is_not_applied_again(b10: ProgramEnv) -> None:
    _watched_once(b10)
    applied = b10.run('apply_return.py', b10.home / 'inbox' / 'S3.author.json')
    assert applied.returncode == 0, _output(applied)
    ledger = _ledger(b10)
    put_record(b10, 'final-exited')

    proc = watch(b10)

    assert proc.returncode == 0, _output(proc)
    assert any(
        'already applied' in line and 'returns/S3.author.json' in line
        for line in lines_about(proc, S3_TITLE)
    )
    assert _ledger(b10) == ledger


REGISTRY = load_session_registry()
EXIT_REQUESTED_AT = '2026-10-19T10:30:00Z'
LIVE_STANDING_LABELS = {
    (True, 'exited'): 'final',
    (False, 'exited'): 'ended at checkpoint — relaunch or resume',
    (True, 'idle'): 'finished, not exited',
    (False, 'idle'): 'checkpoint',
}
"""The label for a hand-back by (its outcome is terminal, its record's status)."""


@pytest.fixture
def s2(env: ProgramEnv) -> ProgramEnv:
    """The S2 author session spawned, with its fixture return in the inbox and no record yet."""
    spawned(env, S2_TITLE, 'S2', 'author')
    env.put_inbox('S2.author.json')
    return env


def put_written_record(env: ProgramEnv, result: str | bytes | None, **record_fields: Any) -> Path:
    """The fixture finished-idle record dir, its record.json fields and its result.md written by the test.

    A *result* of None leaves the dir without a result.md.
    """
    record_dir = put_record(env, 'finished-idle')
    record_json = record_dir / 'record.json'
    record = json.loads(record_json.read_text(encoding='utf-8'))
    record_json.write_text(json.dumps({**record, **record_fields}), encoding='utf-8')
    hand_back = record_dir / 'result.md'
    if result is None:
        hand_back.unlink()
    else:
        hand_back.write_bytes(result.encode('utf-8') if isinstance(result, str) else result)
    return record_dir


def result_md(outcome: str) -> str:
    return f'---\noutcome: {outcome}\nchanged: none\naction_needed: none\n---\nWritten by the test.\n'


def _applied(env: ProgramEnv) -> bool:
    return (env.home / 'returns' / 'S2.author.json').exists()


def _reaped_pid() -> int:
    child = subprocess.Popen(['true'])
    child.wait()
    return child.pid


@pytest.mark.parametrize('status', ['exited', 'idle'])
@pytest.mark.parametrize('outcome', REGISTRY.RESULT_OUTCOMES)
def test_a_hand_back_is_final_iff_its_outcome_is_terminal_and_its_record_exited(
    s2: ProgramEnv, outcome: Any, status: str,
) -> None:
    terminal = outcome in REGISTRY.TERMINAL_RESULT_OUTCOMES
    put_written_record(s2, result_md(str(outcome)), status=status)

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    label = LIVE_STANDING_LABELS[(terminal, status)]
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: {label}')
    assert _applied(s2) is (terminal and status == 'exited')


def _sibling_record(env: ProgramEnv, *, exit_requested_at: str | None, owner_pid: int) -> None:
    put_written_record(
        env, result_md('done'), status='idle', spawn_mode='sibling',
        exit_requested_at=exit_requested_at, claude_owner_pid=owner_pid,
    )


def test_a_sibling_whose_self_exit_owner_is_dead_is_final(s2: ProgramEnv) -> None:
    _sibling_record(s2, exit_requested_at=EXIT_REQUESTED_AT, owner_pid=_reaped_pid())

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: final')
    assert _applied(s2)


def test_a_sibling_whose_self_exit_owner_is_alive_is_finished_not_exited(s2: ProgramEnv) -> None:
    _sibling_record(s2, exit_requested_at=EXIT_REQUESTED_AT, owner_pid=os.getpid())

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: finished, not exited')
    assert not _applied(s2)


def test_a_dead_owner_without_a_self_exit_attempt_is_not_final(s2: ProgramEnv) -> None:
    _sibling_record(s2, exit_requested_at=None, owner_pid=_reaped_pid())

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: finished, not exited')
    assert not _applied(s2)


def test_the_outcome_is_read_by_the_registrys_parser(s2: ProgramEnv) -> None:
    padded = '\n\n---\noutcome:   handed-off  \nchanged: none\n---\nHanded to the lead.\n'
    put_written_record(s2, padded, status='exited', exit_code=0)

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: final')
    assert _applied(s2)


def test_a_hand_back_without_a_contract_outcome_is_a_checkpoint_and_still_archived(
    s2: ProgramEnv,
) -> None:
    put_written_record(s2, result_md('finished'), status='exited', exit_code=0)

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(
        f'{S2_TITLE}: ended at checkpoint — relaunch or resume, '
        'archived handbacks/S2.author.checkpoint-1.md')
    assert not _applied(s2)


def _archived(env: ProgramEnv) -> list[Path]:
    return sorted((env.home / 'handbacks').glob('*'))


@pytest.mark.parametrize(('status', 'label'), [
    ('exited', 'ended without a hand-back — relaunch or resume'),
    ('idle', 'running, no hand-back yet'),
])
def test_a_record_without_a_hand_back_is_running_until_it_ends(
    s2: ProgramEnv, status: str, label: str,
) -> None:
    put_written_record(s2, None, status=status)

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(f'{S2_TITLE}: {label}')
    assert _archived(s2) == []
    assert not _applied(s2)


def test_a_hand_back_that_is_not_utf8_is_unreadable_and_not_archived(s2: ProgramEnv) -> None:
    record_dir = put_written_record(
        s2, b'---\noutcome: done\n---\n\xff\xfe\n', status='exited', exit_code=0,
    )

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE).startswith(
        f'{S2_TITLE}: unreadable: {record_dir / "result.md"}: ')
    assert _archived(s2) == []
    assert not _applied(s2)


def test_a_resolved_record_reaped_later_is_unreadable_and_its_archive_survives(s2: ProgramEnv) -> None:
    record_dir = put_record(s2, 'finished-idle')
    _watched_once(s2)
    shutil.rmtree(record_dir)

    proc = watch(s2)

    assert proc.returncode == 0, _output(proc)
    assert standing_line(proc, S2_TITLE) == (
        f'{S2_TITLE}: unreadable: {record_dir / "record.json"}: absent')
    assert _record_dirs(s2) == {S2_TITLE: str(record_dir)}
    assert _archived(s2) == [s2.home / 'handbacks' / 'S2.author.checkpoint-1.md']
    assert not _applied(s2)


def test_a_corrupt_sessions_json_could_not_run_and_says_why_on_stderr(env: ProgramEnv) -> None:
    sessions_json = env.home / 'sessions.json'
    sessions_json.write_text('{"prd:fixture S2": ', encoding='utf-8')

    proc = watch(env)

    assert proc.returncode == 2, _output(proc)
    assert proc.stdout == ''
    assert proc.stderr.startswith(f'watch_sessions.py: could not run: {sessions_json} is not JSON')
