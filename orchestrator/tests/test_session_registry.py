"""Tests for orchestrator.session_registry module.

Covers: schema/contract round-trip (SCHEMA_VERSION, Status enum, SessionRecord
to/from dict/json); slug sanitization + path/transcript encoding; single-writer
atomic write/read/update; TTL/pid stale-record reaper matrix; CLI subcommands
+ fail-soft; and the G5 two-way boundary test (write by a real spawn -> refresh
by a simulated hook -> reap), exercised jointly with
tests/scripts/test_spawn_claude.py's bash-level harness.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides: object) -> sr.SessionRecord:
    """Build a fully-populated SessionRecord for round-trip/identity tests.

    Every field is given a concrete, distinguishable value so a round-trip
    test can catch a field being dropped/mis-typed; ``overrides`` lets a
    test tweak just the field(s) it cares about.
    """
    # Declared as a bare `dict` (not `dict[str, object]`) so pyright treats it
    # as dict[Unknown, Unknown] at the **fields unpack below -- mirrors
    # test_merge_types_invariants.py's _real_kwargs/_decided_kwargs/_entry_kwargs,
    # the established idiom in this suite for a kwargs-builder helper whose
    # result is unpacked into a strictly-typed constructor.
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.LAUNCHING,
        'title': 'unblock:df#2085 routing-mechanism',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'prompt': '/unblock 2085',
        'cwd': '/home/leo/src/dark-factory',
        'launcher_pid': 4242,
        'start_ts': '2026-07-07T00:00:00+00:00',
        'exit_code': None,
        'result_file': None,
        'transcript_path': '~/.claude/projects/-home-leo-src-dark-factory',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


# A pid virtually guaranteed dead on any host (matches the convention already
# established by orchestrator.harness._pid_alive's own test suite: 2**31 - 1,
# orchestrator/tests/test_reconcile_stranded.py:34).
_DEAD_PID = 2**31 - 1

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)


def _set_mtime(path: Path, now: datetime, age: timedelta) -> None:
    """Backdate *path*'s mtime to ``now - age`` (reap_stale_records' heartbeat clock)."""
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


# ---------------------------------------------------------------------------
# Step-1: schema / contract
# ---------------------------------------------------------------------------


def test_schema_version_is_int() -> None:
    assert isinstance(sr.SCHEMA_VERSION, int)


def test_status_enum_has_exact_six_wire_values() -> None:
    assert issubclass(sr.Status, str)
    values = {member.value for member in sr.Status}
    assert values == {
        'launching',
        'running',
        'awaiting-input',
        'idle',
        'exited',
        'failed-to-start',
    }
    assert len(list(sr.Status)) == 6
    # Spot-check the documented wire-value <-> member mapping.
    assert sr.Status('launching') is sr.Status.LAUNCHING
    assert sr.Status('running') is sr.Status.RUNNING
    assert sr.Status('awaiting-input') is sr.Status.AWAITING_INPUT
    assert sr.Status('idle') is sr.Status.IDLE
    assert sr.Status('exited') is sr.Status.EXITED
    assert sr.Status('failed-to-start') is sr.Status.FAILED_TO_START


def test_session_record_schema_version_defaults_to_schema_version() -> None:
    r = _make_record()
    assert r.schema_version == sr.SCHEMA_VERSION


def test_session_record_dict_round_trip_is_lossless() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_dict(r.to_dict()) == r


def test_session_record_json_round_trip_is_lossless() -> None:
    r = _make_record()
    assert sr.SessionRecord.from_json(r.to_json()) == r


def test_session_record_json_round_trip_with_null_fields() -> None:
    """exit_code/result_file/task_id/escalation_id may legitimately be None
    (a freshly-launched record has no exit code yet); the round trip must
    preserve that, not coerce it to a string or drop the key.
    """
    r = _make_record(
        task_id=None,
        escalation_id=None,
        exit_code=None,
        result_file=None,
        transcript_path=None,
    )
    assert sr.SessionRecord.from_json(r.to_json()) == r


# ---------------------------------------------------------------------------
# Step-3: identity, paths, transcript encoding
# ---------------------------------------------------------------------------


def test_build_session_slug_joins_role_project_task_pid() -> None:
    assert sr.build_session_slug('unblock', 'df', '2085', 4242) == 'unblock-df-2085-4242'


@pytest.mark.parametrize('task_id', [None, ''])
def test_build_session_slug_omits_task_segment_when_absent(task_id: str | None) -> None:
    assert sr.build_session_slug('unblock', 'df', task_id, 4242) == 'unblock-df-4242'


def test_build_session_slug_sanitizes_special_chars() -> None:
    slug = sr.build_session_slug('un block', 'df/prod', '20#85', 4242)
    assert slug == 'un-block-df-prod-20-85-4242'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


def test_fleet_root_defaults_to_dot_claude_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('CLAUDE_FLEET_ROOT', raising=False)
    monkeypatch.setenv('HOME', '/home/fakeuser')
    assert sr.fleet_root() == Path('/home/fakeuser/.claude/fleet')


def test_fleet_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    assert sr.fleet_root() == tmp_path


def test_fleet_root_explicit_root_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', '/should/not/be/used')
    other = tmp_path / 'other'
    assert sr.fleet_root(root=other) == other


def test_record_path_for_slug(tmp_path: Path) -> None:
    path = sr.record_path_for_slug('unblock-df-2085-4242', root=tmp_path)
    assert path == tmp_path / 'sessions' / 'unblock-df-2085-4242' / 'record.json'


def test_transcript_path_for_cwd_encodes_slash() -> None:
    assert (
        sr.transcript_path_for_cwd('/home/leo/src/dark-factory')
        == '~/.claude/projects/-home-leo-src-dark-factory'
    )


def test_transcript_path_for_cwd_encodes_dot() -> None:
    # Regression fixture confirmed against a real ~/.claude/projects/ dir
    # (plans/session-attention-rail-prd.md §3): '.' -> '-' too, so a cwd
    # component starting with '.' yields a doubled '--' (one dash from the
    # preceding '/', one from the '.').
    assert (
        sr.transcript_path_for_cwd('/home/leo/.openclaw-workspace')
        == '~/.claude/projects/-home-leo--openclaw-workspace'
    )


# ---------------------------------------------------------------------------
# Step-5: single-writer atomic write / read / update
# ---------------------------------------------------------------------------


def test_write_record_creates_record_json_matching_the_record(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    assert path.is_file()
    assert sr.SessionRecord.from_json(path.read_text()) == r


def test_write_record_leaves_no_leftover_tmp_file(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    slug_dir = sr.record_path_for_slug(r.session_slug, root=tmp_path).parent
    leftovers = [p for p in slug_dir.iterdir() if p.suffix == '.tmp']
    assert leftovers == []


def test_read_record_returns_equal_record(tmp_path: Path) -> None:
    r = _make_record()
    sr.write_record(r, root=tmp_path)
    assert sr.read_record(r.session_slug, root=tmp_path) == r


def test_update_status_mutates_status_and_exit_code_in_place(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, exit_code=None)
    sr.write_record(r, root=tmp_path)

    sr.update_status(r.session_slug, root=tmp_path, status=sr.Status.EXITED, exit_code=3)

    reread = sr.read_record(r.session_slug, root=tmp_path)
    assert reread.status == sr.Status.EXITED
    assert reread.exit_code == 3
    # Every other field survives the read-modify-write untouched.
    assert reread.session_slug == r.session_slug
    assert reread.title == r.title


def test_refresh_record_updates_existing_record_under_same_key(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING)
    sr.write_record(r, root=tmp_path)

    sr.refresh_record(r.session_slug, root=tmp_path, status=sr.Status.RUNNING)

    path_before = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    reread = sr.read_record(r.session_slug, root=tmp_path)
    assert reread.status == sr.Status.RUNNING
    assert sr.record_path_for_slug(r.session_slug, root=tmp_path) == path_before


# ---------------------------------------------------------------------------
# Step-7: TTL / pid stale-record reaper matrix
# ---------------------------------------------------------------------------


def test_ttl_constants_are_module_level_timedeltas() -> None:
    assert timedelta(hours=24) == sr.TERMINAL_TTL
    assert timedelta(hours=1) == sr.NON_TERMINAL_HEARTBEAT_TTL


def test_reap_removes_terminal_record_past_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.EXITED, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.TERMINAL_TTL + timedelta(hours=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'terminal_ttl'
    assert reaped[0].path == record_path.parent
    assert not record_path.parent.exists()


def test_reap_keeps_terminal_record_within_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.EXITED, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.TERMINAL_TTL - timedelta(hours=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_non_terminal_dead_pid_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'stale_pid'
    assert not record_path.parent.exists()


def test_reap_keeps_non_terminal_live_pid_regardless_of_age(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.RUNNING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, timedelta(days=30))  # past both TTLs

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_keeps_non_terminal_dead_pid_within_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_corrupt_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.write_text('{not valid json')
    _set_mtime(record_path, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_removes_missing_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    """A slug dir with no record.json at all is still reaped by path identity."""
    r = _make_record(status=sr.Status.LAUNCHING, launcher_pid=os.getpid())
    sr.write_record(r, root=tmp_path)
    record_path = sr.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.unlink()
    _set_mtime(record_path.parent, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    """The reaper's whole point is sweeping many records in one pass; pin that
    a kept/terminal_ttl/stale_pid/corrupt mix in a single sessions dir reaps
    exactly the expected subset in one call and leaves survivors untouched.
    """
    kept_running = _make_record(
        session_slug='kept-running', status=sr.Status.RUNNING, launcher_pid=os.getpid()
    )
    kept_recent_terminal = _make_record(
        session_slug='kept-recent-terminal', status=sr.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_terminal = _make_record(
        session_slug='reap-terminal', status=sr.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_stale = _make_record(
        session_slug='reap-stale-pid', status=sr.Status.LAUNCHING, launcher_pid=_DEAD_PID
    )

    for r in (kept_running, kept_recent_terminal, reap_terminal, reap_stale):
        sr.write_record(r, root=tmp_path)

    _set_mtime(
        sr.record_path_for_slug(kept_running.session_slug, root=tmp_path),
        _NOW,
        timedelta(days=30),  # long-lived but a live pid -> kept regardless of age
    )
    _set_mtime(
        sr.record_path_for_slug(kept_recent_terminal.session_slug, root=tmp_path),
        _NOW,
        sr.TERMINAL_TTL - timedelta(hours=1),
    )
    _set_mtime(
        sr.record_path_for_slug(reap_terminal.session_slug, root=tmp_path),
        _NOW,
        sr.TERMINAL_TTL + timedelta(hours=1),
    )
    _set_mtime(
        sr.record_path_for_slug(reap_stale.session_slug, root=tmp_path),
        _NOW,
        sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    # A corrupt-body record dir, aged past the heartbeat TTL.
    corrupt_dir = sr.sessions_dir(root=tmp_path) / 'reap-corrupt'
    corrupt_dir.mkdir(parents=True)
    corrupt_record = corrupt_dir / 'record.json'
    corrupt_record.write_text('{not valid json')
    _set_mtime(corrupt_record, _NOW, sr.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    reaped_by_slug = {r.session_slug: r.reason for r in reaped}
    assert reaped_by_slug == {
        'reap-terminal': 'terminal_ttl',
        'reap-stale-pid': 'stale_pid',
        'reap-corrupt': 'corrupt',
    }

    remaining = {p.name for p in sr.sessions_dir(root=tmp_path).iterdir()}
    assert remaining == {'kept-running', 'kept-recent-terminal'}


def test_reap_continues_sweep_when_one_directory_fails_to_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single unreapable directory (permission error, a file held open, an
    ENOTEMPTY race with a concurrent writer) must not abort the sweep -- a
    later stale directory in the same sessions dir is still reaped.
    """
    poison = _make_record(session_slug='poison', status=sr.Status.EXITED, launcher_pid=os.getpid())
    victim = _make_record(session_slug='victim', status=sr.Status.EXITED, launcher_pid=os.getpid())
    for r in (poison, victim):
        sr.write_record(r, root=tmp_path)
        _set_mtime(
            sr.record_path_for_slug(r.session_slug, root=tmp_path),
            _NOW,
            sr.TERMINAL_TTL + timedelta(hours=1),
        )

    poison_dir = sr.record_path_for_slug(poison.session_slug, root=tmp_path).parent
    real_rmtree = sr.shutil.rmtree

    def _flaky_rmtree(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        if Path(path) == poison_dir:
            raise OSError('simulated ENOTEMPTY race')
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(sr.shutil, 'rmtree', _flaky_rmtree)

    with caplog.at_level(logging.ERROR):
        reaped = sr.reap_stale_records(root=tmp_path, now=_NOW)

    assert {r.session_slug for r in reaped} == {'victim'}
    assert poison_dir.is_dir()  # left behind, not silently claimed as reaped
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Step-9: CLI + fail-soft
# ---------------------------------------------------------------------------


def _launching_env(
    tmp_path: Path,
    *,
    role: str = 'unblock',
    project: str = 'df',
    task_id: str = '2085',
    escalation_id: str = 'esc-9',
    title: str = 'unblock:df#2085 routing-mechanism',
    prompt: str = '/unblock 2085',
    cwd: str = '/home/leo/src/dark-factory',
    launcher_pid: str = '4242',
) -> dict[str, str]:
    """Build the CLAUDE_SPAWN_* env mapping spawn-claude.sh passes to `launching`."""
    return {
        'CLAUDE_FLEET_ROOT': str(tmp_path),
        'CLAUDE_SPAWN_ROLE': role,
        'CLAUDE_SPAWN_PROJECT': project,
        'CLAUDE_SPAWN_TASK_ID': task_id,
        'CLAUDE_SPAWN_ESCALATION_ID': escalation_id,
        'CLAUDE_SPAWN_TITLE': title,
        'CLAUDE_SPAWN_PROMPT': prompt,
        'CLAUDE_SPAWN_CWD': cwd,
        'CLAUDE_SPAWN_LAUNCHER_PID': launcher_pid,
    }


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_main_launching_writes_record_and_prints_only_record_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected_dir)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.LAUNCHING
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.escalation_id == 'esc-9'
    assert record.title == 'unblock:df#2085 routing-mechanism'
    assert record.prompt == '/unblock 2085'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.launcher_pid == 4242


def test_main_exit_sets_status_exited_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(['exit', '--record', record_dir, '--code', '3'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.EXITED
    assert record.exit_code == 3


def test_main_launching_populates_result_file_and_exit_preserves_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Attention Rail T5: `launching` must allocate a deterministic result.md
    path inside the record dir, and the `exit` read-modify-write must preserve
    it through to the EXITED record (result_file is set once, at launch, and
    never touched again).
    """
    _set_env(monkeypatch, _launching_env(tmp_path))

    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    expected_result_file = str(sr.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')

    launching_record = sr.read_record(slug, root=tmp_path)
    assert launching_record.result_file is not None
    assert launching_record.result_file == expected_result_file

    rc = sr.main(['exit', '--record', record_dir, '--code', '3'])
    assert rc == 0

    exited_record = sr.read_record(slug, root=tmp_path)
    assert exited_record.result_file == expected_result_file


def test_main_refresh_sets_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(['refresh', '--record', record_dir, '--status', 'running'])

    assert rc == 0
    slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING


# --- identity fallback (parse_spawn_identity) -------------------------------


def test_parse_spawn_identity_env_first_overrides_conflicting_title() -> None:
    env = {
        'CLAUDE_SPAWN_ROLE': 'review',
        'CLAUDE_SPAWN_PROJECT': 'other',
        'CLAUDE_SPAWN_TASK_ID': '999',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-5',
    }
    identity = sr.parse_spawn_identity(
        env=env,
        title='unblock:df#2085 routing-mechanism',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'review'
    assert identity.project == 'other'
    assert identity.task_id == '999'
    assert identity.escalation_id == 'esc-5'


def test_parse_spawn_identity_falls_back_to_task_scoped_title() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='unblock:df#2085 routing-mechanism',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'unblock'
    assert identity.project == 'df'
    assert identity.task_id == '2085'


def test_parse_spawn_identity_falls_back_to_project_level_title_with_no_hash() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='prd:df attention-rail',
        prompt='',
        cwd='/x',
    )
    assert identity.role == 'prd'
    assert identity.project == 'df'
    assert identity.task_id is None


def test_parse_spawn_identity_defaults_when_env_and_title_absent() -> None:
    identity = sr.parse_spawn_identity(
        env={},
        title='',
        prompt='',
        cwd='/home/leo/src/dark-factory',
    )
    assert identity.role == 'session'
    assert identity.project == 'dark-factory'
    assert identity.task_id is None
    assert identity.escalation_id is None


def test_parse_spawn_identity_defaults_project_unknown_when_cwd_has_no_basename() -> None:
    identity = sr.parse_spawn_identity(env={}, title='', prompt='', cwd='')
    assert identity.role == 'session'
    assert identity.project == 'unknown'


def test_parse_spawn_identity_defaults_project_strips_trailing_slash_from_cwd() -> None:
    # CLAUDE_SPAWN_CWD comes from a bash positional arg and is not guaranteed
    # normalized; a trailing '/' must not make basename() degrade to '' (and
    # thus the project silently fall through to 'unknown').
    identity = sr.parse_spawn_identity(
        env={},
        title='',
        prompt='',
        cwd='/home/leo/src/dark-factory/',
    )
    assert identity.role == 'session'
    assert identity.project == 'dark-factory'


# --- fail-soft ---------------------------------------------------------------


def test_main_launching_fail_soft_when_write_record_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_main_launching_fail_soft_when_fleet_root_under_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A CLAUDE_FLEET_ROOT rooted under a pre-created regular file makes the
    # write's mkdir(parents=True) raise NotADirectoryError deterministically.
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    env = _launching_env(tmp_path)
    env['CLAUDE_FLEET_ROOT'] = str(blocker / 'fleet')
    _set_env(monkeypatch, env)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_main_launching_fail_soft_when_launcher_pid_not_numeric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # CLAUDE_SPAWN_LAUNCHER_PID is spawn-claude.sh's own "$$" and should
    # always be numeric, but a malformed/corrupted env value must fail soft
    # like any other registry fault -- int() raising ValueError inside
    # _run_launching must not escape main(), must not print a bogus record
    # dir, and must leave no record behind.
    env = _launching_env(tmp_path)
    env['CLAUDE_SPAWN_LAUNCHER_PID'] = 'not-a-pid'
    _set_env(monkeypatch, env)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['launching'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not sr.sessions_dir(root=tmp_path).exists()
