"""Tests for orchestrator.session_hooks module (Attention Rail T6).

Covers: identity/slug resolution for hook events (env > default, keyed on
session_id); pure OSC-retitle + display-title helpers; the SessionStart /
Notification / Stop handlers against a tmp session-registry root; the
fail-soft main() CLI dispatch; the settings-merge (MERGE-not-clobber)
function; and a bash-level integration test exercising the real
skills/spawn/hooks/*.sh entrypoints end-to-end.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS_DIR = _REPO_ROOT / 'skills' / 'spawn' / 'hooks'

# ---------------------------------------------------------------------------
# Step-1: identity + slug resolution
# ---------------------------------------------------------------------------


def test_resolve_hook_identity_defaults_when_no_spawn_env() -> None:
    hook_input = {'session_id': 'sess-1', 'cwd': '/home/leo/src/dark-factory'}
    identity = sh.resolve_hook_identity(hook_input, env={})
    assert identity.role == 'session'
    assert identity.project == 'dark-factory'
    assert identity.task_id is None


def test_resolve_hook_identity_env_wins_over_default() -> None:
    hook_input = {'session_id': 'sess-1', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }
    identity = sh.resolve_hook_identity(hook_input, env=env)
    assert identity.role == 'unblock'
    assert identity.project == 'df'
    assert identity.task_id == '2085'


def test_resolve_hook_identity_uses_os_getcwd_when_cwd_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    identity = sh.resolve_hook_identity({'session_id': 'sess-2'}, env={})
    assert identity.project == tmp_path.name


def test_hook_session_slug_keyed_on_session_id_hand_launched() -> None:
    hook_input = {'session_id': 'abc-123', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, env={})
    # session_id ('abc-123', a str) fills the launcher_pid slot deliberately;
    # see session_hooks.hook_session_slug's docstring.
    assert slug == sr.build_session_slug(
        'session',
        'dark-factory',
        None,
        'abc-123',  # type: ignore[arg-type]
    )


def test_hook_session_slug_keyed_on_session_id_spawned() -> None:
    hook_input = {'session_id': 'abc-123', 'cwd': '/x'}
    env = {
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }
    slug = sh.hook_session_slug(hook_input, env=env)
    assert slug == sr.build_session_slug(
        'unblock',
        'df',
        '2085',
        'abc-123',  # type: ignore[arg-type]
    )


def test_hook_session_slug_is_deterministic() -> None:
    hook_input = {'session_id': 'same-id', 'cwd': '/home/leo/src/dark-factory'}
    assert sh.hook_session_slug(hook_input, env={}) == sh.hook_session_slug(hook_input, env={})


def test_hook_session_slug_is_filesystem_safe() -> None:
    hook_input = {'session_id': 'weird id/with#chars', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, env={})
    assert re.fullmatch(r'[A-Za-z0-9._-]+', slug)


def test_hook_session_slug_uses_cwd_fallback_when_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # No 'cwd' key at all -> os.getcwd() fallback still yields a well-formed slug.
    monkeypatch.chdir(tmp_path)
    hook_input = {'session_id': 'sess-3'}
    slug = sh.hook_session_slug(hook_input, env={})
    assert slug == sr.build_session_slug(
        'session',
        tmp_path.name,
        None,
        'sess-3',  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Step-3: pure OSC-retitle + display-title helpers
# ---------------------------------------------------------------------------


def _make_hook_record(**overrides: object) -> sr.SessionRecord:
    """Minimal SessionRecord for hook_display_title's record-title fallback tests."""
    fields: dict = {
        'session_slug': 'session-dark-factory-sess-1',
        'status': sr.Status.RUNNING,
        'title': '',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


def test_osc_retitle_sequence_running_uses_gear_glyph() -> None:
    seq = sh.osc_retitle_sequence(sr.Status.RUNNING, 'unblock:df#2085')
    assert seq == '\033]0;⚙ unblock:df#2085\007'


def test_osc_retitle_sequence_awaiting_input_uses_pause_glyph_and_label() -> None:
    seq = sh.osc_retitle_sequence(sr.Status.AWAITING_INPUT, 'unblock:df#2085')
    assert seq == '\033]0;⏸ AWAITING unblock:df#2085\007'


def test_osc_retitle_sequence_idle_uses_check_glyph() -> None:
    seq = sh.osc_retitle_sequence(sr.Status.IDLE, 'unblock:df#2085')
    assert seq == '\033]0;✅ unblock:df#2085\007'


def test_hook_display_title_prefers_explicit_env_title() -> None:
    identity = sr.SpawnIdentity(role='unblock', project='df', task_id='2085', escalation_id=None)
    env = {'CLAUDE_SPAWN_TITLE': 'unblock:df#2085 routing-mechanism'}
    record = _make_hook_record(title='stale-title')
    assert sh.hook_display_title(identity, env, record) == 'unblock:df#2085 routing-mechanism'


def test_hook_display_title_falls_back_to_record_title_when_no_env_title() -> None:
    identity = sr.SpawnIdentity(role='unblock', project='df', task_id='2085', escalation_id=None)
    record = _make_hook_record(title='unblock:df#2085 routing-mechanism')
    assert sh.hook_display_title(identity, env={}, record=record) == 'unblock:df#2085 routing-mechanism'


def test_hook_display_title_derives_role_project_task_when_no_explicit_title() -> None:
    identity = sr.SpawnIdentity(role='unblock', project='df', task_id='2085', escalation_id=None)
    assert sh.hook_display_title(identity, env={}, record=None) == 'unblock:df#2085'


def test_hook_display_title_derives_role_project_when_no_task_id() -> None:
    identity = sr.SpawnIdentity(role='session', project='dark-factory', task_id=None, escalation_id=None)
    assert sh.hook_display_title(identity, env={}, record=None) == 'session:dark-factory'


def test_hook_display_title_ignores_blank_record_title() -> None:
    identity = sr.SpawnIdentity(role='session', project='dark-factory', task_id=None, escalation_id=None)
    record = _make_hook_record(title='')
    assert sh.hook_display_title(identity, env={}, record=record) == 'session:dark-factory'


# ---------------------------------------------------------------------------
# Step-5: run_session_start (hand-launched-capture + refresh)
# ---------------------------------------------------------------------------


def test_run_session_start_hand_launched_creates_new_running_record(tmp_path: Path) -> None:
    # Case A: no prior record, no CLAUDE_SPAWN_* -> a NEW record.json appears
    # at the session_id-keyed slug (the hand-launched-capture signal).
    hook_input = {'session_id': 'sess-a', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.role == 'session'
    assert record.project == 'dark-factory'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.transcript_path == sr.transcript_path_for_cwd('/home/leo/src/dark-factory')


def test_run_session_start_refreshes_existing_record_preserving_fields(tmp_path: Path) -> None:
    # Case B: a record already exists at the slug -> refreshed to 'running'
    # (status flips, mtime heartbeat bumps) without dropping previously-
    # populated fields.
    hook_input = {'session_id': 'sess-b', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.LAUNCHING,
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    sr.write_record(existing, root=tmp_path)
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    old_ts = record_path.stat().st_mtime - 10
    os.utime(record_path, (old_ts, old_ts))

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    # Previously-populated fields survive the refresh untouched.
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.prompt == '/unblock 2085'
    # mtime heartbeat bumped by the refresh write.
    assert record_path.stat().st_mtime > old_ts


def test_run_session_start_hand_launched_uses_session_leader_as_liveness_pid(
    tmp_path: Path,
) -> None:
    # os.getppid() would resolve to this hook's own short-lived bash
    # entrypoint (dead within seconds) -- reap_stale_records' stale_pid rule
    # would then reclaim a long-idle hand-launched session's record even
    # though its terminal is still alive. os.getsid(0) resolves to the
    # durable POSIX session leader instead (see _hand_launched_liveness_pid).
    hook_input = {'session_id': 'sess-pid', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.launcher_pid == os.getsid(0)


def test_run_session_start_spawned_still_prefers_explicit_launcher_pid_env(
    tmp_path: Path,
) -> None:
    # CLAUDE_SPAWN_LAUNCHER_PID (when present) wins outright -- the liveness-
    # pid fallback only ever applies to the hand-launched (no env) case.
    hook_input = {'session_id': 'sess-pid-2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_LAUNCHER_PID': '4242'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.launcher_pid == 4242


# ---------------------------------------------------------------------------
# Task 2292 step-3: SessionStart parent_session_id stamping (Fleet Cockpit C2)
# ---------------------------------------------------------------------------


def test_run_session_start_create_path_stamps_parent_session_id(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-p1', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_PARENT_ID': 'unblock-df-2085-4242'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.parent_session_id == 'unblock-df-2085-4242'


def test_run_session_start_refresh_path_stamps_parent_session_id_preserving_fields(
    tmp_path: Path,
) -> None:
    hook_input = {'session_id': 'sess-p2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_PARENT_ID': 'unblock-df-2085-4242'}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.LAUNCHING,
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    sr.write_record(existing, root=tmp_path)
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    old_ts = record_path.stat().st_mtime - 10
    os.utime(record_path, (old_ts, old_ts))

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.parent_session_id == 'unblock-df-2085-4242'
    # Previously-populated fields survive the enrichment write.
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.prompt == '/unblock 2085'
    # mtime heartbeat bumped by the refresh write.
    assert record_path.stat().st_mtime > old_ts


def test_run_session_start_no_parent_id_env_leaves_parent_session_id_none(tmp_path: Path) -> None:
    # Hand-launched root: no CLAUDE_SPAWN_PARENT_ID -> stays None.
    hook_input = {'session_id': 'sess-p3', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.parent_session_id is None


# ---------------------------------------------------------------------------
# Step-7: run_notification / run_stop (status flip + OSC return)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('handler_name', 'expected_status', 'expected_glyph_prefix'),
    [
        ('run_notification', 'awaiting-input', '⏸ AWAITING'),
        ('run_stop', 'idle', '✅'),
    ],
)
def test_run_notification_and_run_stop_flip_status_and_return_osc(
    tmp_path: Path,
    handler_name: str,
    expected_status: str,
    expected_glyph_prefix: str,
) -> None:
    hook_input = {'session_id': 'sess-c', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        role='session',
        project='dark-factory',
        cwd='/home/leo/src/dark-factory',
    )
    sr.write_record(existing, root=tmp_path)
    handler = getattr(sh, handler_name)

    osc = handler(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status(expected_status)
    assert osc == f'\033]0;{expected_glyph_prefix} session:dark-factory\007'


def test_run_notification_prefers_persisted_record_title(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        title='unblock:df#2085 routing-mechanism',
    )
    sr.write_record(existing, root=tmp_path)

    osc = sh.run_notification(hook_input, env, root=tmp_path)

    assert osc == '\033]0;⏸ AWAITING unblock:df#2085 routing-mechanism\007'


# ---------------------------------------------------------------------------
# Task 2292 step-1: Notification question capture (Fleet Cockpit C2)
# ---------------------------------------------------------------------------


def test_run_notification_stamps_question_from_message(tmp_path: Path) -> None:
    hook_input = {
        'session_id': 'sess-q1',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Claude needs your permission to run Bash',
    }
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        role='session',
        project='dark-factory',
        cwd='/home/leo/src/dark-factory',
    )
    sr.write_record(existing, root=tmp_path)

    sh.run_notification(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT
    assert record.question is not None
    assert record.question.text == 'Claude needs your permission to run Bash'
    # asked_at is a non-empty, genuinely parseable ISO-8601 timestamp.
    assert record.question.asked_at
    datetime.fromisoformat(record.question.asked_at)


def test_run_notification_question_stamp_preserves_other_fields(tmp_path: Path) -> None:
    # merge-not-clobber, record-level: previously-populated fields (title/
    # role/project/task_id) survive the question-stamping write.
    hook_input = {
        'session_id': 'sess-q2',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Allow file write?',
    }
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        title='unblock:df#2085 routing-mechanism',
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
    )
    sr.write_record(existing, root=tmp_path)

    sh.run_notification(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.title == 'unblock:df#2085 routing-mechanism'
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.question is not None
    assert record.question.text == 'Allow file write?'


@pytest.mark.parametrize(
    'hook_input_extra',
    [
        {},
        {'message': ''},
        {'message': '   '},
    ],
    ids=['absent', 'blank', 'whitespace'],
)
def test_run_notification_no_question_when_message_absent_or_blank(
    tmp_path: Path,
    hook_input_extra: dict[str, str],
) -> None:
    hook_input = {'session_id': 'sess-q3', 'cwd': '/home/leo/src/dark-factory', **hook_input_extra}
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        role='session',
        project='dark-factory',
        cwd='/home/leo/src/dark-factory',
    )
    sr.write_record(existing, root=tmp_path)

    osc = sh.run_notification(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT
    assert record.question is None
    assert osc == '\033]0;⏸ AWAITING session:dark-factory\007'


def test_run_stop_does_not_stamp_question_even_with_message(tmp_path: Path) -> None:
    # Stop (idle) never captures a question, even if the hook's stdin JSON
    # happens to carry a 'message' key.
    hook_input = {
        'session_id': 'sess-q4',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Should not be captured by Stop',
    }
    env: dict[str, str] = {}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.RUNNING,
        role='session',
        project='dark-factory',
        cwd='/home/leo/src/dark-factory',
    )
    sr.write_record(existing, root=tmp_path)

    sh.run_stop(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.IDLE
    assert record.question is None


def test_main_notification_stamps_question_via_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # stdin -> question plumbing: the hooks/notification.sh entrypoint passes
    # the Notification event's stdin JSON straight through to session_hooks
    # via command substitution, so main() itself must thread 'message' into
    # the stamped question with no .sh changes required.
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {
        'session_id': 'sess-main-q',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Claude needs your permission to run Bash',
    }
    _stdin_json(monkeypatch, hook_input)

    rc = sh.main(['notification'])

    assert rc == 0
    slug = sh.hook_session_slug(hook_input, env={})
    record = sr.read_record(slug, root=tmp_path)
    assert record.question is not None
    assert record.question.text == 'Claude needs your permission to run Bash'


# ---------------------------------------------------------------------------
# Step-9: main(argv) CLI dispatch + fail-soft
# ---------------------------------------------------------------------------


def _stdin_json(monkeypatch: pytest.MonkeyPatch, payload: Mapping[str, object]) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(payload)))


def test_main_session_start_creates_running_record_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {'session_id': 'sess-main-1', 'cwd': '/home/leo/src/dark-factory'}
    _stdin_json(monkeypatch, hook_input)

    rc = sh.main(['session-start'])

    assert rc == 0
    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING


def test_main_notification_writes_osc_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {'session_id': 'sess-main-2', 'cwd': '/home/leo/src/dark-factory'}
    _stdin_json(monkeypatch, hook_input)

    rc = sh.main(['notification'])

    assert rc == 0
    expected = sh.osc_retitle_sequence(sr.Status.AWAITING_INPUT, 'session:dark-factory')
    assert capsys.readouterr().out.strip() == expected
    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.AWAITING_INPUT


def test_main_stop_writes_osc_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {'session_id': 'sess-main-3', 'cwd': '/home/leo/src/dark-factory'}
    _stdin_json(monkeypatch, hook_input)

    rc = sh.main(['stop'])

    assert rc == 0
    expected = sh.osc_retitle_sequence(sr.Status.IDLE, 'session:dark-factory')
    assert capsys.readouterr().out.strip() == expected
    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE


def test_main_session_start_fail_soft_on_empty_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setattr('sys.stdin', io.StringIO(''))

    assert sh.main(['session-start']) == 0


def test_main_session_start_fail_soft_on_malformed_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setattr('sys.stdin', io.StringIO('not-json{{{'))

    assert sh.main(['session-start']) == 0


def test_main_fail_soft_when_registry_write_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _stdin_json(monkeypatch, {'session_id': 'sess-main-4', 'cwd': '/home/leo/src/dark-factory'})

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sh.main(['session-start'])

    assert rc == 0
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_main_session_start_notification_stop_share_one_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Identity stability: the SAME session_id across all three events must
    # resolve to exactly ONE record directory, ending at status='idle'.
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {'session_id': 'sess-main-5', 'cwd': '/home/leo/src/dark-factory'}

    _stdin_json(monkeypatch, hook_input)
    assert sh.main(['session-start']) == 0
    _stdin_json(monkeypatch, hook_input)
    assert sh.main(['notification']) == 0
    _stdin_json(monkeypatch, hook_input)
    assert sh.main(['stop']) == 0

    session_dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(session_dirs) == 1
    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE


# ---------------------------------------------------------------------------
# Step-11: merge_hook_settings (MERGE-not-clobber)
# ---------------------------------------------------------------------------


def _real_settings_snapshot() -> dict:
    """A settings dict reproducing the real ~/.claude/settings.json's shape."""
    return {
        'env': {'SOME_VAR': '1'},
        'permissions': {'allow': ['Bash(git *)']},
        'hooks': {
            'PreToolUse': [
                {
                    'matcher': 'Bash',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/skim-rewrite.sh',
                            'timeout': 5,
                        }
                    ],
                },
                {
                    'matcher': 'EnterWorktree',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/worktree-hookspath-capture.sh',
                            'timeout': 10,
                        }
                    ],
                },
            ],
            'PostToolUse': [
                {
                    'matcher': 'ExitWorktree',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/worktree-hookspath-restore.sh',
                            'timeout': 10,
                        }
                    ],
                },
            ],
        },
        'statusLine': {'type': 'command', 'command': 'my-status-line.sh'},
        'enabledPlugins': ['hookify@claude-plugins-official'],
    }


def test_merge_hook_settings_adds_only_the_three_event_keys(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    hooks = after['hooks']
    assert set(hooks) == {'PreToolUse', 'PostToolUse', 'SessionStart', 'Notification', 'Stop'}
    for event, script_name in (
        ('SessionStart', 'session-start.sh'),
        ('Notification', 'notification.sh'),
        ('Stop', 'stop.sh'),
    ):
        entries = hooks[event]
        assert len(entries) == 1
        leaf = entries[0]['hooks'][0]
        assert leaf['type'] == 'command'
        assert leaf['command'] == str(script_dir / script_name)
        assert leaf['timeout'] == 10


def test_merge_hook_settings_leaves_pretooluse_posttooluse_byte_identical(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    assert after['hooks']['PreToolUse'] == before['hooks']['PreToolUse']
    assert after['hooks']['PostToolUse'] == before['hooks']['PostToolUse']


def test_merge_hook_settings_leaves_every_other_top_level_key_byte_identical(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    for key in ('env', 'permissions', 'statusLine', 'enabledPlugins'):
        assert after[key] == before[key]


def test_merge_hook_settings_does_not_mutate_input(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    sh.merge_hook_settings(before, script_dir)

    assert before == _real_settings_snapshot()


def test_merge_hook_settings_is_idempotent(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    once = sh.merge_hook_settings(before, script_dir)
    twice = sh.merge_hook_settings(once, script_dir)

    assert once == twice


def test_merge_hook_settings_creates_hooks_key_when_absent(tmp_path: Path) -> None:
    before = {'env': {'SOME_VAR': '1'}}
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    assert set(after['hooks']) == {'SessionStart', 'Notification', 'Stop'}
    assert after['env'] == before['env']


# ---------------------------------------------------------------------------
# _hooks_script_dir: resolves to the PRIMARY checkout, not a linked worktree
# ---------------------------------------------------------------------------


def _init_primary_checkout_with_hooks(tmp_path: Path) -> Path:
    """Create a temp git checkout under *tmp_path* with a real hooks trio, committed."""
    primary = tmp_path / 'primary'
    primary.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=primary, check=True)
    hooks_dir = primary / 'skills' / 'spawn' / 'hooks'
    hooks_dir.mkdir(parents=True)
    for script_name in ('session-start.sh', 'notification.sh', 'stop.sh'):
        (hooks_dir / script_name).write_text('#!/bin/sh\n')
    (primary / 'README.md').write_text('primary checkout\n')
    subprocess.run(['git', 'add', '-A'], cwd=primary, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=primary, check=True)
    return primary


def _init_primary_checkout_with_partial_hooks(tmp_path: Path) -> Path:
    """Like ``_init_primary_checkout_with_hooks`` but deliberately omits one
    script, so the resolved candidate dir exists (git succeeds) yet fails the
    ``all(...is_file())`` completeness guard in ``_hooks_script_dir``."""
    primary = tmp_path / 'primary'
    primary.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=primary, check=True)
    hooks_dir = primary / 'skills' / 'spawn' / 'hooks'
    hooks_dir.mkdir(parents=True)
    # Deliberately omit stop.sh.
    for script_name in ('session-start.sh', 'notification.sh'):
        (hooks_dir / script_name).write_text('#!/bin/sh\n')
    (primary / 'README.md').write_text('primary checkout with partial hooks\n')
    subprocess.run(['git', 'add', '-A'], cwd=primary, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=primary, check=True)
    return primary


def _fake_session_hooks_path(root: Path) -> Path:
    """Mirror the real repo layout (<root>/orchestrator/src/orchestrator/session_hooks.py)."""
    fake_file = root / 'orchestrator' / 'src' / 'orchestrator' / 'session_hooks.py'
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text('# stub for _hooks_script_dir test\n')
    return fake_file


def test_hooks_script_dir_resolves_to_primary_checkout_from_linked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _init_primary_checkout_with_hooks(tmp_path)
    worktree = tmp_path / 'worktree-2288'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', str(worktree), '-b', 'wt-2288'],
        cwd=primary,
        check=True,
    )
    fake_file = _fake_session_hooks_path(worktree)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    resolved = sh._hooks_script_dir()

    assert resolved == (primary / 'skills' / 'spawn' / 'hooks').resolve()
    assert resolved != fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


def test_hooks_script_dir_falls_back_when_candidate_missing_a_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git resolves fine and the candidate dir exists, but it is missing one
    of the three hook scripts -- the ``all(...is_file())`` guard must reject
    it and fall back to parents[3] rather than returning the incomplete
    candidate (e.g. an installed/partial checkout where skills/spawn/hooks is
    stale). Uses a linked worktree, like the resolves-to-primary-checkout
    test above, so the (rejected) candidate and the fallback are observably
    different directories -- otherwise a regression that wrongly accepted the
    incomplete candidate would go unnoticed, since both would resolve to the
    same path when the fake file lives directly under the primary checkout."""
    primary = _init_primary_checkout_with_partial_hooks(tmp_path)
    worktree = tmp_path / 'worktree-partial'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', str(worktree), '-b', 'wt-partial'],
        cwd=primary,
        check=True,
    )
    fake_file = _fake_session_hooks_path(worktree)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'
    assert resolved != (primary / 'skills' / 'spawn' / 'hooks').resolve()


def test_hooks_script_dir_falls_back_to_parents3_when_not_a_git_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_file = _fake_session_hooks_path(tmp_path)
    monkeypatch.setattr(sh, '__file__', str(fake_file))
    # Hermetic regardless of ambient git topology: cap git's upward .git
    # search at tmp_path so this test's "not a git checkout" premise holds
    # even if /tmp (or some ancestor) happens to sit inside a real repo.
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


def test_hooks_script_dir_falls_back_when_git_common_dir_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct fallback path from the not-a-git-checkout case above: git
    itself succeeds (returncode 0) but reports an empty/blank
    --git-common-dir, which must also degrade to the parents[3] fallback."""
    fake_file = _fake_session_hooks_path(tmp_path)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout='   \n', stderr='')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


# ---------------------------------------------------------------------------
# _run_install: missing-file create, backup-on-existing, atomic JSON write
# ---------------------------------------------------------------------------


def test_run_install_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'

    sh._run_install(settings_path=settings_path)

    assert settings_path.is_file()
    written = json.loads(settings_path.read_text())
    assert set(written['hooks']) == {'SessionStart', 'Notification', 'Stop'}
    assert list(tmp_path.glob('settings.json.*.bak')) == []


def test_run_install_backs_up_pre_merge_bytes_when_file_exists(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'
    before = {'env': {'SOME_VAR': '1'}, 'hooks': {'PreToolUse': []}}
    before_raw = json.dumps(before)
    settings_path.write_text(before_raw)

    sh._run_install(settings_path=settings_path)

    backups = list(tmp_path.glob('settings.json.*.bak'))
    assert len(backups) == 1
    assert backups[0].read_text() == before_raw

    merged = json.loads(settings_path.read_text())
    assert merged['env'] == before['env']
    assert set(merged['hooks']) == {'PreToolUse', 'SessionStart', 'Notification', 'Stop'}


def test_run_install_result_is_valid_json(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps({'env': {}}))

    sh._run_install(settings_path=settings_path)

    # json.loads raises on malformed content -- reaching the assertion below
    # at all is the real assertion here.
    reloaded = json.loads(settings_path.read_text())
    assert reloaded['env'] == {}


# ---------------------------------------------------------------------------
# main(['install', ...]): exit-code propagation (install is NOT fail-soft)
# ---------------------------------------------------------------------------


def test_main_install_succeeds_and_returns_zero(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'

    rc = sh.main(['install', '--settings-path', str(settings_path)])

    assert rc == 0
    assert settings_path.is_file()


def test_main_install_returns_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_path = tmp_path / 'settings.json'

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sh, '_run_install', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sh.main(['install', '--settings-path', str(settings_path)])

    assert rc == 1
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_main_session_start_failure_still_returns_zero_not_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Contrast case for the install-only exit-code change above: the
    # session-start/notification/stop verbs remain fail-soft (always 0).
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _stdin_json(monkeypatch, {'session_id': 'sess-rc', 'cwd': '/home/leo/src/dark-factory'})

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    assert sh.main(['session-start']) == 0


# ---------------------------------------------------------------------------
# Step-13: bash-level integration test (skills/spawn/hooks/*.sh entrypoints)
# ---------------------------------------------------------------------------


def _run_hook_script(
    script_name: str,
    hook_input: Mapping[str, object],
    root: Path,
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env['CLAUDE_FLEET_ROOT'] = str(root)
    return subprocess.run(
        [str(_HOOKS_DIR / script_name)],
        input=json.dumps(hook_input).encode(),
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_session_start_sh_then_notification_sh_flip_status(tmp_path: Path) -> None:
    # Proves the PYTHONPATH=orchestrator/src wiring, stdin passthrough to
    # python, and fail-soft exit for the real bash entrypoints end-to-end.
    hook_input = {'session_id': 'sess-bash-1', 'cwd': '/home/leo/src/dark-factory'}

    start_result = _run_hook_script('session-start.sh', hook_input, tmp_path)
    assert start_result.returncode == 0

    slug = sh.hook_session_slug(hook_input, env={})
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING

    notification_result = _run_hook_script('notification.sh', hook_input, tmp_path)
    assert notification_result.returncode == 0

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT
