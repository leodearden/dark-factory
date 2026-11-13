"""Tests for orchestrator.session_hooks module (Attention Rail T6).

Covers: identity/slug resolution for hook events (env > default, keyed on
session_id), including CLAUDE_SPAWN_SESSION_ID convergence and
sessions-by-claude-id pointer keying (task 7489); the pure OSC-retitle and
display-title helpers; the SessionStart / Notification / Stop handlers
against a tmp session-registry root, including parent_session_id stamping,
question capture and the Stop-only self-exit decision (task 7193); and the
fail-soft main() CLI dispatch.
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree
from _session_hook_fixtures import (
    assert_hook_env_isolated,
    isolate_hook_env,
    owner_ppid_env,
    stdin_json,
)

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


def test_isolate_hook_env_clears_every_spawn_var_and_pins_the_fleet_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_SPAWN_SESSION_ID', 'leaked-slug')
    monkeypatch.setenv('CLAUDE_SPAWN_OWNER_PPID', '1')
    monkeypatch.setenv('CLAUDE_SPAWN_FUTURE_KNOB', 'x')

    isolate_hook_env(monkeypatch, tmp_path / 'fleet')

    assert_hook_env_isolated(tmp_path / 'fleet')
    assert not (tmp_path / 'fleet').exists()


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
# Task 2511 step-3: CLAUDE_SPAWN_SESSION_ID convergence (dual-record fix)
# ---------------------------------------------------------------------------


def test_hook_session_slug_prefers_claude_spawn_session_id_when_present() -> None:
    # The env token wins outright -- session_id/role/project/task_id are all
    # ignored -- because it is already spawn-claude.sh's own launching-record
    # slug; recomposing it through build_session_slug would double-prefix.
    hook_input = {'session_id': 'uuid-x', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': 'session-cockpit-3215033',
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }
    assert sh.hook_session_slug(hook_input, env=env) == 'session-cockpit-3215033'


@pytest.mark.parametrize('spawn_session_id', [None, '', '   '])
def test_hook_session_slug_falls_back_when_claude_spawn_session_id_absent(
    spawn_session_id: str | None,
) -> None:
    # Missing/blank/whitespace-only CLAUDE_SPAWN_SESSION_ID must not clobber
    # the hand-launched session_id-derived fallback slug.
    hook_input = {'session_id': 'abc-123', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}
    if spawn_session_id is not None:
        env['CLAUDE_SPAWN_SESSION_ID'] = spawn_session_id
    slug = sh.hook_session_slug(hook_input, env=env)
    assert slug == sr.build_session_slug(
        'session',
        'dark-factory',
        None,
        'abc-123',  # type: ignore[arg-type]
    )


def test_hook_session_slug_sanitizes_malformed_claude_spawn_session_id() -> None:
    hook_input = {'session_id': 'uuid-x', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': 'bad/id#x'}
    slug = sh.hook_session_slug(hook_input, env=env)
    assert slug == sr.sanitize_slug('bad/id#x')
    assert slug == 'bad-id-x'


@pytest.mark.parametrize(
    ('spawn_session_id', 'expected'),
    [('.', '-'), ('..', '--'), ('...', '---')],
)
def test_hook_session_slug_sanitizes_all_dots_claude_spawn_session_id(
    spawn_session_id: str,
    expected: str,
) -> None:
    # Task 4112: the live, externally-supplied entry point for the '..' escape.
    # CLAUDE_SPAWN_SESSION_ID comes from outside the process and
    # hook_session_slug hands it to sanitize_slug DIRECTLY (deliberately
    # bypassing build_session_slug to avoid double-prefixing -- see its
    # docstring). Covering only the registry unit would leave this
    # attacker-reachable path untested: a future refactor of that bypass could
    # drop the sanitize call with every registry-level test still green.
    #
    # This test owns exactly that routing claim. The downstream containment
    # property (record_path_for_slug's join stays under sessions_dir) is owned
    # by test_record_path_for_slug_all_dots_slug_stays_under_sessions_dir in
    # test_session_registry.py, together with its non-vacuity counterfactual --
    # asserting it here too would drag a registry-layer invariant across the
    # module boundary and break two files for one change.
    hook_input = {'session_id': 'uuid-x', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': spawn_session_id}
    slug = sh.hook_session_slug(hook_input, env=env)
    assert slug == expected


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
    # durable POSIX session leader instead (see
    # session_ownership_probes.py::EventProbes.hand_launched_liveness_pid).
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
# Task 2511 step-3: run_session_start/run_notification/run_stop convergence
# ---------------------------------------------------------------------------


def test_run_session_start_converges_onto_spawn_claude_launching_record(
    tmp_path: Path,
) -> None:
    # A rich LAUNCHING record, as spawn-claude.sh's `launching` write would
    # produce, pre-exists at slug S. SessionStart (with a DIFFERENT Claude
    # Code session_id, but CLAUDE_SPAWN_SESSION_ID=S) must advance that SAME
    # record rather than creating a second, uuid-keyed one.
    slug = 'session-cockpit-3215033'
    result_file = str(sr.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')
    launching = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.LAUNCHING,
        role='session',
        project='cockpit',
        task_id=None,
        prompt='/spawn cockpit',
        cwd='/home/leo/src/dark-factory',
        launcher_pid=3215033,
        result_file=result_file,
    )
    sr.write_record(launching, root=tmp_path)

    hook_input = {'session_id': 'uuid-x', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    # Rich pre-existing fields survive the convergence refresh untouched.
    assert record.role == 'session'
    assert record.project == 'cockpit'
    assert record.prompt == '/spawn cockpit'
    assert record.launcher_pid == 3215033
    assert record.result_file == result_file
    # No separate uuid-keyed duplicate: exactly one dir in sessions_dir.
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_full_lifecycle_converges_on_one_slug_through_exit(tmp_path: Path) -> None:
    slug = 'session-cockpit-3215034'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215034),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-y', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING

    sh.run_notification(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.AWAITING_INPUT

    sh.run_stop(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE

    sr.update_status(slug, root=tmp_path, status=sr.Status.EXITED, exit_code=0)
    final = sr.read_record(slug, root=tmp_path)
    assert final.status == sr.Status.EXITED
    assert final.exit_code == 0

    # One record dir throughout the whole lifecycle -- never a second one.
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_run_notification_and_stop_converge_onto_spawn_claude_slug_not_uuid(
    tmp_path: Path,
) -> None:
    slug = 'session-cockpit-3215035'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.RUNNING, launcher_pid=3215035),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-z', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_notification(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.AWAITING_INPUT

    sh.run_stop(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE

    uuid_slug = sh.hook_session_slug({'session_id': 'uuid-z', 'cwd': hook_input['cwd']}, env={})
    assert uuid_slug != slug
    assert not sr.record_path_for_slug(uuid_slug, root=tmp_path).exists()


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
    stdin_json(monkeypatch, hook_input)

    rc = sh.main(['notification'])

    assert rc == 0
    slug = sh.hook_session_slug(hook_input, env={})
    record = sr.read_record(slug, root=tmp_path)
    assert record.question is not None
    assert record.question.text == 'Claude needs your permission to run Bash'


# ---------------------------------------------------------------------------
# Step-9: main(argv) CLI dispatch + fail-soft
# ---------------------------------------------------------------------------


def test_main_session_start_creates_running_record_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    hook_input = {'session_id': 'sess-main-1', 'cwd': '/home/leo/src/dark-factory'}
    stdin_json(monkeypatch, hook_input)

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
    stdin_json(monkeypatch, hook_input)

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
    stdin_json(monkeypatch, hook_input)

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
    stdin_json(monkeypatch, {'session_id': 'sess-main-4', 'cwd': '/home/leo/src/dark-factory'})

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

    stdin_json(monkeypatch, hook_input)
    assert sh.main(['session-start']) == 0
    stdin_json(monkeypatch, hook_input)
    assert sh.main(['notification']) == 0
    stdin_json(monkeypatch, hook_input)
    assert sh.main(['stop']) == 0

    session_dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(session_dirs) == 1
    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE


def test_main_session_start_failure_still_returns_zero_not_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Contrast case for
    # orchestrator/tests/test_session_hooks_install.py::test_main_install_returns_nonzero_on_failure:
    # the session-start/notification/stop verbs remain fail-soft (always 0).
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    stdin_json(monkeypatch, {'session_id': 'sess-rc', 'cwd': '/home/leo/src/dark-factory'})

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    assert sh.main(['session-start']) == 0


# ---------------------------------------------------------------------------
# Task 7193: the self-exit decision is Stop-only
# ---------------------------------------------------------------------------


def test_notification_takes_no_self_exit_decision(tmp_path: Path) -> None:
    slug = 'prd-df-7193-7193250'
    record_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            self_exit=True,
            claude_session_id='uuid-owner',
            claude_owner_pid=7_193_201,
            launcher_pid=7_193_250,
        ),
        root=tmp_path,
    )
    (record_dir / sr.RESULT_FILENAME).write_text('outcome: done\n', encoding='utf-8')
    marker = record_dir / sr.EXIT_REQUESTED_FILENAME
    marker.write_text('uuid-owner', encoding='utf-8')

    sh.run_notification(
        {'session_id': 'uuid-owner', 'cwd': str(tmp_path), 'background_tasks': [], 'session_crons': []},
        owner_ppid_env(slug, 7_193_200),
        root=tmp_path,
        process_tree=claude_process_tree(7_193_201, 7_193_200),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.exit_request is None
    assert marker.exists()


# ---------------------------------------------------------------------------
# Task 7489: pointer keying -- a session with no CLAUDE_SPAWN_SESSION_ID (a
# T3-hosted one) reaches its record through sessions-by-claude-id, then
# sessions-by-pid, before the hand-launched keying.
# ---------------------------------------------------------------------------

_T3_CWD = '/home/leo/src/dark-factory'
_T3_SLUG = 'session-dark-factory-7489001'
_T3_OWNER = 7_489_101
_NESTED = 7_489_202


def _write_t3_record(
    tmp_path: Path,
    *,
    session_id: str,
    status: sr.Status = sr.Status.RUNNING,
    owner_pid: int = _T3_OWNER,
) -> None:
    sr.write_record(
        sr.SessionRecord(
            session_slug=_T3_SLUG,
            status=status,
            title='t3 child',
            cwd=_T3_CWD,
            launcher_pid=7_489_900,
            display=sr.Display(kind=sr.DisplayKind.T3.value, t3_thread_id='thread-7489'),
            claude_session_id=session_id,
            claude_launch_session_id=session_id,
            claude_owner_pid=owner_pid,
        ),
        root=tmp_path,
    )
    sr.write_claude_session_pointer(session_id, _T3_SLUG, root=tmp_path)
    sr.write_session_pointer(owner_pid, _T3_SLUG, root=tmp_path)


def _reaped_pid() -> int:
    child = subprocess.Popen(['true'])
    child.wait()
    return child.pid


def _hand_slug(session_id: str, tmp_path: Path) -> str:
    """The hand-launched keying's slug: what the hooks resolve against a registry with no pointers."""
    return sh.hook_session_slug(
        _t3_event(session_id),
        {},
        root=tmp_path / 'pointerless-fleet',
        process_tree=claude_process_tree(None),
    )


def _session_slugs(tmp_path: Path) -> set[str]:
    return {path.name for path in sr.sessions_dir(root=tmp_path).iterdir()}


def _t3_event(session_id: str, **extra: str) -> dict[str, str]:
    return {'session_id': session_id, 'cwd': _T3_CWD, **extra}


@pytest.mark.parametrize(
    ('handler', 'expected'),
    [
        (sh.run_session_start, sr.Status.RUNNING),
        (sh.run_notification, sr.Status.AWAITING_INPUT),
        (sh.run_stop, sr.Status.IDLE),
    ],
)
def test_session_id_pointer_resolves_every_event_onto_the_t3_record(
    tmp_path: Path,
    handler: Callable[..., object],
    expected: sr.Status,
) -> None:
    _write_t3_record(tmp_path, session_id='sid-t3', status=sr.Status.LAUNCHING)

    handler(_t3_event('sid-t3'), {}, root=tmp_path, process_tree=claude_process_tree(_T3_OWNER))

    record = sr.read_record(_T3_SLUG, root=tmp_path)
    assert record.status is expected
    assert record.display is not None and record.display.kind == sr.DisplayKind.T3.value
    assert _session_slugs(tmp_path) == {_T3_SLUG}


def test_session_start_keeps_the_t3_display_over_an_inherited_tmux_env(tmp_path: Path) -> None:
    _write_t3_record(tmp_path, session_id='sid-t3')

    sh.run_session_start(
        _t3_event('sid-t3'),
        {'TMUX': '/tmp/tmux-1000/default,1,0', 'TMUX_PANE': '%3'},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_OWNER),
    )

    display = sr.read_record(_T3_SLUG, root=tmp_path).display
    assert display is not None
    assert (display.kind, display.t3_thread_id) == (sr.DisplayKind.T3.value, 'thread-7489')


def test_a_session_id_pointer_the_record_no_longer_vouches_for_falls_through(
    tmp_path: Path,
) -> None:
    _write_t3_record(tmp_path, session_id='sid-current')
    sr.write_claude_session_pointer('sid-stale', _T3_SLUG, root=tmp_path)

    sh.run_notification(
        _t3_event('sid-stale'), {}, root=tmp_path, process_tree=claude_process_tree(_T3_OWNER)
    )

    t3 = sr.read_record(_T3_SLUG, root=tmp_path)
    assert (t3.status, t3.claude_session_id) == (sr.Status.RUNNING, 'sid-current')
    assert _session_slugs(tmp_path) == {_T3_SLUG, _hand_slug('sid-stale', tmp_path)}


@pytest.mark.parametrize('source', ['clear', 'compact'])
def test_pid_pointer_rebinds_the_t3_record_on_a_remint(tmp_path: Path, source: str) -> None:
    _write_t3_record(tmp_path, session_id='sid-before')

    sh.run_session_start(
        _t3_event('sid-after', source=source),
        {},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_OWNER),
    )

    record = sr.read_record(_T3_SLUG, root=tmp_path)
    assert record.claude_session_id == 'sid-after'
    assert record.claude_launch_session_id == 'sid-before'
    assert record.claude_owner_pid == _T3_OWNER
    assert sr.resolve_session_slug_for_claude_session_id('sid-after', root=tmp_path) == _T3_SLUG
    assert sr.resolve_session_slug_for_pid(_T3_OWNER, root=tmp_path) == _T3_SLUG
    assert _session_slugs(tmp_path) == {_T3_SLUG}


@pytest.mark.parametrize('source', ['startup', 'resume'])
def test_pid_pointer_does_not_rebind_the_t3_record_on_a_plain_session_start(
    tmp_path: Path,
    source: str,
) -> None:
    _write_t3_record(tmp_path, session_id='sid-before')

    sh.run_session_start(
        _t3_event('sid-new', source=source),
        {},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_OWNER),
    )

    assert sr.read_record(_T3_SLUG, root=tmp_path).claude_session_id == 'sid-before'
    assert _session_slugs(tmp_path) == {_T3_SLUG, _hand_slug('sid-new', tmp_path)}


def test_a_nested_claude_cannot_capture_the_t3_record(tmp_path: Path) -> None:
    _write_t3_record(tmp_path, session_id='sid-t3')
    nested = claude_process_tree(_NESTED, _T3_OWNER)

    sh.run_session_start(_t3_event('sid-nested', source='resume'), {}, root=tmp_path, process_tree=nested)
    sh.run_session_start(_t3_event('sid-nested-2', source='compact'), {}, root=tmp_path, process_tree=nested)
    sh.run_stop(_t3_event('sid-nested-2'), {}, root=tmp_path, process_tree=nested)

    t3 = sr.read_record(_T3_SLUG, root=tmp_path)
    assert (t3.status, t3.claude_session_id, t3.claude_owner_pid) == (
        sr.Status.RUNNING,
        'sid-t3',
        _T3_OWNER,
    )
    assert sr.resolve_session_slug_for_pid(_T3_OWNER, root=tmp_path) == _T3_SLUG
    assert _session_slugs(tmp_path) == {_T3_SLUG, _hand_slug('sid-nested', tmp_path)}


def test_an_env_slug_binding_writes_a_session_id_pointer(tmp_path: Path) -> None:
    slug = 'session-cockpit-7489300'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=7_489_300),
        root=tmp_path,
    )

    sh.run_session_start(
        _t3_event('sid-env'),
        owner_ppid_env(slug, 7_489_310),
        root=tmp_path,
        process_tree=claude_process_tree(7_489_311, 7_489_310),
    )

    assert sr.resolve_session_slug_for_claude_session_id('sid-env', root=tmp_path) == slug


def test_a_hand_launched_record_carries_its_owner_pid_and_both_pointers(tmp_path: Path) -> None:
    record = sh.run_session_start(
        _t3_event('sid-hand'), {}, root=tmp_path, process_tree=claude_process_tree(7_489_401)
    )

    assert record.session_slug == _hand_slug('sid-hand', tmp_path)
    assert record.claude_owner_pid == 7_489_401
    assert sr.resolve_session_slug_for_pid(7_489_401, root=tmp_path) == record.session_slug
    assert sr.resolve_session_slug_for_claude_session_id('sid-hand', root=tmp_path) == record.session_slug


def test_a_hand_launched_clear_keeps_the_session_on_its_record(tmp_path: Path) -> None:
    tree = claude_process_tree(7_489_501)
    sh.run_session_start(_t3_event('sid-first'), {}, root=tmp_path, process_tree=tree)

    sh.run_session_start(_t3_event('sid-second', source='clear'), {}, root=tmp_path, process_tree=tree)
    sh.run_stop(_t3_event('sid-second'), {}, root=tmp_path, process_tree=tree)

    record = sr.read_record(_hand_slug('sid-first', tmp_path), root=tmp_path)
    assert (record.status, record.claude_session_id) == (sr.Status.IDLE, 'sid-second')
    assert _session_slugs(tmp_path) == {_hand_slug('sid-first', tmp_path)}


_T3_RESUMED = 7_489_102


def test_a_resume_in_a_new_process_reclaims_a_dead_owner_then_rebinds_on_compact(
    tmp_path: Path,
) -> None:
    _write_t3_record(tmp_path, session_id='sid-t3', owner_pid=_reaped_pid())
    resumed = claude_process_tree(_T3_RESUMED)

    sh.run_session_start(_t3_event('sid-t3', source='resume'), {}, root=tmp_path, process_tree=resumed)

    assert sr.read_record(_T3_SLUG, root=tmp_path).claude_owner_pid == _T3_RESUMED
    assert sr.resolve_session_slug_for_pid(_T3_RESUMED, root=tmp_path) == _T3_SLUG

    sh.run_session_start(_t3_event('sid-compacted', source='compact'), {}, root=tmp_path, process_tree=resumed)

    assert sr.read_record(_T3_SLUG, root=tmp_path).claude_session_id == 'sid-compacted'
    assert _session_slugs(tmp_path) == {_T3_SLUG}


def test_a_same_session_id_from_a_new_process_leaves_a_live_owner_in_place(
    tmp_path: Path,
) -> None:
    _write_t3_record(tmp_path, session_id='sid-t3', owner_pid=os.getpid())

    sh.run_session_start(
        _t3_event('sid-t3', source='resume'),
        {},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_RESUMED, os.getpid()),
    )

    assert sr.read_record(_T3_SLUG, root=tmp_path).claude_owner_pid == os.getpid()
    assert sr.resolve_session_slug_for_pid(_T3_RESUMED, root=tmp_path) is None
