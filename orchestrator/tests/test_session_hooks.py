"""Tests for orchestrator.session_hooks module (Attention Rail T6).

Covers: identity/slug resolution for hook events (env > default, keyed on
session_id); pure OSC-retitle + display-title helpers; the SessionStart /
Notification / Stop handlers against a tmp session-registry root; and the
fail-soft main() CLI dispatch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree, counting_walks
from _session_hook_fixtures import (
    isolate_hook_env,
    launching_record,
    owner_ppid_env,
    stdin_json,
    write_bound_parent,
)

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr
from orchestrator.session_ownership_probes import EventProbes, ProcessTree


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_isolate_hook_env_clears_every_spawn_var_and_pins_the_fleet_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_SPAWN_SESSION_ID', 'leaked-slug')
    monkeypatch.setenv('CLAUDE_SPAWN_OWNER_PPID', '1')
    monkeypatch.setenv('CLAUDE_SPAWN_FUTURE_KNOB', 'x')

    isolate_hook_env(monkeypatch, tmp_path / 'fleet')

    assert [k for k in os.environ if k.startswith('CLAUDE_SPAWN_')] == []
    assert os.environ['CLAUDE_FLEET_ROOT'] == str(tmp_path / 'fleet')
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
# Task 2510 step-1: _resolve_wm_window_id resolver (Fleet Cockpit C10 fix)
# ---------------------------------------------------------------------------


def test_resolve_wm_window_id_matches_exact_title() -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x03200007 0 host focus:df#2510 alpha\n'
        )

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=3, sleep=lambda _s: None
    )

    assert result == '0x03200007'
    assert calls == [['wmctrl', '-l']]


def test_resolve_wm_window_id_returns_none_when_no_match() -> None:
    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x03200007 0 host some-other-window\n'
        )

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=1, sleep=lambda _s: None
    )

    assert result is None


def test_resolve_wm_window_id_does_not_match_as_substring() -> None:
    """A marker that is only a PREFIX of a longer, unrelated window title must
    not match -- mirrors WmBackend.is_alive's exact-field guard."""

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x03200007 0 host focus:df#2510 alpha extra\n'
        )

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=1, sleep=lambda _s: None
    )

    assert result is None


def test_resolve_wm_window_id_retries_with_sleep_between_attempts() -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if len(calls) < 3:
            return subprocess.CompletedProcess(argv, returncode=1, stdout='')
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x03200007 0 host focus:df#2510 alpha\n'
        )

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha',
        run=fake_run,
        attempts=5,
        sleep=lambda s: sleeps.append(s),
    )

    assert result == '0x03200007'
    assert len(calls) == 3
    assert len(sleeps) == 2  # one sleep between each of the first 3 attempts, none after success


def test_resolve_wm_window_id_exhausts_attempts_then_none() -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=1, stdout='')

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=4, sleep=lambda _s: None
    )

    assert result is None
    assert len(calls) == 4


def test_resolve_wm_window_id_short_circuits_on_missing_binary_sentinel() -> None:
    """rc=127 is ``_wmctrl_list``'s sentinel for a missing ``wmctrl`` binary
    (see its docstring) -- a permanent failure, not the transient
    window-mapping race the retry loop exists for. It must fail fast on the
    first probe rather than retrying with sleeps in between."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=127, stdout='')

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=5, sleep=lambda _s: None
    )

    assert result is None
    assert len(calls) == 1


def test_resolve_wm_window_id_run_raising_is_caught_as_none() -> None:
    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError('wmctrl not found')

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=3, sleep=lambda _s: None
    )

    assert result is None


def test_resolve_wm_window_id_only_ever_calls_wmctrl_list() -> None:
    """Never issues a focus/activate command -- read-only probing only."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=1, stdout='')

    sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=3, sleep=lambda _s: None
    )

    assert all(call == ['wmctrl', '-l'] for call in calls)


def test_wmctrl_list_missing_binary_returns_rc_127(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely-missing ``wmctrl`` binary (``FileNotFoundError``) is the
    permanent-failure sentinel _resolve_wm_window_id short-circuits on."""

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError('wmctrl not found')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    result = sh._wmctrl_list(['wmctrl', '-l'])

    assert result.returncode == 127


def test_wmctrl_list_timeout_returns_distinct_transient_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer finding: ``_wmctrl_list`` used to map every ``OSError`` /
    ``SubprocessError`` -- including a ``subprocess.TimeoutExpired`` from a
    momentarily-hung ``wmctrl`` -- to the same rc=127 sentinel that
    ``_resolve_wm_window_id`` treats as a permanent missing-binary failure
    and short-circuits on. A timeout is transient, not permanent, so it must
    map to a distinct sentinel the retry loop keeps riding out."""

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=sh._WMCTRL_TIMEOUT_SECS)

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    result = sh._wmctrl_list(['wmctrl', '-l'])

    assert result.returncode == 124
    assert result.returncode != 127


def test_resolve_wm_window_id_retries_through_transient_timeout_sentinel() -> None:
    """rc=124 (the transient-timeout sentinel) must still be retried by the
    loop, unlike rc=127 which short-circuits immediately -- the two failure
    modes _wmctrl_list now distinguishes are handled distinctly end-to-end."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if len(calls) < 3:
            return subprocess.CompletedProcess(argv, returncode=124, stdout='')
        return subprocess.CompletedProcess(
            argv, returncode=0, stdout='0x03200007 0 host focus:df#2510 alpha\n'
        )

    result = sh._resolve_wm_window_id(
        'focus:df#2510 alpha', run=fake_run, attempts=5, sleep=lambda _s: None
    )

    assert result == '0x03200007'
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Task 2292 step-5: SessionStart best-effort display stamping (Fleet Cockpit C2)
# ---------------------------------------------------------------------------


def test_run_session_start_tmux_env_stamps_tmux_display(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d1', 'cwd': '/home/leo/src/dark-factory'}
    env = {'TMUX': '/tmp/tmux-1000/default,123,0', 'TMUX_PANE': '%3'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    # Assert on the wire value directly -- a StrEnum-vs-str choice can't drift.
    assert record.display.kind == 'tmux'
    assert record.display.tmux_target == '%3'


def test_run_session_start_windowid_env_stamps_wm_display(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    assert record.display.wm_title == 'session:dark-factory'


def test_run_session_start_decimal_windowid_stamps_canonical_hex_display(tmp_path: Path) -> None:
    # Real terminal emulators export $WINDOWID as a bare DECIMAL (e.g.
    # '52428807'), not hex. Capture must canonicalize it to wmctrl -l's
    # 0x%08x column form ('0x03200007') so string-consumers (the window-gone
    # reaper, dashboards) and the wm backend agree. 52428807 == 0x03200007
    # is an exact identity.
    hook_input = {'session_id': 'sess-d2b', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '52428807'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'


def test_run_session_start_tmux_takes_precedence_over_windowid(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d3', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'TMUX': '/tmp/tmux-1000/default,123,0',
        'TMUX_PANE': '%3',
        'WINDOWID': '0x3200007',
    }

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'tmux'


def test_run_session_start_no_tmux_or_windowid_leaves_display_none(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d4', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is None


def test_run_session_start_display_stamping_applies_on_refresh_path(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d5', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007'}
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.LAUNCHING,
        title='unblock:df#2085 routing-mechanism',
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    sr.write_record(existing, root=tmp_path)

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    # wm_title resolves from the persisted record title (hook_display_title's
    # existing precedence), not a freshly-derived role:project fallback.
    assert record.display.wm_title == 'unblock:df#2085 routing-mechanism'
    # Previously-populated fields survive.
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.prompt == '/unblock 2085'


# ---------------------------------------------------------------------------
# Task 2510 step-3: SessionStart marker-search display stamping (Fleet Cockpit C10 fix)
# ---------------------------------------------------------------------------


def test_run_session_start_marker_title_env_stamps_wm_display_via_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_input = {'session_id': 'sess-m1', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': 'focus:df#2510 alpha'}
    monkeypatch.setattr(sh, '_resolve_wm_window_id', lambda title: '0x03200007')

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    assert record.display.wm_title == 'focus:df#2510 alpha'


def test_run_session_start_marker_title_resolver_miss_leaves_display_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Best-effort: a resolution miss must be no worse than today -- record.display
    # stays None, exactly like the no-tmux/no-windowid case.
    hook_input = {'session_id': 'sess-m2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': 'focus:df#2510 alpha'}
    monkeypatch.setattr(sh, '_resolve_wm_window_id', lambda title: None)

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is None


def test_run_session_start_windowid_wins_over_marker_title_resolver_not_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_input = {'session_id': 'sess-m3', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007', 'CLAUDE_SPAWN_WM_TITLE': 'focus:df#2510 alpha'}

    def _boom(title: str) -> str | None:
        raise AssertionError('resolver must not be called when WINDOWID is present')

    monkeypatch.setattr(sh, '_resolve_wm_window_id', _boom)

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'


def test_run_session_start_tmux_wins_over_marker_title_resolver_not_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_input = {'session_id': 'sess-m4', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'TMUX': '/tmp/tmux-1000/default,123,0',
        'TMUX_PANE': '%3',
        'CLAUDE_SPAWN_WM_TITLE': 'focus:df#2510 alpha',
    }

    def _boom(title: str) -> str | None:
        raise AssertionError('resolver must not be called when TMUX is present')

    monkeypatch.setattr(sh, '_resolve_wm_window_id', _boom)

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'tmux'


def test_run_session_start_marker_title_stamping_applies_on_refresh_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_input = {'session_id': 'sess-m5', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': 'focus:df#2510 alpha'}
    monkeypatch.setattr(sh, '_resolve_wm_window_id', lambda title: '0x03200007')
    slug = sh.hook_session_slug(hook_input, env)
    existing = sr.SessionRecord(
        session_slug=slug,
        status=sr.Status.LAUNCHING,
        title='unblock:df#2085 routing-mechanism',
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    sr.write_record(existing, root=tmp_path)

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    assert record.display.wm_title == 'focus:df#2510 alpha'
    # Previously-populated fields survive.
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.prompt == '/unblock 2085'


def test_run_session_start_no_marker_title_resolver_never_invoked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No TMUX/WINDOWID/CLAUDE_SPAWN_WM_TITLE -> display stays None (matches
    # test_run_session_start_no_tmux_or_windowid_leaves_display_none) AND the
    # resolver is never invoked -- no fallback to a churny derived title.
    hook_input = {'session_id': 'sess-m6', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    def _boom(title: str) -> str | None:
        raise AssertionError('resolver must not be called when no marker is set')

    monkeypatch.setattr(sh, '_resolve_wm_window_id', _boom)

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = sr.read_record(slug, root=tmp_path)
    assert record.display is None


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
    # test_session_hooks_install.py::test_main_install_returns_nonzero_on_failure:
    # the session-start/notification/stop verbs remain fail-soft (always 0).
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    stdin_json(monkeypatch, {'session_id': 'sess-rc', 'cwd': '/home/leo/src/dark-factory'})

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'write_record', _boom)

    assert sh.main(['session-start']) == 0


# ---------------------------------------------------------------------------
# ONE EVENT, ONE /proc WALK (task 4662)
#
# The fixture below is the exact shape where all three walks fire today: a
# still-LAUNCHING, still-unbound spawn record whose OWNER_PPID verdict is
# True -- the owner's own pre-SessionStart event, task 4193 L2 ruling item
# 4-iii. _env_slug_ownership takes its verdict, _withhold_from_launching
# takes it again, and _bind_claude_session_id stamps claude_owner_pid from a
# third walk. Each assertion pairs the count with an OUTCOME check, so the
# pin cannot be satisfied by simply skipping a probe.
# ---------------------------------------------------------------------------


def _owner_shape(
    slug: str, tmp_path: Path, *, pid: int
) -> tuple[dict[str, str], ProcessTree, list[int]]:
    """LAUNCHING+unbound record whose OWNER_PPID verdict resolves to True."""
    launching_record(slug, tmp_path, pid=pid)
    tree, walks = counting_walks(claude_process_tree(4193501, 4193500))
    return owner_ppid_env(slug, 4193500), tree, walks


def test_notification_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662001'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662001)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    # OUTCOME PARITY (mirrors test_owner_stop_is_not_withheld_...).
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    assert record.claude_session_id == 'uuid-owner'


def test_stop_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662002'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662002)

    sh.run_stop(
        {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.IDLE
    assert record.claude_session_id == 'uuid-owner'


def test_session_start_walks_proc_exactly_once(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662003'
    env, tree, walks = _owner_shape(slug, tmp_path, pid=4662003)

    sh.run_session_start(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'source': 'startup',
        },
        env,
        root=tmp_path,
        process_tree=tree,
    )

    assert len(walks) == 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.RUNNING
    assert record.claude_session_id == 'uuid-owner'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_nested_session_start_walks_proc_at_most_once(tmp_path: Path) -> None:
    """The fork mirror: memoizing must not weaken the nested-inheritor guard.

    Mirrors test_nested_session_start_cannot_capture_an_unbound_launching_record
    -- the spawn record must be left unbound and LAUNCHING -- while pinning
    that the fork path (ownership verdict, the nested-claude liveness pid,
    and the bind-time stamp) also settles on one walk.
    """
    slug = 'session-cockpit-4662004'
    launching_record(slug, tmp_path, pid=4662004)
    env = owner_ppid_env(slug, 4193500)
    # A nested claude's parent is its agent's Bash-tool shell, not the
    # payload bash spawn-claude.sh exported.
    tree, walks = counting_walks(claude_process_tree(4193777, 4193776))

    nested = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'source': 'startup',
    }
    sh.run_session_start(nested, env, root=tmp_path, process_tree=tree)

    assert len(walks) <= 1
    spawn_record = sr.read_record(slug, root=tmp_path)
    assert spawn_record.claude_session_id is None
    assert spawn_record.status is sr.Status.LAUNCHING
    forked = sr.read_record(
        sh.hook_session_slug(nested, env, root=tmp_path, process_tree=tree), root=tmp_path
    )
    assert forked.claude_session_id == 'uuid-nested'
    assert forked.parent_session_id == slug


# ---------------------------------------------------------------------------
# _RecordSnapshot / _read_record_snapshot -- the TRI-STATE read (task 4662)
#
# One read must serve two consumers that want OPPOSITE things from a failed
# one. The ownership probe must fail soft to "adopt" (_env_slug_ownership:
# "every failure mode resolves to adopt (True), never to fork"), so it treats
# corrupt and absent alike. The refresh must NOT: session_registry's
# refresh_record guarantees "a *corrupt* existing body is NOT treated as
# absent". A two-state record|None would let a hook synthesize a fresh record
# over a corrupt body -- silent loss of the record holding role/prompt/
# result_file. Hence ABSENT and UNREADABLE are distinct states.
# ---------------------------------------------------------------------------


def test_record_snapshot_reads_an_existing_record(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662010'
    launching_record(slug, tmp_path, pid=4662010)

    snapshot = sh._read_record_snapshot(slug, tmp_path)

    assert snapshot.slug == slug
    assert snapshot.unreadable is False
    assert snapshot.record is not None
    assert snapshot.record.to_dict() == sr.read_record(slug, root=tmp_path).to_dict()


def test_record_snapshot_absent_is_not_unreadable_and_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ABSENT is the state that MAY be synthesized over -- and it is routine.

    An ordinary fresh spawn must log nothing, matching _env_slug_ownership's
    existing dedicated FileNotFoundError arm ("so an ordinary fresh spawn
    logs nothing").
    """
    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot('session-cockpit-4662011', tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_record_snapshot_corrupt_body_is_unreadable_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    slug = 'session-cockpit-4662012'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    with pytest.raises(sr.CorruptSessionRecord):
        sr.read_record(slug, root=tmp_path)

    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot(slug, tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is True
    # Degradation is never silent (repo's no-silent-fail-soft norm).
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_record_snapshot_arbitrary_oserror_is_unreadable_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'read_record', _boom)

    with caplog.at_level(logging.WARNING):
        snapshot = sh._read_record_snapshot('session-cockpit-4662013', tmp_path)

    assert snapshot.record is None
    assert snapshot.unreadable is True
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_record_snapshot_is_frozen(tmp_path: Path) -> None:
    """Immutable: a snapshot is one observation, not a mutable scratchpad."""
    slug = 'session-cockpit-4662014'
    launching_record(slug, tmp_path, pid=4662014)
    snapshot = sh._read_record_snapshot(slug, tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.record = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _HookSlugResolution -- carry the probe's read forward (task 4662)
#
# The ownership probe already reads the candidate slug's record. On the ADOPT
# branch that candidate IS the slug the handler goes on to write, so its
# snapshot is exactly the record about to be refreshed -- handing it forward
# is what collapses the event onto one read. On the FORK branch the resolver
# returns a DIFFERENT slug, so offering the rejected spawner's snapshot would
# apply a nested claude's decision to the spawning session's record: the
# ownership inversion tasks 4193 and 2511 both exist to prevent. Hence the
# invariant asserted on EVERY shape below.
# ---------------------------------------------------------------------------


def test_resolution_adopts_and_carries_the_probes_snapshot(tmp_path: Path) -> None:
    slug = 'session-cockpit-4662020'
    write_bound_parent(slug, tmp_path, 4662020)
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug == slug
    assert resolution.rejected_env_slug is None
    assert resolution.snapshot is not None
    assert resolution.snapshot.slug == resolution.slug
    assert resolution.snapshot.record is not None
    assert (
        resolution.snapshot.record.to_dict()
        == sr.read_record(slug, root=tmp_path).to_dict()
    )
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_forks_without_offering_the_rejected_snapshot(
    tmp_path: Path,
) -> None:
    """The snapshot belongs to the REJECTED slug, so it must not travel."""
    slug = 'session-cockpit-4662021'
    write_bound_parent(slug, tmp_path, 4662021)
    # Proven mismatch: the record is bound to 'uuid-parent', this is not it.
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug != slug
    assert resolution.rejected_env_slug == slug
    assert resolution.snapshot is None
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_reads_nothing_for_a_hand_launched_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env slug: nothing to probe, so nothing is read (must-not-be-called)."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('nothing to probe for a hand-launched session')

    monkeypatch.setattr(sr, 'read_record', _boom)
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}

    probes = EventProbes({}, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, {}, tmp_path, probes=probes)

    assert resolution.snapshot is None
    assert resolution.rejected_env_slug is None
    assert resolution.may_bind is True
    assert sh.hook_session_slug(hook_input, {}, root=tmp_path) == resolution.slug


def test_resolution_blank_stdin_session_id_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-discriminator early return reads nothing, so carries nothing."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('the blank-session_id early return reads no record')

    monkeypatch.setattr(sr, 'read_record', _boom)
    slug = 'session-cockpit-4662022'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': '  ', 'cwd': '/home/leo/src/dark-factory'}

    probes = EventProbes(env, claude_process_tree(None))
    resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)

    assert resolution.slug == slug
    assert resolution.may_bind is True
    assert resolution.snapshot is None
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == resolution.slug


def test_resolution_snapshot_slug_invariant_holds_on_every_shape(
    tmp_path: Path,
) -> None:
    """INVARIANT: a snapshot is attached only when it IS the returned slug.

    Asserted across all four resolver shapes rather than left to the reader
    of the branch, because a mis-attached snapshot would fail silently and
    only under a nested-claude race.
    """
    adopted = 'session-cockpit-4662023'
    forked = 'session-cockpit-4662024'
    write_bound_parent(adopted, tmp_path, 4662023)
    write_bound_parent(forked, tmp_path, 4662024)
    cwd = '/home/leo/src/dark-factory'
    shapes = [
        ({'session_id': 'uuid-parent', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': adopted}),
        ({'session_id': 'uuid-nested', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': forked}),
        ({'session_id': 'sess-hand', 'cwd': cwd}, {}),
        ({'session_id': '  ', 'cwd': cwd}, {'CLAUDE_SPAWN_SESSION_ID': adopted}),
    ]
    for hook_input, env in shapes:
        tree = claude_process_tree(None)
        probes = EventProbes(env, tree)
        resolution = sh._resolve_hook_slug(hook_input, env, tmp_path, probes=probes)
        assert resolution.snapshot is None or resolution.snapshot.slug == resolution.slug
        # hook_session_slug's public signature and str return are unchanged.
        public = sh.hook_session_slug(hook_input, env, root=tmp_path, process_tree=tree)
        assert isinstance(public, str)
        assert public == resolution.slug


# ---------------------------------------------------------------------------
# ONE READ, ONE DECISION, ONE WRITE for Notification/Stop (task 4662)
#
# Today one adopted-spawn event reads record.json THREE times (the ownership
# probe, the pre-refresh snapshot, and refresh_record's own internal read)
# and writes up to TWICE. The counts are the efficiency half; the
# same-snapshot assertion below is the correctness half -- today the withhold
# decision is computed from read #2 while the body written derives from read
# #3, so two racing hook events decide against stale state and write against
# fresh state.
# ---------------------------------------------------------------------------


class _RegistryIO:
    """Count reads per slug and record every body handed to write_record.

    Each read stamps a per-call marker into the returned body's ``cwd`` so a
    written record can be traced back to the read it derives from -- the
    technique that turns "how many reads" into "which read did the write
    come from", which is the TOCTOU property, not merely an I/O count.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[str] = []
        self.writes: list[sr.SessionRecord] = []
        real_read, real_write = sr.read_record, sr.write_record

        def _read(slug: str, *args: object, **kwargs: object) -> sr.SessionRecord:
            self.reads.append(slug)
            record = real_read(slug, *args, **kwargs)  # type: ignore[arg-type]
            record.cwd = f'read-{len(self.reads)}'
            return record

        def _write(record: sr.SessionRecord, *args: object, **kwargs: object) -> None:
            self.writes.append(record)
            real_write(record, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sr, 'read_record', _read)
        monkeypatch.setattr(sr, 'write_record', _write)

    def reads_of(self, slug: str) -> int:
        return self.reads.count(slug)


def _adopted_bound_record(slug: str, tmp_path: Path) -> dict[str, object]:
    """A record already bound to this event's stdin session_id (adopt path)."""
    write_bound_parent(slug, tmp_path, 4662030)
    return {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}


@pytest.mark.parametrize('handler', ['notification', 'stop'])
def test_refresh_path_reads_the_written_record_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: str
) -> None:
    slug = f'session-cockpit-466203{0 if handler == "notification" else 1}'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    if handler == 'notification':
        sh.run_notification({**hook_input, 'message': 'may I proceed?'}, env, root=tmp_path)
        expected = sr.Status.AWAITING_INPUT
    else:
        sh.run_stop(hook_input, env, root=tmp_path)
        expected = sr.Status.IDLE

    assert io.reads_of(slug) == 1
    # OUTCOME PARITY.
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is expected
    if handler == 'notification':
        assert record.question is not None and record.question.text == 'may I proceed?'


def test_hand_launched_refresh_writes_the_body_its_decision_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(hook_input, {}, root=tmp_path)
    slug = sh.hook_session_slug(hook_input, {}, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, {}, root=tmp_path)

    # Read 1 is the session-id pointer's vouch; read 2 is the one the decision and the write share.
    assert io.reads_of(slug) == 2
    assert [written.cwd for written in io.writes] == ['read-2']
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.IDLE


def test_refresh_writes_the_body_it_decided_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE TOCTOU INVARIANT, not merely a count.

    The withhold/bind decision and the body written must come from the SAME
    read. Today the decision is made on read #2 while refresh_record writes
    a body it re-read as #3, so a concurrent writer's change can land in
    between and be decided against stale state.
    """
    slug = 'session-cockpit-4662032'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert len(io.writes) == 1
    assert io.writes[0].cwd == 'read-1'


def test_first_bind_event_writes_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two-write case: an adoptable record that this event first BINDS.

    Status, question and binding must land in ONE body, which is what the
    handler's docstring already claimed but the two-write shape did not
    deliver.
    """
    slug = 'session-cockpit-4662033'
    # RUNNING + unbound: adoptable (not in the launch window), so this event
    # both refreshes the status and stamps the first binding.
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4662033
        ),
        root=tmp_path,
    )
    env = owner_ppid_env(slug, 4193500)
    io = _RegistryIO(monkeypatch)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=claude_process_tree(4193501, 4193500),
    )

    assert io.reads_of(slug) == 1
    assert len(io.writes) == 1
    written = io.writes[0]
    assert written.status is sr.Status.AWAITING_INPUT
    assert written.question is not None and written.question.text == 'may I proceed?'
    assert written.claude_session_id == 'uuid-owner'


def test_fork_path_reads_each_of_its_two_slugs_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACHIEVABILITY GUARD: a fork legitimately touches TWO records.

    So the bound is per-slug, never one read in total -- asserting the
    latter would be unsatisfiable without breaking the ownership split.
    """
    spawner = 'session-cockpit-4662034'
    write_bound_parent(spawner, tmp_path, 4662034)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': spawner}
    forked = sh.hook_session_slug(hook_input, env, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert io.reads_of(spawner) <= 1
    assert io.reads_of(forked) <= 1
    # The spawner's record is untouched; the fork carries the IDLE.
    assert sr.read_record(spawner, root=tmp_path).status is sr.Status.RUNNING
    assert sr.read_record(forked, root=tmp_path).status is sr.Status.IDLE


def test_refresh_never_overwrites_a_corrupt_body(tmp_path: Path) -> None:
    """CHARACTERIZATION (green before and after) -- pinned because step 12
    puts it at risk. refresh_record's "a *corrupt* existing body is NOT
    treated as absent" contract means run_stop must leave the bytes alone.
    """
    slug = 'session-cockpit-4662035'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    with contextlib.suppress(sr.CorruptSessionRecord):
        sh.run_stop(hook_input, env, root=tmp_path)

    assert record_path.read_text() == 'not-json{{{'


# ---------------------------------------------------------------------------
# The same one-read pin for SessionStart, which shares the resolution path
# (task 4662). Today the adopted path reads twice: _env_slug_ownership's
# probe read, then run_session_start's own read_record.
# ---------------------------------------------------------------------------


def test_session_start_reads_the_written_record_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug = 'session-cockpit-4662040'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**hook_input, 'source': 'startup'}, env, root=tmp_path)

    assert io.reads_of(slug) == 1
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.RUNNING


def test_session_start_writes_the_body_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAME-SNAPSHOT DERIVATION: the record the ownership/remint decision was
    made on is the record that gets written."""
    slug = 'session-cockpit-4662041'
    hook_input = _adopted_bound_record(slug, tmp_path)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**hook_input, 'source': 'startup'}, env, root=tmp_path)

    assert len(io.writes) == 1
    assert io.writes[0].cwd == 'read-1'


def test_session_start_fork_path_reads_each_slug_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = 'session-cockpit-4662042'
    write_bound_parent(spawner, tmp_path, 4662042)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': spawner}
    forked = sh.hook_session_slug(nested, env, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_session_start({**nested, 'source': 'startup'}, env, root=tmp_path)

    assert io.reads_of(spawner) <= 1
    assert io.reads_of(forked) <= 1
    assert sr.read_record(spawner, root=tmp_path).status is sr.Status.RUNNING
    assert sr.read_record(forked, root=tmp_path).parent_session_id == spawner


def test_session_start_hand_launched_reads_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, {}, root=tmp_path)
    io = _RegistryIO(monkeypatch)

    sh.run_session_start(hook_input, {}, root=tmp_path)

    assert io.reads_of(slug) == 1
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.RUNNING


def test_session_start_fault_lane_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHARACTERIZATION of test_main_session_start_fail_soft_when_probe_read_
    raises' scenario, asserted here so the single-snapshot rewrite cannot
    quietly narrow it: a probe read that raises still completes the write,
    without CLAIMING the record (adopting is fail-soft, binding is permanent).
    """
    slug = 'session-cockpit-4662043'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    real_read = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    sh.run_session_start(
        {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
    )

    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.RUNNING
    assert record.claude_session_id is None
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_session_start_first_sight_still_captures_richly(tmp_path: Path) -> None:
    """OUTCOME PARITY for the rich-capture path (no prior record)."""
    hook_input = {
        'session_id': 'sess-rich',
        'cwd': '/home/leo/src/dark-factory',
        'transcript_path': '/tmp/ignored.jsonl',
    }
    env = {
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '4662',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-4662-1',
        'CLAUDE_SPAWN_LAUNCHER_PID': '4662044',
    }

    record = sh.run_session_start(hook_input, env, root=tmp_path)

    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '4662'
    assert record.escalation_id == 'esc-4662-1'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.launcher_pid == 4662044
    assert record.start_ts
    assert record.transcript_path == sr.transcript_path_for_cwd(
        '/home/leo/src/dark-factory'
    )


# ---------------------------------------------------------------------------
# The UNREADABLE-snapshot fault lane must still honour the task-4193
# launch-window withhold guard (task 4662, reviewer finding).
#
# MEASURED on task/4662 @ 38265c5462, same scenario both sides (LAUNCHING +
# unbound spawn record, nested `claude` Notification, read_record raising on
# call #1 then succeeding):
#   main        -> status=launching,      question=None            (withheld)
#   this branch -> status=awaiting-input, question='nested question'
# i.e. the nested session's status and prompt land on the SPAWNING session's
# record -- the ownership inversion tasks 4193/2511 exist to prevent.
#
# Before task 4662 the handler made TWO independent reads (the ownership
# probe's AND _prior_record_or_none's), so a TRANSIENT fault on the first
# left the second free to succeed and the guard still ran. One snapshot now
# serves both, and one fault disarms it -- the fault lane never carried a
# withhold evaluation of its own. The existing fault-lane tests
# (test_run_stop_fail_soft_when_probe_read_raises,
# test_session_start_fault_lane_parity) both use an ABSENT record, so neither
# exercises the interaction; that gap is why this shipped.
# ---------------------------------------------------------------------------


def _boom_once_read(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Fault the FIRST ``read_record`` (the ownership probe's), then behave.

    The suite's established ``_boom_once`` idiom, hoisted so the fault-lane
    tests below share one definition of "transient fault on the probe read".
    """
    real_read = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)
    return calls


def _spawn_launching_record(slug: str, root: Path) -> None:
    """The body that makes the stake real: role/project/prompt/result_file.

    This is the record spawn-claude.sh's ``finish()`` writes ``exited`` to,
    so a nested inheritor rewriting it is not a cosmetic status lie.
    """
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            result_file='/tmp/spawn-4662.json',
            launcher_pid=4662050,
            start_ts=datetime.now(UTC).isoformat(),
        ),
        root=root,
    )


@pytest.mark.parametrize('handler', ['notification', 'stop'])
def test_fault_lane_still_withholds_from_an_unbound_launching_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: str
) -> None:
    """The direct mirror of test_refresh_path_does_not_bind_a_still_launching_
    record / ..._withholds_a_question_during_the_unbound_launch_window, with a
    transient fault injected into the ownership probe's read.

    A fault on ONE read must not cost this event its withhold DECISION. The
    record's provenance is still unknowable, so nothing this nested event
    carries -- not the status, not the question -- may land on it.
    """
    slug = f'session-cockpit-466205{0 if handler == "notification" else 1}'
    _spawn_launching_record(slug, tmp_path)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    calls = _boom_once_read(monkeypatch)

    if handler == 'notification':
        sh.run_notification({**hook_input, 'message': 'nested question'}, env, root=tmp_path)
    else:
        sh.run_stop(hook_input, env, root=tmp_path)

    # The probe was consulted (and degraded) rather than skipped.
    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.LAUNCHING
    assert record.question is None
    assert record.claude_session_id is None
    # Withholding is not forking: the event lands on this slug and is then
    # dropped -- it must not also mint a second, nested-owned row.
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_fault_lane_does_not_withhold_from_a_positively_identified_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARITY IN THE OTHER DIRECTION, so the fix cannot be "always withhold
    when the snapshot faulted".

    The owner's own pre-SessionStart Notification (task 4193 L2 item 4-iii):
    CLAUDE_SPAWN_OWNER_PPID positively identifies this event, so its status
    and question belong on the record whether or not the read faulted.
    """
    slug = 'session-cockpit-4662052'
    _spawn_launching_record(slug, tmp_path)
    env = owner_ppid_env(slug, 4193500)
    calls = _boom_once_read(monkeypatch)

    sh.run_notification(
        {
            'session_id': 'uuid-owner',
            'cwd': '/home/leo/src/dark-factory',
            'message': 'may I proceed?',
        },
        env,
        root=tmp_path,
        process_tree=claude_process_tree(4193501, 4193500),
    )

    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    # The spawn record's own payload survives the refresh untouched.
    assert record.prompt == '/spawn cockpit'
    assert record.result_file == '/tmp/spawn-4662.json'


def test_fault_lane_reads_the_written_slug_at_most_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE STRUCTURAL BOUND: the faulted probe read plus ONE retry.

    Restoring the withhold decision must not degrade into an unbounded or
    third re-read of the record. The outcome is asserted alongside the count
    so the bound cannot be met by skipping the retry.
    """
    slug = 'session-cockpit-4662053'
    _spawn_launching_record(slug, tmp_path)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    reads: list[str] = []
    real_read = sr.read_record

    def _counting_boom_once(slug_arg: str, *args: object, **kwargs: object) -> sr.SessionRecord:
        reads.append(slug_arg)
        if len(reads) == 1:
            raise OSError('disk on fire')
        return real_read(slug_arg, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _counting_boom_once)

    sh.run_stop(hook_input, env, root=tmp_path)

    assert reads.count(slug) <= 2
    # OUTCOME, so the bound cannot be satisfied by skipping the retry: the
    # retry is what supplies the observation the withhold decision needs.
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.LAUNCHING


def test_fault_lane_propagates_a_persistently_corrupt_body(tmp_path: Path) -> None:
    """STRENGTHENS test_refresh_never_overwrites_a_corrupt_body.

    That test suppresses CorruptSessionRecord, so it would also pass if the
    handler silently no-op'd. A body that is corrupt on EVERY read must still
    PROPAGATE to main()'s blanket except -- refresh_record's "a *corrupt*
    existing body is NOT treated as absent" contract -- and leave the bytes
    exactly as it found them.
    """
    slug = 'session-cockpit-4662054'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    with pytest.raises(sr.CorruptSessionRecord):
        sh.run_stop(hook_input, env, root=tmp_path)

    assert record_path.read_text() == 'not-json{{{'


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
