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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS_DIR = _REPO_ROOT / 'skills' / 'spawn' / 'hooks'


@pytest.fixture(autouse=True)
def _clear_claude_spawn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from the REAL process environment (Task 2643).

    sh.main(...) reads env=os.environ directly, so a real
    CLAUDE_SPAWN_SESSION_ID leaking in from a fleet-spawned launching
    context would make hook_session_slug prefer it over the stdin
    session_id these tests assert on, writing the registry record at a
    different slug than the test reads. Clears the CLAUDE_SPAWN_* vars
    spawn-claude.sh sets (see that script for exactly which ones it
    splices into a spawned session's real env), including
    CLAUDE_SPAWN_ROLE/PROJECT/TASK_ID (Task 2940 — an orchestrator-spawned
    launching context exports these too, and an omission here let the
    leaked identity resolve to 'implementer:dark_factory#<id>' instead of
    the clean 'session:dark-factory' default in subprocess-spawning
    tests), so this file is hermetic regardless of launching context;
    tests needing a specific value still set it explicitly via an env=
    mapping or monkeypatch.setenv, unaffected by this fixture running
    first.
    """
    monkeypatch.delenv('CLAUDE_SPAWN_SESSION_ID', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_WM_TITLE', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_RESULT_FILE', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_PARENT_ID', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_LAUNCHER_PID', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_TITLE', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_PROMPT', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_ROLE', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_PROJECT', raising=False)
    monkeypatch.delenv('CLAUDE_SPAWN_TASK_ID', raising=False)


@pytest.fixture(autouse=True)
def _isolate_fleet_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin the fleet root to this test's tmp_path (Task 4193).

    hook_session_slug now READS the session registry to decide whether an
    inherited CLAUDE_SPAWN_SESSION_ID slug is this session's own record or
    one it merely inherited, and the many call sites here that pass no
    explicit root= would otherwise resolve through
    session_registry.fleet_root to the developer's REAL ~/.claude/fleet --
    a hermeticity leak of exactly the kind the neighbouring
    _clear_claude_spawn_env fixture exists to prevent. Pinning the default
    root to the same tmp_path the tests already hand to root= also keeps
    the implicit and explicit roots in agreement. Deliberately creates no
    directories: several tests assert the fleet's sessions dir holds
    exactly one entry. Tests that set CLAUDE_FLEET_ROOT themselves simply
    win over this fixture, which runs first.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))


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
# Task 4193 step-1: SessionStart binds the owning claude session_id
# ---------------------------------------------------------------------------


def test_run_session_start_binds_claude_session_id_on_first_adoption(
    tmp_path: Path,
) -> None:
    # The first hook event to adopt a spawn-claude.sh `launching` record
    # stamps its own Claude Code session_id onto it -- the token that later
    # tells this session apart from a nested claude that merely inherited
    # CLAUDE_SPAWN_SESSION_ID.
    slug = 'session-cockpit-3215040'
    result_file = str(sr.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            cwd='/home/leo/src/dark-factory',
            launcher_pid=3215040,
            result_file=result_file,
        ),
        root=tmp_path,
    )

    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-parent'
    assert record.status == sr.Status.RUNNING
    # The rich launching-vintage fields still survive the binding write.
    assert record.role == 'session'
    assert record.project == 'cockpit'
    assert record.prompt == '/spawn cockpit'
    assert record.launcher_pid == 3215040
    assert record.result_file == result_file
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_run_session_start_binds_claude_session_id_on_hand_launched_record(
    tmp_path: Path,
) -> None:
    # No CLAUDE_SPAWN_SESSION_ID and no pre-existing record: the freshly
    # synthesized record is bound too. Harmless (that slug already embeds
    # the session_id, so it can never mismatch) and it keeps every record
    # self-describing.
    hook_input = {'session_id': 'sess-hl', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_session_start(hook_input, env={}, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'sess-hl'


def test_run_session_start_does_not_rebind_existing_claude_session_id(
    tmp_path: Path,
) -> None:
    # Bind-once: a re-fired SessionStart against an already-bound record
    # must not churn the binding.
    slug = 'session-cockpit-3215042'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215042,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


def test_run_session_start_leaves_claude_session_id_unbound_when_stdin_has_no_session_id(
    tmp_path: Path,
) -> None:
    # hook_session_slug falls back to the literal 'unknown' when stdin
    # carries no session_id; that must NEVER be bound as an owner token, or
    # every discriminator-less session would collide on one bogus binding.
    slug = 'session-cockpit-3215043'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215043),
        root=tmp_path,
    )
    hook_input = {'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    assert record.status == sr.Status.RUNNING


# ---------------------------------------------------------------------------
# Task 4193 step-2: mismatched inheritors fork their own record
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_env_slug_when_record_absent(tmp_path: Path) -> None:
    # First sight: no record at the env slug yet, so this hook event is the
    # one that will create/claim it.
    slug = 'session-cockpit-3215050'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_env_slug_when_record_unbound(tmp_path: Path) -> None:
    # A spawn-claude.sh `launching`-vintage (or pre-task-4193) record carries
    # no binding; the first matching hook adopts and binds it.
    slug = 'session-cockpit-3215051'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215051),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_env_slug_when_binding_matches(tmp_path: Path) -> None:
    slug = 'session-cockpit-3215052'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215052,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_forks_when_binding_mismatches(tmp_path: Path) -> None:
    # A nested claude inherits the whole CLAUDE_SPAWN_* namespace but arrives
    # with its OWN stdin session_id -- the only token that can tell it from
    # the session spawn-claude.sh launched. It must fall through to the
    # unchanged hand-launched keying instead of adopting the parent's slug.
    slug = 'session-cockpit-3215053'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215053,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }

    forked = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked == sr.build_session_slug(
        'unblock',
        'df',
        '2085',
        'uuid-nested',  # type: ignore[arg-type]
    )
    assert forked != slug


def test_run_session_start_nested_inheritor_forks_leaving_parent_untouched(
    tmp_path: Path,
) -> None:
    # The blast radius: run_session_start overwrites status/parent_session_id/
    # display on whatever record the slug resolves to. A nested claude's
    # SessionStart must land on its OWN record, leaving the parent's byte-
    # identical.
    slug = 'session-cockpit-3215041'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            title='parent-title-marker',
            launcher_pid=3215041,
            parent_session_id='root-df-1-1',
            display=sr.Display(kind='wm', wm_title='parent-marker', wm_window_id='0x1a'),
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    snapshot = sr.read_record(slug, root=tmp_path).to_dict()

    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    # CLAUDE_SPAWN_PARENT_ID/CLAUDE_SPAWN_TITLE/WINDOWID are the concrete
    # overwrite vectors: under the pre-task-4193 code they all landed on the
    # parent's record.
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_PARENT_ID': 'some-other-root',
        'CLAUDE_SPAWN_TITLE': 'nested-title',
        'WINDOWID': '0x99',
    }

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).to_dict() == snapshot

    forked_slug = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked_slug != slug
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.status == sr.Status.RUNNING
    assert forked.claude_session_id == 'uuid-nested'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


# ---------------------------------------------------------------------------
# Task 4193 step-3: Stop/Notification honour the record binding
# ---------------------------------------------------------------------------


def test_run_notification_and_stop_keep_writing_to_the_bound_record(
    tmp_path: Path,
) -> None:
    # The owning session's own events still converge on its record, question
    # capture included -- the ownership probe must not disturb the task-2511
    # convergence it gates.
    slug = 'session-cockpit-3215060'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215060,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-parent',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'approve rollout?',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_notification(hook_input, env, root=tmp_path)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT
    assert record.question is not None
    assert record.question.text == 'approve rollout?'

    sh.run_stop(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_run_stop_from_nested_inheritor_does_not_idle_the_parent(tmp_path: Path) -> None:
    # The sharpest statement of the bug: a nested claude finishing its turn
    # must not advertise the spawning session as idle mid-turn.
    slug = 'session-cockpit-3215061'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215061,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
    forked_slug = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked_slug != slug
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.status == sr.Status.IDLE
    assert forked.claude_session_id == 'uuid-nested'


def test_run_notification_from_nested_inheritor_does_not_flip_the_parent(
    tmp_path: Path,
) -> None:
    slug = 'session-cockpit-3215062'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215062,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'nested asks something',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_notification(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.status == sr.Status.RUNNING
    assert parent.question is None
    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.status == sr.Status.AWAITING_INPUT
    assert forked.question is not None
    assert forked.question.text == 'nested asks something'


def test_refresh_path_binds_a_legacy_unbound_record(tmp_path: Path) -> None:
    # Migration safety: an in-flight session whose record predates task 4193
    # (or was written by spawn-claude.sh's `launching`) is bound by the very
    # next Notification/Stop, without waiting for a SessionStart that may
    # never come again -- so the next nested event is already discriminable.
    slug = 'session-cockpit-3215063'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.RUNNING, launcher_pid=3215063),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.IDLE
    assert record.claude_session_id == 'uuid-parent'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_refresh_path_does_not_rebind_or_write_when_already_bound(tmp_path: Path) -> None:
    slug = 'session-cockpit-3215064'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215064,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


# ---------------------------------------------------------------------------
# Task 4193 step-4: ownership probe is fail-soft
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_when_record_is_corrupt(tmp_path: Path) -> None:
    # A corrupt body must degrade to the pre-task-4193 behaviour (adopt), not
    # fork a spurious record; the reaper already has its own 'corrupt' rule.
    slug = 'session-cockpit-3215070'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    with pytest.raises(sr.CorruptSessionRecord):
        sr.read_record(slug, root=tmp_path)

    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_when_read_record_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slug = 'session-cockpit-3215071'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'read_record', _boom)

    with caplog.at_level(logging.WARNING):
        assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug

    # Degradation is never silent (repo's no-silent-fail-soft norm).
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_run_stop_fail_soft_when_probe_read_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end: a fault in the OWNERSHIP PROBE's read must never be what
    # breaks Stop. Scoped to the probe's own read (the first of the two
    # read_record calls run_stop makes: the probe reads, then
    # refresh_record does) because refresh_record deliberately PROPAGATES a
    # corrupt/unreadable body rather than overwriting it -- see its
    # docstring, "A *corrupt* existing body is NOT treated as absent" -- and
    # main()'s blanket except is the backstop for that pre-existing case.
    slug = 'session-cockpit-3215072'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    real_read_record = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    assert sh.run_stop(hook_input, env, root=tmp_path).startswith('\033]0;')
    # The probe was consulted (and degraded) rather than skipped.
    assert len(calls) >= 1
    # Degrading to adopt means the env slug still owns the write.
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE


def test_main_session_start_fail_soft_when_probe_read_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The CLI backstop, scoped to the PROBE's own read (the first of the two
    # read_record calls session-start makes). Booby-trapping every read
    # instead would prove nothing: main()'s pre-existing blanket except
    # would return 0 even if the probe propagated, and the hook would be a
    # no-op rather than fail-soft. So the assertion here is the OUTCOME --
    # the degraded probe still let SessionStart complete its write.
    slug = 'session-cockpit-3215073'
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_SPAWN_SESSION_ID', slug)
    _stdin_json(monkeypatch, {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'})

    real_read_record = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    assert sh.main(['session-start']) == 0

    # The probe was consulted (and degraded), and adopting the env slug
    # unchecked is exactly the pre-task-4193 behaviour: one record, written.
    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    assert record.claude_session_id == 'uuid-parent'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_hand_launched_session_unaffected_by_ownership_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A hand-launched session (no CLAUDE_SPAWN_SESSION_ID) has nothing to
    # probe and must pay no probe cost at all -- the must-not-be-called
    # idiom, not merely an assertion that the outcome is unchanged. Note
    # run_session_start legitimately calls sr.read_record itself, so it is
    # sh._env_slug_is_owned that is booby-trapped here, not the registry.
    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError('ownership probe must not run for a hand-launched session')

    monkeypatch.setattr(sh, '_env_slug_is_owned', _boom)

    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, env={}, root=tmp_path)

    sh.run_session_start(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING

    sh.run_notification(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.AWAITING_INPUT

    sh.run_stop(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE

    assert slug == sr.build_session_slug(
        'session',
        'dark-factory',
        None,
        'sess-hand',  # type: ignore[arg-type]
    )
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_hook_session_slug_adopts_when_stdin_has_no_session_id(tmp_path: Path) -> None:
    # With no discriminator the only safe answer is today's behaviour --
    # forking here would key every discriminator-less session on the one
    # 'unknown' literal hook_session_slug falls back to.
    slug = 'session-cockpit-3215074'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215074,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


# ---------------------------------------------------------------------------
# Task 4193 amendment: one definition of a usable stdin session_id
# ---------------------------------------------------------------------------


def test_hook_session_id_normalizes_every_missing_shape() -> None:
    # The probe, the binding stamp and the slug builder must agree on what
    # counts as a discriminator, so all the "no session_id" shapes collapse
    # to the same empty string.
    assert sh._hook_session_id({}) == ''
    assert sh._hook_session_id({'session_id': None}) == ''
    assert sh._hook_session_id({'session_id': ''}) == ''
    assert sh._hook_session_id({'session_id': '   '}) == ''
    assert sh._hook_session_id({'session_id': ' uuid-x '}) == 'uuid-x'


def test_hook_session_slug_treats_a_blank_session_id_as_unknown(tmp_path: Path) -> None:
    # A whitespace-only stdin session_id used to reach build_session_slug
    # verbatim (no .strip()), yielding a slug with a blank token while the
    # two 4193 helpers already read it as "no discriminator at all".
    blank = sh.hook_session_slug({'session_id': '   ', 'cwd': '/home/leo/src/df'}, {}, root=tmp_path)
    absent = sh.hook_session_slug({'cwd': '/home/leo/src/df'}, {}, root=tmp_path)
    assert blank == absent
    assert blank.endswith('-unknown')


def test_bind_claude_session_id_ignores_a_blank_stdin_session_id() -> None:
    record = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING)
    assert sh._bind_claude_session_id(record, {'session_id': '   '}) is False
    assert record.claude_session_id is None


# ---------------------------------------------------------------------------
# Task 4193 amendment: /clear re-mints session_id inside the OWNING process
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_env_slug_when_a_clear_remints_the_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /clear mints a new session_id in the SAME process, so the owning
    # session's own SessionStart legitimately mismatches its binding.
    # Reading that as an inheritor would re-introduce the task-2511 split.
    # "SAME process" is load-bearing and now checked: the bound
    # claude_owner_pid must match the pid resolved for this event.
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 424_242)
    slug = 'session-cockpit-3215080'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215080,
            claude_session_id='uuid-before-clear',
            claude_owner_pid=424_242,
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-after-clear',
        'source': 'clear',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path, allow_remint=True) == slug
    # ... but only where a `source` field means anything. Without the
    # SessionStart-only opt-in the same event is a plain mismatch and forks.
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) != slug


def test_hook_session_slug_does_not_forgive_a_resume_remint(tmp_path: Path) -> None:
    # `allow_remint` is not blanket forgiveness -- it is scoped to the two
    # sources with no CLI spelling. `--resume`/`--continue` make a brand-new
    # nested process report 'resume' too, so honouring it here would let any
    # inheritor claim its spawner's record even with the opt-in withheld.
    slug = 'session-cockpit-3215087'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215087,
            claude_session_id='uuid-before',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'resume',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path, allow_remint=True) != slug
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) != slug


@pytest.mark.parametrize('source', ['clear', 'compact'])
def test_run_session_start_rebinds_on_an_owner_only_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    # ... and the record is RE-bound to the new id, so the session stays
    # discriminable from a nested claude on every subsequent event.
    #
    # claude_owner_pid is stamped to the SAME process this hook resolves,
    # which is what makes the re-mint provably the owner's. Source alone is
    # not enough: auto-compaction fires source='compact' unprompted, so a
    # nested claude reaches this path too and must NOT be forgiven -- see
    # test_nested_compact_cannot_invert_ownership.
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 424_242)
    slug = f'session-cockpit-321508{["clear", "compact"].index(source) + 1}'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215081,
            claude_session_id='uuid-before',
            claude_owner_pid=424_242,
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-after',
        'source': source,
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-after'
    assert record.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_run_session_start_startup_source_with_mismatched_id_still_forks(
    tmp_path: Path,
) -> None:
    # The other direction: a nested claude is always a fresh process and so
    # always reports source='startup'. It must still fork.
    slug = 'session-cockpit-3215084'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215084,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'startup',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


def test_run_session_start_resume_source_with_mismatched_id_still_forks(
    tmp_path: Path,
) -> None:
    # `resume` is NOT owner-only -- `--resume`/`--continue` make a brand-new
    # process report it, so a nested `claude -c -p ...` would otherwise
    # rebind the spawner's record to itself. It is deliberately excluded from
    # _OWNER_ONLY_SESSION_START_SOURCES; a spawned session /resume'd in place
    # simply forks, which is the fail-safe direction.
    slug = 'session-cockpit-3215086'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215086,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'resume',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


def test_refresh_path_ignores_a_forged_owner_only_source(tmp_path: Path) -> None:
    # Notification/Stop carry no `source` field, so nothing there can vouch
    # for the caller: the re-mint escape hatch is SessionStart-only. A
    # forged one on a Stop must not be honoured -- otherwise any inheritor
    # gets a one-word bypass of the whole ownership check.
    slug = 'session-cockpit-3215085'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215085,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'clear',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


# ---------------------------------------------------------------------------
# Task 4193 amendment: a forked record describes ITSELF, not its spawner
# ---------------------------------------------------------------------------


def _write_bound_parent(slug: str, tmp_path: Path, pid: int) -> None:
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=pid,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )


def test_forked_inheritor_records_its_spawner_as_parent(tmp_path: Path) -> None:
    # CLAUDE_SPAWN_PARENT_ID names the SPAWNER's OWN parent, so copying it
    # onto the fork would render the nested session as its spawner's
    # sibling. The fork path is the one place the true parent is known
    # exactly: it is the env slug that was just rejected.
    slug = 'session-cockpit-3215090'
    _write_bound_parent(slug, tmp_path, 3215090)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_PARENT_ID': 'some-other-root'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.parent_session_id == slug
    # And the spawning session's own record still points where it did.
    assert sr.read_record(slug, root=tmp_path).parent_session_id is None


def test_forked_inheritor_does_not_claim_the_spawners_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CLAUDE_SPAWN_WM_TITLE marks the SPAWNING session's window: resolving
    # it here would give the forked row the parent's window id, so cockpit
    # focus would raise the parent's terminal and
    # mark_windowless_wm_sessions_exited would treat both rows as one
    # window. The resolver must not even be consulted.
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError('a forked inheritor must not resolve the spawner window marker')

    monkeypatch.setattr(sh, '_resolve_wm_window_id', _boom)

    slug = 'session-cockpit-3215091'
    _write_bound_parent(slug, tmp_path, 3215091)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_WM_TITLE': 'parent-marker'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.display is None


def test_adopted_session_still_resolves_the_spawn_window_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate is fork-only: the owning session's own SessionStart keeps the
    # task-2510 / Cockpit C10 marker resolution untouched.
    monkeypatch.setattr(sh, '_resolve_wm_window_id', lambda _marker: '0x1a')

    slug = 'session-cockpit-3215092'
    _write_bound_parent(slug, tmp_path, 3215092)
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_WM_TITLE': 'parent-marker'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    display = sr.read_record(slug, root=tmp_path).display
    assert display is not None
    assert display.wm_title == 'parent-marker'
    assert display.wm_window_id == '0x1a'


# ---------------------------------------------------------------------------
# Task 4193 amendment: a forked record must stay reapable
# ---------------------------------------------------------------------------


def test_parent_pid_of_resolves_this_process_parent() -> None:
    assert sh._parent_pid_of(os.getpid()) == os.getppid()


def test_parent_pid_of_returns_none_for_an_unresolvable_pid() -> None:
    # Never raises: an absent /proc entry (or no /proc at all) is a plain
    # "cannot tell", which every caller treats as a best-effort miss.
    assert sh._parent_pid_of(-1) is None


def test_nested_claude_liveness_pid_prefers_the_grandparent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # claude -> <hook>.sh -> python, so the grandparent IS the nested claude
    # whose death should make the forked record reclaimable.
    monkeypatch.setattr(sh, '_parent_pid_of', lambda _pid: 424242)
    assert sh._nested_claude_liveness_pid() == 424242


@pytest.mark.parametrize('unresolvable', [None, 1])
def test_nested_claude_liveness_pid_falls_back_to_the_durable_pid(
    monkeypatch: pytest.MonkeyPatch,
    unresolvable: int | None,
) -> None:
    # No /proc, a mid-read race, or a hook reparented to init: degrade to
    # the coarse-but-durable pid rather than guessing.
    monkeypatch.setattr(sh, '_parent_pid_of', lambda _pid: unresolvable)
    assert sh._nested_claude_liveness_pid() == sh._hand_launched_liveness_pid()


def test_forked_inheritor_record_gets_a_pid_that_dies_with_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _hand_launched_liveness_pid's session leader outlives every nested
    # claude, so a fork stamped with it would never satisfy
    # reap_stale_records' stale_pid rule -- one permanent extra row per
    # nested `claude -p` an agent shells out to.
    monkeypatch.setattr(sh, '_nested_claude_liveness_pid', lambda: 424242)

    slug = 'session-cockpit-3215093'
    _write_bound_parent(slug, tmp_path, 3215093)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    # An inherited CLAUDE_SPAWN_LAUNCHER_PID is the SPAWNER's, so it must
    # not win here either.
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_LAUNCHER_PID': '3215093'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.launcher_pid == 424242


def test_hand_launched_record_keeps_the_durable_session_leader_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The nested-pid resolver is fork-only: a hand-launched session still
    # gets the terminal-lifetime pid _hand_launched_liveness_pid documents.
    def _boom() -> int:
        raise AssertionError('the nested-claude pid resolver is for forked inheritors only')

    monkeypatch.setattr(sh, '_nested_claude_liveness_pid', _boom)
    monkeypatch.setattr(sh, '_hand_launched_liveness_pid', lambda: 515151)

    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(hook_input, env={}, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).launcher_pid == 515151


# ---------------------------------------------------------------------------
# Task 4193 amendment: an inheritor cannot claim an unsighted launching record
# ---------------------------------------------------------------------------


def test_refresh_path_does_not_bind_a_still_launching_record(tmp_path: Path) -> None:
    # spawn-claude.sh's `launching` write has not been sighted by its
    # owner's SessionStart yet (delayed, timed out under the 10s hook
    # timeout, or failed). A nested claude's Stop landing first must not be
    # allowed to bind ITS id into that record -- that would invert ownership
    # permanently, since the launching record is the one holding
    # role/project/prompt/result_file and the one finish() writes `exited`
    # to.
    slug = 'session-cockpit-3215100'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            launcher_pid=3215100,
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    # ...and the STATUS is withheld too, not just the binding. Asserting only
    # the binding once let the real defect through: refresh_record is a
    # read-modify-WRITE, so a guard sitting after it still left the nested
    # Stop advertising a still-launching spawn as `idle` in the cockpit.
    assert record.status == sr.Status.LAUNCHING
    # Blast radius, stated explicitly: withholding is not forking. The probe
    # ADOPTS an unbound record (that is the legacy-migration path), so the
    # event lands on this slug and is then dropped -- it must not also mint a
    # second, nested-owned row for the cockpit to show alongside the spawn.
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_refresh_path_withholds_a_question_during_the_unbound_launch_window(
    tmp_path: Path,
) -> None:
    # The question is the attention rail's payload: a nested claude's prompt
    # landing on the spawner's record would make the cockpit claim the SPAWN
    # is the thing awaiting input. Withheld on the same unknowable-provenance
    # grounds as the status and the binding.
    slug = 'session-cockpit-3215103'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215103
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Do you want to proceed?',
    }

    sh.run_notification(hook_input, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.question is None
    assert record.status == sr.Status.LAUNCHING
    assert record.claude_session_id is None


def test_refresh_path_writes_normally_to_a_bound_launching_record(
    tmp_path: Path,
) -> None:
    # The window is LAUNCHING *and unbound*. A LAUNCHING record that already
    # carries a binding HAS a discriminator, so _env_slug_is_owned has proved
    # this event belongs to the owner -- withholding there would strand a
    # legitimately-owned record at `launching`.
    slug = 'session-cockpit-3215104'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215104,
            claude_session_id='uuid-owner',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_stop(hook_input, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.IDLE
    assert record.claude_session_id == 'uuid-owner'


def test_owning_session_start_still_wins_the_launching_record_after_a_nested_stop(
    tmp_path: Path,
) -> None:
    # The whole inversion story end to end: the nested event refrains, then
    # the owner's SessionStart binds, and from there the nested claude forks
    # onto its own record exactly as it should.
    slug = 'session-cockpit-3215101'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215101,
            result_file='/tmp/result-3215101.md',
        ),
        root=tmp_path,
    )
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    owner = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_stop(nested, env, root=tmp_path)
    # Assert HERE, before run_session_start's unconditional `status = RUNNING`
    # masks it: that reset is what hid the original defect from this test.
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING
    sh.run_session_start(owner, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.result_file == '/tmp/result-3215101.md'

    # The nested claude's next event now forks instead of flipping it.
    sh.run_stop(nested, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
    forked = sr.read_record(sh.hook_session_slug(nested, env, root=tmp_path), root=tmp_path)
    assert forked.status == sr.Status.IDLE
    assert forked.claude_session_id == 'uuid-nested'


def test_refresh_path_still_binds_a_legacy_record_that_is_past_launching(
    tmp_path: Path,
) -> None:
    # The LAUNCHING guard must not cost the migration case it sits next to:
    # a pre-task-4193 in-flight record is RUNNING/AWAITING_INPUT/IDLE, never
    # LAUNCHING, so it is still bound by the very next event.
    slug = 'session-cockpit-3215102'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.AWAITING_INPUT, launcher_pid=3215102
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_stop(hook_input, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


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
    assert record.display.wm_window_id == '0x3200007'
    assert record.display.wm_title == 'session:dark-factory'


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
    assert record.display.wm_window_id == '0x3200007'
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
    assert record.display.wm_window_id == '0x3200007'


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
    *,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env['CLAUDE_FLEET_ROOT'] = str(root)
    if extra_env:
        env.update(extra_env)
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


# ---------------------------------------------------------------------------
# Task 2511 step-3: bash-level CLAUDE_SPAWN_SESSION_ID convergence
# ---------------------------------------------------------------------------


def test_session_start_sh_converges_onto_pre_written_launching_record(tmp_path: Path) -> None:
    # Proves CLAUDE_SPAWN_SESSION_ID propagates through the real .sh -> python
    # wiring (not just the pure-python run_session_start path exercised
    # above): a pre-written LAUNCHING record at slug S, keyed the way
    # spawn-claude.sh's own `launching` write keys it, must advance to
    # RUNNING via the real session-start.sh entrypoint -- with a DIFFERENT
    # Claude Code session_id -- when CLAUDE_SPAWN_SESSION_ID=S is in env.
    slug = 'session-cockpit-3215099'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215099),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-bash-convergence', 'cwd': '/home/leo/src/dark-factory'}

    result = _run_hook_script(
        'session-start.sh',
        hook_input,
        tmp_path,
        extra_env={'CLAUDE_SPAWN_SESSION_ID': slug},
    )

    assert result.returncode == 0
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    # No separate uuid-keyed duplicate.
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_nested_compact_cannot_invert_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Automatic compaction fires SessionStart source='compact' with NO user
    # action, so any nested `claude -p` that runs long enough reaches the
    # remint path. Source alone proves the emitter holds *a* session, not
    # *this* one -- so without the pid condition a nested compact REBOUND the
    # spawner's record to itself and forced the true owner to fork, i.e. a
    # full ownership inversion plus the task-2511 split, reached by routine
    # agent behaviour with no hook fault.
    slug = 'session-cockpit-4400010'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4400010,
            result_file='/tmp/result-4400010.md',
        ),
        root=tmp_path,
    )
    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(owner, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).claude_owner_pid is not None

    # A DIFFERENT process (the nested claude) presenting an auto-compact.
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 999_001)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(nested, env, root=tmp_path)

    spawner = sr.read_record(slug, root=tmp_path)
    assert spawner.claude_session_id == 'uuid-owner'      # NOT rebound
    assert spawner.result_file == '/tmp/result-4400010.md'
    forked = sr.read_record(sh.hook_session_slug(nested, env, root=tmp_path), root=tmp_path)
    assert forked.claude_session_id == 'uuid-nested'


def test_owner_auto_compact_still_converges_on_its_own_record(tmp_path: Path) -> None:
    # The other direction, and the reason 'compact' is not simply dropped
    # from the owner-only set: auto-compaction is routine, so forking on it
    # would re-introduce the task-2511 split for the OWNER on a far more
    # common event than the inversion it would prevent. Same process => same
    # pid => the re-mint is forgiven and the record stays converged.
    slug = 'session-cockpit-4400011'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4400011),
        root=tmp_path,
    )
    sh.run_session_start({'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}, env, root=tmp_path)

    remint = {'session_id': 'uuid-owner-2', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-owner-2'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_legacy_record_without_owner_pid_adopts_rather_than_splitting_the_owner(
    tmp_path: Path,
) -> None:
    # A record bound before claude_owner_pid existed cannot prove ownership
    # EITHER WAY. Task 4193 L2 ruling item 4-ii: unprovable is not disproved,
    # so it must take the fail-soft direction this module documents at
    # _env_slug_is_owned ("every failure mode resolves to adopt, never to
    # fork"). Forking here would split the OWNER's own record on every
    # routine automatic compaction wherever /proc is unavailable (macOS) --
    # a universal regression, strictly worse than the rare inversion it
    # would prevent. The fork is reserved for a pid that RESOLVES and
    # mismatches (see the test below).
    slug = 'session-cockpit-4400012'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4400012,
            claude_session_id='uuid-owner',  # bound, but no owner pid
        ),
        root=tmp_path,
    )
    remint = {'session_id': 'uuid-other', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-other'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_unresolvable_current_owner_pid_adopts_rather_than_splitting_the_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mirror of the case above, and the one the pre-existing pin could
    # never observe (both _owning_claude_pid() calls happened inside one
    # pytest process): the record HAS an owner pid, but the CURRENT probe
    # cannot resolve one -- exactly macOS, where _parent_pid_of has no /proc
    # to read. Unprovable again means adopt.
    slug = 'session-cockpit-4400013'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4400013,
            claude_session_id='uuid-owner', claude_owner_pid=4400099,
        ),
        root=tmp_path,
    )
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: None)
    remint = {'session_id': 'uuid-owner-2', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-owner-2'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


# ---------------------------------------------------------------------------
# Task 4193 L2 ruling item 4: the UNBOUND launch window.
#
# Before the owner's own SessionStart lands, the spawn-created record carries
# no claude_session_id, so the stdin-session_id discriminator does not exist
# yet and the FIRST event to arrive captures the record -- owner or inheritor.
# The stateless CLAUDE_SPAWN_OWNER_PPID probe decides there instead.
# ---------------------------------------------------------------------------


def _owner_ppid_env(slug: str, ppid: int) -> dict[str, str]:
    return {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_OWNER_PPID': str(ppid)}


def _launching_record(slug: str, root: Path, *, pid: int = 4193001) -> None:
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=pid,
            role='cockpit',
            start_ts=datetime.now(UTC).isoformat(),
        ),
        root=root,
    )


def test_nested_session_start_cannot_capture_an_unbound_launching_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE REVIEWER'S BLOCKING CASE (esc-4193-9). SessionStart is the event
    # that reaches the launch window FIRST -- a nested claude always fires
    # its own SessionStart before any Notification/Stop -- so a guard wired
    # only into the refresh path never gets a chance to fire. Here the
    # nested claude's SessionStart arrives at a LAUNCHING+unbound record and
    # must NOT bind it: it forks onto its own slug, leaving the record that
    # holds role/prompt/result_file (the one spawn-claude.sh writes 'exited'
    # to) untouched and still available to its true owner.
    slug = 'session-cockpit-4193001'
    _launching_record(slug, tmp_path)
    env = _owner_ppid_env(slug, 4193500)
    # The nested claude's parent is its agent's Bash-tool shell, not the
    # payload bash spawn-claude.sh exported.
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193777)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193776)

    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path)

    spawn_record = sr.read_record(slug, root=tmp_path)
    assert spawn_record.claude_session_id is None
    assert spawn_record.status is sr.Status.LAUNCHING
    assert spawn_record.role == 'cockpit'
    forked = sr.read_record(sh.hook_session_slug(nested, env, root=tmp_path), root=tmp_path)
    assert forked.claude_session_id == 'uuid-nested'
    assert forked.parent_session_id == slug


def test_owner_session_start_binds_the_unbound_launching_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other half: the session spawn-claude.sh actually launched is the
    # direct child of the payload bash whose pid it exported, so it adopts
    # and binds the spawn-created record -- exactly one row, converged.
    slug = 'session-cockpit-4193002'
    _launching_record(slug, tmp_path, pid=4193002)
    env = _owner_ppid_env(slug, 4193500)
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193501)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193500)

    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-owner'
    assert record.status is sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_nested_then_owner_session_start_leaves_the_spawn_record_with_the_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The end-to-end ordering the reviewer reproduced against the branch:
    # nested SessionStart -> nested Stop -> owner SessionStart. Previously
    # the nested SessionStart inverted ownership permanently. Now the spawn
    # record survives untouched until its real owner arrives.
    slug = 'session-cockpit-4193003'
    _launching_record(slug, tmp_path, pid=4193003)
    env = _owner_ppid_env(slug, 4193500)

    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193777)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193776)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path)
    sh.run_stop(nested, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.LAUNCHING

    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193501)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193500)
    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-owner'
    assert record.status is sr.Status.RUNNING


def test_owner_stop_is_not_withheld_from_a_still_launching_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ruling item 4-iii. The launch-window withholding suppressed status AND
    # the pending question for EVERY event, including the legitimate
    # owner's. A SessionStart killed by _HOOK_TIMEOUT_SECS (reachable:
    # _resolve_wm_window_id's own docstring prices the pathological wmctrl
    # case at ~10.8s against a 10s budget) writes nothing, so the record
    # stays LAUNCHING+unbound and the session went invisible for its whole
    # life. An event whose OWNER_PPID matches must write.
    slug = 'session-cockpit-4193004'
    _launching_record(slug, tmp_path, pid=4193004)
    env = _owner_ppid_env(slug, 4193500)
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193501)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193500)

    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_notification({**owner, 'message': 'may I proceed?'}, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    assert record.claude_session_id == 'uuid-owner'


def test_unknown_provenance_event_is_still_withheld_inside_the_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No verdict at all (a session spawned by a pre-task-4193
    # spawn-claude.sh, or a platform with no /proc): provenance is genuinely
    # unknowable, so the conservative withholding stands.
    slug = 'session-cockpit-4193005'
    _launching_record(slug, tmp_path, pid=4193005)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: None)

    sh.run_stop({'session_id': 'uuid-anon', 'cwd': '/home/leo/src/dark-factory'}, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.LAUNCHING
    assert record.claude_session_id is None


def test_withholding_expires_so_a_stuck_record_is_never_blind_forever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing else bounds the window: no LAUNCHING reaper, and
    # reap_stale_records needs a DEAD launcher_pid, which is
    # spawn-claude.sh's own $$ -- alive for the whole session. Past the
    # bound, a possibly-wrong status beats a permanently blind record.
    slug = 'session-cockpit-4193006'
    stale = datetime.now(UTC) - timedelta(seconds=sh._LAUNCH_WINDOW_WITHHOLD_MAX_SECS + 60)
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4193006,
            start_ts=stale.isoformat(),
        ),
        root=tmp_path,
    )
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: None)

    sh.run_stop({'session_id': 'uuid-late', 'cwd': '/home/leo/src/dark-factory'}, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.IDLE
    assert record.claude_session_id == 'uuid-late'


def test_sibling_mode_running_unbound_record_is_still_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ruling item 4-i: resolve_sibling() runs `refresh --status running`
    # immediately at launch on EVERY backend branch, so a
    # CLAUDE_SPAWN_MODE=sibling record is RUNNING-and-unbound and a
    # `status is LAUNCHING` predicate never matches it at all. Keying the
    # ownership probe on UNBOUND-ness instead covers it.
    slug = 'session-cockpit-4193007'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4193007, role='cockpit'
        ),
        root=tmp_path,
    )
    env = _owner_ppid_env(slug, 4193500)
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193777)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193776)

    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id is None
    assert sr.read_record(
        sh.hook_session_slug(nested, env, root=tmp_path), root=tmp_path
    ).claude_session_id == 'uuid-nested'


def test_owner_ppid_verdict_is_none_for_every_unresolvable_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail-soft: never False (fork) on an input the probe simply cannot read.
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193501)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: 4193500)
    assert sh._owner_ppid_verdict({}) is None
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '   '}) is None
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': 'not-a-pid'}) is None
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '1'}) is None
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}) is True
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193999'}) is False

    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: None)
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}) is None
    monkeypatch.setattr(sh, '_owning_claude_pid', lambda: 4193501)
    monkeypatch.setattr(sh, '_parent_pid_of', lambda pid: None)
    assert sh._owner_ppid_verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}) is None
