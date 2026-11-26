"""Tests for orchestrator.session_registry module.

Covers: CLI subcommands + fail-soft, the reap and lease-* verbs included; the
bare-shell entrypoint registry; and the G5 two-way boundary test (write by a
real spawn -> refresh by a simulated hook -> reap), exercised jointly with
tests/scripts/test_spawn_claude.py's bash-level harness. The record schema and
store are covered in test_session_records.py, the lease library in
test_session_leases.py, the reaper in test_session_reaper.py.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import pytest
from _decision_fixtures import EVIDENCE, make_decision
from _process_identity_variants import own_identity
from _reap_report_wait import await_reap_report
from _session_record_fixtures import make_record, set_mtime, write_t3_record
from shared.project_id import PROJECT_ID_ALIASES, fold_project_token

from orchestrator import decision_registry as dr
from orchestrator import session_leases, session_reaper, session_records
from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A pid virtually guaranteed dead on any host (matches the convention in
# shared/tests/test_proc_liveness.py: 2**31 - 1).
_DEAD_PID = 2**31 - 1

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)


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
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = session_records.record_path_for_slug(slug, root=tmp_path).parent
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected_dir)

    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status.LAUNCHING
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.escalation_id == 'esc-9'
    assert record.title == 'unblock:df#2085 routing-mechanism'
    assert record.prompt == '/unblock 2085'
    assert record.cwd == '/home/leo/src/dark-factory'
    assert record.launcher_pid == 4242


def test_main_launching_stamps_the_launcher_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path, launcher_pid=str(os.getpid())))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = session_records.build_session_slug('unblock', 'df', '2085', os.getpid())
    record_file = session_records.record_path_for_slug(slug, root=tmp_path)
    assert capsys.readouterr().out.strip() == str(record_file.parent)
    assert session_records.read_record(slug, root=tmp_path).launcher_identity == own_identity()
    on_disk = json.loads(record_file.read_text())
    assert on_disk['launcher_identity']['host'] == socket.gethostname()


def test_main_launching_with_a_dead_launcher_still_writes_an_unqualified_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path, launcher_pid=str(_DEAD_PID)))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = session_records.build_session_slug('unblock', 'df', '2085', _DEAD_PID)
    expected_dir = session_records.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out.strip() == str(expected_dir)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.launcher_pid == _DEAD_PID
    assert record.launcher_identity is None


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
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status.EXITED
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

    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    expected_result_file = str(session_records.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')

    launching_record = session_records.read_record(slug, root=tmp_path)
    assert launching_record.result_file is not None
    assert launching_record.result_file == expected_result_file

    rc = sr.main(['exit', '--record', record_dir, '--code', '3'])
    assert rc == 0

    exited_record = session_records.read_record(slug, root=tmp_path)
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
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status.RUNNING


@pytest.mark.parametrize('live_status', ['running', 'idle', 'awaiting-input'])
def test_main_refresh_clears_exit_code_when_record_leaves_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    live_status: str,
) -> None:
    """A live session falsely swept EXITED/ORPHAN_EXIT_CODE then flipped back
    by a hook must not keep the sweep's terminal exit code."""
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    session_records.update_status(
        slug, root=tmp_path, status=session_records.Status.EXITED, exit_code=session_reaper.ORPHAN_EXIT_CODE
    )

    rc = sr.main(['refresh', '--record', record_dir, '--status', live_status])

    assert rc == 0
    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status(live_status)
    assert record.exit_code is None


def test_main_refresh_repairs_a_live_record_left_holding_an_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A record already on disk as running/ORPHAN_EXIT_CODE has a non-terminal
    prior, yet its next refresh still drops the stale exit code."""
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    launched = session_records.read_record(slug, root=tmp_path)
    session_records.write_record(
        dataclasses.replace(
            launched, status=session_records.Status.RUNNING, exit_code=session_reaper.ORPHAN_EXIT_CODE
        ),
        root=tmp_path,
    )

    rc = sr.main(['refresh', '--record', record_dir, '--status', 'idle'])

    assert rc == 0
    record = session_records.read_record(slug, root=tmp_path)
    assert (record.status, record.exit_code) == (session_records.Status.IDLE, None)


def test_main_refresh_keeps_exit_code_while_record_stays_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    session_records.update_status(slug, root=tmp_path, status=session_records.Status.EXITED, exit_code=3)

    rc = sr.main(['refresh', '--record', record_dir, '--status', 'exited'])

    assert rc == 0
    record = session_records.read_record(slug, root=tmp_path)
    assert (record.status, record.exit_code) == (session_records.Status.EXITED, 3)


def test_main_set_display_stamps_and_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fleet Cockpit C6: the `set-display` verb stamps display post-hoc (the
    tmux target is only known after `tmux new-window` runs, well after
    `launching` allocated the record), and a later `exit` read-modify-write
    must preserve it (both re-read the whole record and re-write it).
    """
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(
        [
            'set-display',
            '--record',
            record_dir,
            '--kind',
            'tmux',
            '--tmux-target',
            'fleet-df:2',
            '--wm-title',
            'watcher:df#2085',
        ]
    )

    assert rc == 0
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'tmux'
    assert record.display.tmux_target == 'fleet-df:2'
    assert record.display.wm_title == 'watcher:df#2085'
    assert record.status == session_records.Status.LAUNCHING
    assert record.session_slug == slug

    rc = sr.main(['exit', '--record', record_dir, '--code', '0'])

    assert rc == 0
    exited_record = session_records.read_record(slug, root=tmp_path)
    assert exited_record.status == session_records.Status.EXITED
    assert exited_record.display is not None
    assert exited_record.display.kind == 'tmux'
    assert exited_record.display.tmux_target == 'fleet-df:2'


@pytest.mark.parametrize(
    ('given', 'stored'),
    [
        pytest.param('52428807', '0x03200007', id='decimal'),
        pytest.param('0x3200007', '0x03200007', id='unpadded-hex'),
        pytest.param('not-a-window', 'not-a-window', id='unparseable-kept-raw'),
    ],
)
def test_main_set_display_stores_the_one_window_id_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    given: str,
    stored: str,
) -> None:
    """set-display is a Display.wm_window_id writer, so it writes the PRD D10 form SessionStart writes.

    session_windows.canonical_window_id's padded ``0x%08x``, or the raw value
    when it does not parse, as session_hooks keeps an unparseable $WINDOWID.
    """
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(
        ['set-display', '--record', record_dir, '--kind', 'wm', '--wm-title', 'unblock:df#2085',
         '--wm-window-id', given]
    )

    assert rc == 0
    display = session_records.read_record(Path(record_dir).name, root=tmp_path).display
    assert display == session_records.Display(kind='wm', wm_title='unblock:df#2085', wm_window_id=stored)


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

    monkeypatch.setattr(session_records, 'write_record', _boom)

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


def test_main_logs_a_verb_the_parser_accepts_but_no_runner_handles(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A verb registered on the parser but dispatched nowhere is an ERROR, not a silent rc-0 no-op."""

    def parser_with_an_unrouted_verb() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest='verb', required=True).add_parser('unrouted-verb')
        return parser

    monkeypatch.setattr(sr, '_build_parser', parser_with_an_unrouted_verb)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['unrouted-verb'])

    assert rc == 0
    [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert 'unrouted-verb' in record.getMessage()
    assert 'no runner handles it' in record.getMessage()


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
    assert not session_records.sessions_dir(root=tmp_path).exists()


# ---------------------------------------------------------------------------
# Task 2511 step-7: driving the liveness sweep from `launching`/`reap`
# ---------------------------------------------------------------------------


def test_launching_returns_while_a_reap_holds_the_lock_and_its_detached_reap_drains_after(
    tmp_path: Path,
) -> None:
    """An inline sweep would block on the held lock, and a child that inherited
    stdout or stderr would keep ``run`` from returning: either way it times out.
    """
    session_records.write_record(
        make_record(
            session_slug='orphan-o', status=session_records.Status.RUNNING, launcher_pid=_DEAD_PID, display=None
        ),
        root=tmp_path,
    )
    set_mtime(
        session_records.record_path_for_slug('orphan-o', root=tmp_path),
        datetime.now(UTC),
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )
    _write_aged_exited('old-terminal', tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE_SPAWN_')}
    env.update(_launching_env(tmp_path))
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)

    with session_reaper.reap_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, str(Path(sr.__file__).resolve()), 'launching'],
            env=env,
            capture_output=True,
            timeout=120,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(session_records.record_path_for_slug(slug, root=tmp_path).parent)
        assert session_records.read_record(slug, root=tmp_path).status == session_records.Status.LAUNCHING
        assert session_records.read_record('orphan-o', root=tmp_path).status == session_records.Status.RUNNING
        assert (session_records.sessions_dir(root=tmp_path) / 'old-terminal').is_dir()

    await_reap_report(tmp_path)
    orphan = session_records.read_record('orphan-o', root=tmp_path)
    assert orphan.status == session_records.Status.EXITED
    assert orphan.exit_code == session_reaper.ORPHAN_EXIT_CODE
    assert not (session_records.sessions_dir(root=tmp_path) / 'old-terminal').exists()


def _pids_with_cwd_under(directory: Path) -> list[int]:
    pids: list[int] = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = Path(os.readlink(proc / 'cwd'))
        except OSError:
            continue
        if cwd == directory or directory in cwd.parents:
            pids.append(int(proc.name))
    return pids


def test_launchings_detached_reap_leaves_the_callers_cwd_and_drains_its_relative_fleet_root(
    tmp_path: Path,
) -> None:
    """A reap child whose cwd stayed in the caller's worktree would hold that
    worktree in use for as long as it queued on the held lock.
    """
    worktree = tmp_path / 'worktree'
    fleet = worktree / 'fleet'
    _write_aged_exited('old-terminal', fleet)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE_SPAWN_')}
    env.update(_launching_env(fleet))
    env['CLAUDE_FLEET_ROOT'] = 'fleet'

    with session_reaper.reap_lock(fleet):
        result = subprocess.run(
            [sys.executable, str(Path(sr.__file__).resolve()), 'launching'],
            cwd=worktree,
            env=env,
            capture_output=True,
            timeout=120,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert _pids_with_cwd_under(worktree) == []

    await_reap_report(fleet)
    assert not (session_records.sessions_dir(root=fleet) / 'old-terminal').exists()


def test_main_launching_fail_soft_when_the_detached_reap_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'no-such-python'))

    rc = sr.main(['launching'])

    assert rc == 0
    slug = session_records.build_session_slug('unblock', 'df', '2085', 4242)
    expected_dir = session_records.record_path_for_slug(slug, root=tmp_path).parent
    assert capsys.readouterr().out == f'{expected_dir}\n'
    assert session_records.read_record(slug, root=tmp_path).status == session_records.Status.LAUNCHING


def test_main_reap_marks_then_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    # A fresh stale orphan: non-terminal, dead pid, past the heartbeat TTL --
    # this pass must MARK it exited (its mtime is bumped by the mark, so it
    # is not also independently terminal_ttl-stale in this SAME pass).
    fresh_orphan = make_record(
        session_slug='fresh-orphan', status=session_records.Status.RUNNING, launcher_pid=_DEAD_PID
    )
    session_records.write_record(fresh_orphan, root=tmp_path)
    set_mtime(
        session_records.record_path_for_slug('fresh-orphan', root=tmp_path),
        datetime.now(UTC),
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    # A separately-aged terminal record, well past TERMINAL_TTL -- this pass
    # must DELETE it (reason='terminal_ttl'), unchanged from today.
    old_terminal = make_record(
        session_slug='old-terminal', status=session_records.Status.EXITED, exit_code=0, launcher_pid=os.getpid()
    )
    session_records.write_record(old_terminal, root=tmp_path)
    set_mtime(
        session_records.record_path_for_slug('old-terminal', root=tmp_path),
        datetime.now(UTC),
        session_reaper.TERMINAL_TTL + timedelta(hours=1),
    )

    rc = sr.main(['reap'])

    assert rc == 0
    fresh_dir = session_records.record_path_for_slug('fresh-orphan', root=tmp_path).parent
    assert fresh_dir.is_dir()  # marked, NOT deleted, this pass
    fresh_reloaded = session_records.read_record('fresh-orphan', root=tmp_path)
    assert fresh_reloaded.status == session_records.Status.EXITED
    assert fresh_reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE

    assert not session_records.record_path_for_slug('old-terminal', root=tmp_path).parent.exists()


def _write_aged_exited(slug: str, root: Path) -> None:
    session_records.write_record(
        make_record(session_slug=slug, status=session_records.Status.EXITED, exit_code=0), root=root
    )
    set_mtime(
        session_records.record_path_for_slug(slug, root=root),
        datetime.now(UTC),
        session_reaper.TERMINAL_TTL + timedelta(hours=1),
    )


def test_main_reap_drains_every_eligible_record_in_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))
    aged_slugs = [f'aged-exited-{i:03d}' for i in range(150)]
    for slug in aged_slugs:
        _write_aged_exited(slug, tmp_path)
    session_records.write_record(
        make_record(session_slug='drain-orphan', status=session_records.Status.RUNNING, launcher_pid=_DEAD_PID),
        root=tmp_path,
    )
    set_mtime(
        session_records.record_path_for_slug('drain-orphan', root=tmp_path),
        datetime.now(UTC),
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    rc = sr.main(['reap'])

    assert rc == 0
    for slug in aged_slugs:
        assert not (session_records.sessions_dir(root=tmp_path) / slug).exists()
    orphan = session_records.read_record('drain-orphan', root=tmp_path)
    assert orphan.status == session_records.Status.EXITED
    assert orphan.exit_code == session_reaper.ORPHAN_EXIT_CODE

    report_path = session_reaper.reap_report_path(tmp_path)
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    requested_at = datetime.fromisoformat(report['requested_at'])
    started_at = datetime.fromisoformat(report['started_at'])
    assert requested_at <= started_at <= datetime.fromisoformat(report['finished_at'])
    assert report['remaining_record_dirs'] == 1
    phases = report['phases']
    assert [phase['name'] for phase in phases] == [
        'index_unpointed_t3_records',
        'mark_orphaned_sessions_exited',
        'mark_windowless_wm_sessions_exited',
        'reap_stale_records',
        'reap_stale_session_pointers',
    ]
    for phase in phases:
        assert isinstance(phase['seconds'], (int, float))
        assert phase['seconds'] >= 0
        assert phase['error'] is None
    by_name = {phase['name']: phase for phase in phases}
    assert by_name['mark_orphaned_sessions_exited']['count'] == 1
    assert by_name['reap_stale_records']['count'] == 150


def test_main_reap_runs_later_phases_when_one_phase_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))

    def _boom(*_args: object, **_kwargs: object) -> list[session_records.SessionRecord]:
        raise OSError('mark phase on fire')

    monkeypatch.setattr(session_reaper, 'mark_orphaned_sessions_exited', _boom)
    _write_aged_exited('aged-despite-fault', tmp_path)

    rc = sr.main(['reap'])

    assert rc == 0
    assert not (session_records.sessions_dir(root=tmp_path) / 'aged-despite-fault').exists()
    phases = json.loads(session_reaper.reap_report_path(tmp_path).read_text())['phases']
    [failed] = [phase for phase in phases if phase['name'] == 'mark_orphaned_sessions_exited']
    assert 'OSError' in failed['error']
    assert failed['count'] is None


def _write_report_started(root: Path, started_at: datetime) -> str:
    report = json.dumps({'started_at': started_at.isoformat()})
    session_reaper.reap_report_path(root).write_text(report)
    return report


def test_main_reap_skips_only_when_a_reap_started_after_it_was_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))
    _write_aged_exited('aged', tmp_path)
    later = _write_report_started(tmp_path, datetime.now(UTC) + timedelta(hours=1))

    assert sr.main(['reap']) == 0

    assert (session_records.sessions_dir(root=tmp_path) / 'aged').is_dir()
    assert session_reaper.reap_report_path(tmp_path).read_text() == later

    _write_report_started(tmp_path, datetime.now(UTC) - timedelta(hours=1))

    assert sr.main(['reap']) == 0

    assert not (session_records.sessions_dir(root=tmp_path) / 'aged').exists()


# ---------------------------------------------------------------------------
# Task 2934 step-9/10: driving the WM-window sweep from `launching`/`reap`
# ---------------------------------------------------------------------------


def test_main_reap_marks_windowless_wm_session_before_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    windowless = make_record(
        session_slug='reap-windowless',
        status=session_records.Status.AWAITING_INPUT,
        launcher_pid=os.getpid(),  # ALIVE -- the pid/TTL sweep alone would never catch this
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    session_records.write_record(windowless, root=tmp_path)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    wmctrl = bin_dir / 'wmctrl'
    wmctrl.write_text("#!/bin/sh\nprintf '0x99  0 host  still-open window\\n'\n")
    wmctrl.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')

    rc = sr.main(['reap'])

    assert rc == 0
    # Marked, not deleted, in this SAME pass -- the window sweep ran in the
    # mark phase before reap_stale_records (mark-then-delete order).
    reloaded_dir = session_records.record_path_for_slug('reap-windowless', root=tmp_path).parent
    assert reloaded_dir.is_dir()
    reloaded = session_records.read_record('reap-windowless', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE


# ---------------------------------------------------------------------------
# CLI lease-* verbs + fail-open fail-soft
# ---------------------------------------------------------------------------


def test_main_lease_claim_on_free_name_acquires_and_prints_decision_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
    )

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    # The WHOLE acquired-path contract, not just line 1: a successful claim has
    # no contending holder, so it must not print `holder_liveness=held` (which
    # reads as "someone else has it"). `none` is the third token in the
    # vocabulary, and it is pinned here because nothing else exercises it.
    assert lines[0] == 'decision=acquired'
    assert lines[2] == 'holder_liveness=none'
    # Task 4248 appended a `slug=` line after it (pinned in that task's own
    # section), and task 4237 a `holder_record=` line after that; anchored by
    # INDEX here so this keeps asserting the position of `holder_liveness`,
    # which is what this test is about, rather than silently re-aiming at
    # whatever line happens to be last.
    assert len(lines) == 5
    assert session_leases.lease_path_for_name('watcher-df', root=tmp_path).is_file()


def test_main_lease_claim_when_held_by_live_holder_stands_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
    )
    capsys.readouterr()  # discard the first claim's output

    rc = sr.main(
        [
            'lease-claim',
            '--name',
            'watcher-df',
            '--slug',
            'watcher-df-200',
            '--pid',
            str(os.getpid()),
            '--policy',
            'stand-down',
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == 'decision=stand-down'
    assert re.search(
        r'lease held by \S+ \(pid \d+ alive, heartbeat \d+s ago\) — standing down', lines[1]
    )
    # the on-disk holder must still be the ORIGINAL claimant (no clobber).
    body = session_leases.lease_path_for_name('watcher-df', root=tmp_path).read_text()
    assert 'watcher-df-100' in body
    assert 'watcher-df-200' not in body


def test_main_lease_heartbeat_bumps_mtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())])
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    rc = sr.main(['lease-heartbeat', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    assert lease_path.stat().st_mtime > backdated_mtime


def test_main_lease_release_removes_the_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())])
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert lease_path.is_file()

    rc = sr.main(['lease-release', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    assert not lease_path.exists()


# --- CLI holder_liveness orphan signal (task 3994 defect 4) ---------------
#
# SINGLE-SIGNAL by construction: `holder_liveness=orphaned` means exactly
# "the pid recorded in the lease body is not running". These tests run under
# PRODUCTION-REALISTIC conditions -- no session record is written anywhere,
# because a lease's `session_slug` is a claimant-chosen ownership token
# (`watcher-<project>-<pid>`) and NOT a session-registry record key, so no
# production path can ever put a record there (see
# test_session_leases.py::TestLeaseSlugIsNotASessionRecordKey). The originally-designed
# session-record corroboration was withdrawn for exactly that reason; the
# emitted values are byte-identical either way.


def _contend_via_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *, holder_pid: int
) -> list[str]:
    """Seed a lease held by watcher-df-100/holder_pid, contend it, return stdout lines."""
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(holder_pid)])
    capsys.readouterr()
    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-200', '--pid', str(os.getpid())]
    )
    assert rc == 0
    return capsys.readouterr().out.splitlines()


def test_main_lease_claim_orphan_signal_is_additive_after_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    lines = _contend_via_cli(tmp_path, capsys, holder_pid=os.getpid())

    # `decision=` STAYS line 1 and the message line 2, so a skill parser that
    # reads only the first two lines is provably unaffected by the addition.
    assert lines[0] == 'decision=stand-down'
    assert lines[1].startswith('lease held by watcher-df-100')
    assert lines[2] == 'holder_liveness=held'


def test_main_lease_claim_reports_orphaned_for_a_dead_holder_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    # No session record is written -- that is the production shape, and the
    # dead pid is the whole of the evidence.
    lines = _contend_via_cli(tmp_path, capsys, holder_pid=_DEAD_PID)

    assert lines[0] == 'decision=stand-down'
    # Anchored by INDEX, not [-1]: task 4248 appended a `slug=` line after this
    # one, and [-1] would have silently re-aimed at it instead of failing.
    assert lines[2] == 'holder_liveness=orphaned'


def test_main_lease_claim_reports_held_for_an_unreadable_lease_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-200', '--pid', str(os.getpid())]
    )

    assert rc == 0
    # 'unknown' must never be promoted to an orphan finding: fail toward held.
    # Index, not [-1] -- task 4248's `slug=` line now follows it.
    assert capsys.readouterr().out.splitlines()[2] == 'holder_liveness=held'




def test_a_degraded_pid_lease_still_ages_out_and_is_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: an unresolvable CLAUDE_PID must not make a lease immortal.

    The whole point of returning 0 rather than a durable-but-unrelated pid:
    both staleness rules need a DEAD pid AND an aged heartbeat, so a fallback
    pid that stays alive (the interactive shell) would make a crashed holder's
    lease permanently unreapable — recoverable only by a human `--force`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-degraded'])
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert json.loads(lease_path.read_text())['pid'] == 0

    # Fresh: NOT reaped — the heartbeat half of the guard still protects it.
    set_mtime(lease_path, _NOW, timedelta(seconds=60))
    assert session_leases.reap_stale_leases(root=tmp_path, now=_NOW) == []
    assert lease_path.is_file()

    # Aged past the TTL: reaped, exactly as pre-3994 heartbeat-only behaviour.
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    reaped = session_leases.reap_stale_leases(root=tmp_path, now=_NOW)

    assert [r.reason for r in reaped] == ['stale_pid']
    assert not lease_path.exists()


def test_main_lease_claim_without_pid_writes_the_resolved_session_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    # Deliberately NOT this process's own pid: that is what the old
    # `--pid` default was, so an env-blind implementation would still pass.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    # The correct pid is STRUCTURAL: it no longer depends on every skill doc
    # getting one shell token right (the `--pid $$` defect).
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()).pid == _DEAD_PID


def test_main_lease_claim_explicit_pid_still_overrides_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(_DEAD_PID)]
    )

    assert rc == 0
    # The operator-override path operators already used by hand.
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()).pid == _DEAD_PID


def test_main_lease_claim_without_slug_writes_the_derived_lease_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--slug` is now optional on lease-claim and defaults like `--pid` does.

    Same shape as the `--pid` pair above, on purpose: the two CLI-owned tokens
    should read as ONE pattern rather than two conventions.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    # Deliberately NOT this process's own pid, so an env-blind implementation
    # (e.g. one reaching for os.getpid()) cannot pass by coincidence.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert (
        session_leases.LeaseHolder.from_json(lease_path.read_text()).session_slug
        == f'watcher-df-{_DEAD_PID}'
    )


def test_main_lease_claim_derives_the_body_pid_and_the_slug_pid_from_one_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A no-flag claim's IDENTITY pid and its LIVENESS pid come from ONE resolution.

    The slug embeds a pid and the lease body records one. If each were resolved
    independently they would agree only because both happen to call the same
    function today: any later change making `resolve_session_pid`
    non-deterministic (a cache, a fallback probe, a re-read of a mutated env)
    would silently desynchronise the identity token from the pid
    `holder_liveness` probes — a lease whose slug names a different session than
    its own liveness check. So `main()` resolves once and feeds both, and this
    pins the resulting equality rather than the implementation.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    holder = session_leases.LeaseHolder.from_json(session_leases.lease_path_for_name('watcher-df', root=tmp_path).read_text())
    assert holder.pid == int(holder.session_slug.rsplit('-', 1)[1])
    assert holder.pid == _DEAD_PID


def test_an_explicit_pid_does_not_satisfy_an_underivable_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--pid` is the body's LIVENESS pid, not this session's IDENTITY.

    The two CLI-owned tokens read as one pattern in the skills ("the CLI owns
    both"), so an operator on the degraded path may reach for the documented
    `--pid` and expect it to unblock a slug-less claim. It deliberately does
    not, and this pins that: `lease-heartbeat`/`lease-release` have no `--pid`
    at all, so a slug derived from one would be derivable by the CLAIM alone —
    an identity the session's own later release could not reproduce, which is
    the exact drift task 4248 removes. `--slug` is the escape hatch precisely
    because all three verbs honour it. The refusal must SAY so, or the operator
    just retries with a bigger `--pid`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    with pytest.raises(SystemExit) as excinfo:
        sr.main(['lease-claim', '--name', 'watcher-df', '--pid', str(_DEAD_PID)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert '--pid' in err
    assert '--slug' in err
    assert not session_leases.lease_path_for_name('watcher-df', root=tmp_path).exists()


def test_main_lease_claim_explicit_slug_still_overrides_the_derived_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-OPERATOR'])

    assert rc == 0
    # An explicit --slug remains a deliberate operator override, never shadowed.
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()).session_slug == 'watcher-df-OPERATOR'


# --- lease-claim's `slug=` diagnostic line (task 4248) ---------------------
#
# ADDITIVE and LAST, under 3994's discipline: `decision=` stays line 1 and the
# human message line 2, so a not-yet-reloaded skill parser reading only the
# first two lines is unaffected. What the line is FOR, now that the skills no
# longer pass `--slug`: it makes the CLI-derived identity LEGIBLE, so an agent
# can cross-check it against `lease-show`'s holder_slug when a heartbeat comes
# back `result=refused`. It is a diagnostic, NOT a value to thread into the
# next call -- a shell variable cannot survive to the next Claude Code Bash
# tool call, which is exactly why the derivation had to move into the CLI.


def test_main_lease_claim_prints_the_derived_slug_last_on_the_acquired_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=acquired'
    assert lines[2] == 'holder_liveness=none'
    assert lines[3] == f'slug=watcher-df-{_DEAD_PID}'
    # Task 4237 appended `holder_record=` after it (pinned in
    # test_session_record_linkage.py); `slug=` keeps its index.
    assert len(lines) == 5


def test_main_lease_claim_slug_line_names_this_caller_not_the_holder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a stand-down the `slug=` line is about US; `holder_liveness=` is about THEM.

    Confusing the two would make the line actively misleading in the one
    situation it exists for — reconciling a refusal against `lease-show`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli(slug='watcher-df-OTHER', pid=_DEAD_PID)
    capsys.readouterr()
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))

    rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=stand-down'
    # DETERMINISTIC, not "either value": the holder above was seeded with
    # `_DEAD_PID`, which `test_main_lease_claim_reports_orphaned_for_a_dead_holder_pid`
    # already pins as `orphaned`. Accepting `held` too would buy nothing and
    # would read as if the outcome were nondeterministic.
    assert lines[2] == 'holder_liveness=orphaned'
    assert lines[3] == f'slug=watcher-df-{os.getpid()}'
    assert 'watcher-df-OTHER' not in lines[3]


def test_main_lease_claim_prints_the_slug_line_even_when_it_fails_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fail-open path keeps its 3994 silence about the HOLDER, but not about US.

    `holder_liveness=` stays ABSENT on a substrate fault, deliberately: we know
    nothing about a holder and must not assert one either way. The slug is a
    different fact — it is this caller's own derived identity, which the fault
    did not make unknown — so it is still emitted, and is in fact most useful
    here, where the claim's outcome is least certain.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    def _boom(*_args: object, **_kwargs: object) -> session_leases.LeaseClaim:
        raise OSError('lease substrate on fire')

    monkeypatch.setattr(session_leases, 'claim_lease', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(['lease-claim', '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=proceed'
    assert lines[2] == f'slug=watcher-df-{_DEAD_PID}'
    assert not any(line.startswith('holder_liveness=') for line in lines)


def test_main_lease_claim_echoes_an_explicit_slug_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-OPERATOR'])

    assert rc == 0
    # The line reports the slug ACTUALLY claimed with, not the one that would
    # have been derived — otherwise it would misreport the override path.
    assert capsys.readouterr().out.splitlines()[3] == 'slug=watcher-df-OPERATOR'


# --- CLI ownership on the mutating verbs (task 3994 defect 1) -------------


def _claim_via_cli(slug: str = 'watcher-df-100', pid: int | None = None) -> None:
    sr.main(
        [
            'lease-claim',
            '--name',
            'watcher-df',
            '--slug',
            slug,
            '--pid',
            str(os.getpid() if pid is None else pid),
        ]
    )


@pytest.mark.parametrize('verb', ['lease-claim', 'lease-heartbeat', 'lease-release'])
def test_main_lease_verbs_refuse_loudly_when_the_slug_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """An underivable slug FAILS LOUDLY — it never drifts to a colliding token.

    This is the half of task 4248 that is not "move it into the CLI": having
    moved it, the CLI must refuse rather than synthesize. `{name}-0` would be
    IDENTICAL for two concurrently-degraded sessions, so each would pass the
    other's `_is_lease_owner` check — 3994 defect 1 reopened. Exit 2 is not a
    regression: pre-4248 a slug-less lease verb ALSO exited 2, so the
    underivable case keeps its exit code and only the derivable case changed.

    The stderr assertions matter as much as the code: an operator who is
    refused must also be told the escape hatch, not merely told no.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))
    _claim_via_cli()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()
    capsys.readouterr()

    monkeypatch.delenv('CLAUDE_PID', raising=False)
    with pytest.raises(SystemExit) as excinfo:
        sr.main([verb, '--name', 'watcher-df'])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert 'CLAUDE_PID' in err
    assert '--slug' in err
    # Nothing was mutated on the way out.
    assert lease_path.read_text() == original_body


def test_the_read_only_lease_verbs_stay_usable_when_the_slug_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DIAGNOSIS must survive the degraded path the exit-2 refusal creates.

    `lease-show` and `lease-reap` are exactly what a refused or degraded
    operator is sent to next — `_run_lease_mutation`'s `result=refused` message
    ends "inspect with `lease-show --name <name>`", and both watcher SKILLs say
    the same — so the guard must cover only the three SLUG-BEARING verbs.
    Nothing else pins that:
    the guard is scoped by a hardcoded verb tuple, and widening it (or hoisting
    it above the verb check) would leave every new test green while removing
    the operator's only way to see WHY they were refused.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(os.getpid()))
    _claim_via_cli()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    capsys.readouterr()

    monkeypatch.delenv('CLAUDE_PID', raising=False)

    # Read-only inspection: still prints its normal key=value report.
    assert sr.main(['lease-show', '--name', 'watcher-df']) == 0
    shown = dict(line.split('=', 1) for line in capsys.readouterr().out.splitlines())
    assert shown['name'] == 'watcher-df'
    assert shown['holder_slug'] == 'watcher-df-100'

    # The sweep: also unguarded, and a live holder is still not reaped.
    assert sr.main(['lease-reap']) == 0
    assert lease_path.is_file()


def test_an_explicit_slug_is_still_the_escape_hatch_under_an_unresolvable_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal above is escapable, and does not break the degraded-pid path.

    `test_a_degraded_pid_lease_still_ages_out_and_is_reaped` depends on being
    able to CLAIM a lease with an unresolvable `CLAUDE_PID` (recording pid 0, so
    a crashed holder still ages out). Refusing to DERIVE a slug must not refuse
    to ACCEPT one — the operator supplies a stable token and the pid degrades
    exactly as 3994 designed.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.delenv('CLAUDE_PID', raising=False)

    rc = sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-degraded'])

    assert rc == 0
    holder = session_leases.LeaseHolder.from_json(session_leases.lease_path_for_name('watcher-df', root=tmp_path).read_text())
    assert holder.session_slug == 'watcher-df-degraded'
    assert holder.pid == 0


# `test_main_lease_mutating_verbs_require_slug` lived here until task 4248.
# It asserted that OMITTING `--slug` exits 2 — true only while the token had
# to be spelled in shell. It is superseded, not dropped: its surviving half
# (an UNDERIVABLE slug exits 2 rather than defaulting to a colliding token) is
# now pinned across all THREE lease verbs by
# `test_main_lease_verbs_refuse_loudly_when_the_slug_cannot_be_derived`, and
# the ownership guarantee it was really protecting is pinned directly by
# `..._still_refuse_a_stranger_with_a_derived_slug` below. It was also quietly
# env-dependent: it never unset `CLAUDE_PID`, so post-4248 it would pass or
# fail on ambient environment rather than on behaviour.


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_derive_the_owners_slug_when_it_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """THE POINT of task 4248: the CLI re-derives the same slug across invocations.

    Claim and mutate are separate Claude Code Bash tool calls in production —
    separate `/bin/bash -c` processes — so nothing can be carried between them.
    Both derive from `$CLAUDE_PID`, so the owner matches itself without ever
    spelling the token in shell.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))
    sr.main(['lease-claim', '--name', 'watcher-df'])
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df'])

    assert rc == 0
    assert capsys.readouterr().out.splitlines()[0] == 'result=applied'


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_still_refuse_a_stranger_with_a_derived_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """THE GUARD TEST: defaulting `--slug` is a CALLER-DERIVED default, not a fallback.

    3994's `required=True` comment reads "a silent no-slug fallback would
    preserve the 'any caller may evict/refresh any lease' defect indefinitely".
    This pins that the defect stays fixed. The defect 3994 closed was that both
    verbs mutated UNCONDITIONALLY — no ownership check ran at all. The check
    still runs; it now compares against a slug derived from THIS caller's own
    `$CLAUDE_PID`, so a DIFFERENT session derives a DIFFERENT slug and is
    refused, with the lease body untouched. What 4248 retires is only the
    requirement that the token be spelled in shell — which is what let it drift.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID))
    sr.main(['lease-claim', '--name', 'watcher-df'])
    capsys.readouterr()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()

    # A different session: same lease name, its own (different) session pid.
    monkeypatch.setenv('CLAUDE_PID', str(_DEAD_PID - 1))
    rc = sr.main([verb, '--name', 'watcher-df'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=refused'
    # The human line names the REAL holder — the first session's derived slug.
    assert f'watcher-df-{_DEAD_PID}' in lines[1]
    assert lease_path.read_text() == original_body


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_print_result_applied_for_the_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    # Mirrors lease-claim's `decision=` convention: one parse rule for all
    # four lease verbs.
    assert capsys.readouterr().out.splitlines()[0] == 'result=applied'


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_refuse_a_stranger_without_changing_rc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    original_body = lease_path.read_text()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-STRANGER'])

    # Fail-soft: a refusal is an OUTCOME, not an exit-code change.
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=refused'
    # The human line must name the REAL holder, so an agent reading stdout
    # learns who owns it rather than only that it was told no.
    assert re.search(
        rf'held by watcher-df-100 \(pid {os.getpid()} alive, heartbeat \d+s ago\), not by watcher-df-STRANGER ',
        lines[1],
    )
    assert lease_path.read_text() == original_body


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_force_overrides_a_stranger_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    capsys.readouterr()

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-STRANGER', '--force'])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'result=forced'
    # The holder is read BEFORE the mutation: a forced release deletes the
    # lease body, so a read after it could not name the displaced holder.
    assert re.search(
        rf'FORCED by watcher-df-STRANGER over holder watcher-df-100 \(pid {os.getpid()} alive, heartbeat \d+s ago\)$',
        lines[1],
    )


@pytest.mark.parametrize('verb', ['lease-heartbeat', 'lease-release'])
def test_main_lease_mutating_verbs_report_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main([verb, '--name', 'watcher-df', '--slug', 'watcher-df-100'])

    assert rc == 0
    assert capsys.readouterr().out.splitlines()[0] == 'result=absent'


# --- CLI lease-show (task 3994 defect 3) ----------------------------------


def _parse_kv(out: str) -> dict[str, str]:
    """Parse lease-show's `key=value` lines -- the whole point of the verb."""
    return dict(line.split('=', 1) for line in out.splitlines() if '=' in line)


def _assert_age_between_clock_reads(
    reported_secs: str, *, backdated_by: timedelta, before: datetime, after: datetime
) -> None:
    """lease-show's age is its own clock read, taken somewhere in [before, after], minus an mtime of before - backdated_by.

    The 1 s of upper slack covers filesystem mtime granularity.
    """
    reported = timedelta(seconds=int(reported_secs))
    assert backdated_by <= reported <= backdated_by + (after - before) + timedelta(seconds=1)


def test_main_lease_show_prints_machine_readable_fields_for_a_held_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    before = datetime.now(UTC)
    set_mtime(lease_path, before, timedelta(hours=1))
    capsys.readouterr()

    rc = sr.main(['lease-show', '--name', 'watcher-df'])
    after = datetime.now(UTC)

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    assert fields['name'] == 'watcher-df'
    assert fields['state'] == 'held'
    assert fields['holder_slug'] == 'watcher-df-100'
    assert fields['holder_pid'] == str(os.getpid())
    assert fields['holder_pid_alive'] == 'true'
    assert fields['reclaimable'] == 'false'
    # The freshness clock `cat <lease>` cannot show, replacing `stat -c %y`.
    _assert_age_between_clock_reads(
        fields['heartbeat_age_secs'], backdated_by=timedelta(hours=1), before=before, after=after
    )
    assert datetime.fromisoformat(fields['heartbeat_ts']).tzinfo is not None


def test_main_lease_show_reports_a_dead_and_expired_holder_reclaimable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _claim_via_cli(pid=_DEAD_PID)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, datetime.now(UTC), session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    capsys.readouterr()

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    assert fields['holder_pid_alive'] == 'false'
    assert fields['reclaimable'] == 'true'


def test_main_lease_show_on_an_absent_lease_reports_state_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(['lease-show', '--name', 'watcher-df'])

    assert rc == 0
    assert 'state=absent' in capsys.readouterr().out.splitlines()


def test_main_lease_show_on_a_corrupt_body_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')
    before = datetime.now(UTC)
    set_mtime(lease_path, before, timedelta(hours=1))

    rc = sr.main(['lease-show', '--name', 'watcher-df'])
    after = datetime.now(UTC)

    assert rc == 0
    fields = _parse_kv(capsys.readouterr().out)
    # Marks the body unreadable rather than asserting a fake holder -- and
    # still reports a REAL heartbeat age, the axis that decides reapability.
    assert fields['state'] == 'unreadable'
    _assert_age_between_clock_reads(
        fields['heartbeat_age_secs'], backdated_by=timedelta(hours=1), before=before, after=after
    )
    assert lease_path.is_file()  # read-only: inspection never reaps


def test_main_lease_reap_removes_a_stale_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    sr.main(['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(_DEAD_PID)])
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    rc = sr.main(['lease-reap'])

    assert rc == 0
    assert not lease_path.exists()


def test_main_lease_claim_is_fail_open_never_stand_down_on_a_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> session_leases.LeaseClaim:
        raise OSError('lease substrate on fire')

    monkeypatch.setattr(session_leases, 'claim_lease', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sr.main(
            ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', str(os.getpid())]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == 'decision=proceed'
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Project-token canonicalization (task 3807)
# ---------------------------------------------------------------------------

def test_solar_challenge_alias_was_decided_and_declined() -> None:
    """The naming mismatch is a DECIDED, standing state -- not a pending gap.

    Folding merges solar-challenge's two filed spellings into one bucket --
    an improvement, since a reap scoped to either previously missed the
    other (re-measured 2026-09-07 over 748 records: 3 OPEN under
    ``solar-challenge``, 2 under ``solar_challenge``, ZERO under
    ``my_solar_challenge``; unchanged from 2026-08-07). But that bucket is
    ``solar_challenge``, while
    ``/home/leo/src/solar-challenge/dark-factory-orchestrator.yaml`` declares
    ``my_solar_challenge``, and no alias bridges them. So a reaper passing
    the config-declared token matches ZERO of those 5 rows.

    Task 3813 was the decision task for that bridge, and it DECLINED it --
    chiefly because the fold left no split to heal (an alias would rename a
    populated bucket onto an empty one) and because the identity question is
    an open human gate in that project, marked "Do NOT auto-act". See
    ``dr.PROJECT_TOKEN_ALIASES_DECLINED`` for the full evidence.

    So this is no longer "pending"; the assertions below pin a settled
    outcome. They keep their fail-loudly-on-promotion property -- adding the
    alias breaks the last two -- which
    test_project_token_alias_declines_registry_shape now also enforces from
    the other side, via the disjointness of the two tables. The skills'
    ``--project`` guidance carries the same, now-permanent caveat for the
    humans and watchers that read it, and both CLI verbs warn if you pass
    the config-declared token.
    """
    assert fold_project_token('solar-challenge') == 'solar_challenge'
    assert fold_project_token('solar-challenge') == fold_project_token(
        'solar_challenge'
    )
    assert fold_project_token('my_solar_challenge') != fold_project_token(
        'solar-challenge'
    )
    assert 'solar_challenge' not in PROJECT_ID_ALIASES


def test_project_token_aliases_maps_folded_to_folded() -> None:
    """Every PROJECT_ID_ALIASES key AND value must already be canonical
    under the fold, or a lookup would miss (key) or emit a non-canonical
    token (value) -- the alias map is applied AFTER folding, never before.
    """
    assert PROJECT_ID_ALIASES['df'] == 'dark_factory'
    for alias, canonical in PROJECT_ID_ALIASES.items():
        assert fold_project_token(alias) == canonical
        assert fold_project_token(canonical) == canonical


#: Repo root of this worktree, resolved from THIS FILE so the subprocess
#: contract test below is CWD-independent (the suite is run from
#: ``<worktree>/orchestrator``, but a scratch-CWD run must behave identically).
_SR_REPO_ROOT = Path(__file__).resolve().parents[2]


def _non_venv_interpreter() -> str | None:
    """Return an interpreter that has no workspace packages installed, or None.

    ``sys.executable`` is the venv interpreter, which HAS ``shared`` installed
    and therefore cannot discriminate — the contract under test is precisely
    "imports without a venv".  Prefer the system interpreter, then the base
    interpreter this venv was built from.
    """
    candidates = ['/usr/bin/python3', str(Path(sys.base_prefix) / 'bin' / 'python3')]
    for candidate in candidates:
        if Path(candidate).is_file() and Path(candidate).resolve() != Path(
            sys.executable
        ).resolve():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Bare-shell entrypoint registry + mutation matrix (task 4027)
# ---------------------------------------------------------------------------
#
# Production runs orchestrator modules from plain shell scripts with NO venv
# and no install, under MORE THAN ONE environment.  Each such invocation is
# registered below as a ROW, and `_MUST_BE_REJECTED` mutation-tests those rows
# so the guard is observed FAILING rather than merely observed green.
#
# Register a new bare-shell entrypoint as a ROW; do not write another ad-hoc
# test.  The rows are NOT interchangeable — see
# `test_tier1_is_strictly_stronger_than_tier2` below.

#: The directory that plays the role of ``orchestrator/src`` — i.e. the PARENT
#: of the ``orchestrator`` package.  A mutation run substitutes a sandbox for
#: it (see ``_inject_import_sandbox``), which is why every path below is
#: expressed relative to this rather than hardcoded.
_SR_PKG_ROOT = _SR_REPO_ROOT / 'orchestrator' / 'src'

#: The one repo-relative ``PYTHONPATH`` entry the hook shells export.  Any row
#: entry that RESOLVES to the same directory (``_row_env`` compares resolved
#: paths, not strings) is rewritten to the sandbox root during a mutation run,
#: so the row measures the MUTATED copy rather than the real tree.
_SR_PKG_ROOT_ENTRY = 'orchestrator/src'


class _EntrypointShape(StrEnum):
    """How a bare-shell caller invokes an entrypoint. The shapes differ.

    ``SCRIPT`` puts only the script's OWN directory on ``sys.path``; ``IMPORT``
    resolves off ``PYTHONPATH``.  A module can satisfy one and break the other,
    so a row records which one its callers actually use.
    """

    SCRIPT = 'script'
    """``python /abs/path/to/module.py <argv...>``"""

    IMPORT = 'import'
    """``python -c 'import <dotted.name>'``"""


@dataclasses.dataclass(frozen=True)
class _BareShellEntrypoint:
    """One production invocation of an orchestrator module without a venv.

    Attributes:
        name: human label, used verbatim as the pytest parametrization id.
        path: absolute path of the entrypoint, under ``_SR_PKG_ROOT``.
        pythonpath: repo-relative ``PYTHONPATH`` entries.  ``()`` means the key
            is ABSENT from the env — NOT set to ``''``.  CPython reads an empty
            entry as "the current directory", which would silently put the
            neutral cwd on ``sys.path`` and change what the row measures.
        shape: ``SCRIPT`` or ``IMPORT`` (see ``_EntrypointShape``).
        argv: argv tail for a ``SCRIPT`` row; unused by ``IMPORT`` rows.
            ``('--help',)`` exercises the module import PLUS ``_build_parser()``
            construction — the two things a bad module-scope import breaks —
            and exits 0 without touching ``~/.claude/fleet``.
        probe: an import that MUST FAIL under THIS row's own env.  If it
            succeeds, the row cannot discriminate a compliant module from a
            violating one, and the mutation test skips loudly rather than
            passing vacuously.  Per-row rather than global because the rows
            forbid different sets (tier-1 forbids strictly more than tier-2).
            It is a general health check for the row, NOT a stand-in for the
            payload: ``test_forbidden_import_is_rejected`` additionally requires
            the payload itself to be unreachable under this env, because probe
            and payload can name different modules.
        callers: the shell scripts that make this invocation, so a failure
            names the production path that breaks.
    """

    name: str
    path: Path
    pythonpath: tuple[str, ...]
    shape: _EntrypointShape
    argv: tuple[str, ...]
    probe: str
    callers: str


_BARE_SHELL_ENTRYPOINTS: tuple[_BareShellEntrypoint, ...] = (
    _BareShellEntrypoint(
        name='session_registry.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'downstream Python consumers that import orchestrator.session_registry '
            'with orchestrator/src on PYTHONPATH but no install (T4 verify, T5 '
            'result, T6 hooks, T7 leases). Nothing on that path reaches `shared`; '
            "the module's own bootstrap appends this checkout's shared/src, so only "
            'its stdlib-only shared leaves are importable. This row carries the '
            'assertion formerly made by test_imports_under_a_bare_interpreter_with_'
            'no_workspace_packages (task 3223 step-31).'
        ),
    ),
    _BareShellEntrypoint(
        name='session_registry.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py',
        # THE POINT OF THIS ROW: running a script by absolute path puts only
        # the script's OWN directory on sys.path. Nothing else is reachable
        # from the env — not `shared`, not `orchestrator.<sibling>` — so the
        # only `shared` the module sees is the checkout's shared/src its own
        # bootstrap appends. That is a STRICTLY stronger contract than the
        # tier-2 row above, which the PYTHONPATH entry makes able to reach
        # every intra-orchestrator sibling.
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from orchestrator import stop_instruction',
        callers=(
            'skills/spawn/spawn-claude.sh lines 145 (launching), 428 (exit), 596 '
            'and 693 (refresh), 871 (set-display) — `python3 "$SESSION_REGISTRY_PY"` '
            'with no venv, no install and no PYTHONPATH — and the SKILL.md verb '
            'lines that run it the same way. FOUR OF THE FIVE spawn lines end in '
            '`|| true`, so a break here is SILENT: the subprocess simply never '
            'writes record.json and the session goes missing from the fleet.'
        ),
    ),
    _BareShellEntrypoint(
        name='fleet_files.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'fleet_files.py',
        # An import-only module: running it by path executes its module scope
        # and exits 0, which is exactly what a bad module-scope import breaks.
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py (and decision_registry.py) import this module at '
            'module scope as a bare sibling when skills/spawn/spawn-claude.sh, the '
            'watcher SKILL.md verb lines and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every one of those verbs.'
        ),
    ),
    _BareShellEntrypoint(
        name='decision_registry.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'decision_registry.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'decision_cli.py imports this module at module scope as a bare sibling, '
            'and session_registry.py imports decision_cli the same way, when '
            'skills/spawn/spawn-claude.sh, the watcher SKILL.md verb lines '
            '(write-decision, reap-decisions, close-decision, file-and-close-decision, '
            'reopen-decision, migrate-decision-projects) and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every decision verb.'
        ),
    ),
    _BareShellEntrypoint(
        name='decision_cli.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'decision_cli.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling and hands it the six decision verbs (write-decision, '
            'reap-decisions, close-decision, file-and-close-decision, '
            'reopen-decision, migrate-decision-projects) when '
            'skills/spawn/spawn-claude.sh, the '
            'watcher SKILL.md verb lines and scripts/sitting/payloads.py run '
            'session_registry.py by absolute path with no venv, no install and no '
            'PYTHONPATH. A non-stdlib import here breaks every decision verb.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_identity.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_identity.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling when skills/spawn/spawn-claude.sh and the SKILL.md verb lines '
            'run session_registry.py by absolute path with no venv, no install and '
            'no PYTHONPATH. A non-stdlib import here breaks every registry verb, '
            'the lease verbs that resolve the session identity included.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_records.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_records.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling when skills/spawn/spawn-claude.sh and the SKILL.md verb lines '
            'run session_registry.py by absolute path with no venv, no install and '
            'no PYTHONPATH; tests/scripts/review_all_program_fixture.py::'
            'load_session_records loads it by path under a foreign module name; and '
            'session_hooks, spawn_t3_lane, the cockpit and the scripts import '
            'orchestrator.session_records. A non-stdlib import here breaks every '
            'registry verb, and with it every record write a spawn makes.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_leases.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_leases.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling and runs the lease verbs (lease-claim, lease-heartbeat, '
            'lease-release, lease-show, lease-reap) through it when the watcher '
            'SKILL.md verb lines run session_registry.py by absolute path with no '
            'venv, no install and no PYTHONPATH. A non-stdlib import here breaks '
            'every registry verb, and with it every watcher rotation.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_windows.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_windows.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_reaper.py imports this module at module scope as a bare '
            'sibling when skills/spawn/spawn-claude.sh and the SKILL.md verb lines '
            'run session_registry.py by absolute path with no venv, no install and '
            'no PYTHONPATH, so a non-stdlib import here breaks the reap verb and '
            'every verb that loads the reaper; and session_hooks.py imports '
            'orchestrator.session_windows for the SessionStart display, so it also '
            'breaks every hook.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_reaper.py (tier-1: bare shell, NO PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_reaper.py',
        pythonpath=(),
        shape=_EntrypointShape.SCRIPT,
        argv=(),
        probe='from orchestrator import stop_instruction',
        callers=(
            'session_registry.py imports this module at module scope as a bare '
            'sibling and runs the reap verb, and the detached reap every launching '
            'starts, through it when skills/spawn/spawn-claude.sh and the SKILL.md '
            'verb lines run session_registry.py by absolute path with no venv, no '
            'install and no PYTHONPATH. A non-stdlib import here breaks every '
            'registry verb, and with it every spawn; spawn_t3_lane imports '
            'orchestrator.session_reaper for its thread sweeps too.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_hooks.py (tier-2: bare shell + PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_hooks.py',
        # A THIRD distinct shape: script-by-absolute-path WITH PYTHONPATH. Not
        # reducible to either row above — the tier-1 row has no PYTHONPATH, and
        # the tier-2 IMPORT row does not put the script's own directory on
        # sys.path. This one is exactly what session-start.sh:25 + :29 does.
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from shared import safe_io',
        callers=(
            'skills/spawn/hooks/{session-start,stop,notification,install-hooks}.sh '
            '— each exports PYTHONPATH="$REPO_ROOT/orchestrator/src" (line 25) and '
            'then runs session_hooks.py by absolute path (line 29). These fire on '
            'EVERY session start / stop / notification in every project, including '
            'hand-launched ones.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_ownership_probes.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_ownership_probes.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'session_hooks.py imports this module at module scope, and '
            'skills/spawn/hooks/{session-start,stop,notification}.sh run '
            'session_hooks.py by absolute path with PYTHONPATH=orchestrator/src and '
            'no venv. A non-stdlib import here makes session_hooks.py unimportable, '
            'which silently disables every hook.'
        ),
    ),
    _BareShellEntrypoint(
        name='spawn_t3_lane.py (tier-2: bare shell + PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'spawn_t3_lane.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.SCRIPT,
        argv=('--help',),
        probe='from shared import safe_io',
        callers=(
            'skills/spawn/spawn-claude.sh runs it by absolute path for the t3 lane '
            '(detect, launch, await-started, wait, self-record, pair, status) with '
            'PYTHONPATH="$REPO_ROOT/orchestrator/src" and no venv. A non-stdlib import '
            'here turns every t3-lane spawn into a launcher failure.'
        ),
    ),
    _BareShellEntrypoint(
        name='t3_mcp_client.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 't3_mcp_client.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'spawn_t3_lane.py imports this module at module scope, and '
            'skills/spawn/spawn-claude.sh runs spawn_t3_lane.py by absolute path '
            'with PYTHONPATH="$REPO_ROOT/orchestrator/src" and no venv. A non-stdlib '
            'import here makes the lane unimportable, so every t3-lane spawn becomes '
            'a launcher failure.'
        ),
    ),
    _BareShellEntrypoint(
        name='session_self_exit.py (tier-2: import with PYTHONPATH)',
        path=_SR_PKG_ROOT / 'orchestrator' / 'session_self_exit.py',
        pythonpath=(_SR_PKG_ROOT_ENTRY,),
        shape=_EntrypointShape.IMPORT,
        argv=(),
        probe='from shared import safe_io',
        callers=(
            'session_hooks.py imports this module at module scope, and '
            'skills/spawn/hooks/stop.sh (with its session-start/notification '
            'siblings) runs session_hooks.py by absolute path with '
            'PYTHONPATH=orchestrator/src and no venv. A non-stdlib import here makes '
            'session_hooks.py unimportable, which silently disables every hook.'
        ),
    ),
)

#: Mutation matrix: ``(entrypoint filename, import injected at module scope)``.
#: Each pair MUST be rejected by at least one registry row covering that
#: entrypoint.  This is the load-bearing signal — a guard that has never been
#: observed to fail is not evidence that it guards anything.
_MUST_BE_REJECTED: tuple[tuple[str, str], ...] = (
    # session_registry's own bootstrap puts this checkout's shared/src on
    # sys.path, and its payloads are injected AFTER that (see
    # ``_INJECTION_ANCHORS``), so `shared` itself is reachable here by design.
    # What must still be refused is a shared module with a third-party
    # dependency: config_models fails on its pydantic import.
    ('session_registry.py', 'from shared import config_models'),
    ('session_registry.py', 'from orchestrator import config'),
    ('session_registry.py', 'from orchestrator import stop_instruction'),
    # The dependency-footprint half of the contract: a plain third-party import
    # at module scope. Without it the matrix only ever exercises workspace
    # packages, and nothing observes the rows rejecting `pip install`-ware.
    #
    # `pydantic` rather than `yaml` DELIBERATELY, and it is measured, not
    # assumed: on this host `import yaml` resolves out of the distro's
    # /usr/lib/python3/dist-packages and SUCCEEDS under every row's env (it is
    # not user-site, so PYTHONNOUSERSITE does not remove it), which would make
    # the payload non-discriminating and skip the row instead of pinning it.
    # `pydantic` is absent from the bare interpreter — that absence is the same
    # one that makes `from orchestrator import config` fail below.
    ('session_registry.py', 'import pydantic'),
    # session_hooks.py's contract is the WEAKER tier-2 one: it does
    # `from orchestrator import session_registry` at module scope (line 50) and
    # its callers export PYTHONPATH for exactly that, so `orchestrator.*` is
    # ALLOWED. `shared` and third-party are not: its env puts neither shared/src
    # nor a site-packages on the path, and the payload lands at the
    # `from __future__` anchor, before the session_registry import whose
    # bootstrap would otherwise make `shared` reachable as a side effect.
    ('session_hooks.py', 'from shared import safe_io'),
    ('session_hooks.py', 'import pydantic'),
    ('session_ownership_probes.py', 'from shared import safe_io'),
    ('session_ownership_probes.py', 'import pydantic'),
    ('session_self_exit.py', 'from shared import safe_io'),
    ('session_self_exit.py', 'import pydantic'),
    ('spawn_t3_lane.py', 'from shared import safe_io'),
    ('spawn_t3_lane.py', 'import pydantic'),
    # Lands at the `from __future__` anchor, BEFORE t3_mcp_client's own
    # session_registry import, whose bootstrap is what makes `shared` reachable.
    ('t3_mcp_client.py', 'from shared import safe_io'),
    ('t3_mcp_client.py', 'import pydantic'),
    ('fleet_files.py', 'import pydantic'),
    ('fleet_files.py', 'from orchestrator import stop_instruction'),
    ('decision_registry.py', 'import pydantic'),
    ('decision_registry.py', 'from orchestrator import stop_instruction'),
    ('decision_cli.py', 'import pydantic'),
    ('decision_cli.py', 'from orchestrator import stop_instruction'),
    ('session_identity.py', 'import pydantic'),
    ('session_identity.py', 'from orchestrator import stop_instruction'),
    ('session_records.py', 'import pydantic'),
    ('session_records.py', 'from orchestrator import stop_instruction'),
    ('session_leases.py', 'import pydantic'),
    ('session_leases.py', 'from orchestrator import stop_instruction'),
    ('session_windows.py', 'import pydantic'),
    ('session_windows.py', 'from orchestrator import stop_instruction'),
    ('session_reaper.py', 'import pydantic'),
    ('session_reaper.py', 'from orchestrator import stop_instruction'),
)


def _registry_row(filename: str, shape: _EntrypointShape) -> _BareShellEntrypoint:
    """The unique registry row for ``filename`` invoked as ``shape``.

    Anything needing "the tier-2 env" must read it off the ROW via ``_row_env``
    rather than hand-rolling a dict, so there is exactly ONE definition of each
    row's environment.  Two copies drift: a scrub added to ``_row_env`` later
    (``PYTHONNOUSERSITE`` was one) would reach the row but not the helper that
    claims to characterise it.
    """
    matches = [
        row
        for row in _BARE_SHELL_ENTRYPOINTS
        if row.path.name == filename and row.shape is shape
    ]
    assert len(matches) == 1, (
        f'expected exactly one _BARE_SHELL_ENTRYPOINTS row for {filename} invoked '
        f'as {shape}, found {[row.name for row in matches]}'
    )
    return matches[0]


def _row_env(row: _BareShellEntrypoint, *, pkg_root: Path) -> dict[str, str]:
    """Build ``row``'s scrubbed env, with ``pkg_root`` standing in for src.

    ``pkg_root`` is the real ``orchestrator/src`` for a positive run and a
    mutation sandbox for a rejection run.
    """
    env = {
        'PATH': '/usr/bin:/bin',
        'HOME': os.environ.get('HOME', '/tmp'),
        # Every row, always: a sandbox mirrors the package as SYMLINKS, so
        # bytecode written through one would land in the real source tree.
        'PYTHONDONTWRITEBYTECODE': '1',
        # STRICTER than production, deliberately.  The shells do not set this,
        # but ``~/.local/lib/pythonX.Y/site-packages`` is on ``sys.path`` by
        # default (``site.ENABLE_USER_SITE`` is True on the system interpreter),
        # so one developer's ``pip install --user`` of anything named ``shared``
        # would make these rows accept a module that breaks on every other box —
        # and would silently defeat the vacuity probes below, which measure
        # importability under exactly this env.  Erring toward the stricter
        # environment can only produce a loud false RED, never a false green.
        'PYTHONNOUSERSITE': '1',
    }
    # An empty tuple leaves the key genuinely ABSENT — see the dataclass
    # docstring; PYTHONPATH='' is not the same environment.
    if not row.pythonpath:
        return env

    real_pkg_root = _SR_PKG_ROOT.resolve()
    is_mutation_run = pkg_root.resolve() != real_pkg_root
    entries: list[str] = []
    for entry in row.pythonpath:
        resolved = (_SR_REPO_ROOT / entry).resolve()
        if resolved == real_pkg_root:
            # THE substitution that makes a mutation run measure the sandbox.
            entries.append(str(pkg_root))
            continue
        # Resolve-and-compare rather than string-equality against
        # ``_SR_PKG_ROOT_ENTRY``: a row spelled 'orchestrator/src/',
        # './orchestrator/src', or carrying a second entry that also reaches the
        # package, would otherwise miss the substitution and point a MUTATION run
        # at the real, UNMUTATED tree.  That row would then always exit 0 and
        # quietly stop pinning anything — masked, because
        # ``test_forbidden_import_is_rejected`` only needs ANY row to reject.
        assert not (is_mutation_run and resolved.is_relative_to(real_pkg_root)), (
            f'{row.name}: PYTHONPATH entry {entry!r} resolves inside the real '
            f'{real_pkg_root} but was not substituted for the mutation sandbox '
            f'{pkg_root}, so this run would import the UNMUTATED module and the '
            f'mutation test would measure nothing. Spell it exactly '
            f'{_SR_PKG_ROOT_ENTRY!r} so it is substituted.'
        )
        entries.append(str(resolved))
    env['PYTHONPATH'] = os.pathsep.join(entries)
    return env


def _run_row(
    row: _BareShellEntrypoint, *, pkg_root: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    """Run ``row``'s production invocation shape against ``pkg_root``."""
    interpreter = _non_venv_interpreter()
    assert interpreter is not None, 'callers skip when no non-venv interpreter exists'

    target = pkg_root / row.path.relative_to(_SR_PKG_ROOT)
    if row.shape is _EntrypointShape.SCRIPT:
        argv = [interpreter, str(target), *row.argv]
    else:
        dotted = '.'.join(target.relative_to(pkg_root).with_suffix('').parts)
        argv = [interpreter, '-c', f'import {dotted}']

    # ``cwd=tmp_path`` here and in ``_probe_discriminates`` is LOAD-BEARING,
    # not tidiness (task 3223 step-31's measured finding, carried forward).
    # Python puts the process's own cwd on sys.path, so a run from the REPO
    # ROOT resolves ``<repo>/shared`` — the wrapper directory, NOT the real
    # ``shared/src/shared`` package — as an empty NAMESPACE package.  The probe's
    # ``shared`` import then succeeds, the test skips, and the contract goes
    # unpinned.  Measured: inert from the repo root, live from ``orchestrator/``.
    # The canonical test_command runs ``cd orchestrator && pytest tests/``, so it
    # only bit an ad-hoc ``pytest orchestrator/tests/...`` from the root — which
    # is exactly what someone checking this one file types.  A neutral cwd holds
    # either way.  (conftest.py's autouse ``_df_no_mock_derived_stray_dirs`` is a
    # second reason never to let a subprocess inherit the suite's cwd.)
    return subprocess.run(
        argv,
        env=_row_env(row, pkg_root=pkg_root),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


class _BareShellProbes:
    """Session-scoped memo for the subprocess measurements the rows need.

    Two kinds, each a pure function of (row env, subject) and therefore
    identical for every test that asks:

    - ``rejects(row, statement)`` — does ``statement`` FAIL as a bare import
      under ``row``'s own documented env?  A row can only be counted as
      discriminating when this is True; if the interpreter reaches the package
      anyway, the row would accept a violating module too and its green is
      unearned.
    - ``control_run(row, entrypoint)`` — does ``row`` run CLEAN against a
      sandbox with NOTHING injected?

    Memoised because the mutation matrix asks the same questions repeatedly:
    unmemoised, every (matrix entry x covering row) pair spawns its own
    interpreter, so the two session_registry rows are re-probed once per payload
    with identical inputs.  That count grows multiplicatively as rows and
    payloads are added — exactly what a registry design invites — for results
    that cannot differ.
    """

    def __init__(self, cwd: Path) -> None:
        #: Neutral cwd for every subprocess — LOAD-BEARING, see ``_run_row``.
        self._cwd = cwd
        self._rejects: dict[tuple[str, str], bool] = {}
        self._control_sandboxes: dict[str, Path] = {}
        self._control_runs: dict[tuple[str, str], subprocess.CompletedProcess[str]] = {}

    def rejects(self, row: _BareShellEntrypoint, statement: str) -> bool:
        """True when ``statement`` FAILS as a bare import under ``row``'s env."""
        key = (row.name, statement)
        if key not in self._rejects:
            interpreter = _non_venv_interpreter()
            assert interpreter is not None, (
                'callers skip when no non-venv interpreter exists'
            )
            probe = subprocess.run(
                [interpreter, '-c', statement],
                env=_row_env(row, pkg_root=_SR_PKG_ROOT),
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._rejects[key] = probe.returncode != 0
        return self._rejects[key]

    def control_run(
        self, row: _BareShellEntrypoint, entrypoint: str
    ) -> subprocess.CompletedProcess[str]:
        """Run ``row`` against a sandbox of ``entrypoint`` with NOTHING injected.

        The control for every rejection.  It proves the symlink farm and the
        source rewrite still produce a RUNNABLE module, so a non-zero exit on a
        mutated copy is attributable to the payload.  Without it, a silently
        degraded sandbox — a missing symlink, an anchor line that lands inside a
        docstring, a syntax-breaking rewrite — makes every covering row exit
        non-zero and every mutation test PASS while measuring nothing, which is
        the same silent-pass mode this registry exists to close.
        """
        if entrypoint not in self._control_sandboxes:
            # Empty payload: same code path, same symlink farm, same rewrite —
            # only the injected line differs from a mutation run.
            self._control_sandboxes[entrypoint] = _inject_import_sandbox(
                self._cwd, entrypoint, ''
            )
        key = (row.name, entrypoint)
        if key not in self._control_runs:
            self._control_runs[key] = _run_row(
                row, pkg_root=self._control_sandboxes[entrypoint], tmp_path=self._cwd
            )
        return self._control_runs[key]


@pytest.fixture(scope='session')
def bare_shell_probes(tmp_path_factory: pytest.TempPathFactory) -> _BareShellProbes:
    """Session-scoped so each (row, statement) pair is measured exactly once."""
    return _BareShellProbes(tmp_path_factory.mktemp('bare-shell-probes'))


#: Intra-orchestrator modules documented stdlib-only for LAYERING reasons
#: (cycle avoidance), NOT for bare-shell executability. Each is therefore
#: importable under the tier-2 env and NOT under tier-1 — which is exactly the
#: difference `test_tier1_is_strictly_stronger_than_tier2` demonstrates.
#:
#: `orchestrator.config` is deliberately NOT in this list. Measured: it fails
#: the TIER-2 env too, because orchestrator/config.py imports pydantic, which
#: the bare interpreter lacks — a dependency-footprint failure, not the
#: layering one being demonstrated. Using it as the probe would make the test
#: tautological and mislead the next reader. It stays in `_MUST_BE_REJECTED`
#: (it must fail tier-1, and does), it just cannot serve as the tier-2-accepts
#: probe.
_BARE_IMPORTABLE_SIBLINGS = ('stop_instruction', 'fm_retry', 'routing')


def _first_bare_importable_sibling(probes: _BareShellProbes) -> str | None:
    """Return the first sibling importable under the tier-2 env, or None.

    Probed at runtime rather than hardcoded: if the chosen module later grows a
    pydantic/yaml dependency it stops being bare-importable, and a hardcoded
    reference would turn its caller into a confusing false failure instead of a
    skip. Probing the list degrades to None only when ALL candidates stop
    qualifying.

    The env comes off the tier-2 ROW itself (see ``_registry_row``), never a
    local copy — this probe exists to characterise what that row accepts, so
    measuring a different environment than the row would silently weaken
    ``test_tier1_is_strictly_stronger_than_tier2``'s premise.
    """
    tier2 = _registry_row('session_registry.py', _EntrypointShape.IMPORT)
    for candidate in _BARE_IMPORTABLE_SIBLINGS:
        if not probes.rejects(tier2, f'from orchestrator import {candidate}'):
            return candidate
    return None


#: The line a payload is injected after, per entrypoint; any other entrypoint
#: uses ``from __future__ import annotations``.  session_registry's,
#: session_records', session_leases', session_reaper's, decision_registry's and
#: decision_cli's payloads go after their last shared import, so each is judged
#: with the module's own shared/src bootstrap already applied: that is the
#: environment a later module-scope import in that file really runs in.
_INJECTION_ANCHORS: dict[str, str] = {
    'session_registry.py': 'from shared.session_liveness import ',
    'session_records.py': 'from shared.process_identity import ',
    'session_leases.py': 'from shared.proc_liveness import ',
    'session_reaper.py': 'from shared.session_liveness import ',
    'decision_registry.py': 'from shared.project_id import ',
    'decision_cli.py': 'from shared.project_id import ',
}
_DEFAULT_INJECTION_ANCHOR = 'from __future__ import annotations'


def _inject_after_anchor(source: str, entrypoint_filename: str, payload: str) -> str:
    anchor = _INJECTION_ANCHORS.get(entrypoint_filename, _DEFAULT_INJECTION_ANCHOR)
    lines = source.splitlines(keepends=True)
    anchored = [index for index, line in enumerate(lines) if line.startswith(anchor)]
    assert anchored, (
        f'{entrypoint_filename} has no module-scope line starting `{anchor}` to '
        'anchor the injection on; pick another anchor rather than injecting blindly'
    )
    insert_at = anchored[0] + 1
    return ''.join([*lines[:insert_at], f'{payload}\n', *lines[insert_at:]])


def _inject_import_sandbox(tmp_path: Path, entrypoint_filename: str, payload: str) -> Path:
    """Mirror the checkout layout, with ``payload`` injected into one file.

    Returns the directory that stands in for ``orchestrator/src``.

    The sandbox is laid out like a checkout: the package sits at
    ``<sandbox>/orchestrator/src/orchestrator`` and ``<sandbox>/shared/src``
    links the real ``shared/src``, so a mutated session_registry.py resolves its
    ``__file__``-relative shared bootstrap exactly as the real file does.

    Every entry of the real package is mirrored as a SYMLINK (``__pycache__``
    excepted) and only the mutated entrypoint is a real file.  Faithfulness and
    cost: in production a ``SCRIPT`` row's own directory holds all ~90 sibling
    modules, so a lone-file temp copy would be over-strict and could manufacture
    false rejections; a full ``cp -r`` would copy 11M per mutation.  The symlink
    farm is both faithful and one inode.
    """
    slug = re.sub(r'\W+', '-', f'{entrypoint_filename}-{payload}').strip('-')
    sandbox = tmp_path / f'sandbox-{slug}'
    pkg_root = sandbox / 'orchestrator' / 'src'
    pkg = pkg_root / 'orchestrator'
    pkg.mkdir(parents=True)
    (sandbox / 'shared').mkdir()
    (sandbox / 'shared' / 'src').symlink_to(_SR_REPO_ROOT / 'shared' / 'src')

    real_pkg = _SR_PKG_ROOT / 'orchestrator'
    for entry in real_pkg.iterdir():
        if entry.name == '__pycache__':
            continue
        (pkg / entry.name).symlink_to(entry)

    source = (real_pkg / entrypoint_filename).read_text()
    (pkg / entrypoint_filename).unlink()
    (pkg / entrypoint_filename).write_text(
        _inject_after_anchor(source, entrypoint_filename, payload)
    )
    return pkg_root


class TestStdlibOnlySelfContainment:
    """The bare-shell entrypoints must run with NO venv, install or third-party deps.

    This is an architectural contract, not a style preference.  Two production
    shell paths execute orchestrator modules directly, and they do it under
    DIFFERENT environments — so this class pins a REGISTRY
    (``_BARE_SHELL_ENTRYPOINTS``) of invocations, not one module.  What such a
    module may import is the stdlib plus this checkout's stdlib-only ``shared``
    leaves, and the only route to those is session_registry's own bootstrap,
    which appends the checkout's ``shared/src`` (derived from ``__file__``) to
    ``sys.path``; no caller exports anything for it.

    - **tier-1 — bare shell, NO PYTHONPATH.**  ``skills/spawn/spawn-claude.sh``
      and the SKILL.md verb lines run ``session_registry.py`` by absolute path
      with nothing exported, so only the script's own directory is on
      ``sys.path``: no third-party, no ``orchestrator.<sibling>``, and no
      ``shared`` beyond what the bootstrap appends.
    - **tier-2 — bare shell (or plain import) WITH PYTHONPATH=orchestrator/src.**
      ``skills/spawn/hooks/*.sh`` run ``session_hooks.py``, and Python consumers
      import ``orchestrator.session_registry``.  Intra-orchestrator siblings are
      reachable here; third-party is still not, and ``shared`` only through the
      same bootstrap.

    Tier-1 is STRICTLY STRONGER, so the rows are not interchangeable and neither
    may be deleted as a duplicate of the other — pinned by
    ``test_tier1_is_strictly_stronger_than_tier2``.

    The registry is MUTATION-TESTED (``_MUST_BE_REJECTED``): each forbidden
    import is injected into a sandbox and some covering row must be observed
    REJECTING it, with a ``ModuleNotFoundError`` naming the payload's own
    dependency and against a control run of the SAME sandbox with nothing
    injected — so a degraded sandbox cannot masquerade as a working guard.  A
    guard that has never been observed to fail is not evidence of anything —
    task 3223's consolidation added ``from shared import safe_io`` before any
    bootstrap existed and broke exactly this contract (escalation esc-3223-2)
    while the only symptom in the suite was two downstream
    ``test_session_hooks.py`` failures.  A row counts
    as able to discriminate only when BOTH its own vacuity ``probe`` AND the
    payload under test are unreachable under its documented env; otherwise it
    SKIPS loudly rather than reporting a green it did not earn.

    Adding a bare-shell entrypoint?  Register a ROW plus a matrix entry.  Do not
    write another ad-hoc test — a second one drifts from what the shells run.

    DO NOT WIDEN THIS INTO A REPO-WIDE RULE.  "stdlib-only" names at least three
    different contracts in this repo, and only the first is what these rows pin:

    (a) bare-shell executability — ``session_registry``, ``fleet_files``,
        ``decision_registry``, ``decision_cli``, ``session_identity``,
        ``session_records``, ``session_leases``, ``session_windows``,
        ``session_reaper``, ``session_hooks``, ``session_ownership_probes``,
        ``session_self_exit``;
    (b) layering / cycle avoidance — ``routing``, ``stop_instruction``,
        ``fm_retry``, ``agents/triage``, ``fused_memory/config``/``schema``,
        where importing an intra-package sibling is perfectly fine;
    (c) dependency footprint — ``memory_eval_limits``, ``evals/report``
        ("no scipy/numpy"), which already import pydantic and
        ``shared.memory_eval_metrics`` by design.

    A blanket "no non-stdlib import in any file whose docstring says
    stdlib-only" check false-positives on (c) and misreads (b).
    """

    def test_every_matrix_entrypoint_has_a_registry_row(self):
        """Every entrypoint named in the mutation matrix is actually registered.

        Without this, a matrix entry naming an unregistered entrypoint makes
        ``test_forbidden_import_is_rejected`` iterate an EMPTY row set — it
        would neither fail nor skip while the contract went unpinned, which is
        precisely the silent-pass failure mode this registry exists to close.
        """
        registered = {row.path.name for row in _BARE_SHELL_ENTRYPOINTS}
        named = {entrypoint for entrypoint, _payload in _MUST_BE_REJECTED}

        assert named <= registered, (
            'these entrypoints are mutation-tested but have no '
            f'_BARE_SHELL_ENTRYPOINTS row: {sorted(named - registered)}. Register '
            'the production invocation (env + shape + vacuity probe + callers) as '
            'a row rather than leaving the mutation unpinned.'
        )

    @pytest.mark.parametrize('row', _BARE_SHELL_ENTRYPOINTS, ids=lambda row: row.name)
    def test_entrypoint_runs_clean_under_documented_env(self, row, tmp_path):
        """Every registered bare-shell invocation succeeds against the REAL tree."""
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        result = _run_row(row, pkg_root=_SR_PKG_ROOT, tmp_path=tmp_path)

        assert result.returncode == 0, (
            f'{row.name} must run with no venv/install. Its callers:\n'
            f'  {row.callers}\n'
            'Remove the offending module-scope import (or, if the dependency is '
            'genuinely required, the caller can no longer run it bare).\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )

    @pytest.mark.parametrize('flags', [(), ('-I', '-B')], ids=['default', 'isolated'])
    def test_bare_shell_write_decision_folds_project_through_shared(self, tmp_path, flags):
        """The production spawn-claude/SKILL invocation reaches the shared fold.

        ``write-decision`` run by absolute path with no PYTHONPATH stores its
        project folded by ``shared.project_id.fold_project_token``, so the fold
        the cockpit reads with is reachable from a bare shell. The isolated case
        (``-I``, which implies ``-P``) leaves the script's own directory off
        ``sys.path``, so the sibling imports must not depend on it being there.
        """
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')
        fleet = tmp_path / 'fleet'

        result = subprocess.run(
            [
                interpreter,
                *flags,
                str(_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'),
                'write-decision',
                '--id', 'esc-1-1',
                '--project', 'Dark-Factory',
                '--text', 'Approve the bare-shell filing?',
                '--escalations-dir', str(tmp_path / 'escalations'),
            ],
            env={
                'PATH': '/usr/bin:/bin',
                'HOME': str(tmp_path),
                'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'CLAUDE_FLEET_ROOT': str(fleet),
            },
            # Neutral cwd, LOAD-BEARING for the reasons ``_run_row`` gives.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f'stderr:\n{result.stderr}'
        assert [decision.project for decision in dr.list_decisions(root=fleet)] == [
            fold_project_token('Dark-Factory')
        ], f'stderr:\n{result.stderr}'

    _BARE_SHELL_QUEUE = '/a/data/escalations'

    @pytest.mark.parametrize(
        ('argv', 'seeded_state', 'stdout', 'final_state'),
        [
            pytest.param(
                ['close-decision', '--id', 'esc-1-1', '--state', 'answered', '--evidence', EVIDENCE,
                 '--project', 'df', '--escalations-dir', _BARE_SHELL_QUEUE],
                dr.DecisionState.OPEN, 'esc-1-1', dr.DecisionState.ANSWERED,
                id='close-decision',
            ),
            pytest.param(
                ['reopen-decision', '--id', 'esc-1-1', '--project', 'df',
                 '--escalations-dir', _BARE_SHELL_QUEUE],
                dr.DecisionState.DROPPED, 'esc-1-1', dr.DecisionState.OPEN,
                id='reopen-decision',
            ),
            pytest.param(
                ['file-and-close-decision', '--id', 'esc-1-1', '--project', 'df', '--text', 'approve?',
                 '--escalations-dir', _BARE_SHELL_QUEUE, '--state', 'answered', '--evidence', EVIDENCE],
                None, 'dark_factory-esc-1-1', dr.DecisionState.ANSWERED,
                id='file-and-close-decision',
            ),
            pytest.param(
                ['reap-decisions', '--project', 'df', '--escalations-dir', _BARE_SHELL_QUEUE],
                None, '', None,
                id='reap-decisions',
            ),
            pytest.param(
                ['migrate-decision-projects', '--dry-run'],
                None, '', None,
                id='migrate-decision-projects',
            ),
        ],
    )
    def test_bare_shell_runs_the_other_decision_verbs(
        self,
        tmp_path: Path,
        argv: list[str],
        seeded_state: dr.DecisionState | None,
        stdout: str,
        final_state: dr.DecisionState | None,
    ) -> None:
        """Every SKILL-documented decision verb runs at ``session_registry.py <verb>`` from a bare shell.

        The in-process ``sr.main`` tests import the package, so they cannot see
        a break in the bare sibling imports that only a run by absolute path
        takes. write-decision has its own case above.
        """
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')
        fleet = tmp_path / 'fleet'
        if seeded_state is not None:
            seeded = make_decision(
                id='esc-1-1', escalations_dir=self._BARE_SHELL_QUEUE, state=seeded_state,
                closing_evidence=EVIDENCE if seeded_state != dr.DecisionState.OPEN else '',
            )
            dr.write_decision(seeded, root=fleet)

        result = subprocess.run(
            [interpreter, str(_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'), *argv],
            env={
                'PATH': '/usr/bin:/bin',
                'HOME': str(tmp_path),
                'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'CLAUDE_FLEET_ROOT': str(fleet),
            },
            # Neutral cwd, LOAD-BEARING for the reasons ``_run_row`` gives.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f'stderr:\n{result.stderr}'
        assert result.stdout.strip() == stdout, f'stderr:\n{result.stderr}'
        if final_state is not None:
            assert [decision.state for decision in dr.list_decisions(root=fleet)] == [final_state]

    def test_bare_shell_runs_the_lease_verbs(self, tmp_path: Path) -> None:
        """The watcher skills' lease verbs, and ``reap``, run at ``session_registry.py <verb>`` from a bare shell.

        They reach session_leases and the reaper through the bare sibling
        imports that only a run by absolute path takes, so the in-process
        ``sr.main`` tests cannot see them break. Runs claim, show, lease-reap
        and reap in order against one fleet root, as a rotation would.
        """
        interpreter = _non_venv_interpreter()
        if interpreter is None:
            pytest.skip('no non-venv interpreter available on this host')
        fleet = tmp_path / 'fleet'
        holder_pid = str(os.getpid())

        def run_verb(*argv: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [interpreter, str(_SR_PKG_ROOT / 'orchestrator' / 'session_registry.py'), *argv],
                env={
                    'PATH': '/usr/bin:/bin',
                    'HOME': str(tmp_path),
                    'PYTHONNOUSERSITE': '1',
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'CLAUDE_FLEET_ROOT': str(fleet),
                },
                # Neutral cwd, LOAD-BEARING for the reasons ``_run_row`` gives.
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

        claim = run_verb('lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-100', '--pid', holder_pid)
        show = run_verb('lease-show', '--name', 'watcher-df')
        lease_reap = run_verb('lease-reap')
        reap = run_verb('reap')

        assert claim.returncode == 0, f'stderr:\n{claim.stderr}'
        assert claim.stdout.splitlines()[0] == 'decision=acquired', f'stderr:\n{claim.stderr}'
        assert show.returncode == 0, f'stderr:\n{show.stderr}'
        assert 'state=held' in show.stdout.splitlines(), f'stderr:\n{show.stderr}'
        assert (lease_reap.returncode, lease_reap.stdout) == (0, ''), f'stderr:\n{lease_reap.stderr}'
        assert (reap.returncode, reap.stdout) == (0, ''), f'stderr:\n{reap.stderr}'
        assert 'Traceback' not in lease_reap.stderr + reap.stderr
        # A live holder's lease survives the sweep.
        assert (fleet / 'leases' / 'watcher-df.lease').is_file()

    def test_the_extracted_modules_import_only_downward(self):
        """Each module extracted from session_registry imports only below it, never session_registry.

        fleet_files <- decision_registry <- decision_cli; session_identity and
        session_windows are leaves; fleet_files, session_identity <-
        session_records; fleet_files, session_identity, session_records <-
        session_leases; and fleet_files, session_records, session_windows <-
        session_reaper.

        Under tier-1 session_registry is ``__main__``, so a module below it
        that imported it back would silently load a second copy rather than
        fail. Read off every import statement, function-local ones included,
        in each of the relative, ``orchestrator.``-qualified and bare spellings.
        """
        package = _SR_PKG_ROOT / 'orchestrator'

        def is_sibling(name: str) -> bool:
            return (package / f'{name}.py').is_file() or (package / name / '__init__.py').is_file()

        def sibling_imports(module: str) -> set[str]:
            found: set[str] = set()
            for node in ast.walk(ast.parse((package / f'{module}.py').read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    head = (node.module or '').split('.')
                    if node.level == 1:
                        found.update([head[0]] if node.module else [alias.name for alias in node.names])
                    elif head[0] == 'orchestrator':
                        found.update([head[1]] if len(head) > 1 else [alias.name for alias in node.names])
                    elif is_sibling(head[0]):
                        found.add(head[0])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        head = alias.name.split('.')
                        if head[0] == 'orchestrator' and len(head) > 1:
                            found.add(head[1])
                        elif is_sibling(head[0]):
                            found.add(head[0])
            return found

        expected = {
            'fleet_files': set(),
            'decision_registry': {'fleet_files'},
            'decision_cli': {'decision_registry'},
            'session_identity': set(),
            'session_records': {'fleet_files', 'session_identity'},
            'session_leases': {'fleet_files', 'session_identity', 'session_records'},
            'session_windows': set(),
            'session_reaper': {'fleet_files', 'session_records', 'session_windows'},
        }

        assert {module: sibling_imports(module) for module in expected} == expected

    @pytest.mark.parametrize(
        'extracted', ['session_identity', 'session_leases', 'session_reaper', 'session_records', 'session_windows']
    )
    def test_no_extracted_name_is_re_exported_from_session_registry(self, extracted):
        """A name moved out of session_registry no longer resolves through it.

        session_registry imports each extracted module AS A MODULE, so a
        consumer left pointing at ``session_registry.<moved name>`` fails
        loudly instead of reaching the moved object through a silent shim.
        Compared by identity, so session_registry's own same-named ``logger``
        is not a leak.
        """
        module = importlib.import_module(f'orchestrator.{extracted}')
        tree = ast.parse(Path(module.__file__ or '').read_text(encoding='utf-8'))
        bound: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        absent = object()

        assert bound
        assert sorted(name for name in bound if getattr(sr, name, absent) is getattr(module, name)) == []

    @pytest.mark.parametrize(
        ('filename', 'siblings', 'module_siblings'),
        [
            pytest.param(
                'session_registry.py',
                (
                    'decision_cli', 'decision_registry', 'fleet_files', 'session_identity',
                    'session_leases', 'session_reaper', 'session_records', 'session_windows',
                ),
                (
                    'decision_cli', 'session_identity', 'session_leases', 'session_reaper', 'session_records',
                    'session_windows',
                ),
                id='session_registry',
            ),
            pytest.param(
                'session_records.py',
                ('fleet_files', 'session_identity'),
                ('session_identity',),
                id='session_records',
            ),
        ],
    )
    def test_loads_by_path_under_a_foreign_dotless_module_name(
        self, monkeypatch, filename, siblings, module_siblings
    ):
        """The third load mode: by path, under a module name outside the package.

        tests/scripts/review_all_program_fixture.py::load_session_records loads
        this checkout's session_records.py with ``spec_from_file_location``
        under a dotless name, without putting its directory on ``sys.path``, so
        that a suite takes the record schema from this checkout rather than from
        whichever ``orchestrator`` the venv resolves; session_registry.py must
        load the same way. There ``__package__`` is ``''``, so no relative
        sibling import resolves. The siblings must come from the file's own
        directory too, and that directory must not be left on the importer's
        ``sys.path``.
        """
        name = 'sr_foreign_dotless_load_probe'
        path = _SR_PKG_ROOT / 'orchestrator' / filename
        assert str(path.parent) not in sys.path
        assert not any(sibling in sys.modules for sibling in siblings)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        path_before = list(sys.path)

        try:
            spec.loader.exec_module(module)

            assert module.__package__ == ''
            assert sys.path == path_before
            assert {sibling: Path(sys.modules[sibling].__file__ or '') for sibling in siblings} == {
                sibling: path.parent / f'{sibling}.py' for sibling in siblings
            }
            assert {sibling: getattr(module, sibling) for sibling in module_siblings} == {
                sibling: sys.modules[sibling] for sibling in module_siblings
            }
            assert module.fleet_root is sys.modules['fleet_files'].fleet_root
        finally:
            for sibling in siblings:
                sys.modules.pop(sibling, None)

    def test_tier1_is_strictly_stronger_than_tier2(self, tmp_path, bare_shell_probes):
        """The two session_registry rows encode DIFFERENT contracts, not one twice.

        They read like duplicates ("both just check that session_registry
        imports"), so a tidy-up could delete the tier-1 row as redundant and
        silently restore the exact hole task 4027 closed. They are not: tier-2
        ACCEPTS an intra-orchestrator sibling import that tier-1 REJECTS, and
        tier-1 is what spawn-claude.sh actually runs.
        """
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        sibling = _first_bare_importable_sibling(bare_shell_probes)
        if sibling is None:
            pytest.skip(
                'no intra-orchestrator sibling is currently bare-importable under '
                'the tier-2 env, so the two tiers cannot be told apart by this '
                'probe (they still differ — see the row definitions)'
            )

        rows = {
            row.shape: row
            for row in _BARE_SHELL_ENTRYPOINTS
            if row.path.name == 'session_registry.py'
        }
        assert set(rows) == {_EntrypointShape.SCRIPT, _EntrypointShape.IMPORT}, (
            'session_registry.py must keep BOTH a SCRIPT row (tier-1, no '
            'PYTHONPATH — what spawn-claude.sh runs) and an IMPORT row (tier-2, '
            f'what Python consumers do). Present: {sorted(rows)}'
        )

        sandbox = _inject_import_sandbox(
            tmp_path, 'session_registry.py', f'from orchestrator import {sibling}'
        )
        tier2 = _run_row(rows[_EntrypointShape.IMPORT], pkg_root=sandbox, tmp_path=tmp_path)
        tier1 = _run_row(rows[_EntrypointShape.SCRIPT], pkg_root=sandbox, tmp_path=tmp_path)

        both_encode_different_contracts = (
            'The two session_registry rows are NOT duplicates and neither may be '
            'deleted as a copy of the other: tier-2 (import with PYTHONPATH) '
            'tolerates intra-orchestrator siblings, tier-1 (bare shell, no '
            'PYTHONPATH) does not, and tier-1 is the one spawn-claude.sh runs.\n'
            f'payload: from orchestrator import {sibling}\n'
        )
        assert tier2.returncode == 0, (
            f'{both_encode_different_contracts}'
            f'tier-2 was expected to ACCEPT it but exited {tier2.returncode}. If '
            f'`{sibling}` has grown a non-stdlib dependency it is no longer a valid '
            f'probe — extend _BARE_IMPORTABLE_SIBLINGS.\nstderr:\n{tier2.stderr}'
        )
        assert tier1.returncode != 0, (
            f'{both_encode_different_contracts}'
            'tier-1 was expected to REJECT it but exited 0, which means the tier-1 '
            'row is no longer measuring the no-PYTHONPATH environment.\n'
            f'stdout:\n{tier1.stdout}'
        )

    @pytest.mark.parametrize(('entrypoint', 'payload'), _MUST_BE_REJECTED)
    def test_forbidden_import_is_rejected(
        self, entrypoint, payload, tmp_path, bare_shell_probes
    ):
        """Inject a forbidden import and demand that some covering row REJECTS it.

        This is the signal that the rows above discriminate at all. A guard that
        has never been observed to fail is not evidence of anything.
        """
        if _non_venv_interpreter() is None:
            pytest.skip('no non-venv interpreter available on this host')

        covering = [row for row in _BARE_SHELL_ENTRYPOINTS if row.path.name == entrypoint]
        assert covering, (
            f'no _BARE_SHELL_ENTRYPOINTS row covers {entrypoint}, so the loop below '
            'would iterate zero rows and this test would PASS while the contract '
            'went unpinned — the exact silent-pass failure mode the registry exists '
            'to close. Register a row for it.'
        )

        # Discrimination is judged against THIS payload, not the row's fixed
        # `probe` alone.  The two can be about different modules, and then a row
        # is counted as discriminating on a probe unrelated to what it is being
        # asked to reject.  Concretely: the tier-1 row's probe is `from
        # orchestrator import stop_instruction` (no PYTHONPATH, so it fails on
        # essentially any host), while one payload it must reject is `from shared
        # import safe_io`.  On a host where `shared` IS reachable by the bare
        # interpreter, tier-2 is correctly filtered out on its own probe, tier-1
        # survives the filter on an unrelated one, and the mutated tier-1 run then
        # exits 0 — a hard RED reporting a fact about the HOST rather than about
        # the code.  The honest outcome there is the skip below.  Both conditions
        # are required: the row probe guards the row in general, the payload
        # guards this parametrization in particular.
        discriminating = [
            row
            for row in covering
            if bare_shell_probes.rejects(row, row.probe)
            and bare_shell_probes.rejects(row, payload)
        ]
        if not discriminating:
            pytest.skip(
                f'every registry row covering {entrypoint} can already import '
                f'`{payload}` (or its own vacuity probe) under its documented env '
                '(workspace or third-party packages reachable by the bare '
                'interpreter?), so none of them can discriminate a compliant module '
                'from a violating one'
            )

        # CONTROL BEFORE CONCLUSION: the identical sandbox with NOTHING injected
        # must run clean, or a non-zero exit on the mutated copy says nothing
        # about the payload.
        for row in discriminating:
            control = bare_shell_probes.control_run(row, entrypoint)
            assert control.returncode == 0, (
                f'the UNMUTATED sandbox of {entrypoint} already fails under '
                f'{row.name} (rc={control.returncode}), so any rejection below would '
                'be attributable to the sandbox rather than to the injected import. '
                '_inject_import_sandbox has degraded — a missing symlink, or an '
                'anchor line that no longer sits at module scope.\n'
                f'stderr:\n{control.stderr}'
            )

        sandbox = _inject_import_sandbox(tmp_path, entrypoint, payload)
        outcomes = [
            (row, _run_row(row, pkg_root=sandbox, tmp_path=tmp_path))
            for row in discriminating
        ]

        detail = '\n'.join(
            f'- {row.name}: rc={result.returncode}\n'
            f'  stderr: {result.stderr.strip()[:400]}'
            for row, result in outcomes
        )
        # ``ModuleNotFoundError``, not merely ``rc != 0``: the rejection must be
        # caused by the module the payload names — directly, or transitively (`from
        # orchestrator import config` fails on config.py's own pydantic import under
        # tier-2).  A SyntaxError or a crash also exits non-zero, so accepting any
        # non-zero exit would let a broken injection masquerade as a working guard.
        rejected = [
            row.name
            for row, result in outcomes
            if result.returncode != 0 and 'ModuleNotFoundError' in result.stderr
        ]
        assert rejected, (
            f'{entrypoint} with `{payload}` injected at module scope was not rejected '
            'for the right reason by any registry row covering it: each row either '
            'ACCEPTED it — nothing pins that part of the contract, so add the row '
            'whose environment rejects it — or failed WITHOUT a ModuleNotFoundError, '
            'meaning the injection broke the module instead of exercising the import '
            'guard.\n'
            f'{detail}'
        )


# ---------------------------------------------------------------------------
# Task 7489: the T3 lane's registry primitives
# ---------------------------------------------------------------------------


def test_main_set_display_stamps_a_t3_thread_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_env(monkeypatch, _launching_env(tmp_path))
    sr.main(['launching'])
    record_dir = capsys.readouterr().out.strip()

    rc = sr.main(
        ['set-display', '--record', record_dir, '--kind', 't3', '--t3-thread-id', 'th-9',
         '--wm-title', 'unblock:df#2085']
    )

    assert rc == 0
    display = session_records.read_record(Path(record_dir).name, root=tmp_path).display
    assert display == session_records.Display(kind='t3', wm_title='unblock:df#2085', t3_thread_id='th-9')


_SID = '0b6a3c1e-1111-4222-8333-444455556666'


def _bound_record(tmp_path: Path, slug: str, session_id: str | None) -> session_records.SessionRecord:
    record = make_record(session_slug=slug, claude_session_id=session_id)
    session_records.write_record(record, root=tmp_path)
    return record


class TestClaudeSessionPointers:
    def test_pointer_dir_is_a_sibling_of_sessions(self, tmp_path: Path) -> None:
        assert session_records.claude_session_pointers_dir(tmp_path) == tmp_path / 'sessions-by-claude-id'
        assert session_records.claude_session_pointer_path(_SID, tmp_path).parent.parent == tmp_path

    def test_a_written_pointer_resolves_while_the_record_vouches(self, tmp_path: Path) -> None:
        _bound_record(tmp_path, 'session-df-7489-1', _SID)

        assert session_records.write_claude_session_pointer(_SID, 'session-df-7489-1', root=tmp_path)
        assert session_records.claude_session_pointer_path(_SID, tmp_path).read_text() == 'session-df-7489-1'
        assert session_records.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) == (
            'session-df-7489-1'
        )

    def test_a_record_bound_to_another_session_does_not_vouch(self, tmp_path: Path) -> None:
        _bound_record(tmp_path, 'session-df-7489-1', 'some-other-session')
        session_records.write_claude_session_pointer(_SID, 'session-df-7489-1', root=tmp_path)

        assert session_records.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None

    def test_a_missing_record_or_pointer_resolves_none(self, tmp_path: Path) -> None:
        assert session_records.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None
        session_records.write_claude_session_pointer(_SID, 'gone-record', root=tmp_path)
        assert session_records.resolve_session_slug_for_claude_session_id(_SID, root=tmp_path) is None

    @pytest.mark.parametrize('session_id', ['', '../escape', 'a/b', 'x' * 129])
    def test_an_unsafe_session_id_is_never_written_or_read(
        self, tmp_path: Path, session_id: str
    ) -> None:
        assert not session_records.write_claude_session_pointer(session_id, 'session-df-1', root=tmp_path)
        assert session_records.resolve_session_slug_for_claude_session_id(session_id, root=tmp_path) is None
        assert not session_records.claude_session_pointers_dir(tmp_path).exists()

    def test_a_non_record_key_slug_is_not_written(self, tmp_path: Path) -> None:
        assert not session_records.write_claude_session_pointer(_SID, '..', root=tmp_path)
        assert not session_records.write_claude_session_pointer(_SID, 'a/b', root=tmp_path)


_THREAD = '7d1f0c2a-5b3e-4c8d-9a6f-1e2d3c4b5a69'
_OTHER_THREAD = 'mcp:launch:' + '0f9e8d7c6b5a49388271605f4e3d2c1b' * 2


class TestT3ThreadPointers:
    def test_pointer_dir_is_a_sibling_of_sessions(self, tmp_path: Path) -> None:
        assert session_records.t3_thread_pointers_dir(tmp_path) == tmp_path / 'sessions-by-t3-thread'
        assert session_records.t3_thread_pointer_path(_THREAD, tmp_path).parent.parent == tmp_path

    def test_a_written_pointer_resolves_while_a_live_t3_record_names_the_thread(
        self, tmp_path: Path
    ) -> None:
        write_t3_record(tmp_path, 'session-df-8124-1', _THREAD, cwd=str(tmp_path))

        assert session_records.write_t3_thread_pointer(_THREAD, 'session-df-8124-1', root=tmp_path)
        assert session_records.t3_thread_pointer_path(_THREAD, tmp_path).read_text() == 'session-df-8124-1'
        assert session_records.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == (
            'session-df-8124-1'
        )

    @pytest.mark.parametrize(
        ('thread_id', 'status', 'kind'),
        [
            pytest.param(_THREAD, session_records.Status.RUNNING, 'wm', id='not-t3'),
            pytest.param(_OTHER_THREAD, session_records.Status.RUNNING, 't3', id='another-thread'),
            pytest.param(_THREAD, session_records.Status.EXITED, 't3', id='terminal'),
        ],
    )
    def test_a_record_that_does_not_vouch_resolves_none(
        self, tmp_path: Path, thread_id: str, status: session_records.Status, kind: str
    ) -> None:
        write_t3_record(tmp_path, 'held', thread_id, cwd=str(tmp_path), status=status, kind=kind)
        session_records.write_t3_thread_pointer(_THREAD, 'held', root=tmp_path)

        assert session_records.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None

    def test_a_missing_record_or_pointer_resolves_none(self, tmp_path: Path) -> None:
        assert session_records.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None
        session_records.write_t3_thread_pointer(_THREAD, 'gone-record', root=tmp_path)
        assert session_records.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) is None

    @pytest.mark.parametrize('thread_id', ['', '../escape', 'a/b', 'x' * 129])
    def test_an_unsafe_thread_id_is_never_written_or_read(
        self, tmp_path: Path, thread_id: str
    ) -> None:
        assert not session_records.write_t3_thread_pointer(thread_id, 'session-df-1', root=tmp_path)
        assert session_records.resolve_session_slug_for_t3_thread(thread_id, root=tmp_path) is None
        assert not session_records.t3_thread_pointers_dir(tmp_path).exists()

    def test_a_non_record_key_slug_is_not_written(self, tmp_path: Path) -> None:
        assert not session_records.write_t3_thread_pointer(_THREAD, '..', root=tmp_path)
        assert not session_records.write_t3_thread_pointer(_THREAD, 'a/b', root=tmp_path)

    def test_the_exit_verb_drops_the_pointer_naming_the_exited_record(
        self, tmp_path: Path
    ) -> None:
        write_t3_record(tmp_path, 'held', _THREAD, cwd=str(tmp_path))
        session_records.write_t3_thread_pointer(_THREAD, 'held', root=tmp_path)

        assert sr.main(['exit', '--record', str(session_records.sessions_dir(tmp_path) / 'held'), '--code', '0']) == 0

        assert session_records.read_record('held', root=tmp_path).status == session_records.Status.EXITED
        assert not session_records.t3_thread_pointer_path(_THREAD, tmp_path).exists()

    def test_the_exit_verb_keeps_a_pointer_another_live_record_holds(
        self, tmp_path: Path
    ) -> None:
        write_t3_record(tmp_path, 'holder', _THREAD, cwd=str(tmp_path))
        write_t3_record(tmp_path, 'bystander', _THREAD, cwd=str(tmp_path))
        session_records.write_t3_thread_pointer(_THREAD, 'holder', root=tmp_path)

        sr.main(['exit', '--record', str(session_records.sessions_dir(tmp_path) / 'bystander'), '--code', '0'])

        assert session_records.read_record('bystander', root=tmp_path).status == session_records.Status.EXITED
        assert session_records.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == 'holder'

    def test_a_failed_to_start_refresh_drops_the_pointer_naming_the_record(
        self, tmp_path: Path
    ) -> None:
        write_t3_record(tmp_path, 'held', _OTHER_THREAD, cwd=str(tmp_path), pointer=True)
        record_dir = str(session_records.sessions_dir(tmp_path) / 'held')

        assert sr.main(['refresh', '--record', record_dir, '--status', 'failed-to-start']) == 0

        assert session_records.read_record('held', root=tmp_path).status == session_records.Status.FAILED_TO_START
        assert not session_records.t3_thread_pointer_path(_OTHER_THREAD, tmp_path).exists()
