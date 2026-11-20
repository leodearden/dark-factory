"""Tests for how SessionStart stamps a record's display: tmux, WINDOWID, or the spawn's marker title resolved through the injected window probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _session_hook_fixtures import assert_hook_env_isolated, isolate_hook_env

from orchestrator import session_hooks as sh
from orchestrator import session_records, session_windows


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


def _windows(resolved: str | None = None, *, asked: list[str] | None = None) -> session_windows.WindowProbe:
    """A fake probe resolving every title to *resolved*, recording each title it is asked into *asked*."""

    def _resolve_by_title(title: str) -> str | None:
        if asked is not None:
            asked.append(title)
        return resolved

    return session_windows.WindowProbe(live_window_ids=lambda: None, resolve_by_title=_resolve_by_title)


# ---------------------------------------------------------------------------
# Task 2292 step-5: SessionStart best-effort display stamping (Fleet Cockpit C2)
# ---------------------------------------------------------------------------


def test_run_session_start_tmux_env_stamps_tmux_display(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d1', 'cwd': '/home/leo/src/dark-factory'}
    env = {'TMUX': '/tmp/tmux-1000/default,123,0', 'TMUX_PANE': '%3'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.display is not None
    # Assert on the wire value directly -- a StrEnum-vs-str choice can't drift.
    assert record.display.kind == 'tmux'
    assert record.display.tmux_target == '%3'


def test_run_session_start_windowid_env_stamps_wm_display(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
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
    record = session_records.read_record(slug, root=tmp_path)
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
    record = session_records.read_record(slug, root=tmp_path)
    assert record.display is not None
    assert record.display.kind == 'tmux'


def test_run_session_start_no_tmux_or_windowid_leaves_display_none(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d4', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}

    sh.run_session_start(hook_input, env, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.display is None


def test_run_session_start_display_stamping_applies_on_refresh_path(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-d5', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007'}
    slug = sh.hook_session_slug(hook_input, env)
    existing = session_records.SessionRecord(
        session_slug=slug,
        status=session_records.Status.LAUNCHING,
        title='unblock:df#2085 routing-mechanism',
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    session_records.write_record(existing, root=tmp_path)

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status.RUNNING
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

_MARKER = 'focus:df#2510 alpha'


def test_run_session_start_marker_title_env_stamps_wm_display_via_resolver(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-m1', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': _MARKER}
    asked: list[str] = []

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows('0x03200007', asked=asked))

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert asked == [_MARKER]
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    assert record.display.wm_title == _MARKER


def test_run_session_start_marker_title_resolves_through_wmctrl_to_the_canonical_id(
    tmp_path: Path,
) -> None:
    """The live probe's resolve path stamps wmctrl's padded form, however wmctrl spelled it."""
    hook_input = {'session_id': 'sess-m1b', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': _MARKER}

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode=0, stdout=f'0x3200007 0 host {_MARKER}\n')

    windows = session_windows.wmctrl_window_probe(run=fake_run, sleep=lambda _s: None)

    sh.run_session_start(hook_input, env, root=tmp_path, windows=windows)

    record = session_records.read_record(sh.hook_session_slug(hook_input, env), root=tmp_path)
    assert record.display is not None
    assert record.display.wm_window_id == '0x03200007'


def test_run_session_start_marker_title_resolver_miss_leaves_display_none(tmp_path: Path) -> None:
    # Best-effort: a resolution miss must be no worse than today -- record.display
    # stays None, exactly like the no-tmux/no-windowid case.
    hook_input = {'session_id': 'sess-m2', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': _MARKER}

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows(None))

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert record.display is None


def test_run_session_start_windowid_wins_over_marker_title_resolver_not_called(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-m3', 'cwd': '/home/leo/src/dark-factory'}
    env = {'WINDOWID': '0x3200007', 'CLAUDE_SPAWN_WM_TITLE': _MARKER}
    asked: list[str] = []

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows('0x0badbeef', asked=asked))

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert asked == []
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'


def test_run_session_start_tmux_wins_over_marker_title_resolver_not_called(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-m4', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'TMUX': '/tmp/tmux-1000/default,123,0',
        'TMUX_PANE': '%3',
        'CLAUDE_SPAWN_WM_TITLE': _MARKER,
    }
    asked: list[str] = []

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows('0x0badbeef', asked=asked))

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert asked == []
    assert record.display is not None
    assert record.display.kind == 'tmux'


def test_run_session_start_marker_title_stamping_applies_on_refresh_path(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-m5', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_WM_TITLE': _MARKER}
    slug = sh.hook_session_slug(hook_input, env)
    existing = session_records.SessionRecord(
        session_slug=slug,
        status=session_records.Status.LAUNCHING,
        title='unblock:df#2085 routing-mechanism',
        role='unblock',
        project='df',
        task_id='2085',
        cwd='/home/leo/src/dark-factory',
        prompt='/unblock 2085',
    )
    session_records.write_record(existing, root=tmp_path)

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows('0x03200007'))

    record = session_records.read_record(slug, root=tmp_path)
    assert record.status == session_records.Status.RUNNING
    assert record.display is not None
    assert record.display.kind == 'wm'
    assert record.display.wm_window_id == '0x03200007'
    assert record.display.wm_title == _MARKER
    # Previously-populated fields survive.
    assert record.role == 'unblock'
    assert record.project == 'df'
    assert record.task_id == '2085'
    assert record.prompt == '/unblock 2085'


def test_run_session_start_no_marker_title_resolver_never_invoked(tmp_path: Path) -> None:
    # No TMUX/WINDOWID/CLAUDE_SPAWN_WM_TITLE -> display stays None (matches
    # test_run_session_start_no_tmux_or_windowid_leaves_display_none) AND the
    # resolver is never invoked -- no fallback to a churny derived title.
    hook_input = {'session_id': 'sess-m6', 'cwd': '/home/leo/src/dark-factory'}
    env: dict[str, str] = {}
    asked: list[str] = []

    sh.run_session_start(hook_input, env, root=tmp_path, windows=_windows('0x0badbeef', asked=asked))

    slug = sh.hook_session_slug(hook_input, env)
    record = session_records.read_record(slug, root=tmp_path)
    assert asked == []
    assert record.display is None
