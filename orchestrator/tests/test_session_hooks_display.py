"""Tests for how SessionStart stamps a record's display (tmux, WINDOWID, or the wmctrl marker-title resolver) and for that resolver itself."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _session_hook_fixtures import assert_hook_env_isolated, isolate_hook_env

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


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
