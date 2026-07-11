"""Live harness for cockpit/tests/smoke -- disposable X11/tmux resources + observers.

Loaded by pytest via conftest.py auto-discovery regardless of
--import-mode=importlib (see test_live_backends.py's module docstring for
why the test module can't import this file by name -- every helper below is
exposed to tests as a fixture, not an importable symbol).

Safety: every resource this harness creates is uniquely pid-stamped
(`cockpit-smoke-<pid>-*` window titles, `cockpit-smoke-<pid>` tmux session)
and every observation/op below targets only those artifacts, so a run on a
real desktop never touches the operator's real windows or tmux sessions.
Teardown is fail-soft (wrapped so it never raises) for every fixture.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest

from cockpit.backends.base import run_command

logger = logging.getLogger(__name__)

_REQUIRED_BINARIES = ('wmctrl', 'xdotool', 'xprop', 'tmux')
_SPAWN_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.1
_CLOSE_TIMEOUT_SECONDS = 3.0

# Launched via `sys.executable -c _TK_SRC <title>`: the only disposable-window
# candidate probed on the target host that (a) shows up in `wmctrl -l` and
# (b) reliably accepts input focus, so `wmctrl -a` / WmBackend.focus can
# actually raise it -- xclock/xmessage/zenity do neither (see plan.json
# design decisions). xterm isn't installed on the target host; python3-tk is.
_TK_SRC = (
    "import tkinter as tk,sys; r=tk.Tk(); r.title(sys.argv[1]); r.geometry('240x120'); r.mainloop()"
)


def live_skip_reason() -> str | None:
    """Why this host can't run the live smoke tests, or None if it can."""
    if not os.environ.get('DISPLAY'):
        return 'no DISPLAY set (headless host)'
    missing = [b for b in _REQUIRED_BINARIES if shutil.which(b) is None]
    if missing:
        return f'missing required binaries: {", ".join(missing)}'
    if importlib.util.find_spec('tkinter') is None:
        return 'tkinter not importable'
    return None


@pytest.fixture(autouse=True)
def _require_live_host():
    """Skip cleanly (not fail) on a host that can't run these tests -- headless CI safety net."""
    reason = live_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _wait_until(pred, timeout=_SPAWN_TIMEOUT_SECONDS, interval=_POLL_INTERVAL_SECONDS) -> bool:
    """Poll pred() until truthy or timeout elapses -- async X/tmux state is never instant."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


@pytest.fixture
def wait_until():
    return _wait_until


def _active_window_id() -> str:
    """The decimal window id xdotool considers focused ('' if none/unavailable)."""
    return run_command(['xdotool', 'getactivewindow']).stdout.strip()


@pytest.fixture
def active_window_id():
    return _active_window_id


def _find_window_id(title: str) -> str | None:
    """The decimal id of the first window matching title, via `xdotool search --name`."""
    stdout = run_command(['xdotool', 'search', '--name', title]).stdout.strip()
    if not stdout:
        return None
    return stdout.splitlines()[0]


@pytest.fixture
def find_window_id():
    return _find_window_id


def _close_window_fail_soft(proc: subprocess.Popen, title: str) -> None:
    """terminate() -> wait(timeout) -> kill() on timeout; wrapped to never raise."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
    except Exception:
        logger.warning('disposable_wm_window teardown: failed to close %r', title, exc_info=True)


@dataclass
class Window:
    title: str
    id: str
    proc: subprocess.Popen


def _spawn_tk_window(suffix: str) -> Window:
    title = f'cockpit-smoke-{os.getpid()}-{suffix}'
    proc = subprocess.Popen([sys.executable, '-c', _TK_SRC, title])

    found_id: str | None = None

    def _check() -> bool:
        nonlocal found_id
        found_id = _find_window_id(title)
        return found_id is not None

    if not _wait_until(_check):
        _close_window_fail_soft(proc, title)
        pytest.fail(f'timed out waiting for window {title!r} to appear (xdotool search --name)')
    assert (
        found_id is not None
    )  # narrowed: the last _check() call above set it before returning True
    return Window(title=title, id=found_id, proc=proc)


@pytest.fixture
def disposable_wm_window():
    """Factory fixture: make(suffix) spawns a uniquely-titled Tk window, returns a Window.

    Every window `make()` creates is torn down fail-soft at fixture teardown,
    regardless of how many were made or whether the test itself failed.
    """
    created: list[Window] = []

    def make(suffix: str) -> Window:
        window = _spawn_tk_window(suffix)
        created.append(window)
        return window

    yield make

    for window in created:
        _close_window_fail_soft(window.proc, window.title)


def _wm_stack_order() -> list[str]:
    """The managed-window stack: each `wmctrl -l` line's window-id column, in order."""
    order = []
    for line in run_command(['wmctrl', '-l']).stdout.splitlines():
        columns = line.split(None, 1)
        if columns:
            order.append(columns[0])
    return order


@pytest.fixture
def wm_stack_order():
    return _wm_stack_order


def _wm_urgency_set(win_id: str) -> bool:
    """Whether `xprop -id win_id WM_HINTS` reports the urgency hint bit as set."""
    stdout = run_command(['xprop', '-id', win_id, 'WM_HINTS']).stdout
    return 'urgency hint bit is set' in stdout.lower()


@pytest.fixture
def wm_urgency_set():
    return _wm_urgency_set


def _tmux_active_window_id(session: str) -> str:
    """The stable @id of session's current/active window."""
    return run_command(
        ['tmux', 'display-message', '-p', '-t', session, '#{window_id}']
    ).stdout.strip()


@pytest.fixture
def tmux_active_window_id():
    return _tmux_active_window_id


@dataclass
class TmuxSession:
    """A disposable tmux session, confined to by the `disposable_tmux_session` fixture."""

    name: str

    def windows(self) -> list[tuple[str, str, str]]:
        """(index, window_id, active-flag) rows for every live window, in `tmux list-windows` order."""
        stdout = run_command(
            [
                'tmux',
                'list-windows',
                '-t',
                self.name,
                '-F',
                '#{window_index} #{window_id} #{window_active}',
            ]
        ).stdout
        rows = []
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) == 3:
                rows.append((parts[0], parts[1], parts[2]))
        return rows

    def select(self, index: int) -> None:
        """Mark window `index` as this session's current/active window."""
        run_command(['tmux', 'select-window', '-t', f'{self.name}:{index}'])


def _kill_tmux_session_fail_soft(name: str) -> None:
    try:
        run_command(['tmux', 'kill-session', '-t', name])
    except Exception:
        logger.warning('disposable_tmux_session teardown: failed to kill %r', name, exc_info=True)


@pytest.fixture
def disposable_tmux_session():
    """A uniquely-named session with >=3 windows, killed fail-soft at teardown.

    tmux assigns each new window's index by searching for the lowest unused
    index starting at the *operator's own* `base-index` tmux.conf setting (0
    by default, but commonly 1) -- this fixture does not force that setting,
    so callers MUST NOT assume the three windows land at indices 0,1,2.
    Discover the real indices via the returned TmuxSession's `.windows()`
    (in `tmux list-windows` order, i.e. ascending by index).

    Every tmux op the smoke tests issue is confined to this session name
    (TmuxBackend.reorder addresses only `<name>:<index>`, parking at
    `<name>:9000+`), so a run never touches a real operator tmux session.
    """
    name = f'cockpit-smoke-{os.getpid()}'
    run_command(['tmux', 'new-session', '-d', '-s', name])
    run_command(['tmux', 'new-window', '-t', name])
    run_command(['tmux', 'new-window', '-t', name])

    yield TmuxSession(name=name)

    _kill_tmux_session_fail_soft(name)
