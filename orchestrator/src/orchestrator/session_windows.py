"""orchestrator.session_windows — X11 window ids and the ``wmctrl -l`` probes.

One window-id form: wmctrl's zero-padded ``0x%08x``. ``canonical_window_id``
parses with base autodetection (``int(raw, 0)``), so a terminal's decimal
``$WINDOWID`` and wmctrl's hex column name the same window identically.

It is a stdlib-only leaf: it imports no orchestrator sibling and no ``shared``,
so ``session_hooks`` (the SessionStart display) and ``session_reaper`` (the
window sweep) both import it without a cycle. The bare-shell contract is
pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``;
the interface is C6 of
``plans/review-all-s19-package-layering-and-private-seams-prd.md``.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WmctrlRun = Callable[[list[str]], subprocess.CompletedProcess[str]]

WMCTRL_TIMEOUT_SECS = 2
"""Per-probe ``wmctrl`` timeout, seconds."""

RESOLVE_ATTEMPTS = 5
"""Default number of ``wmctrl -l`` probes ``resolve_window_id_by_title`` makes."""

_RESOLVE_RETRY_SLEEP_SECS = 0.2

_MISSING_BINARY_RC = 127
_TRANSIENT_FAILURE_RC = 124


def run_wmctrl(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *argv* (a ``wmctrl`` command line), never raising.

    A missing ``wmctrl`` binary is a permanent failure and yields rc 127. Any
    other ``OSError`` or ``SubprocessError``, notably a timeout from a
    momentarily-hung ``wmctrl``, is transient and yields rc 124, so a retrying
    caller can tell the two apart.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=WMCTRL_TIMEOUT_SECS)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, returncode=_MISSING_BINARY_RC, stdout='')
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(argv, returncode=_TRANSIENT_FAILURE_RC, stdout='')


@dataclass(frozen=True)
class WmWindow:
    """One ``wmctrl -l`` line. *window_id* is canonical; a missing column is ``''``."""

    window_id: str
    desktop: str
    host: str
    title: str


def window_id_int(raw: str | int | None) -> int | None:
    """*raw* as a non-negative window-id int, or None when it is blank, unparseable, negative or not a str or int."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw, 0)
        except ValueError:
            return None
    else:
        return None
    return value if value >= 0 else None


def canonical_window_id(raw: str | int | None) -> str | None:
    """*raw* in wmctrl's padded ``0x%08x`` form, or None when ``window_id_int`` rejects it."""
    value = window_id_int(raw)
    return None if value is None else f'0x{value:08x}'


def parse_wmctrl_list(stdout: str) -> tuple[WmWindow, ...]:
    """The windows in ``wmctrl -l`` output: ``<id> <desktop> <host> <title>`` per line.

    A line whose first column is not a window id is skipped. A shorter line is
    kept with ``''`` for its missing columns: a window with an empty title
    prints only three, and dropping it would let the reaper mark a live
    session's window gone.
    """
    windows: list[WmWindow] = []
    for line in stdout.splitlines():
        columns = line.split(None, 3)
        window_id = canonical_window_id(columns[0]) if columns else None
        if window_id is None:
            continue
        desktop, host, title = (*columns[1:], '', '', '')[:3]
        windows.append(WmWindow(window_id=window_id, desktop=desktop, host=host, title=title))
    return tuple(windows)


def live_window_ids(run: WmctrlRun = run_wmctrl) -> frozenset[str] | None:
    """The canonical ids of every window ``wmctrl -l`` lists.

    None means "cannot prove": wmctrl failed or *run* raised. It is never the
    same as an empty set, and a caller holding None must keep every session.
    """
    try:
        result = run(['wmctrl', '-l'])
    except Exception:
        logger.warning('live_window_ids: wmctrl -l invocation raised', exc_info=True)
        return None
    if result.returncode != 0:
        return None
    return frozenset(window.window_id for window in parse_wmctrl_list(result.stdout))


def resolve_window_id_by_title(
    title: str,
    *,
    run: WmctrlRun = run_wmctrl,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = RESOLVE_ATTEMPTS,
) -> str | None:
    """The canonical id of the window titled exactly *title*, or None.

    Terminals map their window asynchronously, so this probes up to *attempts*
    times, sleeping between attempts and never after the last. The title must
    match whole, never as a substring, so a short marker cannot match a longer
    unrelated title. An empty *title* returns None without probing. A missing
    ``wmctrl`` (rc 127) stops at the first probe; a transient failure (rc 124)
    is retried like a miss. It never raises: a fault in *run* or *sleep* is
    logged and returns None.

    Worst-case latency, since SessionStart runs this synchronously: the common
    "window not mapped yet" miss costs ``(attempts - 1)`` sleeps of
    ``_RESOLVE_RETRY_SLEEP_SECS`` plus *attempts* fast probes (about 0.8 s at
    the defaults). A ``wmctrl`` that times out on every probe adds up to
    ``attempts * WMCTRL_TIMEOUT_SECS`` on top (about 10.8 s in all).
    """
    if not title:
        return None
    try:
        for attempt in range(attempts):
            result = run(['wmctrl', '-l'])
            if result.returncode == _MISSING_BINARY_RC:
                return None
            if result.returncode == 0:
                for window in parse_wmctrl_list(result.stdout):
                    if window.title == title:
                        return window.window_id
            if attempt < attempts - 1:
                sleep(_RESOLVE_RETRY_SLEEP_SECS)
    except Exception:
        logger.warning('resolve_window_id_by_title: wmctrl window-id resolution failed', exc_info=True)
    return None


@dataclass(frozen=True)
class WindowProbe:
    """Where a consumer reads windows from. Neither callable raises.

    *live_window_ids* returning None means "cannot prove", and callers keep.
    """

    live_window_ids: Callable[[], frozenset[str] | None]
    resolve_by_title: Callable[[str], str | None]


def wmctrl_window_probe(
    *, run: WmctrlRun = run_wmctrl, sleep: Callable[[float], None] = time.sleep
) -> WindowProbe:
    """The live probe, reading ``wmctrl -l`` through *run*: every consumer's default."""
    return WindowProbe(
        live_window_ids=lambda: live_window_ids(run=run),
        resolve_by_title=lambda title: resolve_window_id_by_title(title, run=run, sleep=sleep),
    )
