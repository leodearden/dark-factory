"""cockpit.clipboard — put text on THIS host's system clipboard, fail-soft.

One purpose: hand *text* to a local clipboard helper (wl-copy / xclip /
xsel) and report structurally what happened. No Textual import, no event
loop, no widget — the whole surface is exercisable from a plain unit test
(see cockpit/tests/test_clipboard.py). The POLICY built on top of that
report — whether to also write the OSC 52 fallback, and what to tell the
operator — lives in cockpit/src/cockpit/app.py::CockpitApp.action_copy.

Nothing here raises. The cockpit is a view, never a dependency (PRD §2), so
a missing helper, an unusable DISPLAY or a hung helper all degrade to a
return value the caller can act on.

WHY THIS MODULE EXISTS. Task 2517 shipped the 'y' copy affordance as
Textual's App.copy_to_clipboard alone — an OSC 52 escape sequence — which
is a measured total no-op on Konsole 23.08.5: the operator pressed 'y' and
nothing reached the clipboard and nothing said so (task 5448). OSC 52 is
still the right fallback when no local helper can reach a clipboard (the
over-SSH case); it is just not sufficient on its own.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# Mirrors cockpit/src/cockpit/backends/base.py::_COMMAND_TIMEOUT_SECONDS, so
# the two external-command seams read alike.
_COPY_TIMEOUT_SECONDS = 5.0

# POSIX's "command not executable/found" code, reused for every degraded
# outcome: the caller only ever asks "was this zero?".
_UNAVAILABLE = 126

# Structured argv tuples, not strings split at call time (heuristic 12 — no
# ad-hoc parser between here and subprocess). Each entry is (env_var,
# argv): the variable whose presence means that display server is reachable,
# and the helper that talks to it. Order IS the preference order — Wayland's
# own helper before the X11 pair, xclip before xsel — rather than whatever
# order PATH happens to resolve.
_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('WAYLAND_DISPLAY', ('wl-copy',)),
    ('DISPLAY', ('xclip', '-selection', 'clipboard')),
    ('DISPLAY', ('xsel', '--clipboard', '--input')),
)


def available_copy_commands(
    *,
    environ: Mapping[str, str] = os.environ,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[tuple[str, ...], ...]:
    """Clipboard-helper argvs usable on this host, in preference order.

    A candidate survives only if its display variable is set in *environ*
    AND its binary resolves through *which*. Both are injected so callers
    (and tests) can ask the question for an environment other than this
    process's own.

    WHY THE ENVIRONMENT GATES THE LIST rather than just trying each binary:
    with neither WAYLAND_DISPLAY nor DISPLAY set — an ssh session, which is
    exactly the case the OSC 52 fallback exists for — no local helper can
    reach a clipboard at all, so there is nothing to learn from exec'ing
    one. Checking the variable first turns that case into a zero-subprocess
    fall-through instead of a doomed exec on every keypress.

    The gate is advisory, not load-bearing: a set-but-broken DISPLAY still
    yields candidates, and the helper's return code remains the arbiter
    (measured: xclip against an unusable DISPLAY exits 1 in 0.06s, which
    copy_to_system_clipboard degrades through HELPER_FAILED to the
    fallback).
    """
    return tuple(
        argv
        for env_var, argv in _CANDIDATES
        if environ.get(env_var) and which(argv[0]) is not None
    )


def run_clipboard_command(
    argv: Sequence[str], text: str, *, timeout: float = _COPY_TIMEOUT_SECONDS
) -> int:
    """Run *argv*, feeding *text* on its stdin; return its exit code, fail-soft.

    Never raises (PRD §2): a missing or non-executable binary (OSError) and
    a helper that never exits (TimeoutExpired) both log at WARNING and
    return a nonzero code. Mirrors cockpit/src/cockpit/backends/base.py::
    run_command's nonzero-means-unavailable reading — a nonzero code here
    says only "this helper did not take the text", which lets the caller
    try the next candidate or fall back to OSC 52.

    WHY stdout/stderr ARE DEVNULL AND NOT CAPTURED — do not "improve" this
    into capture_output to log the helper's stderr. A clipboard helper owns
    the X/Wayland selection by forking a background child that outlives the
    exec'd process and inherits its pipes, so a captured call blocks until
    the CHILD exits, i.e. until the operator replaces the clipboard.
    Measured in this worktree: 5.01s against a 5s background sleeper with
    capture_output=True, versus 0.19s with DEVNULL. This call happens on
    the cockpit's UI thread on every 'y' press, so capturing would be an
    unbounded interface freeze, not a cosmetic choice. The exit code is the
    only signal the caller needs.
    """
    try:
        completed = subprocess.run(  # noqa: S603 -- argv comes from _CANDIDATES, not shell text
            list(argv),
            input=text.encode('utf-8'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning('run_clipboard_command: %s failed: %s', list(argv), exc)
        return _UNAVAILABLE
    return completed.returncode
