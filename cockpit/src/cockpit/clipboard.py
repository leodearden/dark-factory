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
from dataclasses import dataclass
from enum import Enum
from typing import Literal

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


class CopyOutcome(Enum):
    """What the local-helper path achieved. The caller's whole decision input."""

    COPIED = 'copied'
    NO_HELPER = 'no_helper'
    HELPER_FAILED = 'helper_failed'


@dataclass(frozen=True)
class CopyAttempt:
    """The outcome of one copy_to_system_clipboard call, plus the helper it names.

    *command* is the argv that succeeded (COPIED), the last one tried
    (HELPER_FAILED), or empty (NO_HELPER, and the guarded-exception case in
    CockpitApp.action_copy). Structured rather than a status string the
    caller would have to parse, so the toast can name the mechanism that
    actually ran.
    """

    outcome: CopyOutcome
    command: tuple[str, ...] = ()


# argv + text -> POSIX return code. Deliberately NOT cockpit.backends.base's
# CommandRunner: that alias carries no stdin channel and returns a
# CommandResult whose stdout/stderr a clipboard helper never produces (they
# are DEVNULL by design, see run_clipboard_command). The
# nonzero-means-unavailable reading is shared; the type is not.
ClipboardRunner = Callable[[Sequence[str], str], int]


def copy_to_system_clipboard(
    text: str,
    *,
    environ: Mapping[str, str] = os.environ,
    which: Callable[[str], str | None] = shutil.which,
    runner: ClipboardRunner = run_clipboard_command,
) -> CopyAttempt:
    """Hand *text* to the first local clipboard helper that takes it.

    Tries available_copy_commands() in order and stops at the first zero
    exit code. Never raises: with no usable helper the result is NO_HELPER
    and nothing is spawned; with every helper refusing it is HELPER_FAILED
    naming the last one tried. Either way the caller still owes the
    operator an OSC 52 fallback and a toast — see copy_feedback.
    """
    commands = available_copy_commands(environ=environ, which=which)
    for command in commands:
        if runner(command, text) == 0:
            return CopyAttempt(CopyOutcome.COPIED, command)
        logger.warning('copy_to_system_clipboard: %s did not take the payload', list(command))
    if not commands:
        return CopyAttempt(CopyOutcome.NO_HELPER)
    return CopyAttempt(CopyOutcome.HELPER_FAILED, commands[-1])


# A generic stand-in when a failure carries no argv (CockpitApp.action_copy's
# guarded-exception path): fail-soft must not mean a toast reading 'None'.
_ANONYMOUS_HELPER = 'clipboard helper'

_FALLBACK_NOTE = 'wrote the OSC 52 fallback instead (some terminals ignore it)'


@dataclass(frozen=True)
class CopyFeedback:
    """What to tell the operator, and whether the OSC 52 fallback still runs.

    *severity* is Textual's own SeverityLevel vocabulary narrowed to the two
    levels this module emits, so pyright checks the value at the App.notify
    call site instead of it being a free-form string.
    """

    message: str
    severity: Literal['information', 'warning']
    write_osc52: bool


def copy_feedback(attempt: CopyAttempt) -> CopyFeedback:
    """Map a CopyAttempt to the operator's toast and the fallback decision.

    Pure — no clock, no IO, no Textual — so the wording and the policy are
    both pinnable without a terminal (see cockpit/tests/test_clipboard.py::
    TestCopyFeedback). A success toast names the mechanism that ran, which
    is what makes the diagnosis Leo had to run by hand ("is anything
    happening at all, and through which path?") readable from the UI.
    """
    helper = ' '.join(attempt.command) if attempt.command else _ANONYMOUS_HELPER
    if attempt.outcome is CopyOutcome.COPIED:
        return CopyFeedback(f'Copied to clipboard ({helper})', 'information', write_osc52=False)
    if attempt.outcome is CopyOutcome.NO_HELPER:
        return CopyFeedback(
            f'No clipboard helper on this host — {_FALLBACK_NOTE}', 'warning', write_osc52=True
        )
    return CopyFeedback(f'{helper} failed — {_FALLBACK_NOTE}', 'warning', write_osc52=True)
