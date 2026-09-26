"""Live clipboard round-trip for cockpit.clipboard against the real host (task 5448).

Proves the leaf signal task 2517 shipped without (the incident account is
in cockpit/src/cockpit/clipboard.py's module docstring): that the copy path
actually puts text where the operator's OTHER applications can paste it.
The headless suite (cockpit/tests/test_clipboard.py) pins the argv and the
stdin bytes handed to the process boundary, which is the right boundary for
a deterministic test but still stops one step short of a real selection
owner -- this module closes that last step by reading the clipboard back
through a separate reader process.

WARNING: this test necessarily OVERWRITES the operator's real clipboard.
Unlike the disposable windows and tmux sessions the rest of this directory
creates, the X clipboard is a single shared, un-namespaceable resource --
there is nothing to pid-stamp our way around, so the sentinel is stamped
(`cockpit-smoke-<pid>-...`) only to make the round-trip assertion
unambiguous, not to protect anything. Whatever the operator had copied is
gone after a run.

@pytest.mark.smoke, deselected by default in BOTH the root pyproject.toml
and cockpit/pyproject.toml (see test_live_backends.py's docstring for why
both are needed) -- select explicitly with `pytest cockpit/tests/smoke -m
smoke`. This module also inherits the directory's autouse
`_require_live_host` fixture, which is STRICTER than it needs to be: it
demands wmctrl/xdotool/xprop/tmux/tkinter as well as DISPLAY, none of which
a clipboard round-trip touches. That is accepted rather than parameterized,
because relaxing the shared guard would change the skip behaviour of every
other smoke module for one test's benefit. The module adds only its own
helper checks, so it degrades to a skip rather than a failure on a host
that satisfies the harness but has no clipboard helper (or no matching
reader) installed.
"""

from __future__ import annotations

import os
import shutil

import pytest

from cockpit.backends.base import run_command
from cockpit.clipboard import CopyOutcome, available_copy_commands, copy_to_system_clipboard

# How to read back what each helper wrote, keyed by the helper's binary. The
# read MUST address the same selection the write did: on a Wayland host
# running XWayland -- which is exactly what this directory's autouse
# _require_live_host fixture admits -- copy_to_system_clipboard's first
# candidate is wl-copy, and reading that back through `xclip -o` would be
# testing the compositor's X11 clipboard bridge instead of this module,
# reporting a missing or lagging bridge as an opaque string mismatch.
_READ_BACK: dict[str, tuple[str, ...]] = {
    'wl-copy': ('wl-paste', '--no-newline'),
    'xclip': ('xclip', '-o', '-selection', 'clipboard'),
    'xsel': ('xsel', '--clipboard', '--output'),
}


@pytest.mark.smoke
def test_copy_to_system_clipboard_is_readable_by_another_process():
    """The real default path puts the payload where a separate process can read it.

    Deliberately calls copy_to_system_clipboard with NO injected environ/
    which/runner: the point of this module is that the production defaults
    resolve a real helper on a real host and that the selection survives
    the helper process exiting. Which helper won decides how the assertion
    reads it back, so the round-trip stays within one selection.
    """
    if not available_copy_commands():
        pytest.skip('no clipboard helper reachable from this session')

    sentinel = f'cockpit-smoke-{os.getpid()}-clipboard-round-trip'

    attempt = copy_to_system_clipboard(sentinel)

    assert attempt.outcome is CopyOutcome.COPIED
    reader = _READ_BACK.get(attempt.command[0])
    assert reader is not None, f'no read-back reader wired for {attempt.command[0]}'
    if shutil.which(reader[0]) is None:
        pytest.skip(f'{attempt.command[0]} took the payload, but {reader[0]} cannot read it back')

    assert run_command(list(reader)).stdout == sentinel
