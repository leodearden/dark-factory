"""Live X11 round-trip for cockpit.clipboard against the real host (task 5448).

Proves the leaf signal task 2517 shipped without: that pressing the copy
path actually puts text where the operator's OTHER applications can paste
it. The headless suite (cockpit/tests/test_clipboard.py) pins the argv and
the stdin bytes handed to the process boundary, which is the right
boundary for a deterministic test but still stops one step short of a real
selection owner -- this module closes that last step by reading the
clipboard back through a separate `xclip -o` process.

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
xclip check, so it degrades to a skip rather than a failure on a host that
satisfies the harness but has no xclip installed.
"""

from __future__ import annotations

import os
import shutil

import pytest

from cockpit.backends.base import run_command
from cockpit.clipboard import CopyOutcome, copy_to_system_clipboard


@pytest.mark.smoke
def test_copy_to_system_clipboard_is_readable_by_another_process():
    """The real default path puts the payload where a separate process can read it.

    Deliberately calls copy_to_system_clipboard with NO injected environ/
    which/runner: the point of this module is that the production defaults
    resolve a real helper on a real host and that the selection survives
    the helper process exiting.
    """
    if shutil.which('xclip') is None:
        pytest.skip('xclip not installed')

    sentinel = f'cockpit-smoke-{os.getpid()}-clipboard-round-trip'

    attempt = copy_to_system_clipboard(sentinel)

    assert attempt.outcome is CopyOutcome.COPIED
    assert run_command(['xclip', '-o', '-selection', 'clipboard']).stdout == sentinel
