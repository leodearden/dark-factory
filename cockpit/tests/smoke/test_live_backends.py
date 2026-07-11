"""Live smoke tests for cockpit.backends against the real X11/tmux host (PRD C-smoke).

Proves the C4 focus/arrange backends and the "signal-don't-move" invariant
end-to-end against real wmctrl/xdotool/xprop/tmux subprocesses -- the
headless suites (test_backends_wm.py, test_backends_tmux.py, test_app.py's
B3) exercise the same contracts against a scripted/fake CommandRunner, never
a real window manager or tmux server. This module is the live re-proof PRD
C-smoke requires before the C10 human gate.

All tests are @pytest.mark.smoke; deselected by default
(cockpit/pyproject.toml addopts = "-m 'not smoke'") so routine/headless runs
never spawn windows or tmux sessions -- select explicitly with `pytest
cockpit/tests/smoke -m smoke`. The autouse `_require_live_host` fixture
(smoke/conftest.py) also skips cleanly on a host missing
DISPLAY/wmctrl/xdotool/xprop/tmux/tkinter.

The live harness (skip guard, disposable-resource fixtures, observation
helpers) lives entirely in smoke/conftest.py and is never imported by name
here: the repo runs pytest with --import-mode=importlib (root pyproject.toml
addopts), so a test module's own directory is not added to sys.path and it
cannot `import` a sibling helper module -- conftest.py is discovered and
loaded by pytest by path regardless of import mode, so fixtures are the only
cross-file channel available. Every harness helper this module uses
(disposable_wm_window, active_window_id, wait_until, ...) is therefore
requested as a fixture, never imported.

cockpit.backends.wm/base already exist (C4 landed), so importing them below
is a plain top-level import -- only the *harness* is new here. Each new test
is RED (its fixture is unresolved -- "fixture not found") until the
conftest.py step that adds the fixture it needs lands, mirroring the
"ImportError-RED until the impl step" convention documented in
test_backends_wm.py / test_backends_tmux.py.
"""

from __future__ import annotations

import pytest

from cockpit.backends.base import DisplayTarget
from cockpit.backends.wm import WmBackend


@pytest.mark.smoke
def test_wm_focus_raises_exactly_the_disposable_window(
    disposable_wm_window, active_window_id, wait_until
):
    """B5 (positive control): with B focused, an explicit focus(A) call must
    raise EXACTLY A -- proving the real `wmctrl -a` path in WmBackend.focus
    activates the target it was given, not merely *some* window. Paired with
    B4 (test_signal_change_sets_urgency_and_moves_no_window_nor_steals_focus,
    below) which proves the refresh/signal path does NOT do this -- together
    they bind the signal-don't-move invariant live.
    """
    window_a = disposable_wm_window('a')
    window_b = disposable_wm_window('b')
    target_a = DisplayTarget(kind='wm', wm_title=window_a.title, wm_window_id=window_a.id)
    target_b = DisplayTarget(kind='wm', wm_title=window_b.title, wm_window_id=window_b.id)
    backend = WmBackend()

    # Establish B as a known non-A baseline -- a single window can be
    # auto-focused on map, which would make "focus raised A" trivially true.
    assert backend.focus(target_b).ok is True
    assert wait_until(lambda: active_window_id() == window_b.id), 'window B never became active'

    result = backend.focus(target_a)

    assert result.ok is True
    assert wait_until(lambda: active_window_id() == window_a.id), (
        'focus() did not raise the disposable window A'
    )


@pytest.mark.smoke
def test_signal_change_sets_urgency_and_moves_no_window_nor_steals_focus(
    disposable_wm_window, active_window_id, wait_until, wm_stack_order, wm_urgency_set
):
    """B4 (negative control), paired with B5 above (positive control): B5
    proves focus() DOES raise the disposable window; this proves the
    refresh/signal path's exact backend call set for a newly-urgent target --
    set_urgency(target, True) then reorder([target]), exactly what
    app._update_attention issues for a target newly needing attention
    (app.py:311-347), never focus()/tile() -- does NOT raise or reorder
    anything. Together B5 and B4 bind the signal-don't-move invariant live:
    this is the fired-and-observed rejection binding -- if the signal path
    erroneously focused/raised A, the active window would flip B->A and this
    test would fail.
    """
    window_a = disposable_wm_window('a')
    window_b = disposable_wm_window('b')
    target_a = DisplayTarget(kind='wm', wm_title=window_a.title, wm_window_id=window_a.id)
    target_b = DisplayTarget(kind='wm', wm_title=window_b.title, wm_window_id=window_b.id)
    backend = WmBackend()

    # Establish B as a known non-A baseline, same as B5 above.
    assert backend.focus(target_b).ok is True
    assert wait_until(lambda: active_window_id() == window_b.id), 'window B never became active'

    stack0 = wm_stack_order()
    active0 = active_window_id()

    # Exactly _update_attention's on-attention call set for target A -- never
    # focus()/tile(), which live exclusively in explicit-action handlers.
    backend.set_urgency(target_a, True)
    backend.reorder([target_a])

    assert wait_until(lambda: wm_urgency_set(window_a.id)), "window A's urgency hint was not set"
    assert wm_stack_order() == stack0, 'the managed-window stack changed'
    assert active_window_id() == active0, 'input focus moved off window B -- A was raised/focused'
