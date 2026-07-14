"""Tests for cockpit.backends.wm — WmBackend, the wm/X11 implementation (PRD §6.2 / C4).

WmBackend is exercised headlessly via an injected recording fake
CommandRunner (ScriptedRunner below) that scripts CommandResults per argv —
no live X11/wmctrl/xdotool. `cockpit.backends.wm` doesn't exist yet, so every
test imports WmBackend inline and is ImportError-RED until step-6 (mirrors
test_backends_base.py / test_backends_fakes.py conventions).
"""

from __future__ import annotations

import logging

import pytest


class ScriptedRunner:
    """A recording fake CommandRunner: logs every argv, returns per-argv scripted results.

    `results` maps an exact argv tuple to the CommandResult it should return;
    anything not scripted falls back to `default` (success, rc=0, unless
    overridden) so a test only has to script the one or two argvs it cares
    about.
    """

    def __init__(self, results=None, default=None):
        from cockpit.backends.base import CommandResult

        self.calls: list[list[str]] = []
        self._results = {tuple(k): v for k, v in (results or {}).items()}
        self._default = default if default is not None else CommandResult(returncode=0)

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        return self._results.get(tuple(argv), self._default)


class TestWmBackendFocus:
    def test_focus_success_via_wmctrl(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        result = backend.focus(target)

        assert result.ok is True
        assert ['wmctrl', '-a', 'session-a'] in runner.calls

    def test_focus_falls_back_to_xdotool_when_wmctrl_fails(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={('wmctrl', '-a', 'session-a'): CommandResult(returncode=1)}
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        result = backend.focus(target)

        assert result.ok is True
        assert ['xdotool', 'search', '--name', 'session-a', 'windowactivate'] in runner.calls

    def test_focus_gone_when_both_commands_fail(self, caplog):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(default=CommandResult(returncode=1))
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='gone-window')

        with caplog.at_level(logging.WARNING):
            result = backend.focus(target)

        assert result.ok is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestWmBackendFocusPrefersWindowId:
    """Task 2510 (Fleet Cockpit C10 fix): focus() prefers the stable
    wm_window_id over the churny wm_title, mirroring set_urgency/is_alive's
    existing prefer-wm_window_id pattern.
    """

    def test_focus_prefers_wm_window_id_no_title_roundtrip(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a', wm_window_id='0x123')

        result = backend.focus(target)

        assert result.ok is True
        assert runner.calls == [['wmctrl', '-i', '-a', '0x123']]

    def test_focus_falls_back_to_title_when_window_id_activate_fails(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={('wmctrl', '-i', '-a', '0x123'): CommandResult(returncode=1)}
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a', wm_window_id='0x123')

        result = backend.focus(target)

        assert result.ok is True
        assert ['wmctrl', '-i', '-a', '0x123'] in runner.calls
        assert ['wmctrl', '-a', 'session-a'] in runner.calls

    def test_focus_falls_back_to_xdotool_when_window_id_and_title_both_fail(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={
                ('wmctrl', '-i', '-a', '0x123'): CommandResult(returncode=1),
                ('wmctrl', '-a', 'session-a'): CommandResult(returncode=1),
            }
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a', wm_window_id='0x123')

        result = backend.focus(target)

        assert result.ok is True
        assert ['xdotool', 'search', '--name', 'session-a', 'windowactivate'] in runner.calls

    def test_focus_only_window_id_no_title_succeeds(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_window_id='0x123')

        result = backend.focus(target)

        assert result.ok is True
        assert runner.calls == [['wmctrl', '-i', '-a', '0x123']]

    def test_focus_only_window_id_activate_fails_no_title_or_xdotool_fallback(self, caplog):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(default=CommandResult(returncode=1))
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_window_id='0x123')

        with caplog.at_level(logging.WARNING):
            result = backend.focus(target)

        assert result.ok is False
        assert runner.calls == [['wmctrl', '-i', '-a', '0x123']]
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_focus_no_title_and_no_window_id_warns_without_invoking_runner(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm')  # no wm_title, no wm_window_id

        with caplog.at_level(logging.WARNING):
            result = backend.focus(target)

        assert result.ok is False
        assert runner.calls == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestWmBackendSetUrgency:
    def test_sets_urgency_on_by_title(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        backend.set_urgency(target, True)

        assert runner.calls == [
            ['xdotool', 'search', '--name', 'session-a', 'set_window', '--urgency', '1']
        ]

    def test_sets_urgency_off_by_title(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        backend.set_urgency(target, False)

        assert runner.calls == [
            ['xdotool', 'search', '--name', 'session-a', 'set_window', '--urgency', '0']
        ]

    def test_prefers_wm_window_id_when_present_no_id_roundtrip(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a', wm_window_id='0x123')

        backend.set_urgency(target, True)

        assert runner.calls == [['xdotool', 'set_window', '--urgency', '1', '0x123']]


class TestWmBackendReorder:
    def test_reorder_is_a_noop_and_logs_debug(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        targets = [DisplayTarget(kind='wm', wm_title='a'), DisplayTarget(kind='wm', wm_title='b')]

        with caplog.at_level(logging.DEBUG):
            backend.reorder(targets)

        assert runner.calls == []
        assert any(r.levelno == logging.DEBUG for r in caplog.records)


class TestWmBackendTile:
    def test_tile_issues_wmctrl_move_resize_from_zone(self):
        from cockpit.backends.base import DisplayTarget, Zone
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')
        zone = Zone(x=10, y=20, width=800, height=600, gravity=0)

        backend.tile([target], zone)

        assert runner.calls == [['wmctrl', '-r', 'session-a', '-e', '0,10,20,800,600']]


class TestWmBackendIsAlive:
    def test_true_when_title_present_in_wmctrl_output(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={
                ('wmctrl', '-l'): CommandResult(returncode=0, stdout='0x01 0 host session-a\n')
            }
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        assert backend.is_alive(target) is True

    def test_false_when_title_absent_from_wmctrl_output(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={
                ('wmctrl', '-l'): CommandResult(returncode=0, stdout='0x01 0 host other-session\n')
            }
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='session-a')

        assert backend.is_alive(target) is False

    def test_false_when_wm_title_is_only_a_substring_of_a_different_title(self):
        """Title 'a' must not match a longer, unrelated title like 'session-a'."""
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={
                ('wmctrl', '-l'): CommandResult(returncode=0, stdout='0x01 0 host session-a\n')
            }
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='a')

        assert backend.is_alive(target) is False

    def test_true_when_matched_by_wm_window_id_even_if_title_differs(self):
        """A stale-title/renamed-window is still found via the window id column."""
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(
            results={
                ('wmctrl', '-l'): CommandResult(returncode=0, stdout='0x01 0 host session-a\n')
            }
        )
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='renamed-title', wm_window_id='0x01')

        assert backend.is_alive(target) is True


class TestWmBackendGoneTarget:
    """Every op no-ops + warns (never raises) on a gone/unaddressable target — PRD §6.2 invariant."""

    @pytest.mark.parametrize('op', ['set_urgency', 'tile', 'is_alive'])
    def test_nonzero_rc_is_a_noop_and_warns(self, op, caplog):
        from cockpit.backends.base import CommandResult, DisplayTarget, Zone
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner(default=CommandResult(returncode=1))
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm', wm_title='gone-window')

        with caplog.at_level(logging.WARNING):
            if op == 'set_urgency':
                backend.set_urgency(target, True)
            elif op == 'tile':
                backend.tile([target], Zone(x=0, y=0, width=100, height=100))
            else:
                backend.is_alive(target)

        assert any(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.parametrize('op', ['set_urgency', 'tile', 'is_alive'])
    def test_missing_wm_title_is_a_noop_and_warns_without_invoking_runner(self, op, caplog):
        from cockpit.backends.base import DisplayTarget, Zone
        from cockpit.backends.wm import WmBackend

        runner = ScriptedRunner()
        backend = WmBackend(run=runner)
        target = DisplayTarget(kind='wm')  # no wm_title, no wm_window_id

        with caplog.at_level(logging.WARNING):
            if op == 'set_urgency':
                backend.set_urgency(target, True)
            elif op == 'tile':
                backend.tile([target], Zone(x=0, y=0, width=100, height=100))
            else:
                backend.is_alive(target)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == []
