"""Tests for cockpit.backends.tmux — TmuxBackend, the tmux implementation (PRD §6.2 / C4).

TmuxBackend is exercised headlessly via an injected recording fake
CommandRunner (ScriptedRunner below) that scripts CommandResults per argv —
no live tmux. `cockpit.backends.tmux` doesn't exist yet, so every test
imports TmuxBackend inline and is ImportError-RED until step-8 (mirrors
test_backends_wm.py conventions).
"""

from __future__ import annotations

import logging


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


class TestTmuxBackendFocus:
    def test_focus_issues_select_window_and_switch_client(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        result = backend.focus(target)

        assert result.ok is True
        assert runner.calls == [
            ['tmux', 'select-window', '-t', 's:0'],
            ['tmux', 'switch-client', '-t', 's:0'],
        ]

    def test_focus_gone_when_select_window_fails_switch_client_not_issued(self, caplog):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(default=CommandResult(returncode=1))
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='gone:0')

        with caplog.at_level(logging.WARNING):
            result = backend.focus(target)

        assert result.ok is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == [['tmux', 'select-window', '-t', 'gone:0']]

    def test_focus_missing_tmux_target_is_a_noop_and_warns(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux')  # no tmux_target

        with caplog.at_level(logging.WARNING):
            result = backend.focus(target)

        assert result.ok is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == []


class TestTmuxBackendReorder:
    def test_reorder_issues_move_window_in_priority_order(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        targets = [
            DisplayTarget(kind='tmux', tmux_target='s:2'),
            DisplayTarget(kind='tmux', tmux_target='s:0'),
        ]

        backend.reorder(targets)

        assert runner.calls == [
            ['tmux', 'move-window', '-s', 's:2', '-t', 's:0'],
            ['tmux', 'move-window', '-s', 's:0', '-t', 's:1'],
        ]

    def test_reorder_issues_zero_focus_commands(self):
        """Focus-preserving shape: reorder never touches select-window/switch-client."""
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux', tmux_target='s:0')]

        backend.reorder(targets)

        assert not any(argv[1] in ('select-window', 'switch-client') for argv in runner.calls)

    def test_reorder_skips_target_with_missing_tmux_target_and_warns(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux'), DisplayTarget(kind='tmux', tmux_target='s:0')]

        with caplog.at_level(logging.WARNING):
            backend.reorder(targets)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == [['tmux', 'move-window', '-s', 's:0', '-t', 's:1']]


class TestTmuxBackendSetUrgency:
    def test_is_a_noop_and_logs_debug(self, caplog):
        """tmux has no per-window urgency hint under the signal-don't-move model."""
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        with caplog.at_level(logging.DEBUG):
            backend.set_urgency(target, True)

        assert runner.calls == []
        assert any(r.levelno == logging.DEBUG for r in caplog.records)


class TestTmuxBackendTile:
    def test_is_a_noop_and_logs_debug(self, caplog):
        """tmux windows aren't X11-tiled; tile is wm-only."""
        from cockpit.backends.base import DisplayTarget, Zone
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        with caplog.at_level(logging.DEBUG):
            backend.tile([target], Zone(x=0, y=0, width=100, height=100))

        assert runner.calls == []
        assert any(r.levelno == logging.DEBUG for r in caplog.records)


class TestTmuxBackendIsAlive:
    def test_true_when_target_present_in_list_windows_output(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'list-windows', '-a'): CommandResult(returncode=0, stdout='s:0: bash\n')
            }
        )
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        assert backend.is_alive(target) is True

    def test_false_when_target_absent_from_list_windows_output(self):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'list-windows', '-a'): CommandResult(
                    returncode=0, stdout='other:0: bash\n'
                )
            }
        )
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        assert backend.is_alive(target) is False


class TestTmuxBackendGoneTarget:
    """Every op no-ops + warns (never raises) on a gone/unaddressable target — PRD §6.2 invariant."""

    def test_is_alive_missing_tmux_target_is_a_noop_and_warns(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux')  # no tmux_target

        with caplog.at_level(logging.WARNING):
            result = backend.is_alive(target)

        assert result is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == []

    def test_is_alive_command_failure_returns_false_and_warns(self, caplog):
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(default=CommandResult(returncode=1))
        backend = TmuxBackend(run=runner)
        target = DisplayTarget(kind='tmux', tmux_target='s:0')

        with caplog.at_level(logging.WARNING):
            result = backend.is_alive(target)

        assert result is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)
