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


class FakeTmuxWindowTable:
    """Simulates just enough of tmux's move-window semantics to catch collisions.

    Tracks a live `{"session:index": window_id}` table. A move whose
    destination index is already occupied by a *different* window fails
    exactly like real tmux does ("index in use"), so a reorder implementation
    that doesn't stage through a scratch range first gets caught colliding
    instead of silently producing plausible-looking-but-wrong argv.
    """

    def __init__(self, windows):
        self.windows = dict(windows)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        from cockpit.backends.base import CommandResult

        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] != ['tmux', 'move-window']:
            return CommandResult(returncode=0)

        src = argv[argv.index('-s') + 1]
        dst = argv[argv.index('-t') + 1]
        if src not in self.windows:
            return CommandResult(returncode=1, stderr=f'no such window: {src}')
        if dst in self.windows:
            return CommandResult(returncode=1, stderr="can't move window: index in use")

        self.windows[dst] = self.windows.pop(src)
        return CommandResult(returncode=0)


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
    def test_reorder_parks_then_places_in_priority_order(self):
        """Two-phase argv shape: every target is parked at a scratch index first,
        then placed at its final compacted index — never moved straight to a
        destination another target in the same reorder might still occupy.
        """
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
            ['tmux', 'display-message', '-p', '-t', 's', '#{window_id}'],
            ['tmux', 'move-window', '-d', '-s', 's:2', '-t', 's:9000'],
            ['tmux', 'move-window', '-d', '-s', 's:0', '-t', 's:9001'],
            ['tmux', 'move-window', '-d', '-s', 's:9000', '-t', 's:0'],
            ['tmux', 'move-window', '-d', '-s', 's:9001', '-t', 's:1'],
        ]

    def test_reorder_succeeds_against_a_live_occupied_destination_index(self):
        """Outcome-level proof: with destinations occupied by other targets awaiting
        their own move (the common reorder case), a naive direct move-window would
        fail with tmux's real 'index in use' error. The backend must still land the
        final ordering, not just emit plausible-looking argv.
        """
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        table = FakeTmuxWindowTable({'s:2': 'w2', 's:0': 'w0'})
        backend = TmuxBackend(run=table)
        targets = [
            DisplayTarget(kind='tmux', tmux_target='s:2'),
            DisplayTarget(kind='tmux', tmux_target='s:0'),
        ]

        backend.reorder(targets)

        assert table.windows == {'s:0': 'w2', 's:1': 'w0'}

    def test_reorder_restores_active_window_and_never_switch_client(self):
        """Signal-don't-move shape: move-window inevitably churns a session's
        current window, so reorder snapshots the active window by its stable
        @id up front and re-selects exactly that window at the end (restoring
        operator focus to where it already was). The only focus command it may
        issue is that restoring select-window -- never switch-client, and never
        a select-window for any window other than the one that was active.
        """
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'display-message', '-p', '-t', 's', '#{window_id}'): CommandResult(
                    returncode=0, stdout='@7\n'
                )
            }
        )
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux', tmux_target='s:0')]

        backend.reorder(targets)

        assert not any(argv[1] == 'switch-client' for argv in runner.calls)
        select_calls = [argv for argv in runner.calls if argv[1] == 'select-window']
        assert select_calls == [['tmux', 'select-window', '-t', '@7']]
        # ...and the restore is the LAST thing reorder does, after every move.
        assert runner.calls[-1] == ['tmux', 'select-window', '-t', '@7']

    def test_reorder_restores_active_window_per_session_when_targets_span_sessions(self):
        """Multi-session snapshot/restore: reorder() snapshots and restores
        each involved session's active window INDEPENDENTLY -- exactly one
        select-window per session, each using that session's own @id -- not
        just a single session's worth of restore logic (every other
        reorder() test here drives one session, 's').
        """
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'display-message', '-p', '-t', 's1', '#{window_id}'): CommandResult(
                    returncode=0, stdout='@11\n'
                ),
                ('tmux', 'display-message', '-p', '-t', 's2', '#{window_id}'): CommandResult(
                    returncode=0, stdout='@22\n'
                ),
            }
        )
        backend = TmuxBackend(run=runner)
        targets = [
            DisplayTarget(kind='tmux', tmux_target='s1:0'),
            DisplayTarget(kind='tmux', tmux_target='s2:0'),
        ]

        backend.reorder(targets)

        assert not any(argv[1] == 'switch-client' for argv in runner.calls)
        select_calls = [argv for argv in runner.calls if argv[1] == 'select-window']
        assert select_calls == [
            ['tmux', 'select-window', '-t', '@11'],
            ['tmux', 'select-window', '-t', '@22'],
        ]
        # ...and both restores happen last, after every move for both sessions.
        assert runner.calls[-2:] == select_calls

    def test_reorder_skips_restore_for_session_when_display_message_fails(self, caplog):
        """If a session's pre-reorder active-window snapshot can't be read
        (display-message returns a nonzero rc, e.g. the session vanished),
        reorder must not guess -- it skips restoring that session's focus
        entirely (no select-window at all) rather than issuing one with a
        bogus/empty target, and it warns so the skip is observable.
        """
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'display-message', '-p', '-t', 's', '#{window_id}'): CommandResult(
                    returncode=1, stderr='session not found: s'
                ),
            }
        )
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux', tmux_target='s:0')]

        with caplog.at_level(logging.WARNING):
            backend.reorder(targets)

        assert not any(argv[1] == 'select-window' for argv in runner.calls)
        assert not any(argv[1] == 'switch-client' for argv in runner.calls)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_reorder_warns_when_restore_select_window_fails(self, caplog):
        """If the final restore select-window fails (e.g. the snapshotted window
        vanished mid-reorder), reorder must not silently no-op -- it warns so a
        failed focus restoration is observable, mirroring every other _run call
        in this method (move-window and the display-message snapshot read both
        already warn on a nonzero rc).
        """
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'display-message', '-p', '-t', 's', '#{window_id}'): CommandResult(
                    returncode=0, stdout='@7\n'
                ),
                ('tmux', 'select-window', '-t', '@7'): CommandResult(
                    returncode=1, stderr='window not found: @7'
                ),
            }
        )
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux', tmux_target='s:0')]

        with caplog.at_level(logging.WARNING):
            backend.reorder(targets)

        # The restore is still attempted (and is still the last call reorder
        # issues) even though it fails -- only the diagnostic changes.
        assert runner.calls[-1] == ['tmux', 'select-window', '-t', '@7']
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_reorder_skips_target_with_missing_tmux_target_and_warns(self, caplog):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner()
        backend = TmuxBackend(run=runner)
        targets = [DisplayTarget(kind='tmux'), DisplayTarget(kind='tmux', tmux_target='s:0')]

        with caplog.at_level(logging.WARNING):
            backend.reorder(targets)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert runner.calls == [
            ['tmux', 'display-message', '-p', '-t', 's', '#{window_id}'],
            ['tmux', 'move-window', '-d', '-s', 's:0', '-t', 's:9000'],
            ['tmux', 'move-window', '-d', '-s', 's:9000', '-t', 's:1'],
        ]


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

    def test_false_when_target_is_only_a_substring_of_another_windows_line(self):
        """'s:0' must not match a line for 's:01' or a session literally named 'somes'."""
        from cockpit.backends.base import CommandResult, DisplayTarget
        from cockpit.backends.tmux import TmuxBackend

        runner = ScriptedRunner(
            results={
                ('tmux', 'list-windows', '-a'): CommandResult(
                    returncode=0, stdout='s:01: bash\nsomes:0: zsh\n'
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
