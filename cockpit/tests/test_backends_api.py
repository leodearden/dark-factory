"""Tests for the cockpit.backends package public surface (PRD §6.2 / C4).

cockpit.backends.__init__ re-exports the full public API, so every test here
imports through the *package* — `from cockpit.backends import ...` — rather
than the submodules. This is the single import surface C5b and C-smoke
consume.
"""

from __future__ import annotations

import logging

import pytest


class TestPublicApiSurface:
    def test_public_api_names_resolve(self):
        from cockpit.backends import (
            CommandResult,
            CommandRunner,
            DisplayTarget,
            FakeBackend,
            FocusArrangeBackend,
            FocusResult,
            TmuxBackend,
            WmBackend,
            Zone,
            run_command,
        )

        assert FocusArrangeBackend is not None
        assert DisplayTarget is not None
        assert Zone is not None
        assert FocusResult is not None
        assert CommandResult is not None
        assert CommandRunner is not None
        assert callable(run_command)
        assert WmBackend is not None
        assert TmuxBackend is not None
        assert FakeBackend is not None

    def test_dunder_all_lists_the_public_surface(self):
        import cockpit.backends as backends_pkg

        assert set(backends_pkg.__all__) == {
            'FocusArrangeBackend',
            'DisplayTarget',
            'Zone',
            'FocusResult',
            'CommandResult',
            'CommandRunner',
            'run_command',
            'WmBackend',
            'TmuxBackend',
            'FakeBackend',
        }


class TestWaylandReadiness:
    def test_all_three_backends_satisfy_the_protocol(self):
        """A future kdotool/KWin backend only needs to implement the Protocol — no cockpit change."""
        from cockpit.backends import FakeBackend, FocusArrangeBackend, TmuxBackend, WmBackend

        assert isinstance(WmBackend(), FocusArrangeBackend)
        assert isinstance(TmuxBackend(), FocusArrangeBackend)
        assert isinstance(FakeBackend(), FocusArrangeBackend)


class TestCrossBackendGoneInvariant:
    @pytest.mark.parametrize('backend_name', ['wm', 'tmux'])
    def test_every_method_is_fail_soft_on_a_gone_target(self, backend_name, caplog):
        from cockpit.backends import CommandResult, DisplayTarget, TmuxBackend, WmBackend, Zone

        def all_failing_runner(argv):
            return CommandResult(returncode=1, stderr='no such window')

        if backend_name == 'wm':
            backend = WmBackend(run=all_failing_runner)
            target = DisplayTarget(kind='wm', wm_title='gone-window')
        else:
            backend = TmuxBackend(run=all_failing_runner)
            target = DisplayTarget(kind='tmux', tmux_target='gone:0')

        with caplog.at_level(logging.WARNING):
            focus_result = backend.focus(target)
            backend.set_urgency(target, True)
            backend.reorder([target])
            backend.tile([target], Zone(x=0, y=0, width=100, height=100))
            alive = backend.is_alive(target)

        # Never raises: reaching this line already proves it.
        assert focus_result.ok is False
        assert alive is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestRefreshShapeIntegrationLite:
    def test_simulated_refresh_leaves_focus_and_tile_empty(self):
        """B3-shape via the public package surface — the mechanism C5b's real B3 relies on."""
        from cockpit.backends import DisplayTarget, FakeBackend

        fake = FakeBackend()
        targets = [
            DisplayTarget(kind='wm', wm_title='a'),
            DisplayTarget(kind='tmux', tmux_target='s:0'),
        ]

        def simulated_refresh(backend, ordered_targets):
            for t in ordered_targets:
                backend.set_urgency(t, True)
            backend.reorder(ordered_targets)

        simulated_refresh(fake, targets)

        assert fake.focus_calls == []
        assert fake.tile_calls == []
        assert len(fake.set_urgency_calls) == 2
        assert fake.reorder_calls == [targets]
