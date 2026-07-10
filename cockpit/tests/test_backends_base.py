"""Tests for cockpit.backends.base — the FocusArrangeBackend seam types (PRD §6.2 / C4).

Every cockpit symbol is imported inline per test, so this file is
ImportError-RED until cockpit.backends.base exists (step-2). DisplayTarget
mirrors orchestrator.session_registry.Display's field shape but is a local,
concrete dataclass — backends read attributes off a target, never isinstance
it — so cockpit stays decoupled from orchestrator (mirrors priority.py's
ScoringItem stand-in for the same reason).
"""

from __future__ import annotations

import dataclasses
import logging

import pytest


class TestDisplayTarget:
    def test_frozen_dataclass_with_defaults(self):
        from cockpit.backends.base import DisplayTarget

        target = DisplayTarget(kind='wm')

        assert target.kind == 'wm'
        assert target.wm_title == ''
        assert target.wm_window_id is None
        assert target.tmux_target is None

    def test_is_frozen(self):
        from cockpit.backends.base import DisplayTarget

        target = DisplayTarget(kind='tmux', tmux_target='session:0')

        with pytest.raises(dataclasses.FrozenInstanceError):
            target.kind = 'wm'

    def test_duck_typed_object_exposes_same_attribute_surface(self):
        """Backends read attrs off any target-shaped object, not isinstance — see module docstring."""
        from cockpit.backends.base import DisplayTarget

        class PlainDuckTarget:
            kind = 'tmux'
            wm_title = ''
            wm_window_id = None
            tmux_target = 'session:1'

        duck = PlainDuckTarget()

        assert not isinstance(duck, DisplayTarget)
        assert duck.kind == 'tmux'
        assert duck.wm_title == ''
        assert duck.wm_window_id is None
        assert duck.tmux_target == 'session:1'


class TestZone:
    def test_frozen_dataclass_with_default_gravity(self):
        from cockpit.backends.base import Zone

        zone = Zone(x=0, y=10, width=800, height=600)

        assert zone.x == 0
        assert zone.y == 10
        assert zone.width == 800
        assert zone.height == 600
        assert zone.gravity == 0

    def test_gravity_is_overridable(self):
        from cockpit.backends.base import Zone

        zone = Zone(x=0, y=0, width=100, height=100, gravity=5)

        assert zone.gravity == 5

    def test_is_frozen(self):
        from cockpit.backends.base import Zone

        zone = Zone(x=0, y=0, width=100, height=100)

        with pytest.raises(dataclasses.FrozenInstanceError):
            zone.x = 1


class TestFocusResult:
    def test_frozen_dataclass_with_default_note(self):
        from cockpit.backends.base import FocusResult

        result = FocusResult(ok=True)

        assert result.ok is True
        assert result.note == ''

    def test_note_is_overridable(self):
        from cockpit.backends.base import FocusResult

        result = FocusResult(ok=False, note='window gone')

        assert result.ok is False
        assert result.note == 'window gone'

    def test_is_frozen(self):
        from cockpit.backends.base import FocusResult

        result = FocusResult(ok=True)

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.ok = False


class TestCommandResult:
    def test_frozen_dataclass_with_defaults(self):
        from cockpit.backends.base import CommandResult

        result = CommandResult(returncode=0)

        assert result.returncode == 0
        assert result.stdout == ''
        assert result.stderr == ''

    def test_stdout_and_stderr_are_overridable(self):
        from cockpit.backends.base import CommandResult

        result = CommandResult(returncode=1, stdout='out', stderr='err')

        assert result.returncode == 1
        assert result.stdout == 'out'
        assert result.stderr == 'err'

    def test_is_frozen(self):
        from cockpit.backends.base import CommandResult

        result = CommandResult(returncode=0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.returncode = 1


class TestFocusArrangeBackendProtocol:
    def test_conforming_object_is_a_structural_instance(self):
        from cockpit.backends.base import FocusArrangeBackend, FocusResult

        class Conforming:
            def focus(self, target):
                return FocusResult(ok=True)

            def set_urgency(self, target, on):
                return None

            def reorder(self, targets):
                return None

            def tile(self, targets, zone):
                return None

            def is_alive(self, target):
                return True

        assert isinstance(Conforming(), FocusArrangeBackend)

    def test_object_missing_a_method_is_not_a_structural_instance(self):
        from cockpit.backends.base import FocusArrangeBackend

        class MissingTile:
            def focus(self, target):
                return None

            def set_urgency(self, target, on):
                return None

            def reorder(self, targets):
                return None

            def is_alive(self, target):
                return True

        assert not isinstance(MissingTile(), FocusArrangeBackend)


class TestRunCommand:
    def test_missing_binary_is_fail_soft(self, caplog):
        """Never raises; returns a nonzero CommandResult instead (cockpit's fail-soft contract)."""
        from cockpit.backends.base import CommandResult, run_command

        with caplog.at_level(logging.WARNING):
            result = run_command(['definitely-not-a-real-binary-xyz'])

        assert isinstance(result, CommandResult)
        assert result.returncode != 0
