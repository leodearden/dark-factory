"""Tests for cockpit.backends.fakes — FakeBackend, the Protocol-level spy (PRD §6.2 / C4).

FakeBackend is the mechanism C5b's B3 relies on: a refresh path that issues
only set_urgency/reorder must leave focus_calls/tile_calls empty. Distinct
from the injected fake CommandRunner used against the real wm/tmux backends
(see test_backends_wm.py / test_backends_tmux.py) — this fake implements
FocusArrangeBackend directly; no subprocess is involved.
"""

from __future__ import annotations


class TestFakeBackendProtocolConformance:
    def test_is_a_structural_instance_of_focus_arrange_backend(self):
        from cockpit.backends.base import FocusArrangeBackend
        from cockpit.backends.fakes import FakeBackend

        assert isinstance(FakeBackend(), FocusArrangeBackend)


class TestFakeBackendCallBuckets:
    def test_focus_is_recorded(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        target = DisplayTarget(kind='wm', wm_title='session-a')

        fake.focus(target)

        assert fake.focus_calls == [target]

    def test_set_urgency_is_recorded_with_on_flag(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        target = DisplayTarget(kind='wm', wm_title='session-a')

        fake.set_urgency(target, True)

        assert fake.set_urgency_calls == [(target, True)]

    def test_reorder_is_recorded_with_full_ordered_list(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        targets = [DisplayTarget(kind='wm', wm_title='a'), DisplayTarget(kind='wm', wm_title='b')]

        fake.reorder(targets)

        assert fake.reorder_calls == [targets]

    def test_tile_is_recorded_with_targets_and_zone(self):
        from cockpit.backends.base import DisplayTarget, Zone
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        targets = [DisplayTarget(kind='wm', wm_title='a')]
        zone = Zone(x=0, y=0, width=100, height=100)

        fake.tile(targets, zone)

        assert fake.tile_calls == [(targets, zone)]

    def test_reorder_call_is_a_snapshot_not_a_live_reference(self):
        """Mutating the caller's list after the call must not retroactively rewrite history."""
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        targets = [DisplayTarget(kind='wm', wm_title='a')]

        fake.reorder(targets)
        targets.append(DisplayTarget(kind='wm', wm_title='b'))

        assert fake.reorder_calls == [[DisplayTarget(kind='wm', wm_title='a')]]

    def test_tile_call_is_a_snapshot_not_a_live_reference(self):
        """Mutating the caller's list after the call must not retroactively rewrite history."""
        from cockpit.backends.base import DisplayTarget, Zone
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        targets = [DisplayTarget(kind='wm', wm_title='a')]
        zone = Zone(x=0, y=0, width=100, height=100)

        fake.tile(targets, zone)
        targets.append(DisplayTarget(kind='wm', wm_title='b'))

        assert fake.tile_calls == [([DisplayTarget(kind='wm', wm_title='a')], zone)]

    def test_is_alive_is_recorded(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()
        target = DisplayTarget(kind='wm', wm_title='a')

        fake.is_alive(target)

        assert fake.is_alive_calls == [target]


class TestFakeBackendFocusResult:
    def test_focus_defaults_to_ok_true(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()

        result = fake.focus(DisplayTarget(kind='wm', wm_title='a'))

        assert result.ok is True

    def test_focus_result_is_configurable_for_gone_targets(self):
        from cockpit.backends.base import DisplayTarget, FocusResult
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend(focus_result=FocusResult(ok=False, note='gone'))

        result = fake.focus(DisplayTarget(kind='wm', wm_title='a'))

        assert result == FocusResult(ok=False, note='gone')


class TestFakeBackendIsAlive:
    def test_defaults_to_true(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend()

        assert fake.is_alive(DisplayTarget(kind='wm', wm_title='a')) is True

    def test_scriptable_per_target_via_mapping(self):
        """A mapping keyed by target scripts per-target gone/alive for tests."""
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        gone = DisplayTarget(kind='wm', wm_title='gone-window')
        present = DisplayTarget(kind='wm', wm_title='present-window')
        fake = FakeBackend(alive={gone: False})

        assert fake.is_alive(gone) is False
        assert fake.is_alive(present) is True

    def test_scriptable_via_callable(self):
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

        fake = FakeBackend(alive=lambda target: target.wm_title != 'gone-window')

        assert fake.is_alive(DisplayTarget(kind='wm', wm_title='gone-window')) is False
        assert fake.is_alive(DisplayTarget(kind='wm', wm_title='present-window')) is True


class TestFakeBackendRefreshShape:
    def test_refresh_only_calls_leave_focus_and_tile_empty(self):
        """B3-shape: a simulated refresh calling only set_urgency/reorder never touches focus/tile.

        This is the mechanism C5b's real B3 relies on to assert the refresh
        path issues zero focus/tile calls against the wired backend.
        """
        from cockpit.backends.base import DisplayTarget
        from cockpit.backends.fakes import FakeBackend

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
