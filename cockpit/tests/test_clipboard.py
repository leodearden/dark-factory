"""Tests for cockpit.clipboard — the real system-clipboard path (task 5448).

Task 2517 shipped 'y' as Textual's App.copy_to_clipboard alone, i.e. an OSC
52 escape sequence, which is a measured total no-op on Konsole 23.08.5 —
and its tests asserted on `app._clipboard`, an attribute Textual sets
BEFORE the escape write and regardless of terminal support, so they passed
against a copy that did nothing. This module asserts at the only boundary
that can tell those apart: the exact argv and the exact stdin bytes handed
to the clipboard helper process.

Every dependency is injected as a plain dict/function/class (never
MagicMock — fused-memory/scripts/check_bare_magicmock_config.py scans
cockpit/tests), so the environment-gated and command-selection tests are
hermetic on a host with or without a display. The default runner's own
tests deliberately do NOT monkeypatch subprocess: the process boundary is
the thing under test, so they run real `sh` helpers.

`cockpit.clipboard` doesn't exist yet, so every test imports it inline and
is ImportError-RED until its impl step (mirrors test_backends_wm.py /
test_backends_base.py conventions).
"""

from __future__ import annotations


def _which_all(name):
    """A `shutil.which` double for which every clipboard helper resolves."""
    return f'/usr/bin/{name}'


class RecordingWhich:
    """A `shutil.which` double: records every lookup, resolves all but `missing`."""

    def __init__(self, missing=()):
        self.lookups: list[str] = []
        self._missing = set(missing)

    def __call__(self, name):
        self.lookups.append(name)
        return None if name in self._missing else f'/usr/bin/{name}'


_XCLIP = ('xclip', '-selection', 'clipboard')
_XSEL = ('xsel', '--clipboard', '--input')
_WL_COPY = ('wl-copy',)


class TestAvailableCopyCommands:
    def test_wayland_session_prefers_wl_copy(self):
        """WAYLAND_DISPLAY set -> the Wayland helper is the FIRST candidate.

        Preference order is fixed by the module, not by PATH order, so a
        Wayland session never reaches for an X11 helper first.
        """
        from cockpit.clipboard import available_copy_commands

        commands = available_copy_commands(
            environ={'WAYLAND_DISPLAY': 'wayland-0'}, which=_which_all
        )

        assert commands[0] == _WL_COPY

    def test_x11_session_yields_xclip_then_xsel(self):
        """DISPLAY set (no Wayland) -> exactly the two X11 helpers, xclip first."""
        from cockpit.clipboard import available_copy_commands

        commands = available_copy_commands(environ={'DISPLAY': ':0'}, which=_which_all)

        assert commands == (_XCLIP, _XSEL)

    def test_a_helper_missing_from_path_is_dropped(self):
        """`which` is the arbiter of presence: no xclip -> only the xsel candidate survives."""
        from cockpit.clipboard import available_copy_commands

        which = RecordingWhich(missing={'xclip'})

        commands = available_copy_commands(environ={'DISPLAY': ':0'}, which=which)

        assert commands == (_XSEL,)

    def test_no_display_environment_yields_no_candidates_and_no_lookups(self):
        """The over-SSH case: neither display var set -> (), and `which` is never consulted.

        With no display there is nothing a local helper could reach, so the
        caller falls straight through to the OSC 52 fallback without paying
        for a doomed subprocess — the environment gate runs BEFORE the PATH
        lookup, which is what the empty `lookups` list proves.
        """
        from cockpit.clipboard import available_copy_commands

        which = RecordingWhich()

        commands = available_copy_commands(environ={}, which=which)

        assert commands == ()
        assert which.lookups == []

    def test_both_display_vars_set_puts_wayland_first_then_x11(self):
        """A Wayland session running XWayland exports both vars: wl-copy leads, X11 follows."""
        from cockpit.clipboard import available_copy_commands

        commands = available_copy_commands(
            environ={'WAYLAND_DISPLAY': 'wayland-0', 'DISPLAY': ':0'}, which=_which_all
        )

        assert commands == (_WL_COPY, _XCLIP, _XSEL)
