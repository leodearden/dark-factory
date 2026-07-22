"""cockpit.backends.wm — WmBackend, the wm/X11 implementation (PRD §6.2 / C4).

Every op is fail-soft: a missing binary, a nonzero return code, or a target
with no wm_title/wm_window_id logs a warning and no-ops rather than raising
(cockpit's hard constraint that a view must never be a dependency).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cockpit.backends.base import CommandRunner, DisplayTarget, FocusResult, Zone, run_command

logger = logging.getLogger(__name__)


def _window_id_int(raw: str | None) -> int | None:
    """Parse a window id string to its integer value, base-autodetecting.

    ``int(raw, 0)`` accepts both a ``0x``-prefixed hex string (wmctrl -l's
    canonical column form, e.g. ``'0x03200007'``) and a bare decimal (what a
    real terminal emulator exports in ``$WINDOWID``, e.g. ``'52428807'``) and
    yields the SAME integer for the two encodings of one window. Fail-soft: a
    missing (``None``) or unparseable id returns ``None`` so the caller skips
    the id path rather than raising.

    NOTE: this parse/canonicalize logic is mirrored (duplicated, not
    shared/imported, since this is a cross-package boundary — cockpit must
    never import orchestrator) in
    ``orchestrator/src/orchestrator/session_hooks.py::_canonical_window_id``,
    which keys on the identical ``int(s, 0)`` contract. Keep the two in sync.
    """
    try:
        return int(raw, 0)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


class WmBackend:
    """Focus/arrange sessions running under an X11 window manager (wmctrl/xdotool)."""

    def __init__(self, run: CommandRunner = run_command) -> None:
        self._run = run

    def focus(self, target: DisplayTarget) -> FocusResult:
        """Raise target's window.

        Prefers the stable `wm_window_id` (`wmctrl -i -a <id>`) over
        `wm_title`, since a spawned terminal's title churns during launch
        and thereafter carries status glyphs from OSC retitling, making a
        captured title snapshot unreliable within seconds. Falls back to
        `wmctrl -a <title>` then `xdotool windowactivate` when no id is
        available or activating by id fails.
        """
        if not target.wm_window_id and not target.wm_title:
            logger.warning(
                'WmBackend.focus: target has no wm_title or wm_window_id: %r', target
            )
            return FocusResult(ok=False, note='no target')

        if target.wm_window_id:
            id_result = self._run(['wmctrl', '-i', '-a', target.wm_window_id])
            if id_result.returncode == 0:
                return FocusResult(ok=True)

        if not target.wm_title:
            logger.warning(
                'WmBackend.focus: could not focus window id %r (wmctrl rc=%s)',
                target.wm_window_id,
                id_result.returncode,
            )
            return FocusResult(ok=False, note='window not found')

        result = self._run(['wmctrl', '-a', target.wm_title])
        if result.returncode == 0:
            return FocusResult(ok=True)

        fallback = self._run(['xdotool', 'search', '--name', target.wm_title, 'windowactivate'])
        if fallback.returncode == 0:
            return FocusResult(ok=True)

        logger.warning(
            'WmBackend.focus: could not focus %r (wmctrl rc=%s, xdotool rc=%s)',
            target.wm_title,
            result.returncode,
            fallback.returncode,
        )
        return FocusResult(ok=False, note='window not found')

    def set_urgency(self, target: DisplayTarget, on: bool) -> None:
        """Set/clear the urgency hint via a single xdotool set_window --urgency command."""
        flag = '1' if on else '0'
        if target.wm_window_id:
            argv = ['xdotool', 'set_window', '--urgency', flag, target.wm_window_id]
        elif target.wm_title:
            argv = ['xdotool', 'search', '--name', target.wm_title, 'set_window', '--urgency', flag]
        else:
            logger.warning(
                'WmBackend.set_urgency: target has no wm_title or wm_window_id: %r', target
            )
            return

        result = self._run(argv)
        if result.returncode != 0:
            logger.warning(
                'WmBackend.set_urgency: %s failed (rc=%s): %s',
                argv,
                result.returncode,
                result.stderr,
            )

    def reorder(self, targets: Sequence[DisplayTarget]) -> None:
        """No-op: wm windows are never auto-reordered (signal-don't-move; PRD §6.2)."""
        logger.debug(
            "WmBackend.reorder: wm backend does not auto-reorder windows (signal-don't-move)"
        )

    def tile(self, targets: Sequence[DisplayTarget], zone: Zone) -> None:
        """Move/resize each target into zone via `wmctrl -r -e`."""
        mvarg = f'{zone.gravity},{zone.x},{zone.y},{zone.width},{zone.height}'
        for target in targets:
            if not target.wm_title:
                logger.warning('WmBackend.tile: target has no wm_title: %r', target)
                continue

            argv = ['wmctrl', '-r', target.wm_title, '-e', mvarg]
            result = self._run(argv)
            if result.returncode != 0:
                logger.warning(
                    'WmBackend.tile: %s failed (rc=%s): %s', argv, result.returncode, result.stderr
                )

    def is_alive(self, target: DisplayTarget) -> bool:
        """Whether target still resolves to a live window in `wmctrl -l`'s output.

        `wmctrl -l` lines are `<id> <desktop> <host> <title>`; we parse those
        fixed columns and compare the title field exactly (or the window id,
        when known) rather than a raw substring, so e.g. title 'a' can't
        false-positive against a longer title like 'session-a'.

        The window-id comparison is by INTEGER identity (via `_window_id_int`
        / `int(s, 0)`), not exact string: a session captured from a DECIMAL
        `$WINDOWID` (e.g. '52428807') still matches wmctrl -l's canonical
        zero-padded hex column ('0x03200007') because both parse to the same
        int. An unparseable/missing id falls through fail-soft to the title
        match.

        NOTE: this exact `split(None, 3)` parse is duplicated (not
        shared/imported, since this is a cross-package boundary) in
        `orchestrator/src/orchestrator/session_hooks.py::_resolve_wm_window_id`
        -- if the column layout or matching rule here ever changes, mirror
        the change there too. The integer-identity window-id compare is
        likewise mirrored by `session_hooks._canonical_window_id`'s
        `int(s, 0)` contract (see `_window_id_int` above).
        """
        if not target.wm_title and not target.wm_window_id:
            logger.warning(
                'WmBackend.is_alive: target has no wm_title or wm_window_id: %r', target
            )
            return False

        result = self._run(['wmctrl', '-l'])
        if result.returncode != 0:
            logger.warning(
                'WmBackend.is_alive: wmctrl -l failed (rc=%s): %s', result.returncode, result.stderr
            )
            return False

        target_id = _window_id_int(target.wm_window_id)
        for line in result.stdout.splitlines():
            columns = line.split(None, 3)
            if len(columns) < 4:
                continue
            window_id, _desktop, _host, title = columns
            if target_id is not None and _window_id_int(window_id) == target_id:
                return True
            if target.wm_title and title == target.wm_title:
                return True
        return False
