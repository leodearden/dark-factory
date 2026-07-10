"""cockpit.backends.tmux — TmuxBackend, the tmux implementation (PRD §6.2 / C4).

Every op is fail-soft: a missing binary, a nonzero return code, or a target
with no tmux_target logs a warning and no-ops rather than raising (cockpit's
hard constraint that a view must never be a dependency).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cockpit.backends.base import CommandRunner, DisplayTarget, FocusResult, Zone, run_command

logger = logging.getLogger(__name__)


class TmuxBackend:
    """Focus/arrange sessions running inside tmux."""

    def __init__(self, run: CommandRunner = run_command) -> None:
        self._run = run

    def focus(self, target: DisplayTarget) -> FocusResult:
        """Select then switch-client onto target's window. No switch-client on a failed select."""
        if not target.tmux_target:
            logger.warning('TmuxBackend.focus: target has no tmux_target: %r', target)
            return FocusResult(ok=False, note='no tmux_target')

        result = self._run(['tmux', 'select-window', '-t', target.tmux_target])
        if result.returncode != 0:
            logger.warning(
                'TmuxBackend.focus: select-window %r failed (rc=%s): %s',
                target.tmux_target,
                result.returncode,
                result.stderr,
            )
            return FocusResult(ok=False, note='window not found')

        self._run(['tmux', 'switch-client', '-t', target.tmux_target])
        return FocusResult(ok=True)

    def set_urgency(self, target: DisplayTarget, on: bool) -> None:
        """No-op: tmux has no per-window urgency hint (signal-don't-move)."""
        logger.debug(
            "TmuxBackend.set_urgency: tmux has no per-window urgency hint (signal-don't-move)"
        )

    def reorder(self, targets: Sequence[DisplayTarget]) -> None:
        """Move each target to its priority-order index within its own session via move-window.

        Focus-preserving: issues only move-window, never select-window/switch-client.
        """
        for index, target in enumerate(targets):
            if not target.tmux_target:
                logger.warning('TmuxBackend.reorder: target has no tmux_target: %r', target)
                continue

            session = target.tmux_target.split(':', 1)[0]
            dst = f'{session}:{index}'
            result = self._run(['tmux', 'move-window', '-s', target.tmux_target, '-t', dst])
            if result.returncode != 0:
                logger.warning(
                    'TmuxBackend.reorder: move-window %r -> %r failed (rc=%s): %s',
                    target.tmux_target,
                    dst,
                    result.returncode,
                    result.stderr,
                )

    def tile(self, targets: Sequence[DisplayTarget], zone: Zone) -> None:
        """No-op: tmux windows aren't X11-tiled; tile is wm-only."""
        logger.debug('TmuxBackend.tile: tmux windows are not X11-tiled; tile is wm-only')

    def is_alive(self, target: DisplayTarget) -> bool:
        """Whether target still appears in `tmux list-windows -a`'s output."""
        if not target.tmux_target:
            logger.warning('TmuxBackend.is_alive: target has no tmux_target: %r', target)
            return False

        result = self._run(['tmux', 'list-windows', '-a'])
        if result.returncode != 0:
            logger.warning(
                'TmuxBackend.is_alive: tmux list-windows failed (rc=%s): %s',
                result.returncode,
                result.stderr,
            )
            return False

        return target.tmux_target in result.stdout
