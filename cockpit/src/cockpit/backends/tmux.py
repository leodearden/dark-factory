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

# A tmux session's live window indices are small (dozens at most in practice).
# Staging every reorder move through this scratch range first guarantees phase
# 1 can never collide with a destination index a later phase still needs to
# vacate (or with its own current index) — see reorder()'s docstring.
_REORDER_SCRATCH_BASE = 9000


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
        """Move each target into its priority-order index within its own session.

        Two-phase (park-then-place): a naive single-pass move-window can hit
        tmux's "index in use" error when a destination index is still
        occupied by another target awaiting its own move (e.g. targets
        [s:2, s:0] — moving s:2 straight to s:0 collides with s:0 before it
        has vacated). Phase 1 parks every valid target at a scratch index
        (_REORDER_SCRATCH_BASE + its position among valid targets, which
        never collides with the final 0..N-1 destination range); phase 2
        moves each target from its scratch index to its final compacted
        index. Focus-preserving throughout: only move-window is issued, never
        select-window/switch-client.
        """
        # (index, tmux_target) — captured as a plain str (not target.tmux_target,
        # which stays str | None) so the loop below is narrowing-clean for pyright.
        valid: list[tuple[int, str]] = []
        for index, target in enumerate(targets):
            if not target.tmux_target:
                logger.warning('TmuxBackend.reorder: target has no tmux_target: %r', target)
                continue
            valid.append((index, target.tmux_target))

        parked: list[tuple[int, str, str]] = []  # (final_index, session, scratch_target)
        for position, (index, tmux_target) in enumerate(valid):
            session = tmux_target.split(':', 1)[0]
            scratch = f'{session}:{_REORDER_SCRATCH_BASE + position}'
            result = self._run(['tmux', 'move-window', '-s', tmux_target, '-t', scratch])
            if result.returncode != 0:
                logger.warning(
                    'TmuxBackend.reorder: move-window %r -> %r failed (rc=%s): %s',
                    tmux_target,
                    scratch,
                    result.returncode,
                    result.stderr,
                )
                continue
            parked.append((index, session, scratch))

        for index, session, scratch in parked:
            dst = f'{session}:{index}'
            result = self._run(['tmux', 'move-window', '-s', scratch, '-t', dst])
            if result.returncode != 0:
                logger.warning(
                    'TmuxBackend.reorder: move-window %r -> %r failed (rc=%s): %s',
                    scratch,
                    dst,
                    result.returncode,
                    result.stderr,
                )

    def tile(self, targets: Sequence[DisplayTarget], zone: Zone) -> None:
        """No-op: tmux windows aren't X11-tiled; tile is wm-only."""
        logger.debug('TmuxBackend.tile: tmux windows are not X11-tiled; tile is wm-only')

    def is_alive(self, target: DisplayTarget) -> bool:
        """Whether target still appears in `tmux list-windows -a`'s output.

        Each line is formatted `session:index: ...`; we compare the exact
        `session:index` token rather than a raw substring, so e.g. target
        's:0' can't false-positive against a line for 's:01' or a session
        literally named 'somes' at window 0.
        """
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

        for line in result.stdout.splitlines():
            parts = line.split(':', 2)
            if len(parts) < 2:
                continue
            session, index = parts[0], parts[1]
            if f'{session}:{index}' == target.tmux_target:
                return True
        return False
