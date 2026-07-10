"""Base seam types for the Fleet Cockpit focus/arrange backends (PRD §6.2 / C4).

FocusArrangeBackend is the single interface wm/X11 and tmux backends
implement; DisplayTarget is a local, concrete stand-in for orchestrator's
session_registry.Display (C1) so cockpit never imports orchestrator (PRD
§5.1/§5.7) — mirrors priority.py's ScoringItem pattern. Backends read
DisplayTarget's attributes structurally, never isinstance it.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_COMMAND_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class DisplayTarget:
    """Where to find/focus a session's terminal.

    Mirrors orchestrator.session_registry.Display's field shape (kind,
    wm_title, wm_window_id, tmux_target). Any object exposing these four
    attributes works — backends read attributes, not isinstance — so a real
    orchestrator Display satisfies this structurally with zero import
    coupling; C5b maps Display -> DisplayTarget at the call site.
    """

    kind: str
    wm_title: str = ''
    wm_window_id: str | None = None
    tmux_target: str | None = None


@dataclass(frozen=True)
class Zone:
    """A screen rectangle to tile a window into (wm-only; PRD §6.2)."""

    x: int
    y: int
    width: int
    height: int
    gravity: int = 0


@dataclass(frozen=True)
class FocusResult:
    """Outcome of a focus() call. ok=False on a gone/unaddressable target, never an exception."""

    ok: bool
    note: str = ''


@dataclass(frozen=True)
class CommandResult:
    """Outcome of running an external command (wmctrl/xdotool/tmux)."""

    returncode: int
    stdout: str = ''
    stderr: str = ''


CommandRunner = Callable[[Sequence[str]], CommandResult]


def run_command(argv: Sequence[str]) -> CommandResult:
    """Run *argv*, fail-soft: a missing binary/timeout returns a nonzero CommandResult.

    Never raises — the cockpit's hard constraint that a view must never be a
    dependency. Callers (WmBackend/TmuxBackend) treat a nonzero returncode as
    the target being gone/unaddressable and no-op with a warning.
    """
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=_COMMAND_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning('run_command: %s failed: %s', list(argv), exc)
        return CommandResult(returncode=127, stderr=str(exc))
    return CommandResult(
        returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
    )


@runtime_checkable
class FocusArrangeBackend(Protocol):
    """Uniform focus/arrange surface for a session's terminal (PRD §6.2 / C4).

    wm/X11 (WmBackend) and tmux (TmuxBackend) each implement this; a future
    kdotool/KWin (Wayland) backend only needs to satisfy this Protocol — no
    cockpit change required (PRD §5.7).
    """

    def focus(self, target: DisplayTarget) -> FocusResult:
        """Raise/activate target's window or pane. No-op + warning if gone."""
        ...

    def set_urgency(self, target: DisplayTarget, on: bool) -> None:
        """Set or clear an attention hint on target. No-op + warning if gone."""
        ...

    def reorder(self, targets: Sequence[DisplayTarget]) -> None:
        """Reorder targets by priority without touching focus."""
        ...

    def tile(self, targets: Sequence[DisplayTarget], zone: Zone) -> None:
        """Arrange targets into zone. wm-only; no-op elsewhere."""
        ...

    def is_alive(self, target: DisplayTarget) -> bool:
        """Whether target still resolves to a live window/pane."""
        ...
