"""Fleet Cockpit focus/arrange backends (PRD §6.2 / C4).

Single import surface for C5b (focus wiring) and C-smoke (live proof).
"""

from __future__ import annotations

from cockpit.backends.base import (
    CommandResult,
    CommandRunner,
    DisplayTarget,
    FocusArrangeBackend,
    FocusResult,
    Zone,
    run_command,
)
from cockpit.backends.fakes import FakeBackend
from cockpit.backends.tmux import TmuxBackend
from cockpit.backends.wm import WmBackend

__all__ = [
    'CommandResult',
    'CommandRunner',
    'DisplayTarget',
    'FakeBackend',
    'FocusArrangeBackend',
    'FocusResult',
    'TmuxBackend',
    'WmBackend',
    'Zone',
    'run_command',
]
