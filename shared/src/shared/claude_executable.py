"""Which claude executable a launch surface runs: the surface's own override variable, else bare ``claude`` on PATH.

Each surface has exactly one override variable, and no surface reads another
surface's. ``skills/spawn/spawn-claude.sh`` runs this file by path under
``python3 -I``, so it imports nothing from ``shared``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    'DEFAULT_CLAUDE_EXECUTABLE',
    'ClaudeLaunchSurface',
    'claude_executable',
    'main',
    'override_var',
]

DEFAULT_CLAUDE_EXECUTABLE = 'claude'


class ClaudeLaunchSurface(StrEnum):
    """A place that launches the claude CLI and may override which executable it runs."""

    #: Agents spawned through ``shared.cli_invoke``.
    FLEET = 'fleet'
    #: Interactive sessions ``skills/spawn/spawn-claude.sh`` launches on its
    #: tmux and terminal-emulator lanes.
    SPAWN = 'spawn'


_OVERRIDE_VARS: Mapping[ClaudeLaunchSurface, str] = MappingProxyType({
    ClaudeLaunchSurface.FLEET: 'CLAUDE_BINARY',
    ClaudeLaunchSurface.SPAWN: 'CLAUDE_SPAWN_CLAUDE_BIN',
})


def override_var(surface: ClaudeLaunchSurface) -> str:
    """The one environment variable that overrides ``surface``'s executable."""
    return _OVERRIDE_VARS[surface]


def claude_executable(surface: ClaudeLaunchSurface, environ: Mapping[str, str]) -> str:
    """``surface``'s override from ``environ``, or ``'claude'`` when it is unset or empty.

    The result is a bare name (resolved on PATH by whoever execs it) or a path.
    Never raises.
    """
    return environ.get(override_var(surface)) or DEFAULT_CLAUDE_EXECUTABLE


def main(argv: Sequence[str] | None = None) -> int:
    """Print the executable for the surface named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('surface', choices=[surface.value for surface in ClaudeLaunchSurface])
    surface = ClaudeLaunchSurface(parser.parse_args(argv).surface)
    print(claude_executable(surface, os.environ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
