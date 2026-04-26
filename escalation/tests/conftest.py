"""pytest configuration — ensure local src takes precedence over installed package.

Without this conftest, ``import escalation`` resolves to the editable
install in the prod venv (which points at /home/leo/src/dark-factory/),
not the worktree under test.  Injecting the worktree's ``src`` at the
front of ``sys.path`` mirrors the orchestrator/tests pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
