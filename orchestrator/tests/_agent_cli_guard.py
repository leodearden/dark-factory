"""No orchestrator test may start an agent-CLI process (task 4701).

A stale monkeypatch target degrades to calling the real CLI instead of
erroring (the task-882 incident), so the guard sits at
``subprocess.Popen.__init__``: the one spawn chokepoint no name binding can
bypass.
"""

from __future__ import annotations

import os
from typing import Any

AGENT_CLI_BINARIES: frozenset[str] = frozenset({'claude', 'codex', 'gemini', 'pi'})


def agent_cli_in(args: Any) -> str | None:
    """The agent-CLI binary named by any element of ``args``, else None."""
    elements = [args] if isinstance(args, str | bytes | os.PathLike) else args
    for element in elements:
        name = os.path.basename(os.fsdecode(element))
        if name in AGENT_CLI_BINARIES:
            return name
    return None
