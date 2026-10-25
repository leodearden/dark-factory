"""No orchestrator test may start an agent-CLI process (task 4701).

A stale monkeypatch target degrades to calling the real CLI instead of
erroring (the task-882 incident), so the guard sits at
``subprocess.Popen.__init__``: the one spawn chokepoint no name binding can
bypass.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

AGENT_CLI_BINARIES: frozenset[str] = frozenset({'claude', 'codex', 'gemini', 'pi'})


_REMEDY = (
    "Patch invoke_agent at the CONSUMING module's import site, e.g. "
    'orchestrator.workflow.invoke_agent, not orchestrator.agents.invoke.invoke_agent: '
    'a stale target silently calls the real CLI. Mark the test '
    '@pytest.mark.spawns_agent_cli only if it deliberately spawns an agent-CLI '
    'process, e.g. a fake `claude` script on PATH.'
)
_ABSORBED = (
    ' A caller (a background task or a broad except) may have absorbed the raise '
    'at the spawn seam.'
)


def _argv(args: Any) -> tuple[str, ...]:
    elements = [args] if isinstance(args, str | bytes | os.PathLike) else args
    return tuple(os.fsdecode(element) for element in elements)


def agent_cli_in(args: Any) -> str | None:
    """The agent-CLI binary named by any element of ``args``, else None."""
    for element in _argv(args):
        name = os.path.basename(element)
        if name in AGENT_CLI_BINARIES:
            return name
    return None


def _spawn_message(nodeid: str, attempts: list[tuple[str, ...]], note: str = '') -> str:
    spawned = '; '.join(f'{agent_cli_in(argv)!r} via {argv!r}' for argv in attempts)
    return (
        f'orchestrator tests must not spawn an agent CLI, but {nodeid} spawned '
        f'{spawned}.{note} {_REMEDY}'
    )


class AgentCliSpawnGuard:
    """One test's agent-CLI spawn attempts, blocked at ``Popen.__init__``."""

    def __init__(self, nodeid: str) -> None:
        self._nodeid = nodeid
        self._attempts: list[tuple[str, ...]] = []

    def guarded(self, real_init: Callable[..., None]) -> Callable[..., None]:
        def init(popen_self: Any, args: Any, *a: Any, **kw: Any) -> None:
            if isinstance(args, Iterator):
                args = list(args)
            if agent_cli_in(args) is None:
                return real_init(popen_self, args, *a, **kw)
            argv = _argv(args)
            self._attempts.append(argv)
            pytest.fail(_spawn_message(self._nodeid, [argv]))

        return init

    def take_attempts(self) -> list[tuple[str, ...]]:
        attempts, self._attempts = self._attempts, []
        return attempts

    def violation_message(self) -> str | None:
        if not self._attempts:
            return None
        return _spawn_message(self._nodeid, self._attempts, _ABSORBED)
