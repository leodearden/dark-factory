"""Test doubles for the ``shared.cli_invoke.CliRunner`` spawn seam.

    pinned = pin_claude_binary(monkeypatch, tmp_path)
    runner = RecordingRunner(canned_claude_result())
    await invoke_claude_agent(..., runner=runner)
    [spawn] = runner.calls

``RecordingRunner`` stands in for ``run_cli_subprocess``: it spawns nothing,
records each call and answers the same result every time.
``canned_claude_result`` is a successful ``claude --output-format json``
answer, and ``pin_claude_binary`` points ``CLAUDE_BINARY`` at a do-nothing
executable so ``build_claude_argv`` resolves argv[0] to a known path whatever
the host's PATH holds.

It lives in the shipped ``shared`` package, beside
:mod:`shared.testing_fake_claude`, because ``shared.tests`` is not importable
from the orchestrator pytest rootdir and both suites drive the seam. Like
:mod:`shared.testing_streams` it does not import pytest at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.cli_invoke import SubprocessResult

if TYPE_CHECKING:
    import pytest

__all__ = [
    'CANNED_CLAUDE_OUTPUT',
    'CANNED_CLAUDE_SESSION',
    'RecordedSpawn',
    'RecordingRunner',
    'canned_claude_result',
    'pin_claude_binary',
]

CANNED_CLAUDE_OUTPUT = 'canned-output'
CANNED_CLAUDE_SESSION = 'canned-session'


@dataclass(frozen=True)
class RecordedSpawn:
    """One call a ``RecordingRunner`` received."""

    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    kw: dict[str, Any]

    def wrapped_cmd(self) -> list[str]:
        """The argv after the call's ``wrap_argv``, as the real runner would apply it."""
        wrap_argv = self.kw.get('wrap_argv')
        return wrap_argv(self.cmd) if wrap_argv is not None else list(self.cmd)


class RecordingRunner:
    """A ``CliRunner`` that records every call and answers *result* to each."""

    def __init__(self, result: SubprocessResult) -> None:
        self.result = result
        self.calls: list[RecordedSpawn] = []

    async def __call__(
        self, cmd: list[str], cwd: Path, env: dict[str, str], **kw: Any,
    ) -> SubprocessResult:
        self.calls.append(RecordedSpawn(cmd=list(cmd), cwd=cwd, env=dict(env), kw=kw))
        return self.result


def canned_claude_result(**payload: Any) -> SubprocessResult:
    """A clean claude exit whose JSON envelope is the defaults updated by *payload*."""
    envelope = {
        'result': CANNED_CLAUDE_OUTPUT,
        'is_error': False,
        'subtype': 'success',
        'session_id': CANNED_CLAUDE_SESSION,
        'num_turns': 1,
        'duration_ms': 100,
        **payload,
    }
    return SubprocessResult(stdout=json.dumps(envelope), stderr='', returncode=0, duration_ms=100)


def pin_claude_binary(monkeypatch: pytest.MonkeyPatch, root: Path) -> str:
    """Write a do-nothing ``claude`` under *root*, pin ``CLAUDE_BINARY`` at it, return its path."""
    executable = root / 'claude'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BINARY', str(executable))
    return str(executable)
