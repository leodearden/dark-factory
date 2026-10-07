"""Every orchestrator-dispatched Claude session resets its Bash cwd after each call.

The finding is task 5971, a cross-repo refile of reify legibility candidate
#7924 and the same mechanism as dark_factory codebook entry
entry-cand-20260729-4: in a dispatched session the Bash cwd persists across
calls, so a repo-root-relative path used after an earlier `cd` fails as
not-found.

The remedy under test is the Claude CLI env var
CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR, defaulted into every dispatched
Claude session. Per the CLI's post-Bash-call check, it makes the CLI silently
return the shell to its launch directory after every Bash call.

The observation is the env the real child process saw: a fake `claude` on
PATH records the var, driven through the public ``invoke_agent``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.invoke import invoke_agent

#: Spelled as a literal, never imported: the name is the Claude CLI's external
#: contract, and importing the production constant would let a typo pass.
_RESET_VAR = 'CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR'

_UNSET = '__unset__'

_SUCCESS_JSON = json.dumps({
    'result': 'ok',
    'is_error': False,
    'subtype': 'success',
    'cost_usd': 0.0,
    'duration_ms': 1,
    'num_turns': 1,
    'session_id': 's',
})


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], str]:
    """Put a `claude` on PATH that records the reset var's value; return its reader."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    dump = tmp_path / 'env-dump'
    script = bin_dir / 'claude'
    script.write_text(
        '#!/bin/sh\n'
        'cat >/dev/null\n'
        f'printf %s "${{{_RESET_VAR}-{_UNSET}}}" > "$FAKE_CLAUDE_ENV_DUMP"\n'
        f"printf '%s\\n' '{_SUCCESS_JSON}'\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ.get("PATH", "")}')
    monkeypatch.setenv('FAKE_CLAUDE_ENV_DUMP', str(dump))
    for var in (_RESET_VAR, 'DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)

    return dump.read_text


@contextmanager
def _bwrap_identity_sandbox() -> Iterator[None]:
    """Reach the sandboxed branch through public sandbox_dispatch names only."""
    with (
        patch('orchestrator.agents.sandbox_dispatch.resolve_active_backend', return_value='bwrap'),
        patch(
            'orchestrator.agents.sandbox_dispatch.wrap_command',
            side_effect=lambda cmd, *args, **kwargs: cmd,
        ),
    ):
        yield


_DISPATCH_PATHS = pytest.mark.parametrize(
    ('sandbox_modules', 'dispatch_context'),
    [
        pytest.param(None, nullcontext, id='unsandboxed'),
        pytest.param(['shared'], _bwrap_identity_sandbox, id='sandboxed'),
    ],
)


async def _dispatch(
    workdir: Path,
    sandbox_modules: list[str] | None,
    dispatch_context: Callable[[], AbstractContextManager[None]],
    env_overrides: dict[str, str] | None,
) -> None:
    workdir.mkdir(exist_ok=True)
    with dispatch_context():
        result = await invoke_agent(
            prompt='hi',
            system_prompt='s',
            cwd=workdir,
            model='opus',
            max_turns=1,
            max_budget_usd=0.01,
            sandbox_modules=sandbox_modules,
            env_overrides=env_overrides,
        )
    assert result.success, f'fake claude dispatch did not succeed: {result!r}'


@pytest.mark.asyncio
@_DISPATCH_PATHS
async def test_dispatched_claude_session_resets_bash_cwd_by_default(
    fake_claude, tmp_path, sandbox_modules, dispatch_context,
):
    await _dispatch(tmp_path / 'work', sandbox_modules, dispatch_context, env_overrides=None)

    assert fake_claude() == '1', (
        f'the dispatched claude child must see {_RESET_VAR}=1 so the CLI puts its '
        'Bash shell back at the dispatch root after every call. Default it at the '
        'single env seam in orchestrator/src/orchestrator/agents/invoke.py::'
        '_invoke_claude_with_sandbox, before the sandbox branch, so both sub-paths carry it.'
    )
    assert _RESET_VAR not in os.environ, (
        'the default belongs to the child env only; it must not leak into the '
        "orchestrator's own process environment."
    )


@pytest.mark.asyncio
@_DISPATCH_PATHS
async def test_explicit_env_override_wins(
    fake_claude, tmp_path, sandbox_modules, dispatch_context,
):
    overrides = {_RESET_VAR: '0'}

    await _dispatch(tmp_path / 'work', sandbox_modules, dispatch_context, env_overrides=overrides)

    assert fake_claude() == '0', (
        f'an explicit {_RESET_VAR} from the caller (e.g. config.role_env_overrides, '
        'the operator opt-out) must win over the default: use setdefault, not assignment.'
    )
    assert overrides == {_RESET_VAR: '0'}, (
        "the caller's env_overrides dict must not be mutated; transform a copy."
    )
