"""Integration tests: exercise ``claude --resume`` against the real CLI.

These tests invoke the real Claude CLI with haiku to minimize cost (~$0.002/call).
They require at least one OAuth token in env (`CLAUDE_OAUTH_TOKEN_[BCDEF]`) for
the `_need_one_account` tests, and two for `_need_two_accounts`. A full run of
TestCrossAccountResume takes ~6 min of wall clock and costs real money.

**Mechanism.** A Claude CLI session is a LOCAL JSONL transcript at
``<config_dir>/projects/<cwd-slug>/<session_id>.jsonl`` — not a server-side,
account-scoped object — and ``--resume`` replays that local file.  So what
governs whether a resume keeps its context is transcript REACHABILITY (same
config dir, same cwd), not which OAuth account issues the call.  These tests
pass no ``config_dir``, so the CLI inherits the ambient ``CLAUDE_CONFIG_DIR``
(``invoke_claude_agent`` copies ``os.environ``); the transcript lands there,
NOT necessarily under ``~/.claude``.  Production's reachability guard lives in
``shared.cli_invoke.invoke_with_cap_retry`` — see the measurement comment above
its cap-hit resume branch, which is the single source of truth for what has and
has not been established about cross-account resume.

**DESELECTED BY DEFAULT, and the ``-m integration`` marker is mandatory to run
them.** The marker is registered + deselected (``addopts = "-m 'not
integration'"``) in BOTH ``shared/pyproject.toml`` and (as of task 3444) the
ROOT ``pyproject.toml``. pytest reads only ONE [tool.pytest.ini_options] -- the
rootdir's inifile -- and never merges the two, so mirroring at the root is what
keeps a repo-root-bound run (a bare ``pytest`` from the repo root, ``-c
pyproject.toml``, or any arg set spanning two subprojects) from spending live
CLI budget here. No ordinary run, from any directory, collects these; without
``-m integration`` every test in this module is silently deselected and the run
looks green while having executed nothing.

Run explicitly, from the repo root:
    uv run --project shared --directory shared \\
        pytest tests/test_cli_invoke_integration.py -xvs -m integration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from _capacity_skip import result_looks_like_capacity_failure

from shared.cli_invoke import invoke_claude_agent
from shared.config_dir import TaskConfigDir

# Discover available OAuth tokens from env
_TOKEN_ENV_VARS = [f'CLAUDE_OAUTH_TOKEN_{c}' for c in 'BCDEF']
_AVAILABLE_TOKENS: list[tuple[str, str]] = [
    (var, os.environ[var])
    for var in _TOKEN_ENV_VARS
    if os.environ.get(var)
]

_need_one_account = pytest.mark.skipif(
    len(_AVAILABLE_TOKENS) < 1,
    reason='Requires at least 1 OAuth account in env',
)
_need_two_accounts = pytest.mark.skipif(
    len(_AVAILABLE_TOKENS) < 2,
    reason='Requires at least 2 OAuth accounts in env',
)

# ---------------------------------------------------------------------------
# Shared invocation kwargs to minimize cost.
# dict[str, Any] is intentional: invoke_claude_agent parameters have
# heterogeneous types (Path/str/int/float/list), so a concrete dict type
# would lose per-parameter type checking at the ** call site.
_INVOKE_DEFAULTS: dict[str, Any] = {
    'system_prompt': 'You are a helpful assistant. Be very brief.',
    'cwd': Path('/tmp'),
    'model': 'haiku',
    'max_turns': 1,
    'max_budget_usd': 0.01,
    'allowed_tools': [],
    'effort': 'low',
}


@pytest.mark.integration
@pytest.mark.asyncio
class TestCrossAccountResume:

    @_need_one_account
    async def test_invoke_returns_session_id(self):
        """Baseline: a normal invocation returns a non-empty session_id."""
        _name, token = _AVAILABLE_TOKENS[0]
        result = await invoke_claude_agent(
            prompt='Say exactly: PONG',
            oauth_token=token,
            **_INVOKE_DEFAULTS,
        )
        assert result.session_id, f'Expected session_id, got: {result.session_id!r}'
        assert result.success

    @_need_one_account
    async def test_session_resume_same_account_baseline(self):
        """Control: resume on the same account recalls prior context."""
        _name, token = _AVAILABLE_TOKENS[0]

        # Start session with a codeword
        r1 = await invoke_claude_agent(
            prompt='Remember this codeword: FLAMINGO. Just say OK.',
            oauth_token=token,
            **_INVOKE_DEFAULTS,
        )
        if not r1.success and result_looks_like_capacity_failure(r1):
            pytest.skip(f'Capacity failure: {r1.output!r}')
        assert r1.success and r1.session_id

        # Resume and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
        if not r2.success and result_looks_like_capacity_failure(r2):
            pytest.skip(f'Capacity failure: {r2.output!r}')
        assert r2.success
        assert 'FLAMINGO' in r2.output.upper(), (
            f'Expected FLAMINGO in resumed output, got: {r2.output!r}'
        )

    @_need_two_accounts
    async def test_session_resume_preserves_context_across_accounts(self):
        """Probe: does a resume issued on account B recall context started on A?

        The name of the thing being probed is deliberately a QUESTION, not a
        claim.  As of 2026-08-01 (claude CLI 2.1.220, task 3454) this has NOT
        been answered: 4 of the 5 accounts in env were capped, and the probe
        needs two simultaneously-uncapped accounts, so there were 0 valid runs.
        What IS established is that the transcript-reachability explanation is
        ruled out — the r1 transcript was on disk both times, and a resume on a
        different account appended to that same local file — so a failure here
        would NOT be explained by an unreachable session.  Full measurement:
        the comment above the cap-hit resume branch in ``shared.cli_invoke``.

        The ZEPPELIN assertion is retained deliberately and must not be weakened:
        it is the only signal that would distinguish outcome (a) from (b).

        KNOWN MISLEADING FAILURE.  When account B is capped, the CLI answers the
        resumed turn with e.g. "You've hit your weekly limit · resets Aug 5,
        11am".  ``_looks_like_capacity_failure`` does NOT match that text (its
        marker is "you've hit your usage"), so this test does not skip — it
        fails the assertion below with a cap message as the "output", which is
        indistinguishable at a glance from genuine context loss.  Check the
        reported output for a limit message before reading a red run as a
        regression.  (The marker list is intentionally left alone here; it is
        owned by the task that narrowed it.)
        """
        _name_a, token_a = _AVAILABLE_TOKENS[0]
        _name_b, token_b = _AVAILABLE_TOKENS[1]

        # Start session on account A with a codeword
        r1 = await invoke_claude_agent(
            prompt='Remember this codeword: ZEPPELIN. Just say OK.',
            oauth_token=token_a,
            **_INVOKE_DEFAULTS,
        )
        if not r1.success and result_looks_like_capacity_failure(r1):
            pytest.skip(f'Capacity failure: {r1.output!r}')
        assert r1.success and r1.session_id

        # Resume on account B and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token_b,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
        if not r2.success and result_looks_like_capacity_failure(r2):
            pytest.skip(f'Capacity failure: {r2.output!r}')
        assert r2.success
        assert 'ZEPPELIN' in r2.output.upper(), (
            f'Expected ZEPPELIN in cross-account resumed output, got: {r2.output!r}'
        )


@pytest.mark.integration
@pytest.mark.asyncio
class TestConfigDirCredentials:
    """Test credential passing via env var with and without TaskConfigDir.

    These tests MUST use real agents because they validate the actual Claude
    CLI's OAuth token handling.  Stale or invalid tokens produce "Invalid API
    key" or "You're out of extra usage" — errors that only manifest with the
    real CLI, not mocks.
    """

    @_need_one_account
    async def test_env_var_auth_succeeds(self):
        """OAuth token via CLAUDE_CODE_OAUTH_TOKEN env var authenticates."""
        _name, token = _AVAILABLE_TOKENS[0]
        result = await invoke_claude_agent(
            prompt='Say exactly: PONG',
            oauth_token=token,
            **_INVOKE_DEFAULTS,
        )
        # Budget may be exceeded (cost > $0.01) but the CLI must authenticate
        assert 'invalid api key' not in result.output.lower(), (
            f'Token rejected as invalid: {result.output!r}'
        )
        assert 'not logged in' not in result.output.lower(), (
            f'Token not recognized: {result.output!r}'
        )

    @_need_one_account
    async def test_config_dir_plus_env_var_auth_succeeds(self):
        """Auth with both config dir and env var — the orchestrator pattern."""
        _name, token = _AVAILABLE_TOKENS[0]
        config_dir = TaskConfigDir('test-config-dir-both')
        try:
            config_dir.write_credentials(token)
            result = await invoke_claude_agent(
                prompt='Say exactly: PONG',
                oauth_token=token,
                config_dir=config_dir.path,
                **_INVOKE_DEFAULTS,
            )
            assert 'invalid api key' not in result.output.lower(), (
                f'Token rejected as invalid: {result.output!r}'
            )
        finally:
            config_dir.cleanup()
