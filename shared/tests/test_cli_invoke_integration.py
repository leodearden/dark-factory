"""Integration tests: validate that claude --resume works across OAuth accounts.

These tests invoke the real Claude CLI with haiku to minimize cost (~$0.002/call).
They require at least one OAuth token in env; cross-account tests need two.

Run explicitly:  uv run pytest tests/test_cli_invoke_integration.py -xvs -m integration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from shared.cli_invoke import AgentResult, invoke_claude_agent
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
# Capacity-failure detection helper
# ---------------------------------------------------------------------------

_CAPACITY_FAILURE_MARKERS: tuple[str, ...] = (
    'capped',
    'rate limit',
    'unavailable',
    'out of extra usage',
    'usage limit',
    "you've hit",
    "you've used",
)


def _looks_like_capacity_failure(result: AgentResult) -> bool:
    """Return True when *result* looks like a Claude CLI capacity / quota failure.

    The helper inspects both ``result.output`` and ``result.stderr``
    (case-insensitive substring match) against a small focused list of markers
    drawn from real Claude CLI cap messages (see ``shared.usage_gate`` inline
    comments for verbatim examples).

    **Conservative bias (fail loudly when uncertain).** This helper is used
    at ``pytest.skip`` call sites, so a false positive — skipping on a real
    regression — is the exact failure mode we are trying to prevent. The list
    is therefore intentionally small and obvious; anything not matching a
    well-known capacity signal falls through to an ``assert`` that fails the
    test loudly.

    **Local list, not imported from ``shared.usage_gate``.** The production
    cap detector (``usage_gate.detect_cap_hit``) requires BOTH a prefix AND a
    confirm-keyword match — a strict combined policy designed to avoid marking
    healthy accounts as capped. Re-using those lists here would either collapse
    the combined check to a loose OR (pulling in confirm keywords like
    ``"resets"`` as standalone signals) or miss real cap messages that arrive
    without the expected prefix. A purpose-built substring list is the correct
    shape for this use-case.
    """
    haystack = f'{result.output}\n{result.stderr}'.lower()
    return any(marker in haystack for marker in _CAPACITY_FAILURE_MARKERS)


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
        if not r1.success and _looks_like_capacity_failure(r1):
            pytest.skip(f'Capacity failure: {r1.output!r}')
        assert r1.success and r1.session_id

        # Resume and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
        if not r2.success and _looks_like_capacity_failure(r2):
            pytest.skip(f'Capacity failure: {r2.output!r}')
        assert r2.success
        assert 'FLAMINGO' in r2.output.upper(), (
            f'Expected FLAMINGO in resumed output, got: {r2.output!r}'
        )

    @_need_two_accounts
    async def test_session_resume_preserves_context_across_accounts(self):
        """Key test: start on account A, resume on account B, verify context preserved."""
        _name_a, token_a = _AVAILABLE_TOKENS[0]
        _name_b, token_b = _AVAILABLE_TOKENS[1]

        # Start session on account A with a codeword
        r1 = await invoke_claude_agent(
            prompt='Remember this codeword: ZEPPELIN. Just say OK.',
            oauth_token=token_a,
            **_INVOKE_DEFAULTS,
        )
        if not r1.success and _looks_like_capacity_failure(r1):
            pytest.skip(f'Capacity failure: {r1.output!r}')
        assert r1.success and r1.session_id

        # Resume on account B and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token_b,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
        if not r2.success and _looks_like_capacity_failure(r2):
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


class TestLooksLikeCapacityFailure:
    """Unit tests for _looks_like_capacity_failure helper.

    No @pytest.mark.integration marker so these run in normal CI.
    """

    @pytest.mark.parametrize('marker', [
        'capped',
        'rate limit',
        'unavailable',
        'out of extra usage',
        'usage limit',
        "you've hit",
        "you've used",
    ])
    def test_marker_in_output_returns_true(self, marker):
        result = AgentResult(success=False, output=f'Error: {marker} condition', stderr='')
        assert _looks_like_capacity_failure(result)

    @pytest.mark.parametrize('marker', [
        'capped',
        'rate limit',
        'unavailable',
        'out of extra usage',
        'usage limit',
        "you've hit",
        "you've used",
    ])
    def test_marker_in_stderr_returns_true(self, marker):
        result = AgentResult(success=False, output='', stderr=f'Error: {marker} condition')
        assert _looks_like_capacity_failure(result)

    @pytest.mark.parametrize('output,stderr', [
        ('RATE LIMIT exceeded', ''),
        ('Account is Capped', ''),
        ('', 'USAGE LIMIT reached'),
    ])
    def test_case_insensitive_returns_true(self, output, stderr):
        result = AgentResult(success=False, output=output, stderr=stderr)
        assert _looks_like_capacity_failure(result)

    def test_generic_failure_returns_false(self):
        result = AgentResult(success=False, output='process spawn failed: ENOENT', stderr='Traceback ...')
        assert not _looks_like_capacity_failure(result)

    def test_empty_result_returns_false(self):
        result = AgentResult(success=False, output='', stderr='')
        assert not _looks_like_capacity_failure(result)
