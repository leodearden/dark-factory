"""Integration tests: validate that claude --resume works across OAuth accounts.

These tests invoke the real Claude CLI with haiku to minimize cost (~$0.002/call).
They require at least one OAuth token in env; cross-account tests need two.

Run explicitly:  uv run pytest tests/test_cli_invoke_integration.py -xvs -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shared.cli_invoke import invoke_claude_agent

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

# Shared invocation kwargs to minimize cost
_INVOKE_DEFAULTS = dict(
    system_prompt='You are a helpful assistant. Be very brief.',
    cwd=Path('/tmp'),
    model='haiku',
    max_turns=1,
    max_budget_usd=0.01,
    allowed_tools=[],
    effort='low',
)


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
        assert r1.session_id

        # Resume and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
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
        assert r1.session_id

        # Resume on account B and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token_b,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )
        assert 'ZEPPELIN' in r2.output.upper(), (
            f'Expected ZEPPELIN in cross-account resumed output, got: {r2.output!r}'
        )
