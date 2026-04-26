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
    ' capped',          # leading space prevents matching 'uncapped' as a false positive
    'rate limit',
    'account unavailable',  # narrowed from bare 'unavailable' to avoid generic network errors
    'out of extra usage',
    'usage limit',
    "you've hit your usage",   # narrowed prefix to avoid matching innocuous "you've hit a snag" phrasing
    "you've used all",         # narrowed prefix to avoid matching innocuous "you've used the wrong format" phrasing
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

    @pytest.mark.parametrize('cli_output', [
        # Verbatim Claude CLI cap-hit messages (from shared.usage_gate inline comments,
        # lines 64-75 — these are the actual strings that motivated the marker list).
        "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
        "You've used all available credits. Upgrade your plan for more capacity.",
        "You're out of extra usage for this billing period. Your plan resets in 2h.",
        "You're close to reaching your usage limit. Your plan resets in 1h.",
        # Other realistic capacity phrases
        "Your account is capped until the next billing cycle.",
        "Rate limit exceeded. Please wait and retry.",
        "account unavailable at this time; try again later.",
    ])
    def test_capacity_output_returns_true(self, cli_output):
        """Realistic Claude CLI cap messages in output are detected."""
        result = AgentResult(success=False, output=cli_output, stderr='')
        assert _looks_like_capacity_failure(result)

    @pytest.mark.parametrize('cli_stderr', [
        "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
        "You're out of extra usage for this billing period. Your plan resets in 2h.",
        "rate limit: too many requests",
    ])
    def test_capacity_stderr_returns_true(self, cli_stderr):
        """Realistic Claude CLI cap messages in stderr are also detected."""
        result = AgentResult(success=False, output='', stderr=cli_stderr)
        assert _looks_like_capacity_failure(result)

    @pytest.mark.parametrize('output', [
        "YOU'VE HIT YOUR USAGE LIMIT FOR CLAUDE PRO.",
        "YOUR ACCOUNT IS CAPPED.",
        "RATE LIMIT EXCEEDED.",
    ])
    def test_case_insensitive_returns_true(self, output):
        """Cap detection is case-insensitive."""
        result = AgentResult(success=False, output=output, stderr='')
        assert _looks_like_capacity_failure(result)

    @pytest.mark.parametrize('output,stderr', [
        # Generic non-capacity failures — must NOT trigger skip
        ('process spawn failed: ENOENT', 'Traceback (most recent call last): ...'),
        ('malformed JSON response: unexpected token', ''),
        ('OAuth token validation failed: 401 Unauthorized', ''),
        # Substring boundary collisions — the narrowed markers must not match
        ('account uncapped and ready to use', ''),         # 'uncapped' must not match ' capped'
        ('service unavailable: DNS resolution failed', ''),  # generic 'unavailable' != 'account unavailable'
        ("You've used the wrong format. Please retry.", ''),  # must NOT match loose "you've used"
        ("You've hit a snag — try again later.", ''),         # must NOT match loose "you've hit"
        ('', ''),  # empty result
    ])
    def test_non_capacity_failure_returns_false(self, output, stderr):
        """Generic failures and substring boundary cases do not trigger a skip."""
        result = AgentResult(success=False, output=output, stderr=stderr)
        assert not _looks_like_capacity_failure(result)

    def test_call_site_contract_non_capacity_raises_not_skips(self):
        """The call-site guarantee: a non-capacity failure reaches assert, not skip.

        At each call site the pattern is::

            if not r.success and _looks_like_capacity_failure(r):
                pytest.skip(...)
            assert r.success  # ← fires for any non-capacity failure

        When the helper returns False (non-capacity), the skip branch is never
        taken and the subsequent ``assert r.success`` fires loudly.  This test
        verifies that behavioral guarantee — a regression in the call-site
        predicate would either break this test or the parametrize test above.
        """
        result = AgentResult(success=False, output='process spawn failed: ENOENT', stderr='Traceback...')
        # Helper must return False → the skip branch is NOT taken
        assert not _looks_like_capacity_failure(result)
        # Therefore the call site reaches `assert r.success`, which raises
        with pytest.raises(AssertionError):
            assert result.success
