"""Integration tests: exercise ``claude --resume`` against the real CLI.

These tests invoke the real Claude CLI with haiku to minimize cost (~$0.002/call).
They require at least one OAuth token in env (`CLAUDE_OAUTH_TOKEN_[BCDEFG]`) for
the `_need_one_account` tests, and two for `_need_two_accounts`. A full run of
TestCrossAccountResume takes ~6 min of wall clock and costs real money.

**Aiming the cross-account measurement (task 3484).** By default the pair is the
first two available tokens in ``B,C,D,E,F,G`` order — but the accounts that are
UNCAPPED at any given moment are not the first two, and hard-coding them is why
this measurement has twice produced 0 valid runs (2026-08-01, task 3454; again
2026-08-05).  Two env vars, both read via ``_cross_account_evidence``:

- ``CROSS_ACCOUNT_RESUME_TOKENS='F,C'`` — measure on those accounts, in that
  order (first starts the session, second issues the ``--resume``).  Bare letters
  or full ``CLAUDE_OAUTH_TOKEN_*`` names both work.  A value naming an unset var
  does NOT fall back to the default pair: the cross-account test skips with the
  reason, because a silent fall-back is how a run gets aimed at a capped account
  without anyone noticing.
- ``CROSS_ACCOUNT_EVIDENCE_PATH=/tmp/evidence.jsonl`` — also append each run's
  evidence record there.  One JSON line per run is written to stdout regardless
  (visible under ``-s``, prefixed ``CROSS_ACCOUNT_RESUME_EVIDENCE``), including
  for runs that fail or skip.

Probe which accounts are healthy BEFORE spending, and pick the pair from it:

    for v in B C D E F G; do tok_var="CLAUDE_OAUTH_TOKEN_$v"; tok="${!tok_var}"; \\
      out=$(cd /tmp && CLAUDE_CODE_OAUTH_TOKEN="$tok" timeout 60 claude \\
        -p "Say exactly: PONG" --model haiku --max-turns 1 2>&1 | head -c 200); \\
      echo "TOKEN_$v: $out"; done

An account is healthy iff it answers PONG; any "You've hit your weekly/session
limit" text means capped.  A cross-account resume needs TWO simultaneously
healthy accounts, and a probe older than ~15 minutes is stale.

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
import sys
from pathlib import Path
from typing import Any

import pytest
from _capacity_skip import result_looks_like_capacity_failure
from _cross_account_evidence import (
    available_tokens,
    emit_run_evidence,
    format_run_evidence,
    select_token_pair,
)

from shared.cli_invoke import (
    invoke_claude_agent,
    read_transcript_records,
    transcript_exists,
)
from shared.config_dir import TaskConfigDir

# Discover available OAuth tokens from env (scan order B,C,D,E,F,G).
_AVAILABLE_TOKENS: list[tuple[str, str]] = available_tokens(os.environ)

# The measured pair, steerable via CROSS_ACCOUNT_RESUME_TOKENS.  Resolved
# DEFENSIVELY: any resolution failure (too few tokens, an override naming an
# unset var) leaves the pair None and is turned into a skip below, rather than
# raising here — a ValueError at module scope would break collection of every
# test in this file, including the ones that need only one account.
_ACCOUNT_PAIR: tuple[tuple[str, str], tuple[str, str]] | None
_PAIR_ERROR: str | None
try:
    _ACCOUNT_PAIR = select_token_pair(os.environ)
    _PAIR_ERROR = None
except ValueError as exc:
    _ACCOUNT_PAIR = None
    _PAIR_ERROR = str(exc)

# Account A — the session-STARTING account, used by the single-account tests too
# so the control and the cross-account r1 run on the same account.  Falls back to
# the first available token when no pair could be resolved, so a one-token
# environment (or a mis-set override) still exercises the single-account tests.
_ACCOUNT_A: tuple[str, str] | None = (
    _ACCOUNT_PAIR[0] if _ACCOUNT_PAIR else (_AVAILABLE_TOKENS[0] if _AVAILABLE_TOKENS else None)
)

_need_one_account = pytest.mark.skipif(
    _ACCOUNT_A is None,
    reason='Requires at least 1 OAuth account in env',
)
_need_two_accounts = pytest.mark.skipif(
    _ACCOUNT_PAIR is None,
    reason=(
        'Requires 2 resolvable OAuth accounts in env'
        + (f' — {_PAIR_ERROR}' if _PAIR_ERROR else '')
    ),
)

# Did the same-account control pass in THIS process?  None = it did not run (or
# skipped on a capacity failure), which is not the same as failing.  pytest runs
# tests in definition order within a class, so the control below runs before the
# cross-account test and its result is available to the evidence record; a
# targeted ``-k`` run that skips the control simply records None.
_CONTROL_PASSED: bool | None = None


def _effective_config_dir() -> Path:
    """The config dir the CLI actually writes transcripts under, for these tests.

    They pass no ``config_dir``, so ``invoke_claude_agent`` copies ``os.environ``
    and the CLI honours the ambient ``CLAUDE_CONFIG_DIR``; the transcript lands
    THERE, not necessarily under ``~/.claude``.  Globbing the wrong directory
    would report a transcript as absent and wrongly implicate
    transcript-reachability in a failed resume.
    """
    ambient = os.environ.get('CLAUDE_CONFIG_DIR')
    return Path(ambient) if ambient else Path.home() / '.claude'

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
        assert _ACCOUNT_A is not None  # guaranteed by _need_one_account
        _name, token = _ACCOUNT_A
        result = await invoke_claude_agent(
            prompt='Say exactly: PONG',
            oauth_token=token,
            **_INVOKE_DEFAULTS,
        )
        assert result.session_id, f'Expected session_id, got: {result.session_id!r}'
        assert result.success

    @_need_one_account
    async def test_session_resume_same_account_baseline(self):
        """Control: resume on the same account recalls prior context.

        Runs on account A — the same account the cross-account test starts its
        session on — and records its outcome into ``_CONTROL_PASSED`` for that
        test's evidence record.  A FAILING control means the harness, not the
        account, is the variable, and the cross-account result is uninterpretable.
        """
        global _CONTROL_PASSED
        assert _ACCOUNT_A is not None  # guaranteed by _need_one_account
        _name, token = _ACCOUNT_A

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
        # Stamp BEFORE asserting, so a red control is recorded as False rather
        # than left at None (unknown) by the raised AssertionError.
        recalled = 'FLAMINGO' in r2.output.upper()
        _CONTROL_PASSED = bool(r2.success and recalled)
        assert r2.success
        assert recalled, (
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

        READING A RED RUN.  A capped account now SKIPS: the guard is
        ``_capacity_skip.result_looks_like_capacity_failure``, pinned (task
        3483) against the verbatim weekly text observed here — "You've hit
        your weekly limit · resets Aug 5, 11am" — which the guard previously
        did not match, so a capped account used to fail the assertion below
        with a cap message dressed up as lost context.  The remaining reason
        to read a red run carefully is a cap message the guard has never seen:
        check the reported output for a limit message before concluding
        regression, and if it is one, add it to ``REAL_CLI_CAP_MESSAGES``.
        """
        assert _ACCOUNT_PAIR is not None  # guaranteed by _need_two_accounts
        (name_a, token_a), (name_b, token_b) = _ACCOUNT_PAIR

        # Start session on account A with a codeword
        r1 = await invoke_claude_agent(
            prompt='Remember this codeword: ZEPPELIN. Just say OK.',
            oauth_token=token_a,
            **_INVOKE_DEFAULTS,
        )
        if not r1.success and result_looks_like_capacity_failure(r1):
            pytest.skip(f'Capacity failure on {name_a} (r1): {r1.output!r}')
        assert r1.success and r1.session_id

        # Probe r1's transcript on disk BEFORE the resume: "was it reachable"
        # is the alternative explanation for a failed resume, and it can only be
        # ruled in or out from the state that existed when r2 was issued.
        config_dir = _effective_config_dir()
        r1_transcript_present = transcript_exists(config_dir, r1.session_id)
        _r1_records = read_transcript_records(config_dir, r1.session_id)
        # None (not 0) when unreadable/absent — read_transcript_records returns
        # None on both, so pair it with the presence check above rather than
        # inferring absence from a missing count.
        r1_transcript_records = len(_r1_records) if _r1_records is not None else None

        # Resume on account B and ask for the codeword
        r2 = await invoke_claude_agent(
            prompt='What was the codeword I told you? Reply with just the word.',
            oauth_token=token_b,
            resume_session_id=r1.session_id,
            **_INVOKE_DEFAULTS,
        )

        # Emit the evidence BEFORE the cap-skip and the assertion below.  A red
        # or void run is exactly when the record matters most: the 2026-08-01
        # attempt was only diagnosable as void because someone read r2's turn
        # verbatim and recognised a weekly-limit message in it.
        record = format_run_evidence(
            account_a=name_a,
            account_b=name_b,
            r1=r1,
            r2=r2,
            r1_transcript_present=r1_transcript_present,
            r1_transcript_records=r1_transcript_records,
            control_passed=_CONTROL_PASSED,
        )
        evidence = emit_run_evidence(record, os.environ, sys.stdout)

        if not r2.success and result_looks_like_capacity_failure(r2):
            pytest.skip(
                f'Capacity failure on {name_b} (r2) — run is VOID for the '
                f'context question, not evidence of context loss: {evidence}'
            )
        assert r2.success, f'r2 failed: {evidence}'
        assert 'ZEPPELIN' in r2.output.upper(), (
            f'Expected ZEPPELIN in cross-account resumed output. {evidence}'
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
        assert _ACCOUNT_A is not None  # guaranteed by _need_one_account
        _name, token = _ACCOUNT_A
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
        assert _ACCOUNT_A is not None  # guaranteed by _need_one_account
        _name, token = _ACCOUNT_A
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
