"""RED/GREEN tests for CodebaseVerifier.verify() agent-failure handling (site 22).

Step-1 (RED): Failure-path tests confirm that when AgentLoop.run() returns a
warning/no-verdict shape, verify() emits a WARNING and sets
summary='agent-failed:<token>' rather than silently laundering the failure as
an empty-summary inconclusive.

Step-2 (GREEN): Implement the fix in verify.py (extract_agent_verdict).
The success-path test guards behavior-preservation across the refactor.

Task 4343 extends site-22's task-1811 work with STRUCTURED fields: the
``agent-failed:<token>`` sentinel lives in the human-facing ``summary`` prose,
so any consumer wanting the fact has to string-parse it — which design
invariant INV-2 ``structured-facts-at-failure`` forbids.  ``VerificationResult``
now also carries ``agent_failed: bool`` and ``failure_token: str``, and the
assertions below pin them on every path (both failure shapes, the non-dict
shape, and the success path where both must stay at their defaults).

Amendment pass: the two fields encode one fact twice, so
``TestFailureFieldsCannotDesync`` pins the model validator that keeps them in
lockstep, and ``test_verify_failure_token_coerces_malformed_origin`` pins that a
malformed ``warning_origin`` degrades through the preference chain rather than
raising ValidationError out of ``verify()``.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from _git_root_helper import make_git_root
from pydantic import ValidationError

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import VerificationResult, VerificationVerdict
from fused_memory.reconciliation.verify import CodebaseVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> ReconciliationConfig:
    """ReconciliationConfig with all defaults — suffices for unit tests."""
    return ReconciliationConfig()


@pytest.fixture
def git_root(tmp_path):
    """A usable per-call codebase root for verify() (task 4722).

    ``verify()`` now takes ``codebase_root`` as a required keyword and refuses
    a root that does not look like a checkout, so every call below needs a
    real one.  The assertions in this file are unchanged — it remains task
    4343's failure-token census guard; only the call shape moved.

    The shape itself comes from the shared ``make_git_root`` helper so this
    file cannot drift from the other two that build the same thing.
    """
    return make_git_root(tmp_path)


# ---------------------------------------------------------------------------
# FAILURE-PATH tests (RED in step-1, GREEN after step-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_return',
    [
        ({'warning': 'no_tool_calls', 'text': 'gave up'}, []),
        ({'warning': 'max_steps_reached'}, []),
    ],
    ids=['no_tool_calls', 'max_steps_reached'],
)
async def test_verify_agent_failure_emits_warning_and_sentinel_summary(
    agent_return, caplog, git_root
):
    """When AgentLoop.run() returns a warning shape, verify() must:

    - return verdict='inconclusive' (VerificationVerdict constraint)
    - return summary='agent-failed:<token>' (NOT empty)
    - return confidence=0.0, evidence=[], git_context=None
    - emit at least one WARNING record containing the failure token
    - (task 4343) set agent_failed=True and failure_token=<token> as
      STRUCTURED fields, so a downstream consumer never has to parse the
      ``agent-failed:`` prefix back out of the summary prose
    """
    result_dict, journal = agent_return
    expected_token = result_dict['warning']

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(result_dict, journal))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify(claim='Task X completed', codebase_root=git_root)

    assert result.verdict == 'inconclusive', (
        f'Expected inconclusive but got {result.verdict!r}'
    )
    assert result.summary == f'agent-failed:{expected_token}', (
        f'Expected agent-failed sentinel but got {result.summary!r}'
    )
    assert result.confidence == 0.0
    assert result.evidence == []
    assert result.git_context is None

    # Task 4343: the failure must also be legible WITHOUT parsing the summary.
    assert result.agent_failed is True, (
        f'Expected agent_failed=True but got {result.agent_failed!r}'
    )
    assert result.failure_token == expected_token, (
        f'Expected failure_token={expected_token!r} but got {result.failure_token!r}'
    )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, 'Expected at least one WARNING record, got none'
    assert any(expected_token in r.message for r in warning_records), (
        f'Expected a WARNING message containing {expected_token!r}, '
        f'got: {[r.message for r in warning_records]}'
    )


# ---------------------------------------------------------------------------
# NON-DICT failure-path test (guards the `verdict.raw or {}` fallback in
# verify.py against future refactors — extract_agent_verdict has its own
# unit coverage for this, but the verify()-level round-trip is distinct).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_agent_failure_non_dict_uses_error_summary(caplog, git_root):
    """When AgentLoop.run() returns a non-dict result (e.g. None),
    extract_agent_verdict falls back to error_summary='verify_failed' as the
    token.  verify() must:

    - return verdict='inconclusive' (VerificationVerdict constraint)
    - return summary='agent-failed:verify_failed' (error_summary token)
    - return confidence=0.0, evidence=[], git_context=None
      (all from ``verdict.raw or {}`` which resolves to ``{}`` when raw is None)
    - emit at least one WARNING record containing 'verify_failed'
    - (task 4343) set agent_failed=True and failure_token='verify_failed' —
      the structured fields must survive the raw-is-None path too, where
      there is no ``warning`` key to read a token from
    """
    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(None, []))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify(claim='Task X completed', codebase_root=git_root)

    assert result.verdict == 'inconclusive', (
        f'Expected inconclusive but got {result.verdict!r}'
    )
    assert result.summary == 'agent-failed:verify_failed', (
        f'Expected agent-failed:verify_failed but got {result.summary!r}'
    )
    assert result.confidence == 0.0
    assert result.evidence == []
    assert result.git_context is None

    # Task 4343: structured fields survive the raw-is-None path.
    assert result.agent_failed is True, (
        f'Expected agent_failed=True but got {result.agent_failed!r}'
    )
    assert result.failure_token == 'verify_failed', (
        f"Expected failure_token='verify_failed' but got {result.failure_token!r}"
    )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, 'Expected at least one WARNING record, got none'
    assert any('verify_failed' in r.message for r in warning_records), (
        f'Expected a WARNING message containing "verify_failed", '
        f'got: {[r.message for r in warning_records]}'
    )


# ---------------------------------------------------------------------------
# ORIGIN-PREFERENCE tests (task 4343): failure_token must carry the SPECIFIC
# CLI origin when AgentLoop.run() supplies one, and degrade to the generic
# token when it does not.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    'origin_token',
    ['cli_output_empty', 'cli_output_unparseable'],
)
async def test_verify_failure_token_prefers_cli_warning_origin(origin_token, git_root):
    """failure_token must prefer `warning_origin` over the generic `warning`.

    The shape below is exactly what AgentLoop.run() now emits for a CLI
    no-tool-call exit.  'cli_output_empty' (the CLI returned nothing) and
    'cli_output_unparseable' (the CLI returned junk) call for different
    operator responses, so the actionable one is what must reach the DB.

    The human-facing `summary` stays keyed to the GENERIC token: task 1811's
    'agent-failed:no_tool_calls' contract and its tests are untouched.  The
    specific diagnosis travels in the structured field instead of being
    smuggled into prose — INV-2 structured-facts-at-failure.
    """
    result_dict = {'warning': 'no_tool_calls', 'warning_origin': origin_token, 'text': ''}

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(result_dict, []))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=git_root)

    assert result.agent_failed is True, (
        f'Expected agent_failed=True but got {result.agent_failed!r}'
    )
    assert result.failure_token == origin_token, (
        f'Expected the specific origin {origin_token!r} but got '
        f'{result.failure_token!r} — the generic token would lose the diagnosis'
    )
    assert result.verdict == 'inconclusive', (
        f'Expected inconclusive but got {result.verdict!r}'
    )
    assert result.summary == 'agent-failed:no_tool_calls', (
        f'Expected the task-1811 sentinel keyed to the generic token but got '
        f'{result.summary!r}'
    )


@pytest.mark.asyncio
async def test_verify_failure_token_falls_back_when_no_origin(git_root):
    """With no `warning_origin`, failure_token degrades to the generic token.

    Guards that the preference chain DEGRADES rather than blanking the token:
    a missing origin must not leave failure_token empty, because an empty
    token is the success-path signal.
    """
    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=({'warning': 'max_steps_reached'}, []))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=git_root)

    assert result.agent_failed is True
    assert result.failure_token == 'max_steps_reached', (
        f"Expected fallback to 'max_steps_reached' but got {result.failure_token!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('raw_origin', 'expected_token'),
    [
        ({'nested': 'dict'}, 'no_tool_calls'),   # non-str: must not reach pydantic
        (17, 'no_tool_calls'),
        (None, 'no_tool_calls'),
        ('', 'no_tool_calls'),
        ('  spaced  ', 'spaced'),                # stripped
        ('x' * 200, 'x' * 64),                   # bounded
    ],
    ids=['dict', 'int', 'none', 'empty', 'stripped', 'bounded'],
)
async def test_verify_failure_token_coerces_malformed_origin(raw_origin, expected_token, git_root):
    """A malformed `warning_origin` must degrade, never raise.

    AgentLoop.run() gates the key on its closed CLI_WARNING_ORIGINS vocabulary,
    but verify() is the pydantic construction boundary and must hold
    independently: a non-str reaching ``failure_token`` would raise
    ValidationError out of verify(), which targeted.py's except arm would then
    record as a generic 'error' row — erasing the very diagnosis this task
    preserves.  So non-strings fall through to the next candidate in the
    preference chain, and strings are stripped and length-bounded so the
    audit-row census stays groupable.
    """
    result_dict = {'warning': 'no_tool_calls', 'warning_origin': raw_origin, 'text': ''}

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(result_dict, []))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=git_root)

    assert result.agent_failed is True, (
        f'Expected agent_failed=True but got {result.agent_failed!r}'
    )
    assert result.failure_token == expected_token, (
        f'Expected failure_token={expected_token!r} for raw origin '
        f'{raw_origin!r} but got {result.failure_token!r}'
    )


# ---------------------------------------------------------------------------
# MODEL-INVARIANT tests (task 4343): the two structured fields cannot desync.
# ---------------------------------------------------------------------------

class TestFailureFieldsCannotDesync:
    """`agent_failed` and `failure_token` must be set together or not at all.

    They encode one fact from two angles — the boolean is what consumers branch
    on, the token is what they store and GROUP BY — so an unvalidated pair is
    redundant state that drifts silently.  Either drift direction recreates the
    ambiguity this task closes: an ``agent_failed`` audit row with no token is
    undiagnosable, and an ``inconclusive`` row carrying a ``failure_token`` in
    its detail reads as a failure that never happened.
    """

    def test_failed_without_token_rejected(self):
        with pytest.raises(ValidationError, match='must agree'):
            VerificationResult(
                verdict=VerificationVerdict.inconclusive,
                confidence=0.0,
                agent_failed=True,
                failure_token='',
            )

    def test_token_without_failed_rejected(self):
        with pytest.raises(ValidationError, match='must agree'):
            VerificationResult(
                verdict=VerificationVerdict.inconclusive,
                confidence=0.4,
                agent_failed=False,
                failure_token='no_tool_calls',
            )

    def test_both_set_accepted(self):
        r = VerificationResult(
            verdict=VerificationVerdict.inconclusive,
            confidence=0.0,
            agent_failed=True,
            failure_token='cli_output_empty',
        )
        assert r.agent_failed is True
        assert r.failure_token == 'cli_output_empty'

    def test_neither_set_accepted(self):
        """The healthy default: both fields absent from the call entirely.

        Guards the many pre-existing construction sites that predate task 4343
        and never mention either field.
        """
        r = VerificationResult(verdict=VerificationVerdict.confirmed, confidence=0.9)
        assert r.agent_failed is False
        assert r.failure_token == ''


# ---------------------------------------------------------------------------
# SUCCESS-PATH test (written now; must stay green before AND after step-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_success_path_preserved(caplog, git_root):
    """When AgentLoop.run() returns a terminal verification_complete dict,
    verify() must preserve all fields and emit NO WARNING.

    This characterizes current behavior and guards against regressions in
    the step-2 refactor.

    (task 4343) The success path must leave BOTH new structured fields at
    their defaults — a healthy verdict that arrives carrying agent_failed=True
    or a non-empty failure_token would make the two states indistinguishable
    in the opposite direction.
    """
    terminal_result = {
        'verdict': 'confirmed',
        'confidence': 0.9,
        'evidence': [{'file_path': 'a.py', 'line_range': '1-10', 'snippet': 'x = 1', 'relevance': 'direct'}],
        'summary': 'looks good',
        'git_context': {'author': 'x', 'latest_relevant_commit': 'abc123'},
    }
    journal: list = []

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(terminal_result, journal))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify(claim='Feature X was shipped', codebase_root=git_root)

    assert result.verdict == 'confirmed'
    assert result.summary == 'looks good'
    assert result.confidence == 0.9
    assert result.evidence == [{'file_path': 'a.py', 'line_range': '1-10', 'snippet': 'x = 1', 'relevance': 'direct'}]
    assert result.git_context == {'author': 'x', 'latest_relevant_commit': 'abc123'}

    # Task 4343: both structured fields stay at their defaults on success.
    assert result.agent_failed is False, (
        f'Expected agent_failed=False on the success path but got {result.agent_failed!r}'
    )
    assert result.failure_token == '', (
        f'Expected an empty failure_token on the success path but got {result.failure_token!r}'
    )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warning_records, (
        f'Expected no WARNING records on success path, got: {[r.message for r in warning_records]}'
    )
