"""Task 6022: an API-side usage-policy refusal is its own failure kind.

The recon CodebaseVerifier's calls were refused by the API's
``reasoning_extraction`` classifier.  The CLI reports such a call as
``is_error=true`` with ``subtype='success'``, and today
``classify_agent_failure`` launders it into ``UNKNOWN`` ('no specific failure
signal').  The refused CLI JSON carries a STRUCTURED marker,
``stop_reason='refusal'``, while the human-readable text varies by model
version — so the kind is keyed on the field, never on the prose
(heuristic 12).

The two fixture texts are the measured outputs of production-refused calls
(Sonnet 5 and Sonnet 5.5).

Kept in its own module-local file (no ``shared/tests/conftest.py`` edit) --
mirrors test_cli_input_rejected.py's stated no-conftest-edit rationale:
verify.py's ``has_conftest`` would otherwise force a full owning-package suite
fallback at merge-verify time.
"""

from __future__ import annotations

import json

import pytest

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    build_failure_message,
    classify_agent_failure,
)
from shared.invocation_outcome import CapHit, Failure, NearCap, classify_invocation

SONNET_5_REFUSAL_TEXT = (
    "API Error: Sonnet 5 can't help with this. Start a new session to continue."
    '\n\nLearn more: https://www.anthropic.com/legal/aup'
    '\n\nDetails: `[reasoning_extraction]`'
    '\n\nRequest ID: req_011CfW1erek1KB375VYgvhd9'
)
SONNET_5_5_REFUSAL_TEXT = (
    "API Error: Sonnet 5.5's safeguards flagged this message "
    '(https://www.anthropic.com/legal/aup). This sometimes happens with safe, '
    "normal conversations. Claude Code can't respond to this message with "
    'Sonnet 5.5.'
)
MEASURED_REFUSAL_TEXTS = pytest.mark.parametrize(
    'refusal_text',
    [SONNET_5_REFUSAL_TEXT, SONNET_5_5_REFUSAL_TEXT],
    ids=['sonnet-5', 'sonnet-5.5'],
)


def _refused_cli_json(refusal_text: str, **overrides: object) -> str:
    """The measured shape of a refused call's CLI result JSON."""
    payload: dict[str, object] = {
        'type': 'result',
        'subtype': 'success',
        'is_error': True,
        'api_error_status': None,
        'stop_reason': 'refusal',
        'terminal_reason': 'api_error',
        'num_turns': 2,
        'total_cost_usd': 0.0085,
        'session_id': 's-1',
        'result': refusal_text,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _parsed(stdout: str, returncode: int = 1) -> AgentResult:
    return _parse_claude_output(
        _SubprocessResult(stdout=stdout, stderr='', returncode=returncode, duration_ms=4288)
    )


def _parsed_refusal(refusal_text: str) -> AgentResult:
    return _parsed(_refused_cli_json(refusal_text))


class TestParseClaudeOutputCarriesStopReason:
    @MEASURED_REFUSAL_TEXTS
    def test_refused_json_is_a_failure_with_refusal_stop_reason(self, refusal_text):
        result = _parsed_refusal(refusal_text)
        assert result.success is False
        assert result.stop_reason == 'refusal'

    def test_normal_success_keeps_its_stop_reason(self):
        stdout = json.dumps(
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'stop_reason': 'end_turn',
                'num_turns': 1,
                'total_cost_usd': 0.01,
                'session_id': 's-2',
                'result': 'done',
            }
        )
        result = _parsed(stdout, returncode=0)
        assert result.success is True
        assert result.stop_reason == 'end_turn'

    def test_json_without_stop_reason_gives_none(self):
        """Older CLIs never emitted the key."""
        stdout = json.dumps(
            {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'ok'}
        )
        assert _parsed(stdout, returncode=0).stop_reason is None

    def test_non_str_stop_reason_gives_none(self):
        stdout = _refused_cli_json(SONNET_5_5_REFUSAL_TEXT, stop_reason=5)
        assert _parsed(stdout).stop_reason is None


class TestClassifyAgentFailureApiRefusal:
    def test_kind_value_is_the_census_token(self):
        assert AgentFailureKind.API_REFUSAL == 'api_refusal'

    @MEASURED_REFUSAL_TEXTS
    def test_measured_refusal_classifies_api_refusal(self, refusal_text):
        failure = classify_agent_failure(_parsed_refusal(refusal_text))
        assert failure.kind == AgentFailureKind.API_REFUSAL

    @MEASURED_REFUSAL_TEXTS
    def test_summary_is_specific_and_not_the_transient_requeue_marker(self, refusal_text):
        """'agent API error: HTTP' is the prefix orchestrator scheduler.py's
        transient requeue lane keys on; a content-driven refusal must never
        be routed there."""
        failure = classify_agent_failure(_parsed_refusal(refusal_text))
        assert 'no specific failure signal' not in failure.summary
        assert 'agent API error: HTTP' not in failure.summary

    @MEASURED_REFUSAL_TEXTS
    def test_failure_message_carries_the_summary(self, refusal_text):
        parsed = _parsed_refusal(refusal_text)
        summary = classify_agent_failure(parsed).summary
        assert summary in build_failure_message('Claude CLI agent', parsed)


class TestRefusalIsKeyedOnTheStructuredField:
    def test_stop_reason_alone_classifies_api_refusal(self):
        result = AgentResult(
            success=False, subtype='success', stop_reason='refusal', output='unrelated text'
        )
        assert classify_agent_failure(result).kind == AgentFailureKind.API_REFUSAL

    def test_refusal_prose_without_stop_reason_is_not_api_refusal(self):
        result = AgentResult(
            success=False, subtype='success', stop_reason=None, output=SONNET_5_5_REFUSAL_TEXT
        )
        assert classify_agent_failure(result).kind == AgentFailureKind.UNKNOWN


class TestRefusalRuleSitsImmediatelyAboveUnknown:
    """The rule may only reclassify results that were UNKNOWN, so every
    existing kind (and the orchestrator routing keyed on it) is unmoved."""

    def test_timed_out_still_wins(self):
        result = AgentResult(
            success=False, output='', subtype='success', stop_reason='refusal', timed_out=True
        )
        assert classify_agent_failure(result).kind == AgentFailureKind.TIMED_OUT

    def test_api_error_status_still_wins(self):
        result = AgentResult(
            success=False,
            output='',
            subtype='success',
            stop_reason='refusal',
            api_error_status=429,
        )
        assert classify_agent_failure(result).kind == AgentFailureKind.API_ERROR

    def test_max_turns_still_wins(self):
        result = AgentResult(
            success=False, output='', subtype='error_max_turns', stop_reason='refusal'
        )
        assert classify_agent_failure(result).kind == AgentFailureKind.MAX_TURNS


class TestRefusalNeverPausesAnAccount:
    @MEASURED_REFUSAL_TEXTS
    def test_refusal_is_a_plain_failure_not_a_cap(self, refusal_text):
        """The measured refusals are sub-5s fast failures, yet the text holds
        no cap confirm-keyword: a content-driven refusal must never be read
        as a cap hit and pause (or fail over) a healthy account."""
        outcome = classify_invocation(_parsed_refusal(refusal_text), strict_confirm=True)
        assert not isinstance(outcome, (CapHit, NearCap))
        assert isinstance(outcome, Failure)
