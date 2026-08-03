"""Task 3143 / esc-3118-1: the CLI fast-exit "prompt never reached stdin" failure.

The claude backend is 100% stdin-dependent (``build_claude_argv`` emits
``['claude', '--print', '--output-format', 'json']`` with NO positional prompt
and no ``-`` stdin marker), so when the prompt never lands on the child's stdin
the CLI exits on ARGUMENT VALIDATION -- before contacting the API and before any
model turn.  The observed payload is zero-cost, zero-turn, NOT timed out, and
carries an opaque CLI argument error on stderr.

Today every such run is laundered into the generic ``'error_empty_output'``
subtype, whose fixed summary ('agent returned empty output') actively
misdescribes the cause.  This module pins the distinct taxonomy.

Kept in its own module-local file (no ``shared/tests/conftest.py`` edit) --
mirrors test_model_not_found.py's stated no-conftest-edit rationale: verify.py's
``has_conftest`` would otherwise force a full owning-package suite fallback at
merge-verify time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    build_failure_message,
    classify_agent_failure,
    invoke_with_cap_retry,
    is_cli_invocation_rejected,
    is_zero_output_timeout,
)
from shared.invocation_outcome import (
    CLI_INPUT_REQUIRED_MARKERS,
    OK,
    CapHit,
    CliLocalError,
    classify_invocation,
)
from shared.testing import make_gate_mock

# The two stderr lines observed verbatim in esc-3118-1 (2026-07-28 ~16:31Z).
FAST_EXIT_STDERR = (
    'Warning: no stdin data received in 3s, proceeding without it. '
    'If piping from a slow command, redirect stdin explicitly: '
    '< /dev/null to skip, or wait longer.\n'
    'Error: Input must be provided either through stdin or as a prompt '
    'argument when using --print\n'
)


def _fast_exit_subprocess(**overrides: object) -> _SubprocessResult:
    """The esc-3118-1 payload as a ``_SubprocessResult``.

    duration_ms=17331 is the measured wall clock of the observed instance --
    deliberately OUTSIDE the pre-existing ``< 5000`` heuristic cap-safety-net
    window, which is why that instance escaped being reported as a synthetic
    cap hit (see TestClassifyInvocationCliInputRejectedIsNotACap for the
    sub-5s variant that does not).
    """
    kwargs: dict[str, object] = {
        'stdout': '',
        'stderr': FAST_EXIT_STDERR,
        'returncode': 1,
        'duration_ms': 17331,
        'timed_out': False,
        'transcript_turns': None,
    }
    kwargs.update(overrides)
    return _SubprocessResult(**kwargs)  # type: ignore[arg-type]


class TestParseClaudeOutputMintsCliInputRejectedSubtype:
    """``_parse_claude_output``'s empty-stdout branch must disambiguate a
    pre-turn CLI *rejection* from a genuine empty-output failure, exactly as
    task 2360 disambiguated a productive timeout kill from one.
    """

    def test_fast_exit_mints_error_cli_input_rejected(self):
        result = _parse_claude_output(_fast_exit_subprocess())
        assert result.subtype == 'error_cli_input_rejected', (
            f'expected the distinct pre-turn-rejection subtype, got '
            f'{result.subtype!r} -- laundering the cause into the generic '
            f'empty-output bucket is the defect this task fixes'
        )
        assert result.success is False

    def test_predicate_is_true_for_the_fast_exit_payload(self):
        result = _parse_claude_output(_fast_exit_subprocess())
        assert is_cli_invocation_rejected(result) is True

    def test_stderr_is_preserved_verbatim(self):
        result = _parse_claude_output(_fast_exit_subprocess())
        assert result.stderr == FAST_EXIT_STDERR, (
            'the CLI stderr is the ONLY evidence of the real cause and must '
            'survive parsing verbatim'
        )
        assert 'Input must be provided' in result.stderr

    def test_genuine_empty_output_is_unchanged(self):
        """NEGATIVE: an empty-stdout failure with unrelated stderr keeps the
        existing generic subtype and the predicate stays False."""
        result = _parse_claude_output(
            _fast_exit_subprocess(stderr='some unrelated warning\n')
        )
        assert result.subtype == 'error_empty_output'
        assert is_cli_invocation_rejected(result) is False

    def test_empty_stderr_is_unchanged(self):
        result = _parse_claude_output(_fast_exit_subprocess(stderr=''))
        assert result.subtype == 'error_empty_output'
        assert is_cli_invocation_rejected(result) is False

    def test_productive_timeout_kill_is_unchanged(self):
        """NEGATIVE: task 2360's productive-kill arm must keep winning."""
        result = _parse_claude_output(
            _fast_exit_subprocess(
                stderr='killed\n', timed_out=True, transcript_turns=3
            )
        )
        assert result.subtype == 'error_timeout_killed_with_progress'
        assert is_cli_invocation_rejected(result) is False

    def test_marker_on_a_timed_out_run_is_not_a_rejection(self):
        """A KILLED run is not a pre-turn rejection, even if the marker text
        happens to be on stderr: the timeout arms stay authoritative."""
        killed_with_marker = _parse_claude_output(
            _fast_exit_subprocess(timed_out=True, transcript_turns=3)
        )
        assert killed_with_marker.subtype == 'error_timeout_killed_with_progress'
        assert is_cli_invocation_rejected(killed_with_marker) is False

        killed_no_progress = _parse_claude_output(
            _fast_exit_subprocess(timed_out=True, transcript_turns=0)
        )
        assert killed_no_progress.subtype == 'error_empty_output'
        assert is_cli_invocation_rejected(killed_no_progress) is False

    def test_predicate_is_false_for_a_successful_result(self):
        assert (
            is_cli_invocation_rejected(
                AgentResult(success=True, output='ok', stderr=FAST_EXIT_STDERR)
            )
            is False
        )

    def test_predicate_is_false_once_work_was_billed(self):
        """turns/cost above zero mean the model WAS reached, so whatever the
        stderr says this is not a pre-first-turn transport rejection."""
        assert (
            is_cli_invocation_rejected(
                AgentResult(
                    success=False,
                    output='',
                    stderr=FAST_EXIT_STDERR,
                    turns=3,
                )
            )
            is False
        )
        assert (
            is_cli_invocation_rejected(
                AgentResult(
                    success=False,
                    output='',
                    stderr=FAST_EXIT_STDERR,
                    cost_usd=0.42,
                )
            )
            is False
        )

    def test_marker_match_is_case_insensitive(self):
        result = _parse_claude_output(
            _fast_exit_subprocess(
                stderr='ERROR: INPUT MUST BE PROVIDED EITHER THROUGH STDIN '
                'OR AS A PROMPT ARGUMENT WHEN USING --PRINT\n'
            )
        )
        assert result.subtype == 'error_cli_input_rejected'
        assert is_cli_invocation_rejected(result) is True

    def test_is_zero_output_timeout_is_false_for_the_fast_exit(self):
        """esc-3118-1's pinned measurement, kept as a permanent regression
        assertion: it is true BOTH before and after this fix, and documents
        WHY no timeout-keyed consumer ever catches this failure -- the
        predicate is keyed to the timeout family and the fast exit is not a
        timeout (``timed_out=False``)."""
        result = _parse_claude_output(_fast_exit_subprocess())
        assert result.timed_out is False
        assert is_zero_output_timeout(result) is False


class TestClassifyAgentFailureCliInputRejected:
    """The escalation summary an operator reads is built from
    ``classify_agent_failure(...).summary``.  For a pre-turn rejection the
    generic EMPTY_OUTPUT rule's FIXED string ('agent returned empty output')
    actively misdescribes the cause and discards the only evidence there is —
    the CLI's own stderr line.  This suite pins the distinct kind and a
    cause-carrying summary.
    """

    @staticmethod
    def _fast_exit_result() -> AgentResult:
        return _parse_claude_output(_fast_exit_subprocess())

    def test_kind_value_is_stable(self):
        assert AgentFailureKind.CLI_INPUT_REJECTED.value == 'cli_input_rejected'

    def test_fast_exit_classifies_as_cli_input_rejected(self):
        cls = classify_agent_failure(self._fast_exit_result())
        assert cls.kind is AgentFailureKind.CLI_INPUT_REJECTED
        assert cls.kind is not AgentFailureKind.EMPTY_OUTPUT

    def test_summary_carries_the_real_cli_error_line(self):
        cls = classify_agent_failure(self._fast_exit_result())
        assert 'Input must be provided' in cls.summary, (
            f'the summary must embed the CLI stderr cause, got {cls.summary!r}'
        )
        assert cls.summary != 'agent returned empty output'

    def test_diagnostic_detail_still_dumps_every_signal(self):
        cls = classify_agent_failure(self._fast_exit_result())
        assert "subtype='error_cli_input_rejected'" in cls.diagnostic_detail
        assert 'turns=' in cls.diagnostic_detail
        assert 'cost_usd=' in cls.diagnostic_detail
        assert 'timed_out=' in cls.diagnostic_detail
        assert 'Input must be provided' in cls.diagnostic_detail

    def test_summary_falls_back_to_the_last_stderr_line_never_a_fabrication(self):
        """When the subtype is set but no stderr line matches (defensive: a
        caller stamped the subtype by hand), the cause must degrade to real
        observed text, never to an invented string."""
        cls = classify_agent_failure(
            AgentResult(
                success=False,
                output='',
                subtype='error_cli_input_rejected',
                stderr='some other terminal noise\n',
            )
        )
        assert cls.kind is AgentFailureKind.CLI_INPUT_REJECTED
        assert 'some other terminal noise' in cls.summary

    def test_timeout_rule_still_outranks_the_new_rule(self):
        """PRECEDENCE: a killed run carrying the marker is still TIMED_OUT."""
        cls = classify_agent_failure(
            AgentResult(
                success=False,
                output='',
                subtype='error_cli_input_rejected',
                stderr=FAST_EXIT_STDERR,
                timed_out=True,
            )
        )
        assert cls.kind is AgentFailureKind.TIMED_OUT

    def test_plain_empty_output_still_classifies_as_empty_output(self):
        """REGRESSION: the existing generic rule is untouched."""
        cls = classify_agent_failure(
            _parse_claude_output(_fast_exit_subprocess(stderr='unrelated\n'))
        )
        assert cls.kind is AgentFailureKind.EMPTY_OUTPUT
        assert cls.summary == 'agent returned empty output'

    def test_build_failure_message_embeds_the_stderr_cause(self):
        message = build_failure_message('architect', self._fast_exit_result())
        assert message.startswith('architect failed: ')
        assert 'Input must be provided' in message


class TestClassifyInvocationCliInputRejectedIsNotACap:
    """The SUB-5s hazard: ``invoke_with_cap_retry``'s heuristic cap safety-net
    fires on ``not success and cost_usd == 0 and turns <= 1 and
    duration_ms < 5000``.  The observed instance escaped it only by accident
    (duration_ms=17331), but the CLI's stdin wait is just 3s, so a faster
    fast-exit lands INSIDE that window.

    Because ``classify_invocation`` currently returns ``Failure('unclassified')``
    for this shape, the net's ``isinstance(outcome, (CliLocalError,
    ServerError))`` escape does not fire and the run is reported as a SYNTHETIC
    CAP HIT — churning the whole OAuth pool through compounding cooldowns for a
    local argument-validation error the API never even saw.  The CLI exits
    BEFORE contacting the API here, so a cap message is impossible by
    construction; this must classify as ``CliLocalError``.
    """

    def test_marker_table_is_non_empty(self):
        """Guard the guard: an empty table would make every assertion below
        vacuous rather than failing."""
        assert CLI_INPUT_REQUIRED_MARKERS

    def test_fast_exit_classifies_as_cli_local_error_not_unclassified(self):
        result = _parse_claude_output(_fast_exit_subprocess())
        outcome = classify_invocation(result, strict_confirm=True, backend='claude')
        assert isinstance(outcome, CliLocalError), (
            f'expected CliLocalError so the cap safety-net escape fires, got '
            f'{outcome!r} -- an unclassified outcome is reported as a synthetic cap'
        )
        assert outcome.marker in CLI_INPUT_REQUIRED_MARKERS
        assert not isinstance(outcome, CapHit)

    def test_cli_local_error_outranks_co_occurring_cap_text(self):
        """PRECEDENCE (reify-3604's structural fix, extended): even when the
        output ALSO carries cap-like text, the local rejection wins."""
        outcome = classify_invocation(
            AgentResult(
                success=False,
                output="You've hit your usage limit. Your plan resets in 3h.",
                stderr=FAST_EXIT_STDERR,
            ),
            strict_confirm=True,
            backend='claude',
        )
        assert isinstance(outcome, CliLocalError)
        assert not isinstance(outcome, CapHit)

    def test_successful_run_quoting_the_marker_is_still_ok(self):
        """The ``if result.success`` short-circuit must keep protecting a run
        that merely QUOTES the marker text (e.g. an agent discussing this very
        failure mode) from being misclassified as a local CLI error."""
        outcome = classify_invocation(
            AgentResult(
                success=True,
                output=(
                    'I diagnosed the failure: the CLI printed "Error: Input must '
                    'be provided either through stdin or as a prompt argument '
                    'when using --print".'
                ),
            ),
            strict_confirm=True,
            backend='claude',
        )
        assert isinstance(outcome, OK)


@pytest.mark.asyncio
class TestCliInputRejectedIsNeverReportedAsACapHit:
    """End-to-end consumer side: a sub-5s fast-exit driven through
    ``invoke_with_cap_retry`` must never mark an account capped."""

    @staticmethod
    def _capture_slots(gate) -> list:
        """Wrap ``gate.invoke_slot`` so every yielded slot is captured.

        ``make_gate_mock`` builds a fresh slot per ``__aenter__`` inside its
        own closure, so the only way to assert on what production passed to
        ``slot.report(...)`` is to intercept the context manager here.
        """
        slots: list = []
        inner = gate.invoke_slot

        def _wrap(*args, **kwargs):
            cm = inner(*args, **kwargs)
            aenter = cm.__aenter__

            async def _capturing_aenter(*a, **kw):
                slot = await aenter(*a, **kw)
                slots.append(slot)
                return slot

            cm.__aenter__ = AsyncMock(side_effect=_capturing_aenter)
            return cm

        gate.invoke_slot = MagicMock(side_effect=_wrap)
        return slots

    @staticmethod
    def _fast_variant() -> AgentResult:
        """duration_ms=2100 -- deliberately INSIDE the pre-existing ``< 5000``
        heuristic cap-safety-net window, unlike the 17331ms observed instance
        that escaped it by luck."""
        return _parse_claude_output(_fast_exit_subprocess(duration_ms=2100))

    async def test_no_account_is_ever_reported_capped(self):
        gate = make_gate_mock(account_count=2)
        slots = self._capture_slots(gate)
        result = self._fast_variant()
        calls: list[dict] = []

        async def fake_invoke(**kwargs) -> AgentResult:
            calls.append(kwargs)
            return result

        # max_cap_retries is a second, independent bound: if the synthetic-cap
        # path ever fires, this turns what would be an unbounded pool-churn
        # loop into a loud AllAccountsCappedException instead of a hang.
        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate,
                'test[cli-input-rejected]',
                invoke_fn=fake_invoke,
                max_cap_retries=2,
                backend='claude',
                prompt='hi',
            )

        assert returned.success is False, 'the rejection is returned to the caller as-is'
        assert calls, 'the fake CLI was never invoked'
        # No CapHit may be handed to slot.report() -- the direct observable.
        reported = [
            call.args[0] for slot in slots for call in slot.report.call_args_list if call.args
        ]
        assert not any(isinstance(outcome, CapHit) for outcome in reported), (
            f'a pre-turn CLI rejection was reported as a cap hit: {reported!r}'
        )
        # ...and the gate's cap-detected transition is not taken by any route.
        gate._handle_cap_detected.assert_not_called()
