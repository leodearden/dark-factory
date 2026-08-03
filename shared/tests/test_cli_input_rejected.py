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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    build_failure_message,
    classify_agent_failure,
    invoke_claude_agent,
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


class _CountingCli:
    """Scripted ``invoke_fn`` that returns *results* in order, recording each
    call's kwargs, and REPEATS the last result once the script is exhausted.

    Modelled on ``test_model_not_found.py::_CountingModelNotFoundCli``: it must
    tolerate over-invocation rather than raising, because an unbounded-retry
    regression should surface as a call-count assertion (or the
    ``max_cap_retries`` bound) rather than as an opaque StopIteration from the
    fake.
    """

    def __init__(self, *results: AgentResult) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs: object) -> AgentResult:
        self.calls.append(dict(kwargs))
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx]


def _rejected(duration_ms: int = 2100) -> AgentResult:
    """A pre-turn CLI rejection.  Defaults to a SUB-5s duration, i.e. inside
    the heuristic cap-safety-net window, so these drives also pin that the new
    retry branch sits ABOVE that net."""
    return _parse_claude_output(_fast_exit_subprocess(duration_ms=duration_ms))


def _plain_empty_output() -> AgentResult:
    """The generic empty-output failure -- same shape, no marker on stderr."""
    return _parse_claude_output(_fast_exit_subprocess(stderr='some unrelated noise\n'))


def _succeeded() -> AgentResult:
    return AgentResult(
        success=True,
        output='the real answer',
        turns=7,
        cost_usd=0.42,
        duration_ms=31_000,
        session_id='sess-success',
    )


@pytest.mark.asyncio
class TestInvokeWithCapRetryRetriesCliInputRejectedOnce:
    """DEFECT A, transport layer: the agent was never asked anything and
    nothing was billed, so a pre-turn rejection is a FREE retry candidate.

    Retrying once here is the single seam that covers every consumer at once
    (workflow, steward, dry_run_unblock, judge, curator) -- none of which can
    distinguish this failure today.  The retry is deliberately BOUNDED at one:
    a second consecutive rejection is deterministic, not a glitch, and must
    reach a human instead of looping.
    """

    async def test_rejection_then_success_retries_once_and_returns_the_success(self):
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_rejected(), _succeeded())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate,
                'test[reject-then-ok]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
                session_id='sess-original',
            )

        assert len(cli.calls) == 2, (
            f'expected exactly one bounded retry (2 calls), got {len(cli.calls)}'
        )
        assert returned.success is True
        assert returned.output == 'the real answer'

    async def test_retry_is_fresh_not_a_resume_and_reuses_no_session_id(self):
        """A rejected attempt may already have committed its pre-allocated
        UUID to disk, so reusing it makes the CLI exit instantly with
        'Session ID ... is already in use' (the reify-3604 wedge)."""
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_rejected(), _succeeded())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            await invoke_with_cap_retry(
                gate,
                'test[reject-fresh]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
                session_id='sess-original',
            )

        first, retry = cli.calls
        assert 'resume_session_id' not in retry, (
            'the retry must be FRESH -- there is no session to resume, the CLI '
            'never started one'
        )
        assert retry['session_id'] != first['session_id'], (
            'a regenerated session_id is what avoids the reify-3604 '
            '"Session ID ... is already in use" collision'
        )
        assert retry['session_id']

    async def test_no_session_id_is_invented_when_the_caller_passed_none(self):
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_rejected(), _succeeded())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            await invoke_with_cap_retry(
                gate,
                'test[reject-no-session]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
            )

        assert len(cli.calls) == 2
        assert not cli.calls[1].get('session_id')

    async def test_retry_carries_the_original_prompt(self):
        """NOT CAP_HIT_RESUME_PROMPT / CRASH_RECOVERY_RESUME_PROMPT: those
        continue an existing session, and here no session ever existed."""
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_rejected(), _succeeded())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            await invoke_with_cap_retry(
                gate,
                'test[reject-prompt]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
            )

        assert cli.calls[1]['prompt'] == 'the original prompt'

    async def test_two_consecutive_rejections_are_bounded_and_returned(self):
        """BOUNDEDNESS: a second consecutive rejection is deterministic, not a
        glitch -- return it for normal steward handling rather than looping."""
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_rejected(), _rejected())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate,
                'test[reject-twice]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
            )

        assert len(cli.calls) == 2, (
            f'expected the retry budget to stop at 2 calls, got {len(cli.calls)}'
        )
        assert returned.success is False
        assert returned.subtype == 'error_cli_input_rejected'

    async def test_neither_scenario_mutates_account_phase_and_the_slot_settles(self):
        gate = make_gate_mock(account_count=2)
        slots = TestCliInputRejectedIsNeverReportedAsACapHit._capture_slots(gate)
        cli = _CountingCli(_rejected(), _rejected())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            await invoke_with_cap_retry(
                gate,
                'test[reject-no-mutation]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
            )

        gate._handle_cap_detected.assert_not_called()
        gate._handle_auth_failure.assert_not_called()
        assert gate.confirm_account_ok.called, 'every attempt must settle its slot'
        assert all(slot._settled for slot in slots), 'a probe slot was left dangling'

    async def test_plain_empty_output_is_not_retried_by_this_branch(self):
        """NARROWNESS: the new retry must not widen into the generic
        empty-output bucket, which has its own (steward-owned) retry policy."""
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_plain_empty_output(), _succeeded())

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate,
                'test[plain-empty]',
                invoke_fn=cli,
                max_cap_retries=2,
                backend='claude',
                prompt='the original prompt',
            )

        assert len(cli.calls) == 1, (
            f'a plain empty-output failure must NOT be retried here, got '
            f'{len(cli.calls)} calls'
        )
        assert returned.success is False
        assert returned.subtype == 'error_empty_output'


class _ScriptedClock:
    """Stand-in for ``shared.cli_invoke.datetime`` returning scripted stamps.

    Lets the fast-path drives below pin WHICH attempt the ``started_at`` /
    ``completed_at`` stamps describe, instead of relying on wall-clock
    resolution to distinguish two invocations microseconds apart.
    """

    def __init__(self, *stamps: str) -> None:
        self._stamps = list(stamps)
        self.issued: list[str] = []

    def now(self, _tz=None):
        stamp = self._stamps[min(len(self.issued), len(self._stamps) - 1)]
        self.issued.append(stamp)
        return SimpleNamespace(isoformat=lambda s=stamp: s)


@pytest.mark.asyncio
class TestInvokeWithCapRetryFastPathRetriesCliInputRejected:
    """The ``usage_gate is None`` fast path is a SECOND dispatch site — taken
    by the fused-memory callers that pass no gate (reconciliation, the judge,
    the curator).  It must carry the same bounded retry, or half the fleet
    keeps eating the failure silently.
    """

    async def test_rejection_then_success_retries_once_on_the_fast_path(self):
        cli = _CountingCli(_rejected(), _succeeded())

        returned = await invoke_with_cap_retry(
            None,
            'test[fastpath-reject-then-ok]',
            invoke_fn=cli,
            backend='claude',
            prompt='the original prompt',
            session_id='sess-original',
        )

        assert len(cli.calls) == 2, (
            f'expected exactly one bounded retry (2 calls), got {len(cli.calls)}'
        )
        assert returned.success is True
        assert returned.output == 'the real answer'

    async def test_fast_path_retry_is_fresh_with_a_regenerated_session_id(self):
        cli = _CountingCli(_rejected(), _succeeded())

        await invoke_with_cap_retry(
            None,
            'test[fastpath-fresh]',
            invoke_fn=cli,
            backend='claude',
            prompt='the original prompt',
            session_id='sess-original',
        )

        first, retry = cli.calls
        assert 'resume_session_id' not in retry
        assert retry['session_id'] != first['session_id']
        assert retry['prompt'] == 'the original prompt'

    async def test_fast_path_is_bounded_at_two_attempts(self):
        cli = _CountingCli(_rejected(), _rejected())

        returned = await invoke_with_cap_retry(
            None,
            'test[fastpath-reject-twice]',
            invoke_fn=cli,
            backend='claude',
            prompt='the original prompt',
        )

        assert len(cli.calls) == 2, (
            f'expected the retry budget to stop at 2 calls, got {len(cli.calls)}'
        )
        assert returned.success is False
        assert returned.subtype == 'error_cli_input_rejected'

    async def test_fast_path_does_not_retry_plain_empty_output(self):
        cli = _CountingCli(_plain_empty_output(), _succeeded())

        returned = await invoke_with_cap_retry(
            None,
            'test[fastpath-plain-empty]',
            invoke_fn=cli,
            backend='claude',
            prompt='the original prompt',
        )

        assert len(cli.calls) == 1
        assert returned.subtype == 'error_empty_output'

    async def test_account_name_is_empty_on_the_fast_path(self):
        """No gate means no account: the stamp must stay '' rather than
        inheriting anything from the discarded first attempt."""
        cli = _CountingCli(_rejected(), _succeeded())

        returned = await invoke_with_cap_retry(
            None,
            'test[fastpath-account-name]',
            invoke_fn=cli,
            backend='claude',
            prompt='the original prompt',
        )

        assert returned.account_name == ''

    async def test_cost_stamps_describe_the_attempt_actually_returned(self):
        """The invocations row must describe the attempt whose result is
        returned.  Leaving the FIRST attempt's stamps in place would report a
        ~2s window for a run that actually took the second attempt's wall
        clock — a silently false cost/latency record."""
        cli = _CountingCli(_rejected(), _succeeded())
        clock = _ScriptedClock('t1-start', 't1-done', 't2-start', 't2-done')
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        with patch('shared.cli_invoke.datetime', clock):
            await invoke_with_cap_retry(
                None,
                'test[fastpath-stamps]',
                invoke_fn=cli,
                cost_store=cost_store,
                backend='claude',
                prompt='the original prompt',
            )

        assert len(cli.calls) == 2
        saved = cost_store.save_invocation.await_args.kwargs
        assert saved['started_at'] == 't2-start', (
            f'started_at describes the DISCARDED first attempt: {saved["started_at"]!r}'
        )
        assert saved['completed_at'] == 't2-done'


@pytest.mark.asyncio
class TestNonBlankPromptPrecondition:
    """The other half of DEFECT A: make the failure impossible to cause from
    OUR side.

    The claude backend is 100% stdin-dependent -- ``build_claude_argv`` emits
    ``['claude', '--print', '--output-format', 'json']`` with NO positional
    prompt and no ``-`` marker -- so a blank prompt is piped happily
    (``stdin_data = prompt.encode()``) and resurfaces only as the opaque CLI
    argument error this task is classifying.  A blank prompt must therefore
    fail LOUDLY at the boundary, before a subprocess is spawned, before temp
    files are written, and before an account slot is consumed.
    """

    @staticmethod
    def _patched_dispatch():
        """Patch the subprocess spawn AND argv build so a leak past the guard
        is visible as a call rather than a real ``claude`` process."""
        return (
            patch(
                'shared.cli_invoke._run_subprocess',
                new_callable=AsyncMock,
                return_value=_SubprocessResult(
                    stdout='', stderr='', returncode=0, duration_ms=10
                ),
            ),
            patch(
                'shared.cli_invoke.build_claude_argv',
                return_value=(['claude', '--print'], []),
            ),
        )

    async def _assert_no_dispatch(self, prompt, **kwargs):
        run_patch, argv_patch = self._patched_dispatch()
        with run_patch as run_mock, argv_patch as argv_mock, pytest.raises(ValueError):
            await invoke_claude_agent(
                prompt=prompt,
                system_prompt='sys',
                cwd=Path('/tmp'),
                **kwargs,
            )
        run_mock.assert_not_called()
        argv_mock.assert_not_called()

    async def test_empty_prompt_raises_before_any_subprocess(self):
        await self._assert_no_dispatch('')

    async def test_whitespace_only_prompt_raises_before_any_subprocess(self):
        await self._assert_no_dispatch('   \n\t  ')

    async def test_guard_fires_even_with_no_resume_session(self):
        """UNCONDITIONAL: today's only prompt guard fires exclusively when a
        resume session IS set, which is why the plain (fresh-invocation) blank
        prompt sails straight through to the CLI."""
        await self._assert_no_dispatch('', resume_session_id=None)

    async def test_blank_prompt_never_consumes_an_account_slot(self):
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_succeeded())

        with pytest.raises(ValueError):
            await invoke_with_cap_retry(
                gate,
                'test[blank-prompt]',
                invoke_fn=cli,
                backend='claude',
                prompt='  ',
            )

        gate.invoke_slot.assert_not_called()
        assert cli.calls == []

    async def test_resume_with_empty_prompt_still_raises_typeerror(self):
        """REGRESSION: the existing resume-path contract is unchanged."""
        gate = make_gate_mock(account_count=2)

        with pytest.raises(TypeError):
            await invoke_with_cap_retry(
                gate,
                'test[resume-empty]',
                invoke_fn=_CountingCli(_succeeded()),
                backend='claude',
                prompt='',
                resume_session_id='abc',
            )

    async def test_resume_with_whitespace_only_prompt_raises_typeerror(self):
        """The resume guard's condition is strengthened to ``.strip()`` --
        same exception, same message, one more shape caught."""
        gate = make_gate_mock(account_count=2)

        with pytest.raises(TypeError):
            await invoke_with_cap_retry(
                gate,
                'test[resume-blank]',
                invoke_fn=_CountingCli(_succeeded()),
                backend='claude',
                prompt='   ',
                resume_session_id='abc',
            )

    async def test_a_normal_prompt_still_dispatches(self):
        """REGRESSION: the guard must not narrow the happy path."""
        run_patch, argv_patch = self._patched_dispatch()
        with run_patch as run_mock, argv_patch:
            await invoke_claude_agent(
                prompt='a real prompt',
                system_prompt='sys',
                cwd=Path('/tmp'),
            )
        run_mock.assert_called_once()

    async def test_a_normal_prompt_still_dispatches_through_cap_retry(self):
        gate = make_gate_mock(account_count=2)
        cli = _CountingCli(_succeeded())

        returned = await invoke_with_cap_retry(
            gate,
            'test[normal-prompt]',
            invoke_fn=cli,
            backend='claude',
            prompt='a real prompt',
        )

        assert returned.success is True
        assert len(cli.calls) == 1
