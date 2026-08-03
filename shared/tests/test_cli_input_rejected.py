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

from shared.cli_invoke import (
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    is_cli_invocation_rejected,
    is_zero_output_timeout,
)

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
