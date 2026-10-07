"""Per-leg classification: what verify knows about a leg beyond its output.

``verify_classify.classify_failure`` answers what a tool's OUTPUT says. Two
facts it cannot see are known only to verify itself (task 4843):

- which SLOT the leg filled (lint, type or test), which names the gate that
  went red when the output names no more specific cause;
- that verify interposed ``/bin/bash`` around every leg, so a positive
  ``128 + N`` exit status is that shell reporting its child's death by
  signal N.

This module pins both, through the public predicate and the leg classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from _verify_config_corpus import ROOT_LINT_COMMAND, ROOT_TYPE_CHECK_COMMAND

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.verify import _summarize_checks
from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
from orchestrator.verify_classify import (
    is_external_kill_rc,
    is_shell_reported_external_kill,
)

_GREEN_LEG = (0, '', None)

_TEST_CMD = 'uv run pytest tests/'


def _summarize(*, test=_GREEN_LEG, lint=_GREEN_LEG, type_=_GREEN_LEG):
    """``_summarize_checks`` over (rc, output, cmd) legs, none timed out."""
    (test_rc, test_out, test_cmd) = test
    (lint_rc, lint_out, lint_cmd) = lint
    (type_rc, type_out, type_cmd) = type_
    return _summarize_checks(
        test_rc, test_out, False, test_cmd,
        lint_rc, lint_out, False, lint_cmd,
        type_rc, type_out, False, type_cmd,
    )


def _lint_leg_category(lint_out: str, lint_cmd: str) -> str:
    return _summarize(lint=(1, lint_out, lint_cmd))[1]


def _type_leg_category(type_out: str, type_cmd: str) -> str:
    return _summarize(type_=(1, type_out, type_cmd))[1]


# Verbatim task-3121 ruff output: the human text a real lint red prints.
_RUFF_I001_OUTPUT = (
    'orchestrator/src/orchestrator/harness.py:7252:1: I001 [*] Import block is '
    'un-sorted or un-formatted\n'
    'Found 1 error.\n'
    '[*] 1 fixable with the `--fix` option.\n'
)

_PYRIGHT_TEXT_OUTPUT = (
    '/repo/orchestrator/src/x.py:1:1 - error: "foo" is not defined '
    '(reportUndefinedVariable)\n'
    '1 error, 0 warnings, 0 informations \n'
)

# Task 3367's mis-resolved-interpreter signature, in pyright's real line
# shape: the baseline sentinel pytest plus enough distinct third-party
# modules unresolved.
_PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT = '\n'.join(
    f'  /repo/src/fused_memory/server.py:{line}:8 - error: Import "{module}" '
    'could not be resolved (reportMissingImports)'
    for line, module in enumerate(
        ('pytest', 'pydantic', 'aiosqlite', 'openai', 'qdrant_client'), start=8,
    )
)


class TestShellReportedExternalKill:
    """``is_shell_reported_external_kill(rc)`` reads a shell's ``128 + N``
    status for the external-kill signals, minus SIGKILL."""

    @pytest.mark.parametrize(
        ('rc', 'signal_name'),
        [(143, 'SIGTERM'), (130, 'SIGINT'), (129, 'SIGHUP')],
    )
    def test_external_kill_signals_are_recognised(self, rc, signal_name):
        assert is_shell_reported_external_kill(rc) is True, (
            f'rc={rc} is a shell reporting its child died of {signal_name}'
        )

    def test_sigkill_is_excluded(self):
        """Positive 137 is the ordinary branch-caused OOM of a grandchild,
        the residual the INFRA_KILL row already documents, so it must stay a
        code-fault exit status."""
        assert is_shell_reported_external_kill(137) is False

    @pytest.mark.parametrize('rc', [139, 134, 135, 132, 136])
    def test_crash_signals_are_code_faults(self, rc):
        assert is_shell_reported_external_kill(rc) is False, (
            f'rc={rc} is a crash signal, a fault of the code under test'
        )

    @pytest.mark.parametrize('rc', [0, 1, 2, 4, 5, 124, 127, 128, 255])
    def test_ordinary_exit_statuses_are_not_kills(self, rc):
        assert is_shell_reported_external_kill(rc) is False

    @pytest.mark.parametrize('rc', [-15, -9])
    def test_raw_negative_returncodes_belong_to_is_external_kill_rc(self, rc):
        assert is_shell_reported_external_kill(rc) is False

    def test_is_external_kill_rc_is_not_widened(self):
        """Task #5132: the bare predicate keeps refusing shell conventions;
        only the caller that interposed the shell may read them."""
        assert is_external_kill_rc(143) is False


class TestChainedLintTypeArmsAreReached:
    """``classify_failure`` RECEIVES the tool recovered from the live chained
    lint/type commands, proven through ``_summarize_checks`` with nothing
    patched.

    Before task 4843 both commands parsed OPAQUE, and the OPAQUE ladder's
    rustc-code regex swallowed any output quoting ``error[E0308]:``. That is
    the C1 token-swallowing the per-tool dispatch exists to prevent: a ruff
    or pyright JSON message quoting the token classified COMPILE_ERROR. Only
    the RUFF/PYRIGHT JSON arms can turn that payload into LINT_ERROR or
    TYPE_ERROR, so this category delta shows the arm is engaged, not merely
    named.
    """

    def test_ruff_json_through_the_live_lint_chain_is_lint_error(self):
        output = json.dumps([{
            'code': 'F821',
            'filename': 'a.py',
            'severity': 'error',
            'message': 'Undefined name `error[E0308]:`',
        }])
        category = _lint_leg_category(output, ROOT_LINT_COMMAND)
        assert category != FailureCategory.COMPILE_ERROR, (
            'the OPAQUE ladder swallowed the quoted rustc token'
        )
        assert category == FailureCategory.LINT_ERROR

    def test_pyright_json_through_the_live_type_chain_is_type_error(self):
        output = json.dumps({
            'version': '1.1.408',
            'generalDiagnostics': [{
                'file': 'a.py',
                'severity': 'error',
                'message': 'Argument mismatch quoting error[E0308]: mismatched types',
            }],
        })
        category = _type_leg_category(output, ROOT_TYPE_CHECK_COMMAND)
        assert category != FailureCategory.COMPILE_ERROR, (
            'the OPAQUE ladder swallowed the quoted rustc token'
        )
        assert category == FailureCategory.TYPE_ERROR


class TestLegSlotNamesTheRedGate:
    """A red lint or type leg whose output names no more specific cause is
    reported as LINT_ERROR or TYPE_ERROR, the gate that went red, instead of
    unknown_test_failure (task 4843, #3975). Only that fall-through is
    relabelled; the test slot never is."""

    def test_human_text_lint_red_is_lint_error(self):
        _, category, _, summary, failing = _summarize(
            lint=(1, _RUFF_I001_OUTPUT, ROOT_LINT_COMMAND),
        )
        assert category != FailureCategory.UNKNOWN_TEST_FAILURE
        assert category == FailureCategory.LINT_ERROR
        assert summary == 'Failures: lint issues'
        assert failing == ['lint_error']

    def test_human_text_type_red_is_type_error(self):
        _, category, _, summary, failing = _summarize(
            type_=(1, _PYRIGHT_TEXT_OUTPUT, ROOT_TYPE_CHECK_COMMAND),
        )
        assert category == FailureCategory.TYPE_ERROR
        assert summary == 'Failures: type errors'
        assert failing == ['type_error']

    def test_specific_lint_diagnosis_survives(self):
        category = _lint_leg_category(
            'flock: failed to acquire lock on /var/lock/mylock\n', ROOT_LINT_COMMAND,
        )
        assert category == FailureCategory.FLOCK_ERROR, (
            'only the unknown fall-through is relabelled'
        )

    def test_interpreter_misresolution_on_type_leg_stays_env_transient(self):
        category = _type_leg_category(
            _PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT, ROOT_TYPE_CHECK_COMMAND,
        )
        assert category == FailureCategory.ENV_TRANSIENT, (
            'only the unknown fall-through is relabelled'
        )

    def test_externally_killed_lint_leg_stays_infra_kill(self):
        _, category, _, _, _ = _summarize(lint=(-9, '', ROOT_LINT_COMMAND))
        assert category == FailureCategory.INFRA_KILL, (
            'only the unknown fall-through is relabelled'
        )

    def test_test_slot_is_never_relabelled(self):
        _, category, _, _, _ = _summarize(test=(5, 'no tests ran in 0.01s\n', _TEST_CMD))
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_test_failure_outranks_lint_error(self):
        _, category, _, _, failing = _summarize(
            test=(1, 'FAILED tests/test_x.py::test_y - assert 1 == 2\n', _TEST_CMD),
            lint=(1, _RUFF_I001_OUTPUT, ROOT_LINT_COMMAND),
        )
        assert category == FailureCategory.TEST_FAILURE
        assert failing == ['test_failure', 'lint_error']

    def test_type_error_outranks_lint_error(self):
        _, category, _, _, _ = _summarize(
            lint=(1, _RUFF_I001_OUTPUT, ROOT_LINT_COMMAND),
            type_=(1, _PYRIGHT_TEXT_OUTPUT, ROOT_TYPE_CHECK_COMMAND),
        )
        assert category == FailureCategory.TYPE_ERROR

    @pytest.mark.asyncio
    async def test_briefing_names_the_lint_gate(self, tmp_path: Path):
        """The literal user-observable signal: the debugger's briefing says
        lint_error, not unknown_test_failure.

        A task worktree always carries ``.task/``, and the briefing's
        Category line renders only beside the attempt logs that
        ``_persist_attempt_logs`` writes there.
        """
        (tmp_path / '.task').mkdir()
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_TEST_CMD,
            lint_command=ROOT_LINT_COMMAND,
            type_check_command='echo type',
        )

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            if 'ruff check' in cmd:
                return 1, _RUFF_I001_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(
                tmp_path, config,
                attempt_id=1, task_id='4843', archive_root=tmp_path / 'archive',
            )

        assert result.category == 'lint_error'
        report = result.failure_report()
        assert 'Category: lint_error' in report
        assert 'unknown_test_failure' not in report


_XDIST_ABORT_FIXTURES = Path(__file__).parent / 'fixtures' / 'verify_xdist_abort'

# The esc-4246-4 sighting's own command; it resolves PYTEST.
_FUSED_MEMORY_TEST_CMD = (
    'uv run --directory fused-memory pytest tests/ --tb=short -q --timeout=300'
)

_SHAPE2_TALLY = '18885 passed, 1 skipped, 249 warnings in 225.85s (0:03:45)'

# A killed -q session's last progress line: it never prints a short summary,
# so its F or E is the only failure evidence left.
_KILLED_PROGRESS_WITH_F = '....F...' + '.' * 64 + ' [ 40%]'


def _xdist_abort_log(name: str) -> str:
    return (_XDIST_ABORT_FIXTURES / name).read_text()


def _with_line(text: str, line: str) -> str:
    separator = '' if text.endswith('\n') else '\n'
    return f'{text}{separator}{line}\n'


def _xdist_test_leg(rc: int, output: str, *, timed_out: bool = False):
    """(category, failing_leg_categories) for a red test leg, lint/type green."""
    _, category, _, _, failing = _summarize_checks(
        rc, output, timed_out, _FUSED_MEMORY_TEST_CMD,
        0, '', False, None,
        0, '', False, None,
    )
    return category, failing


def _assert_infra_abort(category: str) -> None:
    assert category == FailureCategory.INFRA_ABORT
    policy = CATEGORY_POLICY[FailureCategory(category)]
    assert policy.is_infra_transient is True, (
        'a verdictless abort earns the bounded retry, not a debugger dispatch'
    )
    assert policy.verdict_indeterminate is False, (
        'the leg keeps its veto at the per-land cross-check'
    )


class TestVerdictlessXdistExitIsInfraAbort:
    """A pytest-xdist test leg torn down from outside the tests reached no
    verdict about the branch, so it is INFRA_ABORT (task 4843, #5132).

    Two arms, because no single piece of evidence covers every shape: the
    shell-reported SIGTERM/SIGINT/SIGHUP exit status, and the channel-collapse
    signature xdist workers print when they outlive their controller. Every
    log is a real capture (``fixtures/verify_xdist_abort/PROVENANCE.md``).
    Any failure evidence vetoes both arms, and the classifier's own answer
    passes through.
    """

    def test_controller_sigterm_with_kill_rc_is_infra_abort(self):
        """esc-4808-1, esc-3824-1/-2: rc=143 and the signature, both arms."""
        category, failing = _xdist_test_leg(
            143, _xdist_abort_log('shape1_controller_sigterm.log'),
        )
        _assert_infra_abort(category)
        assert failing == ['infra_abort']

    def test_channel_collapse_signature_alone_is_infra_abort(self):
        """esc-4808-1's text with no kill rc: the text arm alone fires."""
        category, _ = _xdist_test_leg(
            1, _xdist_abort_log('shape1_controller_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_group_sigterm_is_caught_by_the_kill_rc_alone(self):
        """esc-4808-1's kill delivered to the whole group leaves no
        signature, so only the rc arm can catch it."""
        category, _ = _xdist_test_leg(
            143, _xdist_abort_log('shape1_group_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_group_sigterm_without_kill_rc_stays_unknown(self):
        """The discriminating twin of the case above: the same text with no
        kill rc carries neither arm's evidence."""
        category, _ = _xdist_test_leg(
            1, _xdist_abort_log('shape1_group_sigterm.log'),
        )
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_green_tally_then_sigterm_is_infra_abort(self):
        """esc-4246-4: the session finished green, then the shell reported a
        SIGTERM. The log carries no signature; the kill rc decides."""
        category, _ = _xdist_test_leg(
            143, _xdist_abort_log('shape2_green_tally.log'),
        )
        _assert_infra_abort(category)

    @pytest.mark.parametrize('rc', [143, 1])
    def test_bringup_sigterm_is_infra_abort_on_the_signature(self, rc):
        """esc-4071-1: its rc was never recorded, so the text signature
        alone must decide."""
        category, _ = _xdist_test_leg(
            rc, _xdist_abort_log('shape3_bringup_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_failed_line_vetoes_the_abort(self):
        """Rule: a FAILED line is a verdict, so TEST_FAILURE survives."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            'FAILED tests/test_x.py::test_y - AssertionError',
        )
        category, _ = _xdist_test_leg(143, output)
        assert category == FailureCategory.TEST_FAILURE

    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_killed_progress_failure_char_vetoes_the_abort(self, status_char):
        """Rule: a killed -q session prints no short summary, so an F or E
        in its progress line is the only failure evidence and must veto."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            _KILLED_PROGRESS_WITH_F.replace('F', status_char),
        )
        category, _ = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize(
        'evidence',
        ['ERROR tests/test_x.py::test_y - fixture', 'E   assert 1 == 2'],
    )
    def test_error_line_vetoes_the_abort(self, evidence):
        """Rule: an ERROR summary line or an assertion E-line is failure
        evidence, so the classifier's own answer passes through."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'), evidence,
        )
        category, _ = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_failing_tally_vetoes_the_abort(self):
        """Rule: a tally that counts a failure is a verdict, even when the
        shell then reports a SIGTERM."""
        green = _xdist_abort_log('shape2_green_tally.log')
        assert _SHAPE2_TALLY in green
        output = green.replace(
            _SHAPE2_TALLY,
            '1 failed, 18884 passed, 1 skipped, 249 warnings in 225.85s (0:03:45)',
        )
        category, _ = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_internalerror_is_decided_by_the_classifier_first(self):
        """Rule: only the unknown fall-through is rewritten, so the
        classifier's PYTEST_INTERNALERROR wins over both arms."""
        output = _with_line(
            _xdist_abort_log('shape3_bringup_sigterm.log'),
            'INTERNALERROR> Traceback (most recent call last):',
        )
        category, _ = _xdist_test_leg(143, output)
        assert category == FailureCategory.PYTEST_INTERNALERROR

    @pytest.mark.parametrize(
        ('rc', 'why'),
        [
            (137, 'SIGKILL is excluded: the branch-OOM residual'),
            (139, 'a crash signal is a fault of the code under test'),
            (1, 'no kill rc and no signature'),
        ],
    )
    def test_non_abort_exit_status_stays_unknown(self, rc, why):
        """Rule: only a shell-reported SIGTERM/SIGINT/SIGHUP is the rc arm."""
        category, _ = _xdist_test_leg(rc, _xdist_abort_log('shape2_green_tally.log'))
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE, why

    def test_raw_negative_returncode_stays_infra_kill(self):
        """Rule: the structural external-kill arm is unchanged and still
        dominates."""
        category, _ = _xdist_test_leg(-15, _xdist_abort_log('shape2_green_tally.log'))
        assert category == FailureCategory.INFRA_KILL

    def test_lint_leg_is_out_of_the_abort_scope(self):
        """Rule: the abort arms are PYTEST-scoped, because their veto set is
        a pytest-output veto set. A killed lint leg reports its gate; this
        pins the documented residual."""
        _, category, _, _, _ = _summarize(
            lint=(143, _xdist_abort_log('shape1_controller_sigterm.log'), ROOT_LINT_COMMAND),
        )
        assert category == FailureCategory.LINT_ERROR

    def test_timed_out_leg_stays_infra_timeout(self):
        """Rule: the classifier's timeout guard is unchanged and decides
        first."""
        category, _ = _xdist_test_leg(
            143, _xdist_abort_log('shape1_controller_sigterm.log'), timed_out=True,
        )
        assert category == FailureCategory.INFRA_TIMEOUT
