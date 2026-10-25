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
from orchestrator.verify import LegSlot, classify_leg
from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
from orchestrator.verify_classify import (
    is_external_kill_rc,
    is_shell_reported_external_kill,
)

_GREEN_LEG = (0, '')

_TEST_CMD = 'uv run pytest tests/'


def _lint_leg_category(lint_out: str, lint_cmd: str) -> FailureCategory:
    return classify_leg(LegSlot.LINT, 1, lint_out, False, lint_cmd)


def _type_leg_category(type_out: str, type_cmd: str) -> FailureCategory:
    return classify_leg(LegSlot.TYPE, 1, type_out, False, type_cmd)


async def _verify_legs(
    project_root: Path, *, test=_GREEN_LEG, lint=_GREEN_LEG, type_=_GREEN_LEG,
) -> verify.VerifyResult:
    """``run_verification`` over the live lint/type chains, each leg scripted
    as (rc, output) at the ``_run_cmd`` seam, none timed out."""
    config = OrchestratorConfig(
        project_root=project_root,
        test_command=_TEST_CMD,
        lint_command=ROOT_LINT_COMMAND,
        type_check_command=ROOT_TYPE_CHECK_COMMAND,
    )

    async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        # 'ruff check' first: the lint chain names df_pytest_isolation.py.
        for marker, (rc, output) in (
            ('ruff check', lint), ('pyright', type_), ('pytest', test),
        ):
            if marker in cmd:
                return rc, output, False
        return 0, '', False

    with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
        return await verify.run_verification(project_root, config)


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
    lint/type commands, proven through ``classify_leg`` with nothing
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

    @pytest.mark.asyncio
    async def test_human_text_lint_red_is_lint_error(self, tmp_path: Path):
        result = await _verify_legs(tmp_path, lint=(1, _RUFF_I001_OUTPUT))
        assert result.category != FailureCategory.UNKNOWN_TEST_FAILURE
        assert result.category == FailureCategory.LINT_ERROR
        assert result.summary == 'Failures: lint issues'
        assert result.failing_leg_categories == ['lint_error']

    @pytest.mark.asyncio
    async def test_human_text_type_red_is_type_error(self, tmp_path: Path):
        result = await _verify_legs(tmp_path, type_=(1, _PYRIGHT_TEXT_OUTPUT))
        assert result.category == FailureCategory.TYPE_ERROR
        assert result.summary == 'Failures: type errors'
        assert result.failing_leg_categories == ['type_error']

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
        category = classify_leg(LegSlot.LINT, -9, '', False, ROOT_LINT_COMMAND)
        assert category == FailureCategory.INFRA_KILL, (
            'only the unknown fall-through is relabelled'
        )

    def test_test_slot_is_never_relabelled(self):
        category = classify_leg(
            LegSlot.TEST, 5, 'no tests ran in 0.01s\n', False, _TEST_CMD,
        )
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

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


def _rewrite(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f'fixture drift: {old!r} is not unique'
    return text.replace(old, new)


# The group-kill log ends in its unterminated partial progress line.
_GROUP_KILL_TAIL = ' [ 18%]\n.......'

# pytest -v outcome lines, verbatim from pytest 9.0.3 / xdist 3.8.0 runs of
# `pytest -v` and `pytest -v -n 2` (2026-10-07). A parametrised node-id may
# carry a space.
_VERBOSE_SERIAL_FAILED = 'tests/test_a.py::test_bad FAILED'.ljust(73) + '[ 40%]'
_VERBOSE_SERIAL_ERROR = 'tests/test_a.py::test_setup_error ERROR'.ljust(73) + '[ 60%]'
_VERBOSE_SERIAL_PARAM_FAILED = 'tests/test_a.py::test_param[a b] FAILED'.ljust(73) + '[ 80%]'
_VERBOSE_SERIAL_PASSED = 'tests/test_a.py::test_ok PASSED'.ljust(73) + '[ 20%]'
_VERBOSE_XDIST_FAILED = '[gw0] [100%] FAILED tests/test_a.py::test_bad '
_VERBOSE_XDIST_ERROR = '[gw1] [ 40%] ERROR tests/test_a.py::test_setup_error '
_VERBOSE_XDIST_PARAM_FAILED = '[gw1] [ 60%] FAILED tests/test_a.py::test_param[a b] '
_VERBOSE_XDIST_PASSED = '[gw0] [ 20%] PASSED tests/test_a.py::test_ok '


def _xdist_test_leg(rc: int, output: str, *, timed_out: bool = False) -> FailureCategory:
    return classify_leg(LegSlot.TEST, rc, output, timed_out, _FUSED_MEMORY_TEST_CMD)


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
        category = _xdist_test_leg(
            143, _xdist_abort_log('shape1_controller_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_channel_collapse_signature_alone_is_infra_abort(self):
        """esc-4808-1's text with no kill rc: the text arm alone fires."""
        category = _xdist_test_leg(
            1, _xdist_abort_log('shape1_controller_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_group_sigterm_is_caught_by_the_kill_rc_alone(self):
        """esc-4808-1's kill delivered to the whole group leaves no
        signature, so only the rc arm can catch it."""
        category = _xdist_test_leg(
            143, _xdist_abort_log('shape1_group_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_group_sigterm_without_kill_rc_stays_unknown(self):
        """The discriminating twin of the case above: the same text with no
        kill rc carries neither arm's evidence."""
        category = _xdist_test_leg(
            1, _xdist_abort_log('shape1_group_sigterm.log'),
        )
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_green_tally_then_sigterm_is_infra_abort(self):
        """esc-4246-4: the session finished green, then the shell reported a
        SIGTERM. The log carries no signature; the kill rc decides."""
        category = _xdist_test_leg(
            143, _xdist_abort_log('shape2_green_tally.log'),
        )
        _assert_infra_abort(category)

    @pytest.mark.parametrize('rc', [143, 1])
    def test_bringup_sigterm_is_infra_abort_on_the_signature(self, rc):
        """esc-4071-1: its rc was never recorded, so the text signature
        alone must decide."""
        category = _xdist_test_leg(
            rc, _xdist_abort_log('shape3_bringup_sigterm.log'),
        )
        _assert_infra_abort(category)

    def test_failed_line_vetoes_the_abort(self):
        """Rule: a FAILED line is a verdict, so TEST_FAILURE survives."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            'FAILED tests/test_x.py::test_y - AssertionError',
        )
        category = _xdist_test_leg(143, output)
        assert category == FailureCategory.TEST_FAILURE

    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_killed_progress_failure_char_vetoes_the_abort(self, status_char):
        """Rule: a killed -q session prints no short summary, so an F or E
        in its progress line is the only failure evidence and must veto."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            _KILLED_PROGRESS_WITH_F.replace('F', status_char),
        )
        category = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('rc', [143, 1])
    def test_uv_forwarded_sigterm_is_infra_abort(self, rc):
        """Production shape: uv forwards the SIGTERM to the controller only,
        so the partial line is glued to a worker's warning path. The glued
        '........' carries no F/E, and the 8 'For more information see …'
        lines are prose that opens with F. Neither may veto."""
        output = _xdist_abort_log('shape1_uv_forwarded_sigterm.log')
        assert '\nFor more information see ' in output
        category = _xdist_test_leg(rc, output)
        _assert_infra_abort(category)

    @pytest.mark.parametrize('rc', [143, 1])
    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_before_glued_terminated_vetoes_the_abort(
        self, status_char, rc,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the unterminated final line, is
        the only failure evidence and must veto. Here bash's 'Terminated' is
        glued onto that line; rc=1 shows the veto also beats the text arm."""
        output = _rewrite(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            '\n..Terminated\n', f'\n..{status_char}.Terminated\n',
        )
        category = _xdist_test_leg(rc, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_opening_the_partial_line_vetoes_the_abort(
        self, status_char,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the very first char of the
        unterminated final line, is the only failure evidence and must veto."""
        output = _rewrite(
            _xdist_abort_log('shape1_controller_sigterm.log'),
            '\n..Terminated\n', f'\n{status_char}Terminated\n',
        )
        category = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('rc', [143, 1])
    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_in_bringup_partial_line_vetoes_the_abort(
        self, status_char, rc,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the unterminated final line, is
        the only failure evidence and must veto, on shape 3 as on shape 1."""
        output = _rewrite(
            _xdist_abort_log('shape3_bringup_sigterm.log'),
            '\n....Terminated\n', f'\n..{status_char}..Terminated\n',
        )
        category = _xdist_test_leg(rc, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('rc', [143, 1])
    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_glued_to_worker_warning_path_vetoes_the_abort(
        self, status_char, rc,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the unterminated final line, is
        the only failure evidence and must veto. In the production shape the
        glued suffix is a worker's warning path, not a bash word, so a
        closed list of bash job-status words would miss it."""
        output = _rewrite(
            _xdist_abort_log('shape1_uv_forwarded_sigterm.log'),
            '\n......../home/', f'\n...{status_char}..../home/',
        )
        category = _xdist_test_leg(rc, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_glued_to_bash_job_report_vetoes_the_abort(
        self, status_char,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the unterminated final line, is
        the only failure evidence and must veto. Here bash's pretty-print
        job report for a SIGHUP is glued onto that line (bash 5.2.21)."""
        text = _xdist_abort_log('shape1_group_sigterm.log')
        assert text.endswith(_GROUP_KILL_TAIL)
        output = _rewrite(
            text, _GROUP_KILL_TAIL,
            f' [ 18%]\n..{status_char}./bin/bash: line 1: 1854676 Hangup'
            '                  /x/python -m pytest -n 8 -q\n',
        )
        category = _xdist_test_leg(129, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('status_char', ['F', 'E'])
    def test_failure_char_in_partial_line_at_eof_vetoes_the_abort(
        self, status_char,
    ):
        """Rule: a killed -q session prints no short summary, so an F or E
        anywhere in its progress, including the unterminated final line, is
        the only failure evidence and must veto. Here nothing is glued on:
        the partial line simply ends at EOF."""
        text = _xdist_abort_log('shape1_group_sigterm.log')
        assert text.endswith(_GROUP_KILL_TAIL)
        output = _rewrite(text, _GROUP_KILL_TAIL, f' [ 18%]\n...{status_char}...')
        category = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('rc', [143, 1])
    @pytest.mark.parametrize(
        'verbose_line',
        [
            pytest.param(_VERBOSE_SERIAL_FAILED, id='serial_failed'),
            pytest.param(_VERBOSE_SERIAL_ERROR, id='serial_error'),
            pytest.param(_VERBOSE_SERIAL_PARAM_FAILED, id='serial_param_with_space'),
            pytest.param(_VERBOSE_XDIST_FAILED, id='xdist_failed'),
            pytest.param(_VERBOSE_XDIST_ERROR, id='xdist_error'),
            pytest.param(_VERBOSE_XDIST_PARAM_FAILED, id='xdist_param_with_space'),
        ],
    )
    def test_verbose_failure_outcome_vetoes_the_abort(self, verbose_line, rc):
        """Rule: under -v each finished test prints its outcome on its own
        line, and a killed session prints no short summary, so a FAILED or
        ERROR outcome there is failure evidence and must veto."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'), verbose_line,
        )
        category = _xdist_test_leg(rc, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize(
        'verbose_line',
        [
            pytest.param(_VERBOSE_SERIAL_PASSED, id='serial_passed'),
            pytest.param(_VERBOSE_XDIST_PASSED, id='xdist_passed'),
        ],
    )
    def test_verbose_passed_outcome_does_not_veto(self, verbose_line):
        """The discriminating twin: a PASSED outcome is no failure evidence."""
        output = _with_line(
            _xdist_abort_log('shape1_controller_sigterm.log'), verbose_line,
        )
        category = _xdist_test_leg(143, output)
        _assert_infra_abort(category)

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
        category = _xdist_test_leg(143, output)
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
        category = _xdist_test_leg(143, output)
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_internalerror_is_decided_by_the_classifier_first(self):
        """Rule: only the unknown fall-through is rewritten, so the
        classifier's PYTEST_INTERNALERROR wins over both arms."""
        output = _with_line(
            _xdist_abort_log('shape3_bringup_sigterm.log'),
            'INTERNALERROR> Traceback (most recent call last):',
        )
        category = _xdist_test_leg(143, output)
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
        category = _xdist_test_leg(rc, _xdist_abort_log('shape2_green_tally.log'))
        assert category == FailureCategory.UNKNOWN_TEST_FAILURE, why

    def test_raw_negative_returncode_stays_infra_kill(self):
        """Rule: the structural external-kill arm is unchanged and still
        dominates."""
        category = _xdist_test_leg(-15, _xdist_abort_log('shape2_green_tally.log'))
        assert category == FailureCategory.INFRA_KILL

    def test_lint_leg_is_out_of_the_abort_scope(self):
        """Rule: the abort arms are PYTEST-scoped, because their veto set is
        a pytest-output veto set. A killed lint leg reports its gate; this
        pins the documented residual."""
        category = classify_leg(
            LegSlot.LINT, 143, _xdist_abort_log('shape1_controller_sigterm.log'),
            False, ROOT_LINT_COMMAND,
        )
        assert category == FailureCategory.LINT_ERROR

    def test_timed_out_leg_stays_infra_timeout(self):
        """Rule: the classifier's timeout guard is unchanged and decides
        first."""
        category = _xdist_test_leg(
            143, _xdist_abort_log('shape1_controller_sigterm.log'), timed_out=True,
        )
        assert category == FailureCategory.INFRA_TIMEOUT


_TEST_FAILURE_OUTPUT = 'FAILED tests/test_x.py::test_y - assert 1 == 2\n'


class TestTheWorstLegDecidesTheRun:
    """The run-wide category is the most severe failing leg's, and a leg's
    branch verdict is never masked by ANOTHER leg's infra-transient category
    (task #4103). ``failing_leg_categories`` keeps every failing leg's own
    category, in test/lint/type order."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('legs', 'expected', 'failing'),
        [
            pytest.param(
                {'test': (1, _TEST_FAILURE_OUTPUT), 'lint': (1, _RUFF_I001_OUTPUT)},
                'test_failure', ['test_failure', 'lint_error'],
                id='test_failure_outranks_lint_error',
            ),
            pytest.param(
                {'lint': (1, _RUFF_I001_OUTPUT), 'type_': (1, _PYRIGHT_TEXT_OUTPUT)},
                'type_error', ['lint_error', 'type_error'],
                id='type_error_outranks_lint_error',
            ),
            pytest.param(
                {
                    'test': (143, _xdist_abort_log('shape1_controller_sigterm.log')),
                    'lint': (1, _RUFF_I001_OUTPUT),
                },
                'lint_error', ['infra_abort', 'lint_error'],
                id='lint_error_outranks_infra_abort',
            ),
            pytest.param(
                {
                    'test': (5, 'no tests ran in 0.01s\n'),
                    'type_': (1, _PYRIGHT_INTERPRETER_MISRESOLVED_OUTPUT),
                },
                'unknown_test_failure', ['unknown_test_failure', 'env_transient'],
                id='unknown_test_failure_outranks_env_transient',
            ),
        ],
    )
    async def test_worst_leg_decides(self, tmp_path: Path, legs, expected, failing):
        result = await _verify_legs(tmp_path, **legs)
        assert result.category == expected
        assert result.failing_leg_categories == failing
