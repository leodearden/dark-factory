"""VerifyResult.failure_report never silently drops a non-empty leg output (task 5357).

A failed leg whose output matches no failure marker used to vanish from the
report, leaving the debugger with nothing but the one-line summary. These tests
pin that every non-empty leg output reaches the report, and that a section
whose output matched no marker is headed and labelled as such rather than
claiming failures nobody found.
"""

import pytest

from orchestrator.verify import VerifyResult

_NO_MARKER_NOTE = 'No failure marker matched in this output'


def _result(**overrides) -> VerifyResult:
    fields = {
        'passed': False,
        'test_output': '',
        'lint_output': '',
        'type_output': '',
        'summary': 'Failures: tests failed',
        'test_leg_failed': True,
    }
    fields.update(overrides)
    return VerifyResult(**fields)


def _section(report: str, heading: str) -> str:
    return report.split(heading, 1)[1].split('\n## ', 1)[0]


class TestNoMarkerOutputIsEmittedLabelled:
    def test_pytest_error_tally_without_failed_lines_is_emitted(self):
        tally = '19639 passed, 8 warnings, 8 errors in 2351.14s (0:39:11)'
        test_output = (
            '....\n'
            'ERROR orchestrator/tests/test_x.py::test_y - RuntimeError: boom\n'
            f'{tally}\n'
        )
        report = _result(test_output=test_output, cause_hint=tally).failure_report()

        assert '## Test Output' in report
        section = _section(report, '## Test Output')
        assert _NO_MARKER_NOTE in section
        assert 'test_y' in section
        assert '8 errors' in section
        assert '## Test Failures' not in report

    def test_clock_stop_timeout_output_follows_the_timeout_section(self):
        test_output = (
            'Command clock-stop timed out (wall-clock budget (3600s), '
            'wall time 3600.0s): uv run pytest -q tests/'
        )
        report = _result(
            test_output=test_output, timed_out=True, category='infra_timeout',
        ).failure_report()

        assert '## Verify Timed Out' in report
        assert '## Test Output' in report
        assert report.index('## Verify Timed Out') < report.index('## Test Output')
        assert 'clock-stop timed out' in _section(report, '## Test Output')

    def test_marker_output_keeps_the_test_failures_heading(self):
        report = _result(
            test_output='FAILED tests/test_x.py::test_real - AssertionError\n',
        ).failure_report()

        assert '## Test Failures' in report
        assert _NO_MARKER_NOTE not in report

    def test_empty_or_blank_test_output_falls_back_to_summary(self):
        for blank in ('', '   \n'):
            result = _result(test_output=blank)
            assert result.failure_report() == result.summary

    def test_type_output_without_error_is_emitted_labelled(self):
        report = _result(
            type_output='Command timed out after 600s: pyright',
        ).failure_report()

        assert '## Type Check Output' in report
        assert _NO_MARKER_NOTE in _section(report, '## Type Check Output')
        assert '## Type Errors' not in report

    def test_green_test_leg_output_is_not_shown_beside_a_lint_failure(self):
        report = _result(
            test_output='....\n19639 passed in 2351s\n',
            lint_output='x.py:1:1: F401 unused import',
            test_leg_failed=False,
            category='lint_failure',
            summary='Failures: lint issues',
        ).failure_report()

        assert '## Lint Issues' in report
        assert '## Test Output' not in report
        assert _NO_MARKER_NOTE not in report

    def test_green_test_leg_output_is_not_shown_beside_a_type_failure(self):
        report = _result(
            test_output='....\n19639 passed in 2351s\n',
            type_output='error: type mismatch',
            test_leg_failed=False,
            category='type_error',
            summary='Failures: type errors',
        ).failure_report()

        assert '## Type Errors' in report
        assert '## Test Output' not in report

    def test_a_result_that_never_recorded_its_test_leg_shows_no_marker_less_test_output(self):
        result = _result(test_output='....\n19639 passed in 2351s\n', test_leg_failed=False)

        assert result.failure_report() == result.summary

    def test_type_output_with_error_keeps_the_type_errors_heading(self):
        report = _result(type_output='error: type mismatch').failure_report()

        assert '## Type Errors' in report


def _long_lint_output() -> str:
    return '\n'.join(f'orchestrator/src/x.py:{i}:1: F401 unused import' for i in range(200))


def _type_output_with_early_traceback() -> str:
    head = (
        'error: pyright crashed\n'
        'Traceback (most recent call last):\n'
        '  File "x.py", line 1\n'
        'RuntimeError: boom\n'
    )
    filler = '\n'.join(f'  /abs/path/f{i}.py:1:1 - warning: w' for i in range(300))
    return head + filler + '\n1 error, 300 warnings'


def _test_output_with_buried_failure() -> str:
    pass_lines = [f'tests/test_mod.py::test_pass_{i} PASSED' for i in range(200)]
    buried = 'FAILED tests/pkg/test_mod.py::test_buried - AssertionError'
    return '\n'.join(pass_lines[:20] + [buried] + pass_lines[20:])


def _test_output_without_marker() -> str:
    pass_lines = [f'tests/test_mod.py::test_pass_{i} PASSED' for i in range(200)]
    return '\n'.join(pass_lines + ['200 passed, 3 errors in 12.0s'])


def _truncation_line(section: str) -> str:
    return next((line for line in section.split('\n') if line.startswith('Excerpt:')), '')


_TRUNCATED_SECTIONS = [
    ({'lint_output': _long_lint_output()}, '## Lint Issues'),
    ({'type_output': _type_output_with_early_traceback()}, '## Type Errors'),
    ({'test_output': _test_output_with_buried_failure()}, '## Test Failures'),
    ({'test_output': _test_output_without_marker()}, '## Test Output'),
]


class TestTruncationIsLabelledAndAnchored:
    def test_long_lint_output_is_labelled_as_cut_and_keeps_its_last_line(self):
        lint_output = _long_lint_output()
        section = _section(_result(lint_output=lint_output).failure_report(), '## Lint Issues')

        assert f'of {len(lint_output):,} characters' in section
        assert lint_output.split('\n')[-1] in section

    def test_short_lint_output_is_byte_identical(self):
        lint_output = 'x.py:1:1: F401 unused import\nx.py:2:1: F401 unused import'
        report = _result(lint_output=lint_output).failure_report()

        assert '## Lint Issues' + _section(report, '## Lint Issues') == (
            '## Lint Issues\n\n```\n' + lint_output + '\n```'
        )

    def test_type_errors_are_anchored_on_the_traceback_not_blind_tailed(self):
        type_output = _type_output_with_early_traceback()
        section = _section(_result(type_output=type_output).failure_report(), '## Type Errors')

        assert 'Traceback (most recent call last)' in section
        assert f'of {len(type_output):,} characters' in section

    def test_cut_test_failures_section_is_labelled(self):
        test_output = _test_output_with_buried_failure()
        section = _section(_result(test_output=test_output).failure_report(), '## Test Failures')

        assert 'test_buried' in section
        assert f'of {len(test_output):,} characters' in section

    def test_cut_test_output_section_is_labelled_and_keeps_its_tail(self):
        test_output = _test_output_without_marker()
        section = _section(_result(test_output=test_output).failure_report(), '## Test Output')

        assert _NO_MARKER_NOTE in section
        assert '200 passed, 3 errors in 12.0s' in section
        assert f'of {len(test_output):,} characters' in section

    def test_truncation_line_points_at_verify_logs_only_when_listed(self):
        lint_output = _long_lint_output()
        with_logs = _result(
            lint_output=lint_output,
            worktree_log_paths=['/wt/.task/verify/attempt-1.lint.log'],
        ).failure_report()
        without_logs = _result(lint_output=lint_output).failure_report()

        assert 'Verify Logs' in _truncation_line(_section(with_logs, '## Lint Issues'))
        assert 'Verify Logs' not in _section(without_logs, '## Lint Issues')

    @pytest.mark.parametrize(
        ('streams', 'heading'), _TRUNCATED_SECTIONS,
        ids=['lint', 'type_errors', 'test_failures', 'test_output'],
    )
    def test_annotations_follow_the_closing_fence(self, streams, heading):
        section = _section(_result(**streams).failure_report(), heading)
        lines = section.split('\n')
        last_fence = max(i for i, line in enumerate(lines) if line == '```')
        excerpt_lines = [i for i, line in enumerate(lines) if line.startswith('Excerpt:')]

        assert section.startswith('\n\n```\n')
        assert excerpt_lines, f'no truncation line in {heading}'
        assert excerpt_lines[0] > last_fence


class TestAggregateKeepsTheTestLegVerdict:
    def test_a_failed_test_leg_in_any_module_marks_the_aggregate(self):
        from orchestrator.verify import _aggregate_results

        green = _result(test_output='5 passed', test_leg_failed=False, lint_output='x: F401')
        red = _result(test_output='boom\n1 error', test_leg_failed=True)

        assert _aggregate_results([green, red]).test_leg_failed is True
        assert _aggregate_results([green, green]).test_leg_failed is False
