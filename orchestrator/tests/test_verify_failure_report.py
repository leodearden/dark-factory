"""VerifyResult.failure_report never silently drops a non-empty leg output (task 5357).

A failed leg whose output matches no failure marker used to vanish from the
report, leaving the debugger with nothing but the one-line summary. These tests
pin that every non-empty leg output reaches the report, and that a section
whose output matched no marker is headed and labelled as such rather than
claiming failures nobody found.
"""

from orchestrator.verify import VerifyResult

_NO_MARKER_NOTE = 'No failure marker matched in this output'


def _result(**overrides) -> VerifyResult:
    fields = {
        'passed': False,
        'test_output': '',
        'lint_output': '',
        'type_output': '',
        'summary': 'Failures: tests failed',
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

    def test_type_output_with_error_keeps_the_type_errors_heading(self):
        report = _result(type_output='error: type mismatch').failure_report()

        assert '## Type Errors' in report
