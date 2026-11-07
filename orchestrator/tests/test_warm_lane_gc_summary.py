"""Unit tests for ``orchestrator.warm_lane_gc_summary`` (task 7857).

The module turns one ``warm-lane-gc.sh reclaim`` run's stdout into a parsed
summary and the orchestrator arm's per-pass log record.  Both script copies
run across the flip, so every case here is spelled out for both field sets.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

import pytest

from orchestrator.warm_lane_gc_summary import (
    ReclaimSummary,
    ReclaimSummaryError,
    log_reclaim_outcome,
    parse_reclaim_summary,
)

DF_LINE = (
    'reclaim: reset=3 removed=1 preserved=5 preserved_live_ref=2 '
    'preserved_assigned=4 downgraded_assigned=0 reset_cold=1 skipped_unchanged=7'
)
REIFY_LINE = 'reclaim: reset=2 removed=0 preserved=6 preserved_live_ref=1 reset_cold=1'
REIFY_9611_LINE = REIFY_LINE + ' skipped_unchanged=3'


def _expected_fields(line: str) -> dict[str, int]:
    body = line.removeprefix('reclaim: ')
    return {name: int(value) for name, value in (t.split('=') for t in body.split())}


def _counts(summary: ReclaimSummary) -> dict[str, int]:
    return dict(summary.fields)


class TestParseAcceptsEitherCopy:
    @pytest.mark.parametrize('line', [DF_LINE, REIFY_LINE, REIFY_9611_LINE])
    def test_every_field_is_readable_by_name(self, line: str) -> None:
        expected = _expected_fields(line)

        summary = parse_reclaim_summary(line + '\n')

        assert _counts(summary) == expected

    def test_parser_keys_on_names_not_positions(self) -> None:
        summary = parse_reclaim_summary(
            'reclaim: preserved=5 reset_cold=1 removed=1 reset=3',
        )

        assert _counts(summary) == {'reset': 3, 'removed': 1, 'preserved': 5, 'reset_cold': 1}

    def test_a_field_the_copy_does_not_emit_is_absent(self) -> None:
        summary = parse_reclaim_summary(REIFY_LINE)

        assert 'preserved_assigned' not in _counts(summary)

    def test_an_appended_field_name_may_carry_digits(self) -> None:
        summary = parse_reclaim_summary(DF_LINE + ' reset_v2=4')

        assert _counts(summary)['reset_v2'] == 4

    @pytest.mark.parametrize('line', [DF_LINE, REIFY_LINE, REIFY_9611_LINE])
    def test_render_reproduces_the_body_in_emitted_order(self, line: str) -> None:
        assert parse_reclaim_summary(line).render() == line.removeprefix('reclaim: ')

    def test_other_stdout_lines_and_blank_lines_are_ignored(self) -> None:
        stdout = f'\nHEADROOM lanes=1\n\n{DF_LINE}\ntrailing chatter\n\n'

        summary = parse_reclaim_summary(stdout)

        assert summary.render() == DF_LINE.removeprefix('reclaim: ')


class TestParseRejectsAnUncountableSummary:
    @pytest.mark.parametrize('stdout', ['', '\n', 'HEADROOM lanes=1\n'])
    def test_no_summary_line_names_the_prefix(self, stdout: str) -> None:
        with pytest.raises(ReclaimSummaryError, match='reclaim:'):
            parse_reclaim_summary(stdout)

    def test_two_summary_lines_are_rejected(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='found 2'):
            parse_reclaim_summary(f'{DF_LINE}\n{REIFY_LINE}\n')

    def test_a_missing_required_field_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='preserved'):
            parse_reclaim_summary('reclaim: reset=1 removed=0')

    @pytest.mark.parametrize('token', ['reset=x', 'reset', 'reset=-1', 'Reset=1', '2reset=1'])
    def test_a_malformed_token_is_named(self, token: str) -> None:
        line = f'reclaim: {token} removed=0 preserved=0'

        with pytest.raises(ReclaimSummaryError, match=f"'{token}'"):
            parse_reclaim_summary(line)

    def test_a_duplicate_field_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='reset'):
            parse_reclaim_summary('reclaim: reset=1 reset=2 removed=0 preserved=0')


class TestReclaimSummaryEnforcesItsInvariantAtConstruction:
    def test_a_missing_required_field_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='preserved'):
            ReclaimSummary(fields=(('reset', 1), ('removed', 0)))

    def test_a_negative_value_names_the_field(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='removed'):
            ReclaimSummary(fields=(('reset', 1), ('removed', -1), ('preserved', 0)))

    def test_a_bool_is_not_a_count(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='reset'):
            ReclaimSummary(fields=(('reset', True), ('removed', 0), ('preserved', 0)))

    def test_a_duplicate_field_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='reset'):
            ReclaimSummary(
                fields=(('reset', 1), ('reset', 2), ('removed', 0), ('preserved', 0)),
            )

    def test_a_malformed_name_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='Bad-Name'):
            ReclaimSummary(
                fields=(('reset', 1), ('removed', 0), ('preserved', 0), ('Bad-Name', 1)),
            )

    def test_a_name_with_digits_after_the_first_letter_is_accepted(self) -> None:
        summary = ReclaimSummary(
            fields=(('reset', 1), ('removed', 0), ('preserved', 0), ('reset_v2', 1)),
        )

        assert _counts(summary)['reset_v2'] == 1

    def test_the_error_is_a_value_error(self) -> None:
        assert issubclass(ReclaimSummaryError, ValueError)

    def test_the_summary_is_frozen(self) -> None:
        summary = parse_reclaim_summary(DF_LINE)

        pytest.raises(dataclasses.FrozenInstanceError, setattr, summary, 'fields', ())


_LOGGER = 'orchestrator.warm_lane_gc_summary'
_CONTEXT = 'some_reclaim_caller'
_PASS_RECORD = f'{_CONTEXT}: reclaim pass'
_SCRIPT = Path('/x/warm-lane-gc.sh')


def _kv(message: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)=(\S+)', message))


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno == level
    ]


def _log(caplog: pytest.LogCaptureFixture, *, rc: int, stdout: str, stderr: str = '',
         origin: str = 'dark-factory', duration_secs: float = 1.25) -> None:
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        log_reclaim_outcome(
            context=_CONTEXT, rc=rc, stdout=stdout, stderr=stderr, script=_SCRIPT,
            origin=origin, duration_secs=duration_secs,
        )


class TestLogReclaimOutcomePassRecord:
    @pytest.mark.parametrize(
        ('origin', 'line'), [('dark-factory', DF_LINE), ('project', REIFY_LINE)],
    )
    def test_one_info_record_carries_origin_script_duration_and_every_field(
        self, caplog: pytest.LogCaptureFixture, origin: str, line: str,
    ) -> None:
        _log(caplog, rc=0, stdout=line + '\n', origin=origin)

        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1, infos
        assert infos[0].startswith(_PASS_RECORD)
        fields = _kv(infos[0])
        assert fields['origin'] == origin
        assert fields['script'] == str(_SCRIPT)
        assert float(fields['duration_s']) == 1.25
        for name, value in _expected_fields(line).items():
            assert fields[name] == str(value)
        assert _records(caplog, logging.WARNING) == []

    def test_a_reify_pass_carries_no_record_gate_field(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _log(caplog, rc=0, stdout=REIFY_LINE, origin='project')

        (info,) = _records(caplog, logging.INFO)
        assert 'preserved_assigned' not in _kv(info)


class TestLogReclaimOutcomeUncountedPass:
    def test_rc0_without_a_summary_warns_uncounted(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _log(caplog, rc=0, stdout='HEADROOM lanes=1\n', origin='project')

        assert _records(caplog, logging.INFO) == []
        (warning,) = _records(caplog, logging.WARNING)
        assert warning.startswith(f'{_CONTEXT}: ')
        assert str(_SCRIPT) in warning
        assert 'project' in warning
        assert 'uncounted' in warning
        assert 'seed script' not in warning

    def test_the_uncounted_warning_quotes_stdout_once(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _log(caplog, rc=0, stdout='HEADROOM lanes=1\n')

        (warning,) = _records(caplog, logging.WARNING)
        assert warning.count('HEADROOM lanes=1') == 1, warning

    def test_rc0_with_a_malformed_summary_names_the_missing_field(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _log(caplog, rc=0, stdout='reclaim: reset=1 removed=0\n')

        assert _records(caplog, logging.INFO) == []
        (warning,) = _records(caplog, logging.WARNING)
        assert 'preserved' in warning
        assert 'uncounted' in warning


class TestLogReclaimOutcomeScriptFailure:
    def test_nonzero_rc_warns_with_stderr_and_records_no_pass(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _log(caplog, rc=2, stdout=DF_LINE + '\n', stderr='boom')

        assert _records(caplog, logging.INFO) == []
        (warning,) = _records(caplog, logging.WARNING)
        assert warning.startswith(f'{_CONTEXT}: script exited 2')
        assert 'boom' in warning
