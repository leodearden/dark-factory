"""Unit tests for ``orchestrator.warm_lane_gc_summary`` (task 7857).

The module turns one ``warm-lane-gc.sh reclaim`` run's stdout into a parsed
summary and the orchestrator arm's per-pass log record.  Both script copies
run across the flip, so every case here is spelled out for both field sets.
"""
from __future__ import annotations

import dataclasses

import pytest

from orchestrator.warm_lane_gc_summary import (
    ReclaimSummary,
    ReclaimSummaryError,
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


class TestParseAcceptsEitherCopy:
    @pytest.mark.parametrize('line', [DF_LINE, REIFY_LINE, REIFY_9611_LINE])
    def test_every_field_is_readable_by_name(self, line: str) -> None:
        expected = _expected_fields(line)

        summary = parse_reclaim_summary(line + '\n')

        assert summary.reset == expected['reset']
        assert summary.removed == expected['removed']
        assert summary.preserved == expected['preserved']
        for name, value in expected.items():
            assert summary.count(name) == value

    def test_parser_keys_on_names_not_positions(self) -> None:
        summary = parse_reclaim_summary(
            'reclaim: preserved=5 reset_cold=1 removed=1 reset=3',
        )

        assert (summary.reset, summary.removed, summary.preserved) == (3, 1, 5)
        assert summary.count('reset_cold') == 1

    def test_a_field_the_copy_does_not_emit_counts_as_none(self) -> None:
        summary = parse_reclaim_summary(REIFY_LINE)

        assert summary.count('preserved_assigned') is None

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
        with pytest.raises(ReclaimSummaryError, match='2'):
            parse_reclaim_summary(f'{DF_LINE}\n{REIFY_LINE}\n')

    def test_a_missing_required_field_is_named(self) -> None:
        with pytest.raises(ReclaimSummaryError, match='preserved'):
            parse_reclaim_summary('reclaim: reset=1 removed=0')

    @pytest.mark.parametrize('token', ['reset=x', 'reset', 'reset=-1', 'Reset=1'])
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

    def test_the_error_is_a_value_error(self) -> None:
        assert issubclass(ReclaimSummaryError, ValueError)

    def test_the_summary_is_frozen(self) -> None:
        summary = parse_reclaim_summary(DF_LINE)

        pytest.raises(dataclasses.FrozenInstanceError, setattr, summary, 'fields', ())
