"""The C1 consumer-side contract: pending_age_secs(task, now) = max(0, now - parse(pending_since))."""

from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.pending_age import AnchorStatus, pending_age_secs, read_pending_age

NOW = datetime(2026, 7, 16, 21, 0, 0, tzinfo=UTC)


def _task(pending_since: object) -> dict:
    return {'id': '1', 'metadata': {'pending_since': pending_since}}


class TestPresentAnchor:
    def test_live_ms_precision_z_anchor_reads_its_age(self):
        reading = read_pending_age(_task('2026-07-16T20:49:48.531Z'), NOW)

        assert reading.status is AnchorStatus.PRESENT
        assert reading.age_secs == pytest.approx(611.469)

    def test_explicit_utc_offset_reads_the_same_as_z(self):
        anchor = NOW - timedelta(seconds=3600)

        z_reading = read_pending_age(_task(anchor.strftime('%Y-%m-%dT%H:%M:%SZ')), NOW)
        offset_reading = read_pending_age(_task(anchor.isoformat()), NOW)

        assert z_reading.age_secs == 3600.0
        assert offset_reading.age_secs == 3600.0
        assert offset_reading.status is AnchorStatus.PRESENT

    def test_non_utc_offset_yields_the_absolute_age(self):
        reading = read_pending_age(_task('2026-07-16T22:00:00+02:00'), NOW)

        assert reading.status is AnchorStatus.PRESENT
        assert reading.age_secs == 3600.0

    def test_naive_anchor_is_read_as_utc(self):
        reading = read_pending_age(_task('2026-07-16T20:00:00'), NOW)

        assert reading.status is AnchorStatus.PRESENT
        assert reading.age_secs == 3600.0

    def test_future_anchor_clamps_to_zero_but_stays_covered(self):
        reading = read_pending_age(_task('2026-07-16T21:00:05Z'), NOW)

        assert reading.age_secs == 0.0
        assert reading.status is AnchorStatus.PRESENT
        assert reading.covered


ABSENT_TASKS = {
    'key_missing': {'id': '1', 'metadata': {}},
    'value_none': _task(None),
    'value_empty': _task(''),
    'value_whitespace': _task('   '),
    'metadata_missing': {'id': '1'},
    'metadata_none': {'id': '1', 'metadata': None},
}

MALFORMED_TASKS = {
    'unparseable_string': _task('not-a-date'),
    'integer_value': _task(1723000000),
    'list_value': _task(['x']),
}


@pytest.mark.parametrize('task', ABSENT_TASKS.values(), ids=ABSENT_TASKS.keys())
def test_absent_anchor_reads_zero_age(task):
    reading = read_pending_age(task, NOW)

    assert reading.status is AnchorStatus.ABSENT
    assert reading.age_secs == 0.0
    assert not reading.covered


@pytest.mark.parametrize('task', MALFORMED_TASKS.values(), ids=MALFORMED_TASKS.keys())
def test_malformed_anchor_reads_zero_age_without_raising(task):
    reading = read_pending_age(task, NOW)

    assert reading.status is AnchorStatus.MALFORMED
    assert reading.age_secs == 0.0
    assert not reading.covered


ALL_TASKS = {
    'present': _task('2026-07-16T20:49:48.531Z'),
    'future': _task('2026-07-16T21:00:05Z'),
    'naive': _task('2026-07-16T20:00:00'),
    **ABSENT_TASKS,
    **MALFORMED_TASKS,
}


@pytest.mark.parametrize('task', ALL_TASKS.values(), ids=ALL_TASKS.keys())
def test_pending_age_secs_is_the_readings_age(task):
    assert pending_age_secs(task, NOW) == read_pending_age(task, NOW).age_secs


@pytest.mark.parametrize('task', ALL_TASKS.values(), ids=ALL_TASKS.keys())
def test_covered_exactly_when_present(task):
    reading = read_pending_age(task, NOW)

    assert reading.covered is (reading.status is AnchorStatus.PRESENT)
