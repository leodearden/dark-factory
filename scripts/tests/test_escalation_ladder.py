"""Tests for scripts/escalation_ladder.py.

Hermetic: every test runs against JSON records written into tmp_path, shaped
like the live store's (data/escalations/archive/2026-09-26/esc-4530-6.json),
never against the live escalations directory the script defaults to.
"""
import json
from datetime import UTC, datetime

import escalation_ladder
import pytest

T0 = datetime(2026, 9, 12, 6, 43, 16, tzinfo=UTC)


def _record(esc_id, **overrides):
    """A live-shaped escalation record, reduced to the keys the ladder reads."""
    task_id = esc_id.removeprefix('esc-').rsplit('-', 1)[0]
    return {
        'id': esc_id,
        'task_id': task_id,
        'level': 0,
        'agent_role': 'implementer',
        'status': 'pending',
        'timestamp': T0.isoformat(),
        'resolved_at': None,
        'resolved_by': None,
        'resolution_action': None,
        **overrides,
    }


def _write(directory, record, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{record['id']}.json")
    path.write_text(record if isinstance(record, str) else json.dumps(record))
    return path


def test_root_and_archive_records_are_both_loaded(tmp_path):
    _write(tmp_path, _record('esc-4377-1'))
    _write(tmp_path / 'archive' / '2026-09-20', _record('esc-4377-2'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert set(corpus.records) == {'esc-4377-1', 'esc-4377-2'}
    assert corpus.get('esc-4377-2').task_id == '4377'


def test_the_root_copy_wins_over_an_archive_copy_of_the_same_escalation(tmp_path):
    """Mirrors escalation/src/escalation/queue.py::iter_all_escalation_paths."""
    _write(tmp_path, _record('esc-4377-1', status='dismissed'))
    _write(tmp_path / 'archive' / '2026-09-20', _record('esc-4377-1', status='pending'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.get('esc-4377-1').status == 'dismissed'
    assert corpus.skipped == 0


def test_an_archive_only_duplicate_under_two_dates_is_counted_once(tmp_path):
    _write(tmp_path / 'archive' / '2026-09-20', _record('esc-4377-1', status='pending'))
    _write(tmp_path / 'archive' / '2026-09-22', _record('esc-4377-1', status='resolved'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert list(corpus.records) == ['esc-4377-1']
    assert corpus.skipped == 0


@pytest.mark.parametrize(
    ('name', 'content'),
    [
        ('esc-1-1.json', 'not json{'),
        ('esc-1-2.json', '[1, 2]'),
        ('esc-1-3.json', json.dumps({'task_id': '1', 'timestamp': T0.isoformat()})),
        ('esc-1-4.json', json.dumps(_record('esc-1-4', level='zero'))),
    ],
    ids=['non-json', 'json-array', 'no-id', 'non-int-level'],
)
def test_an_unusable_file_is_skipped_and_counted_rather_than_raised(tmp_path, name, content):
    _write(tmp_path, content, name=name)
    _write(tmp_path, _record('esc-4377-1'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert list(corpus.records) == ['esc-4377-1']
    assert corpus.skipped == 1


def test_timestamps_are_parsed_to_aware_utc_datetimes(tmp_path):
    _write(tmp_path, _record(
        'esc-4377-1',
        timestamp='2026-09-12T07:43:16+01:00',
        resolved_at='2026-09-12T08:00:00',  # naive reads as UTC
    ))

    record = escalation_ladder.load_escalation_corpus(tmp_path).get('esc-4377-1')

    assert record.timestamp == T0
    assert record.resolved_at == datetime(2026, 9, 12, 8, 0, tzinfo=UTC)


def test_an_unparseable_timestamp_skips_the_record(tmp_path):
    _write(tmp_path, _record('esc-4377-1', timestamp='yesterday'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.records == {}
    assert corpus.skipped == 1


@pytest.mark.parametrize('resolved_at', ['not a time', None, 17])
def test_an_unparseable_or_absent_resolved_at_reads_as_none(tmp_path, resolved_at):
    _write(tmp_path, _record('esc-4377-1', resolved_at=resolved_at))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.get('esc-4377-1').resolved_at is None
    assert corpus.skipped == 0


def test_a_record_with_no_resolved_at_key_reads_as_none(tmp_path):
    record = _record('esc-4377-1')
    del record['resolved_at']
    _write(tmp_path, record)

    assert escalation_ladder.load_escalation_corpus(tmp_path).get('esc-4377-1').resolved_at is None


def test_the_oldest_archive_date_is_the_coverage_bound(tmp_path):
    """Records resolved before the oldest archive date have been pruned, so
    it bounds how far back an escalation-derived baseline can reach."""
    _write(tmp_path / 'archive' / '2026-08-28', _record('esc-1-1'))
    _write(tmp_path / 'archive' / '2026-08-27', _record('esc-1-2'))
    _write(tmp_path / 'archive' / '2026-09-26', _record('esc-1-3'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.oldest_archive_date == '2026-08-27'


def test_no_archive_means_no_coverage_bound(tmp_path):
    _write(tmp_path, _record('esc-4377-1'))

    assert escalation_ladder.load_escalation_corpus(tmp_path).oldest_archive_date is None


def test_a_missing_escalations_dir_raises_rather_than_reading_as_empty(tmp_path):
    """A typo'd path must not render every steward escalation as 'record
    missing' — a plausible-looking measurement of nothing."""
    with pytest.raises(FileNotFoundError):
        escalation_ladder.load_escalation_corpus(tmp_path / 'no-such-dir')


def test_non_escalation_files_at_the_root_are_ignored_not_skipped(tmp_path):
    _write(tmp_path, '# digest', name='afk-digest.md')
    _write(tmp_path, '{"state": 1}', name='b3-state.json')
    _write(tmp_path, '3', name='esc-2119.seq')
    _write(tmp_path, '', name='esc-2119.seq.json.lock')
    _write(tmp_path, _record('esc-4377-1'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert list(corpus.records) == ['esc-4377-1']
    assert corpus.skipped == 0
