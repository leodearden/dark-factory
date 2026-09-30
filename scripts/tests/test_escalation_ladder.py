"""Tests for scripts/escalation_ladder.py.

Hermetic: every test runs against JSON records written into tmp_path, shaped
like the live store's (data/escalations/archive/2026-09-26/esc-4530-6.json),
never against the live escalations directory the script defaults to.
"""
import json
from datetime import UTC, datetime, timedelta

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
    assert corpus.records['esc-4377-2'].task_id == '4377'


def test_the_root_copy_wins_over_an_archive_copy_of_the_same_escalation(tmp_path):
    """Mirrors escalation/src/escalation/queue.py::iter_all_escalation_paths."""
    _write(tmp_path, _record('esc-4377-1', status='dismissed'))
    _write(tmp_path / 'archive' / '2026-09-20', _record('esc-4377-1', status='pending'))

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.records['esc-4377-1'].status == 'dismissed'
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

    record = escalation_ladder.load_escalation_corpus(tmp_path).records['esc-4377-1']

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

    assert corpus.records['esc-4377-1'].resolved_at is None
    assert corpus.skipped == 0


def test_a_record_with_no_resolved_at_key_reads_as_none(tmp_path):
    record = _record('esc-4377-1')
    del record['resolved_at']
    _write(tmp_path, record)

    corpus = escalation_ladder.load_escalation_corpus(tmp_path)

    assert corpus.records['esc-4377-1'].resolved_at is None


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


# --- steward dispositions: how a steward-worked L0 left the ladder ---

Disposition = escalation_ladder.StewardDisposition
AS_OF = T0 + timedelta(days=14)


def _esc(esc_id, **overrides):
    """An in-memory EscalationRecord; resolved in place by default."""
    fields = {
        'id': esc_id,
        'task_id': esc_id.removeprefix('esc-').rsplit('-', 1)[0],
        'level': 0,
        'agent_role': 'implementer',
        'status': 'dismissed',
        'timestamp': T0,
        'resolved_at': T0 + timedelta(hours=1),
        'resolved_by': 'claude-task-4377-steward',
        'resolution_action': 'close_only',
        **overrides,
    }
    return escalation_ladder.EscalationRecord(**fields)


def _corpus(*records):
    return escalation_ladder.EscalationCorpus(
        records={r.id: r for r in records}, skipped=0, oldest_archive_date=None,
    )


def _steward_l1(esc_id, *, before_resolution, task_id='4377', agent_role='steward'):
    """A level-1 record stamped *before_resolution* ahead of the L0's resolved_at."""
    return _esc(
        esc_id, task_id=task_id, level=1, agent_role=agent_role, status='pending',
        timestamp=T0 + timedelta(hours=1) - before_resolution,
        resolved_at=None, resolved_by=None, resolution_action=None,
    )


def _classify(corpus, esc_id='esc-4377-10'):
    return escalation_ladder.classify_steward_disposition(esc_id, corpus, as_of=AS_OF)


def test_an_id_not_in_the_corpus_is_record_missing():
    assert _classify(_corpus()) is Disposition.RECORD_MISSING


@pytest.mark.parametrize(
    'overrides',
    [
        {'status': 'pending', 'resolved_at': None, 'resolved_by': None},
        {'resolved_at': None},
        {'resolved_at': AS_OF},                        # resolved AT the window end
        {'resolved_at': AS_OF + timedelta(hours=1)},   # resolved after it
    ],
    ids=['status-pending', 'no-resolved-at', 'resolved-at-as-of', 'resolved-after-as-of'],
)
def test_an_escalation_unresolved_as_of_the_window_end_is_pending(overrides):
    assert _classify(_corpus(_esc('esc-4377-10', **overrides))) is Disposition.PENDING


def test_an_adjacent_steward_l1_on_the_same_task_is_a_promotion():
    """steward.py::_auto_escalate_to_human files the L1 and then dismisses the
    L0 with resolved_by='steward' — the same attribution an in-place close
    that named no agent gets, so the adjacent L1 is the only witness."""
    corpus = _corpus(
        _esc('esc-4377-10', resolved_by='steward'),
        _steward_l1('esc-4377-11', before_resolution=timedelta(milliseconds=8)),
    )

    assert _classify(corpus) is Disposition.PROMOTED_TO_L1


@pytest.mark.parametrize('resolved_by', ['claude-task-4377-steward', 'steward'])
def test_a_steward_close_with_no_adjacent_l1_is_resolved_in_place(resolved_by):
    corpus = _corpus(_esc('esc-4377-10', resolved_by=resolved_by))

    assert _classify(corpus) is Disposition.RESOLVED_IN_PLACE


def test_an_auto_dismissal_is_its_own_disposition():
    """Includes steward.py::_give_up_with_wip."""
    corpus = _corpus(_esc('esc-4377-10', resolved_by='auto-dismissed'))

    assert _classify(corpus) is Disposition.AUTO_DISMISSED


@pytest.mark.parametrize('resolved_by', ['l2-cascade:esc-1-1', 'leo', None])
def test_any_other_resolver_is_closed_by_other(resolved_by):
    corpus = _corpus(_esc('esc-4377-10', resolved_by=resolved_by))

    assert _classify(corpus) is Disposition.CLOSED_BY_OTHER


@pytest.mark.parametrize(
    'l1',
    [
        # live esc-5587-20 has one steward L1 8 ms before and another 39.7 s
        # before; only the adjacent one is its promotion
        _steward_l1('esc-4377-11', before_resolution=timedelta(seconds=40)),
        _steward_l1('esc-4377-11', before_resolution=timedelta(seconds=-1)),  # after
        _steward_l1('esc-9999-1', task_id='9999',
                    before_resolution=timedelta(milliseconds=8)),
        _steward_l1('esc-4377-11', agent_role='implementer',
                    before_resolution=timedelta(milliseconds=8)),
    ],
    ids=['40s-before', '1s-after', 'other-task', 'non-steward-l1'],
)
def test_an_l1_that_is_not_the_steward_s_own_promotion_is_not_one(l1):
    corpus = _corpus(_esc('esc-4377-10', resolved_by='steward'), l1)

    assert _classify(corpus) is Disposition.RESOLVED_IN_PLACE


def test_only_the_adjacent_l1_counts_when_an_earlier_one_also_exists():
    corpus = _corpus(
        _esc('esc-4377-10', resolved_by='steward'),
        _steward_l1('esc-4377-11', before_resolution=timedelta(seconds=39.7)),
        _steward_l1('esc-4377-12', before_resolution=timedelta(milliseconds=8)),
    )

    assert _classify(corpus) is Disposition.PROMOTED_TO_L1


def _summary_corpus():
    return _corpus(
        _esc('esc-1-1'),                                             # in place
        _esc('esc-2-1', resolved_by='steward'),                      # promoted
        _steward_l1('esc-2-2', task_id='2',
                    before_resolution=timedelta(milliseconds=8)),
        _esc('esc-3-1', resolved_by='steward'),                      # in place
        _esc('esc-4-1', resolved_by='auto-dismissed'),
        _esc('esc-5-1', status='pending', resolved_at=None, resolved_by=None),
    )


def test_the_summary_counts_each_escalation_once_and_unlinked_runs_apart():
    ids = ['esc-1-1', 'esc-1-1', 'esc-1-1', 'esc-2-1', 'esc-3-1', 'esc-4-1',
           'esc-5-1', 'esc-gone-1', None, None]

    summary = escalation_ladder.summarize_steward_dispositions(
        ids, _summary_corpus(), as_of=AS_OF,
    )

    assert summary.unlinked_runs == 2
    assert dict(summary.counts) == {
        Disposition.RECORD_MISSING: 1,
        Disposition.PENDING: 1,
        Disposition.PROMOTED_TO_L1: 1,
        Disposition.RESOLVED_IN_PLACE: 2,
        Disposition.AUTO_DISMISSED: 1,
        Disposition.CLOSED_BY_OTHER: 0,
    }
    assert [d for d, _ in summary.counts] == list(Disposition)
    # decided = 6 distinct ids - 1 missing - 1 pending = 4
    assert summary.decided == 4
    assert summary.resolved_in_place_share == pytest.approx(2 / 4)
    assert summary.promoted_share == pytest.approx(1 / 4)


def test_shares_are_none_when_nothing_was_decided():
    summary = escalation_ladder.summarize_steward_dispositions(
        ['esc-5-1', 'esc-gone-1', None], _summary_corpus(), as_of=AS_OF,
    )

    assert summary.resolved_in_place_share is None
    assert summary.promoted_share is None


# --- L2-tier metrics: what reached the human tier, and how it closed ---


def _l2(esc_id, *, at, agent_role='steward', resolved_at=None, action=None):
    return _esc(
        esc_id, level=2, agent_role=agent_role, timestamp=at,
        status='pending' if resolved_at is None else 'resolved',
        resolved_at=resolved_at, resolved_by=None if resolved_at is None else 'leo',
        resolution_action=action,
    )


def test_l2_metrics_count_the_half_open_window_and_ignore_lower_levels():
    day = timedelta(days=1)
    corpus = _corpus(
        _l2('esc-1-1', at=T0),                                           # in (at since)
        _l2('esc-2-1', at=T0 + day, agent_role='escalation-watcher-auto',
            resolved_at=T0 + day + timedelta(hours=1), action='close_only'),
        _l2('esc-3-1', at=T0 + timedelta(hours=3), agent_role='escalation-watcher-auto',
            resolved_at=T0 + 2 * day + timedelta(hours=1), action='close_only'),
        _l2('esc-4-1', at=T0 + timedelta(hours=5),
            resolved_at=T0 + timedelta(hours=6), action='code_change'),
        _l2('esc-5-1', at=T0 + 2 * day),                                 # at until: out
        _l2('esc-6-1', at=T0 - timedelta(seconds=1)),                    # before since
        _esc('esc-7-1', level=1, timestamp=T0 + timedelta(hours=1)),     # L1: ignored
        _esc('esc-8-1', level=0, timestamp=T0 + timedelta(hours=1)),     # L0: ignored
    )

    metrics = escalation_ladder.l2_tier_metrics(corpus, since=T0, until=T0 + 2 * day)

    assert metrics.filed == 4
    assert metrics.filed_per_day == pytest.approx(2.0)
    assert metrics.watcher_filed == 2
    assert metrics.watcher_filed_per_day == pytest.approx(1.0)
    assert metrics.resolved == 2           # esc-3-1 resolved only after until
    assert metrics.close_only == 1
    assert metrics.close_only_share == pytest.approx(0.5)


def test_l2_metrics_over_an_empty_window_are_zero_with_no_share():
    metrics = escalation_ladder.l2_tier_metrics(
        _corpus(), since=T0, until=T0 + timedelta(days=14),
    )

    assert (metrics.filed, metrics.watcher_filed, metrics.resolved, metrics.close_only) == (
        0, 0, 0, 0,
    )
    assert metrics.filed_per_day == 0.0
    assert metrics.close_only_share is None
