"""The escalation corpus datum — ``dashboard.data.escalation_corpus``.

PRD ``plans/dashboard-one-datum-one-path-prd.md`` decision 13 and sketch #10:
every escalation surface reads ONE walk of every queue (root and archive), and
"pending in the live queue" and "open in history" are two named views over it.

Fixtures are real escalation trees under ``tmp_path``: records are written
from ``Escalation(...).to_dict()`` at the queue root or under
``archive/<date>/``, exactly where the escalation server puts them. Only the
module's public names are driven.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.archive import archive_dir_for_date
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.task_tags import DEFAULT_TAG

import dashboard.data.escalation_corpus as escalation_corpus
from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumState, unknown_datum, validate_datum
from dashboard.data.escalation_corpus import (
    CORPUS_FRESHNESS_BOUND_SECONDS,
    CORPUS_TTL_SECONDS,
    HUMAN_LEVEL,
    EscalationCorpus,
    EscalationView,
    Location,
    PendingEscalations,
    QueueKind,
    QueueRef,
    acquire_corpus,
    corpus_queues,
    measure_corpus,
    pending_escalation_ids,
    pending_level2_by_task,
    serve_corpus,
    views_over,
    walk_corpus,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
"""The one injected instant every corpus datum here is stamped against."""

ARCHIVE_DAY = '2026-09-01'


def _record(esc_id: str, *, status: str = 'pending', task_id: str = '7') -> dict:
    """One escalation as the server writes it."""
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category='design_concern',
        summary=f'summary of {esc_id}',
        timestamp='2026-09-01T00:00:00+00:00',
        status=status,
    ).to_dict()


def _write(queue_dir: Path, record: dict, *, archived: bool = False) -> Path:
    directory = queue_dir / 'archive' / ARCHIVE_DAY if archived else queue_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['id']}.json"
    path.write_text(json.dumps(record))
    return path


def _queue(directory: Path, label: str = 'proj') -> QueueRef:
    return QueueRef(
        id=str(directory), label=label, kind=QueueKind.ORCHESTRATOR, directory=directory,
        runs_db=directory / 'runs.db',
    )


def _fresh(corpus: EscalationCorpus) -> Datum[EscalationCorpus]:
    return Datum(corpus, NOW, DatumState.FRESH, None, CORPUS_FRESHNESS_BOUND_SECONDS)


def _reason(datum: Datum) -> str:
    assert datum.reason is not None
    return datum.reason


def _by_task(datum: Datum[Mapping[int, tuple[str, ...]]]) -> dict[int, tuple[str, ...]]:
    assert datum.value is not None, datum.reason
    return dict(datum.value)


def _corpus(datum: Datum[EscalationCorpus]) -> EscalationCorpus:
    assert datum.value is not None
    return datum.value


def _sketch_10(queue_dir: Path) -> None:
    """Sketch #10: 2 pending at the root, 3 pending in the archive, 1 resolved at the root."""
    for n in (1, 2):
        _write(queue_dir, _record(f'esc-1-{n}'))
    for n in (3, 4, 5):
        _write(queue_dir, _record(f'esc-1-{n}'), archived=True)
    _write(queue_dir, _record('esc-1-6', status='resolved'))


class TestCorpusQueues:
    """The one copy of the queue iteration: primary, known roots de-duped, reconciliation."""

    def test_primary_then_known_roots_deduped_then_reconciliation(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()
        config = DashboardConfig(
            project_root=primary, known_project_roots=[primary, reify, reify],
        )

        queues = corpus_queues(config)

        assert queues == (
            QueueRef(
                id=str(primary.resolve()), label='primary',
                kind=QueueKind.ORCHESTRATOR, directory=config.escalations_dir,
                runs_db=config.runs_db,
            ),
            QueueRef(
                id=str(reify.resolve()), label='reify', kind=QueueKind.ORCHESTRATOR,
                directory=reify.resolve() / 'data' / 'escalations',
                runs_db=reify.resolve() / 'data' / 'orchestrator' / 'runs.db',
            ),
            QueueRef(
                id='reconciliation', label='fused-memory',
                kind=QueueKind.RECONCILIATION,
                directory=config.reconciliation_escalations_dir, runs_db=None,
            ),
        )


class TestQueueRef:
    """An orchestrator queue is a project's and carries its run ledger; reconciliation has none."""

    @pytest.mark.parametrize(('kind', 'runs_db'), [
        (QueueKind.ORCHESTRATOR, None),
        (QueueKind.RECONCILIATION, Path('/srv/proj/data/orchestrator/runs.db')),
    ])
    def test_a_runs_db_belongs_to_an_orchestrator_queue_and_only_to_one(self, kind, runs_db):
        with pytest.raises(ValueError, match='orchestrator queue has a runs_db'):
            QueueRef(id='q', label='q', kind=kind, directory=Path('/srv/q'), runs_db=runs_db)


class TestWalkCorpus:
    """One walk per queue, every record placed by where it lay."""

    def test_each_record_carries_its_location(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-1-1'))
        _write(queue_dir, _record('esc-1-2', status='resolved'), archived=True)
        queue = _queue(queue_dir)

        scan = walk_corpus((queue,)).scan(queue.id)

        assert scan.reached is True
        assert scan.unreadable == ()
        assert {(r.escalation.id, r.location) for r in scan.records} == {
            ('esc-1-1', Location.ROOT),
            ('esc-1-2', Location.ARCHIVE),
        }

    def test_a_stem_in_both_root_and_archive_appears_once_as_root(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-1-1'))
        _write(queue_dir, _record('esc-1-1', status='resolved'), archived=True)
        queue = _queue(queue_dir)

        records = walk_corpus((queue,)).scan(queue.id).records

        assert [(r.escalation.id, r.location, r.escalation.status) for r in records] == [
            ('esc-1-1', Location.ROOT, 'pending'),
        ]

    def test_unparseable_files_are_unreadable_entries_not_records(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-1-1'))
        bad_json = queue_dir / 'esc-1-2.json'
        bad_json.write_text('{not json')
        archive_dir = queue_dir / 'archive' / ARCHIVE_DAY
        archive_dir.mkdir(parents=True)
        not_an_escalation = archive_dir / 'esc-1-3.json'
        not_an_escalation.write_text(json.dumps({'id': 'esc-1-3'}))
        queue = _queue(queue_dir)

        scan = walk_corpus((queue,)).scan(queue.id)

        assert [r.escalation.id for r in scan.records] == ['esc-1-1']
        by_path = {u.path: u for u in scan.unreadable}
        assert set(by_path) == {str(bad_json), str(not_an_escalation)}
        assert by_path[str(bad_json)].location is Location.ROOT
        assert by_path[str(not_an_escalation)].location is Location.ARCHIVE
        assert all(u.error for u in scan.unreadable)

    def test_a_missing_queue_dir_is_unreached_and_empty(self, tmp_path):
        queue = _queue(tmp_path / 'never-escalated')

        corpus = walk_corpus((queue,))
        scan = corpus.scan(queue.id)

        assert scan.reached is False
        assert scan.records == ()
        assert scan.unreadable == ()
        assert corpus.reached_any is False

    def test_scans_follow_the_queue_order(self, tmp_path):
        first = _queue(tmp_path / 'a' / 'escalations', 'a')
        second = _queue(tmp_path / 'b' / 'escalations', 'b')

        corpus = walk_corpus((first, second))

        assert [scan.queue for scan in corpus.scans] == [first, second]


class TestViewsOver:
    """Two named views over one walk, stamped with the walk's own instant."""

    def test_sketch_10_counts(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)

        views = views_over(_fresh(walk_corpus((queue,))), [queue.id])

        assert views[EscalationView.QUEUE_PENDING].value == 2
        assert views[EscalationView.OPEN_IN_HISTORY].value == 5

    def test_views_share_the_corpus_as_of_and_validate(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)
        corpus_datum = _fresh(walk_corpus((queue,)))

        views = views_over(corpus_datum, [queue.id])

        assert set(views) == {EscalationView.QUEUE_PENDING, EscalationView.OPEN_IN_HISTORY}
        for datum in views.values():
            assert datum.as_of is corpus_datum.as_of
            assert datum.state is DatumState.FRESH
            assert datum.reason is None
            validate_datum(datum, NOW)

    def test_an_unreadable_file_makes_its_scope_a_lower_bound(self, tmp_path):
        broken_dir = tmp_path / 'broken' / 'escalations'
        _write(broken_dir, _record('esc-1-1'))
        (broken_dir / 'esc-1-2.json').write_text('{not json')
        clean_dir = tmp_path / 'clean' / 'escalations'
        _write(clean_dir, _record('esc-2-1'))
        broken = _queue(broken_dir, 'broken')
        clean = _queue(clean_dir, 'clean')
        corpus_datum = _fresh(walk_corpus((broken, clean)))

        partial = views_over(corpus_datum, [broken.id])
        sibling = views_over(corpus_datum, [clean.id])

        for datum in partial.values():
            assert datum.state is DatumState.LOWER_BOUND
            assert 'broken' in _reason(datum)
            assert '1 file' in _reason(datum)
            validate_datum(datum, NOW)
        assert partial[EscalationView.QUEUE_PENDING].value == 1
        for datum in sibling.values():
            assert datum.state is DatumState.FRESH
            assert datum.reason is None
            validate_datum(datum, NOW)

    def test_an_unreached_queue_dir_makes_its_scope_a_lower_bound(self, tmp_path):
        missing_dir = tmp_path / 'missing' / 'escalations'
        present_dir = tmp_path / 'present' / 'escalations'
        _write(present_dir, _record('esc-2-1'))
        missing = _queue(missing_dir, 'missing')
        present = _queue(present_dir, 'present')
        corpus_datum = _fresh(walk_corpus((missing, present)))

        fleet = views_over(corpus_datum, [missing.id, present.id])

        for datum in fleet.values():
            assert datum.state is DatumState.LOWER_BOUND
            assert 'missing' in _reason(datum)
            assert str(missing_dir) in _reason(datum)
            validate_datum(datum, NOW)
        assert fleet[EscalationView.QUEUE_PENDING].value == 1
        assert fleet[EscalationView.OPEN_IN_HISTORY].value == 1

    def test_views_sum_over_every_queue_in_scope(self, tmp_path):
        first_dir = tmp_path / 'a' / 'escalations'
        second_dir = tmp_path / 'b' / 'escalations'
        _sketch_10(first_dir)
        _write(second_dir, _record('esc-9-1'))
        _write(second_dir, _record('esc-9-2'), archived=True)
        first = _queue(first_dir, 'a')
        second = _queue(second_dir, 'b')

        views = views_over(_fresh(walk_corpus((first, second))), [first.id, second.id])

        assert views[EscalationView.QUEUE_PENDING].value == 3
        assert views[EscalationView.OPEN_IN_HISTORY].value == 7


class TestPendingEscalationIds:
    """Each task's pending live-queue escalations in one queue; None when that is unknown."""

    def test_pending_root_records_are_grouped_by_task_with_sorted_ids(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-2', task_id='7'))
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        _write(queue_dir, _record('esc-9-1', task_id='9'))
        queue = _queue(queue_dir)

        ids = pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id)

        assert ids == PendingEscalations({'7': ('esc-7-1', 'esc-7-2'), '9': ('esc-9-1',)})

    def test_the_queue_names_only_tasks_of_the_default_tag(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        queue = _queue(queue_dir)

        ids = pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id)

        assert ids is not None
        assert ids.pinning(DEFAULT_TAG, 7) == ('esc-7-1',)
        assert ids.pinning(DEFAULT_TAG, 8) == ()
        assert ids.pinning('feature', 7) is None

    def test_resolved_and_archived_records_are_not_in_the_live_queue_view(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        _write(queue_dir, _record('esc-7-2', status='resolved', task_id='7'))
        _write(queue_dir, _record('esc-8-1', task_id='8'), archived=True)
        queue = _queue(queue_dir)

        ids = pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id)

        assert ids == PendingEscalations({'7': ('esc-7-1',)})

    def test_an_unreached_queue_is_unknown(self, tmp_path):
        queue = _queue(tmp_path / 'never-escalated')

        assert pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id) is None

    def test_an_unreadable_file_makes_the_queue_unknown_not_none(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        (queue_dir / 'esc-7-2.json').write_text('{not json')
        queue = _queue(queue_dir)

        assert pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id) is None

    def test_an_unreadable_archived_file_leaves_the_live_queue_known(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        corrupt = _write(queue_dir, _record('esc-8-1', task_id='8'), archived=True)
        corrupt.write_text('{not json')
        queue = _queue(queue_dir)

        ids = pending_escalation_ids(_fresh(walk_corpus((queue,))), queue.id)

        assert ids == PendingEscalations({'7': ('esc-7-1',)})

    def test_an_unread_corpus_is_unknown(self, tmp_path):
        unread = unknown_datum('the escalation corpus was not read', CORPUS_FRESHNESS_BOUND_SECONDS)

        assert pending_escalation_ids(unread, str(tmp_path)) is None

    def test_only_the_named_queue_is_read(self, tmp_path):
        first_dir = tmp_path / 'a' / 'escalations'
        second_dir = tmp_path / 'b' / 'escalations'
        _write(first_dir, _record('esc-7-1', task_id='7'))
        _write(second_dir, _record('esc-7-9', task_id='7'))
        _write(second_dir, _record('esc-3-1', task_id='3'))
        first = _queue(first_dir, 'a')
        second = _queue(second_dir, 'b')

        ids = pending_escalation_ids(_fresh(walk_corpus((first, second))), first.id)

        assert ids == PendingEscalations({'7': ('esc-7-1',)})

    def test_a_queue_the_corpus_does_not_carry_is_unknown(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _write(queue_dir, _record('esc-7-1', task_id='7'))
        queue = _queue(queue_dir)

        ids = pending_escalation_ids(_fresh(walk_corpus((queue,))), str(tmp_path / 'other'))

        assert ids is None



@pytest.fixture(autouse=True)
def _corpus_cache():
    """No corpus walk may cross a test."""
    escalation_corpus._corpus_cache_clear()
    yield
    escalation_corpus._corpus_cache_clear()


def _counts(
    corpus_datum: Datum[EscalationCorpus], queue: QueueRef,
) -> tuple[int | None, int | None]:
    views = views_over(corpus_datum, [queue.id])
    return (
        views[EscalationView.QUEUE_PENDING].value,
        views[EscalationView.OPEN_IN_HISTORY].value,
    )


class TestAcquireCorpus:
    """The ONE cache: every escalation surface reads the corpus through it."""

    async def test_a_clean_walk_is_a_fresh_datum_stamped_now(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)

        corpus_datum = await acquire_corpus((queue,), now=NOW)

        assert isinstance(corpus_datum.value, EscalationCorpus)
        assert corpus_datum.as_of == NOW
        assert corpus_datum.state is DatumState.FRESH
        assert corpus_datum.reason is None
        assert corpus_datum.freshness_bound_seconds == 2 * CORPUS_TTL_SECONDS
        validate_datum(corpus_datum, NOW)

    async def test_a_partial_walk_is_a_lower_bound_naming_the_queue(self, tmp_path):
        present_dir = tmp_path / 'present' / 'escalations'
        _write(present_dir, _record('esc-1-1'))
        missing = _queue(tmp_path / 'missing' / 'escalations', 'missing')

        corpus_datum = await acquire_corpus((_queue(present_dir, 'present'), missing), now=NOW)

        assert corpus_datum.state is DatumState.LOWER_BOUND
        assert 'missing' in _reason(corpus_datum)
        validate_datum(corpus_datum, NOW)

    async def test_a_record_written_within_the_ttl_is_not_seen(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)

        first = await acquire_corpus((queue,), now=NOW)
        _write(queue_dir, _record('esc-1-7'))
        second = await acquire_corpus((queue,), now=NOW + timedelta(seconds=1))

        assert second.as_of == first.as_of == NOW
        assert _counts(second, queue) == (2, 5)

    async def test_a_record_written_before_the_ttl_expires_is_seen_after(
        self, tmp_path, monkeypatch,
    ):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)
        later = NOW + timedelta(seconds=1)

        await acquire_corpus((queue,), now=NOW)
        _write(queue_dir, _record('esc-1-7'))
        monkeypatch.setattr(escalation_corpus, 'CORPUS_TTL_SECONDS', 0.0)
        refreshed = await acquire_corpus((queue,), now=later)

        assert refreshed.as_of == later
        assert _counts(refreshed, queue) == (3, 6)

    async def test_a_walk_that_reached_no_queue_is_served_but_not_cached(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        queue = _queue(queue_dir)

        empty = await acquire_corpus((queue,), now=NOW)
        _sketch_10(queue_dir)
        filled = await acquire_corpus((queue,), now=NOW + timedelta(seconds=1))

        assert _corpus(empty).reached_any is False
        assert _counts(empty, queue) == (0, 0)
        assert _counts(filled, queue) == (2, 5)

    async def test_a_partial_walk_is_cached(self, tmp_path):
        present_dir = tmp_path / 'present' / 'escalations'
        _sketch_10(present_dir)
        present = _queue(present_dir, 'present')
        missing_dir = tmp_path / 'missing' / 'escalations'
        missing = _queue(missing_dir, 'missing')

        first = await acquire_corpus((present, missing), now=NOW)
        _write(missing_dir, _record('esc-2-1'))
        second = await acquire_corpus((present, missing), now=NOW + timedelta(seconds=1))

        assert second is first
        assert _corpus(second).scan(missing.id).reached is False

    @pytest.mark.parametrize('ttl_seconds', [0.001, 3600.0])
    async def test_sketch_10_counts_do_not_depend_on_the_ttl(
        self, tmp_path, monkeypatch, ttl_seconds,
    ):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)
        monkeypatch.setattr(escalation_corpus, 'CORPUS_TTL_SECONDS', ttl_seconds)

        corpus_datum = await acquire_corpus((queue,), now=NOW)

        assert _counts(corpus_datum, queue) == (2, 5)

    async def test_concurrent_cold_acquires_share_one_walk(self, tmp_path):
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        queue = _queue(queue_dir)

        first, second = await asyncio.gather(
            acquire_corpus((queue,), now=NOW),
            acquire_corpus((queue,), now=NOW),
        )

        assert first is second


class _CountedWalk:
    """``walk_corpus`` as patched in: counts its calls, optionally sleeping or raising first."""

    def __init__(self, *, sleep_seconds: float = 0.0, error: Exception | None = None) -> None:
        self.calls = 0
        self.sleep_seconds = sleep_seconds
        self.error = error

    def __call__(self, queues) -> EscalationCorpus:
        self.calls += 1
        time.sleep(self.sleep_seconds)
        if self.error is not None:
            raise self.error
        return walk_corpus(queues)


def _unreadable_walk() -> _CountedWalk:
    return _CountedWalk(error=OSError('queue directory unreadable'))


class TestServeCorpus:
    """The corpus served with a bounded wait, an exception guard and its last good walk."""

    @pytest.fixture
    def queue(self, tmp_path) -> QueueRef:
        queue_dir = tmp_path / 'escalations'
        _sketch_10(queue_dir)
        return _queue(queue_dir)

    async def test_a_walk_that_raises_before_any_success_is_unknown_naming_it(
        self, queue, monkeypatch,
    ):
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', _unreadable_walk())

        served = await serve_corpus((queue,), now=NOW, wait_seconds=None)

        assert served.state is DatumState.UNKNOWN
        assert served.value is None
        assert 'OSError' in _reason(served)
        assert 'queue directory unreadable' in _reason(served)
        assert served.freshness_bound_seconds == CORPUS_FRESHNESS_BOUND_SECONDS

    async def test_a_walk_that_raises_after_a_good_one_serves_the_last_good_walk(
        self, queue, monkeypatch,
    ):
        monkeypatch.setattr(escalation_corpus, 'CORPUS_TTL_SECONDS', 0.0)
        good = await serve_corpus((queue,), now=NOW, wait_seconds=None)
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', _unreadable_walk())

        served = await serve_corpus((queue,), now=NOW + timedelta(seconds=1), wait_seconds=None)

        assert good.state is DatumState.FRESH
        assert served.state is DatumState.STALE
        assert _corpus(served).scans == _corpus(good).scans
        assert served.as_of == NOW
        assert 'OSError' in _reason(served)

    async def test_a_slow_walk_is_not_waited_on_and_lands_for_a_later_call(
        self, queue, monkeypatch,
    ):
        walk = _CountedWalk(sleep_seconds=0.5)
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', walk)

        started = time.monotonic()
        waited = await serve_corpus((queue,), now=NOW, wait_seconds=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < walk.sleep_seconds
        assert waited.state is DatumState.UNKNOWN
        assert 'in flight' in _reason(waited)

        deadline = time.monotonic() + 5.0
        landed = waited
        while landed.state is not DatumState.FRESH and time.monotonic() < deadline:
            landed = await serve_corpus((queue,), now=NOW, wait_seconds=None)

        assert landed.state is DatumState.FRESH
        assert _counts(landed, queue) == (2, 5)
        assert walk.calls == 1

    async def test_the_cache_clear_hook_forgets_the_last_good_walk_too(
        self, queue, monkeypatch,
    ):
        await serve_corpus((queue,), now=NOW, wait_seconds=None)
        escalation_corpus._corpus_cache_clear()
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', _unreadable_walk())

        served = await serve_corpus((queue,), now=NOW + timedelta(seconds=1), wait_seconds=None)

        assert served.state is DatumState.UNKNOWN

    async def test_two_calls_inside_the_ttl_walk_once(self, queue, monkeypatch):
        walk = _CountedWalk()
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', walk)

        first = await serve_corpus((queue,), now=NOW, wait_seconds=None)
        second = await serve_corpus((queue,), now=NOW + timedelta(seconds=1), wait_seconds=None)

        assert walk.calls == 1
        assert first.state is second.state is DatumState.FRESH
        assert second.as_of == NOW


def _file(queue: EscalationQueue, esc_id: str, task_id: str, *, level: int = 2) -> None:
    """One pending escalation, filed through the real queue."""
    queue.submit(Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category='design_concern',
        summary=f'summary of {esc_id}',
        timestamp='2026-09-01T00:00:00+00:00',
        status='pending',
        level=level,
    ))


def _archive_pending(queue: EscalationQueue, esc_id: str) -> None:
    """Move a still-pending record into the archive, where no live escalation lies."""
    archived = archive_dir_for_date(queue.queue_dir, NOW.isoformat())
    archived.mkdir(parents=True, exist_ok=True)
    (queue.queue_dir / f'{esc_id}.json').replace(archived / f'{esc_id}.json')


class TestPendingLevel2ByTask:
    """The ruling input: each task's pending human-level escalations in its project's live queue."""

    @pytest.fixture
    def config(self, tmp_path) -> DashboardConfig:
        root = tmp_path / 'proj'
        root.mkdir()
        config = DashboardConfig(project_root=root)
        config.escalations_dir.mkdir(parents=True)
        config.reconciliation_escalations_dir.mkdir(parents=True)
        return config

    @staticmethod
    def _ruling(config: DashboardConfig) -> Datum:
        return pending_level2_by_task(
            measure_corpus(corpus_queues(config), now=NOW), str(config.project_root),
        )

    def test_a_pending_human_record_at_the_queue_root_names_its_task(self, config):
        _file(EscalationQueue(config.escalations_dir), 'esc-3546-2', '3546')

        datum = self._ruling(config)

        assert datum.state is DatumState.FRESH
        assert datum.reason is None
        assert datum.as_of == NOW
        assert datum.freshness_bound_seconds == CORPUS_FRESHNESS_BOUND_SECONDS
        assert _by_task(datum) == {3546: ('esc-3546-2',)}

    def test_only_pending_human_records_at_the_root_on_a_numbered_task_are_listed(self, config):
        queue = EscalationQueue(config.escalations_dir)
        _file(queue, 'esc-3546-2', '3546')
        _file(queue, 'esc-10-1', '10', level=1)
        _file(queue, 'esc-11-1', '11')
        queue.resolve('esc-11-1', 'ruled')
        _file(queue, 'esc-12-1', '12')
        _archive_pending(queue, 'esc-12-1')
        _file(queue, 'esc-b13-1', 'b13')

        datum = self._ruling(config)

        assert _by_task(datum) == {3546: ('esc-3546-2',)}

    @pytest.mark.parametrize('task_id', [None, 7])
    def test_a_human_record_whose_task_id_is_not_a_digit_string_is_left_out(
        self, config, task_id,
    ):
        _file(EscalationQueue(config.escalations_dir), 'esc-3546-2', '3546')
        record = _record('esc-none-1') | {'task_id': task_id, 'level': HUMAN_LEVEL}
        _write(config.escalations_dir, record)

        datum = self._ruling(config)

        assert datum.state is DatumState.FRESH
        assert _by_task(datum) == {3546: ('esc-3546-2',)}

    def test_two_human_records_on_one_task_are_both_listed_in_order(self, config):
        queue = EscalationQueue(config.escalations_dir)
        _file(queue, 'esc-20-3', '20')
        _file(queue, 'esc-20-1', '20')

        assert _by_task(self._ruling(config)) == {20: ('esc-20-1', 'esc-20-3')}

    def test_another_roots_missing_queue_does_not_make_this_projects_ruling_partial(
        self, config, tmp_path,
    ):
        other = tmp_path / 'other'
        other.mkdir()
        config = DashboardConfig(project_root=config.project_root, known_project_roots=[other])
        _file(EscalationQueue(config.escalations_dir), 'esc-3546-2', '3546')
        corpus_datum = measure_corpus(corpus_queues(config), now=NOW)

        datum = pending_level2_by_task(corpus_datum, str(config.project_root))

        assert corpus_datum.state is DatumState.LOWER_BOUND
        assert datum.state is DatumState.FRESH
        assert datum.reason is None
        assert _by_task(datum) == {3546: ('esc-3546-2',)}

    def test_this_projects_missing_queue_makes_its_ruling_a_lower_bound_naming_it(self, config):
        config.escalations_dir.rmdir()

        datum = self._ruling(config)

        assert datum.state is DatumState.LOWER_BOUND
        assert str(config.escalations_dir) in _reason(datum)
        assert _by_task(datum) == {}

    def test_an_unknown_corpus_is_an_unknown_ruling_with_its_reason(self, config):
        unread = unknown_datum('corpus walk timed out', CORPUS_FRESHNESS_BOUND_SECONDS)

        datum = pending_level2_by_task(unread, str(config.project_root))

        assert datum == unknown_datum('corpus walk timed out', CORPUS_FRESHNESS_BOUND_SECONDS)

    def test_a_queue_the_corpus_does_not_carry_is_an_unknown_ruling(self, config, tmp_path):
        datum = pending_level2_by_task(
            measure_corpus(corpus_queues(config), now=NOW), str(tmp_path / 'elsewhere'),
        )

        assert datum.state is DatumState.UNKNOWN
        assert datum.value is None
        assert str(tmp_path / 'elsewhere') in _reason(datum)
