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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.models import Escalation

import dashboard.data.escalation_corpus as escalation_corpus
from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumState, validate_datum
from dashboard.data.escalation_corpus import (
    CORPUS_FRESHNESS_BOUND_SECONDS,
    CORPUS_TTL_SECONDS,
    EscalationCorpus,
    EscalationView,
    Location,
    QueueKind,
    QueueRef,
    acquire_corpus,
    corpus_queues,
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
    )


def _fresh(corpus: EscalationCorpus) -> Datum[EscalationCorpus]:
    return Datum(corpus, NOW, DatumState.FRESH, None, CORPUS_FRESHNESS_BOUND_SECONDS)


def _reason(datum: Datum) -> str:
    assert datum.reason is not None
    return datum.reason


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
            ),
            QueueRef(
                id=str(reify.resolve()), label='reify', kind=QueueKind.ORCHESTRATOR,
                directory=reify.resolve() / 'data' / 'escalations',
            ),
            QueueRef(
                id='reconciliation', label='fused-memory',
                kind=QueueKind.RECONCILIATION,
                directory=config.reconciliation_escalations_dir,
            ),
        )


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
