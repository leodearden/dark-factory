"""Reading the orchestrator's runs.db: the per-event stream and the whole-store derived readers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _stores import RunsDb
from client_feed.sources.events import (
    EventBatch,
    LandedCommit,
    PhaseChange,
    SkippedRow,
    SkipReason,
    StoreEventType,
    VerifyVerdict,
    read_events,
)
from client_feed.sources.store import StoreProblem, StoreReadError
from shared.project_layout import runs_db

from client_feed.contract import Source, VerifyKind

BASE = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
SHA_A = 'a' * 40
SHA_B = 'b' * 40
SHA_C = 'c' * 40
LIMIT = 100


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / 'root'
    path.mkdir()
    return path


@pytest.fixture
def runs(root: Path) -> RunsDb:
    return RunsDb.create(root)


def read_all(root: Path, *, after_id: int = 0) -> EventBatch:
    return read_events(root, after_id=after_id, limit=LIMIT)


def merge_finalized(state: str, merge_sha: str | None = SHA_A) -> dict:
    """The merge_lane/worker.py::_on_finalized durable-tier payload."""
    return {
        'request_id': 'req-1',
        'branch': 'not-the-task-id',
        'state': state,
        'snapshot_tip': SHA_C,
        'merge_sha': merge_sha,
        'superseded_by': None,
        'generation': 1,
        'reason': 'SECRET-REASON /home/leo/x',
        'landed_via_chain': False,
        'absorbed_request_ids': [],
    }


class TestLandedCommits:
    def test_a_done_merge_finalized_lands_the_task_named_by_the_task_id_column(
        self, root: Path, runs: RunsDb
    ) -> None:
        row = runs.emit(
            'merge_finalized', at=at(0), task_id='7', phase='merge', data=merge_finalized('done')
        )

        batch = read_all(root)

        assert batch.events == (
            LandedCommit(event_id=row, at=at(0), commit=SHA_A, task_ids=(7,)),
        )
        assert batch.skipped == ()

    @pytest.mark.parametrize('state', ['blocked', 'conflict', 'superseded', 'already_merged'])
    def test_a_merge_finalized_that_did_not_land_is_dropped_not_skipped(
        self, root: Path, runs: RunsDb, state: str
    ) -> None:
        runs.emit('merge_finalized', at=at(0), task_id='7', data=merge_finalized(state, None))

        batch = read_all(root)

        assert (batch.events, batch.skipped) == ((), ())

    @pytest.mark.parametrize('task_id', ['_mainprobe-1', '.task-meta', '0', '-3', None])
    def test_a_done_row_without_a_positive_integer_task_id_lands_no_task(
        self, root: Path, runs: RunsDb, task_id: str | None
    ) -> None:
        row = runs.emit('merge_finalized', at=at(0), task_id=task_id, data=merge_finalized('done'))

        assert read_all(root).events == (
            LandedCommit(event_id=row, at=at(0), commit=SHA_A, task_ids=()),
        )

    def test_a_train_merged_lands_every_member_in_order(self, root: Path, runs: RunsDb) -> None:
        row = runs.emit(
            'train_merged',
            at=at(0),
            task_id='5985',
            data={
                'train_id': 'coalesce-5985-a90176e0',
                'member_task_ids': ['5985', '5959'],
                'merge_commit_sha': SHA_B,
                'base_sha': SHA_C,
            },
        )

        assert read_all(root).events == (
            LandedCommit(event_id=row, at=at(0), commit=SHA_B, task_ids=(5985, 5959)),
        )


class TestVerifyVerdicts:
    def test_a_merge_verify_is_a_merge_verdict_with_its_duration(
        self, root: Path, runs: RunsDb
    ) -> None:
        row = runs.emit(
            'merge_verify',
            at=at(0),
            task_id='7',
            data={
                'runner': 'local',
                'merge_sha': SHA_A,
                'passed': False,
                'duration_ms': 1234,
                'attempt': 1,
            },
        )

        assert read_all(root).events == (
            VerifyVerdict(
                event_id=row,
                at=at(0),
                kind=VerifyKind.MERGE,
                passed=False,
                duration_ms=1234,
                commit=SHA_A,
                task_id=7,
            ),
        )

    def test_a_merge_verify_of_no_task_keeps_its_verdict(self, root: Path, runs: RunsDb) -> None:
        runs.emit(
            'merge_verify',
            at=at(0),
            task_id='_mainprobe-abc',
            data={'merge_sha': SHA_A, 'passed': True, 'duration_ms': 10},
        )

        (verdict,) = read_all(root).events
        assert isinstance(verdict, VerifyVerdict)
        assert (verdict.task_id, verdict.passed) == (None, True)

    def test_a_red_workflow_verify_is_a_task_verdict_at_its_tip(
        self, root: Path, runs: RunsDb
    ) -> None:
        row = runs.emit(
            'workflow_verify',
            at=at(0),
            task_id='7',
            data={'passed': False, 'tip_sha': SHA_A, 'base_sha': SHA_C, 'branch': 'task/7'},
        )

        assert read_all(root).events == (
            VerifyVerdict(
                event_id=row,
                at=at(0),
                kind=VerifyKind.TASK,
                passed=False,
                duration_ms=None,
                commit=SHA_A,
                task_id=7,
            ),
        )

    def test_a_workflow_verify_without_a_tip_has_no_commit(
        self, root: Path, runs: RunsDb
    ) -> None:
        runs.emit('workflow_verify', at=at(0), task_id='7', data={'passed': True})

        (verdict,) = read_all(root).events
        assert isinstance(verdict, VerifyVerdict)
        assert (verdict.commit, verdict.passed) == (None, True)


class TestPhaseChanges:
    def test_phase_rows_carry_the_phase_column(self, root: Path, runs: RunsDb) -> None:
        entered = runs.emit('phase_enter', at=at(0), task_id='7', phase='execute')
        left = runs.emit('phase_exit', at=at(1), task_id='7', phase='execute')

        assert read_all(root).events == (
            PhaseChange(event_id=entered, at=at(0), task_id=7, phase='execute', entered=True),
            PhaseChange(event_id=left, at=at(1), task_id=7, phase='execute', entered=False),
        )

    def test_a_phase_row_of_no_task_is_dropped(self, root: Path, runs: RunsDb) -> None:
        runs.emit('phase_enter', at=at(0), task_id='_mainprobe-1', phase='verify')

        batch = read_all(root)

        assert (batch.events, batch.skipped) == ((), ())


def test_free_text_never_reaches_the_batch(root: Path, runs: RunsDb) -> None:
    runs.emit(
        'merge_blocked', at=at(0), task_id='7', data={'reason': 'SECRET-REASON /home/x'}
    )
    runs.emit(
        'escalation_created',
        at=at(1),
        task_id='7',
        data={'escalation_id': 'esc-7-1', 'level': 1, 'summary': 'SECRET-SUMMARY'},
    )
    runs.emit('merge_finalized', at=at(2), task_id='7', data=merge_finalized('done'))

    batch = read_all(root)

    assert len(batch.events) == 1
    assert 'SECRET-' not in repr(batch)
    assert '/home/' not in repr(batch)


class TestPaging:
    def test_rows_at_or_below_the_cursor_are_excluded(self, root: Path, runs: RunsDb) -> None:
        first = runs.emit('phase_enter', at=at(0), task_id='7', phase='plan')
        second = runs.emit('phase_exit', at=at(1), task_id='7', phase='plan')

        batch = read_all(root, after_id=first)

        assert [event.event_id for event in batch.events] == [second]

    def test_the_limit_counts_mapped_rows_read_and_the_cursor_is_the_last_one(
        self, root: Path, runs: RunsDb
    ) -> None:
        entered = runs.emit('phase_enter', at=at(0), task_id='7', phase='plan')
        runs.emit('cost_update', at=at(1), task_id='7', data={'cost_usd': 1.0})
        dropped = runs.emit('merge_finalized', at=at(2), task_id='7', data=merge_finalized('conflict'))
        left = runs.emit('phase_exit', at=at(3), task_id='7', phase='plan')
        runs.emit('cost_update', at=at(4), task_id='7', data={'cost_usd': 2.0})

        first = read_events(root, after_id=0, limit=2)
        second = read_events(root, after_id=first.cursor, limit=2)
        third = read_events(root, after_id=second.cursor, limit=2)

        assert [event.event_id for event in first.events] == [entered]
        assert (first.cursor, first.has_more) == (dropped, True)
        assert [event.event_id for event in second.events] == [left]
        assert (second.cursor, second.has_more) == (left, False)
        assert third == EventBatch(events=(), skipped=(), cursor=left, has_more=False)

    def test_an_empty_store_leaves_the_cursor_where_it_was(self, root: Path, runs: RunsDb) -> None:
        assert read_events(root, after_id=42, limit=10) == EventBatch(
            events=(), skipped=(), cursor=42, has_more=False
        )


class TestSkippedRows:
    def test_a_landed_commit_with_a_malformed_sha_is_skipped(
        self, root: Path, runs: RunsDb
    ) -> None:
        row = runs.emit('merge_finalized', at=at(0), task_id='7', data=merge_finalized('done', 'abc'))

        batch = read_all(root)

        assert batch.events == ()
        assert batch.skipped == (
            SkippedRow(
                event_id=row,
                event_type=StoreEventType.MERGE_FINALIZED,
                reason=SkipReason.MALFORMED_PAYLOAD,
            ),
        )
        assert batch.cursor == row

    def test_a_blob_that_is_not_json_is_skipped(self, root: Path, runs: RunsDb) -> None:
        row = runs.emit('merge_verify', at=at(0), task_id='7', data='not json')

        assert read_all(root).skipped == (
            SkippedRow(
                event_id=row,
                event_type=StoreEventType.MERGE_VERIFY,
                reason=SkipReason.MALFORMED_PAYLOAD,
            ),
        )

    def test_a_non_boolean_passed_is_skipped(self, root: Path, runs: RunsDb) -> None:
        runs.emit('merge_verify', at=at(0), task_id='7', data={'passed': 1, 'merge_sha': SHA_A})

        (skipped,) = read_all(root).skipped
        assert skipped.reason is SkipReason.MALFORMED_PAYLOAD

    @pytest.mark.parametrize('timestamp', ['yesterday', '2026-10-01T10:00:00'])
    def test_a_timestamp_that_is_not_an_aware_iso_instant_is_skipped(
        self, root: Path, runs: RunsDb, timestamp: str
    ) -> None:
        row = runs.emit('phase_enter', at=timestamp, task_id='7', phase='plan')

        assert read_all(root).skipped == (
            SkippedRow(
                event_id=row,
                event_type=StoreEventType.PHASE_ENTER,
                reason=SkipReason.BAD_TIMESTAMP,
            ),
        )

    def test_an_offset_timestamp_reads_as_the_same_utc_instant(
        self, root: Path, runs: RunsDb
    ) -> None:
        runs.emit('phase_enter', at='2026-10-01T12:00:00+02:00', task_id='7', phase='plan')

        (change,) = read_all(root).events
        assert change.at == at(0)
        assert change.at.utcoffset() == timedelta(0)


class TestStoreFailures:
    def test_a_missing_store_is_refused_and_never_created(self, root: Path) -> None:
        with pytest.raises(StoreReadError) as caught:
            read_all(root)

        assert (caught.value.source, caught.value.problem) == (Source.EVENTS, StoreProblem.MISSING)
        assert not runs_db(root).exists()
        assert str(root) not in str(caught.value)

    def test_a_store_that_is_not_sqlite_is_unreadable(self, root: Path) -> None:
        path = runs_db(root)
        path.parent.mkdir(parents=True)
        path.write_bytes(b'this is not a sqlite database at all' * 100)

        with pytest.raises(StoreReadError) as caught:
            read_all(root)

        assert (caught.value.source, caught.value.problem) == (
            Source.EVENTS,
            StoreProblem.UNREADABLE,
        )
        assert str(root) not in str(caught.value)

    def test_a_read_leaves_the_store_bytes_unchanged(self, root: Path, runs: RunsDb) -> None:
        runs.emit('merge_finalized', at=at(0), task_id='7', data=merge_finalized('done'))
        before = runs.path.read_bytes()

        read_all(root)

        assert runs.path.read_bytes() == before
