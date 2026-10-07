"""The link-heal ledger and run lock (tasks 6181, 6184; PRD H1 "Ledger" and "One run at a time", H2)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from fused_memory.maintenance.link_heal import (
    BasisSource,
    HealAction,
    LinkImage,
    PlannedAction,
    Verdict,
)
from fused_memory.maintenance.link_heal_ledger import (
    ActionState,
    AdjudicationRecord,
    AmbiguousRun,
    LinkHealLedger,
    RunLock,
    RunLockHeld,
    RunSource,
    UnknownRun,
)
from fused_memory.server.grouped_read import CONTESTED_METADATA_KEY

PROJECT = 'dark_factory'
PARENT = '22222222-2222-2222-2222-222222222222'
CHILD_SHA = 'c' * 64
PARENT_SHA = 'p' * 64


def _child(n: int) -> str:
    return f'11111111-1111-1111-1111-{n:012d}'


def _planned(
    n: int = 1,
    *,
    action: HealAction = HealAction.DETACH,
    kind: str | None = 'amendment',
    parent: str = PARENT,
    child_sha: str = CHILD_SHA,
    parent_sha: str | None = PARENT_SHA,
    source: BasisSource = BasisSource.CORPUS,
) -> PlannedAction:
    return PlannedAction.planned(
        project_id=PROJECT,
        child_id=_child(n),
        action=action,
        pre_image=LinkImage(parent_id=parent, kind=kind, contested=None),
        basis_source=source,
        basis_key=None if source is BasisSource.DETERMINISTIC else f'H{n:03d}',
        child_sha256=child_sha,
        parent_sha256=parent_sha,
    )


@pytest.fixture
def ledger(tmp_path: Path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


class TestSchema:
    def test_every_table_has_an_autoincrement_key(self, tmp_path):
        db = tmp_path / 'link_heal.db'
        LinkHealLedger(db).close()

        conn = sqlite3.connect(db)
        try:
            tables = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"))
        finally:
            conn.close()
        for name in ('runs', 'adjudications', 'actions'):
            assert re.search(
                r'\bid INTEGER PRIMARY KEY AUTOINCREMENT\b', tables[name],
            ), tables[name]

    def test_sqlite_sequence_proves_a_producer(self, tmp_path):
        db = tmp_path / 'link_heal.db'
        ledger = LinkHealLedger(db)

        def sequence_names() -> list[str]:
            conn = sqlite3.connect(db)
            try:
                return [name for (name,) in conn.execute('SELECT name FROM sqlite_sequence')]
            finally:
                conn.close()

        try:
            assert sequence_names() == []
            ledger.start_run(RunSource.CORPUS, writes=False)
            assert sequence_names() == ['runs']
        finally:
            ledger.close()


class TestRuns:
    def test_start_run_mints_a_32_hex_run_id(self, ledger):
        assert re.fullmatch(r'[0-9a-f]{32}', ledger.start_run(RunSource.CORPUS, writes=False))

    def test_finish_run_stores_counts_and_plan_sha(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)

        ledger.finish_run(run_id, counts={'planned': 3, 'would_escape': []}, plan_sha256='f' * 64)

        run = ledger.resolve_run(run_id)
        assert run.run_id == run_id
        assert run.source is RunSource.CORPUS
        assert run.writes is False
        assert run.finished_at is not None
        assert run.counts == {'planned': 3, 'would_escape': []}
        assert run.plan_sha256 == 'f' * 64

    def test_writing_run_count_counts_only_writing_runs(self, ledger):
        ledger.finish_run(ledger.start_run(RunSource.CORPUS, writes=False), counts={})
        assert ledger.writing_run_count() == 0

        ledger.finish_run(ledger.start_run(RunSource.CORPUS, writes=True), counts={})
        assert ledger.writing_run_count() == 1

    def test_recent_runs_are_newest_first(self, ledger):
        first = ledger.start_run(RunSource.CORPUS, writes=False)
        second = ledger.start_run(RunSource.CORPUS, writes=True)

        assert [run.run_id for run in ledger.recent_runs(10)] == [second, first]
        assert [run.run_id for run in ledger.recent_runs(1)] == [second]

    def test_resolve_run_accepts_the_full_id_or_its_8_hex_prefix(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=True)

        assert ledger.resolve_run(run_id).run_id == run_id
        assert ledger.resolve_run(run_id[:8]).run_id == run_id

    def test_resolve_run_refuses_an_unknown_id(self, ledger):
        ledger.start_run(RunSource.CORPUS, writes=True)
        with pytest.raises(UnknownRun):
            ledger.resolve_run('0' * 8)
        with pytest.raises(UnknownRun):
            ledger.resolve_run('not-hex%')

    def test_resolve_run_refuses_an_ambiguous_prefix(self, ledger, monkeypatch):
        ids = iter(['abcdef01' + '1' * 24, 'abcdef01' + '2' * 24])
        monkeypatch.setattr(
            'fused_memory.maintenance.link_heal_ledger.new_run_id', lambda: next(ids),
        )
        ledger.start_run(RunSource.CORPUS, writes=True)
        ledger.start_run(RunSource.CORPUS, writes=True)

        with pytest.raises(AmbiguousRun):
            ledger.resolve_run('abcdef01')


class TestActions:
    def test_add_planned_round_trips_every_field(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        flagged = PlannedAction.planned(
            project_id=PROJECT,
            child_id=_child(2),
            action=HealAction.RELABEL_FLAG,
            pre_image=LinkImage(parent_id=PARENT, kind='sighting', contested=None),
            basis_source=BasisSource.CORPUS,
            basis_key='H002',
            child_sha256=CHILD_SHA,
            parent_sha256=PARENT_SHA,
        )
        dangling = _planned(3, source=BasisSource.DETERMINISTIC, parent_sha=None)

        ledger.add_planned(run_id, [flagged, dangling])

        rows = ledger.pending_actions(RunSource.CORPUS)
        assert [row.planned for row in rows] == [flagged, dangling]
        assert rows[0].planned.post_image.as_dict()[CONTESTED_METADATA_KEY] is True
        assert all(row.state is ActionState.PLANNED for row in rows)
        assert all(row.run_id == run_id for row in rows)

    def test_pending_is_planned_or_skipped_cap_oldest_first_and_by_source(self, ledger):
        corpus = ledger.start_run(RunSource.CORPUS, writes=False)
        ids = ledger.add_planned(corpus, [_planned(1), _planned(2), _planned(3)])
        other = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_planned(other, [_planned(4)])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)

        ledger.set_outcome(ids[0], ActionState.SKIPPED_CAP, apply_run, None)
        ledger.set_outcome(ids[1], ActionState.APPLIED, apply_run, None)

        pending = ledger.pending_actions(RunSource.CORPUS)
        assert [row.action_id for row in pending] == [ids[0], ids[2]]
        assert [row.state for row in pending] == [ActionState.SKIPPED_CAP, ActionState.PLANNED]

    def test_set_outcome_stores_structured_detail(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)

        ledger.set_outcome(
            action_id, ActionState.SKIPPED_STALE, apply_run, {'stale_field': 'child_sha256'},
        )

        assert ledger.pending_actions(RunSource.CORPUS) == []
        (row,) = ledger.run_actions(run_id)
        assert row.state is ActionState.SKIPPED_STALE
        assert row.detail == {'stale_field': 'child_sha256'}
        assert row.executed_run_id == apply_run

    def test_applied_actions_are_those_of_the_executing_run(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        first, second = ledger.add_planned(run_id, [_planned(1), _planned(2)])
        apply_a = ledger.start_run(RunSource.CORPUS, writes=True)
        apply_b = ledger.start_run(RunSource.CORPUS, writes=True)

        ledger.set_outcome(first, ActionState.APPLIED, apply_a, None)
        ledger.set_outcome(second, ActionState.APPLIED, apply_b, None)

        assert [row.action_id for row in ledger.applied_actions(apply_a)] == [first]

    def test_mark_undone_records_the_undo_run(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)
        ledger.set_outcome(action_id, ActionState.APPLIED, apply_run, None)
        undo_run = ledger.start_run(RunSource.UNDO, writes=True)

        ledger.mark_undone(action_id, undo_run)

        (row,) = ledger.run_actions(run_id)
        assert row.state is ActionState.UNDONE
        assert row.undone_run_id == undo_run
        assert ledger.applied_actions(apply_run) == []

    def test_find_pending_matches_an_identical_pending_row(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])

        found = ledger.find_pending(_planned())

        assert found is not None
        assert found.action_id == action_id
        assert ledger.find_pending(_planned(child_sha='d' * 64)) is None
        assert ledger.find_pending(_planned(kind='sighting')) is None

    def _undone(self, ledger) -> None:
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)
        ledger.set_outcome(action_id, ActionState.APPLIED, apply_run, None)
        ledger.mark_undone(action_id, ledger.start_run(RunSource.UNDO, writes=True))

    def test_an_undone_action_suppresses_the_same_action_at_the_same_hashes(self, ledger):
        self._undone(ledger)
        assert ledger.is_undo_suppressed(_planned()) is True

    @pytest.mark.parametrize(
        'changed',
        [
            {'child_sha': 'd' * 64},
            {'parent_sha': 'e' * 64},
            {'parent': '33333333-3333-3333-3333-333333333333'},
            {'action': HealAction.FLAG},
        ],
        ids=['child-text', 'parent-text', 'parent', 'action'],
    )
    def test_a_changed_text_parent_or_action_reopens_it(self, ledger, changed):
        self._undone(ledger)
        assert ledger.is_undo_suppressed(_planned(**changed)) is False

    def test_a_pending_or_applied_row_does_not_suppress(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        ledger.add_planned(run_id, [_planned()])
        assert ledger.is_undo_suppressed(_planned()) is False

    def test_an_undo_step_is_recorded_under_the_undo_run(self, ledger):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)
        ledger.set_outcome(action_id, ActionState.APPLIED, apply_run, None)
        (original,) = ledger.applied_actions(apply_run)
        undo_run = ledger.start_run(RunSource.UNDO, writes=True)

        step_id = ledger.add_undo_step(
            undo_run,
            original,
            before=original.planned.post_image,
            after=original.planned.pre_image,
            state=ActionState.APPLIED,
            detail=None,
        )

        (step,) = ledger.undo_steps(undo_run)
        assert step.action_id == step_id
        assert step.basis_source is BasisSource.UNDO
        assert step.basis_key == str(action_id)
        assert step.original_action_id == action_id
        assert step.child_id == original.planned.child_id
        assert step.before == original.planned.post_image
        assert step.after == original.planned.pre_image
        assert step.state is ActionState.APPLIED
        assert ledger.pending_actions(RunSource.UNDO) == []
        assert ledger.applied_actions(undo_run) == []

    def test_the_last_applied_undo_step_is_the_newest_that_applied_in_any_undo_run(
        self, ledger,
    ):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)
        (action_id,) = ledger.add_planned(run_id, [_planned()])
        apply_run = ledger.start_run(RunSource.CORPUS, writes=True)
        ledger.set_outcome(action_id, ActionState.APPLIED, apply_run, None)
        (original,) = ledger.applied_actions(apply_run)
        assert ledger.last_applied_undo_step(action_id) is None
        part_way = LinkImage(parent_id=PARENT)
        first_undo = ledger.start_run(RunSource.UNDO, writes=True)
        for before, after, state in (
            (original.planned.post_image, part_way, ActionState.APPLIED),
            (part_way, original.planned.pre_image, ActionState.FAILED),
        ):
            ledger.add_undo_step(
                first_undo, original, before=before, after=after, state=state, detail=None,
            )
        ledger.add_undo_step(
            ledger.start_run(RunSource.UNDO, writes=True), original,
            before=part_way, after=original.planned.pre_image,
            state=ActionState.SKIPPED_STALE, detail=None,
        )

        step = ledger.last_applied_undo_step(action_id)

        assert step is not None
        assert (step.undo_run_id, step.after) == (first_undo, part_way)


class TestPublishingPlanned:
    def test_yields_the_pending_rows_with_the_new_ones_and_commits_them(self, ledger):
        earlier = ledger.start_run(RunSource.CORPUS, writes=False)
        ledger.add_planned(earlier, [_planned(1)])
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)

        with ledger.publishing_planned(run_id, [_planned(2)], RunSource.CORPUS) as pending:
            assert [row.planned for row in pending] == [_planned(1), _planned(2)]

        assert [row.planned for row in ledger.run_actions(run_id)] == [_planned(2)]

    def test_a_raising_body_rolls_the_new_rows_back(self, ledger, tmp_path):
        run_id = ledger.start_run(RunSource.CORPUS, writes=False)

        with (
            pytest.raises(OSError, match='disk full'),
            ledger.publishing_planned(run_id, [_planned(1)], RunSource.CORPUS),
        ):
            raise OSError('disk full')

        assert ledger.run_actions(run_id) == []
        reopened = LinkHealLedger(tmp_path / 'link_heal.db')
        try:
            assert [run.run_id for run in reopened.recent_runs(10)] == [run_id]
            assert reopened.run_actions(run_id) == []
        finally:
            reopened.close()


def _record(
    n: int = 1,
    *,
    verdict: Verdict = Verdict.EXTENDS,
    parent: str = PARENT,
    child_sha: str = CHILD_SHA,
    parent_sha: str = PARENT_SHA,
) -> AdjudicationRecord:
    return AdjudicationRecord(
        project_id=PROJECT,
        child_id=_child(n),
        parent_id=parent,
        child_sha256=child_sha,
        parent_sha256=parent_sha,
        verdict=verdict,
        reason=f'reason {n}',
        model='opus',
    )


class TestAdjudications:
    def test_add_adjudications_returns_increasing_ids_and_proves_a_producer(self, ledger):
        run_id = ledger.start_run(RunSource.ADJUDICATOR, writes=False)

        ids = ledger.add_adjudications(run_id, [_record(1), _record(2), _record(3)])

        assert len(ids) == 3
        assert ids == sorted(ids)
        assert len(set(ids)) == 3
        conn = sqlite3.connect(ledger.db_path)
        try:
            names = {name for (name,) in conn.execute('SELECT name FROM sqlite_sequence')}
        finally:
            conn.close()
        assert 'adjudications' in names

    def test_adjudication_at_returns_the_newest_verdict_at_exactly_those_hashes(self, ledger):
        first = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(first, [_record(1, verdict=Verdict.SAME)])
        second = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        (newest,) = ledger.add_adjudications(second, [_record(1, verdict=Verdict.RELATED)])

        row = ledger.adjudication_at(PROJECT, _child(1), PARENT, CHILD_SHA, PARENT_SHA)

        assert row is not None
        assert row.adjudication_id == newest
        assert row.run_id == second
        assert row.record == _record(1, verdict=Verdict.RELATED)
        assert row.record.verdict is Verdict.RELATED
        assert row.at

    @pytest.mark.parametrize(
        'changed',
        [
            {'child_sha256': 'd' * 64},
            {'parent_sha256': 'q' * 64},
            {'parent_id': '33333333-3333-3333-3333-333333333333'},
        ],
    )
    def test_adjudication_at_misses_on_another_text_or_parent(self, ledger, changed):
        run_id = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(run_id, [_record(1)])
        lookup = {
            'project_id': PROJECT,
            'child_id': _child(1),
            'parent_id': PARENT,
            'child_sha256': CHILD_SHA,
            'parent_sha256': PARENT_SHA,
            **changed,
        }

        assert ledger.adjudication_at(**lookup) is None

    def test_adjudication_verdicts_are_those_of_the_named_runs(self, ledger):
        first = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(first, [_record(1, verdict=Verdict.SAME), _record(2, verdict=Verdict.RELATED)])
        second = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(second, [_record(3, verdict=Verdict.CORRECTS)])
        third = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(third, [_record(4, verdict=Verdict.UNRELATED)])

        verdicts = ledger.adjudication_verdicts({first, second})

        assert sorted(verdicts) == sorted([Verdict.SAME, Verdict.RELATED, Verdict.CORRECTS])
        assert all(isinstance(verdict, Verdict) for verdict in verdicts)

    def test_adjudication_verdicts_of_no_runs_are_none(self, ledger):
        run_id = ledger.start_run(RunSource.ADJUDICATOR, writes=False)
        ledger.add_adjudications(run_id, [_record(1)])

        assert ledger.adjudication_verdicts(set()) == []

    def test_a_record_whose_verdict_is_not_a_verdict_cannot_be_built(self):
        with pytest.raises((TypeError, ValueError), match='MAYBE'):
            AdjudicationRecord(
                project_id=PROJECT,
                child_id=_child(1),
                parent_id=PARENT,
                child_sha256=CHILD_SHA,
                parent_sha256=PARENT_SHA,
                verdict='MAYBE',  # pyright: ignore[reportArgumentType]
                reason='r',
                model='opus',
            )


class TestRunLock:
    def test_acquiring_records_the_holder(self, tmp_path):
        lock_path = tmp_path / 'link_heal.lock'

        with RunLock(lock_path) as lock:
            holder = json.loads(lock_path.read_text())
            assert holder['pid'] == os.getpid()
            assert holder['started_at']
            assert lock.reclaimed_from is None

    def test_a_held_lock_is_refused_naming_its_holder(self, tmp_path):
        lock_path = tmp_path / 'link_heal.lock'

        with RunLock(lock_path):
            holder = json.loads(lock_path.read_text())
            with pytest.raises(RunLockHeld) as excinfo, RunLock(lock_path):
                pass

        held = excinfo.value
        assert held.holder is not None
        assert held.holder.pid == holder['pid']
        assert held.holder.started_at == holder['started_at']
        assert str(holder['pid']) in str(held)
        assert holder['started_at'] in str(held)

    def test_a_released_lock_can_be_taken_again(self, tmp_path):
        lock_path = tmp_path / 'link_heal.lock'
        with RunLock(lock_path):
            pass

        with RunLock(lock_path) as lock:
            assert lock.reclaimed_from is None

    def test_a_dead_holders_lock_is_reclaimed_and_reported(self, tmp_path):
        lock_path = tmp_path / 'link_heal.lock'
        lock_path.write_text(json.dumps({'pid': 2**22 + 12345, 'started_at': '2026-10-01T00:00:00+00:00'}))

        with RunLock(lock_path) as lock:
            assert lock.reclaimed_from is not None
            assert lock.reclaimed_from.pid == 2**22 + 12345
            assert lock.reclaimed_from.started_at == '2026-10-01T00:00:00+00:00'
            assert json.loads(lock_path.read_text())['pid'] == os.getpid()
