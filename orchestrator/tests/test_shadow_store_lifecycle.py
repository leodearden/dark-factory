"""The shadow-cell transition choke point, cell queries and the durable storm pause.

Everything runs through a real ``RunStore`` on a tmp ``runs.db``
(plans/live-shadow-eval-prd.md C1/C2).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestrator.run_store import (
    CellChange,
    RunStore,
    ShadowCell,
    ShadowCellQuery,
    ShadowStormRecord,
)
from orchestrator.shadow_types import (
    CostBasis,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
    ShadowStormCause,
)

_S = ShadowCellState
_OPENED_AT = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
_CELL_ID = '01JCELLAAAAAAAAAAAAAAAAAAA'


def _cell(**overrides) -> ShadowCell:
    fields = {
        'cell_id': _CELL_ID,
        'project_id': 'dark_factory',
        'task_id': '5387',
        'shape': ShadowShape.IMPLEMENTER,
        'candidate': 'glm-5.3-endpoint',
        'incumbent': 'claude-opus-high',
        'stratum': ('df', 'feature', 'full'),
        'base_sha': 'a' * 40,
        'state': _S.OPENED,
        'cost_basis': CostBasis.METERED,
        'opened_at': _OPENED_AT,
        'settle_deadline': _OPENED_AT + timedelta(hours=168),
    }
    fields.update(overrides)
    return ShadowCell(**fields)


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / 'runs.db'


@pytest.fixture
def store(db_path) -> RunStore:
    return RunStore(db_path)


def _raw_row(db_path: Path, cell_id: str) -> tuple:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute('SELECT * FROM shadow_cells WHERE cell_id = ?', (cell_id,)).fetchone()
    finally:
        conn.close()


_AT = _OPENED_AT + timedelta(minutes=7)


class TestAdvanceShadowCell:
    """``advance_shadow_cell`` is the single compare-and-set transition choke point."""

    def test_a_legal_transition_returns_and_persists_the_successor(self, store):
        store.insert_shadow_cell(_cell())

        updated = store.advance_shadow_cell(
            _CELL_ID, expected=_S.OPENED, change=CellChange(state=_S.RUNNING, at=_AT),
        )

        assert updated is not None
        assert updated.state is _S.RUNNING
        assert updated.running_at == _AT
        assert updated.settled_at is None
        assert store.get_shadow_cell(_CELL_ID) == updated

    def test_a_lost_compare_and_set_returns_none_and_writes_nothing(self, store, db_path):
        store.insert_shadow_cell(_cell(state=_S.RUNNING, running_at=_AT))
        before = _raw_row(db_path, _CELL_ID)

        result = store.advance_shadow_cell(
            _CELL_ID, expected=_S.OPENED, change=CellChange(state=_S.RUNNING, at=_AT),
        )

        assert result is None
        assert _raw_row(db_path, _CELL_ID) == before

    @pytest.mark.parametrize(
        'stored,expected,target',
        [
            (_S.AWAITING_REFERENCE, _S.AWAITING_REFERENCE, _S.RUNNING),
            (_S.SETTLED, _S.SETTLED, _S.SETTLED),
        ],
    )
    def test_an_illegal_transition_raises_naming_both_states_and_the_cell(
        self, store, db_path, stored, expected, target,
    ):
        settled = {'settled_at': _AT} if stored is _S.SETTLED else {}
        store.insert_shadow_cell(_cell(state=stored, running_at=_AT, **settled))
        before = _raw_row(db_path, _CELL_ID)
        change = (
            CellChange(state=target, at=_AT, reference_kind=ReferenceKind.NONE)
            if target is _S.SETTLED
            else CellChange(state=target, at=_AT)
        )

        with pytest.raises(ValueError) as excinfo:
            store.advance_shadow_cell(_CELL_ID, expected=expected, change=change)

        message = str(excinfo.value)
        assert _CELL_ID in message
        assert expected.value in message
        assert target.value in message
        assert _raw_row(db_path, _CELL_ID) == before

    @pytest.mark.parametrize(
        'change',
        [
            CellChange(state=_S.SETTLED, at=_AT, reference_kind=ReferenceKind.NONE),
            CellChange(state=_S.EXPIRED, at=_AT, reason='settle_deadline'),
        ],
        ids=lambda c: c.state.value,
    )
    def test_settling_targets_stamp_settled_at(self, store, change):
        store.insert_shadow_cell(_cell(state=_S.AWAITING_REFERENCE, running_at=_OPENED_AT))

        updated = store.advance_shadow_cell(_CELL_ID, expected=_S.AWAITING_REFERENCE, change=change)

        assert updated is not None
        assert updated.settled_at == _AT

    @pytest.mark.parametrize(
        'stored,change',
        [
            (_S.SAMPLED, CellChange(state=_S.OPENED, at=_AT, worktree_path=Path('/tmp/w'))),
            (_S.SAMPLED, CellChange(state=_S.FAILED, at=_AT, reason='runner_raised:OSError')),
            (_S.RUNNING, CellChange(state=_S.AWAITING_REFERENCE, at=_AT)),
            (_S.RUNNING, CellChange(state=_S.CAP_EXCLUDED, at=_AT, reason='usage_cap')),
        ],
        ids=lambda v: v.value if isinstance(v, ShadowCellState) else v.state.value,
    )
    def test_no_other_target_touches_settled_at(self, store, stored, change):
        store.insert_shadow_cell(_cell(state=stored))

        updated = store.advance_shadow_cell(_CELL_ID, expected=stored, change=change)

        assert updated is not None
        assert updated.settled_at is None

    @pytest.mark.parametrize(
        'stored,change',
        [
            (_S.SAMPLED, CellChange(state=_S.FAILED, at=_AT, reason='runner_raised:OSError')),
            (_S.OPENED, CellChange(state=_S.FAILED, at=_AT, reason='cancelled_at_shutdown')),
            (_S.RUNNING, CellChange(state=_S.AWAITING_REFERENCE, at=_AT)),
            (_S.RUNNING, CellChange(state=_S.CAP_EXCLUDED, at=_AT, reason='credits_exhausted')),
        ],
        ids=lambda v: v.value if isinstance(v, ShadowCellState) else v.state.value,
    )
    def test_leaving_a_held_state_stamps_run_ended_at(self, store, stored, change):
        store.insert_shadow_cell(_cell(state=stored))

        updated = store.advance_shadow_cell(_CELL_ID, expected=stored, change=change)

        assert updated is not None
        assert updated.run_ended_at == _AT
        assert store.get_shadow_cell(_CELL_ID) == updated

    @pytest.mark.parametrize(
        'stored,change',
        [
            (_S.SAMPLED, CellChange(state=_S.OPENED, at=_AT, worktree_path=Path('/tmp/w'))),
            (_S.OPENED, CellChange(state=_S.RUNNING, at=_AT)),
        ],
        ids=lambda v: v.value if isinstance(v, ShadowCellState) else v.state.value,
    )
    def test_a_still_held_successor_leaves_run_ended_at_unset(self, store, stored, change):
        store.insert_shadow_cell(_cell(state=stored))

        updated = store.advance_shadow_cell(_CELL_ID, expected=stored, change=change)

        assert updated is not None
        assert updated.run_ended_at is None

    def test_settling_keeps_the_run_end_it_already_has(self, store):
        ended = _AT - timedelta(minutes=3)
        store.insert_shadow_cell(_cell(state=_S.AWAITING_REFERENCE, run_ended_at=ended))
        change = CellChange(state=_S.SETTLED, at=_AT, reference_kind=ReferenceKind.NONE)

        updated = store.advance_shadow_cell(_CELL_ID, expected=_S.AWAITING_REFERENCE, change=change)

        assert updated is not None
        assert (updated.run_ended_at, updated.settled_at) == (ended, _AT)

    def test_opened_carries_the_worktree_path(self, store):
        store.insert_shadow_cell(_cell(state=_S.SAMPLED))
        change = CellChange(state=_S.OPENED, at=_AT, worktree_path=Path('/tmp/eval/shadow_5387'))

        updated = store.advance_shadow_cell(_CELL_ID, expected=_S.SAMPLED, change=change)

        assert updated is not None
        assert updated.worktree_path == Path('/tmp/eval/shadow_5387')
        assert updated.running_at is None

    def test_awaiting_carries_the_result_and_costs(self, store):
        store.insert_shadow_cell(_cell(state=_S.RUNNING, cost_basis=CostBasis.SUBSCRIPTION))
        change = CellChange(
            state=_S.AWAITING_REFERENCE,
            at=_AT,
            result_path=Path('/tmp/results/cell.json'),
            cost_usd=0.42,
            cost_credits=812.5,
        )

        updated = store.advance_shadow_cell(_CELL_ID, expected=_S.RUNNING, change=change)

        assert updated is not None
        assert (updated.result_path, updated.cost_usd, updated.cost_credits) == (
            Path('/tmp/results/cell.json'),
            0.42,
            812.5,
        )
        assert store.get_shadow_cell(_CELL_ID) == updated

    @pytest.mark.parametrize(
        'stored,target,reason',
        [
            (_S.RUNNING, _S.FAILED, 'runner_raised:RuntimeError'),
            (_S.RUNNING, _S.CAP_EXCLUDED, 'credits_exhausted'),
            (_S.AWAITING_REFERENCE, _S.EXPIRED, 'settle_deadline'),
        ],
    )
    def test_non_settled_terminals_carry_their_reason(self, store, stored, target, reason):
        store.insert_shadow_cell(_cell(state=stored))

        updated = store.advance_shadow_cell(
            _CELL_ID, expected=stored, change=CellChange(state=target, at=_AT, reason=reason),
        )

        assert updated is not None
        assert updated.reason == reason
        assert store.get_shadow_cell(_CELL_ID) == updated

    @pytest.mark.parametrize(
        'kind,sha',
        [(ReferenceKind.LANDED, 'b' * 40), (ReferenceKind.NONE, None)],
    )
    def test_settled_carries_its_reference(self, store, kind, sha):
        store.insert_shadow_cell(_cell(state=_S.AWAITING_REFERENCE))
        change = CellChange(state=_S.SETTLED, at=_AT, reference_kind=kind, reference_sha=sha)

        updated = store.advance_shadow_cell(_CELL_ID, expected=_S.AWAITING_REFERENCE, change=change)

        assert updated is not None
        assert (updated.reference_kind, updated.reference_sha) == (kind, sha)
        assert store.get_shadow_cell(_CELL_ID) == updated

    def test_an_unknown_cell_raises_key_error_naming_it(self, store):
        with pytest.raises(KeyError, match='01JNOSUCHCELL'):
            store.advance_shadow_cell(
                '01JNOSUCHCELL', expected=_S.OPENED, change=CellChange(state=_S.RUNNING, at=_AT),
            )


class TestIncoherentChangeIsRefused:
    """An incoherent change is refused before anything is written."""

    @pytest.mark.parametrize(
        'fields',
        [
            {'state': _S.SETTLED},
            {'state': _S.SETTLED, 'reference_kind': ReferenceKind.NONE, 'reference_sha': 'b' * 40},
            {'state': _S.SETTLED, 'reference_kind': ReferenceKind.LANDED},
            {'state': _S.SETTLED, 'reference_kind': ReferenceKind.LANDED, 'reference_sha': '  '},
            {'state': _S.EXPIRED, 'reason': 'settle_deadline', 'reference_kind': ReferenceKind.NONE},
            {'state': _S.FAILED, 'reason': 'x', 'reference_sha': 'b' * 40},
        ],
        ids=[
            'settled-without-kind',
            'sha-on-none',
            'landed-without-sha',
            'landed-with-blank-sha',
            'kind-on-expired',
            'sha-on-failed',
        ],
    )
    def test_the_change_itself_refuses_incoherent_reference_fields(self, fields):
        with pytest.raises(ValueError, match='reference'):
            CellChange(at=_AT, **fields)

    @pytest.mark.parametrize(
        'stored,change',
        [
            (_S.RUNNING, CellChange(state=_S.FAILED, at=_AT)),
            (_S.RUNNING, CellChange(state=_S.AWAITING_REFERENCE, at=_AT, cost_credits=3.0)),
        ],
        ids=['failed-without-reason', 'credits-on-a-metered-cell'],
    )
    def test_the_successor_record_re_validates_and_nothing_is_written(
        self, store, db_path, stored, change,
    ):
        store.insert_shadow_cell(_cell(state=stored))
        before = _raw_row(db_path, _CELL_ID)

        with pytest.raises(ValueError, match=_CELL_ID):
            store.advance_shadow_cell(_CELL_ID, expected=stored, change=change)

        assert _raw_row(db_path, _CELL_ID) == before


def _seed(store: RunStore, n: int, **overrides) -> list[ShadowCell]:
    cells = []
    for i in range(n):
        fields = {
            'cell_id': f'01JSEED{i:02d}',
            'task_id': f't{i}',
            'opened_at': _OPENED_AT + timedelta(minutes=i),
            'settle_deadline': _OPENED_AT + timedelta(minutes=i, hours=1),
        }
        fields.update(overrides)
        cell = _cell(**fields)
        store.insert_shadow_cell(cell)
        cells.append(cell)
    return cells


def _ids(cells: list[ShadowCell]) -> list[str]:
    return [cell.cell_id for cell in cells]


class TestFindShadowCells:
    """``find_shadow_cells`` filters on every query field, scoped to one project."""

    def test_default_is_every_cell_of_the_project_oldest_first(self, store):
        cells = _seed(store, 3)
        store.insert_shadow_cell(_cell(cell_id='01JOTHERPROJECT', project_id='reify'))

        found = store.find_shadow_cells(ShadowCellQuery(project_id='dark_factory'))

        assert _ids(found) == _ids(cells)

    def test_newest_first_and_limit(self, store):
        cells = _seed(store, 4)

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', newest_first=True, limit=2),
        )

        assert _ids(found) == [cells[3].cell_id, cells[2].cell_id]

    def test_states_filter(self, store):
        store.insert_shadow_cell(_cell(cell_id='01JA', task_id='a', state=_S.SAMPLED))
        store.insert_shadow_cell(_cell(cell_id='01JB', task_id='b', state=_S.RUNNING))
        store.insert_shadow_cell(_cell(cell_id='01JC', task_id='c', state=_S.AWAITING_REFERENCE))

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', states=frozenset({_S.SAMPLED, _S.RUNNING})),
        )

        assert sorted(_ids(found)) == ['01JA', '01JB']

    @pytest.mark.parametrize(
        'field,value,query_value',
        [
            ('task_id', 'other-task', 'other-task'),
            ('shape', ShadowShape.ARCHITECT, ShadowShape.ARCHITECT),
            ('candidate', 'minimax-m3', 'minimax-m3'),
            ('run_id', 'run-xyz', 'run-xyz'),
        ],
    )
    def test_equality_filters(self, store, field, value, query_value):
        _seed(store, 2)
        store.insert_shadow_cell(_cell(**{'cell_id': '01JMATCH', 'task_id': 'm', field: value}))

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', **{field: query_value}),
        )

        assert _ids(found) == ['01JMATCH']

    def test_opened_since_is_inclusive(self, store):
        cells = _seed(store, 4)

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', opened_since=cells[2].opened_at),
        )

        assert _ids(found) == _ids(cells[2:])

    def test_run_ended_since_is_inclusive_and_skips_unended_cells(self, store):
        for i in range(3):
            store.insert_shadow_cell(_cell(
                cell_id=f'01JENDED{i}',
                task_id=f'e{i}',
                state=_S.CAP_EXCLUDED,
                reason='credits_exhausted',
                run_ended_at=_OPENED_AT + timedelta(minutes=i + 1),
            ))
        store.insert_shadow_cell(_cell(cell_id='01JUNENDED', task_id='u', state=_S.RUNNING))

        found = store.find_shadow_cells(ShadowCellQuery(
            project_id='dark_factory', run_ended_since=_OPENED_AT + timedelta(minutes=2),
        ))

        assert _ids(found) == ['01JENDED1', '01JENDED2']

    def test_deadline_before_is_strict(self, store):
        cells = _seed(store, 4)

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', deadline_before=cells[2].settle_deadline),
        )

        assert _ids(found) == _ids(cells[:2])

    def test_time_filters_compare_instants_across_offsets(self, store):
        cells = _seed(store, 3)
        plus_five = cells[1].opened_at.astimezone(timezone(timedelta(hours=5)))

        found = store.find_shadow_cells(
            ShadowCellQuery(project_id='dark_factory', opened_since=plus_five),
        )

        assert _ids(found) == _ids(cells[1:])

    def test_an_empty_states_filter_is_refused(self):
        with pytest.raises(ValueError, match='states'):
            ShadowCellQuery(project_id='dark_factory', states=frozenset())


class TestRunEndedAt:
    """``run_ended_at``: when the cell's run ended, added to an already-deployed table."""

    def test_a_naive_run_end_is_refused(self):
        with pytest.raises(ValueError, match='run_ended_at'):
            _cell(run_ended_at=datetime(2026, 10, 19, 12, 5))

    def test_a_table_without_the_column_gains_it_and_old_rows_read_none(self, store, db_path):
        store.insert_shadow_cell(_cell(state=_S.RUNNING))
        conn = sqlite3.connect(db_path)
        try:
            conn.execute('ALTER TABLE shadow_cells DROP COLUMN run_ended_at')
            conn.commit()
        finally:
            conn.close()

        reopened = RunStore(db_path)

        stored = reopened.get_shadow_cell(_CELL_ID)
        assert stored is not None
        assert stored.run_ended_at is None
        ended = reopened.advance_shadow_cell(
            _CELL_ID, expected=_S.RUNNING, change=CellChange(state=_S.AWAITING_REFERENCE, at=_AT),
        )
        assert ended is not None
        assert ended.run_ended_at == _AT


class TestTripleUniqueness:
    """C1 at-most-once: one cell per (project, task, shape, candidate)."""

    def test_a_second_cell_for_the_same_triple_is_refused(self, store):
        store.insert_shadow_cell(_cell())

        with pytest.raises(sqlite3.IntegrityError):
            store.insert_shadow_cell(_cell(cell_id='01JSECONDCELL'))

    def test_another_shape_or_candidate_is_a_different_triple(self, store):
        store.insert_shadow_cell(_cell())
        store.insert_shadow_cell(_cell(cell_id='01JB', shape=ShadowShape.ARCHITECT))
        store.insert_shadow_cell(_cell(cell_id='01JC', candidate='minimax-m3'))

        assert len(store.find_shadow_cells(ShadowCellQuery(project_id='dark_factory'))) == 3


_PAUSED_AT = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)


class TestShadowStorm:
    """The durable one-row-per-project storm pause."""

    def test_a_never_paused_project_reads_an_empty_record(self, store):
        record = store.get_shadow_storm('p')

        assert isinstance(record, ShadowStormRecord)
        assert (record.cause, record.detail, record.paused_at, record.resumed_at) == (
            None,
            None,
            None,
            None,
        )
        assert record.paused is False

    def test_only_the_first_pause_creates_it_and_keeps_its_cause(self, store):
        assert store.pause_shadow_storm(
            'p', ShadowStormCause.FAILURE_STREAK, 'five failed cells', _PAUSED_AT,
        ) is True
        assert store.pause_shadow_storm(
            'p', ShadowStormCause.DAILY_BUDGET, 'spent 3.0 >= 2.0', _PAUSED_AT + timedelta(hours=1),
        ) is False

        record = store.get_shadow_storm('p')
        assert record.paused is True
        assert record.cause is ShadowStormCause.FAILURE_STREAK
        assert record.detail == 'five failed cells'
        assert record.paused_at == _PAUSED_AT

    def test_resume_records_resumed_at_once(self, store):
        store.pause_shadow_storm('p', ShadowStormCause.COST_RATIO, 'ratio 4.1 > 3.0', _PAUSED_AT)
        resumed_at = _PAUSED_AT + timedelta(hours=2)

        assert store.resume_shadow_storm('p', resumed_at) is True
        assert store.resume_shadow_storm('p', resumed_at + timedelta(hours=1)) is False

        record = store.get_shadow_storm('p')
        assert record.paused is False
        assert record.cause is None
        assert record.resumed_at == resumed_at

    def test_resuming_a_never_paused_project_changes_nothing(self, store):
        assert store.resume_shadow_storm('p', _PAUSED_AT) is False
        assert store.get_shadow_storm('p').resumed_at is None

    def test_a_pause_after_a_resume_is_a_new_pause(self, store):
        store.pause_shadow_storm('p', ShadowStormCause.COST_RATIO, 'first', _PAUSED_AT)
        store.resume_shadow_storm('p', _PAUSED_AT + timedelta(hours=1))

        assert store.pause_shadow_storm(
            'p', ShadowStormCause.DAILY_BUDGET, 'second', _PAUSED_AT + timedelta(hours=2),
        ) is True
        assert store.get_shadow_storm('p').cause is ShadowStormCause.DAILY_BUDGET

    def test_the_pause_is_scoped_to_its_project(self, store):
        store.pause_shadow_storm('p', ShadowStormCause.FAILURE_STREAK, 'd', _PAUSED_AT)

        assert store.get_shadow_storm('q').paused is False

    def test_a_new_store_on_the_same_file_sees_the_pause(self, store, db_path):
        store.pause_shadow_storm('p', ShadowStormCause.FAILURE_STREAK, 'd', _PAUSED_AT)

        record = RunStore(db_path).get_shadow_storm('p')

        assert record.paused is True
        assert record.cause is ShadowStormCause.FAILURE_STREAK
        assert record.paused_at == _PAUSED_AT

    def test_the_record_refuses_a_cause_without_a_pause_time(self):
        with pytest.raises(ValueError, match='paused_at'):
            ShadowStormRecord(
                project_id='p',
                cause=ShadowStormCause.FAILURE_STREAK,
                detail='d',
                paused_at=None,
                resumed_at=None,
            )
