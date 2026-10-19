"""Tests for the ``shadow_cells`` record and store (plans/live-shadow-eval-prd.md C1)."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestrator.run_store import RunStore, ShadowCell
from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
)

_OPENED_AT = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _cell(**overrides) -> ShadowCell:
    fields = {
        'cell_id': '01JCELLAAAAAAAAAAAAAAAAAAA',
        'project_id': 'dark_factory',
        'task_id': '5382',
        'shape': ShadowShape.IMPLEMENTER,
        'candidate': 'glm-5.3-endpoint',
        'incumbent': 'claude-opus-high',
        'stratum': ('df', 'feature', 'full'),
        'base_sha': 'a' * 40,
        'state': ShadowCellState.OPENED,
        'cost_basis': CostBasis.METERED,
        'opened_at': _OPENED_AT,
        'settle_deadline': _OPENED_AT + timedelta(hours=168),
    }
    fields.update(overrides)
    return ShadowCell(**fields)


class TestShadowCellInvariants:
    """C1's per-row invariants, refused at construction with an informative message."""

    def test_a_valid_cell_constructs(self):
        cell = _cell()
        assert cell.state is ShadowCellState.OPENED

    @pytest.mark.parametrize(
        'state',
        [ShadowCellState.FAILED, ShadowCellState.EXPIRED, ShadowCellState.CAP_EXCLUDED],
    )
    @pytest.mark.parametrize('reason', [None, '', '   '])
    def test_non_settled_terminal_state_requires_a_reason(self, state, reason):
        with pytest.raises(ValueError) as excinfo:
            _cell(state=state, reason=reason)
        message = str(excinfo.value)
        assert '01JCELLAAAAAAAAAAAAAAAAAAA' in message
        assert state.value in message
        assert 'reason' in message

    def test_settled_needs_no_reason(self):
        assert _cell(state=ShadowCellState.SETTLED, reason=None).reason is None

    @pytest.mark.parametrize(
        'field',
        ['owner', 'cell_id', 'project_id', 'task_id', 'candidate', 'incumbent', 'base_sha'],
    )
    def test_empty_identity_string_is_refused(self, field):
        with pytest.raises(ValueError, match=field):
            _cell(**{field: ''})

    @pytest.mark.parametrize('field', ['opened_at', 'settle_deadline', 'running_at', 'settled_at'])
    def test_naive_timestamp_is_refused(self, field):
        naive = {
            'opened_at': datetime(2026, 10, 19, 12, 0),
            'settle_deadline': datetime(2026, 10, 26, 12, 0),
            'running_at': datetime(2026, 10, 19, 12, 5),
            'settled_at': datetime(2026, 10, 20, 12, 0),
        }[field]
        with pytest.raises(ValueError, match=field):
            _cell(**{field: naive})

    @pytest.mark.parametrize('offset', [timedelta(0), timedelta(seconds=-1)])
    def test_settle_deadline_must_lie_ahead_of_opened_at(self, offset):
        with pytest.raises(ValueError, match='settle_deadline'):
            _cell(settle_deadline=_OPENED_AT + offset)

    @pytest.mark.parametrize('field', ['cost_usd', 'cost_credits'])
    def test_negative_cost_is_refused(self, field):
        overrides = {field: -0.01, 'cost_basis': CostBasis.SUBSCRIPTION}
        with pytest.raises(ValueError, match=field):
            _cell(**overrides)

    def test_a_metered_arm_never_carries_credits(self):
        with pytest.raises(ValueError, match='cost_credits'):
            _cell(cost_basis=CostBasis.METERED, cost_credits=1.0)

    def test_a_subscription_arm_carries_credits_and_an_imputed_cost(self):
        cell = _cell(cost_basis=CostBasis.SUBSCRIPTION, cost_credits=1234.5, cost_usd=0.42)
        assert (cell.cost_credits, cell.cost_usd) == (1234.5, 0.42)

    def test_a_cell_is_never_its_own_parent(self):
        with pytest.raises(ValueError, match='parent_cell_id'):
            _cell(parent_cell_id='01JCELLAAAAAAAAAAAAAAAAAAA')

    @pytest.mark.parametrize(
        'stratum',
        [
            ('df', '', 'full'),
            ('df', 'feat×ure', 'full'),
            ('df', 'feature'),
            ('df', 'feature', 'full', 'extra'),
        ],
    )
    def test_malformed_stratum_is_refused(self, stratum):
        with pytest.raises(ValueError, match='stratum'):
            _cell(stratum=stratum)

    @pytest.mark.parametrize('field_name', ['state', 'reason', 'settled_at'])
    def test_the_record_is_frozen(self, field_name):
        cell = _cell()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(cell, field_name, None)

    def test_defaults(self):
        cell = _cell()
        assert cell.owner == 'shadow_coordinator'
        assert cell.cost_usd == 0.0
        assert cell.cost_credits == 0.0

    def test_an_aware_non_utc_timestamp_is_accepted(self):
        plus_two = timezone(timedelta(hours=2))
        cell = _cell(opened_at=_OPENED_AT.astimezone(plus_two))
        assert cell.opened_at == _OPENED_AT


_PARENT_ID = '01JPARENTAAAAAAAAAAAAAAAAA'
_CHILD_ID = '01JCHILDAAAAAAAAAAAAAAAAAA'


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / 'runs.db'


@pytest.fixture
def store(db_path) -> RunStore:
    return RunStore(db_path)


def _parent() -> ShadowCell:
    return _cell(cell_id=_PARENT_ID, shape=ShadowShape.ARCHITECT)


def _child(**overrides) -> ShadowCell:
    fields = {
        'cell_id': _CHILD_ID,
        'shape': ShadowShape.ARCHITECT_CONSEQUENCE,
        'parent_cell_id': _PARENT_ID,
        'worktree_path': Path('/tmp/eval-worktrees/shadow_5382_child'),
        'plan_source': PlanSource.CANDIDATE,
        'cost_basis': CostBasis.SUBSCRIPTION,
        'cost_credits': 1234.5,
        'cost_usd': 0.37,
        'run_id': 'run-abc123',
        'state': ShadowCellState.RUNNING,
        'running_at': _OPENED_AT + timedelta(minutes=5),
    }
    fields.update(overrides)
    return _cell(**fields)


def _raw_row(db_path: Path, cell_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute('SELECT * FROM shadow_cells WHERE cell_id = ?', (cell_id,)).fetchone()
    finally:
        conn.close()


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute('SELECT COUNT(*) FROM shadow_cells').fetchone()[0]
    finally:
        conn.close()


class TestShadowCellsTable:
    """The shadow_cells table through a real RunStore on a tmp runs.db."""

    def test_columns_are_exactly_the_record_fields(self, store, db_path):
        conn = sqlite3.connect(db_path)
        try:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(shadow_cells)')}
        finally:
            conn.close()
        assert columns == {f.name for f in dataclasses.fields(ShadowCell)}

    def test_a_child_cell_round_trips_with_its_parent_edge(self, store):
        parent, child = _parent(), _child()
        store.insert_shadow_cell(parent)
        store.insert_shadow_cell(child)

        read_back = store.get_shadow_cell(_CHILD_ID)

        assert read_back == child
        assert read_back is not None
        assert read_back.parent_cell_id == _PARENT_ID
        assert read_back.worktree_path == Path('/tmp/eval-worktrees/shadow_5382_child')
        assert read_back.cost_credits == 1234.5
        assert read_back.cost_basis is CostBasis.SUBSCRIPTION

    def test_a_terminal_cell_with_every_optional_column_round_trips(self, store):
        cell = _cell(
            state=ShadowCellState.SETTLED,
            reference_kind=ReferenceKind.LANDED,
            reference_sha='b' * 40,
            result_path=Path('/tmp/results/shadow.json'),
            plan_source=PlanSource.PRODUCTION,
            settled_at=_OPENED_AT + timedelta(hours=3),
        )
        store.insert_shadow_cell(cell)
        assert store.get_shadow_cell(cell.cell_id) == cell

    def test_a_minimal_all_defaults_cell_round_trips(self, store):
        cell = _cell()
        store.insert_shadow_cell(cell)
        assert store.get_shadow_cell(cell.cell_id) == cell

    def test_an_unknown_cell_reads_none(self, store):
        assert store.get_shadow_cell('01JNOSUCHCELL') is None

    def test_a_duplicate_cell_id_is_refused_and_the_row_kept(self, store):
        original = _cell()
        store.insert_shadow_cell(original)
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_shadow_cell(_cell(candidate='another-candidate'))
        assert store.get_shadow_cell(original.cell_id) == original

    def test_a_parent_that_does_not_exist_is_refused(self, store, db_path):
        with pytest.raises(ValueError) as excinfo:
            store.insert_shadow_cell(_child())
        assert _CHILD_ID in str(excinfo.value)
        assert _PARENT_ID in str(excinfo.value)
        assert _row_count(db_path) == 0

    @pytest.mark.parametrize(
        'field,value',
        [('project_id', 'reify'), ('task_id', '9999'), ('base_sha', 'c' * 40)],
    )
    def test_a_parent_from_another_task_or_base_is_refused(self, store, db_path, field, value):
        store.insert_shadow_cell(_parent())
        with pytest.raises(ValueError) as excinfo:
            store.insert_shadow_cell(_child(**{field: value}))
        message = str(excinfo.value)
        assert _CHILD_ID in message
        assert _PARENT_ID in message
        assert field in message
        assert _row_count(db_path) == 1

    def test_reconstructing_the_store_keeps_the_rows(self, store, db_path):
        cell = _cell()
        store.insert_shadow_cell(cell)
        assert RunStore(db_path).get_shadow_cell(cell.cell_id) == cell

    def test_a_pre_shadow_runs_db_gains_the_table(self, db_path):
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                'CREATE TABLE task_results (run_id TEXT NOT NULL, task_id TEXT NOT NULL, '
                'project_id TEXT NOT NULL, outcome TEXT NOT NULL, completed_at TEXT)'
            )
            conn.commit()
        finally:
            conn.close()

        store = RunStore(db_path)
        cell = _cell()
        store.insert_shadow_cell(cell)
        assert store.get_shadow_cell(cell.cell_id) == cell

    def test_stored_text_forms_are_bare_values_and_utc_iso(self, store, db_path):
        plus_two = timezone(timedelta(hours=2))
        store.insert_shadow_cell(_cell(opened_at=_OPENED_AT.astimezone(plus_two)))

        row = _raw_row(db_path, '01JCELLAAAAAAAAAAAAAAAAAAA')

        assert row['stratum'] == 'df×feature×full'
        assert row['shape'] == 'implementer'
        assert row['state'] == 'opened'
        assert row['cost_basis'] == 'metered'
        assert row['opened_at'] == '2026-10-19T12:00:00+00:00'
        assert row['settle_deadline'] == '2026-10-26T12:00:00+00:00'


_RAW_COLUMNS = (
    'cell_id', 'parent_cell_id', 'project_id', 'task_id', 'shape', 'candidate', 'incumbent',
    'stratum', 'base_sha', 'state', 'reason', 'cost_basis', 'cost_usd', 'cost_credits',
    'opened_at', 'settle_deadline', 'owner',
)


def _raw_values(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        'cell_id': '01JRAW',
        'parent_cell_id': None,
        'project_id': 'dark_factory',
        'task_id': '5382',
        'shape': 'implementer',
        'candidate': 'c',
        'incumbent': 'i',
        'stratum': 'df×feature×full',
        'base_sha': 'a' * 40,
        'state': 'opened',
        'reason': None,
        'cost_basis': 'metered',
        'cost_usd': 0,
        'cost_credits': 0,
        'opened_at': '2026-10-19T12:00:00+00:00',
        'settle_deadline': '2026-10-26T12:00:00+00:00',
        'owner': 'shadow_coordinator',
    }
    values.update(overrides)
    return values


def _raw_insert(db_path: Path, **overrides) -> None:
    values = _raw_values(**overrides)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f'INSERT INTO shadow_cells ({", ".join(_RAW_COLUMNS)}) '
            f'VALUES ({", ".join("?" for _ in _RAW_COLUMNS)})',
            [values[c] for c in _RAW_COLUMNS],
        )
        conn.commit()
    finally:
        conn.close()


class TestShadowCellsSchemaRedundancy:
    """The DDL refuses rows that bypass ShadowCell: it is the contract every writer meets."""

    @pytest.mark.parametrize(
        'overrides',
        [
            {'owner': None},
            {'settle_deadline': None},
            {'cost_basis': None},
            {'state': 'failed', 'reason': None},
            {'state': 'failed', 'reason': ''},
            {'state': 'cap_excluded', 'reason': '  '},
            {'cost_basis': 'metered', 'cost_credits': 5.0},
            {'cost_basis': 'subscription', 'cost_usd': -1.0},
            {'cost_basis': 'subscription', 'cost_credits': -1.0},
            {'owner': ''},
            {'parent_cell_id': '01JRAW'},
        ],
        ids=lambda o: ','.join(f'{k}={v!r}' for k, v in o.items()),
    )
    def test_a_raw_insert_violating_c1_is_refused(self, store, db_path, overrides):
        with pytest.raises(sqlite3.IntegrityError):
            _raw_insert(db_path, **overrides)
        assert _row_count(db_path) == 0

    def test_a_raw_update_to_expired_without_reason_is_refused(self, store, db_path):
        store.insert_shadow_cell(_cell())
        conn = sqlite3.connect(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE shadow_cells SET state = 'expired', reason = NULL WHERE cell_id = ?",
                    ('01JCELLAAAAAAAAAAAAAAAAAAA',),
                )
        finally:
            conn.close()

    def test_settled_without_reason_is_accepted(self, store, db_path):
        _raw_insert(db_path, state='settled', reason=None)
        assert _row_count(db_path) == 1

    def test_an_unknown_stored_enum_value_is_refused_on_read(self, store, db_path):
        _raw_insert(db_path, shape='reviewer')
        with pytest.raises(ValueError) as excinfo:
            store.get_shadow_cell('01JRAW')
        message = str(excinfo.value)
        assert '01JRAW' in message
        assert 'shape' in message
        assert 'reviewer' in message
