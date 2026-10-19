"""Tests for ``scripts/deferral_census.py``: every deferred row classified with
the shared deferral model.

The classifier section is pure, over ``deferral_store.TaskRow.from_wire`` rows
and a fixed NOW. Every record is built with ``shared.task_deferral``'s own
models and ``record_to_json``, never hand-written JSON. The CLI section
(boundary row 22) drives ``main(argv)`` over ``FakeFusedMemory``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import deferral_census
import pytest
from deferral_census import RowVerdict, classify_deferred, fails_check, referents
from deferral_fake_fused_memory import FakeFusedMemory
from deferral_store import TaskRow
from shared.process_identity import Liveness, ProcessIdentity, capture
from shared.task_deferral import (
    LAPSE_OVERDUE_AFTER_SECS,
    CarriedByDeferral,
    Deferral,
    HeldBySessionDeferral,
    LegacyUnknownDeferral,
    NeedsHumanReason,
    PlanningDeferral,
    UntilConditionDeferral,
    record_to_json,
)
from shared.task_statuses import TaskStatus

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
HOLDER = ProcessIdentity(pid=4242, start_ticks=99, boot_id='boot-a', host='factory-host')
CARRIER_STATUSES: Mapping[int, TaskStatus] = {
    900: TaskStatus.DONE,
    901: TaskStatus.CANCELLED,
    902: TaskStatus.PENDING,
}


def _stamps(stamped_at: datetime = NOW - DAY) -> dict[str, Any]:
    return {'deferred_at': stamped_at, 'stamped_at': stamped_at, 'stamped_by': 'claude-task-1'}


def carried_by(carrier: int) -> Deferral:
    return CarriedByDeferral(carrier_task_id=carrier, **_stamps())


def until_condition(*, expires_at: datetime | None, stamped_at: datetime = NOW - DAY) -> Deferral:
    return UntilConditionDeferral(condition='upstream lands', expires_at=expires_at, **_stamps(stamped_at))


def held_by_session(holder: ProcessIdentity = HOLDER) -> Deferral:
    return HeldBySessionDeferral(holder=holder, **_stamps())


def planning(*, stamped_at: datetime) -> Deferral:
    return PlanningDeferral(**_stamps(stamped_at))


def legacy_unknown() -> Deferral:
    return LegacyUnknownDeferral(deferred_at=None, stamped_at=NOW - DAY, stamped_by='migrate-deferrals')


def wire_row(
    task_id: int,
    *,
    record: Deferral | None = None,
    status: str = 'deferred',
    metadata: object = None,
    **extra: Any,
) -> dict[str, Any]:
    if metadata is None:
        metadata = dict(extra)
        if record is not None:
            metadata['deferral'] = record_to_json(record)
    return {'id': str(task_id), 'title': f'task {task_id}', 'status': status, 'metadata': metadata}


def row(task_id: int, record: Deferral | None = None, **kwargs: Any) -> TaskRow:
    return TaskRow.from_wire(wire_row(task_id, record=record, **kwargs))


def classify_one(
    task_row: TaskRow, *, liveness: Mapping[int, Liveness | None] | None = None
) -> RowVerdict:
    (verdict,) = classify_deferred(
        [task_row], now=NOW, carrier_statuses=CARRIER_STATUSES, holder_liveness=liveness or {}
    )
    return verdict


class TestClassifyDeferred:
    @pytest.mark.parametrize(
        ('carrier', 'expected'),
        [
            (900, NeedsHumanReason.CLOSABLE),
            (901, NeedsHumanReason.ORPHANED),
            (902, None),
            (903, NeedsHumanReason.CARRIER_MISSING),
        ],
    )
    def test_carried_by_reads_its_carriers_status(
        self, carrier: int, expected: NeedsHumanReason | None
    ) -> None:
        verdict = classify_one(row(10, carried_by(carrier)))

        assert verdict.held == 'carried_by'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('record', 'expected'),
        [
            (until_condition(expires_at=NOW + DAY), None),
            (
                until_condition(expires_at=NOW - timedelta(seconds=LAPSE_OVERDUE_AFTER_SECS + 1)),
                NeedsHumanReason.EXPIRY_OVERDUE,
            ),
            (until_condition(expires_at=None, stamped_at=NOW - 31 * DAY), NeedsHumanReason.STALE_CONDITION),
        ],
        ids=['future-expiry', 'expiry-overdue', 'stale-condition'],
    )
    def test_until_condition(self, record: Deferral, expected: NeedsHumanReason | None) -> None:
        verdict = classify_one(row(11, record))

        assert verdict.held == 'until_condition'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('liveness', 'expected'),
        [
            ({12: Liveness.ALIVE}, None),
            ({12: Liveness.DEAD}, NeedsHumanReason.HOLDER_DEAD),
            ({12: Liveness.OTHER_HOST}, NeedsHumanReason.HOLDER_OTHER_HOST),
            ({12: None}, NeedsHumanReason.LIVENESS_UNKNOWN),
            ({}, NeedsHumanReason.LIVENESS_UNKNOWN),
        ],
        ids=['alive', 'dead', 'other-host', 'unread', 'absent'],
    )
    def test_held_by_session_reads_its_holders_liveness(
        self, liveness: Mapping[int, Liveness | None], expected: NeedsHumanReason | None
    ) -> None:
        verdict = classify_one(row(12, held_by_session()), liveness=liveness)

        assert verdict.held == 'held_by_session'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('age', 'expected'),
        [(timedelta(minutes=5), None), (timedelta(hours=25), NeedsHumanReason.STALE_PLANNING)],
        ids=['fresh', 'stale'],
    )
    def test_planning(self, age: timedelta, expected: NeedsHumanReason | None) -> None:
        verdict = classify_one(row(13, planning(stamped_at=NOW - age)))

        assert verdict.held == 'planning'
        assert verdict.needs_human is expected

    def test_legacy_unknown_needs_a_human(self) -> None:
        verdict = classify_one(row(14, legacy_unknown()))

        assert verdict.held == 'legacy_unknown'
        assert verdict.needs_human is NeedsHumanReason.LEGACY_UNKNOWN
        assert verdict.deflake_owner is False

    def test_a_deflake_owner_is_marked(self) -> None:
        verdict = classify_one(row(15, legacy_unknown(), spawn_context=deferral_census.DEFLAKE_SPAWN_CONTEXT))

        assert verdict.needs_human is NeedsHumanReason.LEGACY_UNKNOWN
        assert verdict.deflake_owner is True

    def test_no_record_is_missing(self) -> None:
        verdict = classify_one(row(16, x_coalesced_into=5255))

        assert verdict.held == 'no record'
        assert verdict.needs_human is NeedsHumanReason.MISSING_RECORD

    def test_a_malformed_record_is_invalid(self) -> None:
        malformed = TaskRow.from_wire(
            wire_row(17, metadata={'deferral': {'kind': 'carried_by', 'carrier_task_id': 'soon'}})
        )

        verdict = classify_one(malformed)

        assert verdict.held == 'invalid record'
        assert verdict.needs_human is NeedsHumanReason.INVALID_RECORD

    @pytest.mark.parametrize('metadata', ['[1, 2]', '"text"', '{not json'])
    def test_unreadable_wire_metadata_is_an_invalid_record(self, metadata: str) -> None:
        verdict = classify_one(TaskRow.from_wire(wire_row(18, metadata=metadata)))

        assert verdict.held == 'invalid record'
        assert verdict.needs_human is NeedsHumanReason.INVALID_RECORD
        assert verdict.deflake_owner is False

    def test_a_json_object_string_is_read_as_metadata(self) -> None:
        text = json.dumps({'deferral': record_to_json(legacy_unknown())})

        verdict = classify_one(TaskRow.from_wire(wire_row(19, metadata=text)))

        assert verdict.held == 'legacy_unknown'

    def test_one_verdict_per_row_in_order_carrying_id_and_title(self) -> None:
        rows = [row(21, carried_by(900)), row(20), row(22, legacy_unknown())]

        verdicts = classify_deferred(rows, now=NOW, carrier_statuses=CARRIER_STATUSES, holder_liveness={})

        assert [(v.task_id, v.title) for v in verdicts] == [
            (21, 'task 21'),
            (20, 'task 20'),
            (22, 'task 22'),
        ]


class TestFailsCheck:
    FAILING = frozenset(
        {NeedsHumanReason.MISSING_RECORD, NeedsHumanReason.INVALID_RECORD, NeedsHumanReason.EXPIRY_OVERDUE}
    )

    @pytest.mark.parametrize('reason', [*NeedsHumanReason, None])
    def test_only_missing_invalid_and_overdue_fail(self, reason: NeedsHumanReason | None) -> None:
        verdict = RowVerdict(task_id=1, title='t', held='x', needs_human=reason, deflake_owner=False)

        assert fails_check(verdict) is (reason in self.FAILING)


class TestReferents:
    def test_carriers_and_holders_of_the_recorded_rows(self) -> None:
        rows = [
            row(30, carried_by(900)),
            row(31, carried_by(901)),
            row(32, carried_by(900)),
            row(33, held_by_session()),
            row(34, until_condition(expires_at=NOW + DAY)),
            row(35),
            TaskRow.from_wire(wire_row(36, metadata='[]')),
        ]

        found = referents(rows)

        assert found.carrier_ids == frozenset({900, 901})
        assert dict(found.holders) == {33: HOLDER}


class TestHolderLiveness:
    def test_an_unreadable_proc_is_unknown_never_dead(self, tmp_path: Path) -> None:
        identity = capture(os.getpid())
        assert identity is not None
        empty_proc = tmp_path / 'proc'
        empty_proc.mkdir()

        assert deferral_census.holder_liveness(identity, proc_root=empty_proc) is None

    def test_this_process_is_alive(self) -> None:
        identity = capture(os.getpid())
        assert identity is not None

        assert deferral_census.holder_liveness(identity) is Liveness.ALIVE


class TestTaskRowFromWire:
    def test_parses_id_status_and_metadata(self) -> None:
        parsed = TaskRow.from_wire({'id': 7, 'title': 'seven', 'status': 'pending', 'metadata': {'a': 1}})

        assert (parsed.task_id, parsed.title, parsed.status) == (7, 'seven', TaskStatus.PENDING)
        assert parsed.metadata == {'a': 1}
        assert parsed.deferral is None

    @pytest.mark.parametrize(
        ('field', 'value'),
        [('id', 'abc'), ('id', '1.5'), ('id', True), ('status', 'parked'), ('title', None)],
    )
    def test_an_unusable_field_is_a_value_error_naming_it(self, field: str, value: object) -> None:
        wire = wire_row(1)
        wire[field] = value

        with pytest.raises(ValueError, match=field):
            TaskRow.from_wire(wire)


# ---------------------------------------------------------------------------
# The CLI, boundary row 22: main(argv) in-process over FakeFusedMemory. Every
# run names --units-dir under tmp_path, so the host's units are never read.
# ---------------------------------------------------------------------------

CARRIER_DONE, CARRIER_CANCELLED, CARRIER_PENDING = 201, 202, 203
UNRECORDED, STRAY = 109, 110
LEGACY, DEFLAKE_OWNER = 107, 108


def _recorded(task_id: int, record: Deferral, **extra: Any) -> dict[str, Any]:
    return wire_row(task_id, record=record, **extra)


def row22_fixture() -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    live_holder = capture(os.getpid())
    assert live_holder is not None
    return [
        _recorded(101, carried_by(CARRIER_DONE)),
        _recorded(102, carried_by(CARRIER_CANCELLED)),
        _recorded(103, carried_by(CARRIER_PENDING)),
        _recorded(104, until_condition(expires_at=now + DAY)),
        _recorded(105, held_by_session(live_holder)),
        _recorded(106, planning(stamped_at=now - timedelta(minutes=5))),
        _recorded(LEGACY, legacy_unknown()),
        _recorded(DEFLAKE_OWNER, legacy_unknown(), spawn_context=deferral_census.DEFLAKE_SPAWN_CONTEXT),
        wire_row(UNRECORDED, x_coalesced_into=5255),
        wire_row(STRAY, status='pending', metadata={'deferral': record_to_json(legacy_unknown())}),
        wire_row(CARRIER_DONE, status='done'),
        wire_row(CARRIER_CANCELLED, status='cancelled'),
        wire_row(CARRIER_PENDING, status='pending'),
    ]


def _needs_human_count(rows: list[dict[str, Any]]) -> int:
    deferred = [TaskRow.from_wire(r) for r in rows if r['status'] == 'deferred']
    found = referents(deferred)
    statuses = {TaskRow.from_wire(r).task_id: TaskStatus(r['status']) for r in rows}
    liveness = {task_id: deferral_census.holder_liveness(holder) for task_id, holder in found.holders.items()}
    verdicts = classify_deferred(
        deferred, now=datetime.now(UTC), carrier_statuses=statuses, holder_liveness=liveness
    )
    return sum(verdict.needs_human is not None for verdict in verdicts)


class Census:
    """One main() run's exit code and output, split into per-project blocks."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.code, self.out, self.err = code, out, err

    def block(self, root: Path) -> list[str]:
        lines = self.out.splitlines()
        (start,) = [n for n, line in enumerate(lines) if line.startswith(str(root)) and not line.startswith(' ')]
        end = next((n for n in range(start + 1, len(lines)) if not lines[n].startswith(' ')), len(lines))
        return lines[start:end]

    def header(self, root: Path) -> str:
        return self.block(root)[0]

    def failing(self, root: Path) -> list[str]:
        return [line.strip() for line in self.block(root) if line.strip().startswith('FAIL ')]

    def listed(self, root: Path, label: str) -> set[int]:
        prefix = f'{label}: '
        (line,) = [line.strip() for line in self.block(root) if line.strip().startswith(prefix)]
        return {int(task_id) for task_id in line.removeprefix(prefix).split(', ')}


@pytest.fixture
def units_dir(tmp_path: Path) -> Path:
    units = tmp_path / 'units'
    units.mkdir()
    return units


@pytest.fixture
def make_root(tmp_path: Path, project_root_with_tasks_db: Any) -> Any:
    def _make(name: str) -> Path:
        root = tmp_path.resolve() / name
        project_root_with_tasks_db(root)
        return root

    return _make


@pytest.fixture
def census(capsys: pytest.CaptureFixture[str], units_dir: Path) -> Any:
    def _run(*argv: str) -> Census:
        code = deferral_census.main(['--fused-memory-url', 'http://fake', '--units-dir', str(units_dir), *argv])
        out, err = capsys.readouterr()
        return Census(code, out, err)

    return _run


@pytest.fixture
def serve(install_fake_httpx: Any) -> Any:
    def _serve(projects: Mapping[str, list[dict[str, Any]]]) -> FakeFusedMemory:
        fake = FakeFusedMemory(projects)
        install_fake_httpx(fake.post)
        return fake

    return _serve


def _write_unit(units_dir: Path, name: str, root: Path) -> None:
    (units_dir / f'orchestrator-{name}.service').write_text(
        '[Service]\n'
        f'ExecStart=/usr/bin/uv run orchestrator run --config {root / "dark-factory-orchestrator.yaml"}\n'
    )


class TestCensusCheck:
    def test_check_fails_naming_the_unrecorded_and_the_stray_row(self, census: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        serve({str(root): row22_fixture()})

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.VIOLATIONS
        failing = result.failing(root)
        assert any(f'#{UNRECORDED}' in line and 'missing_record' in line for line in failing)
        assert any(f'#{STRAY}' in line and 'pending' in line for line in failing)
        assert len(failing) == 2

    def test_after_the_fixes_check_passes_and_lists_what_needs_a_human(
        self, census: Any, serve: Any, make_root: Any
    ) -> None:
        root = make_root('A')
        rows = row22_fixture()
        fake = serve({str(root): rows})
        fake.row(str(root), UNRECORDED)['metadata']['deferral'] = record_to_json(legacy_unknown())
        del fake.row(str(root), STRAY)['metadata']['deferral']

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.OK
        assert result.failing(root) == []
        assert result.listed(root, 'legacy_unknown') == {LEGACY, UNRECORDED}
        assert result.listed(root, 'legacy_unknown (de-flake owner)') == {DEFLAKE_OWNER}
        assert result.listed(root, 'closable') == {101}
        assert result.listed(root, 'orphaned') == {102}
        assert f'needs human: {_needs_human_count(fake.projects[str(root)])}' in [
            line.strip() for line in result.block(root)
        ]
        assert _needs_human_count(fake.projects[str(root)]) == 5

    def test_without_check_violations_are_printed_and_the_exit_is_ok(
        self, census: Any, serve: Any, make_root: Any
    ) -> None:
        root = make_root('A')
        serve({str(root): row22_fixture()})

        result = census('--project-root', str(root))

        assert result.code == deferral_census.CensusExit.OK
        assert len(result.failing(root)) == 2

    def test_an_invalid_record_fails_check_naming_its_detail(self, census: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        malformed = wire_row(120, metadata={'deferral': {'kind': 'carried_by', 'carrier_task_id': 'soon'}})
        serve({str(root): [malformed]})

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.VIOLATIONS
        (line,) = result.failing(root)
        assert '#120' in line and 'invalid_record' in line and 'carrier_task_id' in line

    def test_an_overdue_expiry_fails_check(self, census: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        overdue = datetime.now(UTC) - timedelta(seconds=LAPSE_OVERDUE_AFTER_SECS + 60)
        serve({str(root): [_recorded(121, until_condition(expires_at=overdue))]})

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.VIOLATIONS
        (line,) = result.failing(root)
        assert '#121' in line and 'expiry_overdue' in line

    def test_a_stale_condition_is_listed_and_passes(self, census: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        stale = until_condition(expires_at=None, stamped_at=datetime.now(UTC) - 31 * DAY)
        serve({str(root): [_recorded(122, stale)]})

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.OK
        assert result.listed(root, 'stale_condition') == {122}

    def test_deferred_rows_with_no_record_anywhere_never_read_clean(
        self, census: Any, serve: Any, make_root: Any
    ) -> None:
        root = make_root('A')
        serve({str(root): [wire_row(130), wire_row(131, x_armed_by='later')]})

        result = census('--project-root', str(root))

        assert 'no deferral producer has written' in result.out
        assert 'clean' not in result.out.lower()

    def test_a_project_with_records_and_nothing_failing_reads_clean(
        self, census: Any, serve: Any, make_root: Any
    ) -> None:
        root = make_root('A')
        serve({str(root): [_recorded(140, carried_by(CARRIER_PENDING)), wire_row(CARRIER_PENDING, status='pending')]})

        result = census('--project-root', str(root), '--check')

        assert result.code == deferral_census.CensusExit.OK
        assert any('clean' in line for line in result.block(root))


class TestCensusProjects:
    def test_an_unreadable_project_is_named_and_the_rest_still_report(
        self, census: Any, serve: Any, make_root: Any
    ) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): row22_fixture(), str(root_b): []})
        fake.unreadable.add(str(root_b))

        result = census('--project-root', str(root_a), '--project-root', str(root_b), '--check')

        assert result.code == deferral_census.CensusExit.UNREADABLE
        (unreadable,) = [line for line in result.block(root_b) if 'UNREADABLE' in line]
        assert 'is not a known project' in unreadable
        assert len(result.failing(root_a)) == 2
        assert result.listed(root_a, 'closable') == {101}

    def test_a_root_with_no_task_store_is_named_and_never_asked(
        self, census: Any, serve: Any, make_root: Any, tmp_path: Path
    ) -> None:
        root_a = make_root('A')
        bare = tmp_path.resolve() / 'not-a-project'
        bare.mkdir()
        fake = serve({str(root_a): row22_fixture(), str(bare): []})

        result = census('--project-root', str(root_a), '--project-root', str(bare))

        assert result.code == deferral_census.CensusExit.UNREADABLE
        assert any('UNKNOWN PROJECT' in line for line in result.block(bare))
        assert all(arguments.get('project_root') != str(bare) for _, arguments in fake.calls)

    def test_with_no_project_root_every_unit_root_is_reported_with_its_sweep(
        self, census: Any, serve: Any, make_root: Any, units_dir: Path
    ) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        _write_unit(units_dir, 'a', root_a)
        _write_unit(units_dir, 'b', root_b)
        serve({str(root_a): row22_fixture(), str(root_b): [wire_row(1, record=legacy_unknown())]})

        result = census()

        assert result.code == deferral_census.CensusExit.OK
        assert 'sweep: orchestrator-a.service' in result.header(root_a)
        assert 'sweep: orchestrator-b.service' in result.header(root_b)

    def test_a_hand_root_with_no_unit_is_surfaced_only(
        self, census: Any, serve: Any, make_root: Any, units_dir: Path
    ) -> None:
        root_a, root_c = make_root('A'), make_root('C')
        _write_unit(units_dir, 'a', root_a)
        serve({str(root_a): [], str(root_c): []})

        result = census('--project-root', str(root_c))

        assert 'no sweep: surfaced only' in result.header(root_c)
        assert str(root_a) not in result.out

    def test_no_unit_and_no_root_is_unreadable_naming_the_units_dir(
        self, census: Any, serve: Any, units_dir: Path
    ) -> None:
        serve({})

        result = census()

        assert result.code == deferral_census.CensusExit.UNREADABLE
        assert str(units_dir) in result.err

    def test_every_call_names_an_absolute_project_root(
        self, census: Any, serve: Any, make_root: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = make_root('A')
        fake = serve({str(root_a): row22_fixture()})
        monkeypatch.chdir(tmp_path)

        census('--project-root', 'A')

        assert fake.calls
        assert all(Path(arguments['project_root']).is_absolute() for _, arguments in fake.calls)
        assert {arguments['project_root'] for _, arguments in fake.calls} == {str(root_a)}
