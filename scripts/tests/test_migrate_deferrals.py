"""Tests for ``scripts/migrate_deferrals.py``: every unrecorded deferred row
mapped to the record it should carry, and re-stamped through the guarded write.

The mapping section is pure, over ``deferral_store.TaskRow.from_wire`` rows and
a fixed NOW. Records are built with ``shared.task_deferral``'s own models. The
CLI section (boundary row 23) drives ``main(argv)`` over ``FakeFusedMemory``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import migrate_deferrals
import pytest
from deferral_fake_fused_memory import FakeFusedMemory
from deferral_store import TaskRow
from migrate_deferrals import (
    AlreadyRecorded,
    Fallback,
    RowPlan,
    Source,
    carrier_reason,
    fallback_plan,
    plan_row,
)
from shared.task_deferral import (
    REASON_MAX_CHARS,
    DeferralInvalidReason,
    DeferralKind,
    NeedsHumanReason,
    PlanningDeferral,
    parse_request,
    record_to_json,
)
from shared.task_statuses import TaskStatus

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
HUMAN_DECOMPOSED_NOTE = 'human_decomposed: possibly an uncommitted planning birth'


def wire_row(task_id: int, *, status: str = 'deferred', metadata: object = None, **extra: Any) -> dict[str, Any]:
    return {
        'id': str(task_id),
        'title': f'task {task_id}',
        'status': status,
        'metadata': dict(extra) if metadata is None else metadata,
    }


def row(task_id: int = 1, **metadata: Any) -> TaskRow:
    return TaskRow.from_wire(wire_row(task_id, **metadata))


def planned(task_row: TaskRow) -> RowPlan:
    plan = plan_row(task_row, now=NOW)
    assert isinstance(plan, RowPlan)
    return plan


def planning_record() -> dict[str, Any]:
    return record_to_json(PlanningDeferral(deferred_at=NOW, stamped_at=NOW, stamped_by='claude'))


class TestPlanRow:
    @pytest.mark.parametrize('carrier', [5255, '5255'])
    def test_x_coalesced_into_maps_to_carried_by(self, carrier: object) -> None:
        plan = planned(row(7, x_coalesced_into=carrier))

        assert plan.request['kind'] == DeferralKind.CARRIED_BY
        assert plan.request['carrier_task_id'] == 5255
        assert plan.source == Source.X_COALESCED_INTO == 'x_coalesced_into'
        assert plan.fallback is None
        assert plan.request['reason'] == 'migrated from metadata.x_coalesced_into'
        assert (plan.task_id, plan.title) == (7, 'task 7')

    def test_a_deferred_watch_trigger_maps_to_until_condition(self) -> None:
        plan = planned(row(deferred_watch=True, trigger='the upstream API ships'))

        assert plan.request['kind'] == DeferralKind.UNTIL_CONDITION
        assert plan.request['condition'] == 'the upstream API ships'
        assert plan.source == Source.TRIGGER == 'trigger'

    def test_x_armed_by_maps_to_until_condition(self) -> None:
        plan = planned(row(x_armed_by='esc-4827 is ruled'))

        assert plan.request['kind'] == DeferralKind.UNTIL_CONDITION
        assert plan.request['condition'] == 'esc-4827 is ruled'
        assert plan.source == Source.X_ARMED_BY == 'x_armed_by'

    def test_x_coalesced_into_wins_over_trigger_and_x_armed_by(self) -> None:
        plan = planned(row(x_coalesced_into=5255, deferred_watch=True, trigger='t', x_armed_by='a'))

        assert plan.source == Source.X_COALESCED_INTO

    def test_trigger_wins_over_x_armed_by(self) -> None:
        plan = planned(row(deferred_watch=True, trigger='t', x_armed_by='a'))

        assert plan.source == Source.TRIGGER
        assert plan.request['condition'] == 't'

    @pytest.mark.parametrize(
        'metadata',
        [
            {'x_armed_by': None},
            {'deferred_watch': None, 'trigger': 't'},
            {'x_coalesced_into': None},
            {'deferred_watch': True},
            {'deferred_watch': True, 'trigger': None},
            {},
        ],
        ids=['armed-null', 'watch-null', 'coalesced-null', 'watch-no-trigger', 'trigger-null', 'bare'],
    )
    def test_values_not_keys_decide_and_nothing_maps_to_legacy_unknown(self, metadata: dict[str, Any]) -> None:
        plan = planned(row(**metadata))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.source is None
        assert plan.fallback is None

    @pytest.mark.parametrize(('flag', 'noted'), [(True, True), ('yes', True), (False, False), (None, False)])
    def test_legacy_unknown_notes_a_human_decomposed_row(self, flag: object, noted: bool) -> None:
        plan = planned(row(human_decomposed=flag))

        assert (HUMAN_DECOMPOSED_NOTE in plan.request['reason']) is noted

    def test_a_mapped_request_carries_no_human_decomposed_note(self) -> None:
        plan = planned(row(x_coalesced_into=5255, human_decomposed=True))

        assert HUMAN_DECOMPOSED_NOTE not in plan.request['reason']

    def test_a_row_with_a_valid_record_is_already_recorded(self) -> None:
        plan = plan_row(row(8, deferral=planning_record(), x_coalesced_into=5255), now=NOW)

        assert plan == AlreadyRecorded(task_id=8, title='task 8', kind=DeferralKind.PLANNING)

    def test_a_row_with_an_invalid_record_is_planned(self) -> None:
        plan = planned(row(9, deferral={'kind': 'carried_by'}, x_coalesced_into=5255))

        assert plan.request['kind'] == DeferralKind.CARRIED_BY

    def test_unreadable_metadata_is_planned_legacy_unknown(self) -> None:
        plan = planned(TaskRow.from_wire(wire_row(10, metadata='[1]')))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN

    def test_an_oversized_condition_is_a_predicted_fallback(self) -> None:
        text = 'armed by ' + 'x' * 3000

        plan = planned(row(11, x_armed_by=text))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.source == Source.X_ARMED_BY
        assert plan.fallback == Fallback(
            source=Source.X_ARMED_BY, value=text, reason_code=DeferralInvalidReason.TEXT_TOO_LONG
        )
        reason = plan.request['reason']
        assert len(reason) <= REASON_MAX_CHARS
        assert 'x_armed_by' in reason and 'text_too_long' in reason and text[:40] in reason

    def test_a_non_decimal_carrier_is_a_predicted_fallback(self) -> None:
        plan = planned(row(12, x_coalesced_into='soon'))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.fallback is not None
        assert plan.fallback.source == Source.X_COALESCED_INTO
        assert '"soon"' in plan.request['reason']

    @pytest.mark.parametrize(
        'task_row',
        [
            row(x_coalesced_into=5255),
            row(x_coalesced_into='5255'),
            row(deferred_watch=True, trigger='t'),
            row(x_armed_by='a'),
            row(human_decomposed=True),
            row(x_armed_by='y' * 3000, human_decomposed=True),
            row(),
        ],
    )
    def test_every_planned_request_round_trips_the_shared_parser(self, task_row: TaskRow) -> None:
        request = planned(task_row).request

        parsed = parse_request(request, now=NOW)

        assert parsed.model_dump(mode='json', exclude_none=True) == request
        assert json.loads(json.dumps(request)) == request


class TestFallbackPlan:
    def test_a_server_refusal_falls_back_to_legacy_unknown_naming_source_and_value(self) -> None:
        plan = planned(row(13, x_coalesced_into=5255, human_decomposed=True))

        fallback = fallback_plan(plan, reason_code=DeferralInvalidReason.CARRIER_NOT_FOUND)

        assert (fallback.task_id, fallback.title, fallback.source) == (13, 'task 13', Source.X_COALESCED_INTO)
        assert fallback.fallback == Fallback(
            source=Source.X_COALESCED_INTO, value=5255, reason_code=DeferralInvalidReason.CARRIER_NOT_FOUND
        )
        assert fallback.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        reason = fallback.request['reason']
        assert 'x_coalesced_into' in reason and '5255' in reason and 'carrier_not_found' in reason
        assert HUMAN_DECOMPOSED_NOTE in reason
        assert parse_request(fallback.request, now=NOW).model_dump(mode='json', exclude_none=True) == fallback.request

    def test_the_refused_value_is_the_source_value_as_read(self) -> None:
        plan = planned(row(14, deferred_watch=True, trigger='the API ships'))

        fallback = fallback_plan(plan, reason_code=DeferralInvalidReason.TEXT_TOO_LONG)

        assert fallback.fallback is not None
        assert fallback.fallback.value == 'the API ships'
        assert '"the API ships"' in fallback.request['reason']


class TestCarrierReason:
    STATUSES = {900: TaskStatus.DONE, 901: TaskStatus.CANCELLED, 902: TaskStatus.PENDING}

    @pytest.mark.parametrize(
        ('carrier', 'expected'),
        [
            (900, NeedsHumanReason.CLOSABLE),
            (901, NeedsHumanReason.ORPHANED),
            (903, NeedsHumanReason.CARRIER_MISSING),
            (902, None),
        ],
    )
    def test_a_carried_by_plan_reads_its_carriers_status(
        self, carrier: int, expected: NeedsHumanReason | None
    ) -> None:
        plan = planned(row(x_coalesced_into=carrier))

        assert carrier_reason(plan, self.STATUSES, now=NOW) is expected

    @pytest.mark.parametrize('task_row', [row(x_armed_by='a'), row()], ids=['until-condition', 'legacy'])
    def test_any_other_plan_has_no_carrier_reason(self, task_row: TaskRow) -> None:
        assert carrier_reason(planned(task_row), self.STATUSES, now=NOW) is None

    def test_the_migration_agent_id_is_named(self) -> None:
        assert migrate_deferrals.MIGRATION_AGENT_ID == 'migrate-deferrals'


# ---------------------------------------------------------------------------
# The CLI, boundary row 23: main(argv) in-process over FakeFusedMemory across
# two project roots. Every run names --units-dir under tmp_path.
# ---------------------------------------------------------------------------

CARRIER_DONE, CARRIER_CANCELLED, CARRIER_PENDING = 401, 402, 403
RECORDED = 308
CLASSES_A = {
    'carried_by': {301, 302, 303},
    'until_condition': {304, 305},
    'legacy_unknown': {306, 307},
    'already recorded': {RECORDED},
}
MAPPED_KINDS_A = {301: 'carried_by', 302: 'carried_by', 303: 'carried_by', 304: 'until_condition',
                  305: 'until_condition', 306: 'legacy_unknown', 307: 'legacy_unknown'}


def project_a() -> list[dict[str, Any]]:
    return [
        wire_row(301, x_coalesced_into=CARRIER_DONE),
        wire_row(302, x_coalesced_into=CARRIER_CANCELLED),
        wire_row(303, x_coalesced_into=CARRIER_PENDING),
        wire_row(304, x_armed_by='esc-4827 is ruled'),
        wire_row(305, deferred_watch=True, trigger='the upstream API ships'),
        wire_row(306, human_decomposed=True),
        wire_row(307, x_armed_by=None, deferred_watch=None),
        wire_row(RECORDED, deferral=planning_record()),
        wire_row(CARRIER_DONE, status='done'),
        wire_row(CARRIER_CANCELLED, status='cancelled'),
        wire_row(CARRIER_PENDING, status='pending'),
    ]


def project_b() -> list[dict[str, Any]]:
    return [wire_row(501, x_coalesced_into=601), wire_row(502), wire_row(601, status='pending')]


class Migration:
    """One main() run's exit code and output, split into per-project blocks."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.code, self.out, self.err = code, out, err

    def block(self, root: Path) -> list[str]:
        lines = self.out.splitlines()
        (start,) = [n for n, line in enumerate(lines) if line.startswith(str(root)) and not line.startswith(' ')]
        end = next((n for n in range(start + 1, len(lines)) if not lines[n].startswith(' ')), len(lines))
        return [line.strip() for line in lines[start:end]]

    def ids(self, root: Path, label: str) -> set[int]:
        (line,) = [line for line in self.block(root) if line.startswith(f'{label}: ') or line.startswith(f'{label} (')]
        listed = line.split(': ', 1)[1]
        return {int(task_id) for task_id in listed.split(', ')}

    def classes(self, root: Path) -> dict[str, set[int]]:
        found = {}
        for label in ('carried_by', 'until_condition', 'legacy_unknown', 'already recorded'):
            if any(line.startswith(f'{label} (') for line in self.block(root)):
                found[label] = self.ids(root, label)
        return found

    def count(self, root: Path, label: str) -> int:
        (line,) = [line for line in self.block(root) if line.startswith(f'{label}: ')]
        return int(line.split(': ', 1)[1])

    def lines_with(self, root: Path, prefix: str) -> list[str]:
        return [line for line in self.block(root) if line.startswith(prefix)]


@pytest.fixture
def units_dir(tmp_path: Path) -> Path:
    units = tmp_path / 'units'
    units.mkdir()
    return units


@pytest.fixture
def make_root(tmp_path: Path, project_root_with_tasks_db: Callable[[Path], Path]) -> Callable[[str], Path]:
    def _make(name: str) -> Path:
        root = tmp_path.resolve() / name
        project_root_with_tasks_db(root)
        return root

    return _make


@pytest.fixture
def migrate(capsys: pytest.CaptureFixture[str], units_dir: Path) -> Callable[..., Migration]:
    def _run(*argv: str) -> Migration:
        code = migrate_deferrals.main(['--fused-memory-url', 'http://fake', '--units-dir', str(units_dir), *argv])
        out, err = capsys.readouterr()
        return Migration(code, out, err)

    return _run


@pytest.fixture
def serve(install_fake_httpx: Callable[..., Any]) -> Callable[..., FakeFusedMemory]:
    def _serve(projects: Mapping[str, list[dict[str, Any]]], *, echo: bool = True) -> FakeFusedMemory:
        fake = FakeFusedMemory(projects, echo=echo)
        install_fake_httpx(fake.post)
        return fake

    return _serve


def _roots(*roots: Path) -> list[str]:
    return [argument for root in roots for argument in ('--project-root', str(root))]


def _planned_request(wire: dict[str, Any]) -> dict[str, Any]:
    return planned(TaskRow.from_wire(wire)).request


def _record_kind(fake: FakeFusedMemory, root: Path, task_id: int) -> str | None:
    record = fake.row(str(root), task_id)['metadata'].get('deferral')
    return None if record is None else record['kind']


class TestMigrationRun:
    def test_a_dry_run_prints_the_mapping_per_project_and_writes_nothing(
        self, migrate: Any, serve: Any, make_root: Any
    ) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): project_a(), str(root_b): project_b()})

        result = migrate(*_roots(root_a, root_b))

        assert result.code == migrate_deferrals.MigrationExit.OK
        assert result.classes(root_a) == CLASSES_A
        assert result.classes(root_b) == {'carried_by': {501}, 'legacy_unknown': {502}}
        assert result.ids(root_a, 'closable carried_by') == {301}
        assert result.ids(root_a, 'orphaned carried_by') == {302}
        assert result.count(root_a, 'would stamp') == 7
        assert 'dry-run: nothing written' in result.block(root_a)
        assert 'dry-run: nothing written' in result.block(root_b)
        assert fake.calls_to('set_task_status') == []

    def test_apply_stamps_the_mapping(self, migrate: Any, serve: Any, make_root: Any) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): project_a(), str(root_b): project_b()})
        expected_a = {wire['id']: _planned_request(wire) for wire in project_a()[:7]}

        result = migrate(*_roots(root_a, root_b), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.OK
        writes_a = [call for call in fake.calls_to('set_task_status') if call['project_root'] == str(root_a)]
        assert writes_a == [
            {
                'project_root': str(root_a),
                'id': task_id,
                'status': 'deferred',
                'deferral': request,
                'restamp_only': True,
                'agent_id': migrate_deferrals.MIGRATION_AGENT_ID,
            }
            for task_id, request in expected_a.items()
        ]
        assert {task_id: _record_kind(fake, root_a, task_id) for task_id in MAPPED_KINDS_A} == MAPPED_KINDS_A
        assert _record_kind(fake, root_a, RECORDED) == 'planning'
        assert _record_kind(fake, root_b, 501) == 'carried_by'
        assert result.count(root_a, 'stamped') == 7
        assert result.count(root_b, 'stamped') == 2
        assert all('dry-run' not in line for line in result.out.splitlines())

    def test_a_second_apply_writes_nothing(self, migrate: Any, serve: Any, make_root: Any) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): project_a(), str(root_b): project_b()})
        migrate(*_roots(root_a, root_b), '--apply')
        writes = len(fake.calls_to('set_task_status'))

        result = migrate(*_roots(root_a, root_b), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.OK
        assert len(fake.calls_to('set_task_status')) == writes
        assert result.classes(root_a) == {'already recorded': set(range(301, 309))}
        assert result.classes(root_b) == {'already recorded': {501, 502}}
        assert result.count(root_a, 'stamped') == 0

    def test_a_server_that_does_not_echo_aborts_before_the_next_write(
        self, migrate: Any, serve: Any, make_root: Any
    ) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): project_a(), str(root_b): project_b()}, echo=False)

        result = migrate(*_roots(root_a, root_b), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.PREDATES_GATE
        assert 'server predates the deferral gate' in result.err
        assert str(root_a) in result.err and '301' in result.err
        assert len(fake.calls_to('set_task_status')) == 1
        assert 'stamped' not in result.out

    def test_a_refused_carrier_falls_back_to_legacy_unknown(self, migrate: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        long_text = 'armed by ' + 'z' * 3000
        fake = serve(
            {str(root): [wire_row(310, x_coalesced_into=99999), wire_row(311, x_coalesced_into=311),
                         wire_row(312, x_armed_by=long_text)]}
        )

        result = migrate(*_roots(root), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.OK
        writes = {task_id: [call['deferral'] for call in fake.calls_to('set_task_status') if call['id'] == task_id]
                  for task_id in ('310', '311', '312')}
        for task_id, code in (('310', 'carrier_not_found'), ('311', 'carrier_is_self')):
            refused, fallback = writes[task_id]
            assert refused['kind'] == 'carried_by'
            assert fallback['kind'] == 'legacy_unknown'
            assert 'x_coalesced_into' in fallback['reason'] and code in fallback['reason']
            assert ('99999' if task_id == '310' else '311') in fallback['reason']
        (predicted,) = writes['312']
        assert predicted['kind'] == 'legacy_unknown' and 'text_too_long' in predicted['reason']
        assert {_record_kind(fake, root, task_id) for task_id in (310, 311, 312)} == {'legacy_unknown'}
        fallbacks = result.lines_with(root, 'fallback #')
        assert any('#310' in line and 'carrier_not_found' in line for line in fallbacks)
        assert any('#311' in line and 'carrier_is_self' in line for line in fallbacks)
        assert any('#312' in line and 'text_too_long' in line for line in fallbacks)
        assert result.count(root, 'stamped') == 3

    def test_a_row_moved_since_read_is_reported_and_left_alone(
        self, migrate: Any, serve: Any, make_root: Any
    ) -> None:
        root = make_root('A')
        fake = serve({str(root): project_a()})
        fake.move_before_write(str(root), 303, 'pending')

        result = migrate(*_roots(root), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.OK
        assert any('#303' in line for line in result.lines_with(root, 'moved since read #'))
        assert fake.row(str(root), 303)['status'] == 'pending'
        assert _record_kind(fake, root, 303) is None
        assert result.count(root, 'stamped') == 6

    def test_an_unreadable_project_fails_the_run_and_the_rest_still_apply(
        self, migrate: Any, serve: Any, make_root: Any
    ) -> None:
        root_a, root_b = make_root('A'), make_root('B')
        fake = serve({str(root_a): project_a(), str(root_b): project_b()})
        fake.unreadable.add(str(root_b))

        result = migrate(*_roots(root_a, root_b), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.FAILED
        assert any('UNREADABLE' in line and 'is not a known project' in line for line in result.block(root_b))
        assert result.count(root_a, 'stamped') == 7

    def test_a_root_with_no_task_store_fails_the_run_and_is_never_asked(
        self, migrate: Any, serve: Any, make_root: Any, tmp_path: Path
    ) -> None:
        root_a = make_root('A')
        bare = tmp_path.resolve() / 'not-a-project'
        bare.mkdir()
        fake = serve({str(root_a): project_a(), str(bare): []})

        result = migrate(*_roots(root_a, bare))

        assert result.code == migrate_deferrals.MigrationExit.FAILED
        assert any('UNKNOWN PROJECT' in line for line in result.block(bare))
        assert all(arguments.get('project_root') != str(bare) for _, arguments in fake.calls)

    def test_a_refused_fallback_is_a_failed_write(self, migrate: Any, serve: Any, make_root: Any) -> None:
        root = make_root('A')
        fake = serve({str(root): [wire_row(310, x_coalesced_into=99999), wire_row(313, x_armed_by='later')]})
        fake.refuse_kind(str(root), 310, 'legacy_unknown', DeferralInvalidReason.TEXT_TOO_LONG)

        result = migrate(*_roots(root), '--apply')

        assert result.code == migrate_deferrals.MigrationExit.FAILED
        assert any('#310' in line for line in result.lines_with(root, 'FAILED #'))
        assert result.count(root, 'failed') == 1
        assert result.count(root, 'stamped') == 1
        assert _record_kind(fake, root, 310) is None

    def test_with_no_project_root_every_unit_root_is_migrated(
        self, migrate: Any, serve: Any, make_root: Any, units_dir: Path
    ) -> None:
        root_a = make_root('A')
        (units_dir / 'orchestrator-a.service').write_text(
            f'[Service]\nExecStart=/usr/bin/uv run orchestrator run --config {root_a / "dark-factory-orchestrator.yaml"}\n'
        )
        serve({str(root_a): project_a()})

        result = migrate()

        assert result.code == migrate_deferrals.MigrationExit.OK
        assert 'sweep: orchestrator-a.service' in result.block(root_a)[0]
        assert result.classes(root_a) == CLASSES_A
