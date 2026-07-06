"""Tests for shared.task_metadata — versioned TaskMetadata v1 schema.

Built bottom-up in TDD order (see plans/task-metadata-schema-prd.md §5):
  - TestBeforeDone / TestDoneProvenance / TestMemoryHints / TestExternalDep /
    TestRetryLedger: the five sub-models in isolation.
  - TestTaskMetadataFields: TaskMetadata's own fields + I1 round-trip.
  - TestDeterministicInvariants: the two named cross-field invariants.
  - TestSubmodelRegistry: the W10 extension point.
  - TestMigrations: the versioned v0->v1 migration registry.
  - TestParseMetadataCore / TestParseMetadataFailurePolicy: parse_metadata's
    happy path and its direction/enforce failure-policy matrix.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import shared.task_metadata as task_metadata_module
from shared.task_metadata import BeforeDone, DoneProvenance, ExternalDep, MemoryHints


@pytest.fixture(autouse=True)
def _reset_metadata_registry_state():
    """Snapshot and restore task_metadata's module-global registry/migrations.

    register_metadata_submodel and the migration registry mutate module-global
    dicts; without this, TestSubmodelRegistry / TestMigrations / the registry
    portion of the parse_metadata tests would leak registrations into later
    tests. Uses getattr/hasattr defensively since _SUBMODEL_REGISTRY and
    _MIGRATIONS are added incrementally by later steps in this file's own
    TDD sequence.
    """
    had_registry = hasattr(task_metadata_module, '_SUBMODEL_REGISTRY')
    registry_snapshot = dict(getattr(task_metadata_module, '_SUBMODEL_REGISTRY', {}))
    had_migrations = hasattr(task_metadata_module, '_MIGRATIONS')
    migrations_snapshot = dict(getattr(task_metadata_module, '_MIGRATIONS', {}))
    yield
    if had_registry:
        task_metadata_module._SUBMODEL_REGISTRY.clear()
        task_metadata_module._SUBMODEL_REGISTRY.update(registry_snapshot)
    if had_migrations:
        task_metadata_module._MIGRATIONS.clear()
        task_metadata_module._MIGRATIONS.update(migrations_snapshot)


class TestBeforeDone:
    def test_full_live_shape_constructs(self):
        bd = BeforeDone(
            script='scripts/x.sh',
            args=['a'],
            env={'K': 'V'},
            cwd='.',
            timeout_secs=120,
            target_unit='u.service',
        )
        assert bd.script == 'scripts/x.sh'
        assert bd.args == ['a']
        assert bd.env == {'K': 'V'}
        assert bd.cwd == '.'
        assert bd.timeout_secs == 120
        assert bd.target_unit == 'u.service'

    def test_defaults_when_omitted(self):
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60)
        assert bd.args == []
        assert bd.env == {}
        assert bd.cwd is None
        assert bd.target_unit is None

    def test_script_required(self):
        with pytest.raises(ValidationError):
            BeforeDone(timeout_secs=60)

    def test_script_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            BeforeDone(script='', timeout_secs=60)

    def test_timeout_secs_required(self):
        with pytest.raises(ValidationError):
            BeforeDone(script='scripts/x.sh')

    @pytest.mark.parametrize('bad_timeout', [0, -1])
    def test_timeout_secs_must_be_positive(self, bad_timeout):
        with pytest.raises(ValidationError):
            BeforeDone(script='scripts/x.sh', timeout_secs=bad_timeout)

    def test_unknown_subfield_retained_and_reemitted(self):
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60, x_extra='keep-me')
        dumped = bd.model_dump()
        assert dumped['x_extra'] == 'keep-me'


class TestDoneProvenance:
    def test_merged_with_commit_constructs(self):
        dp = DoneProvenance(kind='merged', commit='abc123')
        assert dp.kind == 'merged'
        assert dp.commit == 'abc123'

    def test_found_on_main_with_commit_and_note_constructs(self):
        dp = DoneProvenance(kind='found_on_main', commit='abc123', note='landed in 999')
        assert dp.commit == 'abc123'
        assert dp.note == 'landed in 999'

    def test_deterministic_deploy_with_pid_unit_timestamp_constructs(self):
        dp = DoneProvenance(
            kind='deterministic-deploy',
            pid=4242,
            unit='fused-memory.service',
            active_enter_timestamp='2026-07-06T00:00:00+00:00',
        )
        assert dp.pid == 4242
        assert dp.unit == 'fused-memory.service'

    def test_deterministic_deploy_scheduled_with_unit_constructs(self):
        dp = DoneProvenance(kind='deterministic-deploy-scheduled', unit='fused-memory.service')
        assert dp.unit == 'fused-memory.service'

    def test_bogus_kind_rejected(self):
        with pytest.raises(ValidationError):
            DoneProvenance(kind='bogus')

    def test_extra_subfields_retained_through_round_trip(self):
        dp = DoneProvenance(
            kind='deterministic-deploy-scheduled',
            unit='fused-memory.service',
            transient_unit='fused-memory-restart-1234.service',
            fire_delay_secs=5,
        )
        dumped = dp.model_dump()
        assert dumped['transient_unit'] == 'fused-memory-restart-1234.service'
        assert dumped['fire_delay_secs'] == 5

    def test_commit_required_for_merged(self):
        with pytest.raises(ValidationError):
            DoneProvenance(kind='merged')

    def test_commit_required_for_found_on_main(self):
        with pytest.raises(ValidationError):
            DoneProvenance(kind='found_on_main', note='no commit')

    def test_note_required_for_found_on_main(self):
        with pytest.raises(ValidationError):
            DoneProvenance(kind='found_on_main', commit='abc123')

    def test_deterministic_deploy_ok_without_commit_or_note(self):
        DoneProvenance(kind='deterministic-deploy')

    def test_deterministic_deploy_scheduled_ok_without_commit_or_note(self):
        DoneProvenance(kind='deterministic-deploy-scheduled')


class TestMemoryHints:
    def test_constructs_with_values(self):
        mh = MemoryHints(entities=['E'], queries=['Q'])
        assert mh.entities == ['E']
        assert mh.queries == ['Q']

    def test_defaults_to_empty_lists(self):
        mh = MemoryHints()
        assert mh.entities == []
        assert mh.queries == []

    def test_non_list_entities_rejected(self):
        with pytest.raises(ValidationError):
            MemoryHints(entities='not-a-list')

    def test_wrong_element_type_rejected(self):
        with pytest.raises(ValidationError):
            MemoryHints(entities=[123])


class TestExternalDep:
    def test_constructs_from_fields(self):
        dep = ExternalDep(project_id='dark_factory', task_id='42')
        assert dep.project_id == 'dark_factory'
        assert dep.task_id == '42'

    def test_parse_splits_wire_form(self):
        dep = ExternalDep.parse('dark_factory:42')
        assert dep.project_id == 'dark_factory'
        assert dep.task_id == '42'

    def test_render_reproduces_wire_form(self):
        dep = ExternalDep(project_id='dark_factory', task_id='42')
        assert dep.render() == 'dark_factory:42'

    @pytest.mark.parametrize('wire', ['dark_factory:42', 'a:b', 'proj-1:99'])
    def test_parse_render_round_trips(self, wire):
        assert ExternalDep.parse(wire).render() == wire

    def test_parse_strips_whitespace(self):
        dep = ExternalDep.parse('  a:b ')
        assert dep.project_id == 'a'
        assert dep.task_id == 'b'

    @pytest.mark.parametrize(
        'malformed',
        ['foo', ':42', 'foo:', 'a:b:c', '', '   '],
    )
    def test_parse_rejects_malformed(self, malformed):
        with pytest.raises((ValueError, ValidationError)):
            ExternalDep.parse(malformed)
