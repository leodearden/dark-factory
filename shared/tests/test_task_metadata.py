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
from shared.task_metadata import (
    BeforeDone,
    DoneProvenance,
    ExternalDep,
    MemoryHints,
    RetryLedger,
    TaskMetadata,
)


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


class TestRetryLedger:
    def test_defaults(self):
        rl = RetryLedger()
        assert rl.consecutive_no_plan_failures == 0
        assert rl.total_no_plan_failures == 0
        assert rl.consecutive_infra_resume_failures == 0
        assert rl.last_infra_resume_iteration_count == 0
        assert rl.consecutive_merge_thrash == 0
        assert rl.last_no_plan_main_sha is None
        assert rl.last_merge_outcome_signature is None
        assert rl.merge_first_enqueued_at is None

    def test_explicit_values_construct(self):
        rl = RetryLedger(
            consecutive_no_plan_failures=2,
            total_no_plan_failures=5,
            last_no_plan_main_sha='abc123',
            consecutive_infra_resume_failures=1,
            last_infra_resume_iteration_count=3,
            consecutive_merge_thrash=4,
            last_merge_outcome_signature='sig-1',
            merge_first_enqueued_at='2026-07-06T00:00:00+00:00',
        )
        assert rl.consecutive_no_plan_failures == 2
        assert rl.total_no_plan_failures == 5
        assert rl.last_no_plan_main_sha == 'abc123'
        assert rl.consecutive_infra_resume_failures == 1
        assert rl.last_infra_resume_iteration_count == 3
        assert rl.consecutive_merge_thrash == 4
        assert rl.last_merge_outcome_signature == 'sig-1'
        assert rl.merge_first_enqueued_at == '2026-07-06T00:00:00+00:00'

    def test_model_dump_round_trips_values(self):
        rl = RetryLedger(consecutive_no_plan_failures=2, last_no_plan_main_sha='abc123')
        dumped = rl.model_dump()
        assert dumped['consecutive_no_plan_failures'] == 2
        assert dumped['last_no_plan_main_sha'] == 'abc123'
        assert RetryLedger(**dumped) == rl

    def test_unknown_counter_retained_through_round_trip(self):
        rl = RetryLedger(x_new_counter=3)
        dumped = rl.model_dump()
        assert dumped['x_new_counter'] == 3


class TestTaskMetadataFields:
    def test_empty_defaults(self):
        tm = TaskMetadata()
        assert tm.schema_version == 1
        assert tm.task_kind == 'normal'
        assert tm.always_escalates is False
        assert tm.before_done is None
        assert tm.done_provenance is None
        assert tm.memory_hints is None
        assert tm.retry_ledger is None
        assert tm.external_deps == []
        assert tm.files == []

    def test_nested_dicts_coerce_to_typed_submodels(self):
        tm = TaskMetadata(
            before_done={'script': 'scripts/x.sh', 'timeout_secs': 60},
            done_provenance={'kind': 'merged', 'commit': 'abc123'},
            memory_hints={'entities': ['E'], 'queries': ['Q']},
            retry_ledger={'consecutive_no_plan_failures': 2},
        )
        assert isinstance(tm.before_done, BeforeDone)
        assert isinstance(tm.done_provenance, DoneProvenance)
        assert isinstance(tm.memory_hints, MemoryHints)
        assert isinstance(tm.retry_ledger, RetryLedger)
        assert tm.before_done.script == 'scripts/x.sh'
        assert tm.done_provenance.commit == 'abc123'
        assert tm.memory_hints.entities == ['E']
        assert tm.retry_ledger.consecutive_no_plan_failures == 2

    def test_external_deps_stays_list_of_str(self):
        tm = TaskMetadata(external_deps=['a:1', 'b:2'])
        assert tm.external_deps == ['a:1', 'b:2']
        assert all(isinstance(dep, str) for dep in tm.external_deps)

    def test_deterministic_task_kind_with_before_done_accepted(self):
        tm = TaskMetadata(
            task_kind='deterministic',
            before_done={'script': 'scripts/x.sh', 'timeout_secs': 60},
        )
        assert tm.task_kind == 'deterministic'

    def test_bad_task_kind_rejected(self):
        with pytest.raises(ValidationError):
            TaskMetadata(task_kind='weird')

    def test_unknown_top_level_keys_round_trip(self):
        blob = {
            'x_foo': 1,
            'legacy_bar': [1, 2],
            'schema_version': 1,
            'task_kind': 'normal',
        }
        dumped = TaskMetadata(**blob).model_dump()
        for key, value in blob.items():
            assert dumped[key] == value
