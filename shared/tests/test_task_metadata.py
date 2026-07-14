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

import copy
import json

import pytest
from pydantic import BaseModel, ValidationError

import shared.task_metadata as task_metadata_module
from shared.task_metadata import (
    BeforeDone,
    DoneProvenance,
    ExternalDep,
    MemoryHints,
    Milestone,
    RetryLedger,
    TaskMetadata,
    apply_migrations,
    parse_metadata,
    register_metadata_submodel,
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
        assert bd.kind == 'deploy'

    def test_defaults_when_omitted(self):
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60)
        assert bd.args == []
        assert bd.env == {}
        assert bd.cwd is None
        assert bd.target_unit is None

    def test_script_required(self):
        with pytest.raises(ValidationError):
            BeforeDone(timeout_secs=60)  # type: ignore[call-arg]

    def test_script_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            BeforeDone(script='', timeout_secs=60)

    def test_timeout_secs_required(self):
        with pytest.raises(ValidationError):
            BeforeDone(script='scripts/x.sh')  # type: ignore[call-arg]

    @pytest.mark.parametrize('bad_timeout', [0, -1])
    def test_timeout_secs_must_be_positive(self, bad_timeout):
        with pytest.raises(ValidationError):
            BeforeDone(script='scripts/x.sh', timeout_secs=bad_timeout)

    def test_unknown_subfield_retained_and_reemitted(self):
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60, x_extra='keep-me')  # type: ignore[call-arg]
        dumped = bd.model_dump()
        assert dumped['x_extra'] == 'keep-me'

    def test_kind_defaults_to_deploy(self):
        # Default preserves every existing deterministic task byte-identically
        # — no existing blob carries `kind`, so the default equals the prior
        # implicit deploy behavior.
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60)
        assert bd.kind == 'deploy'

    def test_kind_predicate_accepted(self):
        bd = BeforeDone(script='scripts/x.sh', timeout_secs=60, kind='predicate')
        assert bd.kind == 'predicate'

    def test_kind_bogus_rejected(self):
        with pytest.raises(ValidationError):
            BeforeDone(script='scripts/x.sh', timeout_secs=60, kind='bogus')  # type: ignore[arg-type]


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
            DoneProvenance(kind='bogus')  # type: ignore[arg-type]

    def test_extra_subfields_retained_through_round_trip(self):
        dp = DoneProvenance(
            kind='deterministic-deploy-scheduled',
            unit='fused-memory.service',
            transient_unit='fused-memory-restart-1234.service',  # type: ignore[call-arg]
            fire_delay_secs=5,  # type: ignore[call-arg]
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

    def test_deterministic_gate_ok_without_commit_or_note(self):
        dp = DoneProvenance(kind='deterministic-gate', note='pure gate resolved')
        assert dp.kind == 'deterministic-gate'
        assert dp.note == 'pure gate resolved'

    def test_deterministic_milestone_ok_without_commit_or_note(self):
        dp = DoneProvenance(kind='deterministic-milestone')
        assert dp.kind == 'deterministic-milestone'
        assert dp.commit is None
        assert dp.note is None

    def test_deterministic_milestone_retains_note(self):
        dp = DoneProvenance(kind='deterministic-milestone', note='<stdout tail>')
        assert dp.kind == 'deterministic-milestone'
        assert dp.note == '<stdout tail>'

    def test_deterministic_bogus_kind_still_rejected(self):
        # Regression guard: adding 'deterministic-milestone' must not open
        # the Literal up to arbitrary deterministic-* strings.
        with pytest.raises(ValidationError):
            DoneProvenance(kind='deterministic-bogus')  # type: ignore[arg-type]


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
            MemoryHints(entities='not-a-list')  # type: ignore[arg-type]

    def test_wrong_element_type_rejected(self):
        with pytest.raises(ValidationError):
            MemoryHints(entities=[123])  # type: ignore[arg-type]


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
        rl = RetryLedger(x_new_counter=3)  # type: ignore[call-arg]
        dumped = rl.model_dump()
        assert dumped['x_new_counter'] == 3

    # -- RetryLedger.normalize_cause_hint (staticmethod) --
    # Ported from test_workflow_signature_loop_guard.py::TestNormalizeCauseHint
    # so the ledger becomes the single signature-keying authority (see
    # design_decisions: "_normalize_cause_hint ... become @staticmethod on
    # shared RetryLedger").

    def test_normalize_cause_hint_strips_file_line_suffix(self):
        h1 = 'FAILED tests/test_x.py:42 — AssertionError'
        h2 = 'FAILED tests/test_x.py:99 — AssertionError'
        assert RetryLedger.normalize_cause_hint(h1) == RetryLedger.normalize_cause_hint(h2)

    def test_normalize_cause_hint_strips_file_line_col_suffix(self):
        h1 = 'foo.py:42:7 some error'
        h2 = 'foo.py:99:3 some error'
        assert RetryLedger.normalize_cause_hint(h1) == RetryLedger.normalize_cause_hint(h2)

    def test_normalize_cause_hint_strips_ansi_escapes(self):
        raw = '\x1b[31mFAILED\x1b[0m'
        assert RetryLedger.normalize_cause_hint(raw) == 'failed'

    def test_normalize_cause_hint_collapses_whitespace(self):
        raw = 'some\t  error\n  detail'
        assert RetryLedger.normalize_cause_hint(raw) == 'some error detail'

    def test_normalize_cause_hint_lowercases_and_trims(self):
        raw = '  ASSERTION ERROR  '
        result = RetryLedger.normalize_cause_hint(raw)
        assert result == result.lower()
        assert result == result.strip()
        assert result == 'assertion error'

    def test_normalize_cause_hint_empty_string_returns_empty_string(self):
        assert RetryLedger.normalize_cause_hint('') == ''

    def test_normalize_cause_hint_none_returns_empty_string(self):
        assert RetryLedger.normalize_cause_hint(None) == ''

    # -- RetryLedger.compute_merge_outcome_signature (staticmethod) --
    # Mirrors the behaviour pinned by
    # test_workflow_merge_thrash.py::test_merge_outcome_signature_* so the
    # module-level _compute_merge_outcome_signature delegator (workflow.py)
    # stays byte-identical once it delegates here.

    def test_compute_merge_outcome_signature_stable_despite_varied_line_and_reason(self):
        sig_a = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', 'StatusBar.tsx:42 error TS2322: Type X not assignable'
        )
        sig_b = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', 'StatusBar.tsx:58 error TS2322: Type X not assignable'
        )
        assert sig_a == sig_b, (
            f'Same fingerprint (category, normalised cause_hint) must yield equal signature; '
            f'got {sig_a!r} vs {sig_b!r}'
        )
        assert len(sig_a) == 16 and all(c in '0123456789abcdef' for c in sig_a), (
            f'Signature must be 16 hex chars, got {sig_a!r}'
        )

    def test_compute_merge_outcome_signature_different_category_differs(self):
        sig_a = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', 'StatusBar.tsx:42 error TS2322: Type X not assignable'
        )
        sig_b = RetryLedger.compute_merge_outcome_signature(
            'test_failure', 'StatusBar.tsx:42 error TS2322: Type X not assignable'
        )
        assert sig_a != sig_b, 'Different category must yield a different signature'

    def test_compute_merge_outcome_signature_different_hint_differs(self):
        sig_a = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', 'StatusBar.tsx:42 error TS2322: Type X not assignable'
        )
        sig_b = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', 'OtherComponent.tsx:10 error TS9999: something else'
        )
        assert sig_a != sig_b, 'Different cause_hint must yield a different signature'

    def test_compute_merge_outcome_signature_falls_back_to_reason_hash(self):
        sig = RetryLedger.compute_merge_outcome_signature(
            '', '', fallback_reason='Post-merge verification failed: TESTS FAILED\n\nextra output'
        )
        assert len(sig) == 16 and all(c in '0123456789abcdef' for c in sig), (
            f'Fallback signature must be 16 hex chars, got {sig!r}'
        )

    def test_compute_merge_outcome_signature_fallback_normalises_reason(self):
        sig_a = RetryLedger.compute_merge_outcome_signature(
            '', '', fallback_reason='Post-merge verification failed: TESTS FAILED\n\nextra output'
        )
        sig_b = RetryLedger.compute_merge_outcome_signature(
            '',
            '',
            fallback_reason='post-merge verification failed:  tests  failed\n\nextra  output',
        )
        assert sig_a == sig_b, (
            'Fallback signatures must match for reasons that differ only by case/whitespace'
        )

    def test_compute_merge_outcome_signature_fallback_different_reason_differs(self):
        sig_a = RetryLedger.compute_merge_outcome_signature(
            '', '', fallback_reason='Post-merge verification failed: TESTS FAILED\n\nextra output'
        )
        sig_b = RetryLedger.compute_merge_outcome_signature(
            '', '', fallback_reason='git merge failed: conflict in foo.py'
        )
        assert sig_a != sig_b

    def test_compute_merge_outcome_signature_category_or_hint_alone_takes_structured_path(self):
        """Either field alone (not just both) is enough to skip the fallback path."""
        sig_category_only = RetryLedger.compute_merge_outcome_signature(
            'gui_tsc', '', fallback_reason='some fallback reason'
        )
        sig_fallback = RetryLedger.compute_merge_outcome_signature(
            '', '', fallback_reason='some fallback reason'
        )
        assert sig_category_only != sig_fallback


class TestMilestone:
    """``metadata.milestone`` — the dated/delayed milestone sub-model (PRD §6.1)."""

    def test_dated_constructs(self):
        m = Milestone(mode='dated', at='2026-08-01T00:00:00+00:00')
        assert m.mode == 'dated'
        assert m.at == '2026-08-01T00:00:00+00:00'
        assert m.after_secs is None

    def test_delayed_constructs(self):
        m = Milestone(mode='delayed', after_secs=604800)
        assert m.mode == 'delayed'
        assert m.after_secs == 604800
        assert m.at is None

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({'mode': 'dated', 'at': None}, id='dated_missing_at'),
            pytest.param({'mode': 'dated', 'at': 'not-a-date'}, id='dated_unparseable_at'),
            pytest.param({'mode': 'delayed', 'after_secs': None}, id='delayed_missing_after_secs'),
            pytest.param({'mode': 'delayed', 'after_secs': 0}, id='delayed_after_secs_zero'),
            pytest.param({'mode': 'delayed', 'after_secs': -1}, id='delayed_after_secs_negative'),
            pytest.param(
                {'mode': 'dated', 'at': '2026-08-01T00:00:00+00:00', 'after_secs': 604800},
                id='dated_with_after_secs_also_set',
            ),
            pytest.param(
                {
                    'mode': 'delayed',
                    'after_secs': 604800,
                    'at': '2026-08-01T00:00:00+00:00',
                },
                id='delayed_with_at_also_set',
            ),
            pytest.param({'mode': 'weird'}, id='mode_weird'),
        ],
    )
    def test_invalid_specs_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            Milestone(**kwargs)  # type: ignore[arg-type]

    def test_unknown_subfield_retained_and_reemitted(self):
        m = Milestone(mode='delayed', after_secs=604800, x_extra='keep')  # type: ignore[call-arg]
        dumped = m.model_dump()
        assert dumped['x_extra'] == 'keep'


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
            task_kind='deterministic',
            before_done={'script': 'scripts/x.sh', 'timeout_secs': 60},  # type: ignore[arg-type]
            done_provenance={'kind': 'merged', 'commit': 'abc123'},  # type: ignore[arg-type]
            memory_hints={'entities': ['E'], 'queries': ['Q']},  # type: ignore[arg-type]
            retry_ledger={'consecutive_no_plan_failures': 2},  # type: ignore[arg-type]
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
            before_done={'script': 'scripts/x.sh', 'timeout_secs': 60},  # type: ignore[arg-type]
        )
        assert tm.task_kind == 'deterministic'

    def test_bad_task_kind_rejected(self):
        with pytest.raises(ValidationError):
            TaskMetadata(task_kind='weird')  # type: ignore[arg-type]

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


class TestDeterministicInvariants:
    """The two named cross-field invariants (CLAUDE.md field-combo presets)."""

    _MINIMAL_BEFORE_DONE = {'script': 'scripts/x.sh', 'timeout_secs': 60}

    def test_auto_deploy_accepted(self):
        # deterministic + before_done + always_escalates=False [auto-deploy]
        TaskMetadata(
            task_kind='deterministic',
            before_done=self._MINIMAL_BEFORE_DONE,  # type: ignore[arg-type]
            always_escalates=False,
        )

    def test_act_then_ask_accepted(self):
        # deterministic + before_done + always_escalates=True [act-then-ask]
        TaskMetadata(
            task_kind='deterministic',
            before_done=self._MINIMAL_BEFORE_DONE,  # type: ignore[arg-type]
            always_escalates=True,
        )

    def test_pure_gate_accepted(self):
        # deterministic + no before_done + always_escalates=True [pure gate]
        TaskMetadata(task_kind='deterministic', always_escalates=True)

    def test_normal_default_accepted(self):
        # task_kind='normal' + no before_done [default]
        TaskMetadata(task_kind='normal')

    def test_deterministic_ill_formed_no_op_rejected(self):
        # deterministic + no before_done + always_escalates=False [ill-formed no-op]
        with pytest.raises(ValidationError):
            TaskMetadata(task_kind='deterministic', always_escalates=False)

    def test_before_done_only_valid_on_deterministic_rejected(self):
        # task_kind='normal' + before_done set [before_done only on deterministic]
        with pytest.raises(ValidationError):
            TaskMetadata(
                task_kind='normal',
                before_done=self._MINIMAL_BEFORE_DONE,  # type: ignore[arg-type]
            )


class _DeployStateStub(BaseModel):
    """Throwaway sub-model standing in for a future W10 registrant."""

    phase: str


class TestSubmodelRegistry:
    """The W10 extension point: register_metadata_submodel + _SUBMODEL_REGISTRY."""

    def test_register_new_key_stored_in_registry(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        assert task_metadata_module._SUBMODEL_REGISTRY['deploy_state'] is _DeployStateStub

    def test_register_same_model_twice_is_idempotent(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        register_metadata_submodel('deploy_state', _DeployStateStub)  # no raise
        assert task_metadata_module._SUBMODEL_REGISTRY['deploy_state'] is _DeployStateStub

    def test_register_different_model_same_key_raises(self):
        class _OtherDeployStateStub(BaseModel):
            phase: str

        register_metadata_submodel('deploy_state', _DeployStateStub)
        with pytest.raises(ValueError):
            register_metadata_submodel('deploy_state', _OtherDeployStateStub)


class TestMilestoneRegistration:
    """Milestone's registration with the W10 extension point + parse_metadata integration."""

    def test_registered_at_import(self):
        assert task_metadata_module._SUBMODEL_REGISTRY['milestone'] is Milestone

    def test_round_trip_no_warnings(self):
        model, warnings = parse_metadata(
            {
                'task_kind': 'deterministic',
                'milestone': {'mode': 'delayed', 'after_secs': 604800},
                'before_done': {
                    'kind': 'predicate',
                    'script': 'scripts/check_merge_flakiness.sh',
                    'timeout_secs': 120,
                },
            },
            direction='write',
        )
        assert warnings == []
        assert model.before_done is not None
        assert model.before_done.kind == 'predicate'
        assert isinstance(model.milestone, Milestone)  # type: ignore[attr-defined]
        dumped_milestone = model.model_dump()['milestone']
        assert not isinstance(dumped_milestone, BaseModel)
        assert dumped_milestone == {'mode': 'delayed', 'at': None, 'after_secs': 604800}

    def test_malformed_slice_read_warns_and_retains_raw(self):
        model, warnings = parse_metadata({'milestone': {'mode': 'delayed'}}, direction='read')
        assert len(warnings) == 1
        assert warnings[0].field == 'milestone'
        assert warnings[0].code == 'invalid_submodel'
        assert model.model_dump()['milestone'] == {'mode': 'delayed'}

    def test_malformed_slice_write_enforce_raises(self):
        with pytest.raises(ValidationError):
            parse_metadata({'milestone': {'mode': 'delayed'}}, direction='write', enforce=True)


class TestMigrations:
    """The versioned v0->v1 migration registry (PRD §3/§5).

    The memory_hints legacy-list normalisation is ported verbatim from the
    fused-memory backend's ``_normalize_legacy_memory_hints_value``
    (sqlite_task_backend.py:1320): entities/queries are deduped
    independently (not per-pair), preserving first-seen order.
    """

    def test_legacy_memory_hints_list_deduped_to_canonical_dict(self):
        blob = {'memory_hints': [{'entity': 'E', 'query': 'Q'}, {'entity': 'E', 'query': 'Q2'}]}
        upgraded = apply_migrations(blob)
        assert upgraded['memory_hints'] == {'entities': ['E'], 'queries': ['Q', 'Q2']}

    def test_legacy_memory_hints_skips_non_dict_and_invalid_entries(self):
        blob = {
            'memory_hints': [
                'not-a-dict',
                {'entity': '', 'query': ''},
                {'entity': 123, 'query': 456},
                {},
                {'entity': 'E', 'query': 'Q'},
            ]
        }
        upgraded = apply_migrations(blob)
        assert upgraded['memory_hints'] == {'entities': ['E'], 'queries': ['Q']}

    def test_already_canonical_memory_hints_dict_unchanged(self):
        blob = {'memory_hints': {'entities': ['E'], 'queries': ['Q']}}
        upgraded = apply_migrations(blob)
        assert upgraded['memory_hints'] == {'entities': ['E'], 'queries': ['Q']}

    def test_missing_schema_version_treated_as_v0_and_stamped_to_1(self):
        upgraded = apply_migrations({})
        assert upgraded['schema_version'] == 1

    def test_already_v1_blob_returned_untouched(self):
        blob = {'schema_version': 1, 'memory_hints': [{'entity': 'E', 'query': 'Q'}]}
        upgraded = apply_migrations(blob)
        assert upgraded == blob

    def test_does_not_mutate_caller_input(self):
        blob = {'memory_hints': [{'entity': 'E', 'query': 'Q'}]}
        original = copy.deepcopy(blob)
        apply_migrations(blob)
        assert blob == original


class TestParseMetadataCore:
    """parse_metadata's happy path for both directions.

    The direction/enforce failure-policy matrix is TestParseMetadataFailurePolicy;
    this class covers only the paths that never warn.
    """

    def test_none_blob_read_direction_is_benign_absent(self):
        model, warnings = parse_metadata(None, direction='read')
        assert model == TaskMetadata()
        assert warnings == []

    def test_none_blob_write_direction_is_benign_absent(self):
        model, warnings = parse_metadata(None, direction='write')
        assert model == TaskMetadata()
        assert warnings == []

    def test_valid_dict_returns_typed_model_no_warnings(self):
        model, warnings = parse_metadata({'task_kind': 'normal'}, direction='read')
        assert isinstance(model, TaskMetadata)
        assert model.task_kind == 'normal'
        assert warnings == []

    def test_valid_json_string_parses_to_same_model(self):
        blob = {'task_kind': 'normal'}
        model, warnings = parse_metadata(json.dumps(blob), direction='read')
        assert model == TaskMetadata(**blob)  # type: ignore[arg-type]
        assert warnings == []

    def test_migration_applied_end_to_end_for_legacy_memory_hints(self):
        model, warnings = parse_metadata(
            {'memory_hints': [{'entity': 'E', 'query': 'Q'}]}, direction='read'
        )
        assert model.memory_hints is not None
        assert model.memory_hints.entities == ['E']
        assert model.memory_hints.queries == ['Q']
        assert warnings == []

    def test_registered_submodel_slice_validated_and_attached_no_warnings(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        model, warnings = parse_metadata({'deploy_state': {'phase': 'rollout'}}, direction='write')
        assert isinstance(model.deploy_state, _DeployStateStub)  # type: ignore[attr-defined]
        assert model.deploy_state.phase == 'rollout'  # type: ignore[attr-defined]
        assert warnings == []

    def test_registered_submodel_round_trips_as_plain_dict_via_model_dump(self):
        # model.deploy_state is a _DeployStateStub instance (asserted above),
        # stored in TaskMetadata.__pydantic_extra__ since 'deploy_state' is
        # not a declared TaskMetadata field. I1 round-trip preservation
        # requires model_dump() to re-emit it as a plain dict — not leave a
        # BaseModel instance sitting in the "JSON-serializable blob" output.
        register_metadata_submodel('deploy_state', _DeployStateStub)
        model, warnings = parse_metadata({'deploy_state': {'phase': 'rollout'}}, direction='write')
        assert warnings == []
        dumped = model.model_dump()
        assert dumped['deploy_state'] == {'phase': 'rollout'}
        assert not isinstance(dumped['deploy_state'], BaseModel)

    def test_unknown_top_level_key_survives_round_trip(self):
        # x_-prefixed is the sanctioned silent namespace regardless of the
        # failure-policy matrix landing later, so this stays stable.
        model, _ = parse_metadata({'x_foo': 1}, direction='write')
        assert model.model_dump()['x_foo'] == 1

    def test_return_type_is_always_model_warnings_tuple(self):
        result = parse_metadata(None, direction='read')
        assert isinstance(result, tuple)
        model, warnings = result
        assert isinstance(model, TaskMetadata)
        assert isinstance(warnings, list)


class TestParseMetadataFailurePolicy:
    """The direction x enforce failure-policy matrix (PRD §5 + I1/I4).

    TestParseMetadataCore covers only the paths that never warn. This class
    covers every path that must warn-and-accept (read, or write+enforce=False)
    or reject-loudly (write+enforce=True) instead of the old silent-``{}``
    discard: unparseable JSON, an invalid typed/registered sub-model slice,
    an unrecognised top-level key, and a deterministic cross-field invariant
    violation.
    """

    _BAD_JSON = '{not json'

    def test_unparseable_json_read_warns_never_raises(self):
        model, warnings = parse_metadata(self._BAD_JSON, direction='read')
        assert model == TaskMetadata()
        assert len(warnings) == 1
        assert warnings[0].code == 'unparseable_json'

    def test_unparseable_json_write_warn_mode_accepts(self):
        model, warnings = parse_metadata(self._BAD_JSON, direction='write', enforce=False)
        assert model == TaskMetadata()
        assert len(warnings) == 1
        assert warnings[0].code == 'unparseable_json'

    def test_unparseable_json_write_enforce_raises(self):
        with pytest.raises(ValueError):
            parse_metadata(self._BAD_JSON, direction='write', enforce=True)

    # done_provenance: Literal rejects 'bogus'. before_done: missing required
    # timeout_secs. Neither sets task_kind='deterministic', so only the named
    # sub-model fails — the outer cross-field invariant never enters into it.
    _INVALID_SUBMODEL_CASES = [
        pytest.param(
            {'done_provenance': {'kind': 'bogus'}},
            'done_provenance',
            id='done_provenance_bogus_kind',
        ),
        pytest.param(
            {'before_done': {'script': 'scripts/x.sh'}},
            'before_done',
            id='before_done_missing_timeout',
        ),
    ]

    @pytest.mark.parametrize('blob,field', _INVALID_SUBMODEL_CASES)
    def test_invalid_typed_submodel_read_warns_best_effort(self, blob, field):
        model, warnings = parse_metadata(blob, direction='read')
        assert len(warnings) == 1
        assert warnings[0].field == field
        assert model.model_dump()[field] == blob[field]

    @pytest.mark.parametrize('blob,field', _INVALID_SUBMODEL_CASES)
    def test_invalid_typed_submodel_write_warn_mode_accepts_raw(self, blob, field):
        model, warnings = parse_metadata(blob, direction='write', enforce=False)
        assert len(warnings) == 1
        assert warnings[0].field == field
        assert model.model_dump()[field] == blob[field]

    @pytest.mark.parametrize('blob,field', _INVALID_SUBMODEL_CASES)
    def test_invalid_typed_submodel_write_enforce_raises(self, blob, field):
        with pytest.raises(ValidationError):
            parse_metadata(blob, direction='write', enforce=True)

    def test_invalid_field_warning_message_scoped_to_that_field_only(self):
        # Two independently-invalid top-level fields in one blob: each
        # field's warning message must describe only that field's own
        # failure. Before the fix both warnings shared `message=str(exc)`
        # — the full multi-error ValidationError dump — so each message
        # would mention *both* offending fields, making the per-field
        # `field` slug the only distinguishing signal.
        blob = {
            'done_provenance': {'kind': 'bogus'},
            'retry_ledger': {'consecutive_no_plan_failures': 'not-an-int'},
        }
        model, warnings = parse_metadata(blob, direction='read')
        by_field = {w.field: w.message for w in warnings if w.code == 'invalid_field'}
        assert set(by_field) == {'done_provenance', 'retry_ledger'}
        assert by_field['done_provenance'] != by_field['retry_ledger']
        assert 'retry_ledger' not in by_field['done_provenance']
        assert 'done_provenance' not in by_field['retry_ledger']
        assert 'validation error' not in by_field['done_provenance'].lower()
        assert 'validation error' not in by_field['retry_ledger'].lower()
        assert model.model_dump()['done_provenance'] == blob['done_provenance']
        assert model.model_dump()['retry_ledger'] == blob['retry_ledger']

    def test_second_independent_invariant_after_pop_falls_back_to_raw_construct(self):
        # before_done is missing its required timeout_secs, so it's popped
        # as the offending field; but the remainder
        # (task_kind='deterministic', always_escalates=False, before_done
        # popped) trips the *separate* deterministic cross-field invariant
        # on its own. Popping the first offending key isn't sufficient to
        # produce a valid remainder, so parse_metadata must fall back to a
        # fully raw, unvalidated model instead of raising.
        blob = {
            'task_kind': 'deterministic',
            'always_escalates': False,
            'before_done': {'script': 'scripts/x.sh'},  # missing required timeout_secs
        }
        model, warnings = parse_metadata(blob, direction='read')
        assert model.task_kind == 'deterministic'
        assert model.always_escalates is False
        assert model.before_done == blob['before_done']
        assert len(warnings) == 1
        assert warnings[0].field == 'before_done'
        assert warnings[0].code == 'invalid_field'

    def test_second_independent_invariant_write_enforce_raises(self):
        blob = {
            'task_kind': 'deterministic',
            'always_escalates': False,
            'before_done': {'script': 'scripts/x.sh'},
        }
        with pytest.raises(ValidationError):
            parse_metadata(blob, direction='write', enforce=True)

    def test_invalid_registered_submodel_slice_write_enforce_raises(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        with pytest.raises(ValidationError):
            parse_metadata({'deploy_state': {}}, direction='write', enforce=True)

    def test_invalid_registered_submodel_slice_warn_mode_accepts_raw(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        model, warnings = parse_metadata({'deploy_state': {}}, direction='write', enforce=False)
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert warnings[0].field == 'deploy_state'
        assert model.model_dump()['deploy_state'] == {}

    # A registered slice whose *value* isn't a mapping at all (list/str/etc.)
    # can't be splatted as `submodel(**value)` — that raises TypeError, not
    # ValidationError. parse_metadata must absorb this the same way as any
    # other malformed sub-model, never raising outside write+enforce=True.
    def test_registered_submodel_slice_non_mapping_value_read_warns_never_raises(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        model, warnings = parse_metadata({'deploy_state': [1, 2]}, direction='read')
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert warnings[0].field == 'deploy_state'
        assert model.model_dump()['deploy_state'] == [1, 2]

    def test_registered_submodel_slice_non_mapping_value_write_warn_mode_accepts(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        model, warnings = parse_metadata({'deploy_state': 'x'}, direction='write', enforce=False)
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert warnings[0].field == 'deploy_state'
        assert model.model_dump()['deploy_state'] == 'x'

    def test_registered_submodel_slice_non_mapping_value_write_enforce_raises(self):
        register_metadata_submodel('deploy_state', _DeployStateStub)
        with pytest.raises((ValidationError, TypeError)):
            parse_metadata({'deploy_state': [1, 2]}, direction='write', enforce=True)

    def test_x_prefixed_unknown_key_silent_zero_warnings(self):
        model, warnings = parse_metadata({'x_experimental': 'v'}, direction='write')
        assert warnings == []
        assert model.model_dump()['x_experimental'] == 'v'

    def test_other_unknown_key_single_warning_in_warn_mode(self):
        model, warnings = parse_metadata({'mystery_field': 'v'}, direction='write')
        assert len(warnings) == 1
        assert warnings[0].code == 'unknown_key'
        assert warnings[0].field == 'mystery_field'
        assert model.model_dump()['mystery_field'] == 'v'

    def test_deterministic_invariant_violation_write_enforce_raises(self):
        with pytest.raises(ValidationError):
            parse_metadata({'task_kind': 'deterministic'}, direction='write', enforce=True)

    def test_deterministic_invariant_violation_write_warn_mode_accepts(self):
        model, warnings = parse_metadata(
            {'task_kind': 'deterministic'}, direction='write', enforce=False
        )
        assert len(warnings) == 1
        assert model.task_kind == 'deterministic'

    # Valid JSON that decodes to something other than an object, plus one
    # direct non-dict/non-str Python input exercising the `else: parsed =
    # blob` branch directly (bypassing json.loads entirely).
    _NON_OBJECT_BLOBS = [
        pytest.param('"null"', id='json_string_literal_null'),
        pytest.param('null', id='json_null'),
        pytest.param('42', id='json_int'),
        pytest.param('[1,2]', id='json_list'),
        pytest.param('"str"', id='json_string'),
        pytest.param([1, 2], id='direct_non_str_non_dict_input'),
    ]

    @pytest.mark.parametrize('blob', _NON_OBJECT_BLOBS)
    def test_non_object_blob_read_warns_never_raises(self, blob):
        model, warnings = parse_metadata(blob, direction='read')  # type: ignore[arg-type]
        assert model == TaskMetadata()
        assert len(warnings) == 1
        assert warnings[0].code == 'not_an_object'
        assert warnings[0].field == task_metadata_module._WHOLE_METADATA_FIELD

    @pytest.mark.parametrize('blob', _NON_OBJECT_BLOBS)
    def test_non_object_blob_write_warn_mode_accepts(self, blob):
        model, warnings = parse_metadata(  # type: ignore[arg-type]
            blob, direction='write', enforce=False
        )
        assert model == TaskMetadata()
        assert len(warnings) == 1
        assert warnings[0].code == 'not_an_object'

    @pytest.mark.parametrize('blob', _NON_OBJECT_BLOBS)
    def test_non_object_blob_write_enforce_raises(self, blob):
        with pytest.raises((TypeError, ValueError)):
            parse_metadata(blob, direction='write', enforce=True)  # type: ignore[arg-type]


class _CheckStub(BaseModel):
    """Throwaway list-element sub-model standing in for capability_manifest's
    future DeliveredCheckMeta registration under 'delivered_checks'."""

    name: str


class TestListValuedSubmodelSlice:
    """Generic list-valued registered slice support in parse_metadata.

    Pins the behavior needed by capability_manifest.DeliveredCheckMeta
    (plans/capability-delivered-checks-prd.md §Contract) before that real
    model is wired in: metadata.delivered_checks is a LIST, but the existing
    splat `submodel(**parsed[key])` only handles mapping slices (a list
    raises TypeError). dict-valued slices (milestone, deploy_state) must
    stay on the byte-identical unchanged path.
    """

    _KEY = 'delivered_checks_stub'

    def test_valid_list_slice_validated_and_typed_no_warnings(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata(
            {self._KEY: [{'name': 'a'}, {'name': 'b'}]}, direction='write'
        )
        assert warnings == []
        slice_value = getattr(model, self._KEY)
        assert isinstance(slice_value, list)
        assert all(isinstance(item, _CheckStub) for item in slice_value)
        assert [item.name for item in slice_value] == ['a', 'b']

    def test_valid_list_slice_round_trips_as_plain_dicts(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata(
            {self._KEY: [{'name': 'a'}, {'name': 'b'}]}, direction='write'
        )
        assert warnings == []
        dumped = model.model_dump()[self._KEY]
        assert dumped == [{'name': 'a'}, {'name': 'b'}]
        assert all(not isinstance(item, BaseModel) for item in dumped)

    def test_empty_list_slice_validates_to_empty_list(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata({self._KEY: []}, direction='write')
        assert warnings == []
        assert getattr(model, self._KEY) == []

    def test_malformed_element_read_warns_and_retains_raw_list(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata({self._KEY: [{'name': 'a'}, {}]}, direction='read')
        assert len(warnings) == 1
        assert warnings[0].field == self._KEY
        assert warnings[0].code == 'invalid_submodel'
        assert model.model_dump()[self._KEY] == [{'name': 'a'}, {}]

    def test_malformed_element_write_enforce_raises(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        with pytest.raises(ValidationError):
            parse_metadata({self._KEY: [{'name': 'a'}, {}]}, direction='write', enforce=True)

    def test_malformed_element_write_warn_mode_accepts_raw_list(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata({self._KEY: [{}]}, direction='write', enforce=False)
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert model.model_dump()[self._KEY] == [{}]

    def test_non_mapping_element_in_list_read_warns_never_raises(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        model, warnings = parse_metadata({self._KEY: ['not-a-dict']}, direction='read')
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert model.model_dump()[self._KEY] == ['not-a-dict']

    def test_non_mapping_element_in_list_write_enforce_raises(self):
        register_metadata_submodel(self._KEY, _CheckStub)
        with pytest.raises((ValidationError, TypeError)):
            parse_metadata({self._KEY: ['not-a-dict']}, direction='write', enforce=True)

    def test_existing_dict_slice_milestone_unaffected(self):
        # Regression guard: dict-valued registered slices (e.g. milestone)
        # must stay on the byte-identical submodel(**raw) path.
        model, warnings = parse_metadata(
            {'milestone': {'mode': 'delayed', 'after_secs': 604800}}, direction='write'
        )
        assert warnings == []
        assert isinstance(model.milestone, Milestone)  # type: ignore[attr-defined]
        assert model.model_dump()['milestone'] == {
            'mode': 'delayed',
            'at': None,
            'after_secs': 604800,
        }
