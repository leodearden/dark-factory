"""DoneProvenance as the single home of the per-kind done_provenance contract.

Drives only the model's public surface — ``model_validate``,
``parse_for_write``, ``COMMIT_REQUIRED_KINDS`` and ``recorded_evidence`` — so
the per-kind table behind it stays free to change shape.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from shared.task_metadata import DoneProvenance

DECLARED_KINDS = get_args(DoneProvenance.model_fields['kind'].annotation)
SHA = '0123456789abcdef0123456789abcdef01234567'
FIELD_REQUIRED = 'done_provenance_field_required'


def _first_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict:
    return dict(exc_info.value.errors()[0])


def _complete_payload(kind: str) -> dict:
    """A payload carrying every field any kind can require."""
    return {'kind': kind, 'commit': SHA, 'note': 'n', 'escalation_id': 'esc-1'}


class TestBlankIsAbsent:
    def test_surrounding_whitespace_is_stripped(self):
        dp = DoneProvenance.model_validate({'kind': ' merged ', 'commit': '  abc  '})
        assert dp.kind == 'merged'
        assert dp.commit == 'abc'

    @pytest.mark.parametrize(
        ('payload', 'kind', 'field'),
        [
            ({'kind': 'merged', 'commit': '   '}, 'merged', 'commit'),
            ({'kind': 'found_on_main', 'commit': SHA, 'note': '  '}, 'found_on_main', 'note'),
            (
                {'kind': 'operational-verified', 'note': 'n', 'escalation_id': ' '},
                'operational-verified',
                'escalation_id',
            ),
        ],
    )
    def test_whitespace_only_required_field_counts_as_missing(self, payload, kind, field):
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.model_validate(payload)
        err = _first_error(exc_info)
        assert err['type'] == FIELD_REQUIRED
        assert err['loc'] == ()
        assert err['ctx']['kind'] == kind
        assert err['ctx']['field'] == field

    def test_blank_evidence_reads_back_as_none(self):
        dp = DoneProvenance.model_validate(
            {'kind': 'deterministic-deploy-scheduled', 'unit': '  ', 'transient_unit': ''}
        )
        assert dp.unit is None
        assert dp.transient_unit is None

    def test_whitespace_only_kind_fails_on_kind(self):
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.model_validate({'kind': '   ', 'commit': SHA})
        assert _first_error(exc_info)['loc'] == ('kind',)


class TestRequiredFieldErrors:
    @pytest.mark.parametrize(
        ('kind', 'field'),
        [
            ('merged', 'commit'),
            ('found_on_main', 'commit'),
            ('found_on_main', 'note'),
            ('operational-verified', 'escalation_id'),
            ('operational-verified', 'note'),
        ],
    )
    def test_missing_required_field_is_a_structured_error(self, kind, field):
        payload = _complete_payload(kind)
        del payload[field]
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.model_validate(payload)
        err = _first_error(exc_info)
        assert err['type'] == FIELD_REQUIRED
        assert err['ctx']['kind'] == kind
        assert err['ctx']['field'] == field
        assert field in err['msg']


class TestWriteOnlyRequirement:
    def test_legacy_gate_blob_without_escalation_id_still_reads(self):
        dp = DoneProvenance.model_validate({'kind': 'deterministic-gate'})
        assert dp.escalation_id is None

    def test_gate_write_requires_escalation_id(self):
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.parse_for_write({'kind': 'deterministic-gate'})
        err = _first_error(exc_info)
        assert err['type'] == FIELD_REQUIRED
        assert err['ctx']['kind'] == 'deterministic-gate'
        assert err['ctx']['field'] == 'escalation_id'

    def test_gate_write_with_escalation_id_succeeds(self):
        dp = DoneProvenance.parse_for_write({'kind': 'deterministic-gate', 'escalation_id': 'esc-1'})
        assert isinstance(dp, DoneProvenance)
        assert dp.escalation_id == 'esc-1'

    def test_write_still_enforces_read_requirements(self):
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.parse_for_write({'kind': 'merged'})
        err = _first_error(exc_info)
        assert err['type'] == FIELD_REQUIRED
        assert err['ctx']['field'] == 'commit'


class TestCommitRequiredKinds:
    def test_is_a_frozenset_of_declared_kinds(self):
        kinds = DoneProvenance.COMMIT_REQUIRED_KINDS
        assert isinstance(kinds, frozenset)
        assert kinds <= set(DECLARED_KINDS)

    def test_pinned_membership(self):
        commit_required = DoneProvenance.COMMIT_REQUIRED_KINDS
        assert commit_required == {'merged', 'found_on_main'}

    @pytest.mark.parametrize('kind', DECLARED_KINDS)
    def test_matches_enforced_behaviour(self, kind):
        payload = {'kind': kind, 'note': 'n', 'escalation_id': 'e'}
        if kind in DoneProvenance.COMMIT_REQUIRED_KINDS:
            with pytest.raises(ValidationError) as exc_info:
                DoneProvenance.parse_for_write(payload)
            err = _first_error(exc_info)
            assert err['type'] == FIELD_REQUIRED
            assert err['ctx']['field'] == 'commit'
        else:
            assert DoneProvenance.parse_for_write(payload).kind == kind


class TestTypedEvidence:
    @pytest.mark.parametrize(
        ('payload', 'loc'),
        [
            (
                {'kind': 'deterministic-deploy-scheduled', 'fire_delay_secs': 'soon'},
                ('fire_delay_secs',),
            ),
            ({'kind': 'deterministic-deploy-scheduled', 'transient_unit': 5}, ('transient_unit',)),
            ({'kind': 'deterministic-deploy', 'pid': 'abc'}, ('pid',)),
            ({'kind': 'deterministic-deploy', 'unit': 7}, ('unit',)),
        ],
    )
    def test_wrongly_typed_evidence_is_refused(self, payload, loc):
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.model_validate(payload)
        assert _first_error(exc_info)['loc'] == loc


class TestRecordedEvidence:
    def test_deploy_keeps_only_its_own_evidence(self):
        dp = DoneProvenance.parse_for_write(
            {
                'kind': 'deterministic-deploy',
                'pid': 4242,
                'unit': 'fused-memory.service',
                'active_enter_timestamp': 'Mon 2026-10-12 08:00:00 UTC',
                'note': 'restarted',
                'escalation_id': 'esc-foreign',
                'transient_unit': 'foreign.service',
                'x_foo': 1,
            }
        )
        assert dp.recorded_evidence() == {
            'kind': 'deterministic-deploy',
            'note': 'restarted',
            'unit': 'fused-memory.service',
            'pid': 4242,
            'active_enter_timestamp': 'Mon 2026-10-12 08:00:00 UTC',
        }

    def test_scheduled_deploy_keeps_schedule_evidence_and_drops_pid(self):
        dp = DoneProvenance.parse_for_write(
            {
                'kind': 'deterministic-deploy-scheduled',
                'unit': 'orchestrator.service',
                'transient_unit': 'orchestrator-restart-1.service',
                'fire_delay_secs': 5,
                'pid': 99,
            }
        )
        assert dp.recorded_evidence() == {
            'kind': 'deterministic-deploy-scheduled',
            'unit': 'orchestrator.service',
            'transient_unit': 'orchestrator-restart-1.service',
            'fire_delay_secs': 5,
        }

    @pytest.mark.parametrize('kind', ['deterministic-gate', 'operational-verified'])
    def test_escalation_citing_kinds_keep_escalation_id(self, kind):
        dp = DoneProvenance.parse_for_write({'kind': kind, 'note': 'n', 'escalation_id': 'esc-7'})
        assert dp.recorded_evidence() == {'kind': kind, 'note': 'n', 'escalation_id': 'esc-7'}

    def test_merged_records_exactly_kind_and_commit(self):
        dp = DoneProvenance.parse_for_write(
            {
                'kind': 'merged',
                'commit': SHA,
                'escalation_id': 'esc-foreign',
                'pid': 1,
                'stamped_at': '2026-10-12T00:00:00+00:00',
                'stale_evidence_override': {'reason': 'r'},
                'x_foo': 1,
            }
        )
        assert dp.recorded_evidence() == {'kind': 'merged', 'commit': SHA}

    def test_absent_fields_are_omitted(self):
        dp = DoneProvenance.parse_for_write({'kind': 'deterministic-deploy', 'unit': 'u.service'})
        assert dp.recorded_evidence() == {'kind': 'deterministic-deploy', 'unit': 'u.service'}

    @pytest.mark.parametrize('kind', DECLARED_KINDS)
    def test_every_declared_kind_records_its_kind(self, kind):
        dp = DoneProvenance.parse_for_write(_complete_payload(kind))
        assert dp.recorded_evidence()['kind'] == kind

    def test_returns_a_fresh_dict_each_call(self):
        dp = DoneProvenance.parse_for_write({'kind': 'merged', 'commit': SHA})
        first = dp.recorded_evidence()
        first['commit'] = 'mutated'
        first['extra'] = True
        assert dp.commit == SHA
        assert dp.recorded_evidence() == {'kind': 'merged', 'commit': SHA}
