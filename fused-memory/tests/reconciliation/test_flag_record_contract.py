"""Unit tests for the Stage-1 flag-family write-boundary contract (task 4863, #3919).

Drives only the public names of ``fused_memory.reconciliation.flag_record_contract``.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.flag_dedup import build_suppression_payload
from fused_memory.reconciliation.flag_record_contract import (
    STAGE1_FLAG_KIND,
    STAGE1_FLAG_MARKER_KIND,
    STAGE1_FLAG_SUPPRESSION_KIND,
    FlagRecordSchemaError,
    ReconStageFlagRecordWriteRefused,
    canonical_flag_types,
    enforce_flag_record_write_contract,
    flag_record_kinds,
    normalize_flag_record_metadata,
    recon_stage_flag_kind_refusal,
)

SINGULAR_FIELD_KINDS = [STAGE1_FLAG_MARKER_KIND, STAGE1_FLAG_KIND]


class TestRecordKindKeys:
    """#3919 divergence A: which metadata keys name the record kind."""

    def test_marker_kind_gains_source_without_mutating_input(self):
        metadata = {'kind': STAGE1_FLAG_MARKER_KIND, 'task_id': '7', 'flag_type': 'x'}
        snapshot = dict(metadata)

        result = normalize_flag_record_metadata(metadata)

        assert result == {**snapshot, 'source': STAGE1_FLAG_MARKER_KIND}
        assert result is not metadata
        assert metadata == snapshot

    def test_marker_source_gains_kind(self):
        result = normalize_flag_record_metadata(
            {'source': STAGE1_FLAG_MARKER_KIND, 'task_id': '7', 'flag_type': 'x'}
        )

        assert result['kind'] == STAGE1_FLAG_MARKER_KIND
        assert result['source'] == STAGE1_FLAG_MARKER_KIND

    def test_marker_explicit_none_kind_key_is_filled_like_an_absent_one(self):
        result = normalize_flag_record_metadata({'kind': STAGE1_FLAG_MARKER_KIND, 'source': None})

        assert result['source'] == STAGE1_FLAG_MARKER_KIND

    def test_marker_kind_with_foreign_source_is_rejected(self):
        with pytest.raises(FlagRecordSchemaError) as excinfo:
            normalize_flag_record_metadata(
                {'kind': STAGE1_FLAG_MARKER_KIND, 'source': 'targeted_recon'}
            )

        message = str(excinfo.value)
        assert STAGE1_FLAG_MARKER_KIND in message
        assert 'targeted_recon' in message

    def test_two_family_kinds_are_rejected(self):
        with pytest.raises(FlagRecordSchemaError) as excinfo:
            normalize_flag_record_metadata(
                {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'source': STAGE1_FLAG_MARKER_KIND}
            )

        message = str(excinfo.value)
        assert STAGE1_FLAG_SUPPRESSION_KIND in message
        assert STAGE1_FLAG_MARKER_KIND in message

    def test_suppression_stays_kind_only(self):
        metadata = {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'task_id': '42'}

        assert normalize_flag_record_metadata(metadata) == metadata

    def test_suppression_source_alone_is_not_a_flag_record(self):
        metadata = {'source': STAGE1_FLAG_SUPPRESSION_KIND}

        assert flag_record_kinds(metadata) == frozenset()
        assert normalize_flag_record_metadata(metadata) == metadata

    @pytest.mark.parametrize(
        'metadata', [{'kind': 'cycle_summary', 'task_id': '1'}, {}], ids=['cycle_summary', 'empty']
    )
    def test_non_family_metadata_passes_through(self, metadata):
        assert flag_record_kinds(metadata) == frozenset()
        assert normalize_flag_record_metadata(metadata) == metadata

    def test_flag_record_kinds_of_none_is_empty(self):
        assert flag_record_kinds(None) == frozenset()

    def test_flag_record_kinds_names_each_family_kind(self):
        assert flag_record_kinds({'kind': STAGE1_FLAG_KIND}) == frozenset({STAGE1_FLAG_KIND})
        assert flag_record_kinds({'source': STAGE1_FLAG_MARKER_KIND}) == frozenset(
            {STAGE1_FLAG_MARKER_KIND}
        )


class TestSingularFlagTypeField:
    """#3919 divergence B for kinds whose canonical field is singular ``flag_type``."""

    @pytest.mark.parametrize('kind', SINGULAR_FIELD_KINDS)
    def test_single_plural_converts_to_singular(self, kind):
        result = normalize_flag_record_metadata({'kind': kind, 'flag_types': ['x']})

        assert result['flag_type'] == 'x'
        assert 'flag_types' not in result

    @pytest.mark.parametrize('kind', SINGULAR_FIELD_KINDS)
    def test_empty_plural_is_dropped(self, kind):
        result = normalize_flag_record_metadata({'kind': kind, 'flag_types': []})

        assert 'flag_types' not in result
        assert 'flag_type' not in result

    @pytest.mark.parametrize('kind', SINGULAR_FIELD_KINDS)
    def test_multi_valued_plural_is_lossy(self, kind):
        with pytest.raises(FlagRecordSchemaError) as excinfo:
            normalize_flag_record_metadata({'kind': kind, 'flag_types': ['x', 'y']})

        assert 'flag_types' in str(excinfo.value)

    @pytest.mark.parametrize('kind', SINGULAR_FIELD_KINDS)
    def test_agreeing_plural_is_dropped(self, kind):
        result = normalize_flag_record_metadata(
            {'kind': kind, 'flag_type': 'x', 'flag_types': ['x']}
        )

        assert result['flag_type'] == 'x'
        assert 'flag_types' not in result

    @pytest.mark.parametrize('kind', SINGULAR_FIELD_KINDS)
    def test_contradicting_fields_are_rejected(self, kind):
        with pytest.raises(FlagRecordSchemaError) as excinfo:
            normalize_flag_record_metadata({'kind': kind, 'flag_type': 'x', 'flag_types': ['y']})

        message = str(excinfo.value)
        assert "'x'" in message
        assert "'y'" in message


class TestPluralFlagTypesField:
    """#3919 divergence B for the suppression, whose canonical field is plural ``flag_types``."""

    def test_singular_converts_to_plural(self):
        result = normalize_flag_record_metadata(
            {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_type': 'x'}
        )

        assert result['flag_types'] == ['x']
        assert 'flag_type' not in result

    def test_empty_singular_is_blanket(self):
        result = normalize_flag_record_metadata(
            {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_type': ''}
        )

        assert 'flag_type' not in result
        assert 'flag_types' not in result

    def test_plural_is_sorted_and_deduplicated(self):
        result = normalize_flag_record_metadata(
            {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_types': ['b', 'a', 'a']}
        )

        assert result['flag_types'] == ['a', 'b']

    def test_bare_string_plural_becomes_a_list(self):
        result = normalize_flag_record_metadata(
            {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_types': 'x'}
        )

        assert result['flag_types'] == ['x']

    def test_empty_plural_is_dropped(self):
        result = normalize_flag_record_metadata(
            {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_types': []}
        )

        assert 'flag_types' not in result

    def test_contradicting_fields_are_rejected(self):
        with pytest.raises(FlagRecordSchemaError):
            normalize_flag_record_metadata(
                {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'flag_type': 'x', 'flag_types': ['y']}
            )


def test_canonical_flag_types_stringifies_sorts_and_deduplicates():
    assert canonical_flag_types([2, 'b', 'a', 'b']) == ['2', 'a', 'b']


class TestReconStageRefusal:
    def test_suppression_from_recon_stage_is_refused(self):
        refusal = recon_stage_flag_kind_refusal(
            'recon-stage-memory_consolidator', {'kind': STAGE1_FLAG_SUPPRESSION_KIND}
        )

        assert refusal is not None
        assert refusal.error_type == 'ReconFlagSuppressionWriteRejected'
        assert refusal.error == 'flag_suppression_write_blocked'
        assert refusal.hint

    def test_marker_from_recon_stage_is_refused(self):
        refusal = recon_stage_flag_kind_refusal(
            'recon-stage-memory_consolidator', {'source': STAGE1_FLAG_MARKER_KIND}
        )

        assert refusal is not None
        assert refusal.error_type == 'ReconFlagMarkerWriteRejected'
        assert refusal.error == 'flag_marker_write_blocked'
        assert refusal.hint

    @pytest.mark.parametrize(
        'agent_id', [None, 'claude-interactive', 42], ids=['none', 'interactive', 'non-str']
    )
    def test_non_recon_agents_are_not_refused(self, agent_id):
        assert (
            recon_stage_flag_kind_refusal(agent_id, {'kind': STAGE1_FLAG_SUPPRESSION_KIND})
            is None
        )

    def test_non_dict_metadata_is_not_refused(self):
        assert recon_stage_flag_kind_refusal('recon-stage-x', 'stage1_flag_suppression') is None

    def test_stage1_flag_is_normalized_not_refused(self):
        assert recon_stage_flag_kind_refusal('recon-stage-x', {'kind': STAGE1_FLAG_KIND}) is None


class TestEnforceWriteContract:
    def test_recon_stage_suppression_write_raises(self):
        with pytest.raises(ReconStageFlagRecordWriteRefused) as excinfo:
            enforce_flag_record_write_contract(
                {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'task_id': '6107'},
                agent_id='recon-stage-x',
            )

        message = str(excinfo.value)
        assert 'ReconFlagSuppressionWriteRejected' in message
        assert 'recon-stage-x' in message
        assert excinfo.value.agent_id == 'recon-stage-x'
        assert excinfo.value.refusal.error_type == 'ReconFlagSuppressionWriteRejected'

    def test_operator_write_returns_normalized_copy(self):
        metadata = {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'task_id': '6107', 'flag_type': 'x'}

        result = enforce_flag_record_write_contract(metadata, agent_id=None)

        assert result == {'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'task_id': '6107', 'flag_types': ['x']}
        assert result is not metadata

    def test_absent_metadata_becomes_empty_dict(self):
        assert enforce_flag_record_write_contract(None, agent_id=None) == {}


class TestProducerFixedPoints:
    """flag_dedup's suppression producer already emits the canonical shape."""

    def test_scoped_suppression_payload_is_a_fixed_point(self):
        metadata = build_suppression_payload(42, flag_types=['b', 'a'])['metadata']

        assert normalize_flag_record_metadata(metadata) == metadata

    def test_blanket_suppression_payload_is_a_fixed_point(self):
        metadata = build_suppression_payload('42')['metadata']

        assert normalize_flag_record_metadata(metadata) == metadata
