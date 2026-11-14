"""Pins flag_identity's persisted forms, as literals, and its flag-subject parsing contract."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest

from fused_memory.reconciliation.flag_identity import (
    CONTENT_FP_FLAG_TYPE,
    CONTENT_FP_HEXLEN,
    CONTENT_FP_PREFIX,
    FlagSubject,
    FlagTypeFamily,
    compute_content_fingerprint_signature,
    compute_flag_signature,
    compute_marker_signature,
    content_fingerprint,
    description_digest,
    extract_flag_uuids,
    is_content_fingerprint_task_id,
    is_valid_marker_task_id,
    normalize_description,
    parse_flag_subject,
    recon_signature_task_id,
    signature_field,
)
from fused_memory.reconciliation.scope_freshness import (
    build_scope_snapshot_metadata,
    snapshot_is_fresh,
)
from fused_memory.server.recon_report import ReconReportState

_FT = 'stale_metadata'

# (id, flag, marker_sig, flag_sig, content_fp_sig, marker_key_valid, recon_task_id)
# Every expected value is a literal captured from the pre-flag_identity code.
_GOLDEN: tuple[tuple[Any, ...], ...] = (
    ('int_task_id', {'task_id': 3105, 'flag_type': _FT},
     ('3105', _FT), ('3105', _FT), None, True, '3105'),
    ('str_task_id', {'task_id': '3105', 'flag_type': _FT},
     ('3105', _FT), ('3105', _FT), None, True, '3105'),
    ('padded_task_id', {'task_id': ' 3105 ', 'flag_type': _FT},
     (' 3105 ', _FT), (' 3105 ', _FT), None, True, '3105'),
    ('zero_task_id', {'task_id': 0, 'flag_type': _FT},
     ('0', _FT), ('0', _FT), None, True, '0'),
    ('composite', {'task_id': '3105,4223', 'flag_type': _FT},
     ('3105,4223', _FT), ('3105,4223', _FT), None, True, '3105,4223'),
    ('composite_reordered', {'task_id': '4223,3105', 'flag_type': _FT},
     ('4223,3105', _FT), ('4223,3105', _FT), None, True, '3105,4223'),
    ('composite_spaced', {'task_id': '3105, 4223', 'flag_type': _FT},
     ('3105, 4223', _FT), ('3105, 4223', _FT), None, True, '3105,4223'),
    ('composite_repeated', {'task_id': '3105,3105', 'flag_type': _FT},
     ('3105,3105', _FT), ('3105,3105', _FT), None, True, '3105'),
    ('composite_blank_part', {'task_id': '3105,,4223', 'flag_type': _FT},
     ('3105,,4223', _FT), ('3105,,4223', _FT), None, False, '3105,4223'),
    ('numeric_order', {'task_id': '12,9', 'flag_type': _FT},
     ('12,9', _FT), ('12,9', _FT), None, True, '9,12'),
    ('alnum_part', {'task_id': 'abc,12', 'flag_type': _FT},
     ('abc,12', _FT), ('abc,12', _FT), None, False, '12,abc'),
    ('qualified_part', {'task_id': 'dark_factory:3105,12', 'flag_type': _FT},
     ('dark_factory:3105,12', _FT), ('dark_factory:3105,12', _FT), None, True,
     '12,dark_factory:3105'),
    ('cited_local',
     {'task_id': 598, 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '3839'}]},
     ('3839,598', _FT), ('3839,598', _FT), None, True, '598'),
    ('cited_foreign',
     {'task_id': 598, 'flag_type': _FT, 'cited_tasks': [{'project_id': 'reify', 'task_id': 650}]},
     ('598,reify:650', _FT), ('598,650', _FT), None, True, '598'),
    ('cited_foreign_padded',
     {'task_id': 598, 'flag_type': _FT,
      'cited_tasks': [{'project_id': ' reify ', 'task_id': ' 650'}]},
     ('598,reify:650', _FT), (' 650,598', _FT), None, True, '598'),
    ('cited_home_padded',
     {'task_id': 598, 'flag_type': _FT,
      'cited_tasks': [{'project_id': ' dark_factory', 'task_id': '3839'}]},
     ('3839,598', _FT), ('3839,598', _FT), None, True, '598'),
    ('cited_invalid_project',
     {'task_id': 598, 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'Reify Project', 'task_id': '650'}]},
     ('598,650', _FT), ('598,650', _FT), None, True, '598'),
    ('cited_nondigit_foreign_id',
     {'task_id': 598, 'flag_type': _FT, 'cited_tasks': [{'project_id': 'reify', 'task_id': 'abc'}]},
     ('598,abc', _FT), ('598,abc', _FT), None, False, '598'),
    ('cited_task_id_none',
     {'task_id': 598, 'flag_type': _FT, 'cited_tasks': [{'project_id': 'reify', 'task_id': None}]},
     ('598', _FT), ('598', _FT), None, True, '598'),
    ('cited_not_a_list',
     {'task_id': 598, 'flag_type': _FT, 'cited_tasks': {'project_id': 'reify', 'task_id': '650'}},
     ('598', _FT), ('598', _FT), None, True, '598'),
    ('cited_only_null_task',
     {'task_id': None, 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '3839'},
                      {'project_id': 'reify', 'task_id': 650}]},
     ('3839,reify:650', _FT), ('3839,650', _FT), None, True, None),
    ('cited_composite_top',
     {'task_id': '3105,12', 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'dark_factory', 'task_id': 9}]},
     ('12,3105,9', _FT), ('12,3105,9', _FT), None, True, '12,3105'),
    ('fp_description',
     {'task_id': None, 'flag_type': _FT, 'description': 'Task 3105 metadata drifted from the tree.'},
     ('fp:af8e27237f9d66e09818f6c7ef207c10', _FT), None,
     ('fp:af8e27237f9d66e09818f6c7ef207c10', _FT), True, None),
    ('fp_description_variant',
     {'task_id': None, 'flag_type': _FT,
      'description': '  task 3105 METADATA drifted\nfrom the tree. '},
     ('fp:af8e27237f9d66e09818f6c7ef207c10', _FT), None,
     ('fp:af8e27237f9d66e09818f6c7ef207c10', _FT), True, None),
    ('fp_blank_description', {'task_id': None, 'flag_type': _FT, 'description': ''},
     None, None, None, None, None),
    ('fp_whitespace_description', {'task_id': None, 'flag_type': _FT, 'description': ' \t\n '},
     None, None, None, None, None),
    ('fp_no_description', {'task_id': None, 'flag_type': _FT},
     None, None, None, None, None),
    ('fp_sentinel_flag_type',
     {'task_id': None, 'flag_type': None, 'description': 'A finding with no flag type.'},
     ('fp:ba319c4e1c583e5633faae547a4770a7', '__content_fp__'), None,
     ('fp:ba319c4e1c583e5633faae547a4770a7', '__content_fp__'), True, None),
    ('fp_empty_flag_type',
     {'task_id': None, 'flag_type': '', 'description': 'A finding with an empty flag type.'},
     ('fp:6fc23fbb39b2bbdf54a0db6820e19c80', '__content_fp__'), None,
     ('fp:6fc23fbb39b2bbdf54a0db6820e19c80', '__content_fp__'), True, None),
    ('fp_cited_null',
     {'task_id': None, 'flag_type': _FT, 'description': 'Cited nothing usable.',
      'cited_tasks': [{'project_id': 'reify', 'task_id': None}]},
     ('fp:b4bddb2b5524841b0d5f58b347510c59', _FT), None,
     ('fp:b4bddb2b5524841b0d5f58b347510c59', _FT), True, None),
    ('fp_key_task_id', {'task_id': 'fp:9216e85ac497b68d93043b64684eb049', 'flag_type': _FT},
     ('fp:9216e85ac497b68d93043b64684eb049', _FT), ('fp:9216e85ac497b68d93043b64684eb049', _FT),
     None, True, 'fp:9216e85ac497b68d93043b64684eb049'),
    ('bool_task_id', {'task_id': True, 'flag_type': _FT},
     ('True', _FT), ('True', _FT), None, False, 'True'),
    ('float_task_id', {'task_id': 3105.0, 'flag_type': _FT},
     ('3105.0', _FT), ('3105.0', _FT), None, False, '3105.0'),
    ('list_task_id', {'task_id': ['3105'], 'flag_type': _FT},
     ("['3105']", _FT), ("['3105']", _FT), None, False, "['3105']"),
    ('empty_task_id', {'task_id': '', 'flag_type': _FT},
     ('', _FT), ('', _FT), None, False, ''),
    ('comma_task_id', {'task_id': ',', 'flag_type': _FT},
     (',', _FT), (',', _FT), None, False, ','),
    ('bool_task_id_cited',
     {'task_id': True, 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'dark_factory', 'task_id': 12}]},
     ('12,True', _FT), ('12,True', _FT), None, False, 'True'),
    ('float_task_id_cited',
     {'task_id': 3105.0, 'flag_type': _FT,
      'cited_tasks': [{'project_id': 'dark_factory', 'task_id': 12}]},
     ('12,3105.0', _FT), ('12,3105.0', _FT), None, False, '3105.0'),
    ('flag_type_none_with_task', {'task_id': 3105, 'flag_type': None},
     None, None, None, None, '3105'),
    ('int_flag_type', {'task_id': 3105, 'flag_type': 7},
     ('3105', '7'), ('3105', '7'), None, True, '3105'),
)

_GOLDEN_IDS = [row[0] for row in _GOLDEN]

# (description, normalized, sha256 digest, content fingerprint)
_DESCRIPTIONS: tuple[tuple[str, str, str, str], ...] = (
    ('', '',
     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
     'fp:e3b0c44298fc1c149afbf4c8996fb924'),
    ('   ', '',
     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
     'fp:e3b0c44298fc1c149afbf4c8996fb924'),
    ('\tTabs\nand\r\nnewlines  ', 'tabs and newlines',
     'c52d62b2ae6d89baee93bc54a797edcf9cdd3b9a46d33ad175b65db030223c40',
     'fp:c52d62b2ae6d89baee93bc54a797edcf'),
    ('MiXeD CaSe Description', 'mixed case description',
     'cf678f7b0140021ad89e211aaa94c991b043f14f25b49093fe2f9a334a45a04f',
     'fp:cf678f7b0140021ad89e211aaa94c991'),
    ('Straße', 'strasse',
     '16d96952087774fee069b7585d3991b24d90c181c09b2129b4908c35baa7f0c0',
     'fp:16d96952087774fee069b7585d3991b2'),
    ('STRASSE', 'strasse',
     '16d96952087774fee069b7585d3991b24d90c181c09b2129b4908c35baa7f0c0',
     'fp:16d96952087774fee069b7585d3991b2'),
    ('The cross-project fix task for this finding was filed in reify but the Stage 1 flag '
     'keeps resurfacing every cycle because its citation set changes.',
     'the cross-project fix task for this finding was filed in reify but the stage 1 flag '
     'keeps resurfacing every cycle because its citation set changes.',
     '81b558c2e0e27bf7d645a50ec16f4ba08478786ee0fd173b0b4434c1b83ab86b',
     'fp:81b558c2e0e27bf7d645a50ec16f4ba0'),
    ('already normalized text', 'already normalized text',
     '5471a4f0c1153548c7fae32cd4a7c458a4d3e42d497f884a5d5addc0e46756aa',
     'fp:5471a4f0c1153548c7fae32cd4a7c458'),
)


class TestGoldenPersistedForms:
    @pytest.mark.parametrize(
        ('row_id', 'flag', 'marker_sig', 'flag_sig', 'content_fp_sig', 'marker_key_valid',
         'recon_task_id'),
        _GOLDEN,
        ids=_GOLDEN_IDS,
    )
    def test_every_identity_function_reproduces_the_stored_form(
        self, row_id, flag, marker_sig, flag_sig, content_fp_sig, marker_key_valid, recon_task_id,
    ):
        assert compute_marker_signature(flag, project_id='dark_factory') == marker_sig
        assert compute_flag_signature(flag) == flag_sig
        assert compute_content_fingerprint_signature(flag) == content_fp_sig
        if marker_sig is not None:
            assert is_valid_marker_task_id(marker_sig[0]) is marker_key_valid
        assert recon_signature_task_id(parse_flag_subject(flag.get('task_id'))) == recon_task_id

    @pytest.mark.parametrize(
        ('description', 'normalized', 'digest', 'fingerprint'), _DESCRIPTIONS,
    )
    def test_description_normalizer_digest_and_fingerprint(
        self, description, normalized, digest, fingerprint,
    ):
        assert normalize_description(description) == normalized
        assert description_digest(description) == digest
        assert content_fingerprint(description) == fingerprint


class TestTotality:
    def test_a_non_str_description_has_no_content_fingerprint_signature(self):
        flag = {'task_id': None, 'description': 42}

        assert compute_content_fingerprint_signature(flag) is None
        assert compute_marker_signature(flag, project_id='dark_factory') is None

    def test_a_flag_with_no_str_values_names_no_uuid(self):
        assert extract_flag_uuids({'a': 1, 'b': None}) == set()


_SNAPSHOT_ARGS: dict[str, Any] = {
    'task_ref': 'dark_factory:2405',
    'flag_key': 'cross_project',
    'subject_project_id': 'dark_factory',
    'subject_task_id': '2405',
    'status': 'pending',
    'updated_at': '2026-07-10T10:00:00Z',
    'run_id': 'run-1',
    'snapshot_at': '2026-07-10T11:00:00Z',
}


def _snapshot_literal(fingerprint: str) -> dict[str, Any]:
    return {
        'kind': 'consolidated_scope_correction',
        'source': 'stage_scope_freshness',
        'task_id': 'dark_factory:2405',
        'flag_type': 'cross_project',
        'subject_project_id': 'dark_factory',
        'subject_task_id': '2405',
        'subject_status': 'pending',
        'subject_updated_at': '2026-07-10T10:00:00Z',
        'subject_description_fingerprint': fingerprint,
        'run_id': 'run-1',
        'snapshot_at': '2026-07-10T11:00:00Z',
        'skip_streak': 0,
    }


def _live(description: str | None) -> dict[str, Any]:
    return {'status': 'pending', 'updatedAt': '2026-07-10T10:00:00Z', 'description': description}


class TestScopeSnapshotRoundTrip:
    _STORED = _snapshot_literal('fp:c6b1d9d9ca3417c224b34597ee6f5551')
    _STORED_NO_DESCRIPTION = _snapshot_literal('fp:e3b0c44298fc1c149afbf4c8996fb924')

    def test_the_write_side_is_byte_equal(self):
        built = build_scope_snapshot_metadata(**_SNAPSHOT_ARGS, description='Some description text.')
        assert built == self._STORED

    def test_an_unchanged_description_reads_back_fresh(self):
        assert snapshot_is_fresh(self._STORED, _live('Some description text.')) is True

    def test_a_whitespace_and_case_variant_reads_back_fresh(self):
        assert snapshot_is_fresh(self._STORED, _live('  some DESCRIPTION\ttext. ')) is True

    def test_a_changed_description_reads_back_stale(self):
        assert snapshot_is_fresh(self._STORED, _live('Another description.')) is False

    def test_no_description_is_the_fingerprint_of_the_empty_string(self):
        built = build_scope_snapshot_metadata(**_SNAPSHOT_ARGS, description=None)
        assert built == self._STORED_NO_DESCRIPTION
        assert snapshot_is_fresh(self._STORED_NO_DESCRIPTION, _live(None)) is True
        assert snapshot_is_fresh(self._STORED_NO_DESCRIPTION, _live('   ')) is True
        assert snapshot_is_fresh(self._STORED_NO_DESCRIPTION, _live('now described')) is False


class TestParseFlagSubject:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('3105', ('3105',)),
            ('3105,4223', ('3105', '4223')),
            ('3105, 4223 ,5080', ('3105', '4223', '5080')),
            ('3105,4223,3105', ('3105', '4223')),
            (3105, ('3105',)),
            ('dark_factory:3105,3105.1', ('dark_factory:3105', '3105.1')),
        ],
    )
    def test_decomposes_str_and_int_values(self, raw: Any, expected: tuple[str, ...]) -> None:
        assert parse_flag_subject(raw).components == expected

    @pytest.mark.parametrize(('raw', 'expected'), [(-5, ('-5',)), ('0,3105', ('0', '3105'))])
    def test_does_not_judge_usability(self, raw: Any, expected: tuple[str, ...]) -> None:
        assert parse_flag_subject(raw).components == expected

    @pytest.mark.parametrize(
        'raw', [True, False, None, 3105.0, [], ['3105'], {'task_id': '3105'}, b'3105'],
    )
    def test_a_value_that_is_not_a_task_id_has_no_components(self, raw: Any) -> None:
        assert parse_flag_subject(raw).components == ()

    @pytest.mark.parametrize('raw', [',', ',,', ' , ', ''])
    def test_a_separator_only_or_blank_value_has_no_components(self, raw: str) -> None:
        assert parse_flag_subject(raw).components == ()

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [(None, None), (3105, '3105'), (True, 'True'), (' 3105 ', ' 3105 '), ('', '')],
    )
    def test_raw_is_the_signature_field_of_the_value(self, raw: Any, expected: str | None) -> None:
        assert parse_flag_subject(raw).raw == expected

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [('3105,4223', True), ('3105,3105', False), ('3105', False), (None, False), (',', False)],
    )
    def test_is_composite_only_for_two_or_more_components(self, raw: Any, expected: bool) -> None:
        assert parse_flag_subject(raw).is_composite is expected

    def test_covered_by_names_the_components_another_set_holds(self):
        assert parse_flag_subject('3105,4223').covered_by(['4223', '9']) == frozenset({'4223'})

    def test_flag_subject_is_frozen(self):
        subject = parse_flag_subject('3105')
        assert isinstance(subject, FlagSubject)
        for field in dataclasses.fields(subject):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(subject, field.name, '4223')


class TestReconSignatureTaskId:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('4223, 3105', '3105,4223'),
            ('12,9', '9,12'),
            ('fp:9216e85ac497b68d93043b64684eb049', 'fp:9216e85ac497b68d93043b64684eb049'),
            ('', ''),
            (',', ','),
            (None, None),
        ],
    )
    def test_recon_form(self, raw: Any, expected: str | None) -> None:
        assert recon_signature_task_id(parse_flag_subject(raw)) == expected

    @pytest.mark.parametrize(('value', 'expected'), [(None, None), (1976, '1976'), ('x', 'x')])
    def test_signature_field(self, value: Any, expected: str | None) -> None:
        assert signature_field(value) == expected


class TestMarkerKeyAntiDrift:
    @pytest.mark.parametrize('description', [row[0] for row in _DESCRIPTIONS])
    def test_every_emitted_fingerprint_is_a_valid_fp_marker_key(self, description):
        fingerprint = content_fingerprint(description)
        assert is_valid_marker_task_id(fingerprint) is True
        assert is_content_fingerprint_task_id(fingerprint) is True

    def test_the_persisted_fingerprint_constants(self):
        assert CONTENT_FP_PREFIX == 'fp:'
        assert CONTENT_FP_HEXLEN == 32
        assert CONTENT_FP_FLAG_TYPE == '__content_fp__'


class TestFlagTypeFamily:
    _FAMILY = FlagTypeFamily(
        name='widget_anomaly',
        spellings=frozenset({'widget_anomaly_detected'}),
        drift_tokens=frozenset({'widget'}),
    )

    def test_without_unseen_tokens_only_canonical_variants_match(self):
        assert self._FAMILY.matches('Detected-Widget Anomaly') is True
        assert self._FAMILY.matches('widget_anomaly_detected_again') is False, (
            'a family with no unseen_spelling_tokens must not admit an added word'
        )

    def test_log_drift_names_the_event_the_family_and_only_unmatched_types(self, caplog):
        flags = [
            {'flag_type': 'widget_anomaly_detected'},
            {'flag_type': 'widget_drift_suspected'},
            {'flag_type': 'stale_metadata'},
            {'flag_type': None},
        ]

        with caplog.at_level(logging.INFO):
            self._FAMILY.log_drift(flags, log_event='reconciliation.widget_filter_possible_drift')

        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1, messages
        assert messages[0].startswith('reconciliation.widget_filter_possible_drift ')
        assert 'family=widget_anomaly' in messages[0]
        assert "unmatched_flag_types=['widget_drift_suspected']" in messages[0]

    def test_log_drift_is_silent_when_every_token_bearer_matches(self, caplog):
        with caplog.at_level(logging.INFO):
            self._FAMILY.log_drift(
                [{'flag_type': 'Widget-Anomaly Detected'}], log_event='reconciliation.x',
            )

        assert caplog.records == []

    def test_log_drift_logs_a_flag_type_carrying_any_drift_token(self, caplog):
        family = FlagTypeFamily(
            name='authorship_like',
            spellings=frozenset({'possible_injection'}),
            drift_tokens=frozenset({'inject', 'fabricat'}),
        )
        flags = [
            {'flag_type': 'injection_risk'},
            {'flag_type': 'Fabrication_Suspected'},
            {'flag_type': 'Possible-Injection'},
            {'flag_type': 'stale_metadata'},
        ]

        with caplog.at_level(logging.INFO):
            family.log_drift(flags, log_event='reconciliation.x_filter_possible_drift')

        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1, messages
        assert messages[0].startswith('reconciliation.x_filter_possible_drift ')
        assert "unmatched_flag_types=['injection_risk', 'Fabrication_Suspected']" in messages[0]

    @pytest.mark.parametrize(
        'drift_tokens',
        [frozenset(), frozenset({''}), frozenset({'Widget'}), 'widget', {'widget'}],
        ids=['empty_set', 'empty_token', 'not_casefolded', 'bare_str', 'mutable_set'],
    )
    def test_construction_rejects_malformed_drift_tokens(self, drift_tokens):
        with pytest.raises(ValueError, match='drift_tokens'):
            FlagTypeFamily(name='w', spellings=frozenset({'w_x'}), drift_tokens=drift_tokens)


_U1 = 'b0057f3d-1234-4abc-8def-0123456789ab'
_U1_UPPER = 'B0057F3D-1234-4ABC-8DEF-0123456789AB'
_U2 = 'a1b2c3d4-5678-4901-8234-567890abcdef'
_U3 = 'ffeeddcc-0011-4223-8445-66778899aabb'


class TestExtractFlagUuids:
    def test_collects_lowercased_uuids_from_every_nested_str_value(self):
        flag = {
            'description': f'deleted {_U1_UPPER}',
            'evidence': [f'see {_U2}', {'note': f'and {_U3}'}],
            'cited_memories': [{'memory_id': _U1, 'store': 'mem0'}],
        }

        assert extract_flag_uuids(flag) == {_U1, _U2, _U3}

    def test_non_str_values_are_ignored(self):
        flag = {'task_id': 165, 'score': 0.5, 'ok': True, 'none': None, 'd': f'{_U2}'}

        assert extract_flag_uuids(flag) == {_U2}

    def test_empty_flag_yields_empty_set(self):
        assert extract_flag_uuids({}) == set()

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('no uuids here at all, just 12345', set()),
            ('', set()),
            (f'the entity {_U1_UPPER} is fine', {_U1}),
            (f'{_U1} and also {_U2}', {_U1, _U2}),
            (f'{_U1} cites {_U1} again', {_U1}),
            (f'{_U1_UPPER} vs {_U1}', {_U1}),
        ],
        ids=['no_uuid', 'empty', 'mixed_case_lowered', 'two_distinct', 'repeat_deduped',
             'case_variants_are_one'],
    )
    def test_uuids_in_a_description(self, text, expected):
        assert extract_flag_uuids({'description': text}) == expected

    def test_uuids_at_every_nesting_depth(self):
        flag = {
            'description': f'{_U1}',
            'evidence': {'deep': {'deeper': f'{_U2}'}},
            'list': [f'{_U3}'],
        }
        assert extract_flag_uuids(flag) == {_U1, _U2, _U3}

    def test_a_flag_whose_values_are_not_str_names_no_uuid(self):
        assert extract_flag_uuids({'a': 1, 'b': None, 'c': [2, 3]}) == set()


_RECON_ROWS = [
    pytest.param(row[1]['task_id'], row[1]['flag_type'], row[6], id=row[0])
    for row in _GOLDEN
    if isinstance(row[1]['flag_type'], str)
    and (row[1]['task_id'] is None or type(row[1]['task_id']) is str)
]


class _AnyTaskInterceptor:
    async def get_task(self, task_id: Any, project_root: str, tag: Any = None) -> dict[str, Any]:
        return {'id': task_id, 'title': f'T-{task_id}'}


class TestReconReportStoresTheGoldenForms:
    """recon_report's stored forms, read only through ReconReportState's public seams."""

    @staticmethod
    def _state() -> ReconReportState:
        state = ReconReportState(
            ttl_seconds=300, clock=lambda: 0.0, task_interceptor=_AnyTaskInterceptor(),
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        return state

    @staticmethod
    def _add(
        state: ReconReportState,
        *,
        task_id: str | None,
        flag_type: str | None,
        description: str = 'd',
    ) -> dict[str, Any]:
        return state.add_finding(
            run_id='r1', severity='moderate', category='memory_stale',
            description=description, suggested_action='a', actionable=True,
            task_id=task_id,
            flag_type=flag_type,
        )

    @pytest.mark.parametrize(('task_id', 'flag_type', 'recon_task_id'), _RECON_ROWS)
    def test_a_finding_stores_the_golden_recon_task_id(self, task_id, flag_type, recon_task_id):
        state = self._state()

        result = self._add(state, task_id=task_id, flag_type=flag_type)

        assert 'finding_id' in result, result
        [stored] = state.get_findings_for_run('r1')
        assert stored['task_id'] == recon_task_id

    @pytest.mark.parametrize('respelled', ['4223, 3105', ' 3105 ,4223,', '4223,3105,3105'])
    def test_a_respelled_composite_is_the_same_finding(self, respelled):
        state = self._state()
        first = self._add(state, task_id='3105,4223', flag_type=_FT)

        second = self._add(state, task_id=respelled, flag_type=_FT, description='other text')

        assert second.get('error') == 'duplicate_finding', second
        assert second.get('existing_finding_id') == first['finding_id']

    def test_null_null_findings_dedup_on_the_normalized_description(self):
        state = self._state()
        first = self._add(state, task_id=None, flag_type=None, description='Stale Edge  d592ca46')

        second = self._add(state, task_id=None, flag_type=None, description=' stale edge\td592CA46 ')

        assert second.get('error') == 'duplicate_finding', second
        assert second.get('existing_finding_id') == first['finding_id']

    def test_a_blank_null_null_description_never_dedups(self):
        state = self._state()

        first = self._add(state, task_id=None, flag_type=None, description='   ')
        second = self._add(state, task_id=None, flag_type=None, description='   ')

        assert 'finding_id' in first, first
        assert 'finding_id' in second, second
        assert first['finding_id'] != second['finding_id']

    @pytest.mark.asyncio
    async def test_a_citation_of_a_composite_member_anchors_the_entity_fold(self):
        state = self._state()
        composite = self._add(state, task_id='5040,5149', flag_type='X')
        cited = await state.cite_task('r1', composite['finding_id'], 'dark_factory', '5149')
        assert 'error' not in cited, cited

        later = self._add(state, task_id=None, flag_type='X', description='worded differently')
        folded = await state.cite_task('r1', later['finding_id'], 'dark_factory', '5149')

        assert folded.get('error') == 'duplicate_finding', folded
        assert folded.get('existing_finding_id') == composite['finding_id']
        assert [f['finding_id'] for f in state.get_findings_for_run('r1')] == [
            composite['finding_id'],
        ]

    @pytest.mark.asyncio
    async def test_a_later_finding_on_the_cited_member_collapses_onto_the_composite(self):
        state = self._state()
        composite = self._add(state, task_id='5040,5149', flag_type='X')
        cited = await state.cite_task('r1', composite['finding_id'], 'dark_factory', '5149')
        assert 'error' not in cited, cited

        later = self._add(state, task_id='5149', flag_type='X', description='worded differently')

        assert later.get('error') == 'duplicate_finding', later
        assert later.get('existing_finding_id') == composite['finding_id']
