"""The durable queue's value vocabulary: ``fused_memory.services.queued_writes``.

One frozen value type per queued operation, a wire that keeps today's flat key
names, legacy rows that still decode, and one error type for a row that cannot
(plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md C1, D1, D2).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import TypeVar

import pytest
from _ast_guard import parse_python_module
from _leaf_import_guard import imported_modules, stdlib_or_allowed
from _queued_write_fixtures import EPISODE_WIRE_KEYS
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.services.queued_writes import (
    NO_REFERENTS,
    DecodedReferents,
    QueueCallback,
    QueueDecodeError,
    QueuedEpisodeWrite,
    QueuedGraphitiMemoryWrite,
    QueuedMem0Add,
    QueuedMem0Fact,
    QueueOperation,
    UnknownQueueOperation,
    WriteEnvelope,
    check_callback_pairing,
    decode_queue_callback,
    decode_queued_write,
)
from fused_memory.utils.canonical_labels import Referent
from fused_memory.utils.referent_resolution import ReferentResolution

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

QUEUED_WRITES = (
    pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory' / 'services' / 'queued_writes.py'
)

ENVELOPE = WriteEnvelope(
    project_id='dark_factory',
    group_id='dark_factory',
    agent_id='agent-1',
    session_id='session-1',
    causation_id='cause-1',
    write_op_id='op-1',
)
FACT_ENVELOPE = dataclasses.replace(ENVELOPE, group_id='mem0_dark_factory', write_op_id=None)
REFERENTS = DecodedReferents(
    (Referent(number='3127'), Referent(number='2500', project_id='reify')),
    'derived',
    (Referent(number='6379', project_id='reify'),),
)


def _episode(**overrides) -> QueuedEpisodeWrite:
    fields = {
        'envelope': ENVELOPE,
        'content': 'Task 3127 landed; reify:2500 is next',
        'name': 'episode_abcd1234',
        'source': 'text',
        'source_description': 'notes',
        'correlation_id': 'ep-1',
        'temporal_context': 'planning',
        'unverified_claim': True,
        'reference_time': datetime(2026, 11, 1, 12, 30, tzinfo=UTC),
        'referents': REFERENTS,
    } | overrides
    return QueuedEpisodeWrite(**fields)


def _graphiti_memory(**overrides) -> QueuedGraphitiMemoryWrite:
    fields = {
        'envelope': ENVELOPE,
        'content': 'Task 3127 chose retries in the caller',
        'name': 'memory_decisions_and_rationale',
        'source': 'text',
        'source_description': 'add_memory:decisions_and_rationale',
        'memory_category': 'decisions_and_rationale',
        'referents': REFERENTS,
        'unverified_claim': True,
    } | overrides
    return QueuedGraphitiMemoryWrite(**fields)


def _mem0_fact(**overrides) -> QueuedMem0Fact:
    fields = {
        'envelope': FACT_ENVELOPE,
        'fact_text': 'Task 3127 RELATES_TO reify:2500',
        'temporal_context': 'planning',
        'unverified_claim': True,
    } | overrides
    return QueuedMem0Fact(**fields)


def _mem0_add(**overrides) -> QueuedMem0Add:
    fields = {
        'envelope': ENVELOPE,
        'content': 'a legacy mem0_add row',
        'metadata': {'category': 'procedural_knowledge', 'task_id': '3127'},
    } | overrides
    return QueuedMem0Add(**fields)


ALL_VALUES = [
    pytest.param(_episode(), id='episode'),
    pytest.param(_episode(reference_time=None, referents=NO_REFERENTS), id='episode-minimal'),
    pytest.param(_graphiti_memory(), id='graphiti-memory'),
    pytest.param(
        _graphiti_memory(
            envelope=dataclasses.replace(ENVELOPE, causation_id=None, write_op_id=None),
            memory_category=None,
        ),
        id='graphiti-memory-replay',
    ),
    pytest.param(_mem0_fact(), id='mem0-fact'),
    pytest.param(_mem0_add(), id='mem0-add'),
]


_Value = TypeVar(
    '_Value', QueuedEpisodeWrite, QueuedGraphitiMemoryWrite, QueuedMem0Fact, QueuedMem0Add,
)


def _decode(value_type: type[_Value], body: dict, *, group_id: str = 'dark_factory') -> _Value:
    value = decode_queued_write(
        value_type.operation.value, json.dumps(body), group_id=group_id, row_id=7,
    )
    assert isinstance(value, value_type)
    return value


class TestVocabulary:
    def test_queue_operations(self):
        assert {member.name: member.value for member in QueueOperation} == {
            'ADD_EPISODE': 'add_episode',
            'ADD_MEMORY_GRAPHITI': 'add_memory_graphiti',
            'MEM0_CLASSIFY_AND_ADD': 'mem0_classify_and_add',
            'MEM0_ADD': 'mem0_add',
        }

    def test_queue_callbacks(self):
        assert {member.name: member.value for member in QueueCallback} == {
            'DUAL_WRITE_EPISODE': 'dual_write_episode',
            'REFRESH_ENTITY_SUMMARIES': 'refresh_entity_summaries',
        }

    @pytest.mark.parametrize(
        ('value_type', 'operation'),
        [
            (QueuedEpisodeWrite, QueueOperation.ADD_EPISODE),
            (QueuedGraphitiMemoryWrite, QueueOperation.ADD_MEMORY_GRAPHITI),
            (QueuedMem0Fact, QueueOperation.MEM0_CLASSIFY_AND_ADD),
            (QueuedMem0Add, QueueOperation.MEM0_ADD),
        ],
    )
    def test_each_value_type_names_its_operation(self, value_type, operation):
        assert value_type.operation is operation

    @pytest.mark.parametrize('operation', list(QueueOperation))
    def test_every_operation_has_a_decoder(self, operation):
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queued_write(operation.value, '{}', group_id='dark_factory', row_id=1)
        assert not isinstance(excinfo.value, UnknownQueueOperation)


class TestFrozenRoundTrip:
    @pytest.mark.parametrize('value', ALL_VALUES)
    def test_from_json_inverts_to_json(self, value):
        assert type(value).from_json(value.to_json(), group_id=value.envelope.group_id) == value

    @pytest.mark.parametrize('value', ALL_VALUES)
    def test_the_row_decoder_inverts_to_json(self, value):
        decoded = decode_queued_write(
            value.operation.value, value.to_json(), group_id=value.envelope.group_id, row_id=1,
        )
        assert decoded == value
        assert type(decoded) is type(value)

    @pytest.mark.parametrize('value', ALL_VALUES)
    def test_every_field_is_frozen(self, value):
        for field in dataclasses.fields(value):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(value, field.name, None)

    @pytest.mark.parametrize('frozen', [ENVELOPE, REFERENTS], ids=['envelope', 'referents'])
    def test_the_envelope_and_referents_are_frozen(self, frozen):
        for field in dataclasses.fields(frozen):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(frozen, field.name, None)

    def test_mem0_add_metadata_cannot_be_mutated_through_the_value(self):
        source = {'category': 'procedural_knowledge'}
        value = _mem0_add(metadata=source)
        source['category'] = 'changed'
        assert value.metadata['category'] == 'procedural_knowledge'
        assert not isinstance(value.metadata, MutableMapping)


class TestWireKeys:
    def test_episode_keys_are_exactly_todays(self):
        assert set(json.loads(_episode().to_json())) == EPISODE_WIRE_KEYS

    def test_graphiti_memory_keys_cover_todays(self):
        assert set(json.loads(_graphiti_memory().to_json())) >= {
            'name', 'content', 'source', 'group_id', 'source_description',
            'memory_category', '_causation_id', '_write_op_id', 'referents',
            'unverified_claim',
        }

    def test_mem0_fact_keys_cover_todays(self):
        assert set(json.loads(_mem0_fact().to_json())) >= {
            'fact_text', 'project_id', 'agent_id', 'session_id', '_causation_id',
            'temporal_context', 'unverified_claim',
        }

    def test_the_envelope_writes_todays_flat_names(self):
        body = json.loads(_episode().to_json())
        assert {key: body[key] for key in (
            'project_id', 'group_id', 'agent_id', 'session_id', '_causation_id', '_write_op_id',
        )} == {
            'project_id': 'dark_factory',
            'group_id': 'dark_factory',
            'agent_id': 'agent-1',
            'session_id': 'session-1',
            '_causation_id': 'cause-1',
            '_write_op_id': 'op-1',
        }

    def test_reference_time_encodes_as_isoformat(self):
        stamp = datetime(2026, 11, 1, 12, 30, 15, 123456, tzinfo=UTC)
        body = json.loads(_episode(reference_time=stamp).to_json())
        assert body['reference_time'] == stamp.isoformat()

    def test_an_absent_reference_time_encodes_as_null(self):
        assert json.loads(_episode(reference_time=None).to_json())['reference_time'] is None

    def test_the_referent_blob_is_todays_shape(self):
        assert json.loads(_episode().to_json())['referents'] == {
            'source': 'derived',
            'refs': [
                {'kind': 'task', 'project_id': '', 'number': '3127'},
                {'kind': 'task', 'project_id': 'reify', 'number': '2500'},
            ],
            'ambiguous': [{'kind': 'task', 'project_id': 'reify', 'number': '6379'}],
        }


class TestLegacyRowsDecode:
    """Rows in the shapes producers wrote before the value types (``6551b75aba``)."""

    def test_add_memory_graphiti_without_project_id_derives_it_from_the_group(self):
        value = _decode(QueuedGraphitiMemoryWrite, {
            'name': 'memory_decisions_and_rationale',
            'content': 'Task 3127 chose retries in the caller',
            'source': 'text',
            'group_id': 'reify',
            'source_description': 'add_memory:decisions_and_rationale',
            'memory_category': 'decisions_and_rationale',
            '_causation_id': 'cause-1',
            '_write_op_id': 'op-1',
            'referents': {'source': 'none', 'refs': [], 'ambiguous': []},
            'unverified_claim': False,
        }, group_id='reify')

        assert isinstance(value, QueuedGraphitiMemoryWrite)
        assert value.envelope.project_id == 'reify'
        assert value.envelope.group_id == 'reify'
        assert value.envelope.write_op_id == 'op-1'
        assert value.memory_category == 'decisions_and_rationale'

    def test_a_replay_from_store_row_decodes_with_no_journal_ids(self):
        value = _decode(QueuedGraphitiMemoryWrite, {
            'name': 'replay_observations_and_summaries',
            'content': 'replayed prose',
            'source': 'text',
            'group_id': 'reify',
            'source_description': 'replay_from_mem0:observations_and_summaries',
            'referents': {'source': 'none', 'refs': [], 'ambiguous': []},
        }, group_id='reify')

        assert isinstance(value, QueuedGraphitiMemoryWrite)
        assert value.envelope.write_op_id is None
        assert value.envelope.causation_id is None
        assert value.memory_category is None
        assert value.unverified_claim is False
        assert value.envelope.project_id == 'reify'

    def test_a_pre_3561_uuid_key_is_ignored_and_warned_about(self, caplog):
        with caplog.at_level('WARNING'):
            value = _decode(QueuedEpisodeWrite, {
                'uuid': 'legacy-uuid',
                'name': 'episode_test',
                'content': 'test content',
                'source': 'text',
                'group_id': 'dark_factory',
                'source_description': 'notes',
            })

        assert isinstance(value, QueuedEpisodeWrite)
        assert not hasattr(value, 'uuid')
        assert 'uuid' not in json.loads(value.to_json())
        assert any("Ignoring legacy 'uuid'" in rec.getMessage() for rec in caplog.records)

    def test_a_row_with_no_referents_key_decodes_to_no_referents(self, caplog):
        with caplog.at_level('WARNING'):
            value = _decode(QueuedEpisodeWrite, {'content': 'c', 'group_id': 'dark_factory'})

        assert value.referents == DecodedReferents((), 'none', None)
        assert value.referents == NO_REFERENTS
        assert not caplog.records

    def test_a_referent_blob_without_ambiguous_decodes_its_refs(self):
        value = _decode(QueuedEpisodeWrite, {
            'content': 'c',
            'group_id': 'dark_factory',
            'referents': {'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127'}]},
        })

        assert value.referents.refs == (Referent(number='3127'),)
        assert value.referents.source == 'derived'
        assert value.referents.ambiguous is None

    def test_a_mem0_fact_without_project_id_defaults_to_main(self):
        value = _decode(QueuedMem0Fact, {
            'fact_text': 'a fact',
            'agent_id': None,
            'session_id': None,
            '_causation_id': 'cause-1',
            'temporal_context': None,
            'unverified_claim': False,
        }, group_id='mem0_main')

        assert isinstance(value, QueuedMem0Fact)
        assert value.envelope.project_id == 'main'
        assert value.envelope.group_id == 'mem0_main'
        assert value.envelope.causation_id == 'cause-1'
        assert value.envelope.write_op_id is None

    def test_an_invalid_reference_time_decodes_to_none_and_warns(self, caplog):
        with caplog.at_level('WARNING'):
            value = _decode(QueuedEpisodeWrite, {
                'content': 'c', 'group_id': 'dark_factory', 'reference_time': 'not-a-time',
            })

        assert value.reference_time is None
        assert any(
            'Invalid reference_time' in rec.getMessage()
            and 'treating as None' in rec.getMessage()
            for rec in caplog.records
        )

    def test_a_valid_reference_time_decodes_to_a_datetime(self):
        stamp = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)
        value = _decode(QueuedEpisodeWrite, {
            'content': 'c', 'group_id': 'dark_factory', 'reference_time': stamp.isoformat(),
        })
        assert value.reference_time == stamp

    def test_unknown_extra_keys_are_ignored(self):
        value = _decode(QueuedEpisodeWrite, {
            'content': 'c', 'group_id': 'dark_factory', 'some_future_key': [1, 2],
        })
        assert value.content == 'c'
        assert 'some_future_key' not in json.loads(value.to_json())

    def test_a_mem0_add_row_decodes(self):
        value = _decode(QueuedMem0Add, {
            'content': 'c', 'project_id': 'dark_factory', 'metadata': {'category': 'x'},
        }, group_id='mem0_dark_factory')
        assert isinstance(value, QueuedMem0Add)
        assert value.envelope.project_id == 'dark_factory'
        assert dict(value.metadata) == {'category': 'x'}

    @pytest.mark.parametrize('absent', [{}, dict.fromkeys(
        ('name', 'source', 'source_description', 'correlation_id', 'temporal_context',
         'unverified_claim', 'reference_time', 'referents', 'memory_category'),
    )], ids=['absent', 'null'])
    @pytest.mark.parametrize(
        ('value_type', 'required', 'group_id'),
        [
            (QueuedEpisodeWrite, {'content': 'c'}, 'dark_factory'),
            (QueuedGraphitiMemoryWrite, {'content': 'c'}, 'dark_factory'),
            (QueuedMem0Fact, {'fact_text': 'c', 'project_id': 'dark_factory'}, 'mem0_dark_factory'),
        ],
    )
    def test_an_absent_or_null_optional_key_takes_the_constructor_default(
        self, value_type, required, group_id, absent,
    ):
        expected = value_type(
            envelope=WriteEnvelope(project_id='dark_factory', group_id=group_id),
            **{key: value for key, value in required.items() if key != 'project_id'},
        )
        assert _decode(value_type, required | absent, group_id=group_id) == expected


class TestDecodeErrors:
    def test_an_unknown_operation(self):
        payload = json.dumps({'content': 'looks like an episode', 'group_id': 'dark_factory'})
        with pytest.raises(UnknownQueueOperation) as excinfo:
            decode_queued_write('typo_op', payload, group_id='dark_factory', row_id=42)

        exc = excinfo.value
        assert isinstance(exc, QueueDecodeError)
        assert isinstance(exc, ValueError)
        assert exc.operation == 'typo_op'
        assert exc.row_id == 42
        assert exc.raw_payload == payload
        assert 'row 42' in str(exc)
        assert "'typo_op'" in str(exc)

    @pytest.mark.parametrize(
        'payload',
        [
            pytest.param('{not json', id='non-json'),
            pytest.param('[]', id='json-array'),
            pytest.param('"text"', id='json-string'),
            pytest.param(json.dumps({'group_id': 'dark_factory'}), id='content-missing'),
            pytest.param(json.dumps({'content': None}), id='content-null'),
            pytest.param(json.dumps({'content': 7}), id='content-not-a-string'),
            pytest.param(
                json.dumps({'content': 'c', 'group_id': 'reify'}), id='body-group-disagrees',
            ),
            pytest.param(json.dumps({'content': 'c', 'agent_id': 7}), id='agent-id-not-a-string'),
        ],
    )
    def test_an_undecodable_episode_body(self, payload):
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queued_write('add_episode', payload, group_id='dark_factory', row_id=9)

        exc = excinfo.value
        assert not isinstance(exc, UnknownQueueOperation)
        assert exc.operation == 'add_episode'
        assert exc.row_id == 9
        assert exc.raw_payload == payload
        assert str(exc).startswith("row 9 ('add_episode'): ")

    def test_a_group_mismatch_names_both_groups(self):
        with pytest.raises(QueueDecodeError) as excinfo:
            _decode(QueuedEpisodeWrite, {'content': 'c', 'group_id': 'reify'}, group_id='dark_factory')
        assert 'reify' in str(excinfo.value)
        assert 'dark_factory' in str(excinfo.value)

    def test_the_raw_payload_is_not_in_the_message(self):
        payload = json.dumps({'group_id': 'dark_factory', 'secret_body': 'x' * 50})
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queued_write('add_episode', payload, group_id='dark_factory', row_id=1)
        assert 'secret_body' not in str(excinfo.value)

    def test_mem0_add_requires_project_id(self):
        with pytest.raises(QueueDecodeError):
            _decode(QueuedMem0Add, {'content': 'c'}, group_id='mem0_main')

    def test_mem0_fact_requires_fact_text(self):
        with pytest.raises(QueueDecodeError):
            _decode(QueuedMem0Fact, {'project_id': 'main'}, group_id='mem0_main')

    def test_the_cause_is_chained(self):
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queued_write('add_episode', '{not json', group_id='g', row_id=1)
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


#: Every pairing a producer makes, and the only ones a row may name.
_PAIRINGS = [
    pytest.param(QueueCallback.DUAL_WRITE_EPISODE, _episode(), id='dual-write-episode'),
    pytest.param(QueueCallback.REFRESH_ENTITY_SUMMARIES, _episode(), id='refresh-episode'),
    pytest.param(
        QueueCallback.REFRESH_ENTITY_SUMMARIES, _graphiti_memory(), id='refresh-graphiti-memory',
    ),
]

_MISPAIRINGS = [
    pytest.param(QueueCallback.DUAL_WRITE_EPISODE, _graphiti_memory(), id='dual-write-graphiti-memory'),
    pytest.param(QueueCallback.DUAL_WRITE_EPISODE, _mem0_fact(), id='dual-write-mem0-fact'),
    pytest.param(QueueCallback.REFRESH_ENTITY_SUMMARIES, _mem0_fact(), id='refresh-mem0-fact'),
    pytest.param(QueueCallback.REFRESH_ENTITY_SUMMARIES, _mem0_add(), id='refresh-mem0-add'),
]


class TestDecodeQueueCallback:
    def test_none_is_no_callback(self):
        assert decode_queue_callback(None, _episode(), row_id=1, raw_payload='{}') is None

    @pytest.mark.parametrize(('callback', 'value'), _PAIRINGS)
    def test_a_known_callback_for_a_value_it_takes(self, callback, value):
        decoded = decode_queue_callback(callback.value, value, row_id=1, raw_payload='{}')
        assert decoded is callback

    def test_an_unknown_callback(self):
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queue_callback('typo_cb', _episode(), row_id=3, raw_payload='{}')

        exc = excinfo.value
        assert not isinstance(exc, UnknownQueueOperation)
        assert exc.operation == 'add_episode'
        assert exc.row_id == 3
        assert exc.raw_payload == '{}'
        assert 'typo_cb' in str(exc)

    @pytest.mark.parametrize(('callback', 'value'), _MISPAIRINGS)
    def test_a_callback_named_for_a_value_it_cannot_take(self, callback, value):
        with pytest.raises(QueueDecodeError) as excinfo:
            decode_queue_callback(callback.value, value, row_id=4, raw_payload='{}')

        exc = excinfo.value
        assert exc.operation == value.operation.value
        assert exc.row_id == 4
        assert callback.value in str(exc)
        assert value.operation.value in str(exc)


class TestCheckCallbackPairing:
    @pytest.mark.parametrize(('callback', 'value'), _PAIRINGS)
    def test_a_pairing_a_producer_makes_passes(self, callback, value):
        check_callback_pairing(value, callback)

    @pytest.mark.parametrize(('callback', 'value'), _MISPAIRINGS)
    def test_any_other_pairing_is_a_type_error(self, callback, value):
        with pytest.raises(TypeError, match=callback.value):
            check_callback_pairing(value, callback)


class TestWrittenText:
    def test_episode(self):
        assert _episode(content='body').written_text == 'body'

    def test_graphiti_memory(self):
        assert _graphiti_memory(content='body').written_text == 'body'

    def test_mem0_add(self):
        assert _mem0_add(content='body').written_text == 'body'

    def test_mem0_fact(self):
        assert _mem0_fact(fact_text='the fact').written_text == 'the fact'


class TestReferentWireCodec:
    """Resolution -> JSON-safe blob -> referents."""

    def test_to_wire_emits_plain_json_scalars(self):
        blob = DecodedReferents.from_resolution(ReferentResolution(
            source='derived',
            referents=(
                Referent(number='3127'),
                Referent(number='2500', project_id='reify'),
            ),
            ambiguous=(Referent(number='6379', project_id='reify'),),
        )).to_wire()

        assert blob == {
            'source': 'derived',
            'refs': [
                {'kind': 'task', 'project_id': '', 'number': '3127'},
                {'kind': 'task', 'project_id': 'reify', 'number': '2500'},
            ],
            'ambiguous': [
                {'kind': 'task', 'project_id': 'reify', 'number': '6379'},
            ],
        }

    def test_the_blob_survives_a_json_round_trip(self):
        blob = DecodedReferents.from_resolution(ReferentResolution(
            source='metadata', referents=(Referent(number='3129'),),
        )).to_wire()

        assert json.loads(json.dumps(blob)) == blob

    def test_an_empty_resolution_encodes_to_an_explicit_empty_set(self):
        assert DecodedReferents.from_resolution(ReferentResolution(source='none')).to_wire() == {
            'source': 'none', 'refs': [], 'ambiguous': [],
        }

    def test_ambiguity_is_threaded_and_conflicts_are_not(self):
        """Zeta tells an ambiguous endpoint from a conflation only through the
        producer's own ambiguity set; a conflict is reported to the caller at
        resolution time and has no consumer past the queue."""
        blob = DecodedReferents.from_resolution(ReferentResolution(
            source='derived',
            referents=(Referent(number='3127'),),
            ambiguous=(Referent(number='9'),),
            conflicts=(Referent(number='42'),),
        )).to_wire()

        assert set(blob) == {'source', 'refs', 'ambiguous'}
        assert blob['refs'] == [{'kind': 'task', 'project_id': '', 'number': '3127'}]
        assert blob['ambiguous'] == [{'kind': 'task', 'project_id': '', 'number': '9'}]

    def test_from_wire_rebuilds_the_exact_referent_tuples(self):
        resolution = ReferentResolution(
            source='derived',
            referents=(
                Referent(number='3127'),
                Referent(number='2500', project_id='reify'),
            ),
            ambiguous=(
                Referent(number='6379'),
                Referent(number='4242', project_id='reify'),
            ),
        )

        decoded = DecodedReferents.from_wire(DecodedReferents.from_resolution(resolution).to_wire())

        assert decoded == DecodedReferents(
            (Referent(number='3127'), Referent(number='2500', project_id='reify')),
            'derived',
            (Referent(number='6379'), Referent(number='4242', project_id='reify')),
        )

    def test_digits_survive_verbatim(self):
        """'0132' is a different referent from '132'."""
        blob = DecodedReferents.from_resolution(ReferentResolution(
            source='declared', referents=(Referent(number='0132'),),
        )).to_wire()
        assert blob['refs'][0]['number'] == '0132'

        decoded = DecodedReferents.from_wire(blob)
        assert decoded.refs == (Referent(number='0132'),)
        assert decoded.refs[0].number == '0132'
        assert decoded.source == 'declared'
        assert decoded.ambiguous == ()

    def test_a_legacy_blob_round_trips_without_an_ambiguous_key(self):
        legacy = {'source': 'derived', 'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}]}
        decoded = DecodedReferents.from_wire(legacy)
        assert decoded.ambiguous is None
        assert decoded.to_wire() == legacy

    def test_no_referents_omits_the_ambiguous_key(self):
        assert NO_REFERENTS.to_wire() == {'source': 'none', 'refs': []}

    def test_a_source_outside_the_vocabulary_is_refused_at_construction(self):
        with pytest.raises(ValueError, match='declared_v2'):
            dataclasses.replace(NO_REFERENTS, source='declared_v2')


#: Every way a present referent blob can be unreadable. Each must decode to
#: exactly ``NO_REFERENTS``, never a partial set.
_UNREADABLE_BLOBS = [
    pytest.param('garbage', id='non-dict'),
    pytest.param(['derived'], id='list-not-dict'),
    pytest.param({'refs': []}, id='source-missing'),
    pytest.param({'source': 'declared_v2', 'refs': []}, id='source-outside-vocabulary'),
    pytest.param({'source': 'derived', 'refs': 'not-a-list'}, id='refs-not-a-list'),
    pytest.param({'source': 'derived', 'refs': ['3127']}, id='entry-not-a-dict'),
    pytest.param({'source': 'derived', 'refs': [{'kind': 'task'}]}, id='entry-lacks-number'),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': 3127}]},
        id='int-number-compares-unequal-to-its-str-twin',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': None}]},
        id='none-number-would-name-the-node-Task-None',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': ['3127']}]},
        id='unhashable-number-would-raise-inside-a-consumer-set',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127', 'project_id': None}]},
        id='none-project-id',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127', 'project_id': 7}]},
        id='int-project-id',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127'},
            {'kind': 'task', 'number': 9},
        ]},
        id='partially-mistyped-set',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127'},
            {'kind': 'unregistered_kind', 'number': '9'},
        ]},
        id='partially-malformed-set',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [], 'ambiguous': 'not-a-list'},
        id='ambiguous-not-a-list',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127'}], 'ambiguous': None},
        id='ambiguous-explicit-null',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [], 'ambiguous': ['6379']},
        id='ambiguous-entry-not-a-dict',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [], 'ambiguous': [{'kind': 'task', 'number': 6379}]},
        id='ambiguous-int-number',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [], 'ambiguous': [
            {'kind': 'task', 'number': '6379', 'project_id': 7},
        ]},
        id='ambiguous-int-project-id',
    ),
    pytest.param(
        {'source': 'derived', 'refs': [], 'ambiguous': [
            {'kind': 'unregistered_kind', 'number': '6379'},
        ]},
        id='ambiguous-unregistered-kind',
    ),
    pytest.param(
        {
            'source': 'derived',
            'refs': [{'kind': 'task', 'number': '3127'}],
            'ambiguous': [{'kind': 'unregistered_kind', 'number': '6379'}],
        },
        id='good-refs-beside-a-malformed-ambiguous',
    ),
]


class TestReferentWireCodecDegradation:
    """An unreadable blob degrades to ``NO_REFERENTS`` with a WARNING; it never
    raises (that would dead-letter the memory over a telemetry field) and never
    yields a partial set (zeta would read a dropped referent as a conflation)."""

    @pytest.mark.parametrize('blob', _UNREADABLE_BLOBS)
    def test_an_unreadable_blob_degrades_to_no_referents(self, blob):
        assert DecodedReferents.from_wire(blob) == DecodedReferents((), 'none', None)

    @pytest.mark.parametrize('blob', _UNREADABLE_BLOBS)
    def test_an_unreadable_blob_warns(self, blob, caplog):
        with caplog.at_level('WARNING'):
            DecodedReferents.from_wire(blob)

        assert any(
            rec.levelname == 'WARNING' and 'referents' in rec.getMessage()
            for rec in caplog.records
        ), f'no WARNING naming the payload key; got {[r.getMessage() for r in caplog.records]}'

    @pytest.mark.parametrize('blob', _UNREADABLE_BLOBS)
    def test_an_unreadable_blob_still_decodes_the_write(self, blob):
        value = _decode(QueuedEpisodeWrite, {'content': 'c', 'referents': blob})
        assert value.referents == NO_REFERENTS

    def test_an_absent_blob_does_not_warn(self, caplog):
        with caplog.at_level('WARNING'):
            decoded = DecodedReferents.from_wire(None)

        assert decoded == NO_REFERENTS
        assert not caplog.records

    def test_a_legacy_two_key_blob_decodes_its_refs_and_answers_none(self):
        """``None`` means the producer did not say, so the consumer may re-derive;
        ``()`` means it said nothing was ambiguous, which the consumer believes."""
        decoded = DecodedReferents.from_wire({
            'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127'}],
        })

        assert decoded.refs == (Referent(number='3127'),)
        assert decoded.source == 'derived'
        assert decoded.ambiguous is None

    def test_a_legacy_two_key_blob_does_not_warn(self, caplog):
        with caplog.at_level('WARNING'):
            DecodedReferents.from_wire({
                'source': 'derived', 'refs': [{'kind': 'task', 'number': '3127'}],
            })

        assert not caplog.records

    def test_an_explicitly_empty_ambiguous_is_not_none(self):
        decoded = DecodedReferents.from_wire({
            'source': 'derived',
            'refs': [{'kind': 'task', 'number': '3127'}],
            'ambiguous': [],
        })

        assert decoded.ambiguous == ()
        assert decoded.ambiguous is not None

    def test_an_explicitly_null_ambiguous_is_not_the_legacy_blob(self, caplog):
        """A present JSON ``null`` is a corrupt field, not the absent key."""
        with caplog.at_level('WARNING'):
            decoded = DecodedReferents.from_wire({
                'source': 'derived',
                'refs': [{'kind': 'task', 'number': '3127'}],
                'ambiguous': None,
            })

        assert decoded == NO_REFERENTS
        assert caplog.records

    def test_a_malformed_ambiguous_takes_the_good_refs_down_with_it(self):
        decoded = DecodedReferents.from_wire({
            'source': 'derived',
            'refs': [{'kind': 'task', 'number': '3127'}],
            'ambiguous': [{'kind': 'unregistered_kind', 'number': '6379'}],
        })

        assert decoded == NO_REFERENTS

    def test_the_one_salvageable_referent_does_not_escape(self):
        decoded = DecodedReferents.from_wire({
            'source': 'derived',
            'refs': [
                {'kind': 'task', 'number': '3127'},
                {'kind': 'unregistered_kind', 'number': '9'},
            ],
        })

        assert decoded == NO_REFERENTS
        assert Referent(number='3127') not in decoded.refs

    @pytest.mark.parametrize('blob', _UNREADABLE_BLOBS)
    def test_no_decoded_referent_is_ever_unhashable(self, blob):
        assert set(DecodedReferents.from_wire(blob).refs) == set()

    def test_a_mistyped_number_is_not_silently_coerced(self):
        decoded = DecodedReferents.from_wire({
            'source': 'derived', 'refs': [{'kind': 'task', 'number': 3127}],
        })

        assert decoded == NO_REFERENTS


ALLOWED_FIRST_PARTY = frozenset({
    'fused_memory.utils.referent_resolution',
    'fused_memory.utils.canonical_labels',
    'fused_memory.utils.validation',
})


def test_queued_writes_imports_only_the_stdlib_and_three_utils_modules():
    imported = imported_modules(parse_python_module(QUEUED_WRITES))
    assert imported, 'no import scanned; this guard has lost its subject.'
    disallowed = sorted(
        module for module in imported if not stdlib_or_allowed(module, ALLOWED_FIRST_PARTY)
    )
    assert not disallowed, (
        f'queued_writes.py imports {disallowed}; the leaf may import only the '
        f'stdlib and {sorted(ALLOWED_FIRST_PARTY)}.'
    )
