"""The resolved referent set must reach `_execute_graphiti_write` (task 3670,
PRD leaf epsilon of plans/memory-referent-fidelity-prd.md).

Leaf gamma resolves WHICH referents a write is about; leaf zeta verifies the
episode's graph edges against that set and leaf eta repairs what it finds
misattached.  Between them sits a durable SQLite queue: the write boundary that
knows the content and metadata is not the code that talks to Graphiti.  So the
resolved set rides the identical channel `temporal_context` and
`unverified_claim` already ride (task 3142) — one additional key on the
existing `add_episode` / `add_memory_graphiti` payloads, popped by the executor.

No `payload_version` and no migration: an old consumer ignores one unknown key,
and a new consumer reading an old row finds it absent and treats it as "no
referents" — today's behaviour exactly (PRD "Queue compatibility is free here").

The decode is ALL-OR-NOTHING and the 'none' bucket is COUNTED, because the
regression this guards against (INV-4) is "the plumbing breaks, every row
arrives referent-less, and the feature no-ops in total silence".
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.services.memory_service import MemoryService
from fused_memory.utils.canonical_labels import Referent
from fused_memory.utils.referent_resolution import ReferentResolution


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends and durable queue.

    Copied from tests/test_unverified_claim_tag_propagation.py, the structural
    precedent for this whole channel.  `install_identity_mocks` is REQUIRED,
    not decorative: `_execute_graphiti_write` wraps its critical section in
    `async with self.graphiti._identity_lock_for(...)`, which a bare MagicMock
    cannot satisfy.
    """
    from _fm_helpers import install_identity_mocks

    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[])
    svc.durable_queue.close = AsyncMock()
    return svc


def _graphiti_payload(**overrides):
    payload = {
        'uuid': 'test-uuid',
        'name': 'episode_test',
        'content': 'test content',
        'source': 'text',
        'group_id': 'test',
        'source_description': 'notes',
    }
    payload.update(overrides)
    return payload


class TestReferentWireCodec:
    """The happy-path round trip: resolution -> JSON-safe blob -> resolution."""

    def test_encode_emits_plain_json_scalars(self):
        from fused_memory.services.memory_service import _encode_referents

        blob = _encode_referents(ReferentResolution(
            source='derived',
            referents=(
                Referent(number='3127'),
                Referent(number='2500', project_id='reify'),
            ),
        ))

        assert blob == {
            'source': 'derived',
            'refs': [
                {'kind': 'task', 'project_id': '', 'number': '3127'},
                {'kind': 'task', 'project_id': 'reify', 'number': '2500'},
            ],
        }

    def test_encoded_blob_survives_a_json_round_trip(self):
        """The durable queue persists payloads as JSON TEXT in SQLite, so a
        non-JSON-safe value here would only ever fail in production."""
        from fused_memory.services.memory_service import _encode_referents

        blob = _encode_referents(ReferentResolution(
            source='metadata', referents=(Referent(number='3129'),),
        ))

        assert json.loads(json.dumps(blob)) == blob

    def test_empty_resolution_encodes_to_an_explicit_empty_set(self):
        from fused_memory.services.memory_service import _encode_referents

        assert _encode_referents(ReferentResolution(source='none')) == {
            'source': 'none', 'refs': [],
        }

    def test_decode_rebuilds_the_exact_referent_tuple(self):
        from fused_memory.services.memory_service import (
            _decode_referents,
            _encode_referents,
        )

        resolution = ReferentResolution(
            source='derived',
            referents=(
                Referent(number='3127'),
                Referent(number='2500', project_id='reify'),
            ),
        )
        payload = {'referents': _encode_referents(resolution)}

        assert _decode_referents(payload) == (
            (Referent(number='3127'), Referent(number='2500', project_id='reify')),
            'derived',
        )

    def test_digits_survive_verbatim(self):
        """'0132' is a DIFFERENT referent from '132' (Referent's own contract);
        a wire codec that int-normalized would silently retarget the node."""
        from fused_memory.services.memory_service import (
            _decode_referents,
            _encode_referents,
        )

        blob = _encode_referents(ReferentResolution(
            source='declared', referents=(Referent(number='0132'),),
        ))
        assert blob['refs'][0]['number'] == '0132'

        referents, source = _decode_referents({'referents': blob})
        assert referents == (Referent(number='0132'),)
        assert referents[0].number == '0132'
        assert source == 'declared'

    def test_decode_pops_the_key(self):
        """Matches how `_execute_graphiti_write` already treats
        temporal_context / unverified_claim / reference_time."""
        from fused_memory.services.memory_service import (
            _decode_referents,
            _encode_referents,
        )

        payload = {'referents': _encode_referents(
            ReferentResolution(source='derived', referents=(Referent(number='7'),)),
        )}

        _decode_referents(payload)

        assert 'referents' not in payload


#: Every way the wire value can be unreadable.  Each must decode to exactly
#: ``((), 'none')`` — never a partial set.
_UNREADABLE_PAYLOADS = [
    pytest.param({}, id='key-absent'),
    pytest.param({'referents': None}, id='explicit-none'),
    pytest.param({'referents': 'garbage'}, id='non-dict'),
    pytest.param({'referents': ['derived']}, id='list-not-dict'),
    pytest.param(
        {'referents': {'refs': []}}, id='source-missing',
    ),
    pytest.param(
        {'referents': {'source': 'declared_v2', 'refs': []}},
        id='source-outside-vocabulary',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': 'not-a-list'}},
        id='refs-not-a-list',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': ['3127']}},
        id='entry-not-a-dict',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [{'kind': 'task'}]}},
        id='entry-lacks-number',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127'},
            {'kind': 'unregistered_kind', 'number': '9'},
        ]}},
        id='partially-malformed-set',
    ),
]


class TestReferentWireCodecDegradation:
    """ONE degradation story: every unreadable blob decodes to exactly
    ``((), 'none')``, never a partial set.

    A partial set is WORSE than no set for the consumer this exists to serve.
    Leaf zeta reads "endpoint not in the referent set" as a conflation and leaf
    eta repairs it by repointing the edge — so a referent silently dropped by a
    lenient decoder would manufacture a false conflation and drive destructive
    edge surgery onto the wrong node, the precise failure this PRD exists to
    prevent.  Hence the `partially-malformed-set` case: the one salvageable
    referent must NOT survive.

    Degrading rather than raising is equally deliberate: raising inside the
    queue executor would route the item to `_handle_failure` and eventually
    dead-letter it, LOSING the memory.  Degradation is safe here only because
    the 'none' bucket makes it loud (INV-4).
    """

    @pytest.mark.parametrize('payload', _UNREADABLE_PAYLOADS)
    def test_unreadable_blob_degrades_to_the_empty_set(self, payload):
        from fused_memory.services.memory_service import _decode_referents

        assert _decode_referents(dict(payload)) == ((), 'none')

    @pytest.mark.parametrize('payload', _UNREADABLE_PAYLOADS)
    def test_key_is_popped_in_every_case(self, payload):
        from fused_memory.services.memory_service import _decode_referents

        mutable = dict(payload)

        _decode_referents(mutable)

        assert 'referents' not in mutable

    @pytest.mark.parametrize(
        'payload',
        [p for p in _UNREADABLE_PAYLOADS if p.id not in ('key-absent', 'explicit-none')],
    )
    def test_malformed_blob_warns(self, payload, caplog):
        """Mirrors the invalid-`reference_time` WARNING-and-degrade arm already
        in `_execute_graphiti_write`: degradation is loud, not silent."""
        from fused_memory.services.memory_service import _decode_referents

        with caplog.at_level('WARNING'):
            _decode_referents(dict(payload))

        assert any(
            rec.levelname == 'WARNING' and 'referents' in rec.getMessage()
            for rec in caplog.records
        ), f'no WARNING naming the payload key; got {[r.getMessage() for r in caplog.records]}'

    def test_absent_key_does_not_warn(self, caplog):
        """The old-format row is the LOAD-BEARING back-compat case, not an
        anomaly — warning on it would drown the log during a drain of a queue
        written before this feature."""
        from fused_memory.services.memory_service import _decode_referents

        with caplog.at_level('WARNING'):
            _decode_referents({})

        assert not caplog.records

    def test_the_one_salvageable_referent_does_not_escape(self):
        """Named separately from the parametrized sweep because it is the whole
        reason the decode is all-or-nothing."""
        from fused_memory.services.memory_service import _decode_referents

        referents, source = _decode_referents({'referents': {
            'source': 'derived',
            'refs': [
                {'kind': 'task', 'number': '3127'},
                {'kind': 'unregistered_kind', 'number': '9'},
            ],
        }})

        assert referents == ()
        assert source == 'none'
        assert Referent(number='3127') not in referents


class TestAddMemoryStampsReferents:
    """`add_memory`'s Graphiti leg resolves and stamps the set at enqueue time.

    'decisions_and_rationale' is a GRAPHITI_PRIMARY category, so `write_graphiti`
    is True without needing `dual_write`.
    """

    @pytest.mark.asyncio
    async def test_derived_from_content_when_no_metadata_bridge(self, service):
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['referents'] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
        }

    @pytest.mark.asyncio
    async def test_metadata_task_id_outranks_the_derived_scan(self, service):
        """Same prose naming 3127, but ambient metadata says 3129. The bridge
        wins — and reading an INT proves the resolver sees `meta` AFTER
        `_normalize_task_id_metadata` has coerced it to a str."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
            metadata={'task_id': 3129},
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['referents'] == {
            'source': 'metadata',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
        }

    @pytest.mark.asyncio
    async def test_unresolvable_prose_still_stamps_an_explicit_empty_set(self, service):
        """The key is stamped even when empty, so a new-format row stays
        distinguishable from an old one at the wire level."""
        await service.add_memory(
            content='the merge-lane hardening task',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['referents'] == {'source': 'none', 'refs': []}

    @pytest.mark.asyncio
    async def test_mem0_only_write_never_enqueues_at_all(self, service):
        """No Graphiti leg means no resolution is paid for: the scan is scoped
        to the `write_graphiti` branch."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='observations_and_summaries',
            project_id='dark_factory',
            dual_write=False,
        )

        service.durable_queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_stamped_blob_is_json_safe(self, service):
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert json.loads(json.dumps(payload['referents'])) == payload['referents']

    @pytest.mark.asyncio
    async def test_preexisting_payload_keys_are_untouched(self, service):
        """The referent set is ADDITIVE, never a replacement."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['name'] == 'memory_decisions_and_rationale'
        assert payload['content'] == 'the fix for Task 3127 landed'
        assert payload['source'] == 'text'
        assert payload['source_description'] == 'add_memory:decisions_and_rationale'
        assert '_causation_id' in payload
        assert '_write_op_id' in payload
