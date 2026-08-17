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
from graphiti_core.nodes import EpisodeType

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


class TestAddEpisodeStampsReferents:
    """The second producer. `add_episode` deliberately never persists a
    metadata argument — the same fact that forced task 3142's
    `unverified_claim` onto this payload channel — so the derived scan is the
    only live source here."""

    @pytest.mark.asyncio
    async def test_derived_from_content(self, service):
        await service.add_episode(
            content='Task 3127 was merged', project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['referents'] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
        }

    @pytest.mark.asyncio
    async def test_unresolvable_content_stamps_an_explicit_empty_set(self, service):
        await service.add_episode(
            content='the merge-lane hardening work', project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['referents'] == {'source': 'none', 'refs': []}

    @pytest.mark.asyncio
    async def test_never_reaches_the_metadata_source(self, service):
        """Documents the asymmetry with add_memory: this producer has no
        metadata parameter to bridge from, so 'metadata' is structurally
        unreachable no matter what the prose says."""
        for content in ('Task 3127 was merged', 'an ordinary note'):
            service.durable_queue.enqueue.reset_mock()
            await service.add_episode(content=content, project_id='dark_factory')

            payload = service.durable_queue.enqueue.call_args[1]['payload']
            assert payload['referents']['source'] in ('derived', 'none')

    @pytest.mark.asyncio
    async def test_every_preexisting_payload_key_survives(self, service):
        """The referent set is ADDITIVE, never a replacement."""
        await service.add_episode(
            content='Task 3127 was merged',
            project_id='dark_factory',
            agent_id='a1',
            session_id='s1',
            source_description='notes',
            causation_id='c1',
            temporal_context='planning',
            unverified_claim=True,
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        for key in (
            'uuid', 'name', 'content', 'source', 'group_id', 'source_description',
            'project_id', 'agent_id', 'session_id', '_causation_id', '_write_op_id',
            'temporal_context', 'unverified_claim', 'reference_time',
        ):
            assert key in payload, f'{key} was dropped from the payload'
        assert payload['content'] == 'Task 3127 was merged'
        assert payload['source'] == 'text'
        assert payload['source_description'] == 'notes'
        assert payload['project_id'] == 'dark_factory'
        assert payload['agent_id'] == 'a1'
        assert payload['session_id'] == 's1'
        assert payload['_causation_id'] == 'c1'
        assert payload['temporal_context'] == 'planning'
        assert payload['unverified_claim'] is True
        assert payload['reference_time'] is None

    @pytest.mark.asyncio
    async def test_the_stamped_blob_is_json_safe(self, service):
        await service.add_episode(
            content='Task 3127 was merged', project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert json.loads(json.dumps(payload['referents'])) == payload['referents']


def _encoded(source, *referents):
    from fused_memory.services.memory_service import _encode_referents

    return _encode_referents(
        ReferentResolution(source=source, referents=tuple(referents)),
    )


#: The exact kwargs `_execute_graphiti_write` hands the backend today, for
#: `_graphiti_payload()`. Epsilon must not change ANY of them.
_TODAYS_BACKEND_KWARGS = {
    'name': 'episode_test',
    'content': 'test content',
    'group_id': 'test',
    'source_description': 'notes',
    'uuid': 'test-uuid',
    'temporal_context': None,
    'reference_time': None,
    'unverified_claim': False,
}


class TestExecuteGraphitiWriteConsumesReferents:
    """The executor decodes the blob and stops there — epsilon's job ends at
    making the set a live local. The backend signature is untouched."""

    @pytest.mark.asyncio
    async def test_old_format_row_executes_byte_identically(self, service):
        """The load-bearing back-compat signal: a row written before task 3670
        reaches Graphiti with exactly today's kwargs."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        assert service.graphiti.add_episode.call_count == 1
        assert service.graphiti.add_episode.call_args[1] == {
            **_TODAYS_BACKEND_KWARGS, 'source': EpisodeType.text,
        }
        service._reconcile_episode_identity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_backend_is_never_handed_a_referents_kwarg(self, service):
        """Epsilon stops at the executor. Handing the set to the backend is
        leaf zeta's business, not this leaf's."""
        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert 'referents' not in service.graphiti.add_episode.call_args[1]

    @pytest.mark.asyncio
    async def test_new_format_row_executes_with_the_same_backend_kwargs(self, service):
        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert service.graphiti.add_episode.call_args[1] == {
            **_TODAYS_BACKEND_KWARGS, 'source': EpisodeType.text,
        }

    @pytest.mark.asyncio
    async def test_the_key_is_popped_off_the_payload(self, service):
        """The observable proof the executor CONSUMED the blob rather than
        passing it through to the backend."""
        payload = _graphiti_payload(
            referents=_encoded('derived', Referent(number='3127')),
        )

        await service._execute_graphiti_write('add_episode', payload)

        assert 'referents' not in payload

    @pytest.mark.asyncio
    async def test_a_malformed_blob_still_completes_the_write(self, service):
        """Never dead-letters: a telemetry field must not cost a memory."""
        await service._execute_graphiti_write(
            'add_episode', _graphiti_payload(referents='garbage'),
        )

        assert service.graphiti.add_episode.call_count == 1


class TestReferentSourceCounter:
    """The INV-4 storm-escape gate.

    The regression INV-4 guards against is "the plumbing breaks, every row
    arrives referent-less, and the feature no-ops in total silence".  That
    failure lives on the PRODUCER side, so a counter emitted at the producer
    would go dark in exactly the scenario it exists to detect.  Only the
    CONSUMER sees both new-format and old-format rows, so only the consumer can
    report the rate — and it buckets ALL FOUR sources, because "sustained 100%
    none" is a ratio and an absolute none-count cannot distinguish a broken
    producer from a quiet system.
    """

    def test_a_fresh_service_exposes_every_source_bucket_at_zero(self, service):
        """Expected buckets are built from REFERENT_SOURCES itself, not from
        four re-spelled literals, so a fifth source added to gamma's Literal
        cannot silently escape the counter."""
        from fused_memory.utils.referent_resolution import REFERENT_SOURCES

        assert service.referent_source_counts() == dict.fromkeys(REFERENT_SOURCES, 0)

    @pytest.mark.asyncio
    async def test_an_old_format_row_increments_only_none(self, service):
        """The explicit INV-4 requirement: the absent path is COUNTED, never a
        silent fallthrough."""
        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        counts = service.referent_source_counts()
        assert counts['none'] == 1
        assert counts['derived'] == 0
        assert counts['metadata'] == 0
        assert counts['declared'] == 0

    @pytest.mark.asyncio
    async def test_a_malformed_blob_lands_in_none(self, service):
        """Degrading is only safe because the anomaly is loud somewhere."""
        await service._execute_graphiti_write(
            'add_episode', _graphiti_payload(referents={'source': 'derived', 'refs': 3}),
        )

        assert service.referent_source_counts()['none'] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('source', ['declared', 'metadata', 'derived'])
    async def test_each_resolved_source_increments_its_own_bucket(self, service, source):
        """Bucketing every source is what gives leaf iota a DENOMINATOR, so it
        can compute a rate rather than only see an absolute none-count."""
        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded(source, Referent(number='3127'))),
        )

        counts = service.referent_source_counts()
        assert counts[source] == 1
        assert counts['none'] == 0

    @pytest.mark.asyncio
    async def test_the_accessor_returns_a_copy(self, service):
        """A caller mutating the returned dict must not corrupt the escape
        hatch's own state."""
        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        snapshot = service.referent_source_counts()
        snapshot['none'] = 9999

        assert service.referent_source_counts()['none'] == 1

        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        assert service.referent_source_counts()['none'] == 2

    @pytest.mark.asyncio
    async def test_the_counter_increments_with_no_write_journal(self, service):
        """The escape hatch is UNCONDITIONAL in-process state, so it can never
        itself be silently absent — unlike the journal channel, which is None
        in exactly the degraded configurations where an unnoticed regression is
        least likely to be caught any other way."""
        service._write_journal = None

        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )
        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        counts = service.referent_source_counts()
        assert counts['derived'] == 1
        assert counts['none'] == 1
