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
import pytest_asyncio
from _fm_helpers import poll_until
from graphiti_core.nodes import EpisodeType

from fused_memory.services.durable_queue import DurableWriteQueue
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

    def test_ambiguity_is_deliberately_not_threaded(self):
        """`.ambiguous` and `.conflicts` are DROPPED on the wire — a decision,
        not an oversight, so it gets a named test rather than only a docstring.

        Gamma excludes ambiguous referents from `.referents` on purpose
        ("recorded, not guessed"), so a consumer reading only `refs` sees an
        ambiguous endpoint as a plain non-member — indistinguishable from a
        genuine conflation. Leaf zeta must NOT treat non-membership alone as
        grounds for leaf eta to repoint the edge; it re-derives ambiguity from
        `payload['content']`/`payload['group_id']`, which reproduce the
        producer's set exactly because `.ambiguous` is `scan_content(...)`
        verbatim on every precedence path.

        When the follow-up that carries `'ambiguous'` as a third key lands, this
        test is the thing that fails — which is the point: it routes that author
        through `_encode_referents`' docstring instead of past it.
        """
        from fused_memory.services.memory_service import _encode_referents

        blob = _encode_referents(ReferentResolution(
            source='derived',
            referents=(Referent(number='3127'),),
            ambiguous=(Referent(number='9'),),
            conflicts=(Referent(number='42'),),
        ))

        assert set(blob) == {'source', 'refs'}
        assert blob['refs'] == [{'kind': 'task', 'project_id': '', 'number': '3127'}]

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
    # `Referent.__post_init__` validates `kind` ONLY — `number`/`project_id`
    # take any object at all, so the constructor alone does not harden them and
    # each of these would otherwise mint a bogus referent rather than degrade.
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': 3127},
        ]}},
        id='int-number-compares-unequal-to-its-str-twin',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': None},
        ]}},
        id='none-number-would-name-the-node-Task-None',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': ['3127']},
        ]}},
        id='unhashable-number-would-raise-inside-a-consumer-set',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127', 'project_id': None},
        ]}},
        id='none-project-id',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127', 'project_id': 7},
        ]}},
        id='int-project-id',
    ),
    pytest.param(
        {'referents': {'source': 'derived', 'refs': [
            {'kind': 'task', 'number': '3127'},
            {'kind': 'task', 'number': 9},
        ]}},
        id='partially-mistyped-set',
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

    @pytest.mark.parametrize('payload', _UNREADABLE_PAYLOADS)
    def test_no_decoded_referent_is_ever_unhashable(self, payload):
        """`Referent` validates `kind` only, so an unhashable wire `number`
        would mint a Referent that raises TypeError the moment leaf zeta puts
        it in a set — a raise INSIDE the queue executor, i.e. the
        dead-letter-and-lose-the-memory outcome the decoder's
        degrade-rather-than-raise design exists to prevent."""
        from fused_memory.services.memory_service import _decode_referents

        referents, _ = _decode_referents(dict(payload))

        assert set(referents) == set()

    def test_a_mistyped_number_is_not_silently_coerced(self):
        """An int `number` must NOT decode to `Referent(number='3127')` either:
        coercing would hide a producer writing the wrong type, and the value it
        would produce is indistinguishable from a correctly-encoded row."""
        from fused_memory.services.memory_service import _decode_referents

        referents, source = _decode_referents({'referents': {
            'source': 'derived', 'refs': [{'kind': 'task', 'number': 3127}],
        }})

        assert referents == ()
        assert source == 'none'


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
        unreachable no matter what the prose says.

        Each content is paired with the ONE source it must reach, not asserted
        against the set of both remaining values: 'declared' is structurally
        impossible here (`declared=None` is hardcoded) and 'metadata' is what
        this test rules out, so a membership assertion would admit both
        surviving values for both inputs and would still pass if the derived
        scan regressed and 'Task 3127 was merged' started resolving to 'none'.
        """
        for content, expected_source in (
            ('Task 3127 was merged', 'derived'),
            ('an ordinary note', 'none'),
        ):
            service.durable_queue.enqueue.reset_mock()
            await service.add_episode(content=content, project_id='dark_factory')

            payload = service.durable_queue.enqueue.call_args[1]['payload']
            assert payload['referents']['source'] == expected_source, (
                f'{content!r} resolved to {payload["referents"]["source"]!r}, '
                f'expected {expected_source!r}'
            )

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

    @pytest.mark.asyncio
    async def test_a_failed_write_is_still_counted(self, service):
        """The buckets count write ATTEMPTS, not completed writes, and that is
        deliberate: `DurableWriteQueue._process_item` re-invokes this executor
        on every RETRY with a freshly parsed payload, so retries land in the
        buckets and an item that eventually dead-letters is still counted.

        Counting only successes would instead make the escape go dark during a
        backend outage — exactly when a referent-less write storm is least
        likely to be noticed any other way. Pinned here so nobody 'fixes' the
        attempt-vs-write skew by moving the increment past the backend call
        without re-reading `referent_source_counts`' docstring.
        """
        service.graphiti.add_episode = AsyncMock(side_effect=RuntimeError('backend down'))
        payload = _graphiti_payload(
            referents=_encoded('derived', Referent(number='3127')),
        )

        for _ in range(3):  # the same item, retried
            with pytest.raises(RuntimeError):
                await service._execute_graphiti_write('add_episode', dict(payload))

        assert service.referent_source_counts()['derived'] == 3


@pytest.fixture
def journaling_service(service):
    """`service`, plus a write journal whose backend-op rows are observable."""
    service._write_journal = MagicMock()
    service._write_journal.log_backend_op = AsyncMock()
    return service


def _backend_op_payload(svc):
    return svc._write_journal.log_backend_op.call_args[1]['payload']


class TestReferentSourceIsJournaled:
    """The DURABLE half of the telemetry split.

    The in-process counter is unconditional but process-local; this row is
    per-group_id and time-windowed, which is what leaf iota actually needs to
    read.  Both sinks consume the SAME values from the ONE decode, so they
    cannot drift — this is not lockstep duplication.
    """

    @pytest.mark.asyncio
    async def test_a_new_format_row_journals_its_source_and_count(self, journaling_service):
        await journaling_service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded(
                'derived', Referent(number='3127'), Referent(number='2500'),
            )),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'derived'
        assert payload['referent_count'] == 2

    @pytest.mark.asyncio
    async def test_an_old_format_row_journals_none_and_zero(self, journaling_service):
        await journaling_service._execute_graphiti_write(
            'add_episode', _graphiti_payload(),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'none'
        assert payload['referent_count'] == 0

    @pytest.mark.asyncio
    async def test_a_malformed_blob_journals_none_and_zero(self, journaling_service):
        """The durable channel and the in-process counter agree because both
        read the same decode — a malformed blob is 'none' in both."""
        await journaling_service._execute_graphiti_write(
            'add_episode', _graphiti_payload(referents='garbage'),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'none'
        assert payload['referent_count'] == 0
        assert journaling_service.referent_source_counts()['none'] == 1

    @pytest.mark.asyncio
    async def test_the_existing_journal_keys_are_unchanged(self, journaling_service):
        """Additive: content stays truncated to 200 and group_id stays put."""
        await journaling_service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(
                content='x' * 500,
                group_id='dark_factory',
                referents=_encoded('metadata', Referent(number='3129')),
            ),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['content'] == 'x' * 200
        assert payload['group_id'] == 'dark_factory'
        assert payload['referent_source'] == 'metadata'
        assert payload['referent_count'] == 1


class TestReplayFromStoreStampsReferents:
    """The third and last producer of `add_memory_graphiti` rows.

    Replayed rows carry real prose whose referents the derived scanner can see.
    Leaving them on the absent path would stamp them 'none' and inflate leaf
    iota's undeclared bucket with writes that were plainly derivable — a false
    regression signal in the very counter this task exists to make trustworthy.
    """

    @staticmethod
    def _stub_memories(service, memories):
        service.mem0.get_all = AsyncMock(return_value={'results': memories})

    @pytest.mark.asyncio
    async def test_every_replayed_row_carries_a_referent_blob(self, service):
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'Task 2500 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'the merge-lane hardening work', 'metadata': {}},
        ])

        count = await service.replay_from_store('dark_factory')

        assert count == 3
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 3
        assert all('referents' in item['payload'] for item in batch)

    @pytest.mark.asyncio
    async def test_task_naming_rows_resolve_derived(self, service):
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'Task 2500 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'the merge-lane hardening work', 'metadata': {}},
        ])

        await service.replay_from_store('dark_factory')

        blobs = [
            item['payload']['referents']
            for item in service.durable_queue.enqueue_batch.call_args[0][0]
        ]
        assert blobs[0] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
        }
        assert blobs[1] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '2500'}],
        }
        assert blobs[2] == {'source': 'none', 'refs': []}

    @pytest.mark.asyncio
    async def test_the_metadata_bridge_is_live_on_this_producer(self, service):
        """Unlike add_episode, the replay loop DOES hold a metadata dict — the
        Mem0 record's own — so the bridge can outrank the derived scan here."""
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed',
             'metadata': {'category': 'temporal_facts', 'task_id': '3129'}},
        ])

        await service.replay_from_store('dark_factory')

        blob = service.durable_queue.enqueue_batch.call_args[0][0][0]['payload']['referents']
        assert blob == {
            'source': 'metadata',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
        }

    @pytest.mark.asyncio
    async def test_a_non_str_memory_is_skipped_not_fatal(self, service, caplog):
        """`resolve_referents` raises InputValidationError on a truthy non-str
        content, and this call sits in a loop whose `enqueue_batch` runs only
        AFTER the loop completes — so an unguarded record would abort the WHOLE
        replay and enqueue nothing. The blast radius must stay at one row."""
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': {'not': 'a string'}, 'metadata': {}},
            {'memory': ['also not a string'], 'metadata': {}},
            {'memory': 'Task 2500 landed', 'metadata': {'category': 'temporal_facts'}},
        ])

        with caplog.at_level('WARNING'):
            count = await service.replay_from_store('dark_factory')

        assert count == 2
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert [item['payload']['content'] for item in batch] == [
            'Task 3127 landed', 'Task 2500 landed',
        ]
        assert all(item['payload']['referents']['source'] == 'derived' for item in batch)
        assert sum(
            'not a string' in rec.getMessage() for rec in caplog.records
        ) == 2, f'expected one WARNING per skipped record; got {[r.getMessage() for r in caplog.records]}'

    @pytest.mark.asyncio
    async def test_preexisting_batch_payload_keys_survive(self, service):
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
        ])

        await service.replay_from_store('dark_factory')

        item = service.durable_queue.enqueue_batch.call_args[0][0][0]
        assert item['group_id'] == 'dark_factory'
        assert item['operation'] == 'add_memory_graphiti'
        assert item['callback_type'] == 'refresh_entity_summaries'
        assert item['payload']['name'] == 'replay_temporal_facts'
        assert item['payload']['content'] == 'Task 3127 landed'
        assert item['payload']['source'] == 'text'
        assert item['payload']['group_id'] == 'dark_factory'
        assert item['payload']['source_description'] == 'replay_from_mem0:temporal_facts'
        assert json.loads(json.dumps(item['payload'])) == item['payload']


class TestReferentsSurviveTheRealQueue:
    """End-to-end over a REAL DurableWriteQueue on a tmp_path SQLite file.

    The mocked-`durable_queue` fixtures above structurally cannot exercise the
    `json.dumps` -> SQLite TEXT -> `json.loads` round trip the blob actually
    makes in production.  This is the only level that can.
    """

    @pytest_asyncio.fixture
    async def real_queue(self, tmp_path, service):
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=service._execute_durable_write,
            workers_per_group=1,
            semaphore_limit=2,
            max_attempts=1,
            retry_base_seconds=0.05,
            write_timeout_seconds=5.0,
        )
        await q.initialize()
        service.durable_queue = q
        yield q
        await q.close()

    @pytest.mark.asyncio
    async def test_a_new_format_row_survives_the_sqlite_round_trip(
        self, real_queue, service,
    ):
        await real_queue.enqueue(
            group_id='dark_factory',
            operation='add_episode',
            payload=_graphiti_payload(
                group_id='dark_factory',
                referents=_encoded(
                    'derived', Referent(number='3127'), Referent(number='2500'),
                ),
            ),
        )

        await poll_until(
            lambda: service.referent_source_counts()['derived'] == 1,
            message='the derived bucket never incremented',
        )
        assert service.graphiti.add_episode.call_count == 1
        stats = await real_queue.get_stats()
        assert stats['counts'].get('completed') == 1

    @pytest.mark.asyncio
    async def test_an_old_format_row_still_executes_end_to_end(
        self, real_queue, service,
    ):
        """The load-bearing back-compat signal, all the way through: a row
        written before task 3670 drains to 'completed' and counts only 'none'."""
        await real_queue.enqueue(
            group_id='dark_factory',
            operation='add_episode',
            payload=_graphiti_payload(group_id='dark_factory'),
        )

        await poll_until(
            lambda: service.referent_source_counts()['none'] == 1,
            message='the none bucket never incremented',
        )
        counts = service.referent_source_counts()
        assert counts['derived'] == 0
        stats = await real_queue.get_stats()
        assert stats['counts'].get('completed') == 1

    @pytest.mark.asyncio
    async def test_the_callback_still_sees_the_key_the_executor_popped(
        self, real_queue, service,
    ):
        """Pins the durable_queue._process_item contract that makes popping
        safe: the executor gets one parsed_payload() and the callback a SECOND,
        FRESH one.  A future queue refactor that reused one dict would silently
        starve _dual_write_callback, and only this test would notice."""
        seen: list[dict] = []

        async def _spy(callback_type, result, payload):
            seen.append(payload)

        real_queue.register_callback('dual_write_episode', _spy)
        blob = _encoded('derived', Referent(number='3127'))

        await real_queue.enqueue(
            group_id='dark_factory',
            operation='add_episode',
            payload=_graphiti_payload(group_id='dark_factory', referents=blob),
            callback_type='dual_write_episode',
        )

        await poll_until(lambda: seen, message='the callback never fired')
        assert seen[0]['referents'] == blob


class TestExecuteGraphitiWriteHandsReferentsToZeta:
    """The executor hands the DECODED set to the verification sub-pass (task
    3671, PRD leaf zeta), inside the identity lock — at the call site epsilon's
    own comment reserved for it."""

    @pytest.mark.asyncio
    async def test_a_new_format_row_reaches_the_reconcile_with_the_decoded_set(
        self, service,
    ):
        """The decoded ReferentSet, not the wire blob."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert service._reconcile_episode_identity.call_args[1]['referents'] == (
            Referent(number='3127'),
        )

    @pytest.mark.asyncio
    async def test_an_old_format_row_reaches_it_with_an_empty_set(self, service):
        """The load-bearing back-compat path, still executing unchanged."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        assert service._reconcile_episode_identity.call_args[1]['referents'] == ()

    @pytest.mark.asyncio
    async def test_a_degraded_blob_reaches_it_as_an_empty_set(self, service):
        """`_decode_referents` is all-or-nothing precisely because a PARTIAL set
        would read as a conflation to zeta and drive eta's edge surgery onto the
        wrong node. An empty set no-ops the pass instead."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents={'source': 'derived', 'refs': 'not-a-list'}),
        )

        assert service._reconcile_episode_identity.call_args[1]['referents'] == ()

    @pytest.mark.asyncio
    async def test_the_backend_kwargs_are_still_untouched(self, service):
        """zeta reads the set AFTER the write; the backend never sees it."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert service.graphiti.add_episode.call_args[1] == {
            **_TODAYS_BACKEND_KWARGS, 'source': EpisodeType.text,
        }

    @pytest.mark.asyncio
    async def test_the_verification_runs_while_the_identity_lock_is_held(self, service):
        """No wrongly-attached state may be externally visible between the write
        and its verification, so zeta must run INSIDE the critical section —
        mirrors the TestWriteTimeIdentityGate probe in test_memory_service.py."""
        lock = service.graphiti._identity_lock_for('test')
        observed = {}

        async def _probe(result, *, group_id, referents):
            observed['locked'] = lock.locked()
            observed['same_lock'] = service.graphiti._identity_lock_for(group_id) is lock
            observed['referents'] = referents
            return {}

        service._reconcile_episode_identity = AsyncMock(side_effect=_probe)

        await service._execute_graphiti_write(
            'add_episode',
            _graphiti_payload(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert observed['locked'] is True
        assert observed['same_lock'] is True
        assert observed['referents'] == (Referent(number='3127'),)
        assert lock.locked() is False, 'the lock must be released on return'
