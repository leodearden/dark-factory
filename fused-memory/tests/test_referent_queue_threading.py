"""The resolved referent set must reach `_execute_graphiti_write` (task 3670,
PRD leaf epsilon of plans/memory-referent-fidelity-prd.md).

Leaf gamma resolves WHICH referents a write is about; leaf zeta verifies the
episode's graph edges against that set and leaf eta repairs what it finds
misattached.  Between them sits a durable SQLite queue: the write boundary that
knows the content and metadata is not the code that talks to Graphiti.  So the
resolved set rides the identical channel `temporal_context` and
`unverified_claim` already ride (task 3142) — one additional key on the
existing `add_episode` / `add_memory_graphiti` values, read by the executor.

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
from _fm_helpers import MockNode
from _fm_polling import poll_until
from _queued_write_fixtures import enqueued, queued
from graphiti_core.nodes import EpisodeType

from fused_memory.services.durable_queue import DurableWriteQueue
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.queued_writes import (
    DecodedReferents,
    QueueCallback,
    QueuedEpisodeWrite,
    QueuedGraphitiMemoryWrite,
    QueuedWrite,
)
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


def _graphiti_value(**overrides):
    return queued('add_episode', _graphiti_payload(**overrides))


def _stamped(svc):
    """The Graphiti-bound value the most recent producer call enqueued."""
    return _graphiti_bound(enqueued(svc.durable_queue.enqueue)[-1][0])


def _graphiti_bound(value: QueuedWrite) -> QueuedEpisodeWrite | QueuedGraphitiMemoryWrite:
    assert isinstance(value, QueuedEpisodeWrite | QueuedGraphitiMemoryWrite)
    return value


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

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
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

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'metadata',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
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

        value = _stamped(service)
        assert value.referents.to_wire() == {'source': 'none', 'refs': [], 'ambiguous': []}

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

        value = _stamped(service)
        assert json.loads(json.dumps(value.referents.to_wire())) == value.referents.to_wire()

    @pytest.mark.asyncio
    async def test_preexisting_payload_keys_are_untouched(self, service):
        """The referent set is ADDITIVE, never a replacement."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        value = _stamped(service)
        wire = json.loads(value.to_json())
        assert wire['name'] == 'memory_decisions_and_rationale'
        assert wire['content'] == 'the fix for Task 3127 landed'
        assert wire['source'] == 'text'
        assert wire['source_description'] == 'add_memory:decisions_and_rationale'
        assert '_causation_id' in wire
        assert '_write_op_id' in wire
        assert value.envelope.write_op_id is not None


class TestAddMemoryStampsDeclaredReferents:
    """The tier ABOVE the metadata bridge, reachable for the first time.

    Leaf delta's `entities` parameter fills the `declared=None` seam epsilon
    left annotated at this producer. What lands on the wire is
    `{'source': 'declared', ...}` — the bucket leaf iota counts and leaf zeta
    verifies edges against.

    The service is MECHANISM, not policy, exactly as gamma split reporting from
    rejecting: a declaration that contradicts the prose is LEGAL here and
    stamps normally. The tool-boundary `entities_gate` is what refuses it, and
    tests/server/test_entities_gate_ingestion.py is where that is pinned.
    """

    @pytest.mark.asyncio
    async def test_declared_outranks_the_derived_scan(self, service):
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_declared_outranks_the_metadata_bridge(self, service):
        """The precedence tier that could not be tested before this leaf, and
        the one that proves declared > metadata > derived > none is live end to
        end: prose says 3127, ambient metadata says 3129, the declaration says
        3127 and wins."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
            metadata={'task_id': 3129},
            declared_referents=[{'kind': 'task', 'id': 3127}],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_the_empty_declaration_is_distinct_from_none(self, service):
        """`{'source': 'declared', 'refs': [], 'ambiguous': []}` vs `{'source': 'none', ...}` IS
        the "considered referents and none applied" vs "never looked" signal
        leaf iota counts. Collapsing [] onto None anywhere on this path would
        erase it silently."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
            declared_referents=[],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {'source': 'declared', 'refs': [], 'ambiguous': []}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('omit', [True, False], ids=['omitted', 'explicit-None'])
    async def test_epsilons_behaviour_is_byte_identical_without_a_declaration(
        self, service, omit,
    ):
        """The regression guard for epsilon: adding a tier must not disturb the
        three below it."""
        kwargs = {} if omit else {'declared_referents': None}
        cases = [
            ({'metadata': {'task_id': 3129}}, 'metadata', ['3129']),
            ({}, 'derived', ['3127']),
        ]
        for extra, source, numbers in cases:
            service.durable_queue.enqueue.reset_mock()
            await service.add_memory(
                content='the fix for Task 3127 landed',
                category='decisions_and_rationale',
                project_id='dark_factory',
                **extra,
                **kwargs,
            )
            value = _stamped(service)
            assert value.referents.to_wire() == {
                'source': source,
                'refs': [
                    {'kind': 'task', 'project_id': '', 'number': n} for n in numbers
                ],
                'ambiguous': [],
            }, f'{extra!r}: {value.referents.to_wire()!r}'

        service.durable_queue.enqueue.reset_mock()
        await service.add_memory(
            content='the merge-lane hardening task',
            category='decisions_and_rationale',
            project_id='dark_factory',
            **kwargs,
        )
        value = _stamped(service)
        assert value.referents.to_wire() == {'source': 'none', 'refs': [], 'ambiguous': []}

    @pytest.mark.asyncio
    async def test_declared_digits_are_verbatim(self, service):
        """'0132' is a DIFFERENT referent from '132'; int-normalizing would
        silently repoint the caller's own assertion."""
        await service.add_memory(
            content='the merge-lane hardening task',
            category='decisions_and_rationale',
            project_id='dark_factory',
            declared_referents=[{'id': '0132'}],
        )

        value = _stamped(service)
        assert value.referents.to_wire()['refs'][0]['number'] == '0132'

    @pytest.mark.asyncio
    async def test_the_declared_blob_is_json_safe(self, service):
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='decisions_and_rationale',
            project_id='dark_factory',
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        assert json.loads(json.dumps(value.referents.to_wire())) == value.referents.to_wire()


class TestADeclarationIsScopedToTheGraphitiLeg:
    """A declaration is RECORDED only on a write that reaches Graphiti.

    `resolve_referents` is called inside `if write_graphiti:`, so on a
    MEM0_PRIMARY category without `dual_write` the caller's declaration is
    accepted and then discarded: no resolution, no encoded set, no counter
    increment.  That is deliberate — the referent set is a field on the
    Graphiti queue payload, and a Mem0-primary write produces no queue row to
    carry it and no edges for leaf zeta to verify it against — but it was
    stated nowhere and pinned by nothing, and every other test in this file
    uses a GRAPHITI_PRIMARY category, so the hole was invisible.  These rows
    make the scope a decision on the record.

    The consequence leaf IOTA must price in: its declaration-rate denominator
    is "every Graphiti write", NOT "every add_memory call".  Mem0-primary
    traffic is outside the counter entirely rather than counted as undeclared.

    The tool-boundary `entities_gate` reads the same routing rule
    (`routes_to_graphiti`): on an explicit Mem0-primary category without
    `dual_write` it shape-checks the declaration and never conflict-rejects it
    — pinned separately in tests/server/test_entities_gate_ingestion.py.
    Absence is never rejected, so an agent that omits `entities` cannot lose a
    write here either way.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'category',
        ['procedural_knowledge', 'preferences_and_norms', 'observations_and_summaries'],
    )
    async def test_a_declaration_on_a_mem0_primary_write_is_discarded(
        self, service, category
    ):
        before = service.referent_source_counts()

        await service.add_memory(
            content='the fix for Task 3127 landed',
            category=category,
            project_id='dark_factory',
            dual_write=False,
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        # No queue row, so nothing carries the declaration...
        service.durable_queue.enqueue.assert_not_called()
        # ...and the 'declared' bucket leaf iota reads never moves.
        assert service.referent_source_counts() == before

    @pytest.mark.asyncio
    async def test_the_declaration_is_indistinguishable_from_its_absence_there(
        self, service
    ):
        """The precise shape of the loss: on a Mem0-primary write, declaring
        and not declaring leave IDENTICAL observable state.  Nothing
        downstream can tell the two apart — which is why the docstrings must
        not promise that a declaration is honoured here."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='procedural_knowledge',
            project_id='dark_factory',
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )
        declared_calls = service.durable_queue.enqueue.call_args_list[:]
        declared_counts = service.referent_source_counts()

        service.durable_queue.enqueue.reset_mock()

        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='procedural_knowledge',
            project_id='dark_factory',
        )

        assert service.durable_queue.enqueue.call_args_list == declared_calls == []
        assert service.referent_source_counts() == declared_counts

    @pytest.mark.asyncio
    async def test_dual_write_is_the_documented_way_to_get_it_recorded(self, service):
        """`dual_write=True` puts a Mem0-primary category back on the Graphiti
        leg, and the declaration is stamped exactly as it is for a
        GRAPHITI_PRIMARY one.  This is the escape hatch the corrected
        docstrings name, so it is pinned rather than left as folklore."""
        await service.add_memory(
            content='the fix for Task 3127 landed',
            category='procedural_knowledge',
            project_id='dark_factory',
            dual_write=True,
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
        }


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

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_unresolvable_content_stamps_an_explicit_empty_set(self, service):
        await service.add_episode(
            content='the merge-lane hardening work', project_id='dark_factory',
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {'source': 'none', 'refs': [], 'ambiguous': []}

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

            value = _stamped(service)
            assert value.referents.to_wire()['source'] == expected_source, (
                f'{content!r} resolved to {value.referents.to_wire()["source"]!r}, '
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

        value = _stamped(service)
        wire = json.loads(value.to_json())
        # 'uuid' is deliberately absent: task 3561 removed it from the enqueue
        # payload (a fresh uuid means "load this node" to graphiti_core).
        for key in (
            'name', 'content', 'source', 'group_id', 'source_description',
            'project_id', 'agent_id', 'session_id', '_causation_id', '_write_op_id',
            'temporal_context', 'unverified_claim', 'reference_time',
        ):
            assert key in wire, f'{key} was dropped from the payload'
        assert wire['content'] == 'Task 3127 was merged'
        assert wire['source'] == 'text'
        assert wire['source_description'] == 'notes'
        assert wire['project_id'] == 'dark_factory'
        assert wire['agent_id'] == 'a1'
        assert wire['session_id'] == 's1'
        assert wire['_causation_id'] == 'c1'
        assert wire['temporal_context'] == 'planning'
        assert wire['unverified_claim'] is True
        assert wire['reference_time'] is None

    @pytest.mark.asyncio
    async def test_the_stamped_blob_is_json_safe(self, service):
        await service.add_episode(
            content='Task 3127 was merged', project_id='dark_factory',
        )

        value = _stamped(service)
        assert json.loads(json.dumps(value.referents.to_wire())) == value.referents.to_wire()


class TestAddEpisodeStampsDeclaredReferents:
    """Leaf delta's `entities` fills the second producer's `declared=None` seam.

    Same tier-1 declaration as `add_memory`'s, and it must behave identically —
    the gate stack does not diverge between the two write tools, so neither may
    the stamping. What DOES stay asymmetric is a tier below: this producer has
    no metadata to bridge from, so `declared > derived > none` is the whole
    ladder here and the metadata rung stays structurally unreachable.
    """

    @pytest.mark.asyncio
    async def test_declared_outranks_the_derived_scan(self, service):
        await service.add_episode(
            content='the fix for Task 3127 landed',
            project_id='dark_factory',
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_the_empty_declaration_is_distinct_from_none(self, service):
        """"considered referents and none applied" vs "never looked" — the
        distinction leaf iota counts, and the one a `not entities` shortcut
        anywhere on this path would erase in silence."""
        await service.add_episode(
            content='the fix for Task 3127 landed',
            project_id='dark_factory',
            declared_referents=[],
        )

        value = _stamped(service)
        assert value.referents.to_wire() == {'source': 'declared', 'refs': [], 'ambiguous': []}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('omit', [True, False], ids=['omitted', 'explicit-None'])
    async def test_epsilons_behaviour_is_byte_identical_without_a_declaration(
        self, service, omit,
    ):
        """The regression guard for epsilon at THIS producer."""
        kwargs = {} if omit else {'declared_referents': None}
        cases = [
            ('the fix for Task 3127 landed', 'derived', ['3127']),
            ('the merge-lane hardening work', 'none', []),
        ]
        for content, source, numbers in cases:
            service.durable_queue.enqueue.reset_mock()
            await service.add_episode(
                content=content, project_id='dark_factory', **kwargs,
            )
            value = _stamped(service)
            assert value.referents.to_wire() == {
                'source': source,
                'refs': [
                    {'kind': 'task', 'project_id': '', 'number': n} for n in numbers
                ],
                'ambiguous': [],
            }, f'{content!r}: {value.referents.to_wire()!r}'

    @pytest.mark.asyncio
    async def test_the_metadata_rung_stays_structurally_unreachable(self, service):
        """Adding the declared tier must not quietly wake the dead metadata rung.

        `MemoryService.add_episode` has no `metadata` parameter at all — the
        tool reads metadata for _causation_id/source routing and never forwards
        it — which is what makes epsilon's hardcoded `metadata=None` true by
        CONSTRUCTION rather than by convention. So the pin is the raise itself:
        there is no metadata rung to consult here because the call carrying one
        cannot be made.

        Asserted behaviourally (the TypeError) rather than by
        `inspect.signature`, per the house norm against parameter-name
        introspection meta-tests. Here the two are not equivalent anyway — the
        raise is the actual consequence a caller meets, and the signature is
        merely its cause.

        The regression this guards: `declared_referents` is the first new
        resolve-input to cross into this producer, and the plausible next
        change is adding `metadata` alongside it "for symmetry with
        add_memory". That would hand this producer a metadata rung whose value
        nothing persists, silently re-opening the asymmetry epsilon documented
        — and it would turn this test green-to-red at exactly the right moment.
        """
        with pytest.raises(TypeError, match='metadata'):
            await service.add_episode(
                content='the fix for Task 3127 landed',
                project_id='dark_factory',
                metadata={'task_id': 9999},
            )

        service.durable_queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_declaration_is_additive_to_every_preexisting_key(self, service):
        """A declaration adds a tier; it must not displace the payload."""
        await service.add_episode(
            content='the fix for Task 3127 landed',
            project_id='dark_factory',
            agent_id='a1',
            session_id='s1',
            source_description='notes',
            causation_id='c1',
            temporal_context='planning',
            unverified_claim=True,
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        wire = json.loads(value.to_json())
        # 'uuid' is deliberately absent: task 3561 removed it from the enqueue
        # payload (a fresh uuid means "load this node" to graphiti_core).
        for key in (
            'name', 'content', 'source', 'group_id', 'source_description',
            'project_id', 'agent_id', 'session_id', '_causation_id', '_write_op_id',
            'temporal_context', 'unverified_claim', 'reference_time',
        ):
            assert key in wire, f'{key} was dropped from the payload'
        assert wire['temporal_context'] == 'planning'
        assert wire['unverified_claim'] is True
        assert wire['reference_time'] is None
        assert value.referents.to_wire()['source'] == 'declared'

    @pytest.mark.asyncio
    async def test_declared_digits_are_verbatim(self, service):
        await service.add_episode(
            content='the merge-lane hardening work',
            project_id='dark_factory',
            declared_referents=[{'id': '0132'}],
        )

        value = _stamped(service)
        assert value.referents.to_wire()['refs'][0]['number'] == '0132'

    @pytest.mark.asyncio
    async def test_the_declared_blob_is_json_safe(self, service):
        await service.add_episode(
            content='the fix for Task 3127 landed',
            project_id='dark_factory',
            declared_referents=[{'kind': 'task', 'id': 3129}],
        )

        value = _stamped(service)
        assert json.loads(json.dumps(value.referents.to_wire())) == value.referents.to_wire()


def _encoded(source, *referents, ambiguous=()):
    return DecodedReferents.from_resolution(ReferentResolution(
        source=source, referents=tuple(referents), ambiguous=tuple(ambiguous),
    )).to_wire()


#: The exact kwargs `_execute_graphiti_write` hands the backend today, for
#: `_graphiti_payload()`. Epsilon must not change ANY of them.
#: `uuid` is None rather than absent: task 3561 made the executor pass
#: `uuid=None` unconditionally, so None is what the backend actually receives
#: and this dict stays an exact pin rather than a subset check.
_TODAYS_BACKEND_KWARGS = {
    'name': 'episode_test',
    'content': 'test content',
    'group_id': 'test',
    'source_description': 'notes',
    'uuid': None,
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

        await service._execute_graphiti_write(
            _graphiti_value())

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
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert 'referents' not in service.graphiti.add_episode.call_args[1]

    @pytest.mark.asyncio
    async def test_new_format_row_executes_with_the_same_backend_kwargs(self, service):
        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert service.graphiti.add_episode.call_args[1] == {
            **_TODAYS_BACKEND_KWARGS, 'source': EpisodeType.text,
        }

    @pytest.mark.asyncio
    async def test_a_malformed_blob_still_completes_the_write(self, service):
        """Never dead-letters: a telemetry field must not cost a memory."""
        await service._execute_graphiti_write(
            _graphiti_value(referents='garbage'),
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
        await service._execute_graphiti_write(
            _graphiti_value())

        counts = service.referent_source_counts()
        assert counts['none'] == 1
        assert counts['derived'] == 0
        assert counts['metadata'] == 0
        assert counts['declared'] == 0

    @pytest.mark.asyncio
    async def test_a_malformed_blob_lands_in_none(self, service):
        """Degrading is only safe because the anomaly is loud somewhere."""
        await service._execute_graphiti_write(
            _graphiti_value(referents={'source': 'derived', 'refs': 3}),
        )

        assert service.referent_source_counts()['none'] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('source', ['declared', 'metadata', 'derived'])
    async def test_each_resolved_source_increments_its_own_bucket(self, service, source):
        """Bucketing every source is what gives leaf iota a DENOMINATOR, so it
        can compute a rate rather than only see an absolute none-count."""
        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded(source, Referent(number='3127'))),
        )

        counts = service.referent_source_counts()
        assert counts[source] == 1
        assert counts['none'] == 0

    @pytest.mark.asyncio
    async def test_the_accessor_returns_a_copy(self, service):
        """A caller mutating the returned dict must not corrupt the escape
        hatch's own state."""
        await service._execute_graphiti_write(
            _graphiti_value())

        snapshot = service.referent_source_counts()
        snapshot['none'] = 9999

        assert service.referent_source_counts()['none'] == 1

        await service._execute_graphiti_write(
            _graphiti_value())

        assert service.referent_source_counts()['none'] == 2

    @pytest.mark.asyncio
    async def test_the_counter_increments_with_no_write_journal(self, service):
        """The escape hatch is UNCONDITIONAL in-process state, so it can never
        itself be silently absent — unlike the journal channel, which is None
        in exactly the degraded configurations where an unnoticed regression is
        least likely to be caught any other way."""
        service._write_journal = None

        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
        )
        await service._execute_graphiti_write(
            _graphiti_value())

        counts = service.referent_source_counts()
        assert counts['derived'] == 1
        assert counts['none'] == 1

    @pytest.mark.asyncio
    async def test_a_failed_write_is_still_counted(self, service):
        """The buckets count write ATTEMPTS, not completed writes, and that is
        deliberate: `DurableWriteQueue._process_item` re-invokes this executor
        on every RETRY with the decoded value, so retries land in the
        buckets and an item that eventually dead-letters is still counted.

        Counting only successes would instead make the escape go dark during a
        backend outage — exactly when a referent-less write storm is least
        likely to be noticed any other way. Pinned here so nobody 'fixes' the
        attempt-vs-write skew by moving the increment past the backend call
        without re-reading `referent_source_counts`' docstring.
        """
        service.graphiti.add_episode = AsyncMock(side_effect=RuntimeError('backend down'))
        value = _graphiti_value(
            referents=_encoded('derived', Referent(number='3127')),
        )

        for _ in range(3):  # the same item, retried
            with pytest.raises(RuntimeError):
                await service._execute_graphiti_write(value)

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
            _graphiti_value(referents=_encoded(
                'derived', Referent(number='3127'), Referent(number='2500'),
            )),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'derived'
        assert payload['referent_count'] == 2

    @pytest.mark.asyncio
    async def test_an_old_format_row_journals_none_and_zero(self, journaling_service):
        await journaling_service._execute_graphiti_write(
            _graphiti_value(),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'none'
        assert payload['referent_count'] == 0

    @pytest.mark.asyncio
    async def test_a_malformed_blob_journals_none_and_zero(self, journaling_service):
        """The durable channel and the in-process counter agree because both
        read the same decode — a malformed blob is 'none' in both."""
        await journaling_service._execute_graphiti_write(
            _graphiti_value(referents='garbage'),
        )

        payload = _backend_op_payload(journaling_service)
        assert payload['referent_source'] == 'none'
        assert payload['referent_count'] == 0
        assert journaling_service.referent_source_counts()['none'] == 1

    @pytest.mark.asyncio
    async def test_the_existing_journal_keys_are_unchanged(self, journaling_service):
        """Additive: content stays truncated to 200 and group_id stays put."""
        await journaling_service._execute_graphiti_write(
            _graphiti_value(
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
        batch = enqueued(service.durable_queue.enqueue_batch)
        assert len(batch) == 3
        assert all('referents' in json.loads(value.to_json()) for value, _ in batch)

    @pytest.mark.asyncio
    async def test_task_naming_rows_resolve_derived(self, service):
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'Task 2500 landed', 'metadata': {'category': 'temporal_facts'}},
            {'memory': 'the merge-lane hardening work', 'metadata': {}},
        ])

        await service.replay_from_store('dark_factory')

        blobs = [
            _graphiti_bound(value).referents.to_wire()
            for value, _ in enqueued(service.durable_queue.enqueue_batch)
        ]
        assert blobs[0] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }
        assert blobs[1] == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '2500'}],
            'ambiguous': [],
        }
        assert blobs[2] == {'source': 'none', 'refs': [], 'ambiguous': []}

    @pytest.mark.asyncio
    async def test_the_metadata_bridge_is_live_on_this_producer(self, service):
        """Unlike add_episode, the replay loop DOES hold a metadata dict — the
        Mem0 record's own — so the bridge can outrank the derived scan here."""
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed',
             'metadata': {'category': 'temporal_facts', 'task_id': '3129'}},
        ])

        await service.replay_from_store('dark_factory')

        blob = _graphiti_bound(enqueued(service.durable_queue.enqueue_batch)[0][0]).referents.to_wire()
        assert blob == {
            'source': 'metadata',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
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
        batch = enqueued(service.durable_queue.enqueue_batch)
        assert [_graphiti_bound(value).content for value, _ in batch] == [
            'Task 3127 landed', 'Task 2500 landed',
        ]
        assert all(_graphiti_bound(value).referents.source == 'derived' for value, _ in batch)
        assert sum(
            'not a string' in rec.getMessage() for rec in caplog.records
        ) == 2, f'expected one WARNING per skipped record; got {[r.getMessage() for r in caplog.records]}'

    @pytest.mark.asyncio
    async def test_preexisting_batch_payload_keys_survive(self, service):
        self._stub_memories(service, [
            {'memory': 'Task 3127 landed', 'metadata': {'category': 'temporal_facts'}},
        ])

        await service.replay_from_store('dark_factory')

        value, callback = enqueued(service.durable_queue.enqueue_batch)[0]
        assert value.envelope.group_id == 'dark_factory'
        assert value.operation.value == 'add_memory_graphiti'
        assert callback == QueueCallback.REFRESH_ENTITY_SUMMARIES
        wire = json.loads(value.to_json())
        assert wire['name'] == 'replay_temporal_facts'
        assert wire['content'] == 'Task 3127 landed'
        assert wire['source'] == 'text'
        assert wire['group_id'] == 'dark_factory'
        assert wire['source_description'] == 'replay_from_mem0:temporal_facts'


async def _completed_count_is(queue, expected: int) -> bool:
    """The executor bumps the referent counters BEFORE the queue commits the
    row as 'completed', so a counter barrier alone cannot order that read."""
    stats = await queue.get_stats()
    return stats['counts'].get('completed') == expected


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
            _graphiti_value(
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
        await poll_until(
            lambda: _completed_count_is(real_queue, 1),
            message="the row never reached 'completed'",
        )

    @pytest.mark.asyncio
    async def test_an_old_format_row_still_executes_end_to_end(
        self, real_queue, service,
    ):
        """The load-bearing back-compat signal, all the way through: a row
        written before task 3670 drains to 'completed' and counts only 'none'."""
        await real_queue.enqueue(
            _graphiti_value(group_id='dark_factory'),
        )

        await poll_until(
            lambda: service.referent_source_counts()['none'] == 1,
            message='the none bucket never incremented',
        )
        counts = service.referent_source_counts()
        assert counts['derived'] == 0
        await poll_until(
            lambda: _completed_count_is(real_queue, 1),
            message="the row never reached 'completed'",
        )

    @pytest.mark.asyncio
    async def test_a_flagged_ambiguity_still_vetoes_the_repair_after_the_round_trip(
        self, real_queue, service,
    ):
        """The whole chain in ONE assertion: producer -> JSON -> SQLite ->
        decode -> executor -> the real reconcile -> zeta's VETO 1.

        Everything else in this file pins one hop, and every OTHER executor-seam
        assertion uses `()` or `None` — the values that make the veto a no-op.
        So a defect at ANY hop that lost the set (a dropped key, a fresh empty
        local, a kwarg not forwarded) would leave those green while re-opening
        the exact failure the wire key closes: with the set gone, `Task 6379` is
        simply a non-member of the declared set, `_candidate_pool` falls back to
        that set, and the finding arrives at eta RESOLVABLE — a repair
        instruction to repoint the edge onto `Task 3127`.

        `unresolvable` is the observable because the counter increments exactly
        when `not finding.resolvable`. `set-membership` is asserted beside it so
        a chain that produced NO finding at all (a silently failed sub-pass)
        cannot pass by leaving both at zero.
        """
        from test_referent_verification import _edge, _episode

        service.graphiti.add_episode = AsyncMock(return_value=_episode(
            edges=[_edge('e1', fact='the mirror was reconciled',
                         source='n-end', target='n-mirror')],
            nodes=[MockNode(name='Task 6379', uuid='n-end'),
                   MockNode(name='mirror', uuid='n-mirror')],
        ))

        await real_queue.enqueue(
            _graphiti_value(
                group_id='dark_factory',
                referents=_encoded(
                    'derived', Referent(number='3127'),
                    ambiguous=(Referent(number='6379'),),
                ),
            ),
        )

        await poll_until(
            lambda: service.referent_finding_counts()['set-membership'] == 1,
            message='the chain produced no set-membership finding at all',
        )
        assert service.referent_finding_counts()['unresolvable'] == 1, (
            'the finding came out RESOLVABLE, so the producer ambiguity set was '
            'lost somewhere between enqueue and zeta'
        )


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
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert service._reconcile_episode_identity.call_args[1]['referents'] == (
            Referent(number='3127'),
        )

    @pytest.mark.asyncio
    async def test_an_old_format_row_reaches_it_with_an_empty_set(self, service):
        """The load-bearing back-compat path, still executing unchanged."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            _graphiti_value())

        assert service._reconcile_episode_identity.call_args[1]['referents'] == ()

    @pytest.mark.asyncio
    async def test_a_degraded_blob_reaches_it_as_an_empty_set(self, service):
        """`_decode_referents` is all-or-nothing precisely because a PARTIAL set
        would read as a conflation to zeta and drive eta's edge surgery onto the
        wrong node. An empty set no-ops the pass instead."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            _graphiti_value(referents={'source': 'derived', 'refs': 'not-a-list'}),
        )

        assert service._reconcile_episode_identity.call_args[1]['referents'] == ()

    @pytest.mark.asyncio
    async def test_a_populated_ambiguity_set_reaches_it_intact(self, service):
        """The one value that makes zeta's VETO 1 actually FIRE.

        Every other assertion at this seam is `()` or `None` — the two values
        that make the veto a no-op — so a plumbing defect substituting a fresh
        empty local for the decoded set would leave all of them green. The unit
        halves are covered on either side (`_decode_referents` returns the set;
        `test_referent_verification.py::TestTheWireAmbiguitySetIsPreferred`
        drives the verifier with a populated one); this is the hop between them,
        and dropping it hands eta an AMBIGUOUS endpoint as a destructive repair
        instruction — the failure threading the key exists to close.
        """
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded(
                'derived', Referent(number='3127'),
                ambiguous=(Referent(number='6379'),),
            )),
        )

        kwargs = service._reconcile_episode_identity.call_args[1]
        assert kwargs['ambiguous'] == (Referent(number='6379'),)
        # Asserted together: the two lists are decoded by one helper, so a
        # defect that crossed them would otherwise read as a pass here.
        assert kwargs['referents'] == (Referent(number='3127'),)

    @pytest.mark.asyncio
    async def test_the_backend_kwargs_are_still_untouched(self, service):
        """zeta reads the set AFTER the write; the backend never sees it."""
        service._reconcile_episode_identity = AsyncMock(return_value={})

        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
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

        async def _probe(result, *, group_id, referents, content, referent_source,
                         ambiguous):
            observed['locked'] = lock.locked()
            observed['same_lock'] = service.graphiti._identity_lock_for(group_id) is lock
            observed['referents'] = referents
            # zeta reads the producer's ambiguity set off the wire, falls back to
            # the FULL episode body for a legacy row, and reads the source to
            # decide whether the whole-declared-set fallback is licensed -- all
            # three must reach it through this same locked call, not a second
            # unlocked one.
            observed['ambiguous'] = ambiguous
            observed['content'] = content
            observed['referent_source'] = referent_source
            return {}

        service._reconcile_episode_identity = AsyncMock(side_effect=_probe)

        await service._execute_graphiti_write(
            _graphiti_value(referents=_encoded('derived', Referent(number='3127'))),
        )

        assert observed['locked'] is True
        assert observed['same_lock'] is True
        assert observed['referents'] == (Referent(number='3127'),)
        # `_encoded` builds a post-task-5262 blob, so the producer DID tell us
        # — an empty set, not the `None` of a legacy row.
        assert observed['ambiguous'] == ()
        # The FULL body, not the 200-char journal excerpt: a truncated content
        # would silently lose the second half of an ambiguity pair on the
        # legacy-row fallback.
        assert observed['content'] == 'test content'
        assert observed['referent_source'] == 'derived'
        assert lock.locked() is False, 'the lock must be released on return'


class TestDeclaredReferentsEndToEnd:
    """The whole of leaf delta in ONE path, not only at each seam.

    Everything above pins a single hop: the tool boundary (in
    tests/server/test_entities_gate_ingestion.py, against an AsyncMock
    service), or the producer (above, called directly). Neither can catch a
    wiring defect that lives BETWEEN them — a parameter renamed on one side, a
    gate that returns the right block while the tool forwards the write anyway.
    So this class drives the REAL `MemoryService` behind `create_mcp_server`
    and reads the durable-queue payload the executor will actually receive.

    Each case is one row of the PRD's own table, in the order that table
    states them, so a reader can check the implementation against the spec by
    reading the ids.

    `add_memory` carries the full table because it is the tool with all three
    rungs live (declared > metadata > derived). `add_episode` gets the two rows
    that are not merely a repeat of a seam test: the accepted declaration and
    the rejected one. Those two are what the between-the-seams gap can actually
    hide — a `declared_referents` kwarg the tool forwards under a name the
    producer no longer reads would leave BOTH seam suites green (each asserts
    its own side of the name) and land the write stamped `derived` in silence,
    and a gate whose block the tool returns while still enqueueing is invisible
    to a return-value assertion. The rungs BELOW tier 1 are deliberately not
    repeated here: `add_episode` persists no metadata, so its metadata rung is
    structurally dead and is pinned as such at the producer.
    """

    @pytest.fixture
    def server(self, service):
        from fused_memory.server.tools import create_mcp_server

        return create_mcp_server(service)

    @staticmethod
    def _referents(service) -> dict:
        return _stamped(service).referents.to_wire()

    @pytest.mark.asyncio
    async def test_declared_and_corroborated_lands_as_declared(self, server, service):
        """The PRD's transition out of the transitional auto-derive phase: a
        real declaration, observable at the wire."""
        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'category': 'decisions_and_rationale',
                'entities': [{'kind': 'task', 'id': 3127}],
            },
        )

        assert 'error' not in result, f'the write was blocked: {result!r}'
        assert service.durable_queue.enqueue.call_count == 1, (
            f'{service.durable_queue.enqueue.call_args_list!r}'
        )
        assert self._referents(service) == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_the_headline_conflict_enqueues_nothing_at_all(
        self, server, service,
    ):
        """The rejection's real payoff, and the assertion the return value
        cannot make: NOTHING is enqueued. No episode, so no edges, so nothing
        for leaf zeta to verify and nothing for leaf eta to repair. A gate that
        returned this error while still enqueueing would look identical to the
        caller and would be the exact misattribution this PRD exists to stop.
        """
        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'category': 'decisions_and_rationale',
                'entities': [{'kind': 'task', 'id': 3129}],
            },
        )

        assert result.get('error_type') == 'DeclaredReferentConflictRejected', f'{result!r}'
        assert result.get('conflicts') == ['Task 3129'], f'{result!r}'
        assert result.get('content_referents') == ['Task 3127'], f'{result!r}'
        service.durable_queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_undeclared_but_derivable_lands_as_derived(self, server, service):
        """The PRD's "Undeclared, derivable" row — today's majority, which must
        keep working untouched through the same path."""
        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'category': 'decisions_and_rationale',
            },
        )

        assert 'error' not in result, f'{result!r}'
        assert self._referents(service) == {
            'source': 'derived',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_the_metadata_bridge_is_live_at_the_tool_boundary(
        self, server, service,
    ):
        """Delta's piece 2, delivered by epsilon and pinned here at the seam it
        was never pinned at: `metadata['task_id']` outranks the prose scan
        through the TOOL, not merely through a direct service call. The Graphiti
        enqueue used to discard this metadata entirely, so "it works at the
        service" was never sufficient evidence that an agent's task_id survives.
        """
        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'category': 'decisions_and_rationale',
                'metadata': {'task_id': 3129},
            },
        )

        assert 'error' not in result, f'{result!r}'
        assert self._referents(service) == {
            'source': 'metadata',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3129'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_add_episode_declared_and_corroborated_lands_as_declared(
        self, server, service,
    ):
        """The same accepted-declaration row, through the OTHER write tool.

        PRD Open Question 3 was resolved by giving both tools `entities` in one
        leaf on the grounds that their gate stacks do not diverge. That claim
        is only checkable end to end: the seam tests assert the tool forwards
        `declared_referents` and (separately) that the producer stamps what it
        is handed, and neither can see the hop between them.
        """
        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'entities': [{'kind': 'task', 'id': 3127}],
            },
        )

        assert 'error' not in result, f'the episode was blocked: {result!r}'
        assert service.durable_queue.enqueue.call_count == 1, (
            f'{service.durable_queue.enqueue.call_args_list!r}'
        )
        assert self._referents(service) == {
            'source': 'declared',
            'refs': [{'kind': 'task', 'project_id': '', 'number': '3127'}],
            'ambiguous': [],
        }

    @pytest.mark.asyncio
    async def test_add_episode_headline_conflict_enqueues_nothing_at_all(
        self, server, service,
    ):
        """The rejection's payoff at `add_episode`, where it is worth strictly
        more than at `add_memory`: an ingested episode runs Graphiti's full
        extraction, so a misattributed one deposits edges nobody declared. The
        block alone does not prove that did not happen — only the absence of an
        enqueue does.
        """
        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'the fix for Task 3127 landed',
                'project_id': 'dark_factory',
                'entities': [{'kind': 'task', 'id': 3129}],
            },
        )

        assert result.get('error_type') == 'DeclaredReferentConflictRejected', f'{result!r}'
        assert result.get('conflicts') == ['Task 3129'], f'{result!r}'
        assert result.get('content_referents') == ['Task 3127'], f'{result!r}'
        service.durable_queue.enqueue.assert_not_called()
