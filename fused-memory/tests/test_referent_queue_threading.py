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
