"""`AddMemoryResponse.mem0_write_state`: the one reading of whether a write's Mem0 leg landed."""

import pytest

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import AddMemoryResponse, Mem0WriteState


class TestMem0WriteState:
    def test_a_write_with_an_id_landed(self) -> None:
        response = AddMemoryResponse(memory_ids=['m1'], stores_written=[SourceStore.mem0])
        assert response.mem0_write_state is Mem0WriteState.LANDED

    @pytest.mark.parametrize(
        'stores_written',
        [[], [SourceStore.graphiti]],
        ids=['mem0-only', 'dual-write-whose-graphiti-leg-landed'],
    )
    def test_a_mem0_leg_queued_for_retry_is_pending(self, stores_written) -> None:
        response = AddMemoryResponse(
            memory_ids=[],
            stores_written=stores_written,
            stores_pending_retry=[SourceStore.mem0],
        )
        assert response.mem0_write_state is Mem0WriteState.PENDING_RETRY

    @pytest.mark.parametrize(
        'response',
        [
            AddMemoryResponse(memory_ids=[]),
            AddMemoryResponse(memory_ids=[], stores_written=[SourceStore.graphiti]),
            AddMemoryResponse(),
        ],
        ids=['mem0-leg-failed', 'graphiti-only-write', 'empty-default'],
    )
    def test_no_id_and_nothing_pending_did_not_land(self, response) -> None:
        assert response.mem0_write_state is Mem0WriteState.NOT_LANDED


class TestTheStateIsNotOnTheWire:
    def test_model_dump_carries_only_the_declared_fields(self) -> None:
        dumped = AddMemoryResponse(
            memory_ids=[], stores_pending_retry=[SourceStore.mem0]
        ).model_dump()
        assert 'mem0_write_state' not in dumped
        assert set(dumped) == set(AddMemoryResponse.model_fields)
