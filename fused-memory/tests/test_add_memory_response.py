"""`AddMemoryResponse`: the per-store outcome lists, `mem0_write_state`, and the rendered message."""

import pydantic
import pytest

from fused_memory.models.enums import ClassificationFallback, SourceStore
from fused_memory.models.memory import (
    AddMemoryResponse,
    Mem0WriteState,
    render_add_memory_message,
)


class TestMem0WriteState:
    def test_a_write_with_an_id_landed(self) -> None:
        response = AddMemoryResponse(memory_ids=['m1'], stores_written=[SourceStore.mem0])
        assert response.mem0_write_state is Mem0WriteState.LANDED

    @pytest.mark.parametrize(
        'stores_queued',
        [[], [SourceStore.graphiti]],
        ids=['mem0-only', 'dual-write-whose-graphiti-leg-is-queued'],
    )
    def test_a_mem0_leg_queued_for_retry_is_pending(self, stores_queued) -> None:
        response = AddMemoryResponse(
            memory_ids=[],
            stores_queued=stores_queued,
            stores_pending_retry=[SourceStore.mem0],
        )
        assert response.mem0_write_state is Mem0WriteState.PENDING_RETRY

    @pytest.mark.parametrize(
        'response',
        [
            AddMemoryResponse(memory_ids=[]),
            AddMemoryResponse(memory_ids=[], stores_queued=[SourceStore.graphiti]),
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


class TestStoresQueued:
    def test_a_queued_store_is_on_the_wire(self) -> None:
        dumped = AddMemoryResponse(stores_queued=[SourceStore.graphiti]).model_dump()
        assert dumped['stores_queued'] == ['graphiti']

    def test_nothing_is_queued_by_default(self) -> None:
        assert AddMemoryResponse().stores_queued == []

    @pytest.mark.parametrize(
        ('first', 'second'),
        [
            ('stores_written', 'stores_queued'),
            ('stores_written', 'stores_pending_retry'),
            ('stores_queued', 'stores_pending_retry'),
        ],
    )
    def test_a_store_reported_in_two_lists_is_rejected(self, first, second) -> None:
        with pytest.raises(pydantic.ValidationError) as excinfo:
            AddMemoryResponse.model_validate(
                {first: [SourceStore.graphiti], second: [SourceStore.graphiti]}
            )
        text = str(excinfo.value)
        assert 'graphiti' in text
        assert first in text
        assert second in text


class TestRenderAddMemoryMessage:
    def test_a_mem0_only_write(self) -> None:
        assert (
            render_add_memory_message(stores_written=[SourceStore.mem0])
            == "Memory written to ['mem0']"
        )

    def test_a_graphiti_only_write(self) -> None:
        assert (
            render_add_memory_message(stores_written=[], stores_queued=[SourceStore.graphiti])
            == "Memory queued for ['graphiti']"
        )

    def test_a_dual_write(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[SourceStore.mem0], stores_queued=[SourceStore.graphiti]
            )
            == "Memory written to ['mem0']; queued for ['graphiti']"
        )

    def test_a_mem0_leg_pending_retry_beside_a_queued_graphiti_leg(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[],
                stores_queued=[SourceStore.graphiti],
                stores_pending_retry=[SourceStore.mem0],
                mem0_pending_retry='rate_limited: X',
            )
            == "Memory queued for ['graphiti']; pending retry for ['mem0']"
            ' [mem0_pending_retry: rate_limited: X]'
        )

    def test_nothing_landed(self) -> None:
        assert (
            render_add_memory_message(stores_written=[], mem0_error='qdrant down')
            == 'Memory not written to any store [mem0_error: qdrant down]'
        )

    def test_every_note_renders_in_order(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[],
                graphiti_error='G',
                mem0_error='M',
                mem0_empty_result='empty_result: E',
                mem0_pending_retry='P',
                classification_fallback=ClassificationFallback.llm_error,
            )
            == 'Memory not written to any store'
            ' [graphiti_error: G]'
            ' [mem0_error: M]'
            ' [empty_result: E]'
            ' [mem0_pending_retry: P]'
            ' [classification_fallback: llm_error]'
        )


class TestTheMessageKeepsEachFailedStoresErrorText:
    def test_both_legs_failed(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[],
                graphiti_error='RuntimeError: queue down',
                mem0_error='qdrant down',
            )
            == 'Memory not written to any store'
            ' [graphiti_error: RuntimeError: queue down] [mem0_error: qdrant down]'
        )

    def test_the_graphiti_enqueue_failed_and_mem0_landed(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[SourceStore.mem0],
                graphiti_error='RuntimeError: queue down',
            )
            == "Memory written to ['mem0'] [graphiti_error: RuntimeError: queue down]"
        )

    def test_a_mem0_empty_result_beside_a_queued_graphiti_leg(self) -> None:
        assert (
            render_add_memory_message(
                stores_written=[],
                stores_queued=[SourceStore.graphiti],
                mem0_empty_result='empty_result: mem0 add returned zero memory_ids',
            )
            == "Memory queued for ['graphiti'] [empty_result: mem0 add returned zero memory_ids]"
        )
