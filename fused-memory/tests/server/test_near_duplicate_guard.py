"""Unit tests for the pure near-duplicate selection helper (task 2467).

``find_near_duplicate_memory`` is a pure, synchronous function: given a list
of already-fetched ``MemoryResult`` search hits and a similarity threshold,
it selects the single best near-duplicate candidate (or ``None``). It does
no I/O and makes no assumptions about embedding accuracy — callers inject
explicit ``relevance_score`` values, so these tests exercise pure comparison
logic only.
"""

from __future__ import annotations

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.near_duplicate_guard import find_near_duplicate_memory


def _result(
    id_: str,
    score: float,
    *,
    category: MemoryCategory | None = MemoryCategory.procedural_knowledge,
    source_store: SourceStore = SourceStore.mem0,
    content: str = 'some procedural content',
) -> MemoryResult:
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=source_store,
        relevance_score=score,
    )


class TestFindNearDuplicateMemory:
    """Pure selection logic for the write-time near-duplicate guard."""

    def test_returns_match_when_above_threshold(self):
        results = [_result('m1', 0.97)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm1'

    def test_returns_none_when_all_below_threshold(self):
        results = [_result('m1', 0.50), _result('m2', 0.10)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_score_equal_to_threshold_is_a_match(self):
        results = [_result('m1', 0.92)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm1'

    def test_empty_results_returns_none(self):
        match = find_near_duplicate_memory([], 0.92)
        assert match is None

    def test_ignores_wrong_category(self):
        results = [
            _result('m1', 0.99, category=MemoryCategory.observations_and_summaries),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_ignores_wrong_source_store(self):
        results = [
            _result('m1', 0.99, source_store=SourceStore.graphiti),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_picks_max_score_among_multiple_qualifying_results(self):
        results = [
            _result('m1', 0.93),
            _result('m2', 0.99),
            _result('m3', 0.95),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm2'

    def test_mixed_qualifying_and_disqualified_picks_best_qualifying(self):
        results = [
            _result('m1', 0.999, category=MemoryCategory.observations_and_summaries),
            _result('m2', 0.93),
            _result('m3', 0.999, source_store=SourceStore.graphiti),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm2'
