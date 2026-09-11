"""Unit tests for the single-home search-telemetry summary shape (task 3212, item 1).

`fused_memory/services/read_telemetry.py::summarize_search_results` is the ONE
home for the shape journalled into `write_ops.result_summary` for a search.  It
is pure and I/O-free so the contract can be pinned without standing up an MCP
server, and it is shared by three producers — the MCP `search` tool (grouped
dicts), `MemoryService.search` (raw `MemoryResult`s) and the reconciliation
hint-execution site — so the shape cannot drift into three variants (INV-5).
"""

from __future__ import annotations

import json
import uuid

from fused_memory.models.memory import MemoryCategory, MemoryResult, SourceStore
from fused_memory.services.read_telemetry import (
    SEARCH_TELEMETRY_MAX_RESULTS,
    SEARCH_TELEMETRY_SCHEMA_VERSION,
    SEARCH_TELEMETRY_SIZE_UNIT,
    summarize_search_results,
)


def _make_result(
    content: str = 'a memory body',
    relevance_score: float = 0.8,
    source: SourceStore = SourceStore.mem0,
    metadata: dict | None = None,
    topic_anchored: bool = False,
) -> MemoryResult:
    return MemoryResult(
        id=str(uuid.uuid4()),
        content=content,
        source_store=source,
        category=MemoryCategory.observations_and_summaries,
        relevance_score=relevance_score,
        metadata=metadata if metadata is not None else {},
        topic_anchored=topic_anchored,
    )


class TestSummaryEnvelope:
    """The envelope keys leaf eta (3213) and leaf theta (3214) read."""

    def test_envelope_keys_and_per_result_fields(self):
        """A list of MemoryResults yields the full envelope + per-result detail."""
        first = _make_result(content='x' * 37, relevance_score=0.9)
        second = _make_result(content='y' * 11, relevance_score=0.4)

        summary = summarize_search_results([first, second])

        assert summary['schema_version'] == SEARCH_TELEMETRY_SCHEMA_VERSION, (
            f'schema_version must be stamped so a later shape change is readable, '
            f'got {summary.get("schema_version")!r}. RED: shape is not versioned.'
        )
        assert SEARCH_TELEMETRY_SCHEMA_VERSION == 1, (
            'The first version of the logged shape is 1. '
            f'RED: got {SEARCH_TELEMETRY_SCHEMA_VERSION!r}.'
        )
        assert summary['count'] == 2, (
            f'count must report the TRUE total, got {summary.get("count")!r}. '
            'RED: count not carried through.'
        )
        assert summary['results_logged'] == 2, (
            f'results_logged must report how many per-result entries were kept, '
            f'got {summary.get("results_logged")!r}. RED: key missing.'
        )
        assert summary['results_truncated'] is False, (
            f'results_truncated must be False below the cap, '
            f'got {summary.get("results_truncated")!r}. RED: key missing.'
        )

        entries = summary['results']
        assert [e['id'] for e in entries] == [first.id, second.id], (
            'Per-result ids must be logged in result order — they are leaf eta\'s '
            f'whole question ("was the agent shown this?"), got {entries!r}. '
            'RED: ids not recorded.'
        )
        assert [e['relevance_score'] for e in entries] == [0.9, 0.4], (
            f'Per-result relevance scores must be logged, got {entries!r}. '
            'RED: scores not recorded.'
        )
        assert [e['content_size'] for e in entries] == [37, 11], (
            f'content_size must equal len(content) in characters, got {entries!r}. '
            'RED: sizes not recorded.'
        )
        assert [e['source_store'] for e in entries] == ['mem0', 'mem0'], (
            f'source_store must be logged as a plain string, got {entries!r}. '
            'RED: source_store not recorded (or not coerced off the StrEnum).'
        )
        assert [e['topic_anchored'] for e in entries] == [False, False], (
            'topic_anchored must be logged — a pinned result did not earn its slot '
            f'by rank, so its score is not meaningful. got {entries!r}. RED: key missing.'
        )

    def test_size_unit_is_a_named_key_not_an_implied_one(self):
        """The unit is emitted, not inferred from a field name (task 3212 item 1)."""
        summary = summarize_search_results([_make_result()])

        assert summary['size_unit'] == 'chars', (
            f'size_unit must be the literal "chars", got {summary.get("size_unit")!r}. '
            'RED: unit is implied rather than named.'
        )
        assert SEARCH_TELEMETRY_SIZE_UNIT == 'chars', (
            f'The module constant must name the unit, got {SEARCH_TELEMETRY_SIZE_UNIT!r}.'
        )

    def test_empty_result_list(self):
        """No results is a first-class case, not an error."""
        summary = summarize_search_results([])

        assert summary['count'] == 0, f'RED: empty search must report count 0, got {summary!r}'
        assert summary['results'] == [], f'RED: empty search must report [], got {summary!r}'
        assert summary['results_logged'] == 0, f'RED: got {summary!r}'
        assert summary['results_truncated'] is False, (
            f'An empty list is not a truncated list. RED: got {summary!r}'
        )
        assert summary['size_unit'] == 'chars', (
            f'The unit is stamped even with nothing to measure. RED: got {summary!r}'
        )

    def test_summary_is_json_serialisable(self):
        """The summary is written through json.dumps by WriteJournal.log_write_op."""
        summary = summarize_search_results(
            [_make_result(metadata={'store_rank': 1, 'store_score': 0.51})]
        )

        json.dumps(summary)  # RED: raises TypeError if an enum leaks through.


class TestStoreRankAndScore:
    """Task 3658 stamped per-store truth in metadata; log it, never invent it."""

    def test_store_rank_and_score_lifted_from_metadata(self):
        result = _make_result(metadata={'store_rank': 2, 'store_score': 0.4217})

        entry = summarize_search_results([result])['results'][0]

        assert entry['store_rank'] == 2, (
            f'store_rank must be lifted out of MemoryResult.metadata, got {entry!r}. '
            'RED: 3658 per-store truth not logged.'
        )
        assert entry['store_score'] == 0.4217, (
            f'store_score must be lifted out of MemoryResult.metadata, got {entry!r}. '
            'RED: 3658 per-store truth not logged.'
        )

    def test_absent_rank_and_score_are_none_not_invented(self):
        entry = summarize_search_results([_make_result(metadata={})])['results'][0]

        assert entry['store_rank'] is None, (
            f'An absent store_rank must log as None, never as a fabricated 0/1, '
            f'got {entry!r}. RED: value invented.'
        )
        assert entry['store_score'] is None, (
            f'An absent store_score must log as None, got {entry!r}. RED: value invented.'
        )

    def test_graphiti_none_store_score_passes_through_verbatim(self):
        """Graphiti's contract is store_score=None; a coercion to 0.0 would be a lie."""
        entry = summarize_search_results(
            [
                _make_result(
                    source=SourceStore.graphiti,
                    metadata={'store_rank': 1, 'store_score': None},
                )
            ]
        )['results'][0]

        assert entry['store_rank'] == 1, f'RED: rank dropped, got {entry!r}'
        assert entry['store_score'] is None, (
            'Graphiti reports no per-store score by contract — None must pass through '
            f'verbatim rather than being coerced to 0.0, got {entry!r}. '
            'RED: a coerced 0.0 is indistinguishable from a genuinely worst-ranked hit.'
        )


class TestDictInput:
    """One implementation serves MemoryResults AND their model_dump() dicts."""

    def test_model_dump_dicts_produce_a_byte_identical_summary(self):
        results = [
            _make_result(content='alpha' * 3, metadata={'store_rank': 1, 'store_score': 0.7}),
            _make_result(content='beta', source=SourceStore.graphiti, topic_anchored=True),
        ]

        from_objects = summarize_search_results(results)
        from_dicts = summarize_search_results([r.model_dump() for r in results])

        assert from_dicts == from_objects, (
            'The MCP boundary summarises model_dump() dicts while MemoryService '
            'summarises MemoryResult objects — one implementation must produce the '
            f'SAME summary for both.\nobjects={from_objects!r}\ndicts={from_dicts!r}\n'
            'RED: the dict path diverges.'
        )
        assert json.dumps(from_dicts) == json.dumps(from_objects), (
            'Byte-identical, not merely ==: an enum vs its string value compares equal '
            'but serialises differently. RED: source_store not coerced to str.'
        )

    def test_topic_anchored_is_read_off_a_dict(self):
        result = _make_result(topic_anchored=True)

        entry = summarize_search_results([result.model_dump()])['results'][0]

        assert entry['topic_anchored'] is True, (
            f'topic_anchored must be read off the dict too, got {entry!r}. RED: key dropped.'
        )


class TestGroupedPayload:
    """Grouping (task 3129) folds children INTO a parent — those ids were shown too."""

    def _grouped_entry(self, parent: MemoryResult, child_ids: list[str], amend_ids: list[str]) -> dict:
        entry = parent.model_dump()
        entry['grouped'] = {
            'matched_children': [
                {'id': cid, 'content': 'child body', 'kind': 'amendment', 'matched': True}
                for cid in child_ids
            ],
            'amendments': [
                {'id': aid, 'digest': 'digest body', 'kind': 'amendment'} for aid in amend_ids
            ],
            'amendment_count': len(amend_ids),
            'sighting_count': 0,
        }
        return entry

    def test_folded_child_ids_appear_in_the_summary(self):
        parent = _make_result(content='canonical body')
        entry = self._grouped_entry(parent, ['child-a', 'child-b'], ['amend-a'])

        summary = summarize_search_results([entry])
        logged = summary['results'][0]

        assert logged['id'] == parent.id, f'RED: parent id lost, got {logged!r}'
        assert set(logged.get('folded_child_ids') or []) == {'child-a', 'child-b', 'amend-a'}, (
            'Ids folded into grouped.matched_children and grouped.amendments were SHOWN '
            'to the agent inside the parent entry, so leaf eta must be able to see them; '
            f'got {logged!r}. RED: folded child ids not collected.'
        )

    def test_no_grouped_block_means_no_folded_ids_key(self):
        logged = summarize_search_results([_make_result()])['results'][0]

        assert 'folded_child_ids' not in logged, (
            'An ungrouped hit folded nothing — an empty list would be noise on every row '
            f'of the 1.36% of the table search occupies. got {logged!r}. '
            'RED: key emitted unconditionally.'
        )

    def test_malformed_grouped_block_does_not_break_the_summary(self):
        """A grouped block is produced elsewhere; telemetry must not be its validator."""
        entry = _make_result().model_dump()
        entry['grouped'] = {'matched_children': 'not-a-list', 'amendments': [{'no_id': 1}]}

        summary = summarize_search_results([entry])

        assert summary['count'] == 1, (
            f'A malformed grouped block must not cost the row its telemetry, got {summary!r}. '
            'RED: summariser is not defensive about a foreign payload.'
        )


class TestTruncation:
    """No silent caps: count keeps the truth, the cap is disclosed."""

    def test_cap_is_disclosed_never_silent(self):
        over = SEARCH_TELEMETRY_MAX_RESULTS + 7
        results = [_make_result(content=f'body {i}') for i in range(over)]

        summary = summarize_search_results(results)

        assert summary['count'] == over, (
            f'count must keep the TRUE total ({over}) even when the per-result list is '
            f'capped, got {summary.get("count")!r}. RED: the cap silently rewrote the count.'
        )
        assert len(summary['results']) == SEARCH_TELEMETRY_MAX_RESULTS, (
            f'The per-result list must be capped at SEARCH_TELEMETRY_MAX_RESULTS '
            f'({SEARCH_TELEMETRY_MAX_RESULTS}), got {len(summary["results"])}. RED: no cap.'
        )
        assert summary['results_logged'] == SEARCH_TELEMETRY_MAX_RESULTS, (
            f'results_logged must report the kept count, got {summary.get("results_logged")!r}. '
            'RED: key missing or wrong.'
        )
        assert summary['results_truncated'] is True, (
            f'results_truncated must be True when entries were dropped, '
            f'got {summary.get("results_truncated")!r}. RED: truncation is silent.'
        )

    def test_cap_is_overridable_per_call(self):
        results = [_make_result() for _ in range(5)]

        summary = summarize_search_results(results, max_results=2)

        assert summary['count'] == 5, f'RED: true total lost, got {summary!r}'
        assert summary['results_logged'] == 2, f'RED: cap not applied, got {summary!r}'
        assert summary['results_truncated'] is True, f'RED: got {summary!r}'

    def test_cap_default_is_headroom_over_real_callers(self):
        """briefing.py asks for 5, ReconciliationConfig for 5, the tool defaults to 10."""
        assert SEARCH_TELEMETRY_MAX_RESULTS == 50, (
            f'The cap is derived from observed callers (5-10 results), not a round guess; '
            f'50 is 5-10x headroom and bounds a pathological row to ~6 KB. '
            f'RED: got {SEARCH_TELEMETRY_MAX_RESULTS!r}.'
        )
