"""Unit tests for the topic-anchored canonical recall transform (task 3111).

Covers the ``MemoryResult.topic_anchored`` retrieval-state field and the pure,
synchronous selectors in ``services/topic_anchor.py``.  The service-seam
integration lives in ``test_memory_service_topic_anchored_recall.py``; the MCP
composition proof lives in ``tests/server/test_topic_anchored_recall_mcp.py``.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.topic_anchor import (
    _CHILD_KINDS,
    _DEFAULT_TOPIC_ANCHOR_ENABLED,
    extract_anchor_topics,
    resolve_topic_anchor_enabled,
    select_canonical_payload,
)


def _result(
    id_: str = 'm1',
    *,
    topic: object = 'harness-layout-gate-decision-rule',
    source_store: SourceStore = SourceStore.mem0,
    category: MemoryCategory | None = MemoryCategory.procedural_knowledge,
    content: str = 'a sibling record on the consolidated topic',
) -> MemoryResult:
    """Build a real post-RRF MemoryResult, mirroring _near_duplicate_result.

    *topic* is placed in ``metadata['topic']`` verbatim — including sentinel
    ``...`` to mean "omit the key entirely" — so the selector's defensive
    type handling is exercised against real Pydantic objects, not dicts.
    """
    metadata: dict = {}
    if topic is not ...:
        metadata['topic'] = topic
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=source_store,
        metadata=metadata,
    )


class TestMemoryResultTopicAnchoredField:
    """Pin ``topic_anchored`` as a real, always-present field on MemoryResult."""

    def test_defaults_to_false_when_not_supplied(self):
        """An ordinary search result is not topic-anchored unless the pin says so."""
        result = MemoryResult(
            id='m1',
            content='an ordinary ranked result',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
        )

        assert result.topic_anchored is False

    def test_true_round_trips_through_model_dump(self):
        """The search MCP tool surfaces results via model_dump(), so the flag must survive it."""
        result = MemoryResult(
            id='m1',
            content='the canonical, promoted into the window by rule',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
            topic_anchored=True,
        )

        assert result.topic_anchored is True
        assert result.model_dump()['topic_anchored'] is True

    def test_key_present_in_model_dump_even_when_false(self):
        """Consumers may read the key unconditionally — it is never omitted."""
        result = MemoryResult(
            id='m1',
            content='an ordinary ranked result',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
        )

        dumped = result.model_dump()
        assert 'topic_anchored' in dumped
        assert dumped['topic_anchored'] is False


class TestExtractAnchorTopics:
    """The pure, synchronous topic selector that decides what to look up."""

    def test_returns_distinct_mem0_topics_in_rank_order(self):
        """First-seen (rank) order is the contract — results arrive already sorted."""
        results = [
            _result('m1', topic='topic-a'),
            _result('m2', topic='topic-b'),
            _result('m3', topic='topic-c'),
        ]

        assert extract_anchor_topics(results, max_topics=3) == ['topic-a', 'topic-b', 'topic-c']

    def test_ignores_graphiti_sourced_results(self):
        """metadata.topic is a Mem0 vocabulary key; a Graphiti row can never anchor."""
        results = [
            _result('g1', topic='graphiti-topic', source_store=SourceStore.graphiti),
            _result('m1', topic='mem0-topic'),
        ]

        assert extract_anchor_topics(results, max_topics=3) == ['mem0-topic']

    def test_ignores_missing_empty_and_non_str_topics(self):
        """Raw payload values are not schema-enforced at read time — degrade, never raise."""
        results = [
            _result('m1', topic=...),
            _result('m2', topic=''),
            _result('m3', topic=7),
            _result('m4', topic=['topic-a']),
            _result('m5', topic=None),
            _result('m6', topic=True),
            _result('m7', topic='real-topic'),
        ]

        assert extract_anchor_topics(results, max_topics=5) == ['real-topic']

    def test_deduplicates_repeats_preserving_first_seen_order(self):
        """Nine siblings on one topic must cost ONE lookup, not nine."""
        results = [
            _result('m1', topic='topic-a'),
            _result('m2', topic='topic-b'),
            _result('m3', topic='topic-a'),
            _result('m4', topic='topic-b'),
            _result('m5', topic='topic-c'),
        ]

        assert extract_anchor_topics(results, max_topics=5) == ['topic-a', 'topic-b', 'topic-c']

    def test_truncates_to_max_topics(self):
        """The cap is a blast-radius bound on backend round-trips per search."""
        results = [_result(f'm{i}', topic=f'topic-{i}') for i in range(10)]

        assert extract_anchor_topics(results, max_topics=3) == ['topic-0', 'topic-1', 'topic-2']

    def test_empty_input_returns_empty_list(self):
        """The zero-cost path: no results means no topics means no I/O."""
        assert extract_anchor_topics([], max_topics=3) == []

    def test_no_topics_returns_empty_list(self):
        """The common case on the live corpus today — must short-circuit to no I/O."""
        results = [_result('m1', topic=...), _result('m2', topic=...)]

        assert extract_anchor_topics(results, max_topics=3) == []


def _payload(
    id_: str = 'c1',
    *,
    created_at: object = '2026-08-01T00:00:00+00:00',
    metadata: object = ...,
    **meta_overrides: object,
) -> dict:
    """Build a raw scroll dict of the shape get_memories_by_metadata returns.

    ``{'id', 'created_at', 'metadata': <FULL raw Qdrant payload>}`` — note the
    metadata is the unprocessed payload, so its keys and value types are NOT
    schema-enforced here.  Pass ``metadata=`` explicitly to inject a malformed
    (absent/non-dict) payload; otherwise *meta_overrides* build a well-formed
    one, with ``...`` meaning "omit this key".
    """
    if metadata is ...:
        base: dict = {
            'canonical': True,
            'topic': 'harness-layout-gate-decision-rule',
            'category': 'procedural_knowledge',
            'data': 'the consolidated canonical body',
        }
        for key, value in meta_overrides.items():
            if value is ...:
                base.pop(key, None)
            else:
                base[key] = value
        metadata = base
    return {'id': id_, 'created_at': created_at, 'metadata': metadata}


class TestSelectCanonicalPayload:
    """The pure selector that picks WHICH scrolled payload gets pinned."""

    def test_selects_the_strictly_canonical_payload(self):
        """canonical is a validated bool; the read side matches on identity."""
        payloads = [
            _payload('c0', canonical=...),
            _payload('c1', canonical=True),
        ]

        selected = select_canonical_payload(payloads, allowed_categories=None,
                                            include_planned=False)

        assert selected is not None
        assert selected['id'] == 'c1'

    def test_truthy_non_bools_are_rejected(self):
        """`1 == True` in Python, so truthiness would admit an int as canonical.

        Mirrors the write-side predicate at memory_service.py:940
        (``meta.get('canonical') is not True``) so both sides of the vocabulary
        agree on what "canonical" means.
        """
        for truthy in (1, 'true', 'True', 'yes', [1], 1.0):
            payloads = [_payload('c1', canonical=truthy)]

            assert select_canonical_payload(
                payloads, allowed_categories=None, include_planned=False
            ) is None, f'{truthy!r} must not read as canonical'

    def test_returns_none_when_nothing_is_canonical(self):
        payloads = [_payload('c1', canonical=False), _payload('c2', canonical=...)]

        assert select_canonical_payload(
            payloads, allowed_categories=None, include_planned=False
        ) is None

    def test_returns_none_for_empty_input(self):
        assert select_canonical_payload(
            [], allowed_categories=None, include_planned=False
        ) is None

    def test_allowed_categories_excludes_out_of_scope_canonicals(self):
        """Anchoring may change WHICH member of the permitted set is returned, never widen it."""
        payloads = [_payload('c1', category='observations_and_summaries')]

        assert select_canonical_payload(
            payloads,
            allowed_categories={'procedural_knowledge'},
            include_planned=False,
        ) is None

    def test_allowed_categories_none_permits_any_category(self):
        """An unscoped search filters nothing, so neither does the pin."""
        payloads = [_payload('c1', category='observations_and_summaries')]

        selected = select_canonical_payload(payloads, allowed_categories=None,
                                            include_planned=False)

        assert selected is not None and selected['id'] == 'c1'

    def test_planned_payloads_excluded_unless_include_planned(self):
        payloads = [_payload('c1', planned=True)]

        assert select_canonical_payload(
            payloads, allowed_categories=None, include_planned=False
        ) is None
        included = select_canonical_payload(
            payloads, allowed_categories=None, include_planned=True
        )
        assert included is not None and included['id'] == 'c1'

    def test_child_shaped_payloads_excluded_even_when_canonical(self):
        """A child would be folded into its PARENT at index 0 by grouped_read.

        ``grouped_read._suppress_child`` (:685-691) runs AFTER
        ``MemoryService.search``, so pinning a child-shaped record would let
        the boundary silently swap slot 0 for that record's parent — a
        different record than the one the pin selected.  Exclude at selection
        time rather than relying on well-formed writes.
        """
        for kind in sorted(_CHILD_KINDS):
            payloads = [_payload('c1', kind=kind, parent_id='parent-uuid-0001')]

            assert select_canonical_payload(
                payloads, allowed_categories=None, include_planned=False
            ) is None, f'child-shaped {kind} must not be pinned'

    def test_child_kinds_matches_the_grouped_read_registry(self):
        """Anti-drift pin on a DELIBERATE duplication.

        server/grouped_read.py imports _MEM0_CONTENT_KEYS from
        services/memory_service.py, so a services -> server import here would
        close an import cycle. The constant is therefore restated locally; this
        assertion is what stops the two copies drifting apart silently.
        """
        from fused_memory.server.grouped_read import CHILD_KINDS

        assert _CHILD_KINDS == CHILD_KINDS

    def test_child_kind_without_parent_id_is_not_child_shaped(self):
        """BOTH halves are required — a kind alone cannot be suppressed upward."""
        for parent_id in (..., '', None, 7, []):
            payloads = [_payload('c1', kind='amendment', parent_id=parent_id)]

            selected = select_canonical_payload(
                payloads, allowed_categories=None, include_planned=False
            )
            assert selected is not None, f'parent_id={parent_id!r} is not a real parent link'
            assert selected['id'] == 'c1'

    def test_parent_id_without_child_kind_is_not_child_shaped(self):
        """Ditto the other half — a non-child kind is never suppressed."""
        for kind in (..., '', None, 7, 'decision'):
            payloads = [_payload('c1', kind=kind, parent_id='parent-uuid-0001')]

            selected = select_canonical_payload(
                payloads, allowed_categories=None, include_planned=False
            )
            assert selected is not None, f'kind={kind!r} is not a child kind'

    def test_tie_break_is_most_recent_then_lowest_id(self):
        """Duplicate canonicals are REACHABLE, not theoretical.

        3198's per-(project, topic) uniqueness enforcement ships warn-mode
        first (``memory_metadata.enforce`` defaults false), so two canonicals
        on one topic can land via ordinary writes. The selector must still
        return ONE record, deterministically.
        """
        older = _payload('a-old', created_at='2026-08-01T00:00:00+00:00')
        newer = _payload('b-new', created_at='2026-08-09T00:00:00+00:00')

        selected = select_canonical_payload([older, newer], allowed_categories=None,
                                            include_planned=False)

        assert selected is not None and selected['id'] == 'b-new'

    def test_tie_break_falls_through_to_lowest_id(self):
        """Same timestamp: the id is the total-order tiebreak."""
        same_time = '2026-08-09T00:00:00+00:00'
        payloads = [
            _payload('zzz', created_at=same_time),
            _payload('aaa', created_at=same_time),
            _payload('mmm', created_at=same_time),
        ]

        selected = select_canonical_payload(payloads, allowed_categories=None,
                                            include_planned=False)

        assert selected is not None and selected['id'] == 'aaa'

    def test_tie_break_is_stable_across_input_permutations(self):
        """Order-independence is the point — scroll order is not guaranteed."""
        import itertools

        payloads = [
            _payload('a-old', created_at='2026-08-01T00:00:00+00:00'),
            _payload('b-new', created_at='2026-08-09T00:00:00+00:00'),
            _payload('c-new', created_at='2026-08-09T00:00:00+00:00'),
        ]

        selections = [
            select_canonical_payload(list(perm), allowed_categories=None,
                                     include_planned=False)
            for perm in itertools.permutations(payloads)
        ]
        assert all(s is not None for s in selections)
        picks = {s['id'] for s in selections if s is not None}

        assert picks == {'b-new'}

    def test_missing_or_non_str_created_at_sorts_oldest_and_never_raises(self):
        """An absent timestamp must not win the recency tie-break, nor crash the sort."""
        payloads = [
            _payload('a-undated', created_at=None),
            _payload('b-dated', created_at='2026-01-01T00:00:00+00:00'),
        ]

        selected = select_canonical_payload(payloads, allowed_categories=None,
                                            include_planned=False)

        assert selected is not None and selected['id'] == 'b-dated'

    def test_malformed_payloads_degrade_to_not_canonical(self):
        """Raw payloads are not schema-enforced at read time — degrade, never raise."""
        malformed = [
            {'id': 'c1', 'created_at': None},                       # no metadata key at all
            {'id': 'c2', 'created_at': None, 'metadata': None},     # metadata None
            {'id': 'c3', 'created_at': None, 'metadata': 'nope'},   # metadata not a dict
            {'id': 'c4', 'created_at': None, 'metadata': []},       # metadata wrong container
            'not-a-dict-at-all',                                    # payload itself malformed
            None,
        ]

        assert select_canonical_payload(
            malformed, allowed_categories=None, include_planned=False
        ) is None

    def test_malformed_payloads_do_not_shadow_a_real_canonical(self):
        """Junk in the scroll result must not suppress the record that IS canonical."""
        payloads = [
            {'id': 'junk', 'metadata': 'nope'},
            None,
            _payload('c1', canonical=True),
        ]

        selected = select_canonical_payload(payloads, allowed_categories=None,
                                            include_planned=False)

        assert selected is not None and selected['id'] == 'c1'


def _memory_service_with_reconciliation(**leaves: object) -> types.SimpleNamespace:
    """A config double built from SimpleNamespace, never MagicMock.

    scripts/check_bare_magicmock_config.py rejects a bare MagicMock config: an
    auto-generated attribute reads as a configured value, so a typo'd knob name
    silently "works" in the test and is inert in production.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(reconciliation=types.SimpleNamespace(**leaves))
    )


class TestTopicAnchorConfigResolver:
    """The single enable knob, resolved LIVE off the shared config object.

    There is exactly ONE knob here.  The amended task text item (1) explicitly
    DROPS the planned ``topic_anchored_canonical_kinds`` kind-prefix knob and
    the non-empty-``supersedes`` fallback, because ``metadata.canonical`` is now
    the single marker vocabulary — do not add a second resolver.
    """

    def test_returns_config_value_when_present_and_valid(self):
        memory_service = _memory_service_with_reconciliation(
            topic_anchored_recall_enabled=False
        )

        assert resolve_topic_anchor_enabled(memory_service) is False

    def test_returns_true_when_configured_true(self):
        memory_service = _memory_service_with_reconciliation(
            topic_anchored_recall_enabled=True
        )

        assert resolve_topic_anchor_enabled(memory_service) is True

    def test_falls_back_to_default_when_config_attr_missing(self):
        assert resolve_topic_anchor_enabled(
            types.SimpleNamespace()
        ) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_falls_back_to_default_when_reconciliation_is_none(self):
        memory_service = types.SimpleNamespace(config=types.SimpleNamespace(reconciliation=None))

        assert resolve_topic_anchor_enabled(memory_service) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_falls_back_to_default_when_config_is_none(self):
        memory_service = types.SimpleNamespace(config=None)

        assert resolve_topic_anchor_enabled(memory_service) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_falls_back_to_default_when_leaf_is_missing(self):
        memory_service = _memory_service_with_reconciliation()

        assert resolve_topic_anchor_enabled(memory_service) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_falls_back_to_default_when_leaf_is_non_bool(self):
        """int 1 and str 'true' are NOT bools — a non-measurement must not gate I/O."""
        for non_bool in (1, 0, 'true', 'false', None, [], {}):
            memory_service = _memory_service_with_reconciliation(
                topic_anchored_recall_enabled=non_bool
            )

            assert resolve_topic_anchor_enabled(memory_service) == _DEFAULT_TOPIC_ANCHOR_ENABLED, (
                f'{non_bool!r} must not read as a configured flag'
            )

    def test_falls_back_to_default_for_unspecced_async_mock(self):
        """An unspecced mock's attribute chain yields Mocks, not bools."""
        assert resolve_topic_anchor_enabled(AsyncMock()) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_falls_back_to_default_for_unspecced_magic_mock(self):
        assert resolve_topic_anchor_enabled(MagicMock()) == _DEFAULT_TOPIC_ANCHOR_ENABLED

    def test_default_is_enabled(self):
        """The transform ships ON — it is a no-op wherever coverage is absent."""
        assert _DEFAULT_TOPIC_ANCHOR_ENABLED is True
