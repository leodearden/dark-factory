"""Tests for scripts/normalize_topic_slugs.py — the corpus-wide topic
normalization migration (task 4878).

Sibling of ``test_retro_stamp_topics.py`` and deliberately built from the same
scaffolding: the shared by-path ``load_script_module`` (``scripts/`` is not a
package and is not on PYTHONPATH), the shared store-mutation-preflight contract
(``neutralise_fixture`` for the suite, ``deny`` for the guard's own tests), and
``AsyncMock`` doubles at the single I/O boundary.

What is DIFFERENT from that suite, and why it matters here: this script's sweep
is corpus-wide rather than id-bounded, so its enumeration, its collision
refusals and its residue verification all have to be pinned against injected
fakes.  No test in this file asserts a LIVE corpus count.  The corpus is live
and moving — the very history this task cites (98 -> 103 distinct non-conforming
values while the validator sat in warn mode) proves the number changes on its
own — so a test pinning it would be a doomed RED.  The baseline is a committed
constant and the tests pin the arithmetic against it, never the live number.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from _fm_helpers import load_script_module
from _store_mutation_preflight_contract import SENTINEL, deny, neutralise_fixture

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'normalize_topic_slugs.py'


_mod = load_script_module(SCRIPT_PATH, mod_name='normalize_topic_slugs')


_neutralise = neutralise_fixture(
    _mod,
    note="""``run(..., apply=True)`` runs the preflight once, before it scrolls
    (task 4878). This suite is deliberately MOCK-unit (``AsyncMock`` services
    and the stateful in-memory ``_FakeCorpus``, no live store).
    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.""",
)


# ---------------------------------------------------------------------------
# Record fixtures — the shape ``scroll_all_by_metadata`` actually yields
# ---------------------------------------------------------------------------

def _rec(memory_id: str, topic: object = None, **meta) -> dict:
    """One scroll-shaped record.

    ``topic=None`` means "no ``topic`` key at all", which is a genuinely
    different case from ``topic`` present-and-``None`` — the caller spells the
    latter explicitly via ``metadata``.
    """
    metadata: dict = dict(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def _by_reason(skips: list[dict], reason: str) -> list[dict]:
    return [s for s in skips if s.get('reason') == reason]


# ===========================================================================
# plan_renames — the pure planner
# ===========================================================================

class TestPlanRenames:
    """``plan_renames(records, *, project_id) -> (renames, skips)``.

    Pure: it takes plain scroll-shaped dicts and no service, so a wrong answer
    here can never be masked by a mock.  It decides what the migration WOULD
    do; nothing in it touches a store.
    """

    def test_non_conforming_topic_that_folds_yields_one_rename(self):
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'mem0_tombstone_coverage')], project_id='dark_factory',
        )
        assert skips == []
        assert len(renames) == 1
        (rename,) = renames
        assert rename.project_id == 'dark_factory'
        assert rename.memory_id == 'm1'
        assert rename.old_topic == 'mem0_tombstone_coverage'
        assert rename.new_topic == 'mem0-tombstone-coverage'

    def test_rename_is_frozen(self):
        """A plan row must not be mutated between planning and writing.

        The whole point of planning the corpus before touching it is that the
        artifact and the writes agree; a mutable row lets them diverge.
        """
        (rename,), _ = _mod.plan_renames(
            [_rec('m1', 'a_b')], project_id='dark_factory',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rename.new_topic = 'something-else'  # type: ignore[misc]

    def test_conforming_topic_is_neither_a_rename_nor_a_skip(self):
        """Already-good is not work, and a skip line would MISREPORT it.

        Most of the corpus conforms.  Filing each conforming record as a skip
        would bury the ~141 records that actually need attention under tens of
        thousands of lines saying "nothing to do" — and would make the skip
        buckets, which exist to surface coverage holes, unreadable.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'already-good'), _rec('m2', 'x1-2y')],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_record_without_a_topic_key_is_ignored_entirely(self):
        """No ``topic`` is retro_stamp_topics' job, not this script's.

        This sweep NORMALIZES the value of a topic that exists; stamping one
        onto a record that carries none is the bounded sibling's job, and
        claiming it here would double-report the same corpus gap in two
        artifacts with two different definitions of it.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1'), _rec('m2', canonical=True)], project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_topic_present_but_none_is_ignored_entirely(self):
        """A null ``topic`` is an absent topic, not a malformed one.

        ``derive_topic_slug(None)`` is ``None``, so without this case a null
        would land in the ``topic_unfoldable`` bucket and read as a value a
        human needs to adjudicate — when in fact there is nothing there.
        """
        renames, skips = _mod.plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': None}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_unfoldable_topic_is_a_skip_naming_the_raw_value(self):
        """No honest fold exists — report it, never guess one.

        The skip has to carry the RAW value: an operator resolving this by
        hand needs to see what is actually on the record, and the whole
        reason it is here is that the script refused to name a slug for it.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', '!!!')], project_id='dark_factory',
        )
        assert renames == []
        entries = _by_reason(skips, 'topic_unfoldable')
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['memory_id'] == 'm1'
        assert entry['topic'] == '!!!'
        assert 'new_topic' not in entry, 'a refusal must not carry a guessed slug'

    def test_over_long_topic_is_unfoldable_not_truncated(self):
        """The cap refuses rather than repairs — pinned at the planner too.

        ``derive_topic_slug`` already returns ``None`` here, but the planner
        is where that ``None`` becomes an operator-visible line rather than a
        silently dropped record.
        """
        over = 'a' * 101
        renames, skips = _mod.plan_renames(
            [_rec('m1', over)], project_id='dark_factory',
        )
        assert renames == []
        assert _by_reason(skips, 'topic_unfoldable')[0]['topic'] == over

    def test_non_str_topic_is_unfoldable(self):
        """Untrusted values come straight off live records."""
        renames, skips = _mod.plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': 42}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert _by_reason(skips, 'topic_unfoldable')[0]['topic'] == 42

    def test_many_records_sharing_one_legacy_slug_each_get_a_rename(self):
        """One legacy VALUE can sit on many records; each needs its own write.

        The distinct-value count is what the baseline delta grades (item 4);
        the per-record count is what the write loop does.  Conflating them is
        how a migration reports "1 topic normalized" and leaves 7 records
        behind.
        """
        renames, _ = _mod.plan_renames(
            [_rec(f'm{i}', 'shared_legacy') for i in range(8)],
            project_id='dark_factory',
        )
        assert len(renames) == 8
        assert {r.new_topic for r in renames} == {'shared-legacy'}

    def test_output_is_sorted_for_a_clean_diff(self):
        """Sorted by ``(project_id, new_topic, memory_id)``.

        Two dry runs over an unchanged corpus must produce byte-comparable
        artifacts, and scroll order is not stable.  Without this an operator
        diffing consecutive rehearsals sees churn that means nothing.
        """
        renames, _ = _mod.plan_renames(
            [
                _rec('m9', 'zeta_topic'),
                _rec('m2', 'alpha_topic'),
                _rec('m1', 'zeta_topic'),
                _rec('m5', 'alpha_topic'),
            ],
            project_id='dark_factory',
        )
        assert [(r.new_topic, r.memory_id) for r in renames] == [
            ('alpha-topic', 'm2'),
            ('alpha-topic', 'm5'),
            ('zeta-topic', 'm1'),
            ('zeta-topic', 'm9'),
        ]

    def test_skips_are_sorted_too(self):
        """Same argument as the renames: a rehearsal diff must be readable."""
        _, skips = _mod.plan_renames(
            [_rec('m9', '!!!'), _rec('m1', '???')], project_id='dark_factory',
        )
        assert [s['memory_id'] for s in _by_reason(skips, 'topic_unfoldable')] == [
            'm1', 'm9',
        ]

    def test_record_without_an_id_is_skipped_not_planned(self):
        """A row with no id cannot be written to; refuse it loudly.

        ``update_memory`` is addressed by memory id.  A record whose id is
        missing or empty would otherwise reach the write boundary and fail
        there, one row at a time, inside a report that already claimed it as
        planned work.
        """
        renames, skips = _mod.plan_renames(
            [{'id': '', 'created_at': None, 'metadata': {'topic': 'a_b'}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert len(_by_reason(skips, 'record_without_id')) == 1

    def test_empty_input_is_empty_output(self):
        assert _mod.plan_renames([], project_id='dark_factory') == ([], [])


# ===========================================================================
# Collision classification — the most dangerous case in the migration
# ===========================================================================

class TestSlugCollision:
    """A fold that lands on an OCCUPIED slug is refused, never merged.

    Renaming ``foo_bar`` to ``foo-bar`` when records already carry
    ``foo-bar`` is not a normalization — it MERGES two topic namespaces,
    changing cluster membership for records this migration never examined.

    Worse, ``topic`` is what SCOPES canonical uniqueness
    (``memory_service._check_canonical_uniqueness`` probes
    ``{'topic': T, 'canonical': True}``), so merging two clusters that each
    hold a canonical manufactures exactly the second canonical that invariant
    exists to prevent.  Under the shipped ``memory_metadata.enforce = false``
    the service seam WARN-fails open, so it would land silently — and this
    script would be the thing that created it.

    Refusing makes the verdict a property of the CORPUS rather than of scroll
    order.  That is the same posture ``retro_stamp_topics.stamp_one`` takes for
    ``stray_canonical_on_member``, for the same fail-closed-because-unattended
    -and-bulk reason.
    """

    def test_fold_onto_an_occupied_slug_is_refused(self):
        renames, skips = _mod.plan_renames(
            [
                _rec('m1', 'foo_bar'),
                _rec('m2', 'foo_bar'),
                _rec('incumbent', 'foo-bar'),
            ],
            project_id='dark_factory',
        )
        assert renames == [], 'the merge must not be planned'
        entries = _by_reason(skips, 'slug_collision')
        assert len(entries) == 1, 'one entry per collision, not one per record'
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'foo_bar'
        assert entry['new_topic'] == 'foo-bar'
        assert entry['colliding_count'] == 2
        assert entry['memory_ids'] == ['m1', 'm2']

    def test_the_incumbent_records_are_left_alone(self):
        """The incumbent already conforms — it is not the migration's business.

        It appears in the skip only as the REASON for the refusal.  Touching
        it would be the very namespace merge being refused, in the other
        direction.
        """
        renames, _ = _mod.plan_renames(
            [_rec('m1', 'foo_bar'), _rec('incumbent', 'foo-bar')],
            project_id='dark_factory',
        )
        assert 'incumbent' not in [r.memory_id for r in renames]

    def test_collision_is_scoped_per_project(self):
        """``foo-bar`` in reify does not block ``foo_bar`` in dark_factory.

        Topics are per-corpus; a cross-project "collision" is two unrelated
        clusters that happen to share a name, and refusing on it would strand
        real work for no reason.  ``plan_renames`` sees one project at a time,
        so this is pinned by construction — the test guards the construction.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'foo_bar')], project_id='dark_factory',
        )
        assert [r.new_topic for r in renames] == ['foo-bar']
        assert skips == []

    def test_two_legacy_slugs_folding_to_one_target_is_a_collision(self):
        """``foo_bar`` and ``foo.bar`` both fold to ``foo-bar`` — refuse BOTH.

        Never resolved by scroll order.  Letting the first one through would
        make the outcome depend on which page the backend returned first, and
        would leave the second refused for a reason ("occupied") this script
        itself manufactured a moment earlier.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'foo_bar'), _rec('m2', 'foo.bar')],
            project_id='dark_factory',
        )
        assert renames == []
        entries = _by_reason(skips, 'slug_collision')
        assert len(entries) == 2, 'both source slugs are refused'
        assert sorted(e['old_topic'] for e in entries) == ['foo.bar', 'foo_bar']
        assert {e['new_topic'] for e in entries} == {'foo-bar'}

    def test_two_legacy_slugs_refusal_is_order_independent(self):
        """Reversing the input must not change the verdict for either slug."""
        forward, _ = _mod.plan_renames(
            [_rec('m1', 'foo_bar'), _rec('m2', 'foo.bar')],
            project_id='dark_factory',
        )
        reverse, _ = _mod.plan_renames(
            [_rec('m2', 'foo.bar'), _rec('m1', 'foo_bar')],
            project_id='dark_factory',
        )
        assert forward == reverse == []

    def test_an_unrelated_clean_rename_still_proceeds(self):
        """A refusal is scoped to its slug, not to the whole run.

        ~141 records need this migration; one unresolvable collision must not
        strand the rest.
        """
        renames, skips = _mod.plan_renames(
            [
                _rec('m1', 'foo_bar'),
                _rec('incumbent', 'foo-bar'),
                _rec('m3', 'clean_topic'),
            ],
            project_id='dark_factory',
        )
        assert [(r.memory_id, r.new_topic) for r in renames] == [
            ('m3', 'clean-topic'),
        ]
        assert len(_by_reason(skips, 'slug_collision')) == 1


class TestCanonicalCollision:
    """The severe case: a merge that would manufacture a second canonical.

    Distinct from ``slug_collision`` because the consequence is different in
    kind.  A plain namespace merge changes cluster membership; a merge across
    two clusters that EACH hold a ``canonical: true`` record violates the
    uniqueness invariant ``_check_canonical_uniqueness`` exists to hold — and
    does so through a seam that warn-fails OPEN under the shipped
    ``enforce = false``, so nothing downstream would object.

    Reported separately so an operator triaging the artifact can see at a
    glance which refusals are bookkeeping and which are corruption avoided.
    """

    def test_two_canonicals_across_the_merge_is_the_severe_outcome(self):
        renames, skips = _mod.plan_renames(
            [
                _rec('legacy-canon', 'foo_bar', canonical=True),
                _rec('legacy-member', 'foo_bar'),
                _rec('incumbent-canon', 'foo-bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert renames == []
        assert _by_reason(skips, 'slug_collision') == [], (
            'the severe outcome must not ALSO be filed as the mild one'
        )
        entries = _by_reason(skips, 'canonical_collision')
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'foo_bar'
        assert entry['new_topic'] == 'foo-bar'
        assert entry['old_canonical_ids'] == ['legacy-canon']
        assert entry['new_canonical_ids'] == ['incumbent-canon']

    def test_only_the_source_side_canonical_is_the_mild_outcome(self):
        """One canonical across the merge does not violate uniqueness.

        The merged cluster would still hold exactly one, so the refusal is
        the ordinary namespace-merge one — reported, but not escalated to the
        severe bucket, which would cry wolf and devalue the real cases.
        """
        _, skips = _mod.plan_renames(
            [
                _rec('legacy-canon', 'foo_bar', canonical=True),
                _rec('incumbent', 'foo-bar'),
            ],
            project_id='dark_factory',
        )
        assert len(_by_reason(skips, 'slug_collision')) == 1
        assert _by_reason(skips, 'canonical_collision') == []

    def test_only_the_target_side_canonical_is_the_mild_outcome(self):
        _, skips = _mod.plan_renames(
            [
                _rec('legacy', 'foo_bar'),
                _rec('incumbent-canon', 'foo-bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert len(_by_reason(skips, 'slug_collision')) == 1
        assert _by_reason(skips, 'canonical_collision') == []

    def test_two_legacy_slugs_each_canonical_is_also_severe(self):
        """The two-legacy-slugs collision escalates the same way.

        ``foo_bar`` and ``foo.bar`` each holding a canonical would merge into
        one cluster with two — the same violation, arrived at without any
        incumbent conforming record involved at all.
        """
        _, skips = _mod.plan_renames(
            [
                _rec('c1', 'foo_bar', canonical=True),
                _rec('c2', 'foo.bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert len(_by_reason(skips, 'canonical_collision')) == 2
        assert _by_reason(skips, 'slug_collision') == []

    def test_canonical_must_be_true_not_merely_truthy(self):
        """``canonical: 'false'`` is a string and must not read as canonical.

        Metadata comes off a live store and is not schema-enforced here; a
        truthiness test would turn any stray non-empty value into a phantom
        canonical and escalate a mild refusal to the severe bucket.
        """
        _, skips = _mod.plan_renames(
            [
                _rec('legacy', 'foo_bar', canonical='false'),
                _rec('incumbent', 'foo-bar', canonical='false'),
            ],
            project_id='dark_factory',
        )
        assert _by_reason(skips, 'canonical_collision') == []
        assert len(_by_reason(skips, 'slug_collision')) == 1


class TestCollisionOutcomesFailTheRun:
    """Both refusals are ERROR outcomes: an unresolved collision exits 1."""

    def test_both_are_error_outcomes(self):
        assert 'slug_collision' in _mod.ERROR_OUTCOMES
        assert 'canonical_collision' in _mod.ERROR_OUTCOMES

    def test_both_are_pre_seeded_skip_buckets(self):
        """Pre-seeded so an empty bucket renders as an explicit ``: 0``.

        An ABSENT bucket reads as "nothing was skipped", which is a different
        claim from "we looked and found nothing" — and for the severe bucket
        especially, the difference is the whole point.
        """
        assert 'slug_collision' in _mod.SKIP_BUCKETS
        assert 'canonical_collision' in _mod.SKIP_BUCKETS


# ===========================================================================
# enumerate_topic_bearing — the corpus boundary
# ===========================================================================

def _backend(
    records_by_category: dict[str, list[dict]],
    *,
    counts_by_category: dict[str, int] | None = None,
    recounts_by_category: dict[str, int] | None = None,
    call_log: list | None = None,
    scroll_calls: list | None = None,
    scroll_raises: dict[str, BaseException] | None = None,
) -> AsyncMock:
    """``Mem0Backend`` stand-in — the ``test_census_memory_metadata._backend`` shape.

    ``scroll_all_by_metadata`` is a REAL async generator dispatching on the
    ``category`` filter, not an AsyncMock returning a list: the thing under
    test consumes it with ``async for``, and a mock that returned a list would
    let a non-streaming implementation pass.

    *recounts_by_category* lets a test make the post-scroll count differ from
    the pre-scroll one, which is the only way to exercise the churn-versus-
    truncation distinction the bracket exists to draw.
    """
    log = call_log if call_log is not None else []
    scrolls = scroll_calls if scroll_calls is not None else []
    counts = counts_by_category or {
        c: len(r) for c, r in records_by_category.items()
    }
    recounts = recounts_by_category or {}
    raises = scroll_raises or {}
    seen_counts: dict[str, int] = {}

    async def _scroll_all_by_metadata(scope, filters, **kwargs):
        category = filters['category']
        log.append(('scroll', category))
        scrolls.append((scope, dict(filters), dict(kwargs)))
        for record in records_by_category.get(category, []):
            yield dict(record)
        if category in raises:
            raise raises[category]

    async def _count_by_metadata(scope, filters):
        category = filters['category']
        log.append(('count', category))
        seen = seen_counts.get(category, 0)
        seen_counts[category] = seen + 1
        if seen and category in recounts:
            return recounts[category]
        return counts.get(category, 0)

    backend = AsyncMock()
    backend.config = MagicMock()
    backend.config.mem0.collection_prefix = 'fused'
    backend.scroll_all_by_metadata = _scroll_all_by_metadata
    backend.count_by_metadata = _count_by_metadata
    backend.count.return_value = sum(counts.values())
    return backend


def _service_with(backend: AsyncMock) -> AsyncMock:
    """A ``MemoryService`` double exposing the backend as ``.mem0``.

    Reaching the backend through the service (the
    ``consolidate_namespace_families.py`` pattern) rather than constructing one
    separately is what lets a single injected object serve both the reads and
    the writes — so an end-to-end test cannot accidentally scroll one store and
    write to another.
    """
    service = AsyncMock()
    service.mem0 = backend
    return service


class TestEnumerateTopicBearing:
    """``enumerate_topic_bearing(service, project_id, *, categories, page_size, max_pages)``.

    The corpus boundary.  Three measured constraints force its shape:

    1. ``scroll_all_by_metadata`` and ``scroll_by_metadata`` both raise
       ``ValueError`` on an EMPTY filter dict, explicitly "to avoid silently
       enumerating every memory in the collection" — so there is no direct
       all-records scroll at this seam, and the category partition is how
       ``census_memory_metadata.py`` already covers the whole collection with
       non-empty filters.
    2. A single capped ``get_memories_by_metadata`` silently drops most of a
       ~49k-entry collection, which is why ``scroll_all_by_metadata`` (real
       offset/next_offset pagination) exists at all.
    3. ``CategoryCensus`` deliberately never retains payloads, so the census
       cannot hand over the record ids a migration needs — but retaining only
       the NON-conforming residue keeps peak memory bounded by the residue
       rather than by the corpus.
    """

    @pytest.mark.asyncio
    async def test_walks_every_census_category(self):
        """No cell may be silently uncovered.

        The partition comes from the census's own category set, so the two
        scripts provably cover the same corpus rather than keeping two lists
        that drift.
        """
        scrolls: list = []
        backend = _backend({}, scroll_calls=scrolls)
        await _mod.enumerate_topic_bearing(_service_with(backend), 'dark_factory')
        scrolled = [filters['category'] for _scope, filters, _kw in scrolls]
        assert scrolled == list(_mod.census_categories())
        assert len(scrolled) == len(set(scrolled)), 'one scroll per category'
        assert len(scrolled) >= 6, 'all six categories, Mem0- and Graphiti-primary'

    @pytest.mark.asyncio
    async def test_uses_the_paginating_scroll_not_the_capped_list_read(self):
        """``get_memories_by_metadata`` would cover a prefix and call it whole.

        It returns a bare list with no ``total``, so a capped read is
        indistinguishable from a complete one at this seam — exactly the
        silent truncation this script must not build its coverage claim on.
        """
        backend = _backend({})
        service = _service_with(backend)
        await _mod.enumerate_topic_bearing(service, 'dark_factory')
        service.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forwards_page_size_and_max_pages(self):
        scrolls: list = []
        backend = _backend({}, scroll_calls=scrolls)
        await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', page_size=250, max_pages=7,
        )
        _scope, _filters, kwargs = scrolls[0]
        assert kwargs['page_size'] == 250
        assert kwargs['max_pages'] == 7

    @pytest.mark.asyncio
    async def test_retains_only_the_non_conforming_residue(self):
        """Peak memory is bounded by the RESIDUE, not by the corpus.

        10k conforming records in and nothing comes out.  This is the property
        that makes a corpus-wide sweep affordable at all, and the reason this
        script cannot simply reuse ``CategoryCensus`` (which retains nothing
        and so cannot supply the ids) nor accumulate everything (which would
        hold the whole ~49.4k-record corpus).
        """
        category = _mod.census_categories()[0]
        records = [_rec(f'm{i}', 'conforming-topic') for i in range(10_000)]
        backend = _backend({category: records})
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        assert result['records'] == []

    @pytest.mark.asyncio
    async def test_retains_the_triple_for_a_non_conforming_record(self):
        """``(id, topic, canonical)`` — everything the planner needs, nothing more."""
        category = _mod.census_categories()[0]
        backend = _backend({category: [
            _rec('m1', 'legacy_topic', canonical=True),
            _rec('m2', 'conforming-topic'),
            _rec('m3'),
        ]})
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        assert [r['id'] for r in result['records']] == ['m1']
        assert result['records'][0]['metadata'] == {
            'topic': 'legacy_topic', 'canonical': True,
        }

    @pytest.mark.asyncio
    async def test_each_cell_is_bracketed_by_a_count_before_and_after(self):
        """Two independent reads, or a shortfall is undetectable.

        Counting BEFORE is what makes an under-enumerated scroll detectable at
        all; counting AFTER brackets the scan against a LIVE corpus, so a
        scroll that matches the recount saw churn rather than truncation.
        One read cannot tell those apart.
        """
        category = _mod.census_categories()[0]
        log: list = []
        backend = _backend({category: [_rec('m1', 'a_b')]}, call_log=log)
        await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        assert log == [('count', category), ('scroll', category), ('count', category)]

    @pytest.mark.asyncio
    async def test_a_matching_bracket_is_complete(self):
        category = _mod.census_categories()[0]
        backend = _backend({category: [_rec('m1', 'a_b')]})
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        cell = result['coverage'][category]
        assert cell == {
            'expected': 1, 'scrolled': 1, 'recount': 1, 'delta': 0, 'complete': True,
        }
        assert result['complete'] is True

    @pytest.mark.asyncio
    async def test_a_scroll_matching_neither_count_is_under_enumerated(self):
        """A shortfall is REPORTED, never treated as complete.

        Scrolled 1 where both counts said 5: the sweep saw a fifth of that
        cell.  Calling that complete would let the migration report a clean
        run over a corpus it barely looked at.
        """
        category = _mod.census_categories()[0]
        backend = _backend(
            {category: [_rec('m1', 'a_b')]}, counts_by_category={category: 5},
        )
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        cell = result['coverage'][category]
        assert cell['expected'] == 5
        assert cell['scrolled'] == 1
        assert cell['complete'] is False
        assert result['complete'] is False
        assert len(_by_reason(result['skips'], 'under_enumerated')) == 1

    @pytest.mark.asyncio
    async def test_a_scroll_agreeing_with_the_recount_is_churn_not_truncation(self):
        """Count moved 2 -> 1 while scrolling and the scroll saw 1: complete.

        The corpus is live; orchestrators write while this runs.  Grading only
        against the PRE-scroll count would report ordinary churn as a coverage
        hole, and an operator who learns to ignore that bucket stops reading
        the real ones.
        """
        category = _mod.census_categories()[0]
        backend = _backend(
            {category: [_rec('m1', 'a_b')]},
            counts_by_category={category: 2},
            recounts_by_category={category: 1},
        )
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=[category],
        )
        cell = result['coverage'][category]
        assert (cell['expected'], cell['scrolled'], cell['recount']) == (2, 1, 1)
        assert cell['complete'] is True
        assert result['complete'] is True
        assert _by_reason(result['skips'], 'under_enumerated') == []

    @pytest.mark.asyncio
    async def test_page_budget_exhausted_is_an_explicit_incomplete_outcome(self):
        """NEVER swallowed into a clean empty result.

        ``ScrollPageBudgetExhausted`` means the backend gave up paging. A
        migration that caught it and moved on would report a completed sweep
        over a corpus it had only partly enumerated — and then the residue
        probe would find the legacy slug still populated and blame the writes.
        """
        categories = list(_mod.census_categories()[:2])
        first, second = categories
        backend = _backend(
            {first: [_rec('m1', 'a_b')], second: [_rec('m2', 'c_d')]},
            scroll_raises={first: _mod.ScrollPageBudgetExhausted('budget')},
        )
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=categories,
        )
        assert result['complete'] is False
        entries = _by_reason(result['skips'], 'scroll_budget_exhausted')
        assert len(entries) == 1
        assert entries[0]['category'] == first
        assert entries[0]['project_id'] == 'dark_factory'
        assert 'budget' in entries[0]['error']
        assert result['coverage'][first]['complete'] is False

    @pytest.mark.asyncio
    async def test_an_exhausted_cell_does_not_abort_the_other_cells(self):
        """One dead cell must not cost the coverage of the other five.

        The report is more useful naming which cell failed than crashing
        before any of it is written.
        """
        categories = list(_mod.census_categories()[:2])
        first, second = categories
        backend = _backend(
            {first: [_rec('m1', 'a_b')], second: [_rec('m2', 'c_d')]},
            scroll_raises={first: _mod.ScrollPageBudgetExhausted('budget')},
        )
        result = await _mod.enumerate_topic_bearing(
            _service_with(backend), 'dark_factory', categories=categories,
        )
        assert {r['id'] for r in result['records']} == {'m1', 'm2'}
        assert result['coverage'][second]['complete'] is True

    @pytest.mark.asyncio
    async def test_an_incomplete_enumeration_is_an_error_outcome(self):
        """A partial sweep must not exit 0 and read as a finished migration."""
        assert 'under_enumerated' in _mod.ERROR_OUTCOMES
        assert 'scroll_budget_exhausted' in _mod.ERROR_OUTCOMES
        assert 'under_enumerated' in _mod.SKIP_BUCKETS
        assert 'scroll_budget_exhausted' in _mod.SKIP_BUCKETS

    @pytest.mark.asyncio
    async def test_the_scope_carries_the_project(self):
        """Each scroll is scoped to the project being enumerated."""
        scrolls: list = []
        backend = _backend({}, scroll_calls=scrolls)
        await _mod.enumerate_topic_bearing(_service_with(backend), 'reify')
        scope, _filters, _kw = scrolls[0]
        assert scope.project_id == 'reify'


class TestCensusCategoriesAreImportedNotRestated:
    """The partition has ONE home: ``census_memory_metadata.CENSUS_CATEGORIES``.

    Two lists kept in sync by prose is exactly the drift INV-5 forbids, and
    here the consequence is a silently unenumerated slice of the corpus — the
    failure mode hardest to notice, because the report would look clean.
    """

    def test_categories_match_the_census_module(self):
        census = _mod.load_census_module()
        assert list(_mod.census_categories()) == [
            c.value for c in census.CENSUS_CATEGORIES
        ]

    def test_the_census_module_is_loaded_once(self):
        """``sys.modules``-first, memoized — re-executing hands back new classes."""
        assert _mod.load_census_module() is _mod.load_census_module()


# ===========================================================================
# rename_one — the single write boundary
# ===========================================================================

_UNSET = object()


def _service(
    *,
    record: object = _UNSET,
    update_response: object = _UNSET,
) -> AsyncMock:
    """A stateless ``MemoryService`` double for the write boundary.

    Children are configured through ``.return_value`` rather than reassigned
    to fresh ``AsyncMock``s, because a reassigned child stops propagating into
    the parent's ``mock_calls`` — and the ORDER of the live re-read relative to
    the write is one of the things this suite has to pin.
    """
    service = AsyncMock()
    service.get_memory_by_id.return_value = (
        {'id': 'm1', 'metadata': {'topic': 'legacy_topic'}}
        if record is _UNSET else record
    )
    if update_response is _UNSET:
        service.update_memory.side_effect = lambda **kwargs: {
            'status': 'updated',
            'store': 'mem0',
            'id': kwargs['memory_id'],
            'content_amended': False,
            'metadata_patched': True,
        }
    else:
        service.update_memory.return_value = update_response
    return service


def _rename(**overrides) -> object:
    fields = {
        'project_id': 'dark_factory',
        'memory_id': 'm1',
        'old_topic': 'legacy_topic',
        'new_topic': 'legacy-topic',
    }
    fields.update(overrides)
    return _mod.Rename(**fields)


def _call_names(service: AsyncMock) -> list[str]:
    return [name for name, _a, _kw in service.mock_calls if name]


class TestRenameOne:
    """``rename_one(memory_service, rename, *, apply)``.

    The only function here that touches a store.  Order is load-bearing and
    copied from ``retro_stamp_topics.stamp_one``: live re-read, decide, then
    write.  The re-read is not a formality — the plan's ids come off a scroll
    that may be minutes old on a corpus orchestrators are writing to, so a
    record consolidated away since must become a report line rather than a
    Qdrant ``set_payload`` that acknowledges a write to nothing.
    """

    @pytest.mark.asyncio
    async def test_reads_live_before_it_writes(self):
        service = _service()
        await _mod.rename_one(service, _rename(), apply=True)
        assert _call_names(service) == ['get_memory_by_id', 'update_memory']

    @pytest.mark.asyncio
    async def test_the_write_is_metadata_only_and_carries_only_topic(self):
        """A migration must not touch content, and must not touch a sibling key.

        ``metadata_mode='merge'`` with a patch containing exactly ``topic``
        leaves every other metadata key alone.  A wider patch would let a
        normalization silently clobber ``canonical``, ``supersedes`` or the
        consolidation bookkeeping on ~141 records at once.
        """
        service = _service()
        await _mod.rename_one(service, _rename(), apply=True)
        service.update_memory.assert_awaited_once()
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['memory_id'] == 'm1'
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['metadata_patch'] == {'topic': 'legacy-topic'}
        assert kwargs['metadata_mode'] == 'merge'
        assert 'content' not in kwargs
        assert 'messages' not in kwargs

    @pytest.mark.asyncio
    async def test_the_write_is_attributed_to_this_sweep(self):
        """``_source`` / ``reason`` so the amendment-storm alarm can tell.

        A bulk run under the default ``mcp_tool`` source looks exactly like
        the runaway rewrite that alarm exists to catch.
        """
        service = _service()
        await _mod.rename_one(service, _rename(), apply=True)
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['_source'] == _mod.WRITE_SOURCE == 'normalize_topic_slugs'
        assert kwargs['reason'] == _mod.WRITE_REASON
        assert '4878' in _mod.WRITE_REASON

    @pytest.mark.asyncio
    async def test_no_delete_is_ever_issued(self):
        """A topic rename is a metadata patch, never a delete-and-recreate.

        Re-creating would mint a new memory id, orphaning every reference to
        the old one and destroying ``created_at`` — the field consolidation
        uses to pick a canonical.
        """
        service = _service()
        await _mod.rename_one(service, _rename(), apply=True)
        service.delete_memory.assert_not_awaited()
        assert 'delete_memory' not in _call_names(service)

    @pytest.mark.asyncio
    async def test_a_successful_write_is_renamed(self):
        service = _service()
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'renamed'
        assert result['memory_id'] == 'm1'
        assert result['old_topic'] == 'legacy_topic'
        assert result['new_topic'] == 'legacy-topic'
        assert result['response']['status'] == 'updated'

    @pytest.mark.asyncio
    async def test_dry_run_reads_and_decides_but_never_writes(self):
        """The rehearsal must exercise the SAME decision path as the apply.

        A dry run that skipped the live re-read would rehearse a different
        function from the one ``--apply`` runs, and its report would be a
        prediction rather than a rehearsal.
        """
        service = _service()
        result = await _mod.rename_one(service, _rename(), apply=False)
        assert result['outcome'] == 'would_rename'
        service.get_memory_by_id.assert_awaited_once()
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_vanished_record_is_memory_not_found(self):
        """Consolidated away between the scroll and the write.

        Measured precedent: retro_stamp's gate 3036 id ``19705df4`` no longer
        resolves. Writing to it anyway would be acknowledged by the backend
        and would appear in the report as a successful rename.
        """
        service = _service(record=None)
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'memory_not_found'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_topic_that_moved_since_the_plan_is_refused(self):
        """Somebody else rewrote it — this plan row is stale, so do not write.

        Applying anyway would overwrite a live value with a fold of a value
        that no longer exists, which is a data loss the report would score as
        a success.
        """
        service = _service(record={'id': 'm1', 'metadata': {'topic': 'something_else'}})
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'topic_moved_since_plan'
        assert result['existing_topic'] == 'something_else'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_already_normalized_costs_zero_writes(self):
        """A second run plans zero WRITES, not merely zero net effect.

        A no-op ``update_memory`` would still journal a write op, still count
        toward the amendment-storm alarm, and would inflate the report's
        renamed count with records that gained nothing.
        """
        service = _service(record={'id': 'm1', 'metadata': {'topic': 'legacy-topic'}})
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'already_normalized'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_error_envelope_is_update_failed_not_renamed(self):
        """``update_memory`` reports some rejections by RETURNING, not raising.

        A caller guarding only against exceptions would score a refused write
        as a rename — and then the residue probe would find the legacy slug
        still populated with no explanation in the report.
        """
        service = _service(update_response={
            'error': 'memory not found', 'error_type': 'MemoryNotFound',
        })
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'update_failed'
        assert result['error_type'] == 'MemoryNotFound'
        assert result['error'] == 'memory not found'
        assert result['response'] == {
            'error': 'memory not found', 'error_type': 'MemoryNotFound',
        }

    @pytest.mark.asyncio
    async def test_a_raising_read_is_one_row_not_an_unwound_sweep(self):
        """Every await is individually guarded.

        One backend hiccup must cost one report row, not the whole run — the
        artifact is written after the loop, so an exception escaping here
        would discard everything the sweep had already learned.
        """
        service = _service()
        service.get_memory_by_id.side_effect = RuntimeError('qdrant timeout')
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'rename_error'
        assert 'RuntimeError' in result['error']
        assert 'qdrant timeout' in result['error']

    @pytest.mark.asyncio
    async def test_a_raising_write_is_one_row_too(self):
        service = _service()
        service.update_memory.side_effect = RuntimeError('set_payload failed')
        result = await _mod.rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'rename_error'
        assert 'set_payload failed' in result['error']

    @pytest.mark.asyncio
    async def test_the_memory_id_is_never_changed(self):
        """Whatever the outcome, the row reports the id it was handed.

        The report is how an operator re-addresses a failed row; an id that
        drifted between the plan and the artifact makes it unusable.
        """
        for record in (None, {'id': 'm1', 'metadata': {'topic': 'other_x'}}):
            result = await _mod.rename_one(
                _service(record=record), _rename(), apply=True,
            )
            assert result['memory_id'] == 'm1'

    @pytest.mark.asyncio
    async def test_every_outcome_carries_what_it_needs_to_be_actionable(self):
        """No follow-up query should be needed to act on a report row."""
        result = await _mod.rename_one(_service(), _rename(), apply=True)
        for key in ('project_id', 'memory_id', 'old_topic', 'new_topic', 'outcome'):
            assert key in result


class TestRenameOneOutcomesAreGraded:
    """The write-boundary failures fail the run."""

    def test_failure_outcomes_are_errors(self):
        for outcome in ('memory_not_found', 'update_failed', 'rename_error',
                        'topic_moved_since_plan'):
            assert outcome in _mod.ERROR_OUTCOMES, outcome

    def test_success_outcomes_are_not_errors(self):
        for outcome in ('renamed', 'would_rename', 'already_normalized'):
            assert outcome not in _mod.ERROR_OUTCOMES, outcome

    def test_failure_outcomes_are_pre_seeded_buckets(self):
        for outcome in ('memory_not_found', 'update_failed', 'rename_error',
                        'topic_moved_since_plan'):
            assert outcome in _mod.SKIP_BUCKETS, outcome


# ===========================================================================
# verify_old_slugs_drained — scope item 3, the post-apply verification
# ===========================================================================

def _result(outcome: str, **overrides) -> dict:
    row = {
        'project_id': 'dark_factory',
        'memory_id': 'm1',
        'old_topic': 'legacy_topic',
        'new_topic': 'legacy-topic',
        'outcome': outcome,
    }
    row.update(overrides)
    return row


class TestVerifyOldSlugsDrained:
    """``verify_old_slugs_drained(memory_service, results, skips)``.

    Scope item 3: "verify via ``count_memories_by_metadata`` on each OLD slug
    expecting 0".  Adapted from ``retro_stamp_topics._probe_legacy_topic_
    residue``, with ONE inversion a reader diffing the two must not mistake
    for a copy bug: there, the sweep is id-bounded, so records outside it can
    legitimately still carry the legacy spelling and a non-zero residue is
    expected-and-merely-reported.  HERE the sweep is corpus-wide, so after
    ``--apply`` a non-zero residue means a slug was STRANDED — an error.
    """

    @pytest.mark.asyncio
    async def test_one_probe_per_distinct_project_and_slug(self):
        """Not one per record — the probe is bounded by SLUGS, not by writes.

        Eight records sharing one legacy value is one question, asked once.
        Asking it eight times would multiply a bulk sweep's read load by its
        own target count for no additional information.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 0
        results = [
            _result('renamed', memory_id=f'm{i}') for i in range(8)
        ] + [_result('renamed', memory_id='x1', old_topic='other_slug',
                     new_topic='other-slug')]
        await _mod.verify_old_slugs_drained(service, results, {})
        probed = [
            call.args for call in service.count_memories_by_metadata.await_args_list
        ]
        assert probed == [
            ('dark_factory', {'topic': 'legacy_topic'}),
            ('dark_factory', {'topic': 'other_slug'}),
        ]

    @pytest.mark.asyncio
    async def test_a_drained_slug_files_nothing(self):
        """Zero IS the expected outcome of an apply run — silence is correct.

        Filing a line per successfully-drained slug would bury the stranded
        ones, which are the only reason this pass exists.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 0
        skips: dict = {}
        await _mod.verify_old_slugs_drained(service, [_result('renamed')], skips)
        assert skips.get('legacy_slug_residue', []) == []

    @pytest.mark.asyncio
    async def test_a_stranded_slug_is_reported_with_its_count(self):
        """Corpus-wide means nothing should be left — a remainder is an error."""
        service = _service()
        service.count_memories_by_metadata.return_value = 3
        skips: dict = {}
        await _mod.verify_old_slugs_drained(service, [_result('renamed')], skips)
        entries = skips['legacy_slug_residue']
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['legacy_topic'] == 'legacy_topic'
        assert entry['new_topic'] == 'legacy-topic'
        assert entry['residue_count'] == 3

    @pytest.mark.asyncio
    async def test_a_clean_rehearsal_reports_zero_residue(self):
        """Dry-run rows STILL carry the legacy value, so subtract them first.

        Without this the rehearsal would report every record it is about to
        fix as residue — a false alarm on exactly the run an operator uses to
        decide whether to apply at all.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 8
        skips: dict = {}
        results = [_result('would_rename', memory_id=f'm{i}') for i in range(8)]
        await _mod.verify_old_slugs_drained(service, results, skips)
        assert skips.get('legacy_slug_residue', []) == []

    @pytest.mark.asyncio
    async def test_a_rehearsal_still_surfaces_records_it_did_not_plan(self):
        """8 rehearsed out of 10 counted: 2 records this run never saw.

        That is a genuine coverage signal — an under-enumerated cell, or a
        record whose topic was refused — and subtracting the rehearsed rows is
        precisely what makes it visible instead of drowned.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 10
        skips: dict = {}
        results = [_result('would_rename', memory_id=f'm{i}') for i in range(8)]
        await _mod.verify_old_slugs_drained(service, results, skips)
        assert skips['legacy_slug_residue'][0]['residue_count'] == 2

    @pytest.mark.asyncio
    async def test_the_subtraction_clamps_at_zero(self):
        """More rehearsed than counted is churn, not negative residue.

        A record consolidated away between the plan and the probe makes the
        count smaller than the rehearsal. Reporting -1 residue would be
        nonsense an operator has to decode.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 1
        skips: dict = {}
        results = [_result('would_rename', memory_id=f'm{i}') for i in range(5)]
        await _mod.verify_old_slugs_drained(service, results, skips)
        assert skips.get('legacy_slug_residue', []) == []

    @pytest.mark.asyncio
    async def test_a_failed_write_is_not_subtracted_in_an_apply_run(self):
        """Only ``would_rename`` rows still carry the legacy value by design.

        A row whose write FAILED also still carries it — and that is genuine
        residue, correctly left in.
        """
        service = _service()
        service.count_memories_by_metadata.return_value = 1
        skips: dict = {}
        await _mod.verify_old_slugs_drained(
            service, [_result('update_failed')], skips,
        )
        assert skips['legacy_slug_residue'][0]['residue_count'] == 1

    @pytest.mark.asyncio
    async def test_a_raising_probe_files_unknown_not_zero(self):
        """Unknown residue is NOT zero residue — a bucket entry says so.

        Swallowing the failure would let the report claim a drained corpus on
        the strength of a question that was never answered.
        """
        service = _service()
        service.count_memories_by_metadata.side_effect = RuntimeError('qdrant down')
        skips: dict = {}
        await _mod.verify_old_slugs_drained(service, [_result('renamed')], skips)
        entry = skips['legacy_slug_residue'][0]
        assert 'RuntimeError' in entry['error']
        assert 'residue_count' not in entry, 'an unanswered probe has no count'
        assert 'not the same as zero' in entry['note']

    @pytest.mark.asyncio
    async def test_rows_that_did_not_rename_are_not_probed(self):
        """A refused row never moved a slug, so there is nothing to drain."""
        service = _service()
        service.count_memories_by_metadata.return_value = 0
        await _mod.verify_old_slugs_drained(
            service,
            [_result('memory_not_found'), _result('already_normalized')],
            {},
        )
        service.count_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_probes_are_ordered_for_a_clean_diff(self):
        service = _service()
        service.count_memories_by_metadata.return_value = 0
        results = [
            _result('renamed', project_id='reify', old_topic='z_slug'),
            _result('renamed', old_topic='b_slug'),
            _result('renamed', old_topic='a_slug'),
        ]
        await _mod.verify_old_slugs_drained(service, results, {})
        probed = [
            call.args for call in service.count_memories_by_metadata.await_args_list
        ]
        assert probed == [
            ('dark_factory', {'topic': 'a_slug'}),
            ('dark_factory', {'topic': 'b_slug'}),
            ('reify', {'topic': 'z_slug'}),
        ]

    def test_a_stranded_slug_fails_the_run(self):
        assert 'legacy_slug_residue' in _mod.ERROR_OUTCOMES
        assert 'legacy_slug_residue' in _mod.SKIP_BUCKETS


# ===========================================================================
# pair_gate_blocks + the gate writer — the two-store lockstep
# ===========================================================================
#
# ``metadata.x_recon_consolidation_gate`` is a Tier-C block on TASK metadata,
# not Mem0 metadata: a genuinely different store, reached through the MCP
# server rather than through a backend call.  Renaming the memories without
# moving that block leaves the gate's closure scroll pointed at a slug no
# record carries any more — the uncloseable-gate trap re-created at a NEW
# address, which is strictly worse than leaving the snake_case slug alone.
#
# So the halves are planned as a pair and refused as a pair.  Every test below
# exists to pin one edge of that atomicity.

def _gate_task(
    task_id: str,
    topic: str,
    *,
    project_id: str = 'dark_factory',
    project_root: str = '/repo',
    **block_extras,
) -> dict:
    """One gate task as the MCP ``get_task`` read hands it over.

    ``block_extras`` seed the SIBLING keys inside the block (``provenance``,
    ``considered_and_kept``, ...) — the ones the patch must leave untouched.
    """
    block = {'topic': topic}
    block.update(block_extras)
    return {
        'id': task_id,
        'project_id': project_id,
        'project_root': project_root,
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            _mod.GATE_METADATA_KEY: block,
        },
    }


def _stateful_service(records: dict[str, dict]) -> MagicMock:
    """A ``MemoryService`` double that REMEMBERS what was written to it.

    The gate lockstep's most important property — that a refused gate patch
    leaves the memory half at its ORIGINAL value — is a statement about the
    store's end state, not about a call list.  A stateless double can only
    show the calls; it cannot show that the corpus came back to where it
    started, which is the whole claim.
    """
    service = MagicMock()

    async def _get(*, project_id, memory_id):
        record = records.get(memory_id)
        return None if record is None else {
            'id': memory_id, 'metadata': dict(record['metadata']),
        }

    async def _update(*, memory_id, project_id, metadata_patch, **kwargs):
        record = records.get(memory_id)
        if record is None:
            return {'error_type': 'not_found', 'error': memory_id}
        record['metadata'].update(metadata_patch)
        return {'status': 'updated', 'id': memory_id, 'metadata_patched': True}

    service.get_memory_by_id = AsyncMock(side_effect=_get)
    service.update_memory = AsyncMock(side_effect=_update)
    return service


def _store(*topics: tuple[str, str]) -> dict[str, dict]:
    return {mid: {'metadata': {'topic': topic}} for mid, topic in topics}


def _client(*, result: object = None, error: Exception | None = None) -> MagicMock:
    """A ``FusedMemoryClient`` double.  Injected so no test opens a socket."""
    client = MagicMock()
    if error is not None:
        client.call_tool = AsyncMock(side_effect=error)
    else:
        client.call_tool = AsyncMock(
            return_value=result if result is not None
            else {'id': '4220', 'updated': True, 'message': 'ok'}
        )
    return client


class TestPairGateBlocks:
    """``pair_gate_blocks(renames, gate_tasks) -> (groups, skips)``.

    Pure.  It decides WHICH renames travel with WHICH gate task, before
    anything is written, so the pairing is reproducible from the plan alone.
    """

    def test_a_gate_on_a_renamed_slug_is_paired_with_it(self):
        renames = [_mod.Rename('dark_factory', 'm1', 'mem0_tombstone_coverage',
                               'mem0-tombstone-coverage')]
        gates = [_gate_task('4220', 'mem0_tombstone_coverage')]

        groups, skips = _mod.pair_gate_blocks(renames, gates)

        assert len(groups) == 1
        assert groups[0].old_topic == 'mem0_tombstone_coverage'
        assert groups[0].new_topic == 'mem0-tombstone-coverage'
        assert groups[0].gate_task_id == '4220'
        assert skips == []

    def test_the_patch_moves_the_topic_and_touches_no_sibling_key(self):
        provenance = {'report_run': 'r1', 'observed_members': ['a'],
                      'detector': 'd', 'authoritative': False}
        kept = [{'id': 'x', 'why': 'peer'}]
        gates = [_gate_task('4220', 'gate_topic', provenance=provenance,
                            considered_and_kept=kept)]
        renames = [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')]

        groups, _skips = _mod.pair_gate_blocks(renames, gates)

        assert groups[0].gate_block == {
            'topic': 'gate-topic',
            'provenance': provenance,
            'considered_and_kept': kept,
        }

    def test_the_patch_does_not_mutate_the_source_task(self):
        gates = [_gate_task('4220', 'gate_topic', provenance={'a': 1})]
        renames = [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')]

        _mod.pair_gate_blocks(renames, gates)

        assert gates[0]['metadata'][_mod.GATE_METADATA_KEY]['topic'] == 'gate_topic'

    def test_every_rename_on_the_slug_joins_the_group(self):
        renames = [
            _mod.Rename('dark_factory', f'm{i}', 'gate_topic', 'gate-topic')
            for i in range(4)
        ]
        groups, _skips = _mod.pair_gate_blocks(
            renames, [_gate_task('4220', 'gate_topic')])

        assert len(groups) == 1
        assert [r.memory_id for r in groups[0].renames] == ['m0', 'm1', 'm2', 'm3']

    def test_a_rename_with_no_gate_still_forms_a_group(self):
        """Most topics are not gates; those renames must proceed normally."""
        renames = [_mod.Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')]

        groups, skips = _mod.pair_gate_blocks(renames, [])

        assert len(groups) == 1
        assert groups[0].gate_task_id is None
        assert groups[0].gate_block is None
        assert skips == []

    def test_a_gate_is_matched_within_its_own_project(self):
        renames = [_mod.Rename('reify', 'm1', 'gate_topic', 'gate-topic')]
        gates = [_gate_task('4220', 'gate_topic', project_id='dark_factory')]

        groups, skips = _mod.pair_gate_blocks(renames, gates)

        assert groups[0].gate_task_id is None
        assert _by_reason(skips, 'orphan_gate_topic')

    def test_a_gate_whose_slug_matches_no_record_is_an_orphan_not_a_silence(self):
        gates = [_gate_task('4774', 'ghost_topic')]

        groups, skips = _mod.pair_gate_blocks([], gates)

        assert groups == []
        orphans = _by_reason(skips, 'orphan_gate_topic')
        assert len(orphans) == 1
        assert orphans[0]['gate_task_id'] == '4774'
        assert orphans[0]['gate_topic'] == 'ghost_topic'
        assert orphans[0]['project_id'] == 'dark_factory'

    def test_a_gate_already_on_a_conforming_slug_is_not_an_orphan(self):
        """Nothing to migrate is not the same fact as a dangling gate."""
        groups, skips = _mod.pair_gate_blocks([], [_gate_task('4220', 'good-topic')])

        assert groups == []
        assert skips == []

    def test_a_gate_task_without_a_block_is_ignored(self):
        task = {'id': '9', 'project_id': 'dark_factory', 'metadata': {}}

        groups, skips = _mod.pair_gate_blocks([], [task])

        assert (groups, skips) == ([], [])

    def test_groups_are_sorted_for_a_clean_diff(self):
        renames = [
            _mod.Rename('reify', 'm9', 'z_topic', 'z-topic'),
            _mod.Rename('dark_factory', 'm2', 'b_topic', 'b-topic'),
            _mod.Rename('dark_factory', 'm1', 'a_topic', 'a-topic'),
        ]
        groups, _skips = _mod.pair_gate_blocks(renames, [])

        assert [(g.project_id, g.new_topic) for g in groups] == [
            ('dark_factory', 'a-topic'),
            ('dark_factory', 'b-topic'),
            ('reify', 'z-topic'),
        ]

    def test_the_group_is_frozen(self):
        groups, _skips = _mod.pair_gate_blocks(
            [_mod.Rename('dark_factory', 'm1', 'a_topic', 'a-topic')], [])
        assert dataclasses.is_dataclass(groups[0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            groups[0].new_topic = 'other'  # type: ignore[misc]


class TestAmbiguousGate:
    """A legacy slug carrying MORE THAN ONE consolidation gate is refused.

    A ``GateGroup`` moves exactly one gate.  Moving one of several would strand
    the rest on the vacated slug — the uncloseable-gate trap again — so the
    slug's renames are withheld and both stores stay on the legacy value, the
    posture ``slug_collision`` already takes.
    """

    @staticmethod
    def _gates_on(slug: str, *task_ids: str) -> list[dict]:
        return [_gate_task(task_id, slug) for task_id in task_ids]

    def test_a_doubly_gated_slug_forms_no_group_and_is_refused_once(self):
        renames = [
            _mod.Rename('dark_factory', 'm2', 'shared_gate', 'shared-gate'),
            _mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate'),
        ]

        groups, skips = _mod.pair_gate_blocks(
            renames, self._gates_on('shared_gate', '7718', '7011'))

        assert groups == []
        (entry,) = _by_reason(skips, 'gate_ambiguous')
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'shared_gate'
        assert entry['new_topic'] == 'shared-gate'
        assert entry['gate_task_ids'] == ['7011', '7718']
        assert entry['memory_ids'] == ['m1', 'm2']

    def test_neither_gate_is_reported_as_an_orphan(self):
        """They are matched to live records; "matched to no record" would be false."""
        _groups, skips = _mod.pair_gate_blocks(
            [_mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')],
            self._gates_on('shared_gate', '7011', '7718'))

        assert _by_reason(skips, 'orphan_gate_topic') == []

    def test_the_refusal_is_per_slug(self):
        renames = [
            _mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate'),
            _mod.Rename('dark_factory', 'm2', 'plain_topic', 'plain-topic'),
            _mod.Rename('dark_factory', 'm3', 'single_gate', 'single-gate'),
        ]
        single = _gate_task('4220', 'single_gate')

        groups, _skips = _mod.pair_gate_blocks(
            renames, [*self._gates_on('shared_gate', '7011', '7718'), single])

        assert [(g.old_topic, g.gate_task_id) for g in groups] == [
            ('plain_topic', None),
            ('single_gate', '4220'),
        ]
        unaffected, _ = _mod.pair_gate_blocks(
            [r for r in renames if r.old_topic != 'shared_gate'], [single])
        assert groups == unaffected

    def test_the_verdict_does_not_depend_on_gate_order(self):
        renames = [_mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')]
        gates = self._gates_on('shared_gate', '7011', '7718')

        for ordered in (gates, list(reversed(gates))):
            groups, skips = _mod.pair_gate_blocks(renames, ordered)

            assert groups == []
            (entry,) = _by_reason(skips, 'gate_ambiguous')
            assert entry['gate_task_ids'] == ['7011', '7718']

    def test_gates_in_different_projects_are_not_ambiguous(self):
        """Gates are keyed per project, as the closure scroll that reads them is."""
        renames = [_mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')]
        gates = [
            _gate_task('4220', 'shared_gate', project_id='dark_factory'),
            _gate_task('6852', 'shared_gate', project_id='reify'),
        ]

        groups, skips = _mod.pair_gate_blocks(renames, gates)

        assert [g.gate_task_id for g in groups] == ['4220']
        assert _by_reason(skips, 'gate_ambiguous') == []

    def test_the_markdown_artifact_names_every_gate(self):
        _groups, skips = _mod.pair_gate_blocks(
            [_mod.Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')],
            self._gates_on('shared_gate', '7011', '7718'))

        rendered = _mod.render_markdown({
            'apply': False,
            'skips': {'gate_ambiguous': _by_reason(skips, 'gate_ambiguous')},
        })

        assert '### gate_ambiguous: 1' in rendered
        assert '7011' in rendered
        assert '7718' in rendered

    def test_gate_ambiguous_is_a_pre_seeded_error_bucket(self):
        assert 'gate_ambiguous' in _mod.ERROR_OUTCOMES
        assert 'gate_ambiguous' in _mod.SKIP_BUCKETS


def _one_group(renames, gate_tasks=()) -> object:
    """The single :class:`GateGroup` a writer test drives, unpacked."""
    groups, _skips = _mod.pair_gate_blocks(renames, list(gate_tasks))
    assert len(groups) == 1, groups
    return groups[0]


class TestGateLockstepWriter:
    """``rename_group(memory_service, group, *, apply, client)``.

    The pair is ALL-OR-NOTHING.  Memory renames land first; the gate patch
    follows only once they ALL succeeded; and if either half fails the group
    ends at its original state, so no gate is ever left scrolling a slug no
    record carries.
    """

    @pytest.mark.asyncio
    async def test_memory_renames_land_first_then_the_gate_patch(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = _client()
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert [r['outcome'] for r in results] == ['renamed']
        assert records['m1']['metadata']['topic'] == 'gate-topic'
        assert gate_row['outcome'] == 'gate_patched'
        # The write is a NARROW merge of the one block, never a whole-blob
        # replace: `update_task` refuses any replace carrying `done_provenance`.
        (tool, payload), _ = client.call_tool.call_args
        assert tool == 'update_task'
        assert payload['id'] == '4220'
        assert payload['metadata_mode'] == 'merge'
        assert payload['metadata'] == {
            _mod.GATE_METADATA_KEY: {'topic': 'gate-topic'},
        }

    @pytest.mark.asyncio
    async def test_a_dry_run_writes_neither_half(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = _client()
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        results, gate_row = await _mod.rename_group(
            service, group, apply=False, client=client)

        assert [r['outcome'] for r in results] == ['would_rename']
        assert gate_row['outcome'] == 'would_patch_gate'
        assert records['m1']['metadata']['topic'] == 'gate_topic'
        service.update_memory.assert_not_awaited()
        client.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_memory_failure_withholds_the_gate_patch(self):
        """The other direction of the pair: a failed half holds the whole group."""
        records = _store(('m1', 'gate_topic'))  # m2 is absent -> memory_not_found
        service = _stateful_service(records)
        client = _client()
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             _mod.Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert {r['outcome'] for r in results} >= {'memory_not_found'}
        client.call_tool.assert_not_awaited()
        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert gate_row['half'] == 'memory'

    @pytest.mark.asyncio
    async def test_a_refused_gate_patch_undoes_the_memory_half(self):
        """The dangerous direction, and the reason the group exists at all.

        ``update_task`` reports some rejections by RETURNING an error envelope
        inside a successful JSON-RPC frame, so a caller guarding only against
        exceptions would score a refused gate move as a completed one — and
        leave the gate scrolling a slug the records no longer carry.
        """
        records = _store(('m1', 'gate_topic'), ('m2', 'gate_topic'))
        service = _stateful_service(records)
        client = _client(result={'success': False, 'error': 'done_provenance_via_update_task'})
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             _mod.Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert gate_row['half'] == 'gate'
        assert 'done_provenance_via_update_task' in str(gate_row['error'])
        # Net effect: the slug is exactly where it started in BOTH stores.
        assert records['m1']['metadata']['topic'] == 'gate_topic'
        assert records['m2']['metadata']['topic'] == 'gate_topic'
        assert [r['outcome'] for r in results] == [
            'gate_lockstep_failed', 'gate_lockstep_failed']

    @pytest.mark.asyncio
    async def test_a_raising_gate_patch_undoes_the_memory_half_too(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = _client(error=RuntimeError('mcp down'))
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        _results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert 'mcp down' in str(gate_row['error'])
        assert records['m1']['metadata']['topic'] == 'gate_topic'

    @pytest.mark.asyncio
    async def test_a_failed_undo_is_named_on_the_row_not_swallowed(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = _client(error=RuntimeError('mcp down'))
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        calls = {'n': 0}
        forward = service.update_memory.side_effect

        async def _fail_the_undo(**kwargs):
            calls['n'] += 1
            if calls['n'] > 1:
                raise RuntimeError('undo failed')
            return await forward(**kwargs)

        service.update_memory = AsyncMock(side_effect=_fail_the_undo)

        _results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert gate_row['undo_failures']
        assert 'undo failed' in str(gate_row['undo_failures'])

    @pytest.mark.asyncio
    async def test_an_ungated_group_never_reaches_the_client(self):
        records = _store(('m1', 'plain_topic'))
        service = _stateful_service(records)
        client = _client()
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')])

        results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert [r['outcome'] for r in results] == ['renamed']
        assert gate_row is None
        client.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ungated_group_runs_without_a_client_at_all(self):
        """A run that reaches no gate must not require an MCP handshake."""
        records = _store(('m1', 'plain_topic'))
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')])

        results, gate_row = await _mod.rename_group(
            _stateful_service(records), group, apply=True, client=None)

        assert [r['outcome'] for r in results] == ['renamed']
        assert gate_row is None

    @pytest.mark.asyncio
    async def test_a_gated_group_with_no_client_is_a_lockstep_failure(self):
        """Unable to move the gate is the same verdict as refused to move it."""
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        _results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=None)

        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert records['m1']['metadata']['topic'] == 'gate_topic'

    @pytest.mark.asyncio
    async def test_an_already_normalized_record_is_not_undone(self):
        """Undo reverses THIS run's writes; it does not rewrite what it found."""
        records = _store(('m1', 'gate-topic'), ('m2', 'gate_topic'))
        service = _stateful_service(records)
        client = _client(error=RuntimeError('mcp down'))
        group = _one_group(
            [_mod.Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             _mod.Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [_gate_task('4220', 'gate_topic')])

        _results, gate_row = await _mod.rename_group(
            service, group, apply=True, client=client)

        assert gate_row['outcome'] == 'gate_lockstep_failed'
        assert records['m1']['metadata']['topic'] == 'gate-topic'
        assert records['m2']['metadata']['topic'] == 'gate_topic'


class TestGateLockstepOutcomesAreGraded:
    """A held pair must fail the run — it is unfinished work, not a no-op."""

    def test_gate_lockstep_failed_is_an_error_outcome(self):
        assert 'gate_lockstep_failed' in _mod.ERROR_OUTCOMES

    def test_both_gate_outcomes_are_pre_seeded_buckets(self):
        for bucket in ('gate_lockstep_failed', 'orphan_gate_topic'):
            assert bucket in _mod.SKIP_BUCKETS, bucket

    def test_an_orphan_gate_does_not_by_itself_fail_the_run(self):
        """A dangling gate predates this sweep; reporting it is the deliverable."""
        assert 'orphan_gate_topic' not in _mod.ERROR_OUTCOMES


def _listing(project_id: str, project_root: str, *tasks: dict) -> dict:
    """One ``get_tasks`` response in the measured live shape."""
    return {'tasks': list(tasks), 'project_id': project_id, 'project_root': project_root}


def _task_row(task_id: object, metadata: object) -> dict:
    """One task as ``get_tasks`` lists it, carrying fields the census must drop."""
    return {'id': task_id, 'title': f'task {task_id}', 'status': 'pending',
            'dependencies': [], 'metadata': metadata}


def _census_client(answers: dict[str, object]) -> MagicMock:
    """A client whose ``get_tasks`` answers per root: a response, or an exception.

    Answers are deep-copied, as a JSON transport would hand them over, so the
    census can never alias the double's own state.
    """
    async def _call_tool(name, payload):
        assert name == 'get_tasks', name
        answer = answers[payload['project_root']]
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)

    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=_call_tool)
    return client


_GATE_META = {_mod.GATE_METADATA_KEY: {'topic': 'gate_slug'}}


class TestConsolidationGateCensus:
    """``census_consolidation_gates(client, roots) -> GateCensus``.

    The input the lockstep never had on the live path.  Its contract is to
    state what it could NOT see: a root that cannot be read becomes a
    structured failure, never a silent gap that reads as "no gates here".
    """

    @pytest.mark.asyncio
    async def test_one_unpaged_get_tasks_per_distinct_root(self):
        client = _census_client({
            '/df': _listing('dark_factory', '/df'),
            '/rf': _listing('reify', '/rf'),
        })

        await _mod.census_consolidation_gates(client, ['/df', '/rf', '/df'])

        assert client.call_tool.await_args_list == [
            call('get_tasks', {'project_root': '/df'}),
            call('get_tasks', {'project_root': '/rf'}),
        ]

    @pytest.mark.asyncio
    async def test_only_tasks_carrying_a_gate_block_are_kept(self):
        client = _census_client({'/df': _listing(
            'dark_factory', '/df',
            _task_row(1, _GATE_META),
            _task_row(2, {'task_kind': 'deterministic'}),
            _task_row(3, 'not-a-dict'),
            _task_row(4, {_mod.GATE_METADATA_KEY: 'not-a-dict'}),
            _task_row(5, None),
        )})

        census = await _mod.census_consolidation_gates(client, ['/df'])

        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.asyncio
    async def test_a_kept_task_carries_the_server_stamp_and_nothing_else(self):
        """``project_id``/``project_root`` come off the RESPONSE, never the caller.

        The server normalizes a worktree root to its main checkout and derives
        the project id from that — the derivation the closure scroll runs
        under — so recomputing either here would be a second copy free to
        diverge.
        """
        block = {'topic': 'gate_slug', 'provenance': {'report_run': 'r1'}}
        client = _census_client({'/main/.worktrees/7': _listing(
            'dark_factory', '/main',
            _task_row(4220, {_mod.GATE_METADATA_KEY: block, 'task_kind': 'deterministic'}),
        )})

        census = await _mod.census_consolidation_gates(client, ['/main/.worktrees/7'])

        assert list(census.tasks) == [{
            'id': '4220',
            'project_id': 'dark_factory',
            'project_root': '/main',
            'metadata': {_mod.GATE_METADATA_KEY: block},
        }]

    @pytest.mark.asyncio
    async def test_tasks_are_sorted_by_project_then_id(self):
        client = _census_client({
            '/rf': _listing('reify', '/rf', _task_row(20, _GATE_META), _task_row(10, _GATE_META)),
            '/df': _listing('dark_factory', '/df', _task_row(30, _GATE_META)),
        })

        census = await _mod.census_consolidation_gates(client, ['/rf', '/df'])

        assert [(t['project_id'], t['id']) for t in census.tasks] == [
            ('dark_factory', '30'), ('reify', '10'), ('reify', '20'),
        ]

    @pytest.mark.asyncio
    async def test_covered_projects_are_the_projects_that_answered(self):
        client = _census_client({
            '/df': _listing('dark_factory', '/df'),
            '/rf': _listing('reify', '/rf'),
        })

        census = await _mod.census_consolidation_gates(client, ['/df', '/rf'])

        assert census.covered_projects == {'dark_factory', 'reify'}
        assert list(census.failures) == []

    @pytest.mark.parametrize('answer', [
        pytest.param({'error': 'no such project root', 'error_type': 'ValidationError'},
                     id='error-envelope'),
        pytest.param({'tasks': {'1': {}}, 'project_id': 'ghost', 'project_root': '/bad'},
                     id='tasks-not-a-list'),
        pytest.param({'tasks': [], 'project_root': '/bad'}, id='no-project-id'),
        pytest.param({'tasks': [], 'project_id': 'ghost'}, id='no-project-root'),
        pytest.param(RuntimeError('get_tasks failed: timed out'), id='raises'),
    ])
    @pytest.mark.asyncio
    async def test_a_root_that_cannot_be_read_is_a_failure_not_a_gap(self, answer):
        client = _census_client({
            '/bad': answer,
            '/df': _listing('dark_factory', '/df', _task_row(1, _GATE_META)),
        })

        census = await _mod.census_consolidation_gates(client, ['/bad', '/df'])

        (failure,) = census.failures
        assert failure['reason'] == 'gate_census_incomplete'
        assert failure['project_root'] == '/bad'
        assert failure['error']
        assert census.covered_projects == {'dark_factory'}
        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.parametrize('answer, cause', [
        pytest.param({'error': 'no such project root', 'error_type': 'ValidationError'},
                     'no such project root', id='error-envelope'),
        pytest.param(RuntimeError('get_tasks failed: timed out'), 'timed out', id='raises'),
    ])
    @pytest.mark.asyncio
    async def test_the_failure_carries_the_cause_it_was_given(self, answer, cause):
        census = await _mod.census_consolidation_gates(
            _census_client({'/bad': answer}), ['/bad'])

        (failure,) = census.failures
        assert cause in failure['error']

    @pytest.mark.asyncio
    async def test_one_project_answering_from_two_task_stores_is_ambiguous(self):
        client = _census_client({
            '/a': _listing('reify', '/a'),
            '/b': _listing('reify', '/b'),
            '/df': _listing('dark_factory', '/df'),
        })

        census = await _mod.census_consolidation_gates(client, ['/a', '/b', '/df'])

        assert census.covered_projects == {'dark_factory'}
        (failure,) = census.failures
        assert failure['reason'] == 'gate_census_incomplete'
        assert failure['project_id'] == 'reify'
        assert failure['project_root'] == '/b'
        assert '/a' in failure['error']

    @pytest.mark.asyncio
    async def test_two_roots_normalized_to_one_store_are_deduplicated(self):
        """A worktree path and its main checkout are one task store, read twice."""
        listing = _listing('dark_factory', '/main', _task_row(1, _GATE_META))
        client = _census_client({'/main/.worktrees/7': listing, '/main': listing})

        census = await _mod.census_consolidation_gates(
            client, ['/main/.worktrees/7', '/main'])

        assert list(census.failures) == []
        assert census.covered_projects == {'dark_factory'}
        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.asyncio
    async def test_the_census_records_the_roots_it_attempted(self):
        client = _census_client({
            '/df': _listing('dark_factory', '/df'),
            '/bad': RuntimeError('down'),
        })

        census = await _mod.census_consolidation_gates(client, ['/df', '/bad', '/df'])

        assert list(census.roots) == ['/df', '/bad']

    def test_the_census_is_a_frozen_value(self):
        census = _mod.GateCensus(
            roots=('/df',), tasks=(), covered_projects=frozenset({'dark_factory'}),
            failures=())

        assert dataclasses.is_dataclass(census)
        with pytest.raises(dataclasses.FrozenInstanceError):
            census.tasks = ()  # type: ignore[misc]


# ===========================================================================
# run(), the fail-closed preflight, the artifacts and the CLI
# ===========================================================================

async def run_sweep(service, **kwargs) -> dict:
    """``_mod.run`` under a name that reads as a sweep at the call site."""
    return await _mod.run(service, **kwargs)


def resolve_exit(report: dict) -> int:
    return _mod.resolve_exit_code(report)


class _FakeCorpus:
    """A STATEFUL Mem0 double: scroll, count, read and write over one dict.

    Everything below this point is a claim about the store's END STATE rather
    than about a call list — that an apply run's writes are visible to the
    residue probe, and that a SECOND run over the resulting store plans
    nothing.  A stateless double cannot express either: it would answer the
    verification probe from the same fixture the first run was planned off,
    so a sweep that wrote nothing at all would still look idempotent.
    """

    def __init__(self, records: dict[str, list[dict]]):
        self.by_project: dict[str, dict[str, dict]] = {
            project_id: {r['id']: copy.deepcopy(r) for r in rows}
            for project_id, rows in records.items()
        }
        self.scrolls: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []

    def _rows(self, project_id: str) -> list[dict]:
        return list(self.by_project.get(project_id, {}).values())

    @staticmethod
    def _match(record: dict, filters: dict) -> bool:
        metadata = record.get('metadata') or {}
        return all(metadata.get(key) == value for key, value in filters.items())

    def topic_of(self, project_id: str, memory_id: str) -> object:
        return self.by_project[project_id][memory_id]['metadata'].get('topic')

    def service(self) -> MagicMock:
        corpus = self
        service = MagicMock()
        backend = MagicMock()

        async def _scroll_all_by_metadata(scope, filters, **_kwargs):
            corpus.scrolls.append((scope.project_id, dict(filters)))
            for record in corpus._rows(scope.project_id):
                if corpus._match(record, filters):
                    yield copy.deepcopy(record)

        async def _count_by_metadata(scope, filters):
            return sum(
                1 for r in corpus._rows(scope.project_id) if corpus._match(r, filters)
            )

        backend.scroll_all_by_metadata = _scroll_all_by_metadata
        backend.count_by_metadata = AsyncMock(side_effect=_count_by_metadata)
        service.mem0 = backend

        async def _get_memory_by_id(*, project_id, memory_id):
            record = corpus.by_project.get(project_id, {}).get(memory_id)
            return copy.deepcopy(record) if record is not None else None

        async def _update_memory(*, memory_id, project_id, metadata_patch, **_kw):
            record = corpus.by_project.get(project_id, {}).get(memory_id)
            if record is None:
                return {'error_type': 'not_found', 'error': memory_id}
            corpus.writes.append((memory_id, dict(metadata_patch)))
            record['metadata'].update(metadata_patch)
            return {'status': 'updated', 'id': memory_id, 'metadata_patched': True}

        async def _count_memories_by_metadata(project_id, filters):
            return sum(1 for r in corpus._rows(project_id) if corpus._match(r, filters))

        service.get_memory_by_id = AsyncMock(side_effect=_get_memory_by_id)
        service.update_memory = AsyncMock(side_effect=_update_memory)
        service.count_memories_by_metadata = AsyncMock(
            side_effect=_count_memories_by_metadata)
        return service


def _crec(memory_id: str, topic: object, *,
          category: str = 'procedural_knowledge', **meta) -> dict:
    """A corpus record: like ``_rec`` but carrying the partition key."""
    metadata: dict = {'category': category}
    metadata.update(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def _run_service(records: dict[str, list[dict]] | None = None) -> tuple[MagicMock, _FakeCorpus]:
    corpus = _FakeCorpus(records if records is not None else {})
    return corpus.service(), corpus


class TestRun:
    """``run(memory_service, *, projects, apply, client, gate_tasks)``.

    The whole sweep: enumerate, plan, pair, write, verify, report.
    """

    @pytest.mark.asyncio
    async def test_a_dry_run_plans_the_work_and_writes_nothing(self):
        service, corpus = _run_service({
            'dark_factory': [_crec('m1', 'legacy_topic'), _crec('m2', 'good-topic')],
        })

        report = await run_sweep(service, projects=('dark_factory',), apply=False)

        assert report['apply'] is False
        assert report['outcomes'].get('would_rename') == 1
        assert corpus.writes == []
        assert corpus.topic_of('dark_factory', 'm1') == 'legacy_topic'

    @pytest.mark.asyncio
    async def test_an_apply_run_moves_the_slug(self):
        service, corpus = _run_service({
            'dark_factory': [_crec('m1', 'legacy_topic')],
        })

        report = await run_sweep(service, projects=('dark_factory',), apply=True)

        assert report['outcomes'].get('renamed') == 1
        assert corpus.topic_of('dark_factory', 'm1') == 'legacy-topic'

    @pytest.mark.asyncio
    async def test_a_second_run_over_the_mutated_store_plans_zero_writes(self):
        """Idempotence — only a stateful double can state this at all."""
        service, corpus = _run_service({
            'dark_factory': [_crec('m1', 'legacy_topic'), _crec('m2', 'legacy_topic')],
        })

        await run_sweep(service, projects=('dark_factory',), apply=True)
        writes_after_first = len(corpus.writes)
        second = await run_sweep(service, projects=('dark_factory',), apply=True)

        assert writes_after_first == 2
        assert len(corpus.writes) == 2
        assert second['rename_count'] == 0
        assert second['outcomes'] == {}
        assert resolve_exit(second) == 0

    @pytest.mark.asyncio
    async def test_the_report_carries_every_skip_bucket_pre_seeded(self):
        service, _corpus = _run_service({'dark_factory': []})

        report = await run_sweep(service, projects=('dark_factory',))

        for bucket in _mod.SKIP_BUCKETS:
            assert bucket in report['skips'], bucket
            assert report['skips'][bucket] == []

    @pytest.mark.asyncio
    async def test_it_sweeps_every_requested_project(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'a_topic')],
            'reify': [_crec('m2', 'b_topic')],
        })

        report = await run_sweep(service, projects=('dark_factory', 'reify'))

        assert report['projects'] == ['dark_factory', 'reify']
        assert {r['project_id'] for r in report['results']} == {'dark_factory', 'reify'}

    @pytest.mark.asyncio
    async def test_the_report_states_it_is_NOT_bounded(self):
        """The sibling reports ``bounded: True``; this one must not."""
        service, _corpus = _run_service({'dark_factory': []})

        report = await run_sweep(service, projects=('dark_factory',))

        assert report['bounded'] is False
        assert 'corpus' in report['scope']

    @pytest.mark.asyncio
    async def test_a_collision_is_refused_and_recorded_end_to_end(self):
        service, corpus = _run_service({
            'dark_factory': [_crec('m1', 'dup_topic'), _crec('m2', 'dup-topic')],
        })

        report = await run_sweep(service, projects=('dark_factory',), apply=True)

        assert report['skips']['slug_collision']
        assert corpus.topic_of('dark_factory', 'm1') == 'dup_topic'
        assert resolve_exit(report) == 1

    @pytest.mark.asyncio
    async def test_the_residue_probe_runs_and_files_nothing_when_drained(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'legacy_topic')],
        })

        report = await run_sweep(service, projects=('dark_factory',), apply=True)

        assert report['skips']['legacy_slug_residue'] == []

    @pytest.mark.asyncio
    async def test_coverage_is_reported_per_project_and_category(self):
        service, _corpus = _run_service({'dark_factory': [_crec('m1', 'a_topic')]})

        report = await run_sweep(service, projects=('dark_factory',))

        assert set(report['coverage']['dark_factory']) == set(_mod.census_categories())
        assert report['coverage_complete'] is True


class TestRunApplyStoreMutationPreflight:
    """The fail-closed capability probe, hoisted above the enumeration.

    Probing per record instead would run the check N times and — since
    ``StoreMutationUnavailable`` subclasses ``RuntimeError`` — be swallowed by
    ``rename_one``'s per-record ``except Exception``, downgrading a run-wide
    environment denial into N error rows inside a report that otherwise reads
    as a completed sweep.  The consequence is sharper for a rename than for a
    stamp: a half-applied rename splits one claim across two topic values,
    which is worse for an exact-match ``{'topic': T}`` read than either
    uniform state.
    """

    @pytest.mark.asyncio
    async def test_apply_probes_exactly_once_with_the_operation_named(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw))
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'a_topic'), _crec('m2', 'b_topic')],
        })

        await run_sweep(service, projects=('dark_factory',), apply=True)

        assert calls == [{'operation': 'normalize_topic_slugs --apply'}]

    @pytest.mark.asyncio
    async def test_a_dry_run_never_probes(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw))
        service, _corpus = _run_service({'dark_factory': [_crec('m1', 'a_topic')]})

        await run_sweep(service, projects=('dark_factory',), apply=False)

        assert calls == []

    @pytest.mark.asyncio
    async def test_a_refusal_scrolls_nothing_and_re_raises(self, monkeypatch):
        deny(_mod, monkeypatch)
        service, corpus = _run_service({'dark_factory': [_crec('m1', 'a_topic')]})

        with pytest.raises(_mod.StoreMutationUnavailable, match=SENTINEL):
            await run_sweep(service, projects=('dark_factory',), apply=True)

        assert corpus.scrolls == []
        assert corpus.writes == []

    @pytest.mark.asyncio
    async def test_the_refusal_goes_through_the_logger_not_stdout(
        self, monkeypatch, caplog, capsys,
    ):
        """stdout carries the machine-read artifact; a diagnosis must not."""
        deny(_mod, monkeypatch)
        service, _corpus = _run_service({'dark_factory': []})

        with (
            caplog.at_level('ERROR', logger='normalize_topic_slugs'),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            await run_sweep(service, projects=('dark_factory',), apply=True)

        assert capsys.readouterr().out == ''
        message = '\n'.join(r.message for r in caplog.records)
        assert 'normalize_topic_slugs' in message
        assert 'fail-closed' in message.lower()
        # The remedy must name the hazard, not merely the denial.
        assert 'two topic values' in message or 'half' in message.lower()


class TestReportRenderAndCli:
    """The artifacts, the grade and the argument surface."""

    def test_markdown_states_every_empty_bucket_explicitly(self):
        report = {'apply': False, 'projects': ['dark_factory'],
                  'skips': {bucket: [] for bucket in _mod.SKIP_BUCKETS},
                  'outcomes': {}}

        rendered = _mod.render_markdown(report)

        for bucket in _mod.SKIP_BUCKETS:
            assert f'### {bucket}: 0' in rendered, bucket

    def test_markdown_names_the_mode(self):
        assert 'DRY RUN' in _mod.render_markdown({'apply': False})
        assert 'APPLY' in _mod.render_markdown({'apply': True})

    def test_markdown_says_the_sweep_is_corpus_wide(self):
        rendered = _mod.render_markdown({'apply': False, 'bounded': False})
        assert 'corpus' in rendered.lower()

    def test_markdown_shows_a_populated_bucket_with_its_entries(self):
        report = {'apply': False, 'skips': {'slug_collision': [
            {'reason': 'slug_collision', 'project_id': 'dark_factory',
             'topic': 'dup_topic', 'note': 'occupied'}]}}

        rendered = _mod.render_markdown(report)

        assert '### slug_collision: 1' in rendered
        assert 'dup_topic' in rendered

    def test_json_is_stable_and_never_raises_on_an_odd_value(self):
        import datetime as _dt

        report = {'b': 1, 'a': _dt.datetime(2026, 1, 1)}

        rendered = _mod.render_json(report)

        assert rendered.index('"a"') < rendered.index('"b"')
        assert '2026-01-01' in rendered

    def test_json_of_an_unchanged_report_is_byte_comparable(self):
        one = _mod.render_json({'z': 1, 'a': {'y': 2, 'b': 3}})
        two = _mod.render_json({'a': {'b': 3, 'y': 2}, 'z': 1})
        assert one == two

    def test_exit_code_is_zero_on_a_clean_run(self):
        assert _mod.resolve_exit_code({'outcomes': {'renamed': 12}}) == 0
        assert _mod.resolve_exit_code({}) == 0

    def test_exit_code_is_one_for_any_error_outcome(self):
        for outcome in sorted(_mod.ERROR_OUTCOMES):
            assert _mod.resolve_exit_code({'outcomes': {outcome: 1}}) == 1, outcome

    def test_a_zero_count_error_outcome_is_still_clean(self):
        assert _mod.resolve_exit_code({'outcomes': {'update_failed': 0}}) == 0

    def test_project_replaces_the_default_rather_than_extending_it(self):
        args = _mod._build_parser().parse_args(['--project', 'reify'])
        assert _mod.resolve_projects(args) == ('reify',)

    def test_the_default_project_list_is_both_corpora(self):
        args = _mod._build_parser().parse_args([])
        assert _mod.resolve_projects(args) == ('dark_factory', 'reify')
        assert _mod.DEFAULT_PROJECTS == ('dark_factory', 'reify')

    def test_project_is_repeatable(self):
        args = _mod._build_parser().parse_args(
            ['--project', 'a', '--project', 'b'])
        assert _mod.resolve_projects(args) == ('a', 'b')

    def test_dry_run_is_the_default(self):
        assert _mod._build_parser().parse_args([]).apply is False
        assert _mod._build_parser().parse_args(['--apply']).apply is True

    def test_the_parser_exposes_every_documented_flag(self):
        args = _mod._build_parser().parse_args(
            ['--json-out', '/tmp/a.json', '--md-out', '/tmp/a.md',
             '--config', '/tmp/c.yaml'])
        assert args.json_out == '/tmp/a.json'
        assert args.md_out == '/tmp/a.md'
        assert args.config == '/tmp/c.yaml'

    def test_the_default_artifact_paths_sit_beside_the_siblings(self):
        args = _mod._build_parser().parse_args([])
        assert args.json_out.endswith('plans/topic-slug-normalization-report.json')
        assert args.md_out.endswith('plans/topic-slug-normalization-report.md')


# ===========================================================================
# baseline_delta — scope item 4, the measurement the next reader inherits
# ===========================================================================

class TestBaselineDelta:
    """The report diffs THIS run against one committed, dated measurement.

    Scope item 4 asks for the post-migration count "against the ad707e72
    baseline ... so the next reader can see the delta instead of re-deriving
    it".  Folding it into the report rather than leaving it a manual step is
    what makes it survive: a number an operator has to re-derive is a number
    the next operator re-derives differently.

    NOTHING here asserts a LIVE corpus count.  The very history this block
    carries — 98 -> 103 distinct non-conforming values while the validator sat
    in warn mode — proves the number moves on its own, so a test pinning it
    would be a doomed RED.  The baseline is a constant, the measurement comes
    off an INJECTED corpus, and these tests pin the arithmetic between them.
    """

    def test_the_baseline_is_a_named_committed_constant(self):
        assert _mod.BASELINE_DISTINCT_NON_CONFORMING == 103

    @pytest.mark.asyncio
    async def test_the_report_carries_the_baseline_and_its_provenance(self):
        service, _corpus = _run_service({'dark_factory': []})

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert block['baseline_distinct_non_conforming'] == 103
        assert 'ad707e72' in block['baseline_source']
        assert '2026-08-04' in block['baseline_source']

    @pytest.mark.asyncio
    async def test_the_history_records_the_direction_of_travel(self):
        """98 -> 103 is the fact that makes a pinned live count indefensible."""
        service, _corpus = _run_service({'dark_factory': []})

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert '98' in str(block['baseline_history'])
        assert '103' in str(block['baseline_history'])

    @pytest.mark.asyncio
    async def test_the_measurement_counts_distinct_VALUES_not_records(self):
        """One legacy slug on eight records is ONE non-conforming value.

        This is the census's own partition (``_build_topic_coverage`` item 3),
        and matching it is what makes the delta comparable to the baseline at
        all: counting records instead would report 8 against a baseline of
        distinct values and manufacture a growth that never happened.
        """
        service, _corpus = _run_service({
            'dark_factory': [_crec(f'm{i}', 'legacy_topic') for i in range(8)],
        })

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert block['measured_distinct_non_conforming'] == 1

    @pytest.mark.asyncio
    async def test_the_measurement_spans_every_swept_project(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'a_topic'), _crec('m2', 'b_topic')],
            'reify': [_crec('m3', 'c_topic')],
        })

        block = (await run_sweep(
            service, projects=('dark_factory', 'reify')))['baseline_delta']

        assert block['measured_distinct_non_conforming'] == 3

    @pytest.mark.asyncio
    async def test_a_conforming_corpus_measures_zero(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'fine-topic'), _crec('m2', None)],
        })

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert block['measured_distinct_non_conforming'] == 0

    @pytest.mark.asyncio
    async def test_the_delta_is_measured_minus_baseline(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'a_topic')],
        })

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert block['delta'] == (
            block['measured_distinct_non_conforming']
            - block['baseline_distinct_non_conforming']
        )
        assert block['delta'] == 1 - 103

    @pytest.mark.asyncio
    async def test_it_names_the_standing_re_measurement_instrument(self):
        """The live number stays the census's to define — not this script's."""
        service, _corpus = _run_service({'dark_factory': []})

        block = (await run_sweep(service, projects=('dark_factory',)))['baseline_delta']

        assert 'census_memory_metadata.py' in block['remeasure_command']
        assert 'slug_non_conforming' in block['remeasure_command']
        assert 'topic_coverage' in block['remeasure_command']

    @pytest.mark.asyncio
    async def test_an_incomplete_sweep_says_its_measurement_is_a_lower_bound(self):
        """A cell this run never finished cannot be counted as zero residue."""
        service, _corpus = _run_service({'dark_factory': [_crec('m1', 'a_topic')]})
        report = await run_sweep(service, projects=('dark_factory',))
        report['coverage_complete'] = False

        rendered = _mod.render_markdown(report)

        assert 'LOWER BOUND' in rendered

    def test_markdown_prints_the_whole_block(self):
        report = {
            'apply': False,
            'baseline_delta': {
                'baseline_distinct_non_conforming': 103,
                'baseline_source': 'memory ad707e72, measured 2026-08-04',
                'baseline_history': 'PRD leaf alpha measured 98; ad707e72 103',
                'measured_distinct_non_conforming': 87,
                'measured_by_project': {'dark_factory': 40, 'reify': 47},
                'delta': -16,
                'remeasure_command': (
                    'census_memory_metadata.py -> '
                    'coverage.topic_coverage.slug_non_conforming'
                ),
            },
        }

        rendered = _mod.render_markdown(report)

        assert '103' in rendered
        assert 'ad707e72' in rendered
        assert '2026-08-04' in rendered
        assert '98' in rendered
        assert '87' in rendered
        assert '-16' in rendered
        assert 'census_memory_metadata.py' in rendered
        assert 'slug_non_conforming' in rendered

    def test_markdown_survives_a_report_with_no_baseline_block(self):
        """The renderer is also handed hand-built reports; it must not crash."""
        assert _mod.render_markdown({'apply': False})

    @pytest.mark.asyncio
    async def test_the_measured_count_is_broken_out_per_project(self):
        service, _corpus = _run_service({
            'dark_factory': [_crec('m1', 'a_topic')],
            'reify': [_crec('m2', 'b_topic'), _crec('m3', 'c_topic')],
        })

        block = (await run_sweep(
            service, projects=('dark_factory', 'reify')))['baseline_delta']

        assert block['measured_by_project'] == {'dark_factory': 1, 'reify': 2}


class TestNoLiveCorpusCountIsPinned:
    """The one rule this whole section exists to keep.

    The corpus is live and moving; the baseline's own history proves it. A
    committed constant is a dated measurement a reader can evaluate; a test
    asserting today's live number is a scheduled failure that teaches whoever
    hits it to delete the assertion.
    """

    def test_this_suite_never_reaches_a_live_backend(self):
        """The structural reason no live count can be pinned here.

        Checked over the parsed IMPORT statements rather than by substring,
        so the assertion cannot be satisfied — or defeated — by its own text
        appearing in the file.  This suite loads the script by path and
        injects doubles; it imports nothing from the package, so there is no
        seam through which a live corpus could reach an assertion.
        """
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or '')

        assert not [m for m in imported if m.startswith('fused_memory')], imported


# ===========================================================================
# End to end — every disposition at once, over both corpora
# ===========================================================================

def _stateful_gate_client(tasks: list[dict]) -> MagicMock:
    """A ``FusedMemoryClient`` double that actually APPLIES the gate patch.

    Stateful for the same reason :class:`_FakeCorpus` is: the second-run
    idempotence claim is only meaningful if the first run's gate move is
    visible to it.  A client that acknowledged and forgot would let a sweep
    that never moved the block look idempotent.
    """
    by_id = {str(t['id']): t for t in tasks}

    async def _call_tool(name, payload):
        assert name == 'update_task', name
        task = by_id.get(str(payload['id']))
        if task is None:
            return {'success': False, 'error': 'task_not_found'}
        assert payload['metadata_mode'] == 'merge'
        task['metadata'].update(payload['metadata'])
        return {'id': payload['id'], 'updated': True, 'message': 'ok'}

    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=_call_tool)
    return client


def _e2e_corpus() -> tuple[dict[str, list[dict]], list[dict]]:
    """Every disposition this sweep can reach, present at once.

    Deliberately built as ONE fixture rather than six: the seams between
    enumeration, planning, collision refusal, renaming, the gate lockstep and
    the residue probe are where a per-function suite cannot look, and a
    corpus that exercises one disposition at a time never puts two of them in
    the same scroll page.
    """
    records = {
        'dark_factory': [
            _crec('df1', 'good-topic'),                       # conforming
            _crec('df2', 'clean_slug'),                       # renames
            _crec('df3', 'clean_slug',                        # ... across TWO
                  category='observations_and_summaries'),     #     categories
            _crec('df4', 'dup_topic'),                        # collides
            _crec('df5', 'dup-topic'),                        # ... with this
            _crec('df6', '!!!'),                              # unfoldable
            _crec('df7', 'gate_slug'),                        # gate-backed
            _crec('df8', None),                               # no topic at all
        ],
        'reify': [
            _crec('rf1', 'good-topic'),
            _crec('rf2', 'other_slug'),
            _crec('rf3', None),
        ],
    }
    gates = [_gate_task('4220', 'gate_slug', project_id='dark_factory')]
    return records, gates


_E2E_RENAMED = {'df2', 'df3', 'df7', 'rf2'}
_E2E_UNTOUCHED = {
    'df1': 'good-topic', 'df4': 'dup_topic', 'df5': 'dup-topic',
    'df6': '!!!', 'rf1': 'good-topic',
}


class TestEndToEnd:
    """The whole pipeline as one act, over both default projects."""

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_every_bucket_and_writes_nothing(self):
        records, gates = _e2e_corpus()
        service, corpus = _run_service(records)
        client = _stateful_gate_client(gates)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'),
            apply=False, client=client, gate_tasks=gates)

        assert report['outcomes'].get('would_rename') == len(_E2E_RENAMED)
        assert len(report['skips']['slug_collision']) == 1
        assert len(report['skips']['topic_unfoldable']) == 1
        assert report['skips']['orphan_gate_topic'] == []
        assert report['skips']['legacy_slug_residue'] == []
        assert corpus.writes == []
        client.call_tool.assert_not_awaited()
        assert gates[0]['metadata'][_mod.GATE_METADATA_KEY]['topic'] == 'gate_slug'

    @pytest.mark.asyncio
    async def test_an_apply_run_moves_exactly_the_renameable_records(self):
        records, gates = _e2e_corpus()
        service, corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        assert report['outcomes'].get('renamed') == len(_E2E_RENAMED)
        assert {mid for mid, _patch in corpus.writes} == _E2E_RENAMED
        assert corpus.topic_of('dark_factory', 'df2') == 'clean-slug'
        assert corpus.topic_of('dark_factory', 'df3') == 'clean-slug'
        assert corpus.topic_of('dark_factory', 'df7') == 'gate-slug'
        assert corpus.topic_of('reify', 'rf2') == 'other-slug'

    @pytest.mark.asyncio
    async def test_the_refusals_leave_their_records_exactly_as_found(self):
        records, gates = _e2e_corpus()
        service, corpus = _run_service(records)

        await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        for memory_id, topic in _E2E_UNTOUCHED.items():
            project = 'reify' if memory_id.startswith('rf') else 'dark_factory'
            assert corpus.topic_of(project, memory_id) == topic, memory_id

    @pytest.mark.asyncio
    async def test_the_gate_block_moves_in_lockstep(self):
        records, gates = _e2e_corpus()
        service, _corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        assert gates[0]['metadata'][_mod.GATE_METADATA_KEY]['topic'] == 'gate-slug'
        assert [r['outcome'] for r in report['gate_results']] == ['gate_patched']

    @pytest.mark.asyncio
    async def test_the_renamed_old_slugs_drain_to_zero(self):
        records, gates = _e2e_corpus()
        service, _corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        assert report['skips']['legacy_slug_residue'] == []

    @pytest.mark.asyncio
    async def test_the_run_exits_one_because_refusals_remain(self):
        records, gates = _e2e_corpus()
        service, _corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        assert resolve_exit(report) == 1
        assert report['outcomes'].get('slug_collision') == 1

    @pytest.mark.asyncio
    async def test_a_second_apply_run_plans_nothing_and_fails_for_the_same_reason(self):
        """Idempotence, and the refusal to 'resolve' a refusal by re-running."""
        records, gates = _e2e_corpus()
        service, corpus = _run_service(records)
        client = _stateful_gate_client(gates)

        first = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=client, gate_tasks=gates)
        writes_after_first = len(corpus.writes)
        second = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=client, gate_tasks=gates)

        assert writes_after_first == len(_E2E_RENAMED)
        assert len(corpus.writes) == writes_after_first
        assert second['rename_count'] == 0
        assert second['outcomes'].get('renamed') is None
        assert resolve_exit(second) == resolve_exit(first) == 1
        assert len(second['skips']['slug_collision']) == 1
        assert len(second['skips']['topic_unfoldable']) == 1
        # The gate moved with the records, so it is not an orphan afterwards.
        assert second['skips']['orphan_gate_topic'] == []

    @pytest.mark.asyncio
    async def test_the_baseline_delta_measures_distinct_values_across_both(self):
        records, gates = _e2e_corpus()
        service, _corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'),
            client=_stateful_gate_client(gates), gate_tasks=gates)

        # clean_slug, dup_topic, '!!!', gate_slug in dark_factory; other_slug
        # in reify -- five distinct VALUES over six non-conforming records.
        assert report['baseline_delta']['measured_by_project'] == {
            'dark_factory': 4, 'reify': 1,
        }
        assert report['baseline_delta']['delta'] == 5 - 103

    @pytest.mark.asyncio
    async def test_both_artifacts_render_over_a_real_report(self):
        """A report shape no renderer has seen is where a renderer breaks."""
        records, gates = _e2e_corpus()
        service, _corpus = _run_service(records)

        report = await run_sweep(
            service, projects=('dark_factory', 'reify'), apply=True,
            client=_stateful_gate_client(gates), gate_tasks=gates)

        rendered = _mod.render_markdown(report)
        assert '### slug_collision: 1' in rendered
        assert '### topic_unfoldable: 1' in rendered
        assert 'gate-slug' in rendered
        assert _mod.render_json(report) == _mod.render_json(report)
