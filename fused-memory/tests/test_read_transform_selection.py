"""Tests for read_transform_selection.py — the read-transform arms (task 4004).

Three candidate read transforms over the ratified C write shape, scored
against the flat-read baseline: a PROMOTING topic pin, a TOPIC-KEYED grouped
read, and a topic-diversity cap.  All three are pure post-ranking transforms
over an already-fetched hit list, so every test here injects hand-built
ranked lists with exactly-known answers — which is what permits exact
assertions with no tolerances.

Both scripts are loaded via importlib so they can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file must be free of network, Qdrant and OPENAI_API_KEY
**except a live end-to-end test**, which carries its markers PER-TEST::

    @pytest.mark.integration
    @pytest.mark.timeout(600)
    @qdrant_skipif()
    @pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), ...)

Never via a module-level ``pytestmark``.  ``fused-memory/pyproject.toml``
sets ``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_bake_off_storage_shape.py:9-24``.

This file does NOT extend ``test_bake_off_storage_shape.py``: task 3560 is
in-progress and claims that module.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
SCRIPT_PATH = SCRIPTS_DIR / 'read_transform_selection.py'
BAKE_OFF_PATH = SCRIPTS_DIR / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a standalone script from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    return _load_script(SCRIPT_PATH, 'read_transform_selection')


@functools.cache
def _bake_off() -> types.ModuleType:
    return _load_script(BAKE_OFF_PATH, 'bake_off_storage_shape')


# ---------------------------------------------------------------------------
# Hand-built ranked lists
# ---------------------------------------------------------------------------
#
# Every retrieval below is constructed, not fetched: the answer is known
# exactly, so every assertion is exact and tolerance-free.  The builder is
# the shape of `test_bake_off_storage_shape.py:773-800`, minus the
# `contested` channel the arms in this file deliberately do not read.

import copy  # noqa: E402

import pytest  # noqa: E402


def _rec(record_id, *, topic: str | None = 't', canonical=False, kind=None,
         content='body', claim_ids=(), parent_id=None, cluster_id='c1'):
    """One stored record, in the shape the read transforms consume.

    ``topic=None`` is a first-class case, not a degenerate one: a topic-less
    slab is exactly what an un-migrated write leaves behind, and every
    transform has to survive one.  Hence the explicit optional annotation —
    inferring `str` from the default would make those call sites a type error.
    """
    metadata: dict = {'category': 'procedural_knowledge'}
    if topic is not None:
        metadata['topic'] = topic
    if canonical:
        metadata['canonical'] = True
    if kind is not None:
        metadata['kind'] = kind
    if parent_id is not None:
        metadata['parent_id'] = parent_id
    return _bake_off().ArmRecord(
        record_id=record_id,
        content=content,
        metadata=metadata,
        cluster_id=cluster_id,
        claim_ids=list(claim_ids),
        role='canonical' if canonical else (kind or 'peer'),
    )


def _canonical_index(*records):
    """`topic -> canonical`, built through the bake-off's own indexer.

    Never hand-rolled: `build_canonical_by_topic` enforces bool-identity
    `canonical is True` and RAISES on two canonicals per topic (INV-5), and
    all three arms in this file index through it rather than restating that
    rule.
    """
    return _bake_off().build_canonical_by_topic(list(records))


def _ids(records):
    return [r.record_id for r in records]


# ---------------------------------------------------------------------------
# Arm (1): the PROMOTING topic pin
# ---------------------------------------------------------------------------
#
# `apply_topic_anchor` (bake_off_storage_shape.py:1001) is ADDITIVE: it
# APPENDS a missing canonical after the ranking, deliberately, so that a
# measured improvement is attributable to the canonical becoming reachable at
# all rather than to the transform having hand-placed it high.
#
# That choice has an arithmetic consequence the E2 report's `pin changed
# window = 0.00` under c_peers records but does not explain: `read_path`
# truncates AFTER the transforms (`records[:k]`, :3243), so at an ALREADY-FULL
# window the appended canonical is truncated straight back off and the pin
# provably cannot change anything.  The 0.00 is forced by arithmetic; it is
# not a verdict on anchoring.  Arm (1) is the variant that separates the two:
# same firing rule, but the canonical is PROMOTED into the window instead of
# appended past its edge.


class TestPromotingAnchorFiresOnTheSameRuleAsTheAdditivePin:
    """Same trigger as `apply_topic_anchor`: `metadata['topic']` on a hit."""

    def test_an_absent_canonical_is_promoted_to_the_front_of_the_window(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [_rec('peer-1', topic='alpha'), _rec('peer-2', topic='alpha')]

        promoted = mod.apply_promoting_topic_anchor(
            hits, _canonical_index(canonical, *hits),
        )

        # FRONT, not the tail: this is the entire difference from the
        # additive pin, and the reason the transform can survive `[:k]`.
        assert _ids(promoted) == ['canon-a', 'peer-1', 'peer-2']

    def test_it_selects_exactly_what_the_additive_pin_selects(self):
        """Membership identical to `apply_topic_anchor`; only ORDER differs.

        Pinned as a cross-check against the landed additive pin so the two
        cannot drift apart on *which* canonical fires — if they did, arm (1)
        would stop being a controlled variant of the shipped transform and
        the report's A/B would be comparing two different rules.
        """
        mod = _mod()
        bake_off = _bake_off()
        canon_a = _rec('canon-a', topic='alpha', canonical=True)
        canon_b = _rec('canon-b', topic='beta', canonical=True)
        hits = [
            _rec('peer-1', topic='alpha'),
            _rec('peer-2', topic='beta'),
            _rec('slab-1', topic=None),
        ]
        index = _canonical_index(canon_a, canon_b, *hits)

        additive = bake_off.apply_topic_anchor(hits, index)
        promoting = mod.apply_promoting_topic_anchor(hits, index)

        assert set(_ids(promoting)) == set(_ids(additive))
        assert _ids(promoting) != _ids(additive)

    def test_a_topic_with_no_canonical_promotes_nothing(self):
        """`canonical_by_topic.get(topic) is None` — never synthesize one."""
        mod = _mod()
        hits = [_rec('peer-1', topic='orphan-topic')]

        promoted = mod.apply_promoting_topic_anchor(hits, _canonical_index(*hits))

        assert _ids(promoted) == ['peer-1']

    def test_a_canonical_for_an_absent_topic_never_fires(self):
        mod = _mod()
        unrelated = _rec('canon-z', topic='zeta', canonical=True)
        hits = [_rec('peer-1', topic='alpha')]

        promoted = mod.apply_promoting_topic_anchor(
            hits, _canonical_index(unrelated, *hits),
        )

        assert _ids(promoted) == ['peer-1']


class TestPromotingAnchorNeverDuplicates:
    """An already-ranked canonical MOVES; it is never emitted twice."""

    def test_an_already_ranked_canonical_is_moved_not_copied(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [
            _rec('peer-1', topic='alpha'),
            canonical,
            _rec('peer-2', topic='alpha'),
        ]

        promoted = mod.apply_promoting_topic_anchor(
            hits, _canonical_index(*hits),
        )

        assert _ids(promoted) == ['canon-a', 'peer-1', 'peer-2']
        assert len(promoted) == len(hits)  # nothing gained, nothing lost

    def test_the_relative_order_of_everything_else_is_preserved(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [
            _rec('peer-1', topic='alpha'),
            _rec('slab-1', topic=None),
            canonical,
            _rec('peer-2', topic='alpha'),
            _rec('slab-2', topic=None),
        ]

        promoted = mod.apply_promoting_topic_anchor(hits, _canonical_index(*hits))

        assert _ids(promoted) == [
            'canon-a', 'peer-1', 'slab-1', 'peer-2', 'slab-2',
        ]


class TestPromotingAnchorPassThrough:
    """Nothing pinnable ⇒ the list is returned unchanged, not rebuilt."""

    def test_hits_with_no_topic_metadata_are_returned_by_identity(self):
        mod = _mod()
        # The 300-record distractor slab carries `category` and NOT one
        # reserved vocabulary key (`_distractor_records`, :545), so this is
        # the real no-op case, not a contrived one.
        hits = [_rec('slab-1', topic=None), _rec('slab-2', topic=None)]

        promoted = mod.apply_promoting_topic_anchor(hits, _canonical_index(*hits))

        # Identity of the ELEMENTS, not value equality: `ArmRecord` is a
        # frozen dataclass, so `==` is field-wise and a transform that
        # rebuilt equal-valued copies — allocating on every no-op call, per
        # query per arm on the hot read path — would still satisfy `==`.
        assert all(p is h for p, h in zip(promoted, hits, strict=True))

    def test_an_empty_hit_list_is_returned_empty(self):
        assert _mod().apply_promoting_topic_anchor([], {}) == []


class TestPromotingAnchorIsDeterministic:
    """The promoted prefix follows a stated total order, never a set order."""

    def test_two_promoting_topics_order_by_first_appearance_in_the_ranking(self):
        mod = _mod()
        canon_a = _rec('canon-a', topic='alpha', canonical=True)
        canon_b = _rec('canon-b', topic='beta', canonical=True)
        hits = [_rec('peer-b', topic='beta'), _rec('peer-a', topic='alpha')]
        index = _canonical_index(canon_a, canon_b, *hits)

        promoted = mod.apply_promoting_topic_anchor(hits, index)

        # beta's peer ranked first, so beta's canonical leads: the tie-break
        # is the STORE's ranking, the only ordering signal available here.
        assert _ids(promoted) == ['canon-b', 'canon-a', 'peer-b', 'peer-a']

    def test_reversing_the_input_reverses_the_promoted_prefix_by_the_same_rule(self):
        mod = _mod()
        canon_a = _rec('canon-a', topic='alpha', canonical=True)
        canon_b = _rec('canon-b', topic='beta', canonical=True)
        hits = [_rec('peer-a', topic='alpha'), _rec('peer-b', topic='beta')]
        index = _canonical_index(canon_a, canon_b, *hits)

        promoted = mod.apply_promoting_topic_anchor(hits, index)

        assert _ids(promoted) == ['canon-a', 'canon-b', 'peer-a', 'peer-b']

    def test_repeated_calls_return_the_same_order(self):
        mod = _mod()
        canon_a = _rec('canon-a', topic='alpha', canonical=True)
        canon_b = _rec('canon-b', topic='beta', canonical=True)
        canon_c = _rec('canon-c', topic='gamma', canonical=True)
        hits = [
            _rec('peer-a', topic='alpha'),
            _rec('peer-b', topic='beta'),
            _rec('peer-c', topic='gamma'),
        ]
        index = _canonical_index(canon_a, canon_b, canon_c, *hits)

        runs = {
            tuple(_ids(mod.apply_promoting_topic_anchor(hits, index)))
            for _ in range(5)
        }

        assert len(runs) == 1


class TestPromotingAnchorIsPure:
    """No mutation of the hits, the index, or any record in either."""

    def test_it_mutates_neither_input(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [_rec('peer-1', topic='alpha'), _rec('slab-1', topic=None)]
        index = _canonical_index(canonical, *hits)
        hits_before = copy.deepcopy(hits)
        index_before = copy.deepcopy(index)

        mod.apply_promoting_topic_anchor(hits, index)

        assert hits == hits_before
        assert index == index_before

    def test_it_does_not_mutate_the_promoted_records_metadata(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [_rec('peer-1', topic='alpha')]
        metadata_before = copy.deepcopy(canonical.metadata)

        mod.apply_promoting_topic_anchor(hits, _canonical_index(canonical, *hits))

        assert canonical.metadata == metadata_before


# ---------------------------------------------------------------------------
# THE DECISIVE COMPARISON — a FULL window
# ---------------------------------------------------------------------------

K = 5


def _full_window():
    """*k* topic-carrying hits with their canonical NOT among them."""
    canonical = _rec('canon-a', topic='alpha', canonical=True)
    hits = [_rec(f'peer-{i}', topic='alpha') for i in range(K)]
    return canonical, hits, _canonical_index(canonical, *hits)


class TestFullWindowSeparatesTheTwoPins:
    """At a full window the additive pin CANNOT fire; the promoting one can.

    This is the measurement arm (1) exists to make.  `read_path` truncates
    after the transforms (`records[:k]`, bake_off_storage_shape.py:3243), so
    an appended canonical at |hits| == k is truncated straight back off.
    """

    def test_the_additive_pin_changes_a_full_window_by_exactly_nothing(self):
        bake_off = _bake_off()
        _canonical, hits, index = _full_window()

        anchored = bake_off.apply_topic_anchor(hits, index)

        # It DID pin — the canonical is there pre-truncation...
        assert _ids(anchored) == [*_ids(hits), 'canon-a']
        # ...and the reader's budget then removes it again, unchanged.
        assert _ids(anchored[:K]) == _ids(hits)

    def test_the_promoting_pin_does_change_that_same_full_window(self):
        mod = _mod()
        _canonical, hits, index = _full_window()

        promoted = mod.apply_promoting_topic_anchor(hits, index)

        assert _ids(promoted[:K]) != _ids(hits)
        assert promoted[0].record_id == 'canon-a'
        assert promoted[0].metadata['canonical'] is True

    def test_promotion_at_a_full_window_displaces_the_last_record(self):
        """Stated as a COST, not hidden: this transform is lossy at the edge.

        The transform itself drops nothing — it reorders, and the drop is
        `read_path`'s truncation at the reader's budget.  Both halves are
        pinned so the report can state the consequence rather than let 3111
        discover it: a promoting pin buys canonical reachability by evicting
        the k-th ranked record.
        """
        mod = _mod()
        _canonical, hits, index = _full_window()
        evicted = hits[-1].record_id

        promoted = mod.apply_promoting_topic_anchor(hits, index)

        # Pre-truncation: reordered, nothing lost.
        assert len(promoted) == K + 1
        assert evicted in _ids(promoted)
        # Post-truncation: the k-th ranked record is gone from the window.
        assert evicted not in _ids(promoted[:K])
        assert len(promoted[:K]) == K

    def test_a_window_with_headroom_loses_nothing(self):
        """Below k the promoting pin is order-changing only — no eviction."""
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        hits = [_rec(f'peer-{i}', topic='alpha') for i in range(K - 2)]
        index = _canonical_index(canonical, *hits)

        promoted = mod.apply_promoting_topic_anchor(hits, index)

        assert set(_ids(hits)) <= set(_ids(promoted[:K]))
        assert promoted[0].record_id == 'canon-a'


class TestPromotingAnchorNeedsNoContestedKey:
    """Arm (1) is landable today: it reads `topic` and nothing else.

    `contested` is a hand-labelled bake-off FIXTURE field — it is absent from
    the live `RESERVED_VOCABULARY_KEYS` (fused_memory/memory_metadata.py:601)
    and has no writer — so an arm that needed it would be unimplementable.
    """

    @pytest.mark.parametrize('extra_key', ['contested', 'supersedes'])
    def test_metadata_it_does_not_read_cannot_change_its_answer(self, extra_key):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True)
        plain = _rec('peer-1', topic='alpha')
        marked = _rec('peer-1', topic='alpha')
        marked.metadata[extra_key] = True

        without = mod.apply_promoting_topic_anchor(
            [plain], _canonical_index(canonical, plain),
        )
        with_key = mod.apply_promoting_topic_anchor(
            [marked], _canonical_index(canonical, marked),
        )

        assert _ids(without) == _ids(with_key)


# ---------------------------------------------------------------------------
# Arm (2): the TOPIC-KEYED grouped read
# ---------------------------------------------------------------------------
#
# `apply_grouped_read` (bake_off_storage_shape.py:837) keys on `parent_id`.
# The ratified C write shape stores PEERS — a topic's records carry `topic`
# and `canonical` and NOT a parent link (`_materialize_c_peers`, :653-694) —
# so parent-keyed grouping is structurally inert over it: every C peer is
# "parentless", passes through in place, and the grouped read measures
# nothing.  Arm (2) keys on `topic` instead, which is the corner eval-design
# OQ5 actually asked for (peers PLUS children) and which PRD D9 mislabelled
# as "already C".  It needs no `parent_id` writer, no backfill and no new
# vocabulary key.
#
# It also turns the ONE hard-coded policy in `apply_grouped_read` into a
# dial.  That function credits `[canonical, *amendments, *others]` (:927) and
# collapses sightings to a bare count, and its own comment states the fix if
# a sighting's claim is meant to be recallable: "render its body and pay the
# tokens — not to credit it unrendered".  `render_sightings` is exactly that
# knob, so the report can price the trade instead of inheriting it.


def _topic_index(*records):
    """`topic -> [records]`, the member set arm (2) resolves upward into."""
    index: dict = {}
    for record in records:
        topic = record.metadata.get('topic')
        if topic is None:
            continue
        index.setdefault(topic, []).append(record)
    return index


class TestTopicKeyedGroupsWhatParentKeyedCannot:
    """The whole point: C's peers carry `topic` and no `parent_id`."""

    def test_c_shaped_peers_group_on_topic(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        peer = _rec('peer-1', topic='alpha', kind='amendment', content='AMEND')
        hits = [canonical, peer]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert len(result.records) == 1
        assert result.records[0].content == 'CANON\n[amendment] AMEND'

    def test_the_landed_parent_keyed_read_leaves_that_same_set_flat(self):
        """Cross-check against the shipped transform, not against a claim.

        If `apply_grouped_read` ever started grouping this input, arm (2)
        would stop being the variant it is described as and the report's
        comparison would be measuring the same transform twice.
        """
        bake_off = _bake_off()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        peer = _rec('peer-1', topic='alpha', kind='amendment', content='AMEND')
        hits = [canonical, peer]
        records_by_id = {r.record_id: r for r in hits}

        grouped = bake_off.apply_grouped_read(
            hits, records_by_id, contested_ids=set(),
        )

        # Flat, and by ELEMENT IDENTITY: no `parent_id`, so nothing folded.
        assert all(g is h for g, h in zip(grouped, hits, strict=True))

    def test_a_hit_with_a_parent_link_but_no_topic_is_not_grouped(self):
        """Keyed on `topic`, full stop — `parent_id` is not a second key."""
        mod = _mod()
        orphan = _rec('child-a', topic=None, parent_id='canon-a', kind='amendment')

        result = mod.apply_topic_keyed_grouped_read(
            [orphan], _topic_index(orphan), render_sightings=False,
        )

        assert result.records[0] is orphan


class TestTopicKeyedRenderOrder:
    """Canonical body FIRST, then amendments, then every other kind."""

    def test_canonical_leads_then_amendments_then_others(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        other = _rec('other-1', topic='alpha', kind='clarification', content='CLAR')
        amendment = _rec('amend-1', topic='alpha', kind='amendment', content='AMEND')
        # Deliberately ranked out of render order: the DOCUMENT's order is a
        # rendering rule, not an echo of the store's ranking.
        hits = [other, amendment, canonical]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert result.records[0].content == (
            'CANON\n[amendment] AMEND\n[clarification] CLAR'
        )

    def test_a_child_with_no_kind_renders_as_child_not_as_amendment(self):
        """Same vocabulary as `_render_grouped_document` (:806), reused.

        A fixed `[amendment]` label would rename a record of any third kind
        inside the transform that claims to be the executable specification.
        """
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        kindless = _rec('peer-1', topic='alpha', content='PLAIN')
        hits = [canonical, kindless]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert result.records[0].content == 'CANON\n[child] PLAIN'


class TestSightingCreditingIsAnExplicitKnob:
    """`render_sightings` prices what `apply_grouped_read` hard-codes."""

    @staticmethod
    def _corpus():
        canonical = _rec('canon-a', topic='alpha', canonical=True,
                         content='CANON', claim_ids=['k0'])
        amendment = _rec('amend-1', topic='alpha', kind='amendment',
                         content='AMEND', claim_ids=['k1'])
        sighting_a = _rec('sight-1', topic='alpha', kind='sighting',
                          content='SEEN-A', claim_ids=['k2'])
        sighting_b = _rec('sight-2', topic='alpha', kind='sighting',
                          content='SEEN-B', claim_ids=['k3'])
        return [canonical, amendment, sighting_a, sighting_b]

    def test_uncredited_sightings_collapse_to_a_count_and_lose_their_claims(self):
        mod = _mod()
        hits = self._corpus()

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        document = result.records[0]
        assert document.content == 'CANON\n[amendment] AMEND\n[sightings: 2]'
        assert 'SEEN-A' not in document.content
        assert document.claim_ids == ['k0', 'k1']

    def test_credited_sightings_render_their_bodies_and_carry_their_claims(self):
        mod = _mod()
        hits = self._corpus()

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=True,
        )

        document = result.records[0]
        assert document.content == (
            'CANON\n[amendment] AMEND\n[sighting] SEEN-A\n[sighting] SEEN-B'
        )
        assert '[sightings:' not in document.content  # rendered, not counted
        assert document.claim_ids == ['k0', 'k1', 'k2', 'k3']

    def test_crediting_and_rendering_never_disagree(self):
        """The invariant `apply_grouped_read`'s comment turns on.

        A claim may be credited only if its text actually reached the
        reader; otherwise the arm banks recall AND a token discount for the
        same content — a double advantage in exactly the two columns the
        decision table is read on.
        """
        mod = _mod()
        hits = self._corpus()

        for render in (False, True):
            result = mod.apply_topic_keyed_grouped_read(
                hits, _topic_index(*hits), render_sightings=render,
            )
            document = result.records[0]
            credited_bodies = [
                r.content for r in hits if set(r.claim_ids) <= set(document.claim_ids)
            ]
            for body in credited_bodies:
                assert body in document.content

    def test_crediting_the_sightings_costs_tokens(self):
        """The knob is a DIAL between recall and cost, and it is priced."""
        mod = _mod()
        bake_off = _bake_off()
        hits = self._corpus()
        estimator = bake_off.resolve_token_estimator()

        counted = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )
        rendered = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=True,
        )

        cheap = bake_off.tokens_returned(counted.records, 5, estimator)
        dear = bake_off.tokens_returned(rendered.records, 5, estimator)
        # `tokens_returned` (:1245) returns {'tokens', 'estimator',
        # 'payloads_counted'} — the count is read under its real key rather
        # than a guessed one, so this priced comparison stays directly
        # comparable with the committed E2 table's token column.
        assert dear['tokens'] > cheap['tokens']
        # Both sides must have gone through the SAME estimator, or the
        # comparison prices two different rulers instead of two policies.
        assert dear['estimator'] == cheap['estimator'] == estimator[0]


class TestClaimRecallCeiling:
    """The knob's arithmetic, stated as an identity rather than a hope."""

    def test_uncredited_sightings_cap_recall_at_claims_minus_sightings(self):
        mod = _mod()
        corpus = TestSightingCreditingIsAnExplicitKnob._corpus()

        ceiling = mod.claim_recall_ceiling(corpus, render_sightings=False)

        # (claims - sightings + contested) / claims = (4 - 2 + 0) / 4
        assert ceiling == 0.5

    def test_a_contested_sighting_escapes_suppression_and_raises_the_ceiling(self):
        """The `+ contested` term of the identity — measured, not assumed.

        PRD V2 says a contested child is never suppressed, so its body
        survives as its own hit and its claim stays reachable.  This asserts
        the ARITHMETIC of that term; whether the term can ever be non-zero
        in production is a separate, and currently negative, question — see
        `TestTheContestedTermIsStructurallyZeroToday`.
        """
        mod = _mod()
        corpus = TestSightingCreditingIsAnExplicitKnob._corpus()

        ceiling = mod.claim_recall_ceiling(
            corpus, render_sightings=False, contested_ids={'sight-1'},
        )

        # (4 - 2 + 1) / 4
        assert ceiling == 0.75

    def test_crediting_sightings_lifts_the_ceiling_to_one(self):
        mod = _mod()
        corpus = TestSightingCreditingIsAnExplicitKnob._corpus()

        assert mod.claim_recall_ceiling(corpus, render_sightings=True) == 1.0

    def test_a_claimless_corpus_has_no_ceiling_rather_than_a_zero(self):
        """`None` is no measurement; 0.0 would read as "recalls nothing"."""
        mod = _mod()

        assert mod.claim_recall_ceiling([], render_sightings=False) is None
        assert mod.claim_recall_ceiling(
            [_rec('r1', topic='alpha')], render_sightings=False,
        ) is None

    def test_the_ceiling_equals_what_the_transform_actually_credits(self):
        """Ties the arithmetic to the transform, so the two cannot drift."""
        mod = _mod()
        corpus = TestSightingCreditingIsAnExplicitKnob._corpus()
        total = len({c for r in corpus for c in r.claim_ids})

        for render in (False, True):
            result = mod.apply_topic_keyed_grouped_read(
                corpus, _topic_index(*corpus), render_sightings=render,
            )
            credited = {c for r in result.records for c in r.claim_ids}
            assert len(credited) / total == mod.claim_recall_ceiling(
                corpus, render_sightings=render,
            )


class TestTheContestedTermIsStructurallyZeroToday:
    """Arm (2) suppresses, and cannot implement V2's protection.

    `contested` is a hand-labelled bake-off FIXTURE field: it is absent from
    the live `RESERVED_VOCABULARY_KEYS` (fused_memory/memory_metadata.py:601
    — `{topic, canonical, kind, parent_id, supersedes}`), has no writer and
    no adjudication surface.  So the `+ contested` term above is arithmetic
    that production cannot currently make non-zero, and the report says so
    rather than implying the protection exists.
    """

    def test_contested_is_not_a_live_vocabulary_key(self):
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS  # noqa: PLC0415

        assert 'contested' not in RESERVED_VOCABULARY_KEYS
        assert set(RESERVED_VOCABULARY_KEYS) == {
            'topic', 'canonical', 'kind', 'parent_id', 'supersedes',
        }

class TestTopicKeyedRanking:
    """A group lands at the BEST rank among its members."""

    def test_the_group_takes_the_min_rank_of_its_members(self):
        mod = _mod()
        lead = _rec('slab-1', topic=None, content='SLAB')
        amendment = _rec('amend-1', topic='alpha', kind='amendment', content='AMEND')
        trailer = _rec('slab-2', topic=None, content='SLAB2')
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        hits = [lead, amendment, trailer, canonical]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        # The group inherits rank 1 (its amendment), not rank 3 (its
        # canonical): demoting it to the worst member would make grouping
        # look worse at every k as an artifact of this transform.  Same rule
        # `rescore` applies to SCORE (:3246), so the two agree.
        assert _ids(result.records) == ['slab-1', 'canon-a', 'slab-2']


class TestTopicKeyedNeverInventsRetrieval:
    """Only the canonical is pulled in unranked; members are hits only."""

    def test_a_topic_member_the_store_never_returned_is_not_folded(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True,
                         content='CANON', claim_ids=['k0'])
        ranked = _rec('amend-1', topic='alpha', kind='amendment',
                      content='AMEND', claim_ids=['k1'])
        unranked = _rec('amend-2', topic='alpha', kind='amendment',
                        content='NEVER-RANKED', claim_ids=['k2'])

        result = mod.apply_topic_keyed_grouped_read(
            [canonical, ranked],
            _topic_index(canonical, ranked, unranked),
            render_sightings=False,
        )

        document = result.records[0]
        assert 'NEVER-RANKED' not in document.content
        assert 'k2' not in document.claim_ids

    def test_an_unranked_canonical_IS_pulled_in(self):
        """D6's upward resolution: a member hit must reach its anchor.

        This is the one record the index legitimately contributes — without
        it a peer hit's group would have no canonical body, which is the
        whole benefit being measured.
        """
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True,
                         content='CANON', claim_ids=['k0'])
        ranked = _rec('amend-1', topic='alpha', kind='amendment',
                      content='AMEND', claim_ids=['k1'])

        result = mod.apply_topic_keyed_grouped_read(
            [ranked], _topic_index(canonical, ranked), render_sightings=False,
        )

        assert result.records[0].content == 'CANON\n[amendment] AMEND'
        assert result.records[0].claim_ids == ['k0', 'k1']


class TestTopicKeyedSynthesizesNothingItNeedNot:
    def test_a_topic_with_no_canonical_passes_through_flat(self):
        """Never synthesize a canonical — leaf ε's uniqueness rule is not
        this transform's to invent, and anointing the best-ranked peer
        would make the document depend on ANN order."""
        mod = _mod()
        first = _rec('peer-1', topic='orphan')
        second = _rec('peer-2', topic='orphan')
        hits = [first, second]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert all(r is h for r, h in zip(result.records, hits, strict=True))

    def test_a_lone_canonical_hit_is_returned_by_identity(self):
        """A group of one IS its canonical: no synthesis, no allocation."""
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')

        result = mod.apply_topic_keyed_grouped_read(
            [canonical], _topic_index(canonical), render_sightings=False,
        )

        assert result.records[0] is canonical

    def test_an_empty_hit_list_returns_empty(self):
        assert _mod().apply_topic_keyed_grouped_read(
            [], {}, render_sightings=False,
        ).records == []


class TestTopicKeyedIsPure:
    def test_it_mutates_neither_input(self):
        mod = _mod()
        hits = TestSightingCreditingIsAnExplicitKnob._corpus()
        index = _topic_index(*hits)
        hits_before = copy.deepcopy(hits)
        index_before = copy.deepcopy(index)

        mod.apply_topic_keyed_grouped_read(hits, index, render_sightings=False)

        assert hits == hits_before
        assert index == index_before

    def test_the_document_gets_a_fresh_metadata_dict(self):
        mod = _mod()
        canonical = _rec('canon-a', topic='alpha', canonical=True, content='CANON')
        peer = _rec('peer-1', topic='alpha', kind='amendment', content='AMEND')

        result = mod.apply_topic_keyed_grouped_read(
            [canonical, peer], _topic_index(canonical, peer), render_sightings=False,
        )

        assert result.records[0].metadata is not canonical.metadata
        assert result.records[0].metadata == canonical.metadata


class TestArmTwoIsDeclaredSuppressing:
    """The flag step-19's report column reads, set at the source.

    Two DISTINCT facts, deliberately not collapsed into one boolean: does
    the transform drop ranked records, and would satisfying PRD V2 under it
    require a `contested` key?  Arm (2) is both; arm (1) is neither.
    """

    def test_arm_two_drops_ranked_records_and_would_need_contested(self):
        spec = _mod().ARM_SPECS['topic_keyed_grouped']

        assert spec.drops_ranked_records is True
        assert spec.requires_contested_key_for_v2 is True

    def test_arm_one_drops_nothing_and_needs_no_contested_key(self):
        spec = _mod().ARM_SPECS['promoting_pin']

        assert spec.drops_ranked_records is False
        assert spec.requires_contested_key_for_v2 is False
        # It still costs the window's last slot — via `read_path`'s
        # truncation, which is a different mechanism and gets its own flag
        # rather than being folded into "suppressing".
        assert spec.displaces_at_window_edge is True

    def test_grouped_members_stop_being_independent_hits(self):
        """What `drops_ranked_records is True` actually means, measured."""
        mod = _mod()
        hits = TestSightingCreditingIsAnExplicitKnob._corpus()

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert len(result.records) < len(hits)
        assert 'amend-1' not in _ids(result.records)


# ---------------------------------------------------------------------------
# Arm (3): the TOPIC-DIVERSITY CAP (MMR-style)
# ---------------------------------------------------------------------------
#
# The cheapest member of the family: it renders nothing, credits nothing and
# synthesizes nothing.  It only ever DROPS, which is why it is a suppressing
# read — and yet it reads only `metadata['topic']`, so unlike arm (2) it is
# landable today with no `contested` key in existence.  Those two facts are
# what step-19's report column has to state separately.


def _capped_corpus():
    """Two crowded topics plus untopiced slab records.

    Rank order is the list order.  `t-a` is crowded with its canonical
    ranked SECOND, behind a peer — the case that separates "keep the
    best-ranked member" from "keep the canonical".
    """
    return [
        _rec('a-peer', topic='t-a', content='a peer body'),
        _rec('a-canon', topic='t-a', canonical=True, content='a canonical body'),
        _rec('slab-1', topic=None, content='distractor one'),
        _rec('b-canon', topic='t-b', canonical=True, content='b canonical body'),
        _rec('b-peer', topic='t-b', content='b peer body'),
        _rec('slab-2', topic=None, content='distractor two'),
    ]


class TestTopicDiversityCapKeepsOnePerTopic:
    """At most `per_topic` records of any one topic survive the window."""

    def test_at_most_one_record_per_topic_survives(self):
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        topics = [r.metadata['topic'] for r in kept if 'topic' in r.metadata]
        assert sorted(topics) == ['t-a', 't-b']

    def test_per_topic_is_a_dial_not_a_hardcoded_one(self):
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(
            hits, _canonical_index(*hits), per_topic=2,
        )

        # Both topics had exactly two members, so a cap of 2 drops nothing.
        assert _ids(kept) == _ids(hits)

    def test_the_canonical_wins_even_when_a_peer_outranked_it(self):
        """Diversity picks the ANCHOR, not merely the luckiest ANN result.

        `a-peer` ranks above `a-canon`.  Keeping the peer would hand the
        reader one arbitrary member of the topic; keeping the canonical
        hands them the topic's anchor, which is the whole reason the write
        shape marks one.
        """
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        assert 'a-canon' in _ids(kept)
        assert 'a-peer' not in _ids(kept)

    def test_the_survivor_sits_at_the_topics_BEST_member_rank(self):
        """Best-rank, agreeing with arm (2) and with `rescore` (:3246).

        Demoting the survivor to the canonical's own (worse) rank would
        make the arm look worse at every k as an artifact of THIS
        transform rather than of the read shape.
        """
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        # `a-peer` held rank 0; its topic keeps that slot, with `a-canon` in it.
        assert _ids(kept) == ['a-canon', 'slab-1', 'b-canon', 'slab-2']

    def test_a_topic_whose_canonical_never_ranked_keeps_its_best_peer(self):
        """The cap only ever DROPS — it never pulls an unranked record in.

        Injecting the absent canonical here would be arm (1)'s promotion,
        and would credit this arm for retrieval the store never performed.
        """
        mod = _mod()
        absent_canonical = _rec('c-canon', topic='t-c', canonical=True)
        hits = [
            _rec('c-peer-lo', topic='t-c', content='c lower'),
            _rec('c-peer-hi', topic='t-c', content='c higher'),
        ]

        kept = mod.apply_topic_diversity_cap(
            hits, _canonical_index(*hits, absent_canonical),
        )

        assert _ids(kept) == ['c-peer-lo']
        assert 'c-canon' not in _ids(kept)


class TestTheDistractorSlabIsNeverCapped:
    """Untopiced records are invisible to a topic cap, by construction.

    `_distractor_records` (:545) stamps `category` and NOT ONE reserved
    vocabulary key — deliberately, so a distractor cannot become a right
    answer.  A cap that quietly collapsed them to one would delete the
    contamination variable the whole bake-off is controlled on.
    """

    def test_every_untopiced_record_survives_by_identity(self):
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        slab = [r for r in hits if 'topic' not in r.metadata]
        assert len(slab) == 2
        for record in slab:
            assert any(out is record for out in kept)

    def test_an_all_slab_window_is_returned_completely_untouched(self):
        mod = _mod()
        hits = [_rec(f'slab-{i}', topic=None, content=f'd{i}') for i in range(5)]

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        assert len(kept) == len(hits)
        for out, original in zip(kept, hits, strict=True):
            assert out is original


class TestTheCapSynthesizesNothing:
    """No rendering, no crediting, no new records — the cheapest arm."""

    def test_every_returned_record_is_an_input_record_by_identity(self):
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        for out in kept:
            assert any(out is original for original in hits), (
                f'{out.record_id} was synthesized; the cap must only drop'
            )

    def test_it_returns_a_bare_list_not_a_provenance_carrying_result(self):
        """Nothing is aliased, so there is nothing to disclose.

        Arm (2) returns a `TransformResult` because it MINTS documents under
        a canonical's record_id.  The cap mints nothing, so a provenance
        channel here would be permanently empty ceremony.
        """
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        assert isinstance(kept, list)
        assert not hasattr(kept, 'provenance')


class TestTheCapReadsNoContestedKey:
    """Why this arm is landable today and arm (2) is not."""

    def test_a_contested_looking_metadata_key_changes_nothing(self):
        mod = _mod()
        hits = _capped_corpus()
        marked = copy.deepcopy(hits)
        for record in marked:
            record.metadata['contested'] = True

        assert _ids(mod.apply_topic_diversity_cap(
            marked, _canonical_index(*marked),
        )) == _ids(mod.apply_topic_diversity_cap(hits, _canonical_index(*hits)))


class TestTheCapFreesWindowSlots:
    """The token consequence, pinned rather than left implicit.

    The cap returns a SHORTER LIST; it does not pad.  So at a fixed reader
    budget k the freed slots are BACKFILLED from deeper in the ranking —
    records that ranked below k before the transform are now inside it.
    That is the behaviour chosen, and it is asserted both ways round so it
    cannot drift into silent truncation.
    """

    def test_the_returned_list_is_shorter_than_the_input(self):
        mod = _mod()
        hits = _capped_corpus()

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        assert len(kept) == 4
        assert len(kept) < len(hits)

    def test_a_record_from_below_k_is_backfilled_into_the_window(self):
        mod = _mod()
        hits = _capped_corpus()
        k = 3

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        before = _ids(hits[:k])
        after = _ids(kept[:k])
        assert 'b-canon' not in before
        assert 'b-canon' in after, 'the freed slot must backfill, not blank'

    def test_capping_lowers_the_token_bill_at_a_fixed_budget(self):
        """Why the freed slot is worth measuring at all."""
        mod = _mod()
        bake_off = _bake_off()
        estimator = bake_off.resolve_token_estimator()
        hits = [
            _rec('a-1', topic='t-a', content='x' * 400),
            _rec('a-2', topic='t-a', canonical=True, content='y' * 400),
        ]

        kept = mod.apply_topic_diversity_cap(hits, _canonical_index(*hits))

        flat = bake_off.tokens_returned(hits, 5, estimator)
        capped = bake_off.tokens_returned(kept, 5, estimator)
        assert capped['tokens'] < flat['tokens']
        assert capped['payloads_counted'] == 1


class TestTheCapIsPure:
    """No mutation of the hits, the index, or any record."""

    def test_neither_the_hits_nor_the_index_nor_a_record_is_mutated(self):
        mod = _mod()
        hits = _capped_corpus()
        index = _canonical_index(*hits)
        hits_before = copy.deepcopy(hits)
        index_before = copy.deepcopy(index)

        mod.apply_topic_diversity_cap(hits, index)

        assert _ids(hits) == _ids(hits_before)
        assert [r.metadata for r in hits] == [r.metadata for r in hits_before]
        assert [r.content for r in hits] == [r.content for r in hits_before]
        assert sorted(index) == sorted(index_before)

    def test_it_is_deterministic_across_repeated_application(self):
        mod = _mod()
        hits = _capped_corpus()
        index = _canonical_index(*hits)

        once = mod.apply_topic_diversity_cap(hits, index)
        twice = mod.apply_topic_diversity_cap(hits, index)
        assert _ids(once) == _ids(twice)
        # Idempotent: capping an already-capped window is a no-op.
        assert _ids(mod.apply_topic_diversity_cap(once, index)) == _ids(once)


class TestArmThreeSuppressesYetIsLandableToday:
    """The distinction step-19's report column exists to make.

    Arm (3) DROPS ranked records — it is a suppressing read in the PRD V2
    sense — and yet it needs no `contested` key to compute, so it can ship
    today.  Collapsing those into one "suppressing" boolean would tell 3111
    the opposite of the truth.
    """

    def test_it_drops_ranked_records_but_needs_no_contested_key(self):
        spec = _mod().ARM_SPECS['topic_diversity_cap']

        assert spec.drops_ranked_records is True
        assert spec.requires_contested_key_for_v2 is False
        assert spec.displaces_at_window_edge is False

    def test_all_three_arms_are_declared(self):
        """The decision table cannot render an arm nobody declared."""
        assert sorted(_mod().ARM_SPECS) == [
            'promoting_pin', 'topic_diversity_cap', 'topic_keyed_grouped',
        ]


# ---------------------------------------------------------------------------
# Killing the record-id aliasing in the metric
# ---------------------------------------------------------------------------
#
# `topic_discoverability` (:1116) credits the canonical on
# `hit.record_id == canonical_record_id`, and a grouped read emits its
# document UNDER THE CANONICAL'S OWN record_id (`apply_grouped_read`:934-935,
# and arm (2) here for the same reason — the document IS the canonical,
# amended).  So any member folding upward is scored as "canonical found",
# whether or not the canonical's own stored record ever ranked.  That is the
# mechanism behind b_grouped's 0.97 canonical-in-top-5 in the committed E2
# table, and it is a property of the TRANSFORM, not of retrieval.
#
# The fix is not to redefine the column — that would silently make the new
# table incomparable with the committed one, and would hide that a grouped
# read really does put the canonical's body in the reader's window, which is
# a genuine reader benefit.  Both rates are reported side by side, and the
# GAP between them IS the aliasing, quantified per arm.


def _aliasing_corpus(canonical_ranked: bool):
    """A topic whose canonical did — or did not — itself rank.

    The only difference between the two worlds is whether the store
    returned the canonical.  Everything the transform does is identical,
    which is what isolates aliasing as the cause of any divergence.
    """
    canonical = _rec('anchor', topic='t-a', canonical=True, content='the anchor')
    peer = _rec('peer-1', topic='t-a', kind='amendment', content='an amendment')
    hits = [peer, canonical] if canonical_ranked else [peer]
    return hits, _topic_index(peer, canonical), canonical


class TestSynthesizedDocumentsAlwaysCarryProvenance:
    """A minted record must never be indistinguishable from a stored one."""

    def test_every_synthesized_record_has_a_provenance_entry(self):
        mod = _mod()
        hits, index, _ = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )

        for record in result.records:
            if any(record is original for original in hits):
                continue  # a stored record, passed through — nothing aliased
            assert record.record_id in result.provenance, (
                f'{record.record_id} was synthesized with no disclosure'
            )

    def test_the_disclosure_names_the_members_and_the_canonical_fact(self):
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        disclosure = result.provenance[canonical.record_id]

        assert disclosure.aliased_from == ('peer-1',)
        assert disclosure.canonical_was_itself_ranked is False

    def test_a_pass_through_record_is_disclosed_as_nothing(self):
        """No fold, no provenance — the channel must not cry wolf."""
        mod = _mod()
        hits = [_rec('slab-1', topic=None)]

        result = mod.apply_topic_keyed_grouped_read(
            hits, _topic_index(*hits), render_sightings=False,
        )

        assert result.provenance == {}


class TestCanonicalDiscoverabilityReportsBothRates:
    """One column that cannot be read two ways becomes two columns."""

    def test_it_returns_an_aliased_and_an_unaliased_verdict(self):
        mod = _mod()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        measured = mod.canonical_discoverability(
            hits, 't-a', canonical.record_id, k=5, retrieved=hits,
        )

        assert set(measured) >= {
            'aliased_in_top_k', 'aliased_rank',
            'unaliased_in_top_k', 'unaliased_rank',
        }

    def test_a_flat_arm_reports_the_two_identically(self):
        """The column stays comparable across arms.

        A flat read synthesizes nothing, so there is nothing to alias and
        the two rates MUST agree.  If they could differ here, the pair
        would be measuring something other than aliasing.
        """
        mod = _mod()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        measured = mod.canonical_discoverability(
            hits, 't-a', canonical.record_id, k=5, retrieved=hits,
        )

        assert measured['aliased_in_top_k'] == measured['unaliased_in_top_k']
        assert measured['aliased_rank'] == measured['unaliased_rank']

    def test_it_agrees_with_the_landed_metric_on_the_aliased_half(self):
        """The legacy semantics are preserved exactly, not re-derived.

        Cross-checked against the LANDED `topic_discoverability` rather
        than against a restatement of it, so the two cannot drift and the
        new table's aliased column stays comparable with the committed one.
        """
        mod = _mod()
        bake_off = _bake_off()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        measured = mod.canonical_discoverability(
            hits, 't-a', canonical.record_id, k=5, retrieved=hits,
        )
        landed = bake_off.topic_discoverability(hits, 't-a', canonical.record_id, 5)

        assert measured['aliased_in_top_k'] == landed['canonical_in_top_k']
        assert measured['aliased_rank'] == landed['canonical_rank']

    def test_an_absent_canonical_is_found_by_neither(self):
        mod = _mod()
        hits = [_rec('unrelated', topic='t-z')]

        measured = mod.canonical_discoverability(
            hits, 't-a', 'anchor', k=5, retrieved=hits,
        )

        assert measured['aliased_in_top_k'] is False
        assert measured['unaliased_in_top_k'] is False
        # `None`, never 0 — a 0 rank would collide with a real rank under a
        # 0-based reading and average as "very good" in a mean-rank summary.
        assert measured['aliased_rank'] is None
        assert measured['unaliased_rank'] is None

    def test_ranked_but_outside_the_window_is_not_the_same_as_absent(self):
        """Inherited from `topic_discoverability`: two different findings."""
        mod = _mod()
        canonical = _rec('anchor', topic='t-a', canonical=True)
        hits = [_rec(f'slab-{i}', topic=None) for i in range(5)] + [canonical]

        measured = mod.canonical_discoverability(
            hits, 't-a', 'anchor', k=5, retrieved=hits,
        )

        assert measured['unaliased_in_top_k'] is False
        assert measured['unaliased_rank'] == 6


class TestTheTwoRatesDiverge:
    """THE DECISIVE TEST.

    If no input can make aliased and unaliased differ, the pair is not
    measuring what it claims and the second column is decoration.
    """

    def test_a_fold_credits_the_aliased_rate_and_not_the_unaliased_one(self):
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        measured = mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=result.provenance,
        )

        # The store never returned the canonical...
        assert canonical.record_id not in _ids(hits)
        # ...yet the document carries its record_id, so the legacy column
        # scores it found.  That is the aliasing, isolated.
        assert measured['aliased_in_top_k'] is True
        assert measured['aliased_rank'] == 1
        # The honest column refuses the credit.
        assert measured['unaliased_in_top_k'] is False
        assert measured['unaliased_rank'] is None

    def test_the_gap_closes_when_the_canonical_really_did_rank(self):
        """Same transform, same fold — only retrieval differs.

        This is the control for the test above: a grouped read is NOT
        inherently aliasing, it aliases only when it folds a canonical the
        store never surfaced.  So the unaliased column credits a real
        retrieval even when it arrives inside a synthesized document.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=True)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        measured = mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=result.provenance,
        )

        assert result.provenance[canonical.record_id].canonical_was_itself_ranked
        assert measured['aliased_in_top_k'] is True
        assert measured['unaliased_in_top_k'] is True

    def test_without_the_provenance_the_synthesis_fact_cannot_be_told(self):
        """Aliasing is invisible in the record alone.

        A synthesized document is byte-indistinguishable from a stored one
        at the `record_id` level — that is the entire problem.  So passing
        no provenance must NOT quietly report the credit as genuine: the
        SYNTHESIS fact reports `None`, which the report renders as
        no-measurement rather than as a measured zero.

        The unaliased RANK is unaffected either way: it is read from
        *retrieved*, which no transform and no disclosure can touch.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        measured = mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5, retrieved=hits,
        )

        assert measured['aliased_in_top_k'] is True
        assert measured['aliased_credit_is_synthesized'] is None
        # Retrieval never returned it, so the honest column says so — from
        # `retrieved`, not from the absent disclosure.
        assert measured['unaliased_in_top_k'] is False
        assert measured['unaliased_rank'] is None


class TestTheMetricIsPure:
    def test_it_mutates_neither_the_hits_nor_the_provenance(self):
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)
        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        records_before = _ids(result.records)
        provenance_before = dict(result.provenance)
        retrieved_before = _ids(hits)

        mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=result.provenance,
        )

        assert _ids(result.records) == records_before
        assert result.provenance == provenance_before
        assert _ids(hits) == retrieved_before


class TestOnlyASynthesizedRecordCanAlias:
    """The discriminator behind the SYNTHESIS field, not behind the rank.

    A missing `provenance` must not mean "unknown" unconditionally: a
    STORED record carrying the canonical's id simply IS the canonical, and
    reporting `None` for every flat arm would render the synthesis column
    as no-measurement across most of the table.  Only a SYNTHESIZED
    document can alias, and a synthesized document is self-identifying by
    `role`.

    Note what this class no longer governs: the unaliased RANK.  That is
    read from *retrieved* and is therefore knowable for every arm whether
    or not anything disclosed anything — which is the whole point of
    step-23.
    """

    def test_a_stored_canonical_needs_no_disclosure_to_be_credited(self):
        mod = _mod()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        measured = mod.canonical_discoverability(
            hits, 't-a', canonical.record_id, k=5, retrieved=hits,
        )

        assert measured['unaliased_in_top_k'] is True
        assert measured['unaliased_rank'] == 2
        # Nothing was minted, so the credit is not a synthesis — and that
        # is knowable without a disclosure.
        assert measured['aliased_credit_is_synthesized'] is False

    def test_a_synthesized_document_without_disclosure_is_unknown_not_credited(
        self,
    ):
        """The fallback must fail CLOSED, never open."""
        mod = _mod()
        bake_off = _bake_off()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        document = result.records[0]

        # Self-identifying: the role is what the metric reads, so pin it.
        assert document.role == bake_off.GROUPED_ROLE
        assert mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5, retrieved=hits,
        )['aliased_credit_is_synthesized'] is None

    def test_disclosure_beats_the_role_fallback_when_both_are_present(self):
        """Provenance is EVIDENCE; the role is only a discriminator.

        A grouped document whose canonical really did rank is not a
        synthetic credit — the role alone would have called it one, so the
        disclosure must win.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=True)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )

        assert mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=result.provenance,
        )['aliased_credit_is_synthesized'] is False


# ---------------------------------------------------------------------------
# The unaliased column measures RETRIEVAL, not the transform's placement
# ---------------------------------------------------------------------------
#
# The whole reason `canonical_discoverability` exists is to separate "the
# store found the canonical" from "the transform put something carrying its
# id at the top".  Seeding the unaliased verdict from `aliased_rank` — the
# POST-transform rank — and only ever revising it downward on a disclosure
# reintroduced exactly that conflation: measured on this branch, all three
# transform arms reported `unaliased_rank=1` for a canonical retrieval had
# left at rank 5, and the committed artifact carries the damage
# (`canonical_unaliased_rank` 1.0596 / 1.0455 / 1.0455 against flat_read's
# 3.4935).  It was load-bearing, not cosmetic: `_rank_key` sorted on
# `canonical_unaliased_in_top_k` FIRST, so the corrupted column picked the
# recommended arm.
#
# The fix is structural rather than a correction factor: the unaliased half
# is computed from the PRE-transform hit list alone, so no transform can
# reach it.  Its arm-invariance is now the DEFINITION — a test that can make
# it vary proves the metric is reading the transform again.

def _retrieval_corpus():
    """Four children ranked above the canonical they belong to.

    Retrieval put the canonical at rank 5.  Every transform in the arm set
    puts it — or a document minted under its id — at aliased rank 1.  The
    distance between those two facts is precisely what the two columns
    exist to keep apart.
    """
    children = [
        _rec(f'child-{i}', topic='t-a', kind='amendment', content=f'body {i}')
        for i in range(1, 5)
    ]
    canonical = _rec('anchor', topic='t-a', canonical=True, content='the anchor')
    return [*children, canonical], canonical


def _ranked_hits(records):
    """Wrap stored records as the `ScoredHit`s the arm driver consumes.

    Scores descend with rank and are never read by any metric here
    (`ScoredHit` exists to quarantine them, bake_off:1346-1355); they exist
    only because the fetch cache's element type carries one.
    """
    return [
        _bake_off().ScoredHit(record=record, relevance_score=1.0 - index / 100)
        for index, record in enumerate(records)
    ]


def _seeded(records):
    """The two attributes `apply_arm` reads off a `SeededArm`.

    A stand-in rather than a real `SeededArm`: that class carries eight
    more fields describing a LIVE mem0 collection, none of which any read
    transform touches, and constructing one here would imply this test
    needs a store.
    """
    return types.SimpleNamespace(
        canonical_by_topic=_canonical_index(*records),
        records=list(records),
    )


class TestTheUnaliasedColumnIsReadFromRetrieval:
    """It cannot be computed from the delivered window at all."""

    def test_the_pre_transform_hits_are_a_required_argument(self):
        """Not defaulted: a caller that forgets them must not get a number.

        Defaulting to the window would silently restore the conflation for
        every call site that had not been updated — which is how the bug
        reached a committed artifact in the first place.
        """
        mod = _mod()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        with pytest.raises(TypeError):
            mod.canonical_discoverability(hits, 't-a', canonical.record_id, k=5)

    def test_the_verdict_follows_retrieval_while_the_window_stays_fixed(self):
        """The decisive shape: one window, two retrievals, two answers."""
        mod = _mod()
        canonical = _rec('anchor', topic='t-a', canonical=True)
        window = [canonical]
        shallow = [canonical]
        deep = [_rec('slab-1', topic=None), _rec('slab-2', topic=None), canonical]

        near = mod.canonical_discoverability(
            window, 't-a', 'anchor', k=5, retrieved=shallow,
        )
        far = mod.canonical_discoverability(
            window, 't-a', 'anchor', k=5, retrieved=deep,
        )

        assert near['unaliased_rank'] == 1
        assert far['unaliased_rank'] == 3
        # ...while the window never moved, so the aliased half is identical.
        assert near['aliased_rank'] == far['aliased_rank'] == 1


class TestUnaliasedIsArmInvariantByConstruction:
    """Four arms, one retrieval, one honest rank."""

    def test_every_arm_reports_the_canonicals_real_retrieval_rank(self):
        mod = _mod()
        records, canonical = _retrieval_corpus()
        hits, seeded = _ranked_hits(records), _seeded(records)

        measured = {}
        for arm in mod.ARM_KEYS:
            transform = mod.apply_arm(arm, hits, seeded, 5)
            measured[arm] = mod.canonical_discoverability(
                transform.records[:5], 't-a', canonical.record_id, k=5,
                retrieved=records, provenance=transform.provenance,
            )

        assert {arm: row['unaliased_rank'] for arm, row in measured.items()} == {
            'flat_read': 5,
            'promoting_pin': 5,
            'topic_keyed_grouped': 5,
            'topic_diversity_cap': 5,
        }
        # ...and the aliased column still discriminates, which is why both
        # are printed.  Placement is a real, reportable difference; it is
        # simply not a retrieval one.
        assert {arm: row['aliased_rank'] for arm, row in measured.items()} == {
            'flat_read': 5,
            'promoting_pin': 1,
            'topic_keyed_grouped': 1,
            'topic_diversity_cap': 1,
        }

    def test_no_arm_pulls_the_canonical_into_a_window_retrieval_left_it_out_of(
        self,
    ):
        """The `in_top_k` half of the same fact.

        Asserted at k=4 rather than at the k=5 of the test above because
        `canonical_in_top_k` is `rank <= k` (bake_off:1178): with the
        canonical AT rank 5 and k=5 a `False` is arithmetically impossible,
        so the honest scenario is a window strictly narrower than its
        retrieval rank.
        """
        mod = _mod()
        records, canonical = _retrieval_corpus()
        hits, seeded = _ranked_hits(records), _seeded(records)

        measured = {}
        for arm in mod.ARM_KEYS:
            transform = mod.apply_arm(arm, hits, seeded, 4)
            measured[arm] = mod.canonical_discoverability(
                transform.records[:4], 't-a', canonical.record_id, k=4,
                retrieved=records, provenance=transform.provenance,
            )

        assert [row['unaliased_in_top_k'] for row in measured.values()] == [
            False, False, False, False,
        ]
        # The pin and the grouped read both DELIVER it inside a four-record
        # window retrieval had excluded — a real reader benefit, credited in
        # the aliased column and nowhere else.
        assert measured['promoting_pin']['aliased_in_top_k'] is True
        assert measured['topic_keyed_grouped']['aliased_in_top_k'] is True
        assert measured['flat_read']['aliased_in_top_k'] is False

    def test_the_two_columns_are_not_the_same_number(self):
        """If they never diverged, one of them would be decoration."""
        mod = _mod()
        records, canonical = _retrieval_corpus()
        hits, seeded = _ranked_hits(records), _seeded(records)

        transform = mod.apply_arm('promoting_pin', hits, seeded, 5)
        row = mod.canonical_discoverability(
            transform.records[:5], 't-a', canonical.record_id, k=5,
            retrieved=records, provenance=transform.provenance,
        )

        assert row['aliased_rank'] != row['unaliased_rank']


class TestUnaliasedReadsTheUntruncatedFetch:
    """A canonical below the window is a rank, not an absence."""

    def test_a_canonical_at_retrieval_rank_k_plus_two_reports_that_rank(self):
        mod = _mod()
        canonical = _rec('anchor', topic='t-a', canonical=True)
        records = [_rec(f'slab-{i}', topic=None) for i in range(6)] + [canonical]

        row = mod.canonical_discoverability(
            records[:5], 't-a', 'anchor', k=5, retrieved=records,
        )

        assert row['unaliased_rank'] == 7

    def test_the_flat_baseline_could_not_have_expressed_that_rank(self):
        """Why `baseline` is NOT reused as the pre-transform list.

        `baseline` is `apply_arm('flat_read', ...).records`, already sliced
        to k, so a canonical below the window is indistinguishable there
        from one absent from the corpus — the exact conflation
        `topic_discoverability`'s docstring says the rank exists to avoid.
        """
        mod = _mod()
        bake_off = _bake_off()
        canonical = _rec('anchor', topic='t-a', canonical=True)
        records = [_rec(f'slab-{i}', topic=None) for i in range(6)] + [canonical]
        hits, seeded = _ranked_hits(records), _seeded(records)

        baseline = mod.apply_arm('flat_read', hits, seeded, 5).records

        assert bake_off.topic_discoverability(
            baseline, 't-a', 'anchor', 5,
        )['canonical_rank'] is None
        assert mod.canonical_discoverability(
            records[:5], 't-a', 'anchor', k=5, retrieved=records,
        )['unaliased_rank'] == 7

    def test_a_none_rank_carries_the_depth_it_was_looked_for_in(self):
        """`None` must never be readable as "absent from the corpus".

        It means "absent from the cached fetch", and a fetch has a depth.
        Without it a reader cannot tell a canonical that lost from one that
        was never fetched deeply enough to win.
        """
        mod = _mod()
        records = [_rec(f'slab-{i}', topic=None) for i in range(3)]

        row = mod.canonical_discoverability(
            records, 't-a', 'anchor', k=5, retrieved=records,
        )

        assert row['unaliased_rank'] is None
        assert row['unaliased_in_top_k'] is False
        assert row['retrieval_depth'] == 3

    def test_the_depth_is_reported_even_when_the_canonical_was_found(self):
        mod = _mod()
        records, canonical = _retrieval_corpus()

        row = mod.canonical_discoverability(
            records, 't-a', canonical.record_id, k=5, retrieved=records,
        )

        assert row['retrieval_depth'] == 5


class TestTheSynthesisFactKeepsItsOwnColumn:
    """The arm-dependent half is not deleted — it is renamed and isolated.

    "The record delivered under the canonical's id is a document the
    transform minted, not the stored record" is a true and reportable fact
    about an arm.  It was doing the wrong job as a rank; it keeps doing the
    right one beside it.
    """

    def test_a_fold_over_an_unretrieved_canonical_is_flagged_as_synthesized(self):
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        row = mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=result.provenance,
        )

        assert row['aliased_credit_is_synthesized'] is True
        # ...and the rank column is untouched by that fact.
        assert row['unaliased_rank'] is None

    def test_a_flat_arm_synthesizes_nothing(self):
        mod = _mod()
        records, canonical = _retrieval_corpus()
        hits, seeded = _ranked_hits(records), _seeded(records)

        transform = mod.apply_arm('flat_read', hits, seeded, 5)
        row = mod.canonical_discoverability(
            transform.records[:5], 't-a', canonical.record_id, k=5,
            retrieved=records, provenance=transform.provenance,
        )

        assert row['aliased_credit_is_synthesized'] is False

    def test_it_is_the_column_that_varies_by_arm_now(self):
        """The two columns swap roles, and neither is redundant.

        `unaliased_*` is constant across arms and `aliased_credit_is_
        synthesized` varies — the reverse of the pre-fix state, where the
        rank varied by arm and the synthesis fact was folded into it.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)
        seeded = _seeded([*hits, canonical])
        scored = _ranked_hits(hits)

        grouped = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        flat = mod.apply_arm('flat_read', scored, seeded, 5)

        folded = mod.canonical_discoverability(
            grouped.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=grouped.provenance,
        )
        plain = mod.canonical_discoverability(
            flat.records, 't-a', canonical.record_id, k=5,
            retrieved=hits, provenance=flat.provenance,
        )

        assert folded['aliased_credit_is_synthesized'] is True
        assert plain['aliased_credit_is_synthesized'] is False
        assert folded['unaliased_rank'] == plain['unaliased_rank'] is None


class TestTheScorerThreadsTheUntruncatedFetch:
    """The guard is wired, not merely available.

    `score_arm` holds the full cached hit list; `apply_arm` slices it to k
    before transforming.  If the scorer passed the sliced list — or reused
    `baseline`, which is already sliced — the metric would be correct and
    the report still wrong.
    """

    def _query(self, topic='t-a', query_id='q-1'):
        return types.SimpleNamespace(
            query_id=query_id, topic=topic, expects_claim_ids=['claim-1'],
        )

    def test_every_arm_scores_the_canonical_at_its_full_fetch_rank(self):
        mod = _mod()
        canonical = _rec('anchor', topic='t-a', canonical=True,
                         claim_ids=['claim-1'])
        records = [
            _rec(f'child-{i}', topic='t-a', kind='amendment') for i in range(1, 7)
        ] + [canonical]
        hits, seeded = _ranked_hits(records), _seeded(records)
        query = self._query()

        ranks = {}
        for arm in mod.ARM_KEYS:
            scored = mod.score_arm(
                arm, [query], {'q-1': hits}, seeded, k=5, labeled=True,
            )
            ranks[arm] = scored['rows'][0]['canonical_unaliased_rank']

        # Rank 7 in a 7-deep fetch, scored at k=5: only the untruncated list
        # can say 7, and every arm must say it.
        assert set(ranks.values()) == {7}

    def test_the_row_still_reports_the_arms_own_placement(self):
        mod = _mod()
        records, canonical = _retrieval_corpus()
        hits, seeded = _ranked_hits(records), _seeded(records)
        query = self._query()

        flat = mod.score_arm(
            'flat_read', [query], {'q-1': hits}, seeded, k=5, labeled=True,
        )['rows'][0]
        pinned = mod.score_arm(
            'promoting_pin', [query], {'q-1': hits}, seeded, k=5, labeled=True,
        )['rows'][0]

        assert flat['canonical_aliased_rank'] == 5
        assert pinned['canonical_aliased_rank'] == 1
        assert flat['canonical_unaliased_rank'] == 5
        assert pinned['canonical_unaliased_rank'] == 5


class TestTheRankingRuleDoesNotSortOnAConstant:
    """A primary sort key that cannot vary across arms decides nothing.

    With the unaliased column now arm-invariant BY CONSTRUCTION, leading
    `_rank_key` with it would leave the recommendation to be settled by
    whatever happened to come next — a rule that reads as measured but is
    not.  It is dropped from the ordering and reported beside it.
    """

    def test_the_ranking_ignores_the_arm_invariant_canonical_columns(self):
        mod = _mod()
        left = {
            'canonical_unaliased_in_top_k': 1.0,
            'canonical_aliased_in_top_k': 1.0,
            'claim_recall': 0.5,
            'tokens_per_query': 100.0,
            'window_displacement': 0.0,
        }
        right = dict(left)
        right['canonical_unaliased_in_top_k'] = 0.0
        right['canonical_aliased_in_top_k'] = 0.0

        assert mod._rank_key('a', left)[:-1] == mod._rank_key('a', right)[:-1]

    def test_claim_recall_leads_the_ordering(self):
        mod = _mod()
        better = {'claim_recall': 0.9, 'tokens_per_query': 900.0,
                  'window_displacement': 9.0}
        worse = {'claim_recall': 0.4, 'tokens_per_query': 1.0,
                 'window_displacement': 0.0}

        assert mod._rank_key('a', better) < mod._rank_key('a', worse)

# ---------------------------------------------------------------------------
# The UNLABELED production scoring path
# ---------------------------------------------------------------------------
#
# The E2 query set is blind-authored but LABELED: every `Query` carries
# `expects_claim_ids` (:199), `cross_validate_fixtures` (:477) requires it
# non-empty, and `load_query_set` (:368) validates `kind in QUERY_KINDS`.
# Production queries have none of that — the journal records what was asked,
# never what should have come back.
#
# So they cannot ride the E2 loader, and the two sets must not silently
# merge: a production row entering the labeled corpus would either crash the
# cross-validator or, worse, be scored against an empty expectation set and
# report a recall.  The loaders therefore reject each other's rows BY NAME,
# and the labeled metrics return None — no measurement, never a measured
# zero — for every unlabeled query.


PRODUCTION_FIXTURE = FIXTURES_DIR / 'production_query_sample.jsonl'


def _prod_rows(tmp_path, rows):
    """Write `rows` as a production-shaped JSONL fixture."""
    import json  # noqa: PLC0415

    path = tmp_path / 'prod.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    return path


def _prod_row(query_id='prod-brief-abc', text='project overview architecture goals',
              source='briefing_template', observed_count: object = 74103,
              traffic_share: object = 0.172892, observed_limit=5):
    """Build one production-shaped JSONL row.

    `observed_count` and `traffic_share` are annotated `object`, not narrowed
    to their well-formed types, because this helper's job includes building
    the MALFORMED rows the loader's type refusals are pinned against
    (`traffic_share=None` / `'0.17'` / `True`, `observed_count='74103'`).
    The row is raw JSON on its way to `json.dumps`; the types are the
    loader's to enforce at parse time, which is precisely what those tests
    assert it does.
    """
    return {
        'query_id': query_id,
        'text': text,
        'source': source,
        'observed_count': observed_count,
        'traffic_share': traffic_share,
        'observed_limit': observed_limit,
    }


class TestTheTwoQuerySetsDoNotSilentlyMerge:
    """A labeled loader and an unlabeled loader, each refusing the other's rows."""

    def test_the_production_loader_accepts_label_free_rows(self, tmp_path):
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row()])
        queries = mod.load_production_queries(path)
        assert len(queries) == 1
        assert queries[0].text == 'project overview architecture goals'
        assert queries[0].observed_limit == 5

    def test_the_committed_production_fixture_loads(self):
        """The real harvested sample is loadable as-is."""
        mod = _mod()
        queries = mod.load_production_queries(PRODUCTION_FIXTURE)
        assert len(queries) >= 4
        briefing = [q for q in queries if q.source == 'briefing_template']
        assert len(briefing) == 4

    def test_the_e2_loader_rejects_a_production_row_by_name(self, tmp_path):
        """`load_query_set` must refuse an unlabeled row, loudly."""
        mod, bake = _mod(), _bake_off()
        path = _prod_rows(tmp_path, [_prod_row()])
        with pytest.raises(bake.FixtureError) as exc:
            bake.load_query_set(path)
        assert 'kind' in str(exc.value)
        assert mod # the two loaders coexist; neither shadows the other

    def test_the_production_loader_rejects_a_labeled_row_by_name(self, tmp_path):
        """The symmetric guard: an E2 row must not sneak into the unlabeled set.

        Without this, a labeled row would be scored on the unlabeled path and
        its ground truth silently discarded — the merge this pair exists to
        prevent, in the other direction.
        """
        mod = _mod()
        row = _prod_row()
        row['expects_claim_ids'] = ['claim-1']
        path = _prod_rows(tmp_path, [row])
        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)
        assert 'expects_claim_ids' in str(exc.value)

    def test_a_production_query_exposes_no_expectation_attribute(self, tmp_path):
        """Not merely absent from the file — absent from the object.

        An `expects_claim_ids` attribute defaulting to `[]` would let a
        scorer compute `0/0` and report a recall for a query that has no
        ground truth at all.
        """
        mod = _mod()
        queries = mod.load_production_queries(_prod_rows(tmp_path, [_prod_row()]))
        assert not hasattr(queries[0], 'expects_claim_ids')
        assert not hasattr(queries[0], 'topic')

    def test_a_duplicate_query_id_is_refused(self, tmp_path):
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(), _prod_row()])
        with pytest.raises(mod.ProductionQuerySetError):
            mod.load_production_queries(path)

    def test_a_missing_required_field_is_refused_by_name(self, tmp_path):
        mod = _mod()
        row = _prod_row()
        del row['traffic_share']
        path = _prod_rows(tmp_path, [row])
        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)
        assert 'traffic_share' in str(exc.value)


# ---------------------------------------------------------------------------
# A malformed share is a NAMED refusal, not a raw traceback
# ---------------------------------------------------------------------------
#
# The loader's field check is key-PRESENCE only, so `"traffic_share": null`
# passes it and flows straight through to
# `production_query_provenance`'s `sum(entry['traffic_share'] ...)`, which
# raises `TypeError` — a type NOT in `main`'s handler tuple, so the observed
# result is a traceback rather than the named `return 1`.
#
# This is reachable, not hypothetical: `harvest_production_queries._share`
# returns None whenever `total <= 0` and feeds it to the briefing-template
# rows, so a harvest over a journal with zero parseable search ops emits
# exactly this fixture.  The committed sample happens to be clean (44 rows,
# 0 null shares), which is why the guard has to be provoked here.
#
# The refusal is scoped to the rows the claim rests on.  A `null` share is a
# legal MEASUREMENT for a tail row — it is never summed — but on a
# `briefing_template` row it is the artifact's headline "the four templates
# are N% of all search traffic" being unbuildable, and the fixture is
# malformed.

class TestAMalformedShareIsRefusedByName:
    """Refuse at the loader, so `main`'s existing handler can name it."""

    def test_a_null_share_on_a_template_row_is_refused(self, tmp_path):
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(traffic_share=None)])

        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)

        message = str(exc.value)
        assert 'prod-brief-abc' in message
        assert 'traffic_share' in message

    def test_a_string_share_is_refused_by_type_not_by_presence(self, tmp_path):
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(traffic_share='0.17')])

        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)

        assert 'traffic_share' in str(exc.value)

    def test_a_bool_share_is_refused(self, tmp_path):
        """`isinstance(True, int)` is True — a bare numeric check lets it in."""
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(traffic_share=True)])

        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)

        assert 'traffic_share' in str(exc.value)

    @pytest.mark.parametrize('bad', ['74103', None, -1, 3.5, True])
    def test_a_non_integer_observed_count_is_refused(self, tmp_path, bad):
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(observed_count=bad)])

        with pytest.raises(mod.ProductionQuerySetError) as exc:
            mod.load_production_queries(path)

        assert 'observed_count' in str(exc.value)
        assert 'prod-brief-abc' in str(exc.value)

    def test_a_tail_rows_null_share_is_a_measurement_not_a_defect(self, tmp_path):
        """The harvester emits it legitimately, and nothing sums it.

        Refusing it here would make a real harvest unloadable over a journal
        the loader has no business judging.
        """
        mod = _mod()
        path = _prod_rows(tmp_path, [
            _prod_row(),
            _prod_row(query_id='prod-tail-1', text='an unrepeated question',
                      source='production_tail', traffic_share=None,
                      observed_count=1),
        ])

        queries = mod.load_production_queries(path)

        assert [q.traffic_share for q in queries] == [0.172892, None]

    def test_main_returns_1_with_the_named_error_and_no_traceback(
        self, tmp_path, capsys,
    ):
        """The exit path, not just the exception type.

        `ProductionQuerySetError` is already in `main`'s handler tuple; the
        point of refusing at the loader is that the tuple does not have to
        be widened to swallow bare `TypeError`, which would turn unrelated
        programming errors into a quiet exit 1.
        """
        mod = _mod()
        path = _prod_rows(tmp_path, [_prod_row(traffic_share=None)])

        code = mod.main([
            '--production-queries', str(path),
            '--json-out', str(tmp_path / 'out.json'),
            '--md-out', str(tmp_path / 'out.md'),
        ])

        assert code == 1
        stderr = capsys.readouterr().err
        assert stderr.startswith('ProductionQuerySetError: ')
        assert 'Traceback' not in stderr


class TestTheProvenanceBlockToleratesAnAbsentShare:
    """The two halves of the module must agree that `None` is legal.

    The renderer already handles `share is None`; the provenance builder
    crashed on it.  A pure function over a hand-built or older row list
    must not be the thing that decides — the loader is.
    """

    def _query(self, mod, **kwargs):
        fields = {
            'query_id': 'prod-brief-abc',
            'text': 'project overview architecture goals',
            'source': 'briefing_template',
            'observed_count': 74103,
            'traffic_share': 0.172892,
            'observed_limit': 5,
            'template': 'project overview architecture goals',
            'match': 'literal',
        }
        fields.update(kwargs)
        return mod.ProductionQuery(**fields)

    def test_a_none_share_never_raises(self, ):
        mod = _mod()

        block = mod.production_query_provenance([
            self._query(mod, traffic_share=None),
        ])

        assert block['family_share'] is None

    def test_the_sum_skips_the_none_and_keeps_the_measured_ones(self):
        mod = _mod()

        block = mod.production_query_provenance([
            self._query(mod, traffic_share=0.25),
            self._query(mod, query_id='b', template='b', traffic_share=None),
            self._query(mod, query_id='c', template='c', traffic_share=0.1),
        ])

        assert block['family_share'] == 0.35

    def test_no_template_carries_a_share_reports_no_measurement(self):
        """`None`, never a measured 0.0 — the discipline the artifact states."""
        mod = _mod()

        block = mod.production_query_provenance([
            self._query(mod, traffic_share=None),
            self._query(mod, query_id='b', template='b', traffic_share=None),
        ])

        assert block['family_share'] is None

    def test_the_renderer_prints_a_dash_for_an_absent_observed_count(self):
        """`f'{"—":,}'` raises `ValueError: Cannot specify ',' with 's'`.

        The key is always written today, so this guards a hand-edited or
        older artifact against crashing the renderer instead of printing a
        dash.
        """
        mod = _mod()
        report = mod.build_selection_report(_scored())
        report['production_queries'] = {
            'templates': [
                {'template': 'a template', 'match': 'literal',
                 'traffic_share': None},
            ],
            'family_share': None,
            'sampled_query_count': 1,
            'tail_query_count': 0,
            'observed_limit': 5,
            'unlabeled': True,
            'unlabeled_reason': mod._UNLABELED_REASON,
        }

        markdown = mod.render_selection_markdown(report)

        row = next(
            line for line in markdown.splitlines() if '`a template`' in line
        )
        assert row.count(_bake_off()._NO_MEASUREMENT) == 2


class TestProductionIsScoredAtProductionsOwnK:
    """briefing.py fires at limit=5 (:1376), not the E2 default of 10."""

    def test_the_production_k_is_five(self):
        assert _mod().PRODUCTION_K == 5

    def test_scoring_defaults_to_that_k(self):
        mod = _mod()
        hits = [_rec(f'r{i}', topic=f't{i}') for i in range(8)]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['k'] == 5
        assert scored['window_size'] == 5

    def test_a_shorter_window_reports_its_real_size(self):
        mod = _mod()
        hits = [_rec('r0'), _rec('r1', topic='t2')]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['window_size'] == 2


class TestLabeledMetricsAreNoneNotZero:
    """The whole point: an unlabeled query yields no labeled number."""

    def test_claim_recall_is_none(self):
        mod = _mod()
        hits = [_rec('r0', claim_ids=['c1']), _rec('r1', claim_ids=['c2'])]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['claim_recall'] is None
        assert scored['claim_recall'] != 0.0

    def test_canonical_discoverability_is_none_on_both_rates(self):
        """Even the aliased rate is unavailable: there is no canonical to seek."""
        mod = _mod()
        hits = [_rec('r0', canonical=True), _rec('r1')]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['canonical_aliased_in_top_k'] is None
        assert scored['canonical_unaliased_in_top_k'] is None

    def test_the_reason_travels_with_the_none(self):
        """A `—` in the table must be explainable without reading this file."""
        mod = _mod()
        hits = [_rec('r0')]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['unlabeled'] is True
        assert 'ground truth' in scored['unlabeled_reason'].lower()

    def test_a_canonical_in_the_window_still_scores_none(self):
        """Presence is not ground truth.

        A production query CAN return a canonical record — that says nothing
        about whether it was the RIGHT answer, which is exactly what a
        discoverability rate claims to measure.
        """
        mod = _mod()
        hits = [_rec('r0', canonical=True, topic='t1')]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['canonical_aliased_in_top_k'] is None


class TestComputableMetricsAreProduced:
    """What CAN be measured without labels, is."""

    def test_tokens_per_query_is_measured(self):
        mod = _mod()
        hits = [_rec('r0', content='x' * 100), _rec('r1', content='y' * 100)]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert isinstance(scored['tokens_per_query'], int)
        assert scored['tokens_per_query'] > 0

    def test_the_token_estimator_is_named(self):
        """Inherited from the E2 table: a token count without its estimator
        is not comparable across tables."""
        mod = _mod()
        scored = mod.score_unlabeled_query([_rec('r0')], baseline=[_rec('r0')])
        assert scored['token_estimator']

    def test_topic_diversity_counts_distinct_topics_in_the_window(self):
        mod = _mod()
        hits = [
            _rec('r0', topic='alpha'), _rec('r1', topic='alpha'),
            _rec('r2', topic='beta'), _rec('r3', topic='gamma'),
            _rec('r4', topic='beta'),
        ]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['topic_diversity'] == 3

    def test_untopiced_records_do_not_inflate_diversity(self):
        """The 300-record distractor slab carries only `category`."""
        mod = _mod()
        hits = [_rec('r0', topic='alpha'), _rec('d0', topic=None), _rec('d1', topic=None)]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['topic_diversity'] == 1
        assert scored['untopiced_in_window'] == 2

    def test_baseline_retention_is_one_when_the_window_is_unchanged(self):
        mod = _mod()
        hits = [_rec(f'r{i}', topic=f't{i}') for i in range(5)]
        scored = mod.score_unlabeled_query(hits, baseline=hits)
        assert scored['baseline_retention_at_k'] == 1.0
        assert scored['window_displacement'] == 0

    def test_baseline_retention_falls_when_a_record_is_dropped(self):
        mod = _mod()
        baseline = [_rec(f'r{i}', topic=f't{i}') for i in range(5)]
        transformed = baseline[:4] # the cap dropped one
        scored = mod.score_unlabeled_query(transformed, baseline=baseline)
        assert scored['baseline_retention_at_k'] == 0.8
        assert scored['window_displacement'] == 1

    def test_a_record_pushed_past_the_window_edge_counts_as_displaced(self):
        """Arm (1)'s honest cost: promotion at a full window evicts the k-th."""
        mod = _mod()
        baseline = [_rec(f'r{i}', topic=f't{i}') for i in range(5)]
        promoted = _rec('rX', topic='tX')
        transformed = [promoted, *baseline[:4]] # r4 truncated away at k=5
        scored = mod.score_unlabeled_query(transformed, baseline=baseline)
        assert scored['window_displacement'] == 1
        assert scored['baseline_retention_at_k'] == 0.8

    def test_a_pure_reorder_inside_the_window_displaces_nothing(self):
        """Displacement is eviction, not motion — the two are different costs."""
        mod = _mod()
        baseline = [_rec(f'r{i}', topic=f't{i}') for i in range(5)]
        scored = mod.score_unlabeled_query(list(reversed(baseline)), baseline=baseline)
        assert scored['window_displacement'] == 0
        assert scored['baseline_retention_at_k'] == 1.0
        assert scored['window_reordered'] is True


class TestRetentionAndDisplacementDivergeUnderGrouping:
    """Arm (2)'s exact story: knowledge retained, records displaced.

    A grouped document absorbs its members' bodies, so the KNOWLEDGE of a
    folded member is still in the reader's window — retention holds. But the
    member is no longer an independent hit, which is the suppression cost —
    displacement records it. Collapsing the two into one number would hide
    whichever half a reader cared about.
    """

    def test_a_folded_member_is_retained_but_displaced(self):
        mod = _mod()
        canonical = _rec('c1', topic='alpha', canonical=True)
        peer = _rec('p1', topic='alpha')
        baseline = [canonical, peer, _rec('r2', topic='beta')]
        result = mod.apply_topic_keyed_grouped_read(
            baseline, {'alpha': [canonical, peer]}, render_sightings=False,
        )
        scored = mod.score_unlabeled_query(
            result.records, baseline=baseline, provenance=result.provenance,
        )
        assert scored['baseline_retention_at_k'] == 1.0, 'p1 folded, not lost'
        assert scored['window_displacement'] == 1, 'p1 is no longer its own hit'

    def test_without_provenance_a_folded_member_reads_as_lost(self):
        """The disclosure is load-bearing, not decoration."""
        mod = _mod()
        canonical = _rec('c1', topic='alpha', canonical=True)
        peer = _rec('p1', topic='alpha')
        baseline = [canonical, peer]
        result = mod.apply_topic_keyed_grouped_read(
            baseline, {'alpha': [canonical, peer]}, render_sightings=False,
        )
        scored = mod.score_unlabeled_query(result.records, baseline=baseline)
        assert scored['baseline_retention_at_k'] == 0.5


class TestNoneNeverAveragesInAsZero:
    """`_mean` (:3286) is reused, not restated — and its discipline holds here."""

    def test_an_all_none_column_aggregates_to_none(self):
        mod = _mod()
        rows = [
            {'claim_recall': None, 'tokens_per_query': 10},
            {'claim_recall': None, 'tokens_per_query': 20},
        ]
        agg = mod.aggregate_unlabeled(rows)
        assert agg['claim_recall'] is None
        assert agg['tokens_per_query'] == 15.0

    def test_a_none_does_not_drag_a_measured_column_down(self):
        mod = _mod()
        rows = [
            {'baseline_retention_at_k': 1.0},
            {'baseline_retention_at_k': None},
            {'baseline_retention_at_k': 0.5},
        ]
        agg = mod.aggregate_unlabeled(rows)
        assert agg['baseline_retention_at_k'] == 0.75 # not 0.5

    def test_it_delegates_to_the_bake_offs_mean(self):
        """INV-5: there is not a second mean in this repo."""
        mod = _mod()
        assert mod._mean is _bake_off()._mean

    def test_an_empty_row_set_aggregates_to_none_not_zero(self):
        mod = _mod()
        agg = mod.aggregate_unlabeled([])
        assert agg['tokens_per_query'] is None


class TestUnlabeledScoringIsPure:
    def test_it_mutates_neither_hits_nor_baseline(self):
        mod = _mod()
        hits = [_rec('r0', topic='a'), _rec('r1', topic='b')]
        baseline = [_rec('r0', topic='a'), _rec('r1', topic='b'), _rec('r2', topic='c')]
        before_hits = copy.deepcopy(hits)
        before_baseline = copy.deepcopy(baseline)
        mod.score_unlabeled_query(hits, baseline=baseline)
        assert hits == before_hits
        assert baseline == before_baseline


# ---------------------------------------------------------------------------
# The decision-table artifact contract
# ---------------------------------------------------------------------------
#
# This artifact exists to be READ by whoever lands task 3111, so its contract
# is about disclosure, not only about numbers.  Four things a reader cannot
# reconstruct from a bare table have to be stated per arm:
#
#   (a) the record-id ALIASING — a grouped document is emitted under its
#       canonical's record_id, so a member folding upward scores as "canonical
#       found".  Both rates are printed side by side.
#   (b) the SIGHTING-CREDITING knob — a fixed policy at :927 in the landed
#       code, a dial here, and the claim-recall ceiling moves with it.
#   (c) whether the arm SUPPRESSES, split from whether it needs `contested`,
#       because those are different blockers.
#   (d) that PRD V2's esc-5712 protection is UNIMPLEMENTABLE today, naming
#       the live vocabulary set that does not contain `contested`.
#
# And a refusal: a partial table must never render as a complete one.


REQUIRED_SELECTION_METRICS = (
    'claim_recall',
    'tokens_per_query',
    'topic_diversity',
    'baseline_retention_at_k',
    'window_displacement',
    'canonical_aliased_in_top_k',
    'canonical_unaliased_in_top_k',
)


def _arm_block(arm, *, labeled=True, k=10, drop=None, means=None):
    """A minimal well-formed scored-arm block, with one metric optionally dropped."""
    base = {
        'claim_recall': 0.5,
        'tokens_per_query': 100.0,
        'topic_diversity': 3.0,
        'baseline_retention_at_k': 1.0,
        'window_displacement': 0.0,
        'canonical_aliased_in_top_k': 0.9,
        'canonical_unaliased_in_top_k': 0.4,
    }
    if means:
        base.update(means)
    if drop:
        base.pop(drop)
    return {
        'arm': arm,
        'label': _mod().ARM_LABELS[arm],
        'k': k,
        'labeled': labeled,
        'query_count': 4,
        'rows': [],
        'means': base,
    }


def _scored(drop=None, drop_arm=None, means_by_arm=None, omit_arm=None):
    mod = _mod()
    means_by_arm = means_by_arm or {}
    e2, production = {}, {}
    for arm in mod.ARM_KEYS:
        if arm == omit_arm:
            continue
        e2[arm] = _arm_block(
            arm, labeled=True, k=10,
            drop=drop if arm == (drop_arm or arm) else None,
            means=means_by_arm.get(arm),
        )
        prod_means = {
            'claim_recall': None,
            'canonical_aliased_in_top_k': None,
            'canonical_unaliased_in_top_k': None,
        }
        production[arm] = _arm_block(
            arm, labeled=False, k=5, means=prod_means,
        )
    return {
        'shape': 'c_peers',
        'sighting_crediting': 'uncredited',
        'per_topic_cap': 1,
        'token_estimator': 'char-proxy',
        'e2': e2,
        'production': production,
    }


class TestThePartialTableRefusal:
    """Mirrors `_check_arms` / IncompleteReportError (:1887-1920)."""

    def test_a_complete_scoring_builds(self):
        report = _mod().build_selection_report(_scored())
        assert report['arms']

    def test_a_missing_arm_is_refused(self):
        mod = _mod()
        with pytest.raises(mod.IncompleteSelectionError) as exc:
            mod.build_selection_report(_scored(omit_arm='topic_diversity_cap'))
        assert 'topic_diversity_cap' in str(exc.value)

    def test_an_unknown_arm_is_refused_alongside_the_missing_one(self):
        """The unknown name is the actionable half — a typo, not a crash."""
        mod = _mod()
        scored = _scored(omit_arm='topic_diversity_cap')
        scored['e2']['topic_diversity_kap'] = _arm_block('flat_read')
        with pytest.raises(mod.IncompleteSelectionError) as exc:
            mod.build_selection_report(scored)
        message = str(exc.value)
        assert 'topic_diversity_kap' in message
        assert 'topic_diversity_cap' in message

    @pytest.mark.parametrize('metric', REQUIRED_SELECTION_METRICS)
    def test_a_missing_metric_is_refused_by_name(self, metric):
        mod = _mod()
        with pytest.raises(mod.IncompleteSelectionError) as exc:
            mod.build_selection_report(_scored(drop=metric, drop_arm='promoting_pin'))
        assert metric in str(exc.value)
        assert 'promoting_pin' in str(exc.value)

    def test_a_none_metric_is_NOT_a_missing_one(self):
        """A measured no-measurement is legitimate; an ABSENT key is not.

        The production half is built entirely of Nones by construction — a
        refusal that could not tell the two apart would make the honest
        half of this artifact unrenderable.
        """
        mod = _mod()
        report = mod.build_selection_report(
            _scored(means_by_arm={'flat_read': {'claim_recall': None}}),
        )
        assert report['arms']['flat_read']['e2']['claim_recall'] is None


class TestTheAliasingDisclosure:
    """(a) — the unaliased rate is printed BESIDE the aliased one, per arm."""

    def test_both_rates_appear_for_every_arm(self):
        mod = _mod()
        report = mod.build_selection_report(_scored())
        for arm in mod.ARM_KEYS:
            block = report['arms'][arm]['e2']
            assert 'canonical_aliased_in_top_k' in block
            assert 'canonical_unaliased_in_top_k' in block

class TestTheSightingCreditingKnob:
    """(b) — never left implicit."""

    def test_the_setting_is_stated_in_the_report(self):
        mod = _mod()
        report = mod.build_selection_report(_scored())
        assert report['sighting_crediting'] in ('uncredited', 'rendered')

class TestTheSuppressionColumn:
    """(c) — suppressing, and needing `contested`, are DIFFERENT facts."""

    def test_arm_two_is_marked_suppressing(self):
        mod = _mod()
        report = mod.build_selection_report(_scored())
        spec = report['arms']['topic_keyed_grouped']['spec']
        assert spec['drops_ranked_records'] is True
        assert spec['requires_contested_key_for_v2'] is True

    def test_arm_one_needs_no_contested_key(self):
        mod = _mod()
        spec = mod.build_selection_report(_scored())['arms']['promoting_pin']['spec']
        assert spec['requires_contested_key_for_v2'] is False
        assert spec['drops_ranked_records'] is False
        assert spec['displaces_at_window_edge'] is True

    def test_arm_three_suppresses_yet_needs_no_contested_key(self):
        """The distinction that decides what 3111 can land TODAY."""
        mod = _mod()
        spec = mod.build_selection_report(_scored())['arms']['topic_diversity_cap']['spec']
        assert spec['drops_ranked_records'] is True
        assert spec['requires_contested_key_for_v2'] is False

    def test_the_two_flags_are_not_collapsed_into_one_column(self):
        """Collapsing them would tell 3111 the cap is blocked on a key that
        does not exist — the opposite of the truth."""
        mod = _mod()
        report = mod.build_selection_report(_scored())
        cap = report['arms']['topic_diversity_cap']['spec']
        grouped = report['arms']['topic_keyed_grouped']['spec']
        assert cap['drops_ranked_records'] == grouped['drops_ranked_records']
        assert cap['requires_contested_key_for_v2'] != grouped['requires_contested_key_for_v2']


class TestTheV2Impossibility:
    """(d) — the hard constraint, stated verbatim and evidenced."""

    def test_the_named_set_matches_the_live_one(self):
        """A hard-coded list here would rot silently the day a key is added."""
        from fused_memory.memory_metadata import (  # noqa: PLC0415
            RESERVED_VOCABULARY_KEYS,
        )

        mod = _mod()
        assert set(mod.RESERVED_VOCABULARY_KEYS) == set(RESERVED_VOCABULARY_KEYS)
        assert 'contested' not in RESERVED_VOCABULARY_KEYS

class TestTheRecommendation:
    """The artifact is FOR task 3111. It must answer 3111's question."""

    def test_it_names_exactly_one_winner_from_the_arm_set(self):
        mod = _mod()
        report = mod.build_selection_report(_scored())
        assert report['recommendation']['arm'] in mod.ARM_KEYS

    def test_the_recommendation_carries_its_rationale(self):
        mod = _mod()
        report = mod.build_selection_report(_scored())
        assert len(report['recommendation']['rationale']) > 40

    def test_every_arm_has_exactly_one_row_in_the_markdown(self):
        mod = _mod()
        markdown = mod.render_selection_markdown(mod.build_selection_report(_scored()))
        for arm in mod.ARM_KEYS:
            label = mod.ARM_LABELS[arm]
            rows = [ln for ln in markdown.splitlines()
                    if ln.startswith('| ') and label in ln]
            assert len(rows) == 1, f'{arm}: {len(rows)} rows'


def _ranks(unaliased_by_arm: dict[str, float], aliased: float = 1.1) -> dict:
    """A scoring whose canonical RANK columns are set per arm."""
    return _scored(means_by_arm={
        arm: {'canonical_unaliased_rank': value,
              'canonical_aliased_rank': aliased}
        for arm, value in unaliased_by_arm.items()
    })


class TestTheRetrievalFindingTracksTheTable:
    """The one sentence 3111 needs, DERIVED rather than asserted.

    "No transform improves the canonical's retrieval rank" is true only while
    the unaliased column really is arm-invariant.  A prose constant saying so
    would survive the column going wrong; this paragraph is read off the rows,
    so it changes when they do.
    """

    def test_it_reports_the_shared_rank_when_the_column_is_arm_invariant(self):
        mod = _mod()
        report = mod.build_selection_report(
            _ranks(dict.fromkeys(mod.ARM_KEYS, 3.49)),
        )

        finding = mod._retrieval_finding(report['arms'])

        assert 'No arm improves' in finding
        assert _bake_off()._cell(3.49) in finding

    def test_it_refuses_the_invariance_claim_when_the_ranks_disagree(self):
        """A spread means the column is reading the transform again — the
        exact conflation this table was corrected to remove.  The paragraph
        has to say so instead of restating an invariance that has stopped
        holding."""
        mod = _mod()
        ranks = dict.fromkeys(mod.ARM_KEYS, 3.49)
        ranks['promoting_pin'] = 1.06
        report = mod.build_selection_report(_ranks(ranks))

        finding = mod._retrieval_finding(report['arms'])

        assert 'No arm improves' not in finding
        assert 'NOT identical' in finding
        assert _bake_off()._cell(1.06) in finding

    def test_an_unmeasured_rank_is_not_read_as_agreement(self):
        """A missing column is no measurement, never a silent match."""
        mod = _mod()
        report = mod.build_selection_report(_scored())  # no rank keys at all

        assert 'No arm improves' not in mod._retrieval_finding(report['arms'])

    def test_the_committed_report_carries_the_measured_finding(self):
        """Over the artifact itself: the shared rank in the JSON is the one
        the Recommendation section prints."""
        report = _committed_selection_json()
        shared = {
            report['arms'][arm]['e2']['canonical_unaliased_rank']
            for arm in _mod().ARM_KEYS
        }
        assert len(shared) == 1, shared

        section = _committed_selection_markdown().split('## Recommendation')[1]
        assert _bake_off()._cell(shared.pop()) in section


class TestTheNoMeasurementConvention:
    """A None must never print as 0.00 — reuses `_cell` / `_NO_MEASUREMENT`."""

    def test_a_none_renders_as_the_em_dash(self):
        mod = _mod()
        markdown = mod.render_selection_markdown(mod.build_selection_report(_scored()))
        assert _bake_off()._NO_MEASUREMENT in markdown

    def test_the_production_labeled_columns_render_as_no_measurement(self):
        mod = _mod()
        markdown = mod.render_selection_markdown(mod.build_selection_report(_scored()))
        section = markdown.split('Production')[-1]
        assert '0.00' not in section.split('## ')[0]

    def test_it_reuses_the_bake_offs_cell_renderer(self):
        """INV-5: there is not a second `—` convention in this repo."""
        mod = _mod()
        assert mod._cell is _bake_off()._cell


class TestInv5IsLoadOrderIndependent:
    """The committed INV-5 checks must hold no matter which sibling loads
    `bake_off_storage_shape` last (task 4583).

    `_bake_off()` (:74-76) is `functools.cache`d, so once it has been called
    the returned module instance never changes — but a sibling test module's
    own loader (or a second direct call to this file's `_load_script`, as
    below) can still register a *different*, independently `exec_module`d
    instance under `sys.modules['bake_off_storage_shape']` at any time.
    `read_transform_selection.bake_off()` has no such cache: it reads
    `sys.modules` live on every call.  So `mod._cell is _bake_off()._cell`
    compares "whatever is in `sys.modules` right now" against "whatever
    `_bake_off()` cached the first time it ran" — two objects that are only
    guaranteed equal when nothing has clobbered `sys.modules` in between,
    which is exactly the guarantee xdist scheduling does not provide.  This
    class reproduces that clobber deterministically, in-process, instead of
    depending on which worker happens to run a given test first.
    """

    def _clobber_bake_off_module(self):
        """Register a fresh, independently-exec'd `bake_off_storage_shape`.

        Returns whatever was previously registered (or ``None``) so the
        caller can restore it afterwards.  Uses this file's own
        `_load_script`, which — unlike the production script's loader —
        unconditionally execs a new module and overwrites `sys.modules`,
        exactly what a sibling test module's load does.
        """
        import sys  # noqa: PLC0415

        saved = sys.modules.get('bake_off_storage_shape')
        _load_script(BAKE_OFF_PATH, 'bake_off_storage_shape')
        return saved

    def _restore_bake_off_module(self, saved):
        import sys  # noqa: PLC0415

        if saved is None:
            sys.modules.pop('bake_off_storage_shape', None)
        else:
            sys.modules['bake_off_storage_shape'] = saved

    def test_the_committed_inv5_checks_survive_a_reloaded_bake_off(self):
        """Both hand-picked INV-5 identity checks must survive a clobber.

        Re-clobber immediately before EACH check: a single clobber only
        reddens the first check that runs, because that check's own
        `_bake_off()` call (when its cache is cold) loads and registers yet
        another instance, which re-syncs `sys.modules` with the cache — so a
        second check run right after would pass regardless of whether it is
        itself order-independent.
        """
        saved = self._clobber_bake_off_module()
        try:
            TestTheNoMeasurementConvention().test_it_reuses_the_bake_offs_cell_renderer()
        finally:
            self._restore_bake_off_module(saved)

        saved = self._clobber_bake_off_module()
        try:
            TestNoneNeverAveragesInAsZero().test_it_delegates_to_the_bake_offs_mean()
        finally:
            self._restore_bake_off_module(saved)


# ===========================================================================
# step-21 — the COMMITTED artifact pair, asserted as DATA
# ===========================================================================
#
# `plans/read-transform-selection-report.{json,md}` is this task's
# user-observable signal — the document whoever lands task 3111 actually
# reads — so it is guarded as data rather than described in prose.  Mirrors
# `TestCommittedReportJson`/`TestCommittedReportMarkdown`
# (test_bake_off_storage_shape.py:5736/5952): pure file reads, no network, no
# Qdrant, no key, running in the merge lane on every commit.
#
# What these pin is COMPLETENESS and HONESTY, never a value (PRD gate
# G6/D10 — no arm, knob or window is ever re-tuned to move a column):
#
#   * every cell is a real measurement or `—`, and the two are consistent
#     with the JSON cell-for-cell, so a `None` can never have printed as
#     `0.00` and a measured number can never have been blanked;
#   * the production half's labeled columns are `None` — the honest
#     no-measurement — WITH the reason printed, never a measured zero;
#   * the markdown re-renders byte-identically from the committed JSON, so
#     the two artifacts cannot drift apart unnoticed;
#   * exactly one markdown row per arm named in the JSON;
#   * the four briefing-template traffic shares carried by the report are the
#     ones in the committed `production_query_sample.jsonl`, so the external
#     validity claim is checkable rather than asserted.

import json  # noqa: E402

HARVEST_PATH = SCRIPTS_DIR / 'harvest_production_queries.py'
PRODUCTION_SAMPLE = FIXTURES_DIR / 'production_query_sample.jsonl'

#: `column header -> (half, metric)` for the measurement columns of the
#: committed decision table.  Pinned HERE, as the contract the artifact
#: publishes, so the None-vs-`—` cross-check below is exact rather than
#: positional guesswork.  `None` marks a column that is not a single scalar
#: measurement (the arm label, the two spec flags, and the canonical column,
#: which deliberately prints its two rates in ONE cell so neither can be
#: quoted without its semantics).
SELECTION_TABLE_COLUMNS: tuple[tuple[str, tuple[str, str] | None], ...] = (
    ('arm', None),
    ('claim recall', ('e2', 'claim_recall')),
    ('canonical in top-k', None),
    ('tokens/query', ('e2', 'tokens_per_query')),
    ('topic diversity', ('e2', 'topic_diversity')),
    ('baseline retention', ('e2', 'baseline_retention_at_k')),
    ('displacement', ('e2', 'window_displacement')),
    ('prod tokens/query', ('production', 'tokens_per_query')),
    ('prod displacement', ('production', 'window_displacement')),
    ('drops ranked records', None),
    ('needs `contested` for V2', None),
)


@functools.cache
def _harvest_mod() -> types.ModuleType:
    return _load_script(HARVEST_PATH, 'harvest_production_queries')


@functools.cache
def _committed_selection_json() -> dict:
    path = _mod().DEFAULT_SELECTION_JSON
    assert path.exists(), (
        f'{path} is missing. It is the decision document task 3111 reads; '
        f'regenerate it with `uv run python '
        f'fused-memory/scripts/read_transform_selection.py`.'
    )
    return json.loads(path.read_text(encoding='utf-8'))


@functools.cache
def _committed_selection_markdown() -> str:
    path = _mod().DEFAULT_SELECTION_MD
    assert path.exists(), (
        f'{path} is missing. It is the operator-facing half of the pair; '
        f'regenerate it with `uv run python '
        f'fused-memory/scripts/read_transform_selection.py`.'
    )
    return path.read_text(encoding='utf-8')


def _selection_table_rows(markdown: str) -> list[str]:
    """The decision-table body rows, located by the header the renderer emits."""
    lines = markdown.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith('| arm |')
    )
    rows = []
    for line in lines[start + 2:]:            # +2 skips the `| --- |` rule
        if not line.startswith('| '):
            break
        rows.append(line)
    return rows


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip('|').split('|')]


class TestTheCommittedPairExists:
    """The module names its own artifacts, and the pair is in the tree."""

    def test_the_module_names_the_committed_artifact_pair(self):
        mod = _mod()

        assert mod.DEFAULT_SELECTION_JSON.name == 'read-transform-selection-report.json'
        assert mod.DEFAULT_SELECTION_MD.name == 'read-transform-selection-report.md'
        assert mod.DEFAULT_SELECTION_JSON.parent.name == 'plans'

    def test_it_is_a_SIBLING_and_never_the_e2_artifact(self):
        """In-progress task 3560 owns `plans/e2-storage-shape-bakeoff-report.*`
        and is landing the aliasing disclosure into it.  Writing read-side
        arms there would break its `_check_arms` guard and collide head-on."""
        mod = _mod()
        bake = _bake_off()

        assert mod.DEFAULT_SELECTION_JSON != bake.DEFAULT_REPORT_JSON
        assert mod.DEFAULT_SELECTION_MD != bake.DEFAULT_REPORT_MD


class TestCommittedSelectionJson:
    """The machine-readable half."""

    def test_it_parses_and_declares_its_schema(self):
        report = _committed_selection_json()

        assert report['schema'] == _mod().SELECTION_REPORT_SCHEMA

    def test_every_arm_has_a_row_in_both_halves(self):
        """As a SET: the JSON is written `sort_keys=True` so the committed
        file diffs cleanly, which makes its key order alphabetical by
        construction and not a statement about arm ordering.  Order IS
        asserted — on the markdown, where it is what a reader sees."""
        mod = _mod()
        report = _committed_selection_json()

        assert set(report['arms']) == set(mod.ARM_KEYS)
        for arm in mod.ARM_KEYS:
            assert set(report['arms'][arm]) >= {'label', 'spec', 'e2', 'production'}

    @pytest.mark.parametrize('metric', REQUIRED_SELECTION_METRICS)
    @pytest.mark.parametrize('half', ['e2', 'production'])
    def test_every_required_metric_key_is_present(self, half, metric):
        """An ABSENT key is a broken run; a None VALUE is a measurement that
        was honestly not made.  Only the first is a defect."""
        report = _committed_selection_json()

        for arm in _mod().ARM_KEYS:
            assert metric in report['arms'][arm][half], arm

    def test_the_e2_half_actually_measured_its_labeled_columns(self):
        """The authored set IS labeled, so a `None` there is a half-failed run
        — the "blank cells read as a tie" failure, in the half that has ground
        truth."""
        report = _committed_selection_json()

        for arm in _mod().ARM_KEYS:
            block = report['arms'][arm]['e2']
            for metric in REQUIRED_SELECTION_METRICS:
                assert block[metric] is not None, f'{arm}.{metric}'

    @pytest.mark.parametrize('metric', [
        'claim_recall', 'canonical_aliased_in_top_k',
        'canonical_unaliased_in_top_k',
    ])
    def test_the_production_labeled_columns_are_None_and_NOT_zero(self, metric):
        """The honesty guard, at the source.

        Production queries carry no ground truth, so these are not computable.
        A `0.0` here would say "measured, and the arm found nothing" — the
        exact fabrication the `—` convention exists to prevent.
        """
        report = _committed_selection_json()

        for arm in _mod().ARM_KEYS:
            assert report['arms'][arm]['production'][metric] is None, arm

    def test_the_production_columns_that_ARE_computable_were_computed(self):
        """Otherwise the unlabeled half is decoration: the whole point is that
        tokens and displacement ARE measurable without labels."""
        report = _committed_selection_json()

        for arm in _mod().ARM_KEYS:
            block = report['arms'][arm]['production']
            assert block['tokens_per_query'] is not None, arm
            assert block['window_displacement'] is not None, arm

    def test_the_two_windows_are_the_ones_the_design_names(self):
        """E2 at its own default, production at the briefing assembler's real
        `limit=5` (briefing.py:1376) — a wider production window would measure
        a read no production caller ever gets."""
        report = _committed_selection_json()

        assert report['e2_k'] == 10
        assert report['production_k'] == _mod().PRODUCTION_K == 5

    def test_it_carries_the_fixture_provenance_stamp(self):
        """The blind-authoring audit trail extends to this table too."""
        report = _committed_selection_json()

        assert report['fixture_provenance']

    def test_the_recommendation_names_an_arm_this_run_actually_scored(self):
        mod = _mod()
        report = _committed_selection_json()
        rec = report['recommendation']

        assert rec['arm'] in report['arms']
        assert rec['arm'] in mod.ARM_KEYS
        assert rec['arm'] not in rec['excluded_pending_contested_key']


class TestCommittedSelectionMarkdown:
    """The operator-facing half, and its agreement with the JSON."""

    def test_the_table_header_is_the_pinned_column_set(self):
        markdown = _committed_selection_markdown()

        header = next(
            line for line in markdown.splitlines() if line.startswith('| arm |')
        )

        assert _cells(header) == [name for name, _ in SELECTION_TABLE_COLUMNS]

    def test_every_arm_in_the_json_has_exactly_one_row(self):
        mod = _mod()
        rows = _selection_table_rows(_committed_selection_markdown())
        arms = list(_committed_selection_json()['arms'])

        assert len(rows) == len(arms)
        for arm in arms:
            label = mod.ARM_LABELS[arm]
            assert sum(1 for row in rows if _cells(row)[0] == label) == 1, arm

    def test_the_rows_are_in_arm_order_with_the_baseline_first(self):
        """Every column is read as a delta against the flat read, so the
        baseline is the first row a reader meets — not wherever an alphabetical
        sort happened to put it."""
        mod = _mod()
        rows = _selection_table_rows(_committed_selection_markdown())

        assert [_cells(row)[0] for row in rows] == [
            mod.ARM_LABELS[arm] for arm in mod.ARM_KEYS
        ]
        assert _cells(rows[0])[0] == mod.ARM_LABELS['flat_read']

    def test_every_cell_is_a_real_measurement_or_the_em_dash(self):
        """No blank, no `None`, no `nan` — the three ways a hole gets rendered
        as though it were content."""
        dash = _bake_off()._NO_MEASUREMENT

        for row in _selection_table_rows(_committed_selection_markdown()):
            for cell in _cells(row):
                assert cell, row
                assert 'None' not in cell, row
                assert 'nan' not in cell.lower(), row
                assert cell == dash or cell.strip(dash), row

    def test_a_None_in_the_json_renders_as_the_dash_and_nothing_else(self):
        """The cross-check that makes `0.00` impossible.

        Read the other way it is just as load-bearing: a MEASURED value must
        never render as `—`, or a completed run reads as a half-failed one.
        """
        mod = _mod()
        dash = _bake_off()._NO_MEASUREMENT
        report = _committed_selection_json()
        by_label = {
            _cells(row)[0]: _cells(row)
            for row in _selection_table_rows(_committed_selection_markdown())
        }

        for arm in mod.ARM_KEYS:
            cells = by_label[mod.ARM_LABELS[arm]]
            for column, (name, source) in enumerate(SELECTION_TABLE_COLUMNS):
                if source is None:
                    continue
                half, metric = source
                value = report['arms'][arm][half][metric]
                cell = cells[column]
                assert (cell == dash) is (value is None), f'{arm}.{name}={value!r} -> {cell!r}'

    def test_the_canonical_cell_prints_both_rates_with_their_semantics(self):
        """One cell, two words: a rate cannot be copied out of this table with
        the aliasing silently stripped off — which is exactly how b_grouped's
        0.97 came to be read as a retrieval property."""
        mod = _mod()
        column = [name for name, _ in SELECTION_TABLE_COLUMNS].index(
            'canonical in top-k',
        )
        by_label = {
            _cells(row)[0]: _cells(row)
            for row in _selection_table_rows(_committed_selection_markdown())
        }

        for arm in mod.ARM_KEYS:
            cell = by_label[mod.ARM_LABELS[arm]][column]
            assert 'aliased' in cell and 'unaliased' in cell, cell

    def test_it_re_renders_byte_identically_from_the_committed_json(self):
        """The two artifacts are written from ONE report dict; this is what
        makes it impossible for them to disagree unnoticed — and it is what
        proves no number was hand-edited into the markdown after the run."""
        mod = _mod()

        rendered = mod.render_selection_markdown(_committed_selection_json())

        assert rendered == _committed_selection_markdown()

class TestTheProductionTrafficSharesAreTheMeasuredOnes:
    """The external-validity claim, made checkable.

    The report says the production set is what live traffic actually looks
    like.  That claim is only worth anything if the shares in the artifact
    are the ones in the committed sample — otherwise the number is prose.
    """

    def _fixture_templates(self) -> dict[str, dict]:
        rows = _harvest_mod().read_fixture(PRODUCTION_SAMPLE)
        return {
            row['template']: row
            for row in rows if row.get('source') == 'briefing_template'
        }

    def test_the_report_carries_the_production_query_provenance(self):
        report = _committed_selection_json()

        assert report['production_queries']['templates']

    def test_all_four_briefing_templates_are_carried(self):
        harvest = _harvest_mod()
        report = _committed_selection_json()

        carried = {t['template'] for t in report['production_queries']['templates']}
        assert carried == {*harvest.LITERAL_TEMPLATES, harvest.TASK_TEMPLATE}

    def test_every_share_matches_the_committed_sample_exactly(self):
        """Not "close to": the report copies the fixture, so any drift means
        the table was rendered against a sample that is no longer in the
        tree."""
        report = _committed_selection_json()
        fixture = self._fixture_templates()

        for entry in report['production_queries']['templates']:
            row = fixture[entry['template']]
            assert entry['observed_count'] == row['observed_count'], entry
            assert entry['traffic_share'] == row['traffic_share'], entry

    def test_the_family_share_is_the_sum_of_its_members(self):
        """The 64%-of-live-traffic figure, derived rather than asserted."""
        report = _committed_selection_json()
        block = report['production_queries']

        assert block['family_share'] == pytest.approx(
            sum(t['traffic_share'] for t in block['templates']), abs=1e-6,
        )

    def test_the_markdown_names_each_template_verbatim(self):
        """A reader must be able to see WHICH queries this half was scored on
        without opening the fixture."""
        markdown = _committed_selection_markdown()

        for template in self._fixture_templates():
            assert template in markdown, template

    def test_the_production_rows_carry_no_labels(self):
        """Unlabeled BY CONSTRUCTION — the journal records what was asked,
        never what should have come back.  If a label ever appeared here, the
        `—` cells above would be a lie rather than an honest abstention."""
        for row in _harvest_mod().read_fixture(PRODUCTION_SAMPLE):
            assert 'expects_claim_ids' not in row


# ---------------------------------------------------------------------------
# The production half of the fetch cache needs the SAME depth guard as the E2
# half
# ---------------------------------------------------------------------------
#
# `score_from_cache` already passes `expect_limit=e2_k` to the bake-off's
# `load_fetches`, and `load_fetches` refuses a cache shallower than the depth
# the report will claim.  Its production sibling, in the SAME function, passed
# no such guard — so a cache fetched at a shallower depth scored CLEAN and the
# markdown printed `k=5` over a 3-deep window.  Nothing downstream can notice:
# the window is a plain `records[:k]` slice and the recorded `'k'` is the
# REQUESTED k, never the realized depth.
#
# The compounding half is at the writing end: `--extend-cache-live` fetched at
# `args.k` — the E2 window flag — rather than at a depth that respects
# `production_k`, so `--k 3` wrote exactly the shallow cache the reader had no
# guard against.

COMMITTED_FETCH_CACHE = FIXTURES_DIR / 'e2_fetch_cache.json'

#: Depth of the committed production rankings.  Pinned, not read, because
#: (e) below asserts that a stock `--extend-cache-live` still reproduces it:
#: resolving the live fetch depth to `production_k` would silently SHALLOW
#: the next re-harvest from 10 to 5 and change every backfilled
#: `topic_diversity_cap` window.
COMMITTED_PRODUCTION_DEPTH = 10


@functools.cache
def _selection_seeded():
    """The materialized arm `load_production_fetches` joins a cache against.

    Offline and deterministic over the committed fixtures (~0.3s), and cached
    because every test below wants the same one.
    """
    seeded, _ = _mod().materialize_selection_arm()
    return seeded


def _cache_at_production_depth(tmp_path, limit: int | None, *, drop_key=False):
    """The committed cache, rewritten to a shallower production depth.

    Truncates the rankings as well as the recorded limit, so the file is
    exactly what a `--extend-cache-live --k <limit>` run would have written —
    the point being that such a file is INDISTINGUISHABLE from a full one to
    every reader downstream of the load.
    """
    mod = _mod()
    doc = json.loads(COMMITTED_FETCH_CACHE.read_text(encoding='utf-8'))
    block = doc[mod.PRODUCTION_CACHE_KEY][mod.SELECTION_SHAPE]
    if drop_key:
        block.pop('search_limit', None)
    else:
        block['search_limit'] = limit
    if limit is not None:
        block['queries'] = {
            query_id: ranking[:limit]
            for query_id, ranking in block['queries'].items()
        }
    target = tmp_path / 'e2_fetch_cache.json'
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8',
    )
    return target


class TestTheProductionCacheRefusesAShallowerWindow:
    """(a)/(b) — the reader half of the guard, mirroring `load_fetches`."""

    def test_a_shallower_cache_is_refused_by_name(self, tmp_path):
        mod = _mod()
        shallow = _cache_at_production_depth(tmp_path, 3)

        with pytest.raises(mod.ProductionFetchError) as excinfo:
            mod.load_production_fetches(
                shallow, _selection_seeded(), expect_limit=5,
            )

        message = str(excinfo.value)
        assert '3' in message and '5' in message, message
        # The E2 guard's own words: the danger is not a SMALLER measurement.
        assert 'different experiment' in message.lower(), message

    def test_an_absent_search_limit_is_its_own_refusal(self, tmp_path):
        """A cache written before the guard must not pass by omission.

        The same missing-key/mismatch split `_check_corpus_fingerprint` makes:
        "the depth was never recorded" and "the depth is too shallow" are
        different operator actions, so they are different messages.
        """
        mod = _mod()
        undated = _cache_at_production_depth(tmp_path, None, drop_key=True)

        with pytest.raises(mod.ProductionFetchError) as excinfo:
            mod.load_production_fetches(
                undated, _selection_seeded(), expect_limit=5,
            )

        message = str(excinfo.value)
        assert 'different experiment' not in message.lower(), message
        assert 'search limit' in message.lower() or 'depth' in message.lower()

    def test_a_deeper_cache_loads(self, tmp_path):
        """(c) Truncation to the reader budget is the INTENDED behaviour."""
        mod = _mod()

        replayed = mod.load_production_fetches(
            COMMITTED_FETCH_CACHE, _selection_seeded(), expect_limit=5,
        )

        assert len(replayed) == 44
        assert all(len(hits) == COMMITTED_PRODUCTION_DEPTH
                   for hits in replayed.values())

    def test_the_guard_is_opt_in(self, tmp_path):
        """(f-adjacent) Defaulted to None so a pure round-trip test can
        exercise the join without declaring a depth — the same rationale
        `load_fetches`' docstring already gives for its two guards."""
        mod = _mod()
        shallow = _cache_at_production_depth(tmp_path, 3)

        replayed = mod.load_production_fetches(shallow, _selection_seeded())

        assert all(len(hits) == 3 for hits in replayed.values())


class TestTheGuardIsWiredNotMerelyAvailable:
    """(d) — the asymmetry was WITHIN one function, so fix it there."""

    def test_score_from_cache_refuses_a_shallow_production_half(self, tmp_path):
        mod = _mod()
        shallow = _cache_at_production_depth(tmp_path, 3)

        with pytest.raises(mod.ProductionFetchError):
            mod.score_from_cache(fetch_cache=shallow, production_k=5)

    def test_the_committed_cache_still_scores(self):
        """The guard must not make today's artifact unbuildable."""
        scored = _mod().score_from_cache()

        assert set(scored['e2']) == set(_mod().ARM_KEYS)

    def test_score_from_cache_refuses_a_drifted_query_set_digest(self, tmp_path):
        """`score_from_cache` inherits the E2 half's fixture-digest hole
        identically, so it must inherit the guard too.

        Asserted by tampering with the CACHE rather than a committed fixture:
        the recomputed digest of the real `e2_query_set.jsonl` then disagrees
        with the one the cache claims, which is exactly the disagreement a
        working-tree fixture edit produces — without leaving the tree dirty.
        """
        mod = _mod()
        doc = json.loads(COMMITTED_FETCH_CACHE.read_text(encoding='utf-8'))
        row = next(
            r for r in doc['provenance']['fixtures']
            if r['path'].endswith('e2_query_set.jsonl')
        )
        row['sha256'] = 'deadbeef' * 8
        target = tmp_path / 'e2_fetch_cache.json'
        target.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n',
                          encoding='utf-8')

        with pytest.raises(_bake_off().FetchCacheError) as excinfo:
            mod.score_from_cache(fetch_cache=target)

        assert 'e2_query_set.jsonl' in str(excinfo.value)

    def test_main_reports_the_refusal_without_a_traceback(
        self, tmp_path, capsys,
    ):
        """`ProductionFetchError` is already in main's handler tuple, so the
        wiring must surface as exit 1 plus a named error, not a crash."""
        mod = _mod()
        shallow = _cache_at_production_depth(tmp_path, 3)

        code = mod.main([
            '--fetch-cache', str(shallow),
            '--json-out', str(tmp_path / 'out.json'),
            '--md-out', str(tmp_path / 'out.md'),
        ])

        assert code == 1
        assert 'ProductionFetchError' in capsys.readouterr().err


class TestTheLiveExtendNeverWritesBelowProductionK:
    """(e)/(f) — the writing end of the same asymmetry.

    The live branch fetched at `args.k`, the E2 window flag: `--k 3` wrote a
    3-deep production cache that a later default run scored at
    `production_k=5`.  The resolved depth is asserted directly rather than
    inferred from the file, because it is the quantity the operator sets and
    the one the printed message must name.
    """

    def _resolved_limit(self, monkeypatch, argv):
        mod = _mod()
        seen: dict = {}

        async def _stub(queries, *, limit=None, **kwargs):
            seen['limit'] = limit
            seen['queries'] = len(list(queries))
            return {
                'shape': mod.SELECTION_SHAPE,
                'corpus_fingerprint': 'stub',
                'search_limit': limit,
                'embedder_model': 'stub',
                'queries': {},
            }

        def _no_write(path, fetched):
            seen['written'] = fetched
            return Path(path)

        monkeypatch.setattr(mod, 'fetch_production_rankings', _stub)
        monkeypatch.setattr(mod, 'extend_fetch_cache', _no_write)
        assert mod.main(['--extend-cache-live', *argv]) == 0
        return seen

    def test_stock_flags_reproduce_the_committed_depth(self, monkeypatch):
        seen = self._resolved_limit(monkeypatch, [])

        assert seen['limit'] == COMMITTED_PRODUCTION_DEPTH

    def test_a_shallow_e2_window_cannot_shallow_the_production_cache(
        self, monkeypatch,
    ):
        mod = _mod()
        seen = self._resolved_limit(monkeypatch, ['--k', '3'])

        assert seen['limit'] >= mod.PRODUCTION_K

    def test_a_deeper_e2_window_still_deepens_the_fetch(self, monkeypatch):
        """A cache is allowed to be deeper than either budget — the reader
        truncates.  What it may never be is shallower than the depth it will
        be scored at."""
        seen = self._resolved_limit(monkeypatch, ['--k', '25'])

        assert seen['limit'] >= 25

    def test_the_resolved_depth_is_printed(self, monkeypatch, capsys):
        """An operator must see what was actually fetched, not what was
        asked for."""
        self._resolved_limit(monkeypatch, ['--k', '3'])

        assert f'at limit {_mod().PRODUCTION_K}' in capsys.readouterr().out

    def test_what_the_live_branch_writes_now_scores_clean(
        self, monkeypatch, tmp_path,
    ):
        """(f) closes the loop: build a cache at the depth `--k 3` now
        resolves to, and assert the scorer accepts it."""
        mod = _mod()
        seen = self._resolved_limit(monkeypatch, ['--k', '3'])
        written = _cache_at_production_depth(tmp_path, seen['limit'])

        replayed = mod.load_production_fetches(
            written, _selection_seeded(), expect_limit=mod.PRODUCTION_K,
        )

        assert replayed
