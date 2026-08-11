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


def _rec(record_id, *, topic='t', canonical=False, kind=None,
         content='body', claim_ids=(), parent_id=None, cluster_id='c1'):
    """One stored record, in the shape the read transforms consume."""
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

    def test_the_signature_takes_no_contested_argument(self):
        import inspect  # noqa: PLC0415

        params = inspect.signature(_mod().apply_promoting_topic_anchor).parameters

        assert 'contested_ids' not in params
        assert not any('contested' in name for name in params)

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
        assert dear['total_tokens'] > cheap['total_tokens']


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

    def test_the_transform_takes_no_contested_argument(self):
        import inspect  # noqa: PLC0415

        params = inspect.signature(
            _mod().apply_topic_keyed_grouped_read,
        ).parameters

        assert not any('contested' in name for name in params)


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
