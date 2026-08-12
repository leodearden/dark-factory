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

    def test_the_signature_takes_no_contested_channel(self):
        import inspect  # noqa: PLC0415

        params = inspect.signature(_mod().apply_topic_diversity_cap).parameters

        assert not [p for p in params if 'contested' in p], (
            'the cap must be computable from `topic` alone — `contested` is '
            'absent from RESERVED_VOCABULARY_KEYS and has no writer'
        )

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
            hits, 't-a', canonical.record_id, k=5,
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
            hits, 't-a', canonical.record_id, k=5,
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
            hits, 't-a', canonical.record_id, k=5,
        )
        landed = bake_off.topic_discoverability(hits, 't-a', canonical.record_id, 5)

        assert measured['aliased_in_top_k'] == landed['canonical_in_top_k']
        assert measured['aliased_rank'] == landed['canonical_rank']

    def test_an_absent_canonical_is_found_by_neither(self):
        mod = _mod()
        hits = [_rec('unrelated', topic='t-z')]

        measured = mod.canonical_discoverability(hits, 't-a', 'anchor', k=5)

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

        measured = mod.canonical_discoverability(hits, 't-a', 'anchor', k=5)

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
            provenance=result.provenance,
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
            provenance=result.provenance,
        )

        assert result.provenance[canonical.record_id].canonical_was_itself_ranked
        assert measured['aliased_in_top_k'] is True
        assert measured['unaliased_in_top_k'] is True

    def test_without_the_provenance_the_metric_cannot_tell_and_says_so(self):
        """Aliasing is invisible in the record alone.

        A synthesized document is byte-indistinguishable from a stored one
        at the `record_id` level — that is the entire problem.  So passing
        no provenance must NOT quietly fall back to crediting: it reports
        the aliased rate and `None` for the unaliased one, which the report
        renders as no-measurement rather than as a measured zero.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=False)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )
        measured = mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
        )

        assert measured['aliased_in_top_k'] is True
        assert measured['unaliased_in_top_k'] is None
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

        mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            provenance=result.provenance,
        )

        assert _ids(result.records) == records_before
        assert result.provenance == provenance_before


class TestOnlyASynthesizedRecordCanAlias:
    """The discriminator that lets a flat arm be scored with no disclosure.

    A missing `provenance` must not mean "unknown" unconditionally: a
    STORED record carrying the canonical's id simply IS the canonical, and
    reporting `None` for every flat arm would render the honest column as
    no-measurement across most of the table.  Only a SYNTHESIZED document
    can alias, and a synthesized document is self-identifying by `role`.
    """

    def test_a_stored_canonical_needs_no_disclosure_to_be_credited(self):
        mod = _mod()
        hits, _, canonical = _aliasing_corpus(canonical_ranked=True)

        measured = mod.canonical_discoverability(
            hits, 't-a', canonical.record_id, k=5,
        )

        assert measured['unaliased_in_top_k'] is True
        assert measured['unaliased_rank'] == 2

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
            result.records, 't-a', canonical.record_id, k=5,
        )['unaliased_in_top_k'] is None

    def test_disclosure_beats_the_role_fallback_when_both_are_present(self):
        """Provenance is EVIDENCE; the role is only a discriminator.

        A grouped document whose canonical really did rank is credited —
        the role alone would have refused it, so the disclosure must win.
        """
        mod = _mod()
        hits, index, canonical = _aliasing_corpus(canonical_ranked=True)

        result = mod.apply_topic_keyed_grouped_read(
            hits, index, render_sightings=False,
        )

        assert mod.canonical_discoverability(
            result.records, 't-a', canonical.record_id, k=5,
            provenance=result.provenance,
        )['unaliased_in_top_k'] is True


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
              source='briefing_template', observed_count=74103,
              traffic_share=0.172892, observed_limit=5):
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
