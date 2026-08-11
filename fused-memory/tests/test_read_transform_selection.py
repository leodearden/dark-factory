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
