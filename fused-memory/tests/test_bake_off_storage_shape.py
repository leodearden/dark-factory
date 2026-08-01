"""Tests for bake_off_storage_shape.py — the E2 storage-shape bake-off (task 3199).

The script is loaded via importlib so it can be tested without sys.path
pollution — mirrors test_calibrate_write_triage.py and
test_memory_eval_retrieval_probe.py.  The loader is invoked lazily (via
``_mod()``) rather than bound at import time, so the fixture-contract tests
stay runnable independently of the script.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file is free of network, Qdrant and OPENAI_API_KEY
**except the single live end-to-end test**, which carries its markers
PER-TEST::

    @pytest.mark.integration
    @pytest.mark.timeout(300)
    @qdrant_skipif()
    @pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), ...)

Never via a module-level ``pytestmark``.  ``fused-memory/pyproject.toml``
sets ``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_memory_eval_retrieval_probe.py:3419-3422``.

All retrievals in the pure tests are INJECTED (hand-built ranked hit lists
with exactly-known answers), never embedded.  That is what lets the metric
tests assert exact values with no tolerances.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
ALPHA_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration.jsonl'
REGISTRY_PATH = FIXTURES_DIR / 'memory_eval_topic_registry.json'
ARM_CLAIMS_PATH = FIXTURES_DIR / 'e2_arm_claims.jsonl'
QUERY_SET_PATH = FIXTURES_DIR / 'e2_query_set.jsonl'
DISTRACTOR_SLAB_PATH = FIXTURES_DIR / 'e2_distractor_slab.jsonl'


def _load_module() -> types.ModuleType:
    """Load bake_off_storage_shape.py from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'bake_off_storage_shape'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
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
    return _load_module()


# ===========================================================================
# step-1 — fixture loaders
# ===========================================================================
#
# Contract paths read the COMMITTED fixtures; error paths build tiny
# synthetic fixtures in tmp_path.  No network, no Qdrant.

import json  # noqa: E402

import pytest  # noqa: E402


class TestLoadCalibrationClusters:
    """`load_calibration_clusters` groups the α fixture into its clusters."""

    def test_groups_the_committed_fixture_into_twenty_clusters(self):
        clusters = _mod().load_calibration_clusters(ALPHA_FIXTURE_PATH)

        assert len(clusters) == 20
        assert sum(len(c.members) for c in clusters.values()) == 104

    def test_each_cluster_exposes_its_single_canonical_and_keeps_labels(self):
        clusters = _mod().load_calibration_clusters(ALPHA_FIXTURE_PATH)

        for cluster_id, cluster in clusters.items():
            assert cluster.cluster_id == cluster_id
            # Exactly one canonical, and it is the record whose id IS the
            # cluster id (the α fixture's own invariant).
            canonicals = [r for r in cluster.members if r['label'] == 'canonical']
            assert len(canonicals) == 1
            assert cluster.canonical['memory_id'] == cluster_id
            assert cluster.canonical is canonicals[0]
            # Labels survive grouping — the guard-adequacy probe selects on
            # label=='duplicate', so a loader that dropped them would
            # silently change which record is the probing write.
            assert all(r['label'] for r in cluster.members)

    def test_malformed_line_raises_with_its_one_based_line_number(self, tmp_path):
        path = tmp_path / 'broken.jsonl'
        path.write_text('{"memory_id": "a", "cluster_id": "a", "label": "canonical"}\nnot json\n')

        with pytest.raises(_mod().FixtureError, match=r':2:'):
            _mod().load_calibration_clusters(path)

    def test_cluster_without_a_canonical_is_named_loudly(self, tmp_path):
        path = tmp_path / 'no_canonical.jsonl'
        path.write_text(json.dumps({
            'memory_id': 'm1', 'cluster_id': 'c1', 'label': 'duplicate',
            'content': 'x', 'category': 'procedural_knowledge',
        }) + '\n')

        with pytest.raises(_mod().FixtureError, match='c1'):
            _mod().load_calibration_clusters(path)


class TestLoadRegistryTopics:
    """`load_registry_topics` keys the curator-gate entries by cluster id."""

    def test_returns_the_twenty_curator_gate_entries_keyed_by_cluster_id(self):
        topics = _mod().load_registry_topics(REGISTRY_PATH)

        assert len(topics) == 20
        for cluster_id, topic in topics.items():
            assert topic.cluster_id == cluster_id
            assert topic.topic and topic.topic == topic.topic.strip()
            assert len(topic.phrasings) == 3
            assert any(p['held_out'] for p in topic.phrasings)

    def test_ignores_entries_not_derived_from_the_curator_gate(self, tmp_path):
        path = tmp_path / 'registry.json'
        path.write_text(json.dumps({'schema_version': 1, '_disclosures': {}, 'entries': [
            {'topic': 'kept', 'derived_from': 'curator_gate',
             'phrasings': [{'text': 'p', 'held_out': True}],
             'provenance': {'cluster_id': 'c1'}},
            {'topic': 'dropped', 'derived_from': 'census_topic',
             'phrasings': [{'text': 'p', 'held_out': False}],
             'provenance': {'cluster_id': 'c2'}},
        ]}))

        topics = _mod().load_registry_topics(path)

        assert list(topics) == ['c1']


class TestLoadE2Fixtures:
    """The three E2-owned fixtures parse into their declared shapes."""

    def test_load_arm_claims_reads_every_committed_claim(self):
        claims = _mod().load_arm_claims(ARM_CLAIMS_PATH)

        assert len(claims) == 176
        assert len({c.claim_id for c in claims}) == 176
        assert all(isinstance(c.canonical, bool) and isinstance(c.contested, bool) for c in claims)

    def test_load_query_set_reads_both_query_kinds(self):
        queries = _mod().load_query_set(QUERY_SET_PATH)

        assert len(queries) == 236
        assert sum(1 for q in queries if q.kind == 'topic_phrasing') == 60
        assert sum(1 for q in queries if q.kind == 'claim') == 176
        assert {q.kind for q in queries} == {'topic_phrasing', 'claim'}

    def test_load_distractor_slab_reads_the_frozen_slab(self):
        slab = _mod().load_distractor_slab(DISTRACTOR_SLAB_PATH)

        assert len(slab) == 300
        assert len({d.distractor_id for d in slab}) == 300
        # A distractor must never be topic-anchorable, or it stops being a
        # distractor and starts being a right answer for the pin to find.
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS  # noqa: PLC0415
        for d in slab:
            assert not (RESERVED_VOCABULARY_KEYS & set(d.raw))

    def test_missing_distractor_slab_says_how_to_regenerate_it(self, tmp_path):
        """A missing slab must NEVER fall back to an empty one.

        Seeding no distractors would quietly delete the contamination
        variable the eval doc says matters most, and the run would look
        entirely successful while measuring a different experiment.
        """
        missing = tmp_path / 'absent.jsonl'

        with pytest.raises(_mod().FixtureError) as excinfo:
            _mod().load_distractor_slab(missing)

        message = str(excinfo.value)
        assert 'absent.jsonl' in message
        assert 'README' in message  # points at the regeneration procedure


class TestFixtureCrossValidation:
    """The four fixtures must agree with each other, not merely parse."""

    def test_every_alpha_cluster_has_a_registry_topic_and_at_least_one_claim(self):
        mod = _mod()
        clusters = mod.load_calibration_clusters(ALPHA_FIXTURE_PATH)
        topics = mod.load_registry_topics(REGISTRY_PATH)
        claims = mod.load_arm_claims(ARM_CLAIMS_PATH)

        claimed = {c.cluster_id for c in claims}
        assert set(clusters) == set(topics) == claimed

    def test_every_claim_topic_matches_its_clusters_registry_topic(self):
        mod = _mod()
        topics = mod.load_registry_topics(REGISTRY_PATH)

        for claim in mod.load_arm_claims(ARM_CLAIMS_PATH):
            assert claim.topic == topics[claim.cluster_id].topic

    def test_every_claim_source_memory_id_resolves_inside_its_own_cluster(self):
        mod = _mod()
        clusters = mod.load_calibration_clusters(ALPHA_FIXTURE_PATH)

        for claim in mod.load_arm_claims(ARM_CLAIMS_PATH):
            member_ids = {r['memory_id'] for r in clusters[claim.cluster_id].members}
            assert claim.source_memory_id in member_ids

    def test_exactly_one_canonical_claim_per_cluster(self):
        mod = _mod()
        claims = mod.load_arm_claims(ARM_CLAIMS_PATH)

        per_cluster: dict[str, int] = {}
        for claim in claims:
            per_cluster.setdefault(claim.cluster_id, 0)
            if claim.canonical:
                per_cluster[claim.cluster_id] += 1
        assert set(per_cluster.values()) == {1}

    def test_every_query_expectation_resolves_to_a_claim_in_its_own_cluster(self):
        mod = _mod()
        claims = {c.claim_id: c for c in mod.load_arm_claims(ARM_CLAIMS_PATH)}

        for query in mod.load_query_set(QUERY_SET_PATH):
            assert query.expects_claim_ids
            for claim_id in query.expects_claim_ids:
                assert claim_id in claims
                assert claims[claim_id].cluster_id == query.cluster_id

    def test_cross_validation_names_the_offending_cluster(self, tmp_path):
        """A fixture disagreement must be reported by name, not as a KeyError."""
        mod = _mod()
        claims_path = tmp_path / 'claims.jsonl'
        claims_path.write_text(json.dumps({
            'claim_id': 'orphan-01', 'cluster_id': 'nope', 'topic': 'orphan',
            'text': 'x', 'source_memory_id': 'nope', 'canonical': True,
            'b_arm_role': 'canonical', 'contested': False,
        }) + '\n')

        with pytest.raises(mod.FixtureError, match='nope'):
            mod.cross_validate_fixtures(
                clusters=mod.load_calibration_clusters(ALPHA_FIXTURE_PATH),
                topics=mod.load_registry_topics(REGISTRY_PATH),
                claims=mod.load_arm_claims(claims_path),
                queries=[],
            )


class TestDefaultFixturePaths:
    """Default paths are package-relative — never a per-task worktree path."""
    def test_defaults_point_at_the_committed_fixtures(self):
        mod = _mod()

        assert mod.DEFAULT_ARM_CLAIMS_PATH == ARM_CLAIMS_PATH
        assert mod.DEFAULT_QUERY_SET_PATH == QUERY_SET_PATH
        assert mod.DEFAULT_DISTRACTOR_SLAB_PATH == DISTRACTOR_SLAB_PATH
        assert mod.DEFAULT_ALPHA_FIXTURE_PATH == ALPHA_FIXTURE_PATH
        assert mod.DEFAULT_REGISTRY_PATH == REGISTRY_PATH

    def test_paths_are_derived_from___file___not_baked_in(self):
        """The lesson test_calibrate_write_triage.py:1267 pins.

        A path resolved at AUTHOR time breaks the moment the script runs
        from another checkout — and because this task is itself authored
        inside `.worktrees/3199`, a baked-in literal would be invisible to
        any test that merely inspects the resolved value (it would look
        perfectly plausible). So assert on the SOURCE: no absolute path
        literal anywhere, and every default is a child of the package root
        the module derived from its own ``__file__``.
        """
        mod = _mod()
        source = SCRIPT_PATH.read_text()

        assert '/home/' not in source
        assert '.worktrees' not in source

        package_root = Path(mod.__file__).resolve().parent.parent
        for default in (
            mod.DEFAULT_ARM_CLAIMS_PATH,
            mod.DEFAULT_QUERY_SET_PATH,
            mod.DEFAULT_DISTRACTOR_SLAB_PATH,
            mod.DEFAULT_ALPHA_FIXTURE_PATH,
            mod.DEFAULT_REGISTRY_PATH,
        ):
            assert default.is_absolute()
            assert default.is_relative_to(package_root)
            assert default.exists()


# ===========================================================================
# step-3 — arm materialization
# ===========================================================================
#
# `materialize_arm` turns the SAME knowledge into the three storage shapes
# E2 arbitrates between (eval-design §5 E2 / PRD D9):
#
#   status_quo — the α corpus exactly as it actually existed: long original
#                records, no vocabulary metadata at all.
#   c_peers    — short single-claim peers, flat, all sharing a `topic`, with
#                exactly one carrying `canonical: True` (PRD's Option C).
#   b_grouped  — one canonical plus `parent_id` children (PRD's δ/Option B).
#
# Every arm additionally carries the IDENTICAL frozen distractor slab, which
# is what makes the comparison controlled: the eval doc's contamination
# result says distractors are what actually move retrieval.
#
# All pure — no network, no Qdrant, no embedder.


@functools.cache
def _committed_inputs() -> dict:
    """The four committed fixtures, loaded once for the whole module."""
    mod = _mod()
    return {
        'clusters': mod.load_calibration_clusters(ALPHA_FIXTURE_PATH),
        'topics': mod.load_registry_topics(REGISTRY_PATH),
        'claims': mod.load_arm_claims(ARM_CLAIMS_PATH),
        'distractors': mod.load_distractor_slab(DISTRACTOR_SLAB_PATH),
    }


@functools.cache
def _arm(shape: str) -> tuple:
    """Materialize `shape` from the committed fixtures (cached, immutable view)."""
    return tuple(_mod().materialize_arm(shape, **_committed_inputs()))


def _knowledge(records) -> list:
    """Just the arm's own records — the distractor slab filtered out."""
    return [r for r in records if r.role != 'distractor']


def _distractors(records) -> list:
    return [r for r in records if r.role == 'distractor']


class TestArmShapesBand:
    """The arm set is pinned by EQUALITY, so a shape cannot be added silently.

    E2's whole claim is that it compared *these three* shapes; a fourth
    appearing without the report growing a column would make the decision
    table quietly incomplete.
    """

    def test_arm_shapes_are_exactly_the_three_e2_arms(self):
        assert _mod().ARM_SHAPES == ('status_quo', 'c_peers', 'b_grouped')

    def test_unknown_shape_is_rejected_by_name(self):
        mod = _mod()

        with pytest.raises(ValueError, match='hybrid'):
            mod.materialize_arm('hybrid', **_committed_inputs())


class TestMaterializeStatusQuoArm:
    """Arm (a): the corpus as it actually existed — long originals, no vocabulary."""

    def test_emits_every_alpha_record_verbatim(self):
        records = _knowledge(_arm('status_quo'))
        clusters = _committed_inputs()['clusters']

        originals = {r['memory_id']: r for c in clusters.values() for r in c.members}
        assert len(records) == 104
        assert {r.record_id for r in records} == set(originals)
        for record in records:
            # Verbatim: the status-quo arm must not be "helpfully" cleaned up,
            # or it stops being the baseline the other two are measured against.
            assert record.content == originals[record.record_id]['content']
            assert record.cluster_id == originals[record.record_id]['cluster_id']

    def test_carries_no_reserved_vocabulary_key_at_all(self):
        """The α records genuinely have no `metadata` key — arm (a) reproduces that."""
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS  # noqa: PLC0415

        for record in _knowledge(_arm('status_quo')):
            assert not (RESERVED_VOCABULARY_KEYS & set(record.metadata))

    def test_role_preserves_the_alpha_label(self):
        """The guard-adequacy probe selects the last `duplicate` record, so the
        α labels must survive into the arm rather than being flattened."""
        records = _knowledge(_arm('status_quo'))

        assert {r.role for r in records} == {
            'canonical', 'duplicate', 'pseudo_contradiction', 'distinct',
        }


class TestMaterializeCPeersArm:
    """Arm (c): short single-claim peers sharing a topic, one flagged canonical."""

    def test_emits_one_peer_per_claim_with_the_claim_text(self):
        records = _knowledge(_arm('c_peers'))
        claims = {c.claim_id: c for c in _committed_inputs()['claims']}

        assert len(records) == len(claims) == 176
        for record in records:
            assert record.claim_ids and len(record.claim_ids) == 1
            claim = claims[record.claim_ids[0]]
            assert record.content == claim.text
            assert record.cluster_id == claim.cluster_id

    def test_all_peers_of_a_cluster_share_the_registry_topic(self):
        """PRD D4: one topic namespace — slugs are taken verbatim from E1."""
        topics = _committed_inputs()['topics']

        for record in _knowledge(_arm('c_peers')):
            assert record.metadata['topic'] == topics[record.cluster_id].topic

    def test_exactly_one_peer_per_cluster_is_flagged_canonical(self):
        per_cluster: dict[str, int] = {}
        for record in _knowledge(_arm('c_peers')):
            per_cluster.setdefault(record.cluster_id, 0)
            # β's `invalid_canonical_type` rule is bool-identity, so the arm
            # must emit a real True — a truthy 1 would be a fatal violation.
            if record.metadata.get('canonical') is True:
                per_cluster[record.cluster_id] += 1
        assert len(per_cluster) == 20
        assert set(per_cluster.values()) == {1}

    def test_the_canonical_peer_is_short_not_a_concatenation(self):
        """PRD §3: arm (c)'s canonical is an INDEX claim, not a rolled-up digest.

        This is the load-bearing difference between C and B. If the canonical
        were built by concatenating its cluster's peers, arm (c) would win
        claim-recall trivially (every claim is inside the canonical) and lose
        tokens-per-query for a reason that has nothing to do with the storage
        shape — the experiment would measure a concatenation, not a peer set.
        """
        by_cluster: dict[str, list] = {}
        for record in _knowledge(_arm('c_peers')):
            by_cluster.setdefault(record.cluster_id, []).append(record)

        clusters = _committed_inputs()['clusters']
        for cluster_id, records in by_cluster.items():
            canonical = next(r for r in records if r.metadata.get('canonical') is True)
            peers = [r for r in records if r is not canonical]

            # Not a concatenation: shorter than its own peers put together...
            assert len(canonical.content) < sum(len(p.content) for p in peers)
            # ...and no longer than the longest single peer, i.e. it sits
            # inside the peer band rather than above it.
            assert len(canonical.content) <= max(len(p.content) for p in peers)
            # And dramatically shorter than the long α original it replaces —
            # which is exactly the D4 cost question arm (c) exists to answer.
            assert len(canonical.content) < len(clusters[cluster_id].canonical['content'])

    def test_arm_c_is_flat_no_parent_links(self):
        """Peers-as-default means no parent edge — that is arm (b)'s shape."""
        for record in _knowledge(_arm('c_peers')):
            assert 'parent_id' not in record.metadata


class TestMaterializeBGroupedArm:
    """Arm (b): one canonical per cluster plus `parent_id` children."""

    def test_each_cluster_has_exactly_one_parentless_canonical(self):
        by_cluster: dict[str, list] = {}
        for record in _knowledge(_arm('b_grouped')):
            by_cluster.setdefault(record.cluster_id, []).append(record)

        assert len(by_cluster) == 20
        for records in by_cluster.values():
            canonicals = [r for r in records if r.metadata.get('canonical') is True]
            assert len(canonicals) == 1
            assert 'parent_id' not in canonicals[0].metadata

    def test_every_child_points_at_its_clusters_canonical(self):
        by_cluster: dict[str, list] = {}
        for record in _knowledge(_arm('b_grouped')):
            by_cluster.setdefault(record.cluster_id, []).append(record)

        for records in by_cluster.values():
            canonical = next(r for r in records if r.metadata.get('canonical') is True)
            children = [r for r in records if r is not canonical]
            assert children  # a grouped arm with no children is not grouped
            for child in children:
                assert child.metadata['parent_id'] == canonical.record_id

    def test_children_carry_a_registry_kind_never_the_word_canonical(self):
        """`kind: 'canonical'` is NOT in β's KIND_REGISTRY and would be fatal.

        Canonicality is expressed by the `canonical` key, not by a `kind` —
        this test pins that the arm does not conflate the two.
        """
        from fused_memory.memory_metadata import KIND_REGISTRY  # noqa: PLC0415

        for record in _knowledge(_arm('b_grouped')):
            kind = record.metadata.get('kind')
            if record.metadata.get('canonical') is True:
                assert kind is None
                continue
            assert kind in {'amendment', 'sighting'}
            assert kind in KIND_REGISTRY

    def test_parent_ids_are_canonical_dashed_uuids(self):
        """β's `_is_full_uuid` rule: 36 chars, dashed, and its own str() round-trip.

        A short hex id or a bare hex32 would trip `invalid_parent_id_shape`,
        so the arm cannot mint ids casually.
        """
        import uuid  # noqa: PLC0415

        seen = set()
        for record in _knowledge(_arm('b_grouped')):
            parent_id = record.metadata.get('parent_id')
            if parent_id is None:
                continue
            assert len(parent_id) == 36
            assert str(uuid.UUID(parent_id)) == parent_id
            seen.add(parent_id)
        assert len(seen) == 20  # one canonical per cluster


class TestDistractorSlabIsSharedAndInert:
    """The contamination variable must be IDENTICAL across arms, and inert."""

    def test_every_arm_carries_the_same_slab_in_the_same_order(self):
        slabs = {shape: _distractors(_arm(shape)) for shape in _mod().ARM_SHAPES}

        for records in slabs.values():
            assert len(records) == 300
        reference = [(r.record_id, r.content) for r in slabs['status_quo']]
        for shape, records in slabs.items():
            assert [(r.record_id, r.content) for r in records] == reference, shape

    def test_distractors_are_never_topic_anchorable(self):
        """A distractor carrying a `topic` would stop being a distractor and
        start being a right answer for the pin to find."""
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS  # noqa: PLC0415

        for shape in _mod().ARM_SHAPES:
            for record in _distractors(_arm(shape)):
                assert not (RESERVED_VOCABULARY_KEYS & set(record.metadata))
                assert record.cluster_id is None
                assert record.claim_ids == []


class TestClaimCoverageParity:
    """The mechanical anti-laziness guard for the blind-authoring protocol.

    The eval doc names the experiment's own biggest weakness: "arm quality
    reflects authoring skill — the experiment is gameable by authoring one
    arm well and another lazily". The floor against that is coverage parity:
    an arm cannot be lazily authored by simply dropping claims.

    Deliberately NOT total-content-length parity — arm (a)'s long originals
    versus arm (c)'s short peers differ BY CONSTRUCTION, and that difference
    IS the tokens-per-query metric (see the test below).
    """

    def test_every_arm_realizes_exactly_the_same_claim_set(self):
        mod = _mod()
        all_claim_ids = {c.claim_id for c in _committed_inputs()['claims']}

        realized = {
            shape: {cid for r in _arm(shape) for cid in r.claim_ids}
            for shape in mod.ARM_SHAPES
        }
        for shape, ids in realized.items():
            assert ids == all_claim_ids, f'{shape} does not realize every claim'

    def test_status_quo_realizes_claims_through_its_source_records(self):
        """For arm (a) a claim is realized by its `source_memory_id` record
        being present — that is what makes parity checkable at all for an arm
        that was never decomposed into claims."""
        claims = _committed_inputs()['claims']
        by_record = {r.record_id: r for r in _knowledge(_arm('status_quo'))}

        for claim in claims:
            assert claim.claim_id in by_record[claim.source_memory_id].claim_ids

    def test_arms_deliberately_differ_in_length_that_is_the_d4_metric(self):
        """The inverse guard: assert the arms were NOT length-equalized.

        If some future edit "balanced" the arms to look fair, it would delete
        the tokens-per-query result the D4 question is asked to answer.
        """
        totals = {
            shape: sum(len(r.content) for r in _knowledge(_arm(shape)))
            for shape in _mod().ARM_SHAPES
        }

        assert totals['status_quo'] > 2 * totals['c_peers']
        assert totals['c_peers'] == totals['b_grouped']  # same claim bodies


class TestCategoryIsPreservedAcrossArms:
    """Every arm record carries the SAME category the same knowledge has in
    arm (a) — otherwise the guard replay measures the wrong thing.

    `find_near_duplicate_memory` defensively filters on `category`
    (near_duplicate_guard.py:78), so an arm whose records carried no category
    — or a shifted one — would score zero guard adequacy for a reason that
    has nothing to do with its storage shape. That failure would look exactly
    like a genuine result, which is what makes it worth pinning.
    """

    def test_every_record_in_every_arm_carries_a_category(self):
        for shape in _mod().ARM_SHAPES:
            for record in _arm(shape):
                assert record.metadata.get('category'), (shape, record.record_id)

    def test_a_claims_category_matches_its_source_record_in_arm_a(self):
        claims = {c.claim_id: c for c in _committed_inputs()['claims']}
        status_quo = {r.record_id: r for r in _knowledge(_arm('status_quo'))}

        for shape in ('c_peers', 'b_grouped'):
            for record in _knowledge(_arm(shape)):
                claim = claims[record.claim_ids[0]]
                source = status_quo[claim.source_memory_id]
                assert record.metadata['category'] == source.metadata['category']

    def test_the_two_category_mixed_alpha_clusters_are_not_flattened(self):
        """Two of the twenty α clusters are genuinely category-mixed.

        Deriving category per-cluster instead of per-source-record would
        silently edit the corpus — so assert the mixing survives.
        """
        mixed = 0
        for shape in _mod().ARM_SHAPES:
            per_cluster: dict[str, set] = {}
            for record in _knowledge(_arm(shape)):
                per_cluster.setdefault(record.cluster_id, set()).add(
                    record.metadata['category']
                )
            mixed = max(mixed, sum(1 for v in per_cluster.values() if len(v) > 1))
            assert sum(1 for v in per_cluster.values() if len(v) > 1) == 2, shape
        assert mixed == 2


class TestBetaVocabularyConformance:
    """Every emitted metadata dict must be storable through the real seam.

    The bake-off writes through `Mem0Backend` directly, so it bypasses the
    service-seam validation entirely. Running β/3195's validator over the
    emitted metadata keeps the experiment measuring shapes the system could
    ACTUALLY store — otherwise an arm could win by using a shape the write
    boundary would have rejected in production.
    """

    def test_no_arm_emits_a_fatal_metadata_violation(self):
        from fused_memory.memory_metadata import validate_memory_metadata  # noqa: PLC0415

        for shape in _mod().ARM_SHAPES:
            for record in _arm(shape):
                violations = validate_memory_metadata(
                    dict(record.metadata), enforce_kind_registry=True
                )
                fatal = [v for v in violations if v.fatal]
                assert not fatal, f'{shape}/{record.record_id}: {fatal}'

    def test_every_topic_matches_the_shared_slug_shape(self):
        """PRD D4's one-namespace rule, pinned mechanically rather than by
        convention — the slugs come verbatim from E1's registry."""
        from fused_memory.memory_metadata import (  # noqa: PLC0415
            TOPIC_SLUG_MAX_LEN,
            TOPIC_SLUG_RE,
        )

        topicked = 0
        for shape in _mod().ARM_SHAPES:
            for record in _arm(shape):
                topic = record.metadata.get('topic')
                if topic is None:
                    continue
                topicked += 1
                assert TOPIC_SLUG_RE.match(topic)
                assert len(topic) <= TOPIC_SLUG_MAX_LEN
        assert topicked  # the anchored arms really do carry topics


class TestMaterializationIsDeterministic:
    """A rerun must seed byte-identical collections, or the two runs' reports
    are not comparable and a diff stops being signal."""

    def test_two_materializations_agree_exactly(self):
        mod = _mod()

        for shape in mod.ARM_SHAPES:
            first = mod.materialize_arm(shape, **_committed_inputs())
            second = mod.materialize_arm(shape, **_committed_inputs())

            assert [(r.record_id, r.content, r.metadata, r.role) for r in first] == [
                (r.record_id, r.content, r.metadata, r.role) for r in second
            ]

    def test_record_ids_are_unique_within_an_arm(self):
        for shape in _mod().ARM_SHAPES:
            records = _arm(shape)
            assert len({r.record_id for r in records}) == len(records), shape


# ===========================================================================
# step-5 — apply_grouped_read (arm-local reference implementation of 3129)
# ===========================================================================
#
# `server/grouped_read.py` DOES NOT EXIST: task 3129 is deferred behind gate
# η, which depends on this task. A downstream task structurally cannot supply
# an upstream premise, so the bake-off carries its own arm-local reference
# implementation of PRD V2/D6 — which doubles as the executable specification
# 3129 can port if the gate ratifies grouping.
#
# The two rules that are NOT negotiable:
#
#   D6 — a child hit must resolve UPWARD to its parent's grouped document.
#        Without it a child's content becomes unreachable, which is the whole
#        objection to δ/Option B.
#   V2 — a `contested` child is NEVER suppressed. It survives as its own hit
#        alongside the grouped document (the esc-5712 shape: a contested
#        amendment folded invisibly into a canonical is how a disagreement
#        gets silently resolved in favour of whoever wrote the canonical).
#
# All hit lists here are hand-built with exactly-known answers — no
# embeddings, so every assertion is exact and tolerance-free.


def _hit(record_id, *, parent_id=None, canonical=False, kind=None,
         contested=False, content='body', claim_ids=(), topic='t'):
    """One ranked hit, in the shape `apply_grouped_read` consumes."""
    metadata: dict = {'category': 'procedural_knowledge', 'topic': topic}
    if parent_id is not None:
        metadata['parent_id'] = parent_id
    if canonical:
        metadata['canonical'] = True
    if kind is not None:
        metadata['kind'] = kind
    return _mod().ArmRecord(
        record_id=record_id,
        content=content,
        metadata=metadata,
        cluster_id='c1',
        claim_ids=list(claim_ids),
        role='canonical' if canonical else (kind or 'peer'),
    ), contested


def _index(*pairs):
    """Build (hits, records_by_id, contested_ids) from `_hit` pairs."""
    records = [record for record, _ in pairs]
    contested = {record.record_id for record, is_contested in pairs if is_contested}
    return records, {r.record_id: r for r in records}, contested


PARENT = '11111111-1111-4111-8111-111111111111'


class TestGroupedReadUpwardResolution:
    """D6: a child hit resolves upward — a child's content is never unreachable."""

    def test_a_child_only_hit_returns_its_parents_grouped_document(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        child, _ = _hit('child-a', parent_id=PARENT, kind='amendment',
                        content='AMEND', claim_ids=['k2'])
        records_by_id = {PARENT: parent, 'child-a': child}

        grouped = mod.apply_grouped_read([child], records_by_id, contested_ids=set())

        assert len(grouped) == 1
        assert grouped[0].record_id == PARENT
        # The child's content must be REACHABLE through the group, not lost.
        assert 'AMEND' in grouped[0].content
        assert 'CANON' in grouped[0].content
        # And its claim must be credited to the group, or arm (b) would be
        # unfairly penalised on claim-recall for grouping correctly.
        assert set(grouped[0].claim_ids) == {'k1', 'k2'}

    def test_two_children_of_one_parent_collapse_to_a_single_document(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        first, _ = _hit('child-a', parent_id=PARENT, kind='amendment', claim_ids=['k2'])
        second, _ = _hit('child-b', parent_id=PARENT, kind='sighting', claim_ids=['k3'])
        records_by_id = {PARENT: parent, 'child-a': first, 'child-b': second}

        grouped = mod.apply_grouped_read(
            [first, second], records_by_id, contested_ids=set()
        )

        assert len(grouped) == 1
        assert set(grouped[0].claim_ids) == {'k1', 'k2', 'k3'}

    def test_collapse_preserves_the_better_of_the_collapsed_ranks(self):
        """The group must land where its BEST member ranked.

        Demoting it to the worse rank would make grouping look worse than it
        is at every k — an artifact of the transform, not of the shape.
        """
        mod = _mod()
        other, _ = _hit('unrelated-1', content='X')
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        early, _ = _hit('child-a', parent_id=PARENT, kind='amendment')
        late, _ = _hit('child-b', parent_id=PARENT, kind='sighting')
        records_by_id = {
            PARENT: parent, 'child-a': early, 'child-b': late, 'unrelated-1': other,
        }

        grouped = mod.apply_grouped_read(
            [early, other, late], records_by_id, contested_ids=set()
        )

        assert [r.record_id for r in grouped] == [PARENT, 'unrelated-1']

    def test_a_parent_hit_and_its_own_child_hit_do_not_duplicate_the_group(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        child, _ = _hit('child-a', parent_id=PARENT, kind='amendment', claim_ids=['k2'])
        records_by_id = {PARENT: parent, 'child-a': child}

        grouped = mod.apply_grouped_read(
            [parent, child], records_by_id, contested_ids=set()
        )

        assert [r.record_id for r in grouped] == [PARENT]
        assert set(grouped[0].claim_ids) == {'k1', 'k2'}


class TestGroupedReadDocumentShape:
    """The grouped document is canonical body + amendment digests + a count."""

    def test_document_carries_canonical_body_amendment_digests_and_a_sighting_count(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='THE CANONICAL BODY')
        amend, _ = _hit('child-a', parent_id=PARENT, kind='amendment',
                        content='THE AMENDMENT TEXT')
        seen_one, _ = _hit('child-b', parent_id=PARENT, kind='sighting', content='S1')
        seen_two, _ = _hit('child-c', parent_id=PARENT, kind='sighting', content='S2')
        records_by_id = {
            PARENT: parent, 'child-a': amend, 'child-b': seen_one, 'child-c': seen_two,
        }

        grouped = mod.apply_grouped_read(
            [amend, seen_one, seen_two], records_by_id, contested_ids=set()
        )

        document = grouped[0].content
        assert document.startswith('THE CANONICAL BODY')
        assert 'THE AMENDMENT TEXT' in document
        # Sightings are counted, not pasted — that IS the D4 cost claim
        # grouping makes, so the transform has to actually make it.
        assert 'S1' not in document and 'S2' not in document
        assert '2' in document
        assert grouped[0].role == 'grouped'

    def test_grouped_document_keeps_the_parents_identity_and_topic(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', topic='alpha-topic')
        child, _ = _hit('child-a', parent_id=PARENT, kind='amendment', topic='alpha-topic')
        records_by_id = {PARENT: parent, 'child-a': child}

        grouped = mod.apply_grouped_read([child], records_by_id, contested_ids=set())

        assert grouped[0].record_id == PARENT
        assert grouped[0].metadata['topic'] == 'alpha-topic'
        assert grouped[0].metadata['canonical'] is True


class TestGroupedReadPassThrough:
    """A hit with no parent link is untouched and stays in place."""

    def test_parentless_hits_pass_through_unchanged_and_in_order(self):
        mod = _mod()
        first, _ = _hit('flat-1', content='A')
        second, _ = _hit('flat-2', content='B')
        third, _ = _hit('flat-3', content='C')
        hits, records_by_id, contested = _index(
            (first, False), (second, False), (third, False),
        )

        grouped = mod.apply_grouped_read(hits, records_by_id, contested_ids=contested)

        assert grouped == hits  # identity, not merely equal-length

    def test_an_empty_hit_list_is_returned_empty(self):
        assert _mod().apply_grouped_read([], {}, contested_ids=set()) == []

    def test_a_child_whose_parent_is_absent_from_the_index_survives(self):
        """A dangling parent link must never delete the hit.

        `parent_id` liveness is leaf δ's problem, not this transform's — and
        dropping the hit would silently lose a real answer.
        """
        mod = _mod()
        orphan, _ = _hit('child-a', parent_id=PARENT, kind='amendment', content='ORPHAN')

        grouped = mod.apply_grouped_read(
            [orphan], {'child-a': orphan}, contested_ids=set()
        )

        assert [r.record_id for r in grouped] == ['child-a']
        assert grouped[0].content == 'ORPHAN'


class TestGroupedReadNeverSuppressesContested:
    """PRD V2, the esc-5712 shape: a contested child is NEVER folded away."""

    def test_a_contested_child_stays_as_its_own_hit_beside_the_group(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        plain, _ = _hit('child-a', parent_id=PARENT, kind='amendment', content='PLAIN')
        disputed, _ = _hit('child-b', parent_id=PARENT, kind='amendment',
                           content='DISPUTED')
        records_by_id = {PARENT: parent, 'child-a': plain, 'child-b': disputed}

        grouped = mod.apply_grouped_read(
            [plain, disputed], records_by_id, contested_ids={'child-b'},
        )

        ids = [r.record_id for r in grouped]
        assert PARENT in ids
        assert 'child-b' in ids, 'a contested child was suppressed into the group'
        # It stays VISIBLY itself, not merely counted.
        disputed_hit = next(r for r in grouped if r.record_id == 'child-b')
        assert disputed_hit.content == 'DISPUTED'

    def test_a_contested_childs_body_is_not_folded_into_the_group_document(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        disputed, _ = _hit('child-b', parent_id=PARENT, kind='amendment',
                           content='DISPUTED')
        records_by_id = {PARENT: parent, 'child-b': disputed}

        grouped = mod.apply_grouped_read(
            [disputed], records_by_id, contested_ids={'child-b'},
        )

        group = next((r for r in grouped if r.record_id == PARENT), None)
        if group is not None:
            assert 'DISPUTED' not in group.content

    def test_a_contested_child_is_the_only_survivor_when_it_is_the_sole_hit(self):
        """Even alone, it must not be replaced by its parent's document —
        that would be exactly the silent resolution V2 forbids."""
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        disputed, _ = _hit('child-b', parent_id=PARENT, kind='amendment',
                           content='DISPUTED')

        grouped = mod.apply_grouped_read(
            [disputed], {PARENT: parent, 'child-b': disputed}, contested_ids={'child-b'},
        )

        assert 'child-b' in [r.record_id for r in grouped]
        assert any(r.content == 'DISPUTED' for r in grouped)


class TestGroupedReadIsArmLocalAndPure:
    """The suppression filter must not leak into MemoryService.search.

    PRD V2 forbids it explicitly: at that seam it would break
    `mem0_dedup.find_prior_memories`' post-filter and hide candidates from
    the write guard — the exact failure the grouped read is supposed to
    prevent, relocated one layer down.
    """

    def test_the_transform_itself_never_reaches_for_the_search_seam(self):
        """Scoped to the FUNCTION's own source, not the whole file.

        The module docstring legitimately discusses `MemoryService.search` —
        it has to, since explaining why the transform is arm-local is the
        point. What must stay clean is the transform's body.
        """
        import inspect  # noqa: PLC0415

        source = inspect.getsource(_mod().apply_grouped_read)

        assert 'MemoryService' not in source
        assert 'memory_service' not in source
        # Pure read-side transform: no store, no await, no I/O.
        assert 'await' not in source
        assert 'backend' not in source.lower()

    def test_the_transform_does_not_mutate_its_inputs(self):
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        child, _ = _hit('child-a', parent_id=PARENT, kind='amendment', claim_ids=['k2'])
        records_by_id = {PARENT: parent, 'child-a': child}
        before = (dict(parent.metadata), list(parent.claim_ids), parent.content)

        mod.apply_grouped_read([child], records_by_id, contested_ids=set())

        assert (dict(parent.metadata), list(parent.claim_ids), parent.content) == before
        assert set(records_by_id) == {PARENT, 'child-a'}

    def test_the_transform_is_synchronous_and_deterministic(self):
        import inspect  # noqa: PLC0415

        mod = _mod()
        assert not inspect.iscoroutinefunction(mod.apply_grouped_read)

        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        child, _ = _hit('child-a', parent_id=PARENT, kind='amendment')
        records_by_id = {PARENT: parent, 'child-a': child}

        runs = [
            [r.record_id for r in mod.apply_grouped_read(
                [child], records_by_id, contested_ids=set())]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]


# ===========================================================================
# step-7 — apply_topic_anchor (arm-local reference implementation of 3111)
# ===========================================================================
#
# The topic-anchored pin does not exist in `MemoryService.search` either:
# `memory_service.py` has zero `topic` hits, and 3111 is likewise deferred
# behind gate η. Same reasoning as the grouped read — the bake-off carries
# its own reference implementation, which doubles as 3111's executable spec.
#
# PRD D1: the pin selects `topic == T AND canonical is True` and is
# **ADDITIVE**, never subtractive. That word is load-bearing. A subtractive
# pin — one that dropped non-canonical same-topic hits to make room — would
# hide exactly the candidates `mem0_dedup.find_prior_memories` needs to see,
# turning a discoverability feature into a write-guard regression.
#
# Because the pin is a READ-side transform, arm (d) (each shape ± pin) needs
# no extra seeded collection: pin-on vs pin-off is an exactly-controlled A/B
# over identical stored state, which is the only way it answers its actual
# question ("whether 3111's pin is needed under each shape").


def _anchor_hit(record_id, *, topic=None, canonical=None, content='body'):
    """A hit for the pin tests. `canonical` is passed through VERBATIM so a
    truthy-but-not-True value can be exercised."""
    metadata: dict = {'category': 'procedural_knowledge'}
    if topic is not None:
        metadata['topic'] = topic
    if canonical is not None:
        metadata['canonical'] = canonical
    return _mod().ArmRecord(
        record_id=record_id,
        content=content,
        metadata=metadata,
        cluster_id='c1',
        claim_ids=[],
        role='peer',
    )


class TestTopicAnchorIsAdditive:
    """PRD D1: the pin ADDS the topic's canonical; it never removes a hit."""

    def test_canonical_is_pinned_in_when_absent_from_the_raw_ranking(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)
        member = _anchor_hit('peer-1', topic='alpha')
        other = _anchor_hit('noise-1')

        pinned = mod.apply_topic_anchor(
            [member, other], canonical_by_topic={'alpha': canonical},
        )

        ids = [r.record_id for r in pinned]
        assert 'canon-a' in ids
        # Additive: nothing was displaced to make room.
        assert 'peer-1' in ids and 'noise-1' in ids
        assert len(pinned) == 3

    def test_the_relative_order_of_the_original_hits_is_unchanged(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)
        hits = [
            _anchor_hit('peer-1', topic='alpha'),
            _anchor_hit('noise-1'),
            _anchor_hit('peer-2', topic='alpha'),
        ]

        pinned = mod.apply_topic_anchor(hits, canonical_by_topic={'alpha': canonical})

        original_order = [r.record_id for r in pinned if r.record_id != 'canon-a']
        assert original_order == ['peer-1', 'noise-1', 'peer-2']

    def test_an_already_present_canonical_is_not_duplicated(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)

        pinned = mod.apply_topic_anchor(
            [_anchor_hit('peer-1', topic='alpha'), canonical],
            canonical_by_topic={'alpha': canonical},
        )

        assert [r.record_id for r in pinned].count('canon-a') == 1
        assert len(pinned) == 2

    def test_two_topics_in_one_hit_set_both_get_their_canonical_pinned(self):
        mod = _mod()
        alpha_canon = _anchor_hit('canon-a', topic='alpha', canonical=True)
        beta_canon = _anchor_hit('canon-b', topic='beta', canonical=True)

        pinned = mod.apply_topic_anchor(
            [_anchor_hit('peer-1', topic='alpha'), _anchor_hit('peer-2', topic='beta')],
            canonical_by_topic={'alpha': alpha_canon, 'beta': beta_canon},
        )

        ids = [r.record_id for r in pinned]
        assert 'canon-a' in ids and 'canon-b' in ids
        assert len(pinned) == 4


class TestTopicAnchorIdentityCases:
    """The pin fires only when a same-topic hit is actually present."""

    def test_a_hit_set_with_no_topic_metadata_is_returned_unchanged(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)
        hits = [_anchor_hit('noise-1'), _anchor_hit('noise-2')]

        pinned = mod.apply_topic_anchor(hits, canonical_by_topic={'alpha': canonical})

        assert pinned == hits  # identity — the pin is not a rewrite

    def test_a_topic_with_no_registered_canonical_pins_nothing(self):
        mod = _mod()

        hits = [_anchor_hit('peer-1', topic='orphan-topic')]
        pinned = mod.apply_topic_anchor(hits, canonical_by_topic={})

        assert pinned == hits

    def test_an_empty_hit_list_stays_empty(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)

        assert mod.apply_topic_anchor([], canonical_by_topic={'alpha': canonical}) == []


class TestTopicAnchorCanonicalIsBoolIdentity:
    """`canonical` is matched by `is True`, mirroring β's rule.

    β's `invalid_canonical_type` treats a truthy `1` as a FATAL violation, so
    a pin that accepted it would anchor on records the write boundary would
    have rejected — and E2 would report a discoverability win for a shape
    production cannot store.
    """

    def test_a_truthy_one_is_not_accepted_as_canonical(self):
        mod = _mod()
        impostor = _anchor_hit('impostor', topic='alpha', canonical=1)

        with pytest.raises(ValueError, match='canonical'):
            mod.apply_topic_anchor(
                [_anchor_hit('peer-1', topic='alpha')],
                canonical_by_topic={'alpha': impostor},
            )

    def test_a_real_bool_true_is_accepted(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)

        pinned = mod.apply_topic_anchor(
            [_anchor_hit('peer-1', topic='alpha')],
            canonical_by_topic={'alpha': canonical},
        )

        assert 'canon-a' in [r.record_id for r in pinned]


class TestBuildCanonicalByTopic:
    """The index the live driver populates via `scroll_by_metadata`.

    `Mem0Backend.search` exposes NO arbitrary metadata filter, so the pin's
    canonical lookup cannot be a search — it has to be a scroll. Building the
    index as a pure function over already-fetched records keeps that seam
    testable without a store.
    """

    def test_indexes_exactly_the_canonical_records_by_topic(self):
        mod = _mod()
        records = list(_arm('c_peers'))

        index = mod.build_canonical_by_topic(records)

        assert len(index) == 20
        for topic, record in index.items():
            assert record.metadata['topic'] == topic
            assert record.metadata['canonical'] is True

    def test_a_second_canonical_for_one_topic_is_rejected_by_name(self):
        """Per-(project, topic) canonical uniqueness is leaf ε's rule.

        This transform cannot enforce it globally, but it must not silently
        pick a winner — that would make the pin's answer depend on scroll
        order, i.e. non-deterministic between runs.
        """
        mod = _mod()
        first = _anchor_hit('canon-a', topic='alpha', canonical=True)
        second = _anchor_hit('canon-b', topic='alpha', canonical=True)

        with pytest.raises(ValueError, match='alpha'):
            mod.build_canonical_by_topic([first, second])

    def test_records_without_a_topic_or_canonical_flag_are_ignored(self):
        mod = _mod()
        index = mod.build_canonical_by_topic([
            _anchor_hit('noise-1'),
            _anchor_hit('peer-1', topic='alpha'),
            _anchor_hit('untopicked-canon', canonical=True),
        ])

        assert index == {}

    def test_the_index_is_deterministic_across_input_orderings(self):
        mod = _mod()
        records = list(_arm('c_peers'))

        forward = mod.build_canonical_by_topic(records)
        backward = mod.build_canonical_by_topic(list(reversed(records)))

        assert {t: r.record_id for t, r in forward.items()} == {
            t: r.record_id for t, r in backward.items()
        }


class TestTopicAnchorIsArmLocalAndPure:
    """Same discipline as the grouped read — 3111's pin stays out of the seam."""

    def test_the_transform_itself_never_reaches_for_the_search_seam(self):
        import inspect  # noqa: PLC0415

        source = inspect.getsource(_mod().apply_topic_anchor)

        assert 'MemoryService' not in source
        assert 'memory_service' not in source
        assert 'await' not in source

    def test_the_transform_does_not_mutate_its_inputs(self):
        mod = _mod()
        canonical = _anchor_hit('canon-a', topic='alpha', canonical=True)
        hits = [_anchor_hit('peer-1', topic='alpha')]
        before = [(r.record_id, dict(r.metadata)) for r in hits]

        mod.apply_topic_anchor(hits, canonical_by_topic={'alpha': canonical})

        assert [(r.record_id, dict(r.metadata)) for r in hits] == before
        assert len(hits) == 1  # not appended to in place


# ===========================================================================
# step-9 — the two rank/set-based retrieval metrics
# ===========================================================================
#
# eval-design §1 states the program-wide discipline verbatim: "every
# retrieval metric in this program must be rank-based, never
# absolute-score-based. Re-running 3111's probe today on the canonical's own
# topic phrase returned scores of 0.44-0.51 for the same corpus where the
# task record measured 0.72-0.90 — wording and embedding/config drift move
# the score scale wholesale. Ranks and set-membership (present-in-top-k)
# survive that; thresholds on raw cosine do not."
#
# So these two functions take an ALREADY-RANKED hit list and read only rank
# and set membership. Every hit list below is hand-built, so every expected
# value is exact by construction — no embeddings, no tolerances.


def _ranked(*claim_id_groups, topic=None):
    """A ranked hit list where hit i realizes `claim_id_groups[i]`."""
    mod = _mod()
    hits = []
    for position, claim_ids in enumerate(claim_id_groups):
        metadata: dict = {'category': 'procedural_knowledge'}
        if topic is not None:
            metadata['topic'] = topic
        hits.append(mod.ArmRecord(
            record_id=f'r{position}',
            content='body',
            metadata=metadata,
            cluster_id='c1',
            claim_ids=list(claim_ids),
            role='peer',
        ))
    return hits


class TestClaimRecallAtK:
    """Does the SPECIFIC claim a query targets surface within top-k?"""

    def test_a_claim_at_rank_three_is_recalled_at_k5_but_not_k2(self):
        mod = _mod()
        hits = _ranked([], [], ['target'], [])

        assert mod.claim_recall_at_k(hits, ['target'], 5) == 1.0
        assert mod.claim_recall_at_k(hits, ['target'], 2) == 0.0

    def test_a_claim_exactly_at_k_is_inside_the_window(self):
        """Off-by-one guard: k is inclusive, so rank 5 counts at k=5."""
        mod = _mod()
        hits = _ranked([], [], [], [], ['target'])

        assert mod.claim_recall_at_k(hits, ['target'], 5) == 1.0
        assert mod.claim_recall_at_k(hits, ['target'], 4) == 0.0

    def test_a_grouped_document_realizes_every_claim_it_absorbed(self):
        """Otherwise arm (b) is penalised precisely FOR grouping correctly —
        the metric would punish the shape for doing the thing under test."""
        mod = _mod()
        grouped = _ranked(['k1', 'k2', 'k3'])

        for claim_id in ('k1', 'k2', 'k3'):
            assert mod.claim_recall_at_k(grouped, [claim_id], 5) == 1.0

    def test_recall_over_several_expected_claims_is_the_realized_fraction(self):
        mod = _mod()
        hits = _ranked(['k1'], ['k2'])

        assert mod.claim_recall_at_k(hits, ['k1', 'k2'], 5) == 1.0
        assert mod.claim_recall_at_k(hits, ['k1', 'k2', 'k3'], 5) == pytest.approx(2 / 3)
        assert mod.claim_recall_at_k(hits, ['k3', 'k4'], 5) == 0.0

    def test_a_claim_realized_twice_is_not_double_counted(self):
        mod = _mod()
        hits = _ranked(['k1'], ['k1'], ['k2'])

        assert mod.claim_recall_at_k(hits, ['k1', 'k2'], 5) == 1.0

    def test_no_expected_claims_reports_none_not_a_measured_zero(self):
        """`calibrate_write_triage.compute_recall_at_k`'s own rule: an empty
        denominator is NO measurement, which is not the same as a measured 0.0
        and must not average into the report as one."""
        assert _mod().claim_recall_at_k(_ranked(['k1']), [], 5) is None

    def test_an_empty_hit_list_is_a_genuine_zero(self):
        """Distinct from the case above: the query WAS scorable and returned
        nothing, which is a real miss."""
        assert _mod().claim_recall_at_k([], ['k1'], 5) == 0.0


class TestTopicDiscoverability:
    """Can the topic's canonical be found, and how much of the topic surfaced?"""

    def test_reports_a_one_based_rank_for_a_present_canonical(self):
        mod = _mod()
        hits = _ranked([], [], [], topic='alpha')

        result = mod.topic_discoverability(hits, 'alpha', 'r1', 5)

        assert result['canonical_in_top_k'] is True
        assert result['canonical_rank'] == 2  # r1 is the SECOND hit

    def test_the_first_hit_is_rank_one_not_rank_zero(self):
        result = _mod().topic_discoverability(
            _ranked([], topic='alpha'), 'alpha', 'r0', 5,
        )

        assert result['canonical_rank'] == 1

    def test_an_absent_canonical_reports_rank_none_never_zero(self):
        """0 would collide with a real rank under any 0-based reading and
        would silently average as "very good" in a mean-rank summary."""
        mod = _mod()
        hits = _ranked([], [], topic='alpha')

        result = mod.topic_discoverability(hits, 'alpha', 'missing', 5)

        assert result['canonical_in_top_k'] is False
        assert result['canonical_rank'] is None

    def test_a_canonical_ranked_beyond_k_is_out_of_the_window(self):
        mod = _mod()
        hits = _ranked([], [], [], [], [], [], topic='alpha')

        result = mod.topic_discoverability(hits, 'alpha', 'r5', 5)

        assert result['canonical_in_top_k'] is False
        # Its true rank is still reported — "absent from top-5" and "absent
        # entirely" are different findings and the report must tell them apart.
        assert result['canonical_rank'] == 6

    def test_member_count_counts_only_records_carrying_the_asked_topic(self):
        mod = _mod()
        alpha_hits = _ranked([], [], topic='alpha')
        beta_hit = _ranked([], topic='beta')[0]
        beta_hit = beta_hit.__class__(**{**vars(beta_hit), 'record_id': 'beta-0'})

        result = mod.topic_discoverability([*alpha_hits, beta_hit], 'alpha', 'r0', 5)

        assert result['topic_member_count'] == 2  # the beta hit is not a member

    def test_member_count_respects_the_k_window(self):
        mod = _mod()
        hits = _ranked([], [], [], [], [], [], topic='alpha')

        assert mod.topic_discoverability(hits, 'alpha', 'r0', 5)['topic_member_count'] == 5
        assert mod.topic_discoverability(hits, 'alpha', 'r0', 2)['topic_member_count'] == 2

    def test_an_empty_hit_list_reports_zeroes_and_no_rank(self):
        result = _mod().topic_discoverability([], 'alpha', 'r0', 5)

        assert result == {
            'canonical_in_top_k': False,
            'canonical_rank': None,
            'topic_member_count': 0,
        }


class TestMetricsAreRankBasedNotScoreBased:
    """eval-design §1's discipline, enforced by the metric's own signature.

    This is asserted structurally rather than trusted to convention, because
    the failure mode is invisible: a score-reading metric produces perfectly
    plausible numbers that silently stop being comparable the moment the
    embedding config drifts — which is exactly how the 0.72-0.90 figure in
    the task record became 0.44-0.51 on re-measurement.
    """

    def test_neither_metric_reads_a_score_field(self):
        import inspect  # noqa: PLC0415

        mod = _mod()
        for function in (mod.claim_recall_at_k, mod.topic_discoverability):
            source = inspect.getsource(function)
            assert 'relevance_score' not in source
            assert '.score' not in source
            assert 'threshold' not in source

    def test_arm_records_carry_no_score_field_at_all(self):
        """The strongest form: the metrics COULD NOT read a score if they
        tried, because the type they consume does not have one."""
        import dataclasses  # noqa: PLC0415

        fields = {f.name for f in dataclasses.fields(_mod().ArmRecord)}

        assert 'score' not in fields
        assert 'relevance_score' not in fields

    def test_shuffling_the_payloads_without_changing_order_changes_nothing(self):
        """Rank is the only input that matters: rewriting every content body
        must not move a rank-based metric."""
        mod = _mod()
        hits = _ranked(['k1'], ['k2'], ['k3'], topic='alpha')
        rewritten = [
            r.__class__(**{**vars(r), 'content': f'COMPLETELY DIFFERENT {i}'})
            for i, r in enumerate(hits)
        ]

        assert mod.claim_recall_at_k(hits, ['k2'], 5) == mod.claim_recall_at_k(
            rewritten, ['k2'], 5
        )
        assert mod.topic_discoverability(hits, 'alpha', 'r1', 5) == (
            mod.topic_discoverability(rewritten, 'alpha', 'r1', 5)
        )


# ===========================================================================
# step-11 — the D4 cost metric: tokens returned per query
# ===========================================================================
#
# The question D4 asks is a COST question: "a grouped read returns one long
# document; N peers return N short ones — which costs the reader more?"  So
# the metric sums the payloads a query actually returns within top-k, which
# puts one long document and N short ones on the same footing (a hit-count
# metric would answer a different, useless question).
#
# NOTHING BELOW PINS AN ABSOLUTE TOKEN NUMBER, and nothing pins a
# proxy-vs-tiktoken ratio.  The metric is COMPARATIVE across arms: the report
# reads "arm (a) costs 3.1x arm (c)", never "arm (a) costs 812 tokens".  An
# absolute assertion here would pin this file to one tokenizer build and
# would be asserting a property of tiktoken, not of the bake-off.  Where an
# exact sum IS asserted it is against an explicitly INJECTED estimator whose
# arithmetic is trivially known (word count), which tests the summation
# without pretending to know what a real tokenizer returns.
#
# tiktoken is NOT installed in this venv, so the character proxy is the live
# path today.  The selection logic is still tested in both directions by
# injecting a fake `tiktoken` into sys.modules — a branch that only runs on
# somebody else's machine is a branch nobody has tested.


def _payload(content: str, *, record_id: str = 'r', **metadata):
    """One ranked hit carrying `content` as its returned payload."""
    return _mod().ArmRecord(
        record_id=record_id,
        content=content,
        metadata={'category': 'procedural_knowledge', **metadata},
        cluster_id='c1',
        claim_ids=[],
        role='peer',
    )


#: An injected estimator with trivially-known arithmetic, so a test can assert
#: an exact SUM without asserting anything about a real tokenizer.
_WORDS = ('injected:words', lambda text: len(text.split()))


def _fake_tiktoken(encode=None, *, get_encoding_raises: Exception | None = None):
    """A stand-in `tiktoken` module for testing the selection branch."""
    module = types.ModuleType('tiktoken')

    class _Encoding:
        def encode(self, text):
            return (encode or (lambda t: t.split()))(text)

    def _get_encoding(name):
        if get_encoding_raises is not None:
            raise get_encoding_raises
        return _Encoding()

    module.get_encoding = _get_encoding  # type: ignore[attr-defined]
    return module


def _available_estimators():
    """Every estimator that can actually run here, resolved by name.

    tiktoken is absent from this venv, so this is normally a one-element list
    — but the shared-invariant tests below must cover it wherever it IS
    installed rather than silently testing half of what they claim to.

    Called from inside test BODIES, never from a `parametrize` decorator:
    the latter would load the script at collection time and break this
    module's documented lazy-`_mod()` discipline.
    """
    mod = _mod()
    estimators = [(mod.CHAR_PROXY_ESTIMATOR_NAME, mod.character_proxy_tokens)]
    try:
        import tiktoken  # noqa: PLC0415

        encoding = tiktoken.get_encoding('cl100k_base')
    except Exception:  # noqa: BLE001 — absent OR unable to fetch its BPE file
        return estimators
    estimators.append(
        (mod.TIKTOKEN_ESTIMATOR_NAME, lambda text: len(encoding.encode(text)))
    )
    return estimators


class TestResolveTokenEstimator:
    """Which tokenizer ran must be a REPORTED FACT, never an assumption."""

    def test_returns_a_name_and_a_callable_pair(self):
        name, encode = _mod().resolve_token_estimator()

        assert isinstance(name, str) and name
        assert callable(encode)
        assert isinstance(encode('some text'), int)

    def test_names_tiktoken_when_tiktoken_is_importable(self, monkeypatch):
        import sys  # noqa: PLC0415

        mod = _mod()
        monkeypatch.setitem(sys.modules, 'tiktoken', _fake_tiktoken())

        name, encode = mod.resolve_token_estimator()

        assert name == mod.TIKTOKEN_ESTIMATOR_NAME == 'tiktoken:cl100k_base'
        # ...and it really delegates to the encoding rather than just renaming
        # the proxy: the fake tokenizes on whitespace, which no character
        # proxy would ever agree with on this input.
        assert encode('a bb ccc dddd eeeee ffffff') == 6

    def test_falls_back_to_the_character_proxy_when_tiktoken_is_absent(
        self, monkeypatch
    ):
        import sys  # noqa: PLC0415

        mod = _mod()
        monkeypatch.setitem(sys.modules, 'tiktoken', None)  # forces ImportError

        name, encode = mod.resolve_token_estimator()

        assert name == mod.CHAR_PROXY_ESTIMATOR_NAME
        assert encode('x' * 40) == mod.character_proxy_tokens('x' * 40)

    def test_the_fallback_name_never_claims_tiktoken(self, monkeypatch):
        """The whole point of returning a name: a substitution that reported
        itself as tiktoken would put un-flagged proxy numbers in the report,
        and nobody reading the artifact could tell."""
        import sys  # noqa: PLC0415

        mod = _mod()
        monkeypatch.setitem(sys.modules, 'tiktoken', None)

        name, _ = mod.resolve_token_estimator()

        assert 'tiktoken' not in name
        assert name != mod.TIKTOKEN_ESTIMATOR_NAME
        # The name must say WHAT it is, not merely that it is not tiktoken.
        assert 'char' in name

    def test_an_unusable_tiktoken_falls_back_instead_of_exploding(
        self, monkeypatch
    ):
        """tiktoken imports fine but fetches its BPE file from the network on
        first `get_encoding`. On an offline box that raises — and a cost
        metric must not take the whole bake-off down with it."""
        import sys  # noqa: PLC0415

        mod = _mod()
        monkeypatch.setitem(
            sys.modules,
            'tiktoken',
            _fake_tiktoken(get_encoding_raises=OSError('no network')),
        )

        name, encode = mod.resolve_token_estimator()

        assert name == mod.CHAR_PROXY_ESTIMATOR_NAME  # honest about what ran
        assert encode('x' * 40) == mod.character_proxy_tokens('x' * 40)

    def test_resolution_is_not_cached_across_calls(self, monkeypatch):
        """A cached resolution would make the report state whichever
        estimator the FIRST caller in the process happened to get."""
        import sys  # noqa: PLC0415

        mod = _mod()
        monkeypatch.setitem(sys.modules, 'tiktoken', _fake_tiktoken())
        assert mod.resolve_token_estimator()[0] == mod.TIKTOKEN_ESTIMATOR_NAME

        monkeypatch.setitem(sys.modules, 'tiktoken', None)
        assert mod.resolve_token_estimator()[0] == mod.CHAR_PROXY_ESTIMATOR_NAME


class TestTokensReturned:
    """Sum over the top-k payloads — one long document vs N short ones."""

    def test_sums_the_payloads_of_the_top_k_hits(self):
        mod = _mod()
        hits = [_payload('a b c', record_id='r0'), _payload('d e', record_id='r1')]

        assert mod.tokens_returned(hits, 5, _WORDS)['tokens'] == 5

    def test_only_the_top_k_payloads_are_counted(self):
        mod = _mod()
        hits = [_payload('a b c', record_id='r0'), _payload('d e', record_id='r1')]

        assert mod.tokens_returned(hits, 1, _WORDS)['tokens'] == 3
        assert mod.tokens_returned(hits, 1, _WORDS)['payloads_counted'] == 1

    def test_one_grouped_document_and_n_peers_are_weighed_on_equal_footing(self):
        """The D4 question itself: identical knowledge, two shapes. The metric
        must be blind to how many hits carried it — otherwise it answers "how
        many results?" rather than "how much must the reader read?"."""
        mod = _mod()
        peers = [
            _payload('claim one body', record_id='p0'),
            _payload('claim two body', record_id='p1'),
            _payload('claim three body', record_id='p2'),
        ]
        grouped = [_payload('claim one body claim two body claim three body')]

        assert (
            mod.tokens_returned(grouped, 5, _WORDS)['tokens']
            == mod.tokens_returned(peers, 5, _WORDS)['tokens']
            == 9
        )

    def test_the_estimator_name_is_carried_into_the_result(self):
        """So the report can state which tokenizer produced its numbers.
        Without this the artifact is un-interpretable a month later."""
        mod = _mod()

        result = mod.tokens_returned([_payload('a b')], 5, _WORDS)

        assert result['estimator'] == 'injected:words'

    def test_the_default_estimator_is_the_resolved_one_and_is_named(self):
        mod = _mod()
        expected_name, expected_encode = mod.resolve_token_estimator()

        result = mod.tokens_returned([_payload('x' * 400)], 5)

        assert result['estimator'] == expected_name
        assert result['tokens'] == expected_encode('x' * 400)

    def test_only_the_knowledge_payload_is_counted_not_the_metadata(self):
        """Metadata rendering is a transport/formatting choice η is NOT
        deciding. Folding it in would make the arm comparison move with how
        the server happens to render a result dict — and would charge the
        shapes that carry β vocabulary keys for carrying them, which is a
        real but SEPARATE question from D4's payload cost."""
        mod = _mod()
        bare = [_payload('a b c')]
        adorned = [
            _payload(
                'a b c',
                topic='a-very-long-topic-slug-indeed',
                canonical=True,
                kind='amendment',
            )
        ]

        assert (
            mod.tokens_returned(adorned, 5, _WORDS)['tokens']
            == mod.tokens_returned(bare, 5, _WORDS)['tokens']
        )

    def test_a_k_larger_than_the_hit_list_counts_what_is_there(self):
        result = _mod().tokens_returned([_payload('a b')], 50, _WORDS)

        assert result['tokens'] == 2
        assert result['payloads_counted'] == 1


class TestTokenEstimatorInvariants:
    """Properties that must hold for EVERY estimator, or arms stop comparing."""

    def test_every_estimator_is_monotone_in_content_length(self):
        """Non-decreasing, not strictly increasing: the proxy floors, so two
        near-identical lengths can legitimately tie. What must NEVER happen is
        a longer payload costing less — that would let the verbose arm win."""
        bodies = ['', 'w', 'word ' * 10, 'word ' * 100, 'word ' * 1000]

        for name, encode in _available_estimators():
            estimates = [encode(body) for body in bodies]

            assert estimates == sorted(estimates), f'{name} is not monotone'
            assert estimates[-1] > estimates[1], f'{name} does not grow at all'

    def test_every_estimator_agrees_that_nothing_returned_costs_nothing(self):
        mod = _mod()

        for name, encode in _available_estimators():
            assert encode('') == 0, f'{name} charges for an empty payload'
            assert mod.tokens_returned([], 5, (name, encode)) == {
                'tokens': 0,
                'estimator': name,
                'payloads_counted': 0,
            }

    def test_every_estimator_is_deterministic(self):
        """A rerun must produce a diffable report, not a moving number."""
        body = 'the quick brown fox jumps over the lazy dog ' * 20

        for name, encode in _available_estimators():
            assert encode(body) == encode(body), f'{name} is not deterministic'

    def test_the_character_proxy_documents_its_derivation(self):
        """A magic divisor nobody can trace is how a proxy silently becomes a
        different proxy. It reuses the repo's ONE chars-per-token constant."""
        import inspect  # noqa: PLC0415

        mod = _mod()
        source = inspect.getsource(mod.character_proxy_tokens)

        assert 'estimate_tokens' in source  # reuses context_assembler's proxy
        assert mod.character_proxy_tokens('x' * 400) > 0


# ===========================================================================
# step-13 — near-dup-guard candidate adequacy
# ===========================================================================
#
# eval-doc :324-326's question: "would the write that became duplicate N+1
# have been matched?"  This is the ONE metric in the program that cannot be
# made purely rank-based without ceasing to measure the real thing —
# `find_near_duplicate_memory` IS an absolute-threshold selector in
# production — so it is reported SPLIT IN TWO rather than quietly violating
# eval-design §1:
#
#   part 1  candidate_present  — rank/set-based and score-free: is a true
#           cluster sibling in the arm's top-5 AT ALL?  Drift-proof, and the
#           part that actually discriminates between storage shapes.
#   part 2  guard_matched      — the production selector's verdict at its
#           configured threshold, carrying `threshold_replay: True` so no
#           reader trends it across embedding-config changes as if it were
#           stable.
#
# The two must be INDEPENDENT: a corpus can put the right sibling in front of
# the guard and still score below threshold. A metric that collapsed them
# would report "the guard would not have fired" for a shape that did its job
# perfectly, and the decision table would blame the shape for the threshold.
#
# Part 2 calls the REAL selector — not a reimplementation. Re-deriving its
# defensive category/source_store filter here would measure THIS file's idea
# of the guard, which is exactly the number nobody wants.


def _scored(record_id, score, *, category='procedural_knowledge', content='body',
            cluster_id='c1'):
    """One ranked hit with the score the store returned for it."""
    mod = _mod()
    return mod.ScoredHit(
        record=mod.ArmRecord(
            record_id=record_id,
            content=content,
            metadata={'category': category},
            cluster_id=cluster_id,
            claim_ids=[],
            role='peer',
        ),
        relevance_score=score,
    )


def _top5(*scores, category='procedural_knowledge'):
    """A five-hit window: `scores[i]` for record id `s{i}`."""
    return [
        _scored(f's{i}', score, category=category) for i, score in enumerate(scores)
    ]


class TestGuardAdequacyPartOneCandidatePresent:
    """Rank/set-based and score-free: is a sibling in front of the guard?"""

    def test_a_sibling_in_the_window_is_present(self):
        result = _mod().guard_adequacy(_top5(0.9, 0.8), {'s1'}, threshold=0.92)

        assert result['candidate_present'] is True

    def test_no_sibling_in_the_window_is_not_present(self):
        result = _mod().guard_adequacy(_top5(0.99, 0.99), {'elsewhere'}, threshold=0.5)

        assert result['candidate_present'] is False
        # ...even though the guard itself matched something. The two parts
        # answer different questions and must not be read as one.
        assert result['guard_matched'] is True

    def test_presence_is_independent_of_score(self):
        """The point of splitting the metric: a shape that put the right
        sibling at rank 1 did its job even if the cosine came in low. Scoring
        that as a shape failure would blame the storage shape for the
        threshold."""
        mod = _mod()

        low = mod.guard_adequacy(_top5(0.50, 0.50, 0.50, 0.50, 0.50), {'s2'}, 0.92)

        assert low['candidate_present'] is True
        assert low['guard_matched'] is False
        assert low['guard_matched_id'] is None

    def test_presence_ignores_the_category_filter_the_selector_applies(self):
        """Part 1 asks about the CORPUS, not about the guard's remit."""
        result = _mod().guard_adequacy(
            _top5(0.99, category='observations_and_summaries'), {'s0'}, 0.92,
        )

        assert result['candidate_present'] is True
        assert result['guard_matched'] is False  # part 2 still filters

    def test_an_empty_window_has_no_candidate(self):
        result = _mod().guard_adequacy([], {'s0'}, 0.92)

        assert result['candidate_present'] is False
        assert result['guard_matched_id'] is None


class TestGuardAdequacyPartTwoThresholdReplay:
    """The production selector's verdict, flagged as a threshold replay."""

    def test_a_high_scoring_sibling_matches_at_the_configured_threshold(self):
        result = _mod().guard_adequacy(
            _top5(0.40, 0.95, 0.30), {'s1'}, threshold=0.92,
        )

        assert result['guard_matched_id'] == 's1'
        assert result['guard_matched'] is True

    def test_the_same_window_at_low_scores_does_not_match(self):
        """Same records, same ranks, same everything but the scores — which is
        precisely what embedding-config drift moves."""
        result = _mod().guard_adequacy(
            _top5(0.50, 0.50, 0.50), {'s1'}, threshold=0.92,
        )

        assert result['guard_matched_id'] is None
        assert result['guard_matched'] is False

    def test_a_high_scoring_result_of_the_wrong_category_does_not_match(self):
        """The real selector defensively filters mismatched categories even at
        a high score, because callers may pass unfiltered results. Production
        guards `procedural_knowledge` writes only."""
        result = _mod().guard_adequacy(
            _top5(0.99, category='observations_and_summaries'), {'s0'}, 0.92,
        )

        assert result['guard_matched_id'] is None

    def test_the_best_scoring_qualifying_candidate_wins(self):
        result = _mod().guard_adequacy(
            _top5(0.93, 0.99, 0.94), {'s0', 's1', 's2'}, 0.92,
        )

        assert result['guard_matched_id'] == 's1'

    def test_the_payload_flags_itself_as_a_threshold_replay(self):
        """Without this flag the number reads as a stable measurement, and
        somebody trends it across an embedder change."""
        result = _mod().guard_adequacy(_top5(0.95), {'s0'}, threshold=0.92)

        assert result['threshold_replay'] is True
        assert result['threshold'] == 0.92

    def test_only_the_top_five_are_replayed(self):
        """Production searches with `limit=5`; a caller handing over ten hits
        must not accidentally measure a more generous guard."""
        window = _top5(0.10, 0.10, 0.10, 0.10, 0.10) + _top5(0.99)

        result = _mod().guard_adequacy(window, {'s0'}, 0.92)

        assert result['guard_matched_id'] is None


class TestGuardAdequacyUsesTheRealSelector:
    """A reimplemented selector would measure this file, not the guard."""

    def test_it_delegates_to_near_duplicate_guard(self, monkeypatch):
        from fused_memory.server import near_duplicate_guard  # noqa: PLC0415

        seen = {}

        def _spy(results, threshold, **kwargs):
            seen['results'] = results
            seen['threshold'] = threshold
            seen['kwargs'] = kwargs
            return None

        monkeypatch.setattr(near_duplicate_guard, 'find_near_duplicate_memory', _spy)

        _mod().guard_adequacy(_top5(0.95), {'s0'}, threshold=0.93)

        assert seen['threshold'] == 0.93
        assert len(seen['results']) == 1

    def test_it_hands_the_selector_real_memory_result_objects(self, monkeypatch):
        """The selector reads `.category` / `.source_store` as ENUMS and
        `.relevance_score`; dicts would raise, and dicts-with-strings would
        silently fail every comparison and report "guard never fires"."""
        from fused_memory.models.enums import MemoryCategory, SourceStore  # noqa: PLC0415
        from fused_memory.models.memory import MemoryResult  # noqa: PLC0415
        from fused_memory.server import near_duplicate_guard  # noqa: PLC0415

        captured = []
        monkeypatch.setattr(
            near_duplicate_guard,
            'find_near_duplicate_memory',
            lambda results, threshold, **kw: captured.extend(results) or None,
        )

        _mod().guard_adequacy(_top5(0.95, 0.10), {'s0'}, 0.92)

        assert all(isinstance(r, MemoryResult) for r in captured)
        assert captured[0].category is MemoryCategory.procedural_knowledge
        assert captured[0].source_store is SourceStore.mem0
        assert captured[0].relevance_score == 0.95
        assert captured[0].id == 's0'

    def test_the_adapter_preserves_rank_order(self):
        """The selector takes the max by score, but the report quotes the
        matched id back against the ranked window — a reordering adapter would
        make that evidence point at the wrong record."""
        mod = _mod()

        results = mod.as_memory_results(_top5(0.10, 0.95, 0.50))

        assert [r.id for r in results] == ['s0', 's1', 's2']

    def test_the_adapter_passes_an_unknown_category_through_as_none(self):
        """Rather than inventing one: a record whose category the enum does
        not know is exactly what the selector's defensive filter exists to
        drop, and a guessed category would defeat it."""
        mod = _mod()
        hit = _scored('s0', 0.99, category='not_a_real_category')

        results = mod.as_memory_results([hit])

        assert results[0].category is None


class TestGuardThresholdIsNotHardcodedTwice:
    """0.92 lives in near_duplicate_guard. A copy here would drift silently."""

    def test_the_default_threshold_comes_from_the_guard_module(self):
        from fused_memory.server.near_duplicate_guard import (  # noqa: PLC0415
            _DEFAULT_NEAR_DUP_THRESHOLD,
        )

        result = _mod().guard_adequacy(_top5(0.95), {'s0'})

        assert result['threshold'] == _DEFAULT_NEAR_DUP_THRESHOLD

    def test_the_script_never_restates_the_threshold_value(self):
        import inspect  # noqa: PLC0415

        source = inspect.getsource(_mod())

        assert '0.92' not in source

    def test_a_configured_threshold_is_read_from_the_memory_service(self):
        """The live driver has a MemoryService; the replay must use ITS
        threshold, or the report describes a guard nobody is running."""
        ns = types.SimpleNamespace
        service = ns(config=ns(reconciliation=ns(
            procedural_knowledge_near_dup_threshold=0.77,
        )))

        assert _mod().resolve_guard_threshold(service) == 0.77

    def test_no_memory_service_falls_back_to_the_module_default(self):
        from fused_memory.server.near_duplicate_guard import (  # noqa: PLC0415
            _DEFAULT_NEAR_DUP_THRESHOLD,
        )

        assert _mod().resolve_guard_threshold(None) == _DEFAULT_NEAR_DUP_THRESHOLD


class TestSelectProbingWrite:
    """Which write IS "the one that became duplicate N+1"?"""

    def test_it_is_the_chronologically_last_duplicate(self):
        mod = _mod()
        cluster = mod.CalibrationCluster(
            cluster_id='c1',
            canonical={'memory_id': 'canon', 'label': 'canonical', 'created_at': None},
            members=[
                {'memory_id': 'early', 'label': 'duplicate',
                 'created_at': '2026-07-16T22:16:18.712577+00:00'},
                {'memory_id': 'late', 'label': 'duplicate',
                 'created_at': '2026-07-26T23:58:00.802949+00:00'},
                {'memory_id': 'mid', 'label': 'duplicate',
                 'created_at': '2026-07-26T13:43:01.911185+00:00'},
            ],
        )

        assert mod.select_probing_write(cluster)['memory_id'] == 'late'

    def test_only_duplicate_labelled_records_are_eligible(self):
        """A `distinct` or `pseudo_contradiction` member is by definition NOT
        the write that became duplicate N+1 — probing with one would measure
        whether the guard fires on content it is supposed to let through."""
        mod = _mod()
        cluster = mod.CalibrationCluster(
            cluster_id='c1',
            canonical={'memory_id': 'canon', 'label': 'canonical', 'created_at': None},
            members=[
                {'memory_id': 'dup', 'label': 'duplicate',
                 'created_at': '2026-07-01T00:00:00+00:00'},
                {'memory_id': 'contra', 'label': 'pseudo_contradiction',
                 'created_at': '2026-07-30T00:00:00+00:00'},
                {'memory_id': 'other', 'label': 'distinct',
                 'created_at': '2026-07-31T00:00:00+00:00'},
            ],
        )

        assert mod.select_probing_write(cluster)['memory_id'] == 'dup'

    def test_a_cluster_with_no_duplicate_has_no_probing_write(self):
        """None, not a substituted canonical: 5 of the committed fixture's 20
        clusters really are duplicate-free, and probing them with SOMETHING
        would manufacture 5 fake measurements."""
        mod = _mod()
        cluster = mod.CalibrationCluster(
            cluster_id='c1',
            canonical={'memory_id': 'canon', 'label': 'canonical', 'created_at': None},
            members=[{'memory_id': 'd', 'label': 'distinct', 'created_at': 'x'}],
        )

        assert mod.select_probing_write(cluster) is None

    def test_equal_timestamps_break_deterministically_on_memory_id(self):
        """The 20 canonicals carry `created_at: null` and nothing forbids two
        duplicates sharing a stamp — an unstable tiebreak would make the probe
        query, and therefore the whole arm's guard column, move between runs."""
        mod = _mod()
        same = '2026-07-26T13:43:01.911185+00:00'
        cluster = mod.CalibrationCluster(
            cluster_id='c1',
            canonical={'memory_id': 'canon', 'label': 'canonical', 'created_at': None},
            members=[
                {'memory_id': 'bbb', 'label': 'duplicate', 'created_at': same},
                {'memory_id': 'aaa', 'label': 'duplicate', 'created_at': same},
            ],
        )

        assert mod.select_probing_write(cluster)['memory_id'] == 'bbb'  # max id

    def test_over_the_committed_fixture_every_probe_is_a_duplicate(self):
        mod = _mod()
        clusters = mod.load_calibration_clusters(ALPHA_FIXTURE_PATH)

        probes = {
            cid: mod.select_probing_write(c) for cid, c in clusters.items()
        }
        measurable = {cid: p for cid, p in probes.items() if p is not None}

        assert len(probes) == 20
        assert all(p['label'] == 'duplicate' for p in measurable.values())
        # 5 duplicate-free clusters are unmeasurable BY CONSTRUCTION, and the
        # report must show 15 measurements rather than 20 with 5 invented.
        assert len(measurable) == 15
