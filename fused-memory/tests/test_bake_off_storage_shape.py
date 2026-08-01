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
