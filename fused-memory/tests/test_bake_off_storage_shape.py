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

        with pytest.raises(ValueError, match=r':2:'):
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

    def test_no_worktree_path_is_baked_into_the_module(self):
        # The lesson test_calibrate_write_triage.py:1267 pins: a path
        # resolved at author time would break the moment the script runs
        # from another checkout.
        assert '.worktrees' not in str(_mod().DEFAULT_ARM_CLAIMS_PATH)
