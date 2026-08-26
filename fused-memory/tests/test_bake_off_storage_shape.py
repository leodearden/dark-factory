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
from collections.abc import Container
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
ALPHA_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration.jsonl'
REGISTRY_PATH = FIXTURES_DIR / 'memory_eval_topic_registry.json'
ARM_CLAIMS_PATH = FIXTURES_DIR / 'e2_arm_claims.jsonl'
QUERY_SET_PATH = FIXTURES_DIR / 'e2_query_set.jsonl'
DISTRACTOR_SLAB_PATH = FIXTURES_DIR / 'e2_distractor_slab.jsonl'
REGROWTH_INJECTION_PATH = FIXTURES_DIR / 'e2_regrowth_injection.jsonl'


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
        # Committed, not merely named: a default pointing at a path nobody
        # ever added would only surface on a live run.
        assert all(
            path.exists() for path in (
                ARM_CLAIMS_PATH, QUERY_SET_PATH, DISTRACTOR_SLAB_PATH,
                ALPHA_FIXTURE_PATH, REGISTRY_PATH,
            )
        )

    def test_paths_are_derived_from___file___not_baked_in(self, tmp_path):
        """The lesson test_calibrate_write_triage.py:1267 pins.

        A path resolved at AUTHOR time breaks the moment the script runs from
        another checkout.  Asserting `is_relative_to(package_root)` on the
        module as loaded from THIS worktree could not tell the two apart: a
        hardcoded
        `/home/leo/src/dark-factory/.worktrees/3199/fused-memory/tests/
        fixtures/e2_arm_claims.jsonl` is absolute, is under this package root,
        and exists, so it would satisfy every such assertion.

        So relocate the script — copy it to a package root somewhere else
        entirely and import THAT — and require the defaults to follow it.
        Only a `__file__`-derived path can.
        """
        import importlib.util  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import sys  # noqa: PLC0415

        (tmp_path / 'scripts').mkdir()
        relocated = tmp_path / 'scripts' / SCRIPT_PATH.name
        shutil.copy2(SCRIPT_PATH, relocated)

        name = 'relocated_bake_off_storage_shape'
        spec = importlib.util.spec_from_file_location(name, relocated)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # @dataclass looks itself up here
        try:
            spec.loader.exec_module(module)
            defaults = (
                module.DEFAULT_ARM_CLAIMS_PATH,
                module.DEFAULT_QUERY_SET_PATH,
                module.DEFAULT_DISTRACTOR_SLAB_PATH,
                module.DEFAULT_ALPHA_FIXTURE_PATH,
                module.DEFAULT_REGISTRY_PATH,
            )
        finally:
            sys.modules.pop(name, None)

        for default in defaults:
            assert default.is_absolute()
            assert default.is_relative_to(tmp_path), (
                f'{default} did not follow the script to {tmp_path} — it is '
                f'baked in, so the script only works from the checkout it '
                f'was authored in'
            )
        # Nothing was copied alongside it, so a default that "exists" here
        # would mean it is pointing back at the original tree.
        assert not any(default.exists() for default in defaults)


# ===========================================================================
# 4012 step-1 — the regrowth-injection fixture's loader and cross-validation
# ===========================================================================
#
# The +1-re-emission probe's own fixture: exactly one near-duplicate per
# topic, re-emitting that topic's canonical claim.  Contract paths read the
# COMMITTED fixture; error paths build tiny synthetic JSONL in tmp_path.
# Pure — no network, no Qdrant, no key.


def _injection_row(**overrides) -> dict:
    """One well-formed injection row, before whatever the test breaks."""
    row = {
        'injection_id': 'topic-a-regrowth-01',
        'topic': 'topic-a',
        'cluster_id': 'cluster-a',
        'reemits_claim_id': 'topic-a-01',
        'text': 'a restatement of the canonical claim in different words',
    }
    row.update(overrides)
    return row


def _write_injections(path: Path, rows: list[dict]) -> Path:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def _synthetic_claims(mod, *rows: dict) -> list:
    """Hand-built `ArmClaim`s, so cross-validation has an exact expectation."""
    return [
        mod.ArmClaim(
            claim_id=r['claim_id'],
            cluster_id=r['cluster_id'],
            topic=r['topic'],
            text=r.get('text', 'body'),
            source_memory_id=r.get('source_memory_id', 'm'),
            canonical=bool(r.get('canonical', False)),
            b_arm_role=r.get('b_arm_role', 'canonical' if r.get('canonical') else 'sighting'),
            contested=bool(r.get('contested', False)),
        )
        for r in rows
    ]


class TestLoadRegrowthInjections:
    """`load_regrowth_injections` parses the probe's own fixture strictly."""

    def test_the_committed_fixture_loads_one_injection_per_topic(self):
        injections = _mod().load_regrowth_injections(REGROWTH_INJECTION_PATH)

        assert len(injections) == 20
        assert len({i.injection_id for i in injections}) == 20
        for injection in injections:
            assert injection.injection_id
            assert injection.topic
            assert injection.cluster_id
            assert injection.reemits_claim_id
            assert injection.text

    def test_the_default_path_is_package_relative_and_committed(self):
        mod = _mod()

        assert mod.DEFAULT_REGROWTH_INJECTION_PATH == REGROWTH_INJECTION_PATH
        assert mod.DEFAULT_REGROWTH_INJECTION_PATH.exists()

    def test_the_default_path_follows_a_relocated_script(self, tmp_path):
        """Package-relative, not resolved against the checkout cwd.

        The same lesson `TestDefaultFixturePaths` pins for the five original
        fixtures: a path baked in at author time works only from the tree it
        was written in.
        """
        import importlib.util  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import sys  # noqa: PLC0415

        (tmp_path / 'scripts').mkdir()
        relocated = tmp_path / 'scripts' / SCRIPT_PATH.name
        shutil.copy2(SCRIPT_PATH, relocated)

        name = 'relocated_regrowth_bake_off'
        spec = importlib.util.spec_from_file_location(name, relocated)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            default = module.DEFAULT_REGROWTH_INJECTION_PATH
        finally:
            sys.modules.pop(name, None)

        assert default.is_absolute()
        assert default.is_relative_to(tmp_path)
        assert not default.exists()

    def test_a_duplicate_injection_id_names_the_path_line_and_id(self, tmp_path):
        path = _write_injections(tmp_path / 'inj.jsonl', [
            _injection_row(),
            _injection_row(topic='topic-b', cluster_id='cluster-b'),
        ])

        with pytest.raises(_mod().FixtureError) as excinfo:
            _mod().load_regrowth_injections(path)

        message = str(excinfo.value)
        assert 'inj.jsonl' in message
        assert ':2:' in message
        assert 'topic-a-regrowth-01' in message

    @pytest.mark.parametrize('field', ['topic', 'cluster_id', 'reemits_claim_id', 'text'])
    def test_a_missing_field_names_the_path_line_and_field(self, tmp_path, field):
        row = _injection_row()
        row.pop(field)
        path = _write_injections(tmp_path / 'inj.jsonl', [row])

        with pytest.raises(_mod().FixtureError) as excinfo:
            _mod().load_regrowth_injections(path)

        message = str(excinfo.value)
        assert 'inj.jsonl' in message
        assert ':1:' in message
        assert field in message

    def test_a_missing_injection_id_is_reported_by_name(self, tmp_path):
        row = _injection_row()
        row.pop('injection_id')
        path = _write_injections(tmp_path / 'inj.jsonl', [row])

        with pytest.raises(_mod().FixtureError) as excinfo:
            _mod().load_regrowth_injections(path)

        assert 'injection_id' in str(excinfo.value)


class TestCrossValidateRegrowthInjections:
    """The injections must AGREE with the claims fixture, not merely parse."""

    def test_the_committed_fixtures_cross_validate(self):
        mod = _mod()
        claims = mod.load_arm_claims(ARM_CLAIMS_PATH)
        injections = mod.load_regrowth_injections(REGROWTH_INJECTION_PATH)

        mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        per_topic: dict[str, int] = {}
        for injection in injections:
            per_topic[injection.topic] = per_topic.get(injection.topic, 0) + 1
        assert set(per_topic) == {c.topic for c in claims}
        assert set(per_topic.values()) == {1}, (
            'the "+1" in the probe name is exactly one re-emission per topic'
        )

    def test_a_topic_with_no_injection_is_named(self):
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
            {'claim_id': 'topic-b-01', 'cluster_id': 'cluster-b', 'topic': 'topic-b',
             'canonical': True},
        )
        injections = [mod.RegrowthInjection(**_injection_row())]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        message = str(excinfo.value)
        assert 'topic-b' in message
        assert _mod()._FIXTURE_DOCS in message

    def test_two_injections_on_one_topic_are_a_fixture_defect(self):
        """The quantity IS the experiment: 2 is a defect, not a bigger probe."""
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
        )
        injections = [
            mod.RegrowthInjection(**_injection_row()),
            mod.RegrowthInjection(**_injection_row(injection_id='topic-a-regrowth-02')),
        ]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        message = str(excinfo.value)
        assert 'topic-a' in message
        assert '2' in message

    def test_a_cluster_id_disagreeing_with_the_claims_fixture_is_named(self):
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
        )
        injections = [mod.RegrowthInjection(**_injection_row(cluster_id='cluster-wrong'))]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        message = str(excinfo.value)
        assert 'cluster-wrong' in message
        assert 'cluster-a' in message

    def test_an_unknown_topic_is_named(self):
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
        )
        injections = [
            mod.RegrowthInjection(**_injection_row()),
            mod.RegrowthInjection(**_injection_row(
                injection_id='ghost-regrowth-01', topic='ghost',
                cluster_id='cluster-ghost', reemits_claim_id='ghost-01')),
        ]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        assert 'ghost' in str(excinfo.value)

    def test_a_reemitted_claim_that_does_not_exist_is_named(self):
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
        )
        injections = [mod.RegrowthInjection(**_injection_row(reemits_claim_id='topic-a-99'))]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        assert 'topic-a-99' in str(excinfo.value)

    def test_reemitting_a_non_canonical_claim_is_named(self):
        """The re-emission must name the TRUE canonical, never a peer.

        `claim_ids = [reemits_claim_id]` is what credits the injection with
        realizing the claim it restates; pointing it at a non-canonical peer
        would make the probe measure something other than regrowth of the
        canonical.
        """
        mod = _mod()
        claims = _synthetic_claims(
            mod,
            {'claim_id': 'topic-a-01', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': True},
            {'claim_id': 'topic-a-02', 'cluster_id': 'cluster-a', 'topic': 'topic-a',
             'canonical': False},
        )
        injections = [mod.RegrowthInjection(**_injection_row(reemits_claim_id='topic-a-02'))]

        with pytest.raises(mod.FixtureError) as excinfo:
            mod.cross_validate_regrowth_injections(injections=injections, claims=claims)

        message = str(excinfo.value)
        assert 'topic-a-02' in message
        assert 'canonical' in message

    def test_no_injected_body_is_byte_identical_to_the_claim_it_reemits(self):
        """A copy would make the probe measure deduplication, not regrowth.

        Inequality ONLY.  No similarity threshold is asserted anywhere — a
        bound on how alike a re-emission may be would be a guess dressed as
        a finding (gate G6); the fixture README reports the ratios instead.
        """
        mod = _mod()
        claims = {c.claim_id: c for c in mod.load_arm_claims(ARM_CLAIMS_PATH)}

        for injection in mod.load_regrowth_injections(REGROWTH_INJECTION_PATH):
            assert injection.text != claims[injection.reemits_claim_id].text


# ===========================================================================
# 4012 step-3 — the injected corpus, per mode, and the TRUE-canonical invariant
# ===========================================================================
#
# The probe materialises the ratified `c_peers` write shape twice more, once
# per injection mode, with exactly one extra near-duplicate per topic.  The
# invariant tests below are the task's "scored by TRUE canonical id per
# 3560's correction, never by an aliased record id" requirement made
# mechanical.  Pure — built over the committed fixtures, no embedding.


@functools.cache
def _injections() -> tuple:
    return tuple(_mod().load_regrowth_injections(REGROWTH_INJECTION_PATH))


@functools.cache
def _regrowth_records(mode: str) -> tuple:
    mod = _mod()
    return tuple(mod.materialize_regrowth_injections(
        list(_injections()),
        _committed_inputs()['claims'],
        _committed_inputs()['clusters'],
        mode,
    ))


@functools.cache
def _regrowth_arm(mode: str) -> tuple:
    """`c_peers` + the injections for `mode`, in corpus order."""
    mod = _mod()
    return tuple(mod.regrowth_corpus(
        list(_arm('c_peers')),
        list(_injections()),
        _committed_inputs()['claims'],
        _committed_inputs()['clusters'],
        mode=mode,
    ))


def _indexed(records):
    """Returns the module's `SeededArm`; unannotated so its attributes stay
    reachable — the dynamic `_mod()` import makes the type `Any`, and naming
    it `object` here instead hid every field behind an attribute error."""
    mod = _mod()
    return mod._index_arm(
        'c_peers', 'p', 'c', list(records), _committed_inputs()['claims'],
    )


class TestRegrowthModesArePinned:

    def test_modes_are_pinned_by_equality_unstamped_first(self):
        """Pinned, not derived — the convention `ARM_SHAPES`/`QUERY_KINDS` set.

        `unstamped` leads because it is the case that models reality today:
        esc-3200-3 measured exactly one topic-stamped record for the topic
        whose re-emissions it was reading, so every organic re-emission
        arrived with no topic key.  A reader who stops after the first row
        has then read the real-world case, not the best case.
        """
        assert _mod().REGROWTH_MODES == ('unstamped', 'stamped')

    def test_the_regrowth_role_is_distinct_from_every_other_role(self):
        mod = _mod()

        assert mod.REGROWTH_ROLE == 'regrowth'
        assert mod.REGROWTH_ROLE not in (mod.DISTRACTOR_ROLE, mod.GROUPED_ROLE)


class TestMaterializeRegrowthInjections:
    """One `ArmRecord` per injection, carrying exactly what a re-emission would."""

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_one_record_per_injection_in_fixture_order(self, mode):
        records = _regrowth_records(mode)
        injections = _injections()

        assert len(records) == len(injections)
        for record, injection in zip(records, injections, strict=True):
            assert record.content == injection.text

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_record_ids_are_derived_dashed_uuid5s_distinct_from_the_base_arm(self, mode):
        mod = _mod()
        base_ids = {r.record_id for r in _arm('c_peers')}

        for record, injection in zip(_regrowth_records(mode), _injections(), strict=True):
            expected = mod._derive_record_id(f'regrowth:{mode}', injection.injection_id)
            assert record.record_id == expected
            assert len(record.record_id) == 36 and record.record_id.count('-') == 4
            assert record.record_id not in base_ids

    def test_the_two_modes_derive_disjoint_record_ids(self):
        unstamped = {r.record_id for r in _regrowth_records('unstamped')}
        stamped = {r.record_id for r in _regrowth_records('stamped')}

        assert not (unstamped & stamped)

    def test_unstamped_carries_no_topic_key_at_all(self):
        """`'topic' not in metadata` — NOT `metadata.get('topic') is None`.

        A present-but-None topic key is a different write than no key, and
        the pin's firing rule reads presence.  The unstamped mode models a
        re-emission that never carried the vocabulary at all.
        """
        for record in _regrowth_records('unstamped'):
            assert 'topic' not in record.metadata

    def test_stamped_carries_its_injections_topic(self):
        for record, injection in zip(_regrowth_records('stamped'), _injections(), strict=True):
            assert record.metadata['topic'] == injection.topic

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    @pytest.mark.parametrize('key', ['canonical', 'parent_id', 'contested', 'kind'])
    def test_neither_mode_writes_a_key_a_reemission_would_not_carry(self, mode, key):
        for record in _regrowth_records(mode):
            assert key not in record.metadata

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_category_matches_the_reemitted_claims_own_category(self, mode):
        mod = _mod()
        categories = mod._claim_categories(
            _committed_inputs()['clusters'], _committed_inputs()['claims'],
        )

        for record, injection in zip(_regrowth_records(mode), _injections(), strict=True):
            assert record.metadata['category'] == categories[injection.reemits_claim_id]

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_bookkeeping_credits_the_reemitted_claim_and_its_cluster(self, mode):
        for record, injection in zip(_regrowth_records(mode), _injections(), strict=True):
            assert record.claim_ids == [injection.reemits_claim_id]
            assert record.cluster_id == injection.cluster_id
            assert record.role == _mod().REGROWTH_ROLE

    def test_an_unknown_mode_names_the_modes_it_knows(self):
        mod = _mod()

        with pytest.raises(ValueError, match='REGROWTH_MODES|unstamped') as excinfo:
            mod.materialize_regrowth_injections(
                list(_injections()),
                _committed_inputs()['claims'],
                _committed_inputs()['clusters'],
                'topic_stamped_and_pinned',
            )

        assert 'topic_stamped_and_pinned' in str(excinfo.value)


class TestRegrowthCorpus:
    """The injections go LAST, and the base list is never mutated."""

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_base_records_come_first_in_order_and_injections_are_appended(self, mode):
        base = list(_arm('c_peers'))
        corpus = list(_regrowth_arm(mode))

        assert corpus[:len(base)] == base
        assert corpus[len(base):] == list(_regrowth_records(mode))
        assert len(corpus) == len(base) + len(_injections())

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_the_base_list_is_not_mutated(self, mode):
        mod = _mod()
        base = list(_arm('c_peers'))
        before = len(base)

        mod.regrowth_corpus(
            base, list(_injections()), _committed_inputs()['claims'],
            _committed_inputs()['clusters'], mode=mode,
        )

        assert len(base) == before


class TestRegrowthPreservesTheTrueCanonical:
    """The invariant the whole probe's credibility rests on.

    `canonical_record_ids` is FIRST-MATCH-WINS over the record list, and the
    injection deliberately carries the canonical's claim id.  Prepending it
    would silently rename each cluster's canonical to the re-emission, and
    every discoverability number in the block would then be scored against
    the duplicate — the aliasing failure 3560 had to disclose after the fact
    for `b_grouped`, and the one the task names as the thing to avoid.
    """

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_canonical_by_cluster_is_byte_identical_to_the_uninjected_arm(self, mode):
        base = _indexed(_arm('c_peers'))
        injected = _indexed(_regrowth_arm(mode))

        assert injected.canonical_by_cluster == base.canonical_by_cluster
        injected_ids = {r.record_id for r in _regrowth_records(mode)}
        assert not (set(injected.canonical_by_cluster.values()) & injected_ids)

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_canonical_by_topic_is_byte_identical_and_does_not_raise(self, mode):
        """A stamped injection carrying `canonical: True` would raise here.

        `build_canonical_by_topic` refuses two canonicals on a topic, so this
        pins that the materializer never writes the key.
        """
        base = _indexed(_arm('c_peers'))
        injected = _indexed(_regrowth_arm(mode))

        assert list(injected.canonical_by_topic) == list(base.canonical_by_topic)
        for topic, record in base.canonical_by_topic.items():
            assert injected.canonical_by_topic[topic].record_id == record.record_id

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_contested_ids_are_unchanged(self, mode):
        """An injection must not inherit contested-ness from what it re-emits.

        `contested` has no writer in the live system, so a re-emission that
        acquired it would be modelling something that cannot happen.
        """
        base = _indexed(_arm('c_peers'))
        injected = _indexed(_regrowth_arm(mode))

        assert injected.contested_ids == base.contested_ids

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_records_by_id_gains_exactly_the_injections_and_displaces_nothing(self, mode):
        base = _indexed(_arm('c_peers'))
        injected = _indexed(_regrowth_arm(mode))

        added = set(injected.records_by_id) - set(base.records_by_id)
        assert added == {r.record_id for r in _regrowth_records(mode)}
        assert len(injected.records_by_id) == len(base.records_by_id) + len(_injections())
        for record_id, record in base.records_by_id.items():
            assert injected.records_by_id[record_id] is record

    @pytest.mark.parametrize('mode', ['unstamped', 'stamped'])
    def test_the_injection_joins_its_clusters_sibling_set(self, mode):
        injected = _indexed(_regrowth_arm(mode))

        for record in _regrowth_records(mode):
            assert record.record_id in injected.siblings_by_cluster[record.cluster_id]


class TestRegrowthCorpusFingerprintsAreDistinct:
    """The fetch cache must never replay one pass's rankings as another's."""

    def test_each_injected_corpus_differs_from_the_baseline_and_from_the_other(self):
        mod = _mod()
        base = mod.corpus_fingerprint(list(_arm('c_peers')))
        unstamped = mod.corpus_fingerprint(list(_regrowth_arm('unstamped')))
        stamped = mod.corpus_fingerprint(list(_regrowth_arm('stamped')))

        assert len({base, unstamped, stamped}) == 3


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
        """The `parent_id` shape rule: 36 chars, dashed, own str() round-trip.

        The rule is `fused_memory.utils.validation.is_full_uuid`, enforced by
        β's `validate_memory_metadata`.

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
        # Compared against the OTHER shapes only: including 'status_quo' would
        # compare the reference against itself, a tautology among the three.
        others = {s: r for s, r in slabs.items() if s != 'status_quo'}
        assert others, 'no arm left to compare the slab against'
        for shape, records in others.items():
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
        for shape in _mod().ARM_SHAPES:
            per_cluster: dict[str, set] = {}
            for record in _knowledge(_arm(shape)):
                per_cluster.setdefault(record.cluster_id, set()).add(
                    record.metadata['category']
                )
            mixed = sum(1 for v in per_cluster.values() if len(v) > 1)
            assert mixed == 2, shape


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
        # k3 is the SIGHTING's claim: collapsed to a count, body never
        # rendered, so it is not credited. See the sighting test below.
        assert set(grouped[0].claim_ids) == {'k1', 'k2'}

    def test_a_counted_sightings_claim_is_not_credited_to_the_group(self):
        """Crediting must agree with rendering.

        `_render_grouped_document` collapses sightings to `[sightings: N]` —
        none of their text reaches the reader — while `tokens_returned`
        charges arm (b) only for that count.  Crediting the sighting's claim
        anyway gave arm (b) claim-recall AND the token discount for the same
        content simultaneously, a double advantage in exactly the two columns
        the η decision table is read on.  Material, not theoretical: 34 of the
        176 fixture claims carry `b_arm_role='sighting'` with substantive
        bodies, and 34 of the 236 queries expect one of them.
        """
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        seen, _ = _hit('child-a', parent_id=PARENT, kind='sighting',
                       content='THE SIGHTING BODY', claim_ids=['k-sighting'])
        records_by_id = {PARENT: parent, 'child-a': seen}

        grouped = mod.apply_grouped_read([seen], records_by_id, contested_ids=set())

        assert len(grouped) == 1
        # The body is genuinely absent from the payload ...
        assert 'THE SIGHTING BODY' not in grouped[0].content
        # ... so the claim must not be scored as realized.
        assert set(grouped[0].claim_ids) == {'k1'}

    def test_a_third_kinds_claim_is_still_credited_because_its_body_renders(self):
        """The exclusion is scoped to sightings, not to every non-amendment.

        `others` children ARE pasted into the document verbatim, so their
        claims stay credited — otherwise the fix would swing the bias the
        other way and penalise arm (b) for content it does return.
        """
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON', claim_ids=['k1'])
        other, _ = _hit('child-a', parent_id=PARENT, kind='retraction',
                        content='THE RETRACTION TEXT', claim_ids=['k2'])
        records_by_id = {PARENT: parent, 'child-a': other}

        grouped = mod.apply_grouped_read([other], records_by_id, contested_ids=set())

        assert 'THE RETRACTION TEXT' in grouped[0].content
        assert set(grouped[0].claim_ids) == {'k1', 'k2'}

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

    def test_a_child_of_a_third_kind_is_not_relabelled_an_amendment(self):
        """`apply_grouped_read` buckets members into amendments / sightings /
        `others`, then passed `amendments + others` through the parameter the
        renderer prefixes with the literal `[amendment]`.  Any child carrying
        a kind that is neither amendment nor sighting was therefore rendered
        as an amendment — misattributing what the record actually IS, inside
        the transform whose own docstring calls itself the executable
        specification of PRD D6.

        Latent on the measured path, not theoretical: the committed
        `e2_arm_claims.jsonl` only uses amendment/sighting/canonical, which is
        exactly why no artifact assertion catches it — but the `others` bucket
        exists precisely because the code anticipates other kinds, and a
        grouped read that renames a retraction into an amendment resolves a
        disagreement in the canonical's favour, the esc-5712 shape V2 forbids.
        """
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='THE CANONICAL BODY')
        other, _ = _hit('child-a', parent_id=PARENT, kind='retraction',
                        content='THE RETRACTION TEXT')
        records_by_id = {PARENT: parent, 'child-a': other}

        grouped = mod.apply_grouped_read([other], records_by_id, contested_ids=set())

        document = grouped[0].content
        # D6 still holds: the body resolves upward and stays reachable ...
        assert 'THE RETRACTION TEXT' in document
        # ... but it is not announced as something it is not.
        assert '[amendment]' not in document, (
            f'a retraction was rendered as an amendment: {document!r}'
        )
        assert '[retraction] THE RETRACTION TEXT' in document

    def test_an_amendment_and_a_third_kind_are_labelled_apart_in_one_document(self):
        """The mislabelling is only visible when both kinds share a document:
        a renderer that dropped the prefix entirely would pass the test above
        while making an amendment and a retraction indistinguishable, which is
        the same misattribution in the other direction.
        """
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        amend, _ = _hit('child-a', parent_id=PARENT, kind='amendment',
                        content='AMENDED')
        other, _ = _hit('child-b', parent_id=PARENT, kind='retraction',
                        content='RETRACTED')
        records_by_id = {PARENT: parent, 'child-a': amend, 'child-b': other}

        grouped = mod.apply_grouped_read(
            [amend, other], records_by_id, contested_ids=set()
        )

        document = grouped[0].content
        assert '[amendment] AMENDED' in document
        assert '[retraction] RETRACTED' in document

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

        # Identity of the ELEMENTS, not value equality: `ArmRecord` is a
        # frozen dataclass, so `==` is field-wise and a transform that rebuilt
        # equal-valued copies — allocating on every no-op call, which the hot
        # read path makes per query per arm — would satisfy `==`.
        assert all(g is h for g, h in zip(grouped, hits, strict=True))

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


class TestSuppressionImmunityIsRealInEveryLiveArm:
    """Every V2 test below hands `contested_ids=` in literally, so all of them
    stay green if the LIVE path derives an empty set.

    The live path is `_index_arm` -> `contested_record_ids`, which reads the
    flag through `load_arm_claims`' ``record.get('contested', False)``.  A
    fixture field rename, or a regression in either function, would turn
    suppression-immunity silently OFF in every run — arm (b)'s numbers would
    shift, the artifact gate eta depends on would not say so, and this file
    would still pass.  That is the silent-fail-soft the repo's design
    invariants forbid, so the derivation is asserted against the fixture ON
    DISK rather than against the loader that reads it.
    """

    @staticmethod
    def _raw_claims() -> list[dict]:
        return [
            json.loads(line)
            for line in ARM_CLAIMS_PATH.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def test_the_committed_fixture_still_carries_contested_claims(self):
        """Read from the JSONL, not through the loader whose flag-reading is
        exactly what could regress."""
        contested = [row for row in self._raw_claims() if row.get('contested')]

        assert contested, (
            'no row in e2_arm_claims.jsonl is flagged contested, so every '
            'live arm measures suppression-immunity over an empty set'
        )

    def test_every_arm_derives_a_non_empty_contested_set(self):
        mod = _mod()
        clusters = mod.load_calibration_clusters()
        claims = mod.load_arm_claims()
        topics = mod.load_registry_topics()
        raw_contested = {
            row['claim_id'] for row in self._raw_claims() if row.get('contested')
        }

        for shape in mod.ARM_SHAPES:
            records = mod.materialize_arm(shape, clusters, claims, topics, [])
            seeded = mod._index_arm(shape, 'p', 'c', records, claims)

            assert seeded.contested_ids, (
                f'{shape}: the seeded arm carries no contested record, so the '
                f'grouped read is free to fold every child away'
            )
            # Sound: nothing is marked contested that the fixture on disk does
            # not flag...
            for record_id in seeded.contested_ids:
                realized = set(seeded.records_by_id[record_id].claim_ids)
                assert realized & raw_contested, (
                    f'{shape}/{record_id} is treated as contested but '
                    f'realizes none of the fixture-flagged claims {realized}'
                )
            # ...and complete: every contested claim this arm realizes is
            # covered, so none can be quietly folded.
            for record in records:
                if raw_contested & set(record.claim_ids):
                    assert record.record_id in seeded.contested_ids, (
                        f'{shape}/{record.record_id} realizes a contested '
                        f'claim but is not immune to suppression'
                    )

    def test_every_cluster_keeps_a_canonical_record_in_every_arm(self):
        """`canonical_record_ids` feeds the discoverability column the same
        way, and an empty map would read as "no canonical to discover" rather
        than as a broken derivation."""
        mod = _mod()
        clusters = mod.load_calibration_clusters()
        claims = mod.load_arm_claims()
        topics = mod.load_registry_topics()
        raw_canonical = {
            row['cluster_id']: row['claim_id']
            for row in self._raw_claims() if row.get('canonical')
        }
        assert raw_canonical, 'the fixture flags no canonical claim at all'

        for shape in mod.ARM_SHAPES:
            records = mod.materialize_arm(shape, clusters, claims, topics, [])
            seeded = mod._index_arm(shape, 'p', 'c', records, claims)

            for cluster_id, claim_id in raw_canonical.items():
                record_id = seeded.canonical_by_cluster.get(cluster_id)
                assert record_id is not None, (
                    f'{shape}/{cluster_id}: no canonical record, so this '
                    f'cluster reads as undiscoverable by construction'
                )
                assert claim_id in seeded.records_by_id[record_id].claim_ids, (
                    f'{shape}/{cluster_id}: canonical points at {record_id}, '
                    f'which does not realize the fixture-canonical {claim_id}'
                )


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
        """A group must actually FORM, or this asserts nothing.

        A contested child is emitted as itself and deliberately never
        registered in `group_members`, so a hit list containing only the
        contested child produces no grouped document at all — and a
        `if group is not None:` guard around the real assertion would then
        pass unconditionally, and would keep passing if
        `_render_grouped_document` began folding contested bodies into the
        canonical.  That is precisely the esc-5712 silent-resolution failure
        PRD V2 forbids and this test is named for.  So a NON-contested sibling
        is seeded alongside, the group's existence is asserted rather than
        guarded, and the sibling's body is checked present to prove the
        document was really rendered from its members.
        """
        mod = _mod()
        parent, _ = _hit(PARENT, canonical=True, content='CANON')
        plain, _ = _hit('child-a', parent_id=PARENT, kind='amendment',
                        content='PLAIN')
        disputed, _ = _hit('child-b', parent_id=PARENT, kind='amendment',
                           content='DISPUTED')
        records_by_id = {PARENT: parent, 'child-a': plain, 'child-b': disputed}

        grouped = mod.apply_grouped_read(
            [plain, disputed], records_by_id, contested_ids={'child-b'},
        )

        group = next((r for r in grouped if r.record_id == PARENT), None)
        assert group is not None, 'no group formed, so nothing was tested'
        assert 'PLAIN' in group.content       # the document really was rendered
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
        # No `iscoroutinefunction` assertion: calling the function and
        # iterating the result — which the body does below — already fails
        # loudly on a coroutine, so introspecting the function object first
        # only restates it.
        mod = _mod()

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

        # The documented early-out is `return hits` (bake_off_storage_shape
        # .py:1019), so assert the LIST is the same object — `==` would also
        # pass on `return list(hits)`, which is a rewrite.
        assert pinned is hits

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

    The rejecting half is asserted where the rule is ENFORCED —
    `build_canonical_by_topic`, the sole producer of the index this function
    reads (see TestBuildCanonicalByTopic below).  `apply_topic_anchor` used to
    re-check it on the already-filtered index; that branch was unreachable
    through the module's own wiring, so the only thing that could exercise it
    was a hand-built dict no caller constructs.
    """

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

    def test_a_truthy_one_never_enters_the_index(self):
        """Bool identity, mirroring β's `invalid_canonical_type`.

        This is where the rule is enforced, and it is enforced ONCE: an
        impostor that never enters the index can never be pinned, so
        `apply_topic_anchor` needs no re-check downstream.  A truthy `1` is a
        FATAL violation at β's write boundary, so anchoring on one would
        report a discoverability win for a shape production cannot store.
        """
        mod = _mod()
        impostor = _anchor_hit('impostor', topic='alpha', canonical=1)

        index = mod.build_canonical_by_topic([impostor])

        assert index == {}
        # And therefore nothing to pin — the invariant, end to end.
        hits = [_anchor_hit('peer-1', topic='alpha')]
        assert mod.apply_topic_anchor(hits, canonical_by_topic=index) is hits

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
    """eval-design §1's discipline, asserted behaviorally.

    The failure mode is invisible: a score-reading metric produces perfectly
    plausible numbers that silently stop being comparable the moment the
    embedding config drifts — which is exactly how the 0.72-0.90 figure in
    the task record became 0.44-0.51 on re-measurement.  So the tests below
    perturb what a score-reading metric would be sensitive to and assert the
    numbers do not move.  (The ScoredHit/ArmRecord type split makes the same
    point structurally, but a type shape is not a test.)
    """

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
        # Optional dep, absent from this venv (see the docstring above); the
        # except arm is the normal path, so pyright cannot resolve it.
        import tiktoken  # type: ignore[reportMissingImports]  # noqa: PLC0415

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
        # The two names must be TELLABLE APART, which the identity check above
        # cannot see: were the constants ever set equal, it would still pass
        # while the artifact claimed tiktoken produced proxy numbers.  Compared
        # by value, so neither name's wording is pinned.
        assert name != mod.TIKTOKEN_ESTIMATOR_NAME

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


class TestTheCharProxyNameIsTrueOfItsArithmetic:
    """`CHAR_PROXY_ESTIMATOR_NAME` is printed into the operator-facing
    artifact as a factual claim about how its token numbers were produced.

    Nothing asserted the arithmetic behind it: the only comparison
    (`TestResolveTokenEstimator`) does `encode(...) == character_proxy_tokens(...)`
    where, on a venv without tiktoken, `encode` IS `character_proxy_tokens` —
    compared to itself.  An implementation returning `len(text)` or
    `len(text) // 8` satisfies every monotonicity/zero/determinism test in
    this file, and the report would then misname its own numbers by a factor
    of four.
    """

    def test_it_really_is_four_characters_per_token(self):
        mod = _mod()

        # The boundary, not just a large sample: `// 8` and `len(text)` both
        # break here, and so does an off-by-one divisor.
        assert mod.character_proxy_tokens('x' * 4) == 1
        assert mod.character_proxy_tokens('x' * 3) == 0
        assert mod.character_proxy_tokens('x' * 7) == 1
        assert mod.character_proxy_tokens('x' * 8) == 2
        assert mod.character_proxy_tokens('x' * 40) == 10

    def test_it_delegates_to_the_repos_one_chars_per_token_helper(self):
        """INV-5: a second literal `// 4` here could drift from
        `context_assembler.estimate_tokens` with no test noticing.  Asserted
        by VALUE — introspecting the function object would only restate the
        import."""
        from fused_memory.reconciliation.context_assembler import (  # noqa: PLC0415
            estimate_tokens,
        )

        mod = _mod()
        for text in ('', 'x', 'x' * 40, 'mixed English/JSON {"a": 1}' * 7):
            assert mod.character_proxy_tokens(text) == estimate_tokens(text)


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
        """The selector reads `.category` / `.source_store` as ENUMS and the
        cosine from `.metadata['store_score']`; dicts would raise, and
        dicts-with-strings would silently fail every comparison and report
        "guard never fires".

        The cosine's placement is asserted because getting it wrong is silent:
        since task 3658 the guard thresholds on `metadata['store_score']`, so
        an adapter that left the cosine in `relevance_score` would make
        guard_adequacy report `guard_matched: False` for every arm in the
        program rather than raising.
        """
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
        assert captured[0].id == 's0'
        # The arm's cosine, where the guard actually reads it.
        assert captured[0].metadata['store_score'] == 0.95
        assert captured[1].metadata['store_score'] == 0.10
        # ...and the post-fusion shape production really produces: a 1-based
        # window rank and the ordinal RRF value, not the cosine.
        assert [r.metadata['store_rank'] for r in captured] == [1, 2]
        assert captured[0].relevance_score == pytest.approx(1.0 / 61)

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


class TestAuditThresholdIsNotHardcodedTwice:
    """The D10 mirror of the guard-threshold class above.

    `resolve_audit_threshold` reads the detector's default reflectively so the
    artifact cannot report a threshold nobody runs — but nothing tied it to
    the number the artifact actually SHIPS, and every other test hardcodes
    0.85.  A retune upstream would silently shift the reported audit-recall
    while the committed report went on saying 0.85.

    Deliberately NOT re-asserting `inspect.signature(...)` against itself:
    that form was deleted in an earlier cycle as tautological, and rightly —
    it could only fail if the function stopped using inspect, and a retune
    moved both sides together.  The committed artifact is the independent
    side that makes the comparison mean something.
    """

    def test_the_resolved_threshold_is_the_one_the_committed_artifact_reports(self):
        """A retune upstream must break the build rather than quietly make
        the shipped number describe a detector nobody is running."""
        committed = _committed_report()['audit_recall']

        assert _mod().resolve_audit_threshold() == committed['threshold']

    def test_the_committed_artifact_names_the_detector_it_replayed(self):
        """The threshold alone is meaningless if the reader cannot tell which
        function it was the default of."""
        committed = _committed_report()['audit_recall']
        detector = _mod().load_audit_script().find_near_duplicate_memory_groups

        assert committed['detector'] == (
            f'audit_duplicate_memories.{detector.__name__}'
        )

    def test_a_renamed_threshold_parameter_is_a_clear_error_not_a_keyerror(
        self, monkeypatch
    ):
        """The failure mode this guards is a rename in
        `audit_duplicate_memories`, which surfaces only during a live D10 run.
        A bare `KeyError: 'threshold'` there names neither the upstream that
        moved nor the fix."""
        mod = _mod()

        def _renamed(records, *, cutoff=0.85):  # NOT `threshold`
            return []

        monkeypatch.setattr(
            mod, 'load_audit_script',
            lambda: types.SimpleNamespace(
                find_near_duplicate_memory_groups=_renamed,
            ),
        )

        with pytest.raises(RuntimeError) as excinfo:
            mod.resolve_audit_threshold()

        message = str(excinfo.value)
        assert not isinstance(excinfo.value, KeyError)
        assert 'threshold' in message
        # Names the upstream that moved and what it found instead, so the
        # reader does not have to go and diff the signature by hand.
        assert '_renamed' in message
        assert 'cutoff' in message


class TestDropCollectionsNeverAbandonsTheRemainingNames:
    """The reaper that actually removes the ephemeral collections.

    Every other test in this file monkeypatches `drop_collections` away, so
    its two documented contracts — tolerate an already-absent collection, and
    never abandon the rest after one failure — were unasserted.  A reaper that
    stops at the first error leaks collections into a SHARED Qdrant, which is
    the same class of failure the sibling reaper's own test file was added for
    in this diff.
    """

    @staticmethod
    def _fake_qdrant(monkeypatch, *, fails_on: Container[str] = frozenset()):
        import qdrant_client  # noqa: PLC0415

        state = {'attempted': [], 'closed': 0}

        class _Client:
            def __init__(self, url, timeout=None):
                state['url'] = url

            def delete_collection(self, name):
                state['attempted'].append(name)
                if name in fails_on:
                    raise RuntimeError(f'qdrant said no: {name}')

            def close(self):
                state['closed'] += 1

        monkeypatch.setattr(qdrant_client, 'QdrantClient', _Client)
        return state

    def test_a_failure_on_one_name_still_attempts_every_other(self, monkeypatch):
        """The one that matters: a mid-list failure that aborted the loop
        would leave later collections alive with the run reporting success."""
        state = self._fake_qdrant(monkeypatch, fails_on={'b'})

        _mod().drop_collections(['a', 'b', 'c'], qdrant_url='http://q:6333')

        assert state['attempted'] == ['a', 'b', 'c']
        assert state['closed'] == 1

    def test_an_already_absent_collection_is_not_an_error(self, monkeypatch):
        """Absence is the NORMAL case on the pre-run sweep, so every name
        failing must still return rather than raise into the caller."""
        state = self._fake_qdrant(monkeypatch, fails_on={'a', 'b'})

        _mod().drop_collections(['a', 'b'], qdrant_url='http://q:6333')

        assert state['attempted'] == ['a', 'b']
        assert state['closed'] == 1

    def test_the_client_is_closed_when_the_name_source_itself_raises(
        self, monkeypatch
    ):
        """Chosen because it is the ONLY case that distinguishes the `finally`
        from a plain trailing `close()`.

        A "closed even when a delete failed" test would pass identically
        against a trailing close, since the per-name `except` swallows delete
        failures and the line after the loop is always reached — the exact
        dead test the sibling reaper's review caught.  `names` is typed `Any`,
        so a caller can hand it a lazy iterable that raises mid-iteration; that
        escapes the inner `except` and only `finally` closes the connection.
        """
        state = self._fake_qdrant(monkeypatch)

        def _names():
            yield 'a'
            raise RuntimeError('the name source died')

        with pytest.raises(RuntimeError, match='the name source died'):
            _mod().drop_collections(_names(), qdrant_url='http://q:6333')

        assert state['attempted'] == ['a']
        assert state['closed'] == 1, 'the connection leaked'


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


# ===========================================================================
# step-15 — PRD D10: audit-recall over alpha/3130's labeled fixture
# ===========================================================================
#
# D10 (3136's deferral item 3): "ζ also delivers the audit-recall measurement
# — run audit_duplicate_memories.py against α/3130's labeled fixture and
# report recall on the paraphrase class — the number that decides how much to
# trust the κ report."
#
# NO RATE, BOUND OR TOLERANCE IS ASSERTED AGAINST THE REAL FIXTURE (gate G6).
# The measurement informs a judgement; it does not gate a build. Asserting
# today's recall would also freeze it: a detector improvement would read as a
# test failure, which is the precise opposite of what the number is for.
# So correctness is proven on hand-built corpora with exactly-derivable
# answers, and the real fixture is asserted for STRUCTURE only.
#
# The positive class is split into two bands by each pair's max
# SequenceMatcher ratio against the threshold:
#
#   lexical band     — the detector COULD reach it. "Did it?" is a fair
#                      question about the detector.
#   paraphrase band  — by construction unreachable by a character-level
#                      threshold, no matter how the detector is tuned short
#                      of changing kind. Counting these as detector misses
#                      without saying so would read as "the audit script is
#                      broken" rather than "this class is invisible to it".
#
# GOTCHA (pinned at test_audit_duplicate_memories.py:1116-1119):
# SequenceMatcher.ratio() is ORDER-SENSITIVE — the known exemplar pair scores
# 0.0948 one way and 0.2279 the other on normalised content. The band split
# therefore takes the MAX over both orderings: a pair is only called
# unreachable if NEITHER ordering reaches the threshold, which can never
# over-claim the paraphrase band.

#: Exactly-derivable corpus: one byte-similar positive pair, one paraphrase
#: positive pair, and the four cross-cluster negatives they imply.
_SYNTHETIC_CORPUS = [
    {'memory_id': 'a1', 'cluster_id': 'c1', 'label': 'canonical',
     'content': 'The merge worker retries a failed rebase exactly twice.'},
    {'memory_id': 'a2', 'cluster_id': 'c1', 'label': 'duplicate',
     'content': 'The merge worker retries a failed rebase exactly twice!'},
    {'memory_id': 'b1', 'cluster_id': 'c2', 'label': 'canonical',
     'content': 'Qdrant collections are named by project id.'},
    {'memory_id': 'b2', 'cluster_id': 'c2', 'label': 'duplicate',
     'content': 'Vector store namespaces derive from the canonicalised '
                'project identifier.'},
]

#: Same cluster, but the third record is curator-ruled NOT a duplicate while
#: being byte-similar — the hardest negative there is.
_HARD_NEGATIVE_CORPUS = [
    {'memory_id': 'a1', 'cluster_id': 'c1', 'label': 'canonical',
     'content': 'The merge worker retries a failed rebase exactly twice.'},
    {'memory_id': 'a2', 'cluster_id': 'c1', 'label': 'duplicate',
     'content': 'The merge worker retries a failed rebase exactly twice!'},
    {'memory_id': 'a3', 'cluster_id': 'c1', 'label': 'distinct',
     'content': 'The merge worker retries a failed rebase exactly twice?'},
]


@functools.cache
def _synthetic_audit():
    return _mod().audit_recall_over_labeled_fixture(_SYNTHETIC_CORPUS, 0.85)


@functools.cache
def _fixture_audit():
    """The real measurement. Cached — it is an O(n^2) difflib sweep."""
    mod = _mod()
    return mod.audit_recall_over_labeled_fixture(
        mod.load_labeled_fixture(ALPHA_FIXTURE_PATH), 0.85,
    )


class TestAuditRecallOnAnExactlyDerivableCorpus:
    """Four records, one detectable pair, one paraphrase pair. No tolerances."""

    def test_recall_over_the_positive_class(self):
        result = _synthetic_audit()

        assert result['true_dup']['pairs'] == 2
        assert result['true_dup']['recovered'] == 1  # only the byte-similar one
        assert result['true_dup']['recall'] == 0.5

    def test_the_positive_class_splits_into_two_bands(self):
        result = _synthetic_audit()

        assert result['true_dup']['lexical_band']['pairs'] == 1
        assert result['true_dup']['paraphrase_band']['pairs'] == 1

    def test_per_band_recall_isolates_what_the_detector_could_reach(self):
        """The headline number and the fair number, side by side: 0.5 overall
        reads as a mediocre detector; 1.0 of what it could reach plus 0.0 of
        what it structurally cannot reads as a detector working exactly as
        designed on a corpus that is mostly out of its reach."""
        result = _synthetic_audit()

        assert result['true_dup']['lexical_band']['recall'] == 1.0
        assert result['true_dup']['paraphrase_band']['recall'] == 0.0

    def test_the_bands_partition_the_positive_class_exactly(self):
        result = _synthetic_audit()['true_dup']

        assert (
            result['lexical_band']['pairs'] + result['paraphrase_band']['pairs']
            == result['pairs']
        )
        assert (
            result['lexical_band']['recovered']
            + result['paraphrase_band']['recovered'] == result['recovered']
        )

    def test_a_clean_corpus_reports_no_false_groupings(self):
        result = _synthetic_audit()

        assert result['unrelated']['pairs'] == 4
        assert result['unrelated']['falsely_grouped'] == 0
        assert result['unrelated']['rate'] == 0.0

    def test_an_empty_negative_class_reports_none_not_a_measured_zero(self):
        """This corpus has no hard negatives at all. Reporting 0.0 would put
        a perfect score in the table for something never measured."""
        result = _synthetic_audit()

        assert result['hard_negative']['pairs'] == 0
        assert result['hard_negative']['rate'] is None

    def test_false_groupings_are_counted_when_they_happen(self):
        """Byte-similar but curator-ruled NOT duplicates: the detector unions
        all three transitively, so both hard-negative pairs are false."""
        result = _mod().audit_recall_over_labeled_fixture(
            _HARD_NEGATIVE_CORPUS, 0.85,
        )

        assert result['hard_negative']['pairs'] == 2
        assert result['hard_negative']['falsely_grouped'] == 2
        assert result['hard_negative']['rate'] == 1.0
        # ...and the true positive in the same corpus is still recovered:
        # a false-grouping count is not a recall penalty.
        assert result['true_dup']['recall'] == 1.0

    def test_a_corpus_too_small_to_pair_reports_no_measurement(self):
        result = _mod().audit_recall_over_labeled_fixture(
            _SYNTHETIC_CORPUS[:1], 0.85,
        )

        assert result['true_dup']['pairs'] == 0
        assert result['true_dup']['recall'] is None
        assert result['groups_found'] == 0


class TestMaxLexicalRatio:
    """The band split's ruler — order-insensitive by construction."""

    def test_it_takes_the_max_over_both_argument_orders(self):
        """SequenceMatcher is order-sensitive; a one-directional ruler would
        put a pair in the paraphrase band or not depending on which id sorted
        first, which is not a property of the pair."""
        import difflib  # noqa: PLC0415

        mod = _mod()
        left, right = 'the merge worker retries twice', 'retries twice worker'
        forward = difflib.SequenceMatcher(None, left, right).ratio()
        backward = difflib.SequenceMatcher(None, right, left).ratio()

        measured = mod.max_lexical_ratio(left, right)

        assert measured == max(forward, backward)
        assert measured == mod.max_lexical_ratio(right, left)  # symmetric

    def test_it_normalises_the_way_the_detector_does(self):
        """`(content or '').strip().lower()` — if the ruler and the detector
        disagreed on normalisation, the band split would describe a detector
        nobody is running."""
        mod = _mod()

        assert mod.max_lexical_ratio('  Merge Worker ', 'merge worker') == 1.0

    def test_empty_content_has_no_measurable_ratio(self):
        """The detector explicitly refuses to cluster empty content (an
        unextractable memory must not be deleted as a duplicate), so there is
        no ratio to report — None, not the 1.0 SequenceMatcher would give."""
        assert _mod().max_lexical_ratio('', '') is None
        assert _mod().max_lexical_ratio('body', '   ') is None


@pytest.mark.xdist_group('e2_audit_recall')
class TestAuditRecallOverTheCommittedFixture:
    """STRUCTURE ONLY. G6/D10: the number is reported, never asserted.

    Grouped onto one xdist worker so the O(n^2) difflib sweep over the 104
    committed records is paid ONCE (via the `_fixture_audit` cache) rather
    than once per worker the class happens to be scattered across.
    """

    def test_the_pair_counts_match_the_labeled_partition(self):
        """Fixture facts, not detector rates: `build_pair_sets` produces these
        exact class sizes for the committed 104 records."""
        result = _fixture_audit()

        assert result['true_dup']['pairs'] == 301
        assert result['hard_negative']['pairs'] == 18
        assert result['unrelated']['pairs'] == 5037

    def test_the_bands_sum_to_the_positive_class(self):
        result = _fixture_audit()['true_dup']

        assert (
            result['lexical_band']['pairs'] + result['paraphrase_band']['pairs']
            == 301
        )

    def test_every_reported_rate_is_a_fraction_or_no_measurement(self):
        result = _fixture_audit()
        rates = [
            result['true_dup']['recall'],
            result['true_dup']['lexical_band']['recall'],
            result['true_dup']['paraphrase_band']['recall'],
            result['hard_negative']['rate'],
            result['unrelated']['rate'],
        ]

        for rate in rates:
            assert rate is None or 0.0 <= rate <= 1.0

    def test_the_payload_names_the_detector_it_replayed(self):
        """The report is read months later by somebody deciding how much to
        trust the κ sweep. "Recall was X" is unreadable without "of what"."""
        result = _fixture_audit()

        assert 'find_near_duplicate_memory_groups' in result['detector']
        assert result['threshold'] == 0.85

    def test_the_known_paraphrase_exemplar_is_not_lexically_reachable(self):
        """The independently-measured exemplar (cluster e0a41fcd, cosine
        0.905 at difflib 0.102) proves the paraphrase class is real and
        structurally invisible to the character threshold — asserted as band
        membership, not as a pinned ratio."""
        mod = _mod()
        records = {
            r['memory_id']: r
            for r in mod.load_labeled_fixture(ALPHA_FIXTURE_PATH)
        }
        left = records['243b6dec-f0ce-4123-bb09-16d834b7e9c8']
        right = records['c315352b-6d4e-467d-9a3f-360bc2d53229']

        ratio = mod.max_lexical_ratio(left['content'], right['content'])

        assert left['cluster_id'] == right['cluster_id']  # a true-dup pair
        assert ratio < 0.85

    def test_paraphrase_exemplars_are_emitted_for_hand_auditing(self):
        """A band split nobody can check by hand is a number to be believed
        rather than read. The exemplars are the nearest misses — how far the
        threshold would have to fall to reach the class at all."""
        result = _fixture_audit()
        exemplars = result['paraphrase_exemplars']

        assert exemplars, 'the paraphrase band is non-empty; show some of it'
        for exemplar in exemplars:
            assert exemplar['max_ratio'] < result['threshold']
        # Deterministic order (descending ratio), so a rerun's diff is signal.
        assert [e['max_ratio'] for e in exemplars] == sorted(
            (e['max_ratio'] for e in exemplars), reverse=True,
        )

    def test_the_measurement_is_deterministic(self):
        """No sampling, no set iteration order leaking into the numbers."""
        mod = _mod()
        records = mod.load_labeled_fixture(ALPHA_FIXTURE_PATH)

        first = mod.audit_recall_over_labeled_fixture(records[:30], 0.85)
        second = mod.audit_recall_over_labeled_fixture(records[:30], 0.85)

        assert first == second


# ===========================================================================
# step-17 — the report: build_report + render_markdown
# ===========================================================================
#
# `plans/e2-storage-shape-bakeoff-report.{json,md}` is this task's
# user-observable output and the signal gate leaf η puts in front of an
# operator: the PRD's choice between δ-as-default and peers-as-default gets
# made by reading it. So the artifact is held to artifact standards.
#
#   * SIX arm variants — three shapes x pin on/off. The pin is a read-side
#     transform, so its variants share their shape's seeded collection, but
#     they are separate ROWS: "does the pin help?" is a question the table
#     must answer per shape.
#   * A partial table RAISES. A decision table with a silently blank cell is
#     worse than no table: the reader cannot tell "measured and equal" from
#     "never measured", and the blank always reads as the former.
#   * Rendering is byte-deterministic, so a rerun's diff is signal.
#
# NO metric value, rate or bound is asserted (G6) — every measurement below
# is synthetic, chosen to make a shape assertion legible, and means nothing.


def _arm_measurement(*, recall5=0.8, recall10=0.9, estimator='injected:words',
                     pin_on=False, window_changed_rate=0.25):
    """One arm's metrics, fully populated. Values are arbitrary.

    ``window_changed_rate`` is ``None`` on a pin-OFF arm: the question "did
    the pin change the window?" was never asked there, and a 0.0 would read
    as "asked, and it changed nothing".
    """
    return {
        'pin': {
            'enabled': pin_on,
            'window_changed_rate': window_changed_rate if pin_on else None,
        },
        'claim_recall': {'at_5': recall5, 'at_10': recall10},
        'discoverability': {
            'canonical_in_top_5_rate': 0.7,
            'median_canonical_rank': 2.0,
            'canonical_found_count': 14,
            'canonical_candidates': 20,
            'canonical_rank_window': 10,
            'mean_topic_member_count': 3.0,
            # Deliberately DIFFERENT from the transform-credited values above.
            # Still arbitrary — but a renderer wired to the wrong key would
            # print a number that matches its neighbour, and equal fixtures
            # would let that pass.
            'stored_canonical_in_top_5_rate': 0.4,
            'stored_canonical_median_rank': 5.0,
            'stored_canonical_found_count': 9,
        },
        # EVERY subset carries different numbers — from the pooled block AND
        # from each other.  Identical subsets would let a renderer that
        # sourced every by-kind row from one subset (the first kind in
        # iteration order, say) pass while printing `claim`'s numbers on the
        # `held_out` row — which is precisely the row the transform-blind
        # column exists for, since it is the only one measuring
        # generalisation.  The per-kind offset is what makes that visible.
        'by_query_kind': {
            kind: {
                'queries': 8 + index,
                'claim_recall': {
                    'at_5': round(recall5 - 0.01 * (index + 1), 2),
                    'at_10': round(recall10 - 0.01 * (index + 1), 2),
                },
                'discoverability': {
                    'canonical_in_top_5_rate': round(0.71 + 0.01 * index, 2),
                    'median_canonical_rank': 2.0 + index,
                    'canonical_found_count': 6 + index,
                    'canonical_candidates': 8 + index,
                    'canonical_rank_window': 10,
                    # Distinct from the pooled block's stored values too, so
                    # a by-kind cell sourced from the pooled block is visible.
                    'stored_canonical_in_top_5_rate': round(0.31 + 0.01 * index, 2),
                    'stored_canonical_median_rank': 4.0 + index,
                    'stored_canonical_found_count': 4 + index,
                },
            }
            for index, kind in enumerate(
                (*_mod().QUERY_KINDS, _mod().HELD_OUT_SUBSET),
            )
        },
        'tokens_per_query': {'mean': 412.0, 'estimator': estimator},
        'guard_adequacy': {
            'clusters_measured': 15,
            'candidate_present_rate': 0.6,
            'guard_matched_rate': 0.2,
            'threshold_replay': True,
            'threshold': 0.92,
            'max_observed_score': 0.71,
            'probes': 15,
            'guard_covered_probes': 12,
            'guard_covered_category': 'procedural_knowledge',
        },
    }


def _all_arms():
    return {
        arm: _arm_measurement(pin_on=arm.endswith('+pin'))
        for arm in _mod().ARM_VARIANTS
    }


def _protocol():
    return {
        'blind_authoring': 'single-author-blind-to-metrics (commit ordering)',
        'fixtures': [
            {'path': 'tests/fixtures/e2_arm_claims.jsonl', 'commit': 'abc1234'},
            {'path': 'tests/fixtures/e2_query_set.jsonl', 'commit': 'def5678'},
        ],
        'token_estimator': 'char-proxy:4-chars-per-token',
        'guard_threshold': 0.92,
        'distractor_slab_size': 40,
        'embedder_model': 'text-embedding-3-small',
    }


def _audit_recall():
    return {
        'detector': 'audit_duplicate_memories.find_near_duplicate_memory_groups',
        'threshold': 0.85,
        'groups_found': 0,
        'true_dup': {
            'pairs': 301, 'recovered': 0, 'recall': 0.0,
            'lexical_band': {'pairs': 0, 'recovered': 0, 'recall': None},
            'paraphrase_band': {'pairs': 301, 'recovered': 0, 'recall': 0.0},
        },
        'hard_negative': {'pairs': 18, 'falsely_grouped': 0, 'rate': 0.0},
        'unrelated': {'pairs': 5037, 'falsely_grouped': 0, 'rate': 0.0},
        'paraphrase_exemplars': [{'a': 'x', 'b': 'y', 'max_ratio': 0.55}],
    }


def _report():
    return _mod().build_report(
        arms=_all_arms(), audit_recall=_audit_recall(), protocol=_protocol(),
    )


class TestArmVariants:
    """Three shapes x pin on/off, named once."""

    def test_there_are_exactly_six_named_variants(self):
        mod = _mod()

        assert mod.ARM_VARIANTS == (
            'status_quo', 'status_quo+pin',
            'c_peers', 'c_peers+pin',
            'b_grouped', 'b_grouped+pin',
        )

    def test_every_shape_appears_with_and_without_the_pin(self):
        mod = _mod()

        for shape in mod.ARM_SHAPES:
            assert shape in mod.ARM_VARIANTS
            assert f'{shape}+pin' in mod.ARM_VARIANTS


class TestBuildReportShape:
    """What gate η reads."""

    def test_it_carries_an_entry_for_every_arm_variant(self):
        report = _report()

        assert list(report['arms']) == list(_mod().ARM_VARIANTS)

    def test_every_arm_carries_all_four_e2_metrics(self):
        report = _report()

        for arm, measurement in report['arms'].items():
            assert set(measurement) >= {
                'claim_recall', 'discoverability', 'tokens_per_query',
                'guard_adequacy',
            }, arm

    def test_claim_recall_is_reported_at_both_k(self):
        """k=5 because the near-dup guard lives there; k=10 because a shape
        that merely ranks slower is a different finding from one that loses
        the claim outright."""
        report = _report()

        for measurement in report['arms'].values():
            assert set(measurement['claim_recall']) == {'at_5', 'at_10'}

    def test_guard_adequacy_keeps_both_of_its_parts(self):
        report = _report()

        for measurement in report['arms'].values():
            guard = measurement['guard_adequacy']
            assert 'candidate_present_rate' in guard  # rank/set-based
            assert 'guard_matched_rate' in guard      # threshold replay
            assert guard['threshold_replay'] is True

    def test_it_carries_the_d10_audit_recall_block(self):
        report = _report()

        assert report['audit_recall']['true_dup']['paraphrase_band']['pairs'] == 301

    def test_the_protocol_block_records_how_the_experiment_was_run(self):
        """An arbitration artifact whose provenance is not in it cannot be
        re-read six months later by somebody who was not here."""
        protocol = _report()['protocol']

        assert protocol['blind_authoring']
        assert protocol['fixtures'][0]['commit']       # the audit trail
        assert protocol['token_estimator']             # which numbers these are
        assert protocol['guard_threshold'] == 0.92
        assert protocol['distractor_slab_size'] == 40
        assert protocol['embedder_model']

    def test_the_report_is_json_serializable(self):
        """It is written to disk as JSON; a non-serializable value would fail
        at the very end of an hour-long run."""
        json.dumps(_report())


class TestBuildReportRefusesAPartialTable:
    """A blank cell reads as "measured and equal". It never is."""

    def test_a_missing_arm_raises(self):
        mod = _mod()
        arms = _all_arms()
        del arms['b_grouped+pin']

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'b_grouped+pin' in str(excinfo.value)

    def test_a_missing_metric_names_both_the_arm_and_the_metric(self):
        mod = _mod()
        arms = _all_arms()
        del arms['c_peers']['tokens_per_query']

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'c_peers' in str(excinfo.value)
        assert 'tokens_per_query' in str(excinfo.value)

    def test_a_missing_k_within_claim_recall_raises(self):
        """The nested case is the one that would actually slip through: the
        metric key is present, so a shallow check passes and the column
        renders blank."""
        mod = _mod()
        arms = _all_arms()
        del arms['status_quo']['claim_recall']['at_10']

        with pytest.raises(mod.IncompleteReportError):
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

    def test_a_missing_guard_part_raises(self):
        mod = _mod()
        arms = _all_arms()
        del arms['b_grouped']['guard_adequacy']['candidate_present_rate']

        with pytest.raises(mod.IncompleteReportError):
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

    @pytest.mark.parametrize('kind', ['claim', 'topic_phrasing', 'held_out'])
    def test_a_missing_by_query_kind_subset_raises(self, kind):
        """The same nested case as `claim_recall`, on the block that carries
        the ONLY claim-vs-topic split in the report.  `by_query_kind` is in
        `_REQUIRED_ARM_METRICS` but had no refusal test, so a subset dropped
        by a bad split would publish a by-kind table with a silently absent
        row rather than refusing."""
        mod = _mod()
        arms = _all_arms()
        del arms['b_grouped']['by_query_kind'][kind]

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'b_grouped' in str(excinfo.value)
        assert 'by_query_kind' in str(excinfo.value)

    @pytest.mark.parametrize('kind', ['claim', 'topic_phrasing', 'held_out'])
    @pytest.mark.parametrize('metric,key', [
        ('discoverability', 'stored_canonical_in_top_5_rate'),
        ('discoverability', 'canonical_in_top_5_rate'),
        ('claim_recall', 'at_5'),
    ])
    def test_a_missing_key_INSIDE_a_by_kind_subset_raises(self, kind, metric, key):
        """One level deeper than the subset-name check above.

        The by-kind table subscripts `subset['discoverability'][...]`
        directly, so before this the failure mode was a raw `KeyError` out of
        `render_markdown` — a traceback where every other missing measurement
        in this module produces an error naming the arm, the kind and the key.
        Fail-loud versus fail-obscure: the artifact is refused either way, but
        only one of them tells the operator which subset broke.
        """
        mod = _mod()
        arms = _all_arms()
        del arms['b_grouped']['by_query_kind'][kind][metric][key]

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        message = str(excinfo.value)
        assert 'b_grouped' in message
        assert kind in message
        assert key in message

    @pytest.mark.parametrize('block', ['claim_recall', 'discoverability',
                                       'queries'])
    def test_a_hollow_by_kind_subset_raises(self, block):
        """The subset is PRESENT — so the kind-name check passes — and empty.
        A split that produced the right keys with nothing under them would
        otherwise publish a by-kind row assembled out of a traceback."""
        mod = _mod()
        arms = _all_arms()
        del arms['c_peers+pin']['by_query_kind']['held_out'][block]

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        message = str(excinfo.value)
        assert 'c_peers+pin' in message
        assert 'held_out' in message
        assert block in message

    def test_the_transform_blind_trio_is_registered_not_merely_produced(self):
        """Registration is what obliges the renderer to carry the column.

        `_REQUIRED_ARM_METRICS` is "enumerated ONCE and used by both the
        completeness check and the renderer, so a metric cannot be validated
        into the JSON and then quietly dropped from the table".  A
        transform-blind measurement that lived only in the JSON would leave
        the artifact gate η actually reads exactly as undisclosed as before.
        """
        required = _mod()._REQUIRED_ARM_METRICS['discoverability']

        assert 'stored_canonical_in_top_5_rate' in required
        assert 'stored_canonical_median_rank' in required
        assert 'stored_canonical_found_count' in required

    @pytest.mark.parametrize('key', ['stored_canonical_in_top_5_rate',
                                     'stored_canonical_median_rank',
                                     'stored_canonical_found_count'])
    def test_a_missing_transform_blind_key_raises(self, key):
        """The nested case again: `discoverability` is present, so a shallow
        check passes and the new column renders blank — reading as "measured,
        and equal to its neighbour", which for THIS column is precisely the
        conflation it exists to disclose."""
        mod = _mod()
        arms = _all_arms()
        del arms['c_peers+pin']['discoverability'][key]

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'c_peers+pin' in str(excinfo.value)
        assert key in str(excinfo.value)

    def test_an_unknown_arm_name_raises(self):
        """A typo would otherwise drop a real arm AND add a phantom one, and
        the table would still look complete."""
        mod = _mod()
        arms = _all_arms()
        arms['c_peers_pin'] = arms.pop('c_peers+pin')

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'c_peers_pin' in str(excinfo.value)

    def test_a_missing_protocol_key_raises(self):
        mod = _mod()
        protocol = _protocol()
        del protocol['token_estimator']

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=_all_arms(), audit_recall=_audit_recall(), protocol=protocol,
            )

        assert 'token_estimator' in str(excinfo.value)

    def test_a_none_measurement_is_accepted_and_is_not_a_missing_one(self):
        """"Measured, no denominator" is a legitimate result and must survive
        to the table as such — only an ABSENT key is a broken run."""
        mod = _mod()
        arms = _all_arms()
        arms['status_quo']['claim_recall']['at_10'] = None

        report = mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )

        assert report['arms']['status_quo']['claim_recall']['at_10'] is None


def _decision_table_rows(rendered: str) -> list[str]:
    """The decision table's data rows — header and separator excluded."""
    lines = rendered.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('| arm '))
    rows = []
    for line in lines[start + 2:]:  # skip the header and its `| --- |`
        if not line.startswith('| '):
            break
        rows.append(line)
    return rows


class TestPinCellKeepsNeverFiredApartFromBarelyFired:
    """Three outcomes in one column, and `0.00` is the load-bearing one.

    The reading guide defines `0.00` as "the pin never fired".  Reusing
    `_cell`'s 2-decimal precision made a rate that DID fire print that value,
    so the artifact asserted the opposite of the measurement in the one column
    built to tell the two apart.
    """

    def test_the_pin_off_row_is_no_measurement_not_a_zero(self):
        assert _mod()._pin_cell(None) == '—'

    def test_an_exact_zero_still_prints_as_the_never_fired_value(self):
        """`0.00` must keep meaning what the guide says it means."""
        assert _mod()._pin_cell(0.0) == '0.00'

    def test_the_committed_runs_underflowing_rate_does_not_print_as_zero(self):
        """2 of 487 windows — the value that shipped as `0.00`."""
        mod = _mod()

        assert mod._pin_cell(2 / 487) == '<0.01'

    @pytest.mark.parametrize('rate', [1e-9, 0.0001, 0.004, 0.00499])
    def test_every_rate_that_rounds_to_zero_is_flagged_instead(self, rate):
        assert _mod()._pin_cell(rate) == '<0.01'

    @pytest.mark.parametrize('rate,expected', [
        (0.005, '0.01'),    # the first rate that rounds UP is left alone
        (0.38, '0.38'),
        (1.0, '1.00'),
    ])
    def test_a_rate_that_survives_rounding_is_untouched(self, rate, expected):
        assert _mod()._pin_cell(rate) == expected


class TestRenderMarkdown:
    """The operator-facing decision table."""

    def test_it_renders_exactly_one_row_per_arm_in_the_json(self):
        """Sliced out of the decision table specifically — the artifact holds
        several tables, and "every row somewhere in the document" would pass
        even if an arm rendered into the D10 block by mistake."""
        report = _report()

        rows = _decision_table_rows(_mod().render_markdown(report))

        assert len(rows) == len(report['arms'])
        for arm in report['arms']:
            assert sum(1 for row in rows if row.startswith(f'| {arm} |')) == 1

    def test_it_names_the_estimator_that_produced_the_token_column(self):
        rendered = _mod().render_markdown(_report())

        assert 'char-proxy:4-chars-per-token' in rendered

    def test_a_missing_measurement_renders_as_no_measurement_not_zero(self):
        mod = _mod()
        arms = _all_arms()
        arms['status_quo']['claim_recall']['at_5'] = None
        report = mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )

        row = next(
            line for line in mod.render_markdown(report).splitlines()
            if line.startswith('| status_quo |')
        )

        assert '0.00' not in row.split('|')[2]
        assert '—' in row

    def test_rendering_is_byte_identical_for_identical_input(self):
        """A rerun's diff is only signal if formatting contributes nothing."""
        mod = _mod()

        assert mod.render_markdown(_report()) == mod.render_markdown(_report())

    def test_it_records_the_fixture_commits_that_prove_the_blind_protocol(self):
        rendered = _mod().render_markdown(_report())

        assert 'abc1234' in rendered
        assert 'e2_arm_claims.jsonl' in rendered


_STORED_COLUMN = 'canonical in top-5 (stored)'


def _cells(row: str) -> list[str]:
    """`| a | b |` -> `['a', 'b']`."""
    return [cell.strip() for cell in row.strip().strip('|').split('|')]


def _by_kind_table(rendered: str) -> tuple[list[str], list[str]]:
    """The by-query-kind table's (header cells, data rows).

    Located by the renderer-emitted `## By query kind` heading rather than by
    a line index, so it does not move when a paragraph above it does.  The
    decision table's header ALSO starts with `| arm `, which is why the
    search starts at the heading.
    """
    lines = rendered.splitlines()
    start = lines.index('## By query kind')
    header_at = next(
        i for i, line in enumerate(lines[start:], start)
        if line.startswith('| arm ')
    )
    rows = []
    for line in lines[header_at + 2:]:  # skip the header and its `| --- |`
        if not line.startswith('| '):
            break
        rows.append(line)
    return _cells(lines[header_at]), rows


class TestTheTransformBlindColumnReachesTheTables:
    """The disclosure has to be beside the number it qualifies.

    A caveat three sections away from `canonical in top-5` is not a caveat on
    that column — gate η reads the decision table, and the single biggest
    number in it is the one the grouped read credits.  So the transform-blind
    rate is a COLUMN, immediately after its transform-credited twin, in the
    headline table and in the by-kind table both.
    """

    def test_the_stored_column_sits_immediately_after_the_one_it_qualifies(self):
        columns = _mod().DECISION_TABLE_COLUMNS

        assert _STORED_COLUMN in columns
        assert columns.index(_STORED_COLUMN) == \
            columns.index('canonical in top-5') + 1

    def test_the_rendered_header_is_still_the_pinned_column_set(self):
        """The invariant the committed artifact is already held to, asserted
        here on a synthetic report so a column added to the constant without
        being rendered fails at the unit level rather than at regeneration."""
        mod = _mod()

        rendered = mod.render_markdown(_report())
        header = next(
            line for line in rendered.splitlines() if line.startswith('| arm ')
        )

        assert header == '| ' + ' | '.join(mod.DECISION_TABLE_COLUMNS) + ' |'

    def test_each_arms_cell_comes_from_the_stored_rate(self):
        """Located by column NAME, never by a hardcoded index.

        The fixture's stored rate (0.40) differs from its transform-credited
        neighbour (0.70) on purpose: a cell wired to the wrong key would
        otherwise print a plausible number and pass.
        """
        mod = _mod()
        report = _report()
        column = mod.DECISION_TABLE_COLUMNS.index(_STORED_COLUMN)
        neighbour = mod.DECISION_TABLE_COLUMNS.index('canonical in top-5')

        rows = _decision_table_rows(mod.render_markdown(report))

        assert rows
        for row in rows:
            cells = _cells(row)
            arm = cells[0]
            expected = report['arms'][arm]['discoverability'][
                'stored_canonical_in_top_5_rate']
            # Same precision as the column it qualifies — two numbers a
            # reader is meant to compare must not be formatted differently.
            assert cells[column] == f'{expected:.2f}'
            assert cells[column] != cells[neighbour]

    def test_an_unmeasured_stored_rate_renders_as_no_measurement(self):
        """`—`, never `0.00`.  On THIS column a printed zero would read as
        "retrieval never found the canonical", which is a finding — and the
        opposite of "we did not measure"."""
        mod = _mod()
        arms = _all_arms()
        arms['status_quo']['discoverability'][
            'stored_canonical_in_top_5_rate'] = None
        report = mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )
        column = mod.DECISION_TABLE_COLUMNS.index(_STORED_COLUMN)

        row = next(
            line for line in _decision_table_rows(mod.render_markdown(report))
            if line.startswith('| status_quo |')
        )

        assert _cells(row)[column] == mod._NO_MEASUREMENT

    def test_the_by_kind_table_carries_the_column_too(self):
        """`held_out` is the row that measures generalisation, and it is
        exactly the row where a transform-credited number is least safe to
        read alone."""
        mod = _mod()
        report = _report()

        header, rows = _by_kind_table(mod.render_markdown(report))
        column = header.index(_STORED_COLUMN)

        assert header.index('canonical in top-5') + 1 == column
        assert rows
        by_arm: dict[str, set[str]] = {}
        for row in rows:
            cells = _cells(row)
            arm, kind = cells[0], cells[1]
            expected = report['arms'][arm]['by_query_kind'][kind][
                'discoverability']['stored_canonical_in_top_5_rate']
            assert cells[column] == f'{expected:.2f}'
            by_arm.setdefault(arm, set()).add(cells[column])

        # Anti-vacuity: the fixture gives every subset a DIFFERENT stored
        # rate, so a renderer sourcing all three rows from one subset prints
        # one value three times.  If the fixture is ever re-flattened, this
        # fails here rather than quietly reducing the loop above to a check
        # that `claim`'s number appears on `held_out`'s row.
        for arm, seen in by_arm.items():
            assert len(seen) == len(rows) // len(by_arm), (
                f'{arm}: by-kind rows are not distinguishable ({seen})'
            )

    def test_every_data_row_has_exactly_as_many_cells_as_its_header(self):
        """A column added to a header but not to the row builder shifts every
        cell after it one place left, and the table still LOOKS well-formed."""
        mod = _mod()
        rendered = mod.render_markdown(_report())

        decision_rows = _decision_table_rows(rendered)
        for row in decision_rows:
            assert len(_cells(row)) == len(mod.DECISION_TABLE_COLUMNS)

        header, by_kind_rows = _by_kind_table(rendered)
        for row in by_kind_rows:
            assert len(_cells(row)) == len(header)


class TestTheRunSpecificClaimIsDerivedNotTyped:
    """The reading guide's one run-specific paragraph, held to the table.

    Everything else in this renderer that states a finding about THIS run is
    computed from the report (the pin bullets are the precedent).  A hardcoded
    sentence about the numbers cannot survive the rerun this file exists to
    make cheap: the artifact is regenerated by `--json-out`/`--md-out`, the
    rates already moved once on a rerun with unchanged fixtures, and a stale
    paragraph sits three lines above the table that contradicts it — in the
    artifact gate η reads.  So the claim is derived, and these tests pin it to
    the arms rather than to any wording.
    """

    @staticmethod
    def _bullet(mod, rendered: str, arm: str) -> str:
        """The arm's bullet, located by the anchor the RENDERER emits.

        Same discipline as the pin bullets: selecting on English wording would
        break on a rewording rather than on a wrong number, which is the
        opposite of what this checks.  Exactly one, so a duplicated or
        dropped bullet fails loudly instead of silently skipping an arm.
        """
        prefix = mod.stored_gap_bullet_prefix(arm)
        bullets = [
            line for line in rendered.splitlines() if line.startswith(prefix)
        ]
        assert len(bullets) == 1, f'{arm}: {len(bullets)} bullets'
        return bullets[0]

    def _mixed_report(self):
        """A report where one arm AGREES and the rest diverge.

        `_all_arms()` makes every arm diverge, which would exercise only half
        the block and let "always print the gap sentence" pass.
        """
        mod = _mod()
        arms = _all_arms()
        discoverability = arms['status_quo']['discoverability']
        discoverability['stored_canonical_in_top_5_rate'] = \
            discoverability['canonical_in_top_5_rate']
        return mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )

    def test_every_bullet_restates_its_own_arms_two_rates(self):
        mod = _mod()
        report = self._mixed_report()

        rendered = mod.render_markdown(report)

        for arm, measurement in report['arms'].items():
            disc = measurement['discoverability']
            stored = disc['stored_canonical_in_top_5_rate']
            credited = disc['canonical_in_top_5_rate']
            assert f'{stored:.2f} vs {credited:.2f}' in self._bullet(
                mod, rendered, arm,
            ), arm

    def test_an_arm_is_called_identical_exactly_when_its_columns_agree(self):
        """The claim the old hardcoded paragraph made about `status_quo` and
        `c_peers`.  Made about whichever arms actually agree, so a rerun that
        moves one of them moves the sentence with it."""
        mod = _mod()
        report = self._mixed_report()

        rendered = mod.render_markdown(report)

        agreed = diverged = 0
        for arm, measurement in report['arms'].items():
            disc = measurement['discoverability']
            stored = disc['stored_canonical_in_top_5_rate']
            credited = disc['canonical_in_top_5_rate']
            bullet = self._bullet(mod, rendered, arm)
            if stored == credited:
                agreed += 1
                assert 'identical' in bullet, arm
            else:
                diverged += 1
                assert 'identical' not in bullet, arm
                assert f'gap of {credited - stored:.2f}' in bullet, arm
        # Anti-vacuity: both branches were actually taken.
        assert agreed and diverged, f'{agreed} agreed, {diverged} diverged'

    def test_it_does_not_attribute_agreement_to_the_absence_of_grouping(self):
        """The old paragraph's causal rule was false in general, not merely
        fragile: `apply_topic_anchor` diverges the two columns too — see
        `test_the_pin_moves_the_transformed_column_and_not_the_stored_one` —
        so the columns agree when no read-side transform CHANGED the window,
        not when the shape "runs no grouping transform"."""
        mod = _mod()

        rendered = mod.render_markdown(self._mixed_report())

        bullet = self._bullet(mod, rendered, 'status_quo')
        assert 'grouping' not in bullet
        assert 'no read-side transform changed' in bullet

    def test_an_unmeasured_column_is_reported_as_not_comparable(self):
        """`—` vs a number is not a gap, and subtracting `None` is a crash in
        the renderer that writes the operator's artifact."""
        mod = _mod()
        arms = _all_arms()
        arms['c_peers']['discoverability'][
            'stored_canonical_in_top_5_rate'] = None
        report = mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )

        bullet = self._bullet(mod, mod.render_markdown(report), 'c_peers')

        assert mod._NO_MEASUREMENT in bullet
        assert 'not comparable' in bullet

    def test_a_gap_that_rounds_to_zero_never_prints_as_the_agreement_value(self):
        """`0.00` is the value the surrounding sentence reads as "identical",
        so a real gap of 0.002 must not be allowed to print it — the same
        rule, and the same reason, as the pin column's `<0.01`."""
        mod = _mod()
        arms = _all_arms()
        disc = arms['b_grouped']['discoverability']
        disc['canonical_in_top_5_rate'] = 0.502
        disc['stored_canonical_in_top_5_rate'] = 0.5
        report = mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        )

        bullet = self._bullet(mod, mod.render_markdown(report), 'b_grouped')

        assert f'gap of {mod._PIN_RATE_UNDERFLOW}' in bullet
        assert 'gap of 0.00' not in bullet
        assert 'identical' not in bullet

    def test_the_anchor_does_not_collide_with_the_pin_bullets(self):
        """Both anchors are `- `-prefixed and name the same three shapes.  A
        collision would give `test_the_prose_bullet_restates_the_same_rate...`
        two bullets where it asserts one, breaking a test about a different
        column from a change to this one."""
        mod = _mod()

        rendered = mod.render_markdown(_report())

        for shape in mod.ARM_SHAPES:
            pin_anchor = mod.pin_bullet_prefix(shape)
            assert not mod.stored_gap_bullet_prefix(shape).startswith(pin_anchor)
            assert sum(
                1 for line in rendered.splitlines()
                if line.startswith(pin_anchor)
            ) == 1, shape


class TestTheTwoArtifactsAreWrittenAsAPair:
    """Each file is written atomically; the PAIR has to be atomic too.

    The JSON and the markdown are gate eta's decision input, and a reader has
    no way to tell they disagree.  Writing the JSON first means any raise
    inside `render_markdown` leaves a NEW json beside a STALE markdown
    describing a different run — and `render_markdown` does unguarded
    subscripting of blocks `_check_arms` never validates, so this is
    reachable rather than theoretical.
    """

    def test_a_render_failure_leaves_BOTH_artifacts_untouched(self, tmp_path):
        mod = _mod()
        json_path, md_path = tmp_path / 'r.json', tmp_path / 'r.md'
        json_path.write_text('OLD JSON', encoding='utf-8')
        md_path.write_text('OLD MD', encoding='utf-8')

        # The shape `_check_arms` does not validate: a report that survives
        # to write_artifacts but blows up mid-render.
        broken = _report()
        del broken['audit_recall']['true_dup']

        with pytest.raises(KeyError):
            mod.write_artifacts(broken, json_path, md_path)

        assert json_path.read_text(encoding='utf-8') == 'OLD JSON'
        assert md_path.read_text(encoding='utf-8') == 'OLD MD'

    def test_the_happy_path_still_writes_both(self, tmp_path):
        mod = _mod()
        json_path, md_path = tmp_path / 'r.json', tmp_path / 'r.md'

        written = mod.write_artifacts(_report(), json_path, md_path)

        assert written == (json_path, md_path)
        assert json.loads(json_path.read_text(encoding='utf-8'))['arms']
        # `# ` only: that markdown was written, not what it is titled. The
        # sibling test below asserts md == render_markdown(json), which
        # constrains the content without pinning the title prose.
        assert md_path.read_text(encoding='utf-8').startswith('# ')

    def test_the_markdown_on_disk_is_what_the_committed_json_renders_to(
        self, tmp_path
    ):
        """The same pairing invariant the committed-artifact guard asserts,
        but at the WRITE seam rather than after the fact."""
        mod = _mod()
        json_path, md_path = tmp_path / 'r.json', tmp_path / 'r.md'
        report = _report()

        mod.write_artifacts(report, json_path, md_path)

        assert md_path.read_text(encoding='utf-8') == mod.render_markdown(
            json.loads(json_path.read_text(encoding='utf-8'))
        )


class TestFixtureProvenance:
    """The blind-authoring audit trail is git history; the report cites it."""

    def test_it_reports_the_commit_that_last_touched_a_committed_fixture(self):
        provenance = _mod().fixture_provenance([ARM_CLAIMS_PATH])

        assert len(provenance) == 1
        assert provenance[0]['path'].endswith('e2_arm_claims.jsonl')
        assert len(provenance[0]['commit']) == 40  # a full sha

    def test_an_untracked_path_reports_no_commit_rather_than_a_wrong_one(self,
                                                                        tmp_path):
        untracked = tmp_path / 'never_committed.jsonl'
        untracked.write_text('{}\n')

        provenance = _mod().fixture_provenance([untracked])

        assert provenance[0]['commit'] is None

    def test_paths_are_reported_repo_relative_not_absolute(self):
        """An artifact naming `/home/<someone>/src/...` is not reproducible
        and leaks the checkout it happened to run in."""
        provenance = _mod().fixture_provenance([ARM_CLAIMS_PATH])

        assert not provenance[0]['path'].startswith('/')


# ===========================================================================
# step-19 — the live driver's wiring, exercised WITHOUT a network
# ===========================================================================
#
# The driver is the only part of this script that touches Qdrant or an
# embedder, so it is the only part the pure tests cannot reach directly.
# What CAN be reached — and is what actually goes wrong — is its WIRING:
# which collections it creates, whose config it mutates, how many queries it
# issues, and whether it cleans up when a run dies halfway.  All of that is
# asserted here against a `_FakeMem0`/`_FakeMemoryService` pair, in the
# `_install_run_doubles` spirit of test_audit_duplicate_memories.py:3296-3376,
# with `build_parser()` / `main(argv)` driven directly — never via subprocess,
# which would put the assertions on the far side of a process boundary and
# reduce them to an exit code.
#
# The single LIVE end-to-end test lives in the next section and carries its
# markers PER-TEST.  Everything here runs in the merge lane.

import asyncio  # noqa: E402
import copy  # noqa: E402


class _FakeConfig(types.SimpleNamespace):
    """A config with the leaves the driver reads, plus ``model_copy``.

    ``model_copy(deep=True)`` is the seam that keeps the driver from mutating
    the caller's config in place — so the double has to implement it for real
    rather than returning ``self``, or the test that asserts non-mutation
    would pass against a driver that mutates.
    """

    def model_copy(self, *, deep: bool = False) -> _FakeConfig:
        return copy.deepcopy(self) if deep else copy.copy(self)


def _driver_config() -> _FakeConfig:
    return _FakeConfig(
        mem0=types.SimpleNamespace(
            collection_prefix='fused',  # the DEFAULT — nothing under it is reapable
            qdrant_url='http://localhost:6333',
        ),
        embedder=types.SimpleNamespace(
            model='text-embedding-3-small',
            providers=types.SimpleNamespace(
                openai=types.SimpleNamespace(api_key='sk-fake-must-be-cleared'),
            ),
        ),
        # The SHARED durable-write-queue path, as it comes out of the schema:
        # relative, so from the repo root it resolves onto the live server's
        # own queue DB.  The driver must repoint it at a per-run temp dir
        # before `MemoryService.initialize()` can attach to it.
        queue=types.SimpleNamespace(data_dir='./data/queue'),
    )


class _FakeInstance:
    """What ``Mem0Backend._get_instance`` returns; only ``db`` is touched."""

    def __init__(self):
        # A sentinel, not None: the driver must REPLACE it with its no-op, and
        # `is not sentinel` is how the test proves the stub actually landed.
        self.db = types.SimpleNamespace(add_history=_FakeInstance._SENTINEL)

    _SENTINEL = object()


class _FakeMem0:
    """The Mem0Backend surface the driver touches, and nothing else."""

    def __init__(
        self,
        *,
        search_raises_on: int | None = None,
        add_returns_no_id_on: int | None = None,
        add_ticks: int = 1,
    ):
        self.added: list[tuple[str, str]] = []            # (project_id, content)
        self.searches: list[tuple[str, str, int]] = []    # (project_id, query, limit)
        self.instances: dict[str, _FakeInstance] = {}
        self.inflight = 0
        self.max_inflight = 0
        self.search_inflight = 0
        self.max_search_inflight = 0
        self._search_raises_on = search_raises_on
        #: 1-based ordinal of an `add` that returns a result with no id — the
        #: shape that makes `seed_arm._write` raise SeedingError.
        self._add_returns_no_id_on = add_returns_no_id_on
        #: Event-loop ticks a successful `add` parks for.  >1 keeps a sibling
        #: measurably in flight past the moment a failing peer raises, which
        #: is the only way to tell "awaited every write" from "propagated and
        #: left them running".
        self._add_ticks = add_ticks
        self._add_calls = 0
        self._stored: dict[str, list[dict]] = {}

    async def _get_instance(self, scope):
        return self.instances.setdefault(scope.project_id, _FakeInstance())

    async def add(self, content, scope, metadata=None):
        self._add_calls += 1
        ordinal = self._add_calls
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if ordinal == self._add_returns_no_id_on:
                # Fails WITHOUT suspending, so its peers are still parked below
                # when `seed_arm` sees the error.
                return {'results': []}
            # A REAL suspension point.  Without one, every scheduling policy
            # looks bounded because nothing ever overlaps; with it, an
            # unbounded gather parks all N tasks here at once and
            # `max_inflight` reaches N.
            for _ in range(self._add_ticks):
                await asyncio.sleep(0)
            bucket = self._stored.setdefault(scope.project_id, [])
            stored_id = f'{scope.project_id}-{len(bucket):04d}'
            bucket.append({'id': stored_id, 'memory': content,
                           'metadata': dict(metadata or {})})
            self.added.append((scope.project_id, content))
            return {'results': [{'id': stored_id, 'memory': content, 'event': 'ADD'}]}
        finally:
            self.inflight -= 1

    async def search(self, query, scope, limit=10, categories=None):
        self.searches.append((scope.project_id, query, limit))
        if (self._search_raises_on is not None
                and len(self.searches) >= self._search_raises_on):
            raise RuntimeError('qdrant went away mid-run')
        self.search_inflight += 1
        self.max_search_inflight = max(
            self.max_search_inflight, self.search_inflight,
        )
        try:
            # Same reason as `add`: without a real suspension point the read
            # side cannot be observed to overlap at all, and a serial fetch
            # would be indistinguishable from a bounded-concurrent one.
            await asyncio.sleep(0)
            bucket = self._stored.get(scope.project_id, [])
            return {'results': [
                {'id': item['id'], 'memory': item['memory'],
                 'metadata': item['metadata'], 'score': round(0.99 - 0.01 * rank, 4)}
                for rank, item in enumerate(bucket[:limit])
            ]}
        finally:
            self.search_inflight -= 1


class _FakeMemoryService:
    """Stand-in for MemoryService with the surface the driver actually uses."""

    instances: list = []
    search_raises_on: int | None = None
    add_returns_no_id_on: int | None = None
    add_ticks: int = 1

    def __init__(self, config):
        self.config = config
        self.mem0 = _FakeMem0(
            search_raises_on=type(self).search_raises_on,
            add_returns_no_id_on=type(self).add_returns_no_id_on,
            add_ticks=type(self).add_ticks,
        )
        self.initialized = False
        self.closed = False
        # `config` is held by REFERENCE, so reading `config.queue.data_dir`
        # after the run cannot tell "repointed before the queue was built"
        # apart from "repointed afterwards, too late".  Snapshot it at the
        # moment the real `initialize()` constructs its `DurableWriteQueue` —
        # that is when attaching to the shared path would do the damage.
        self.queue_data_dir_at_initialize: str | None = None
        type(self).instances.append(self)

    async def initialize(self):
        self.initialized = True
        self.queue_data_dir_at_initialize = self.config.queue.data_dir

    async def close(self):
        self.closed = True


class _DropRecorder:
    """Records every collection-drop the driver asks for, in order."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, names, **kwargs):
        self.calls.append(list(names))

    @property
    def dropped(self) -> set[str]:
        return {name for call in self.calls for name in call}


def _install_driver_doubles(
    monkeypatch,
    *,
    search_raises_on: int | None = None,
    add_returns_no_id_on: int | None = None,
    add_ticks: int = 1,
):
    """Patch the three seams the driver reaches through, at their source.

    The driver imports ``FusedMemoryConfig`` and ``MemoryService``
    function-locally, so patching the defining module (never a name already
    bound into this script) is what actually intercepts them.
    ``drop_collections`` is patched on the script itself because it is the
    script's OWN qdrant seam — the alternative, a fake QdrantClient, would
    test qdrant_client's import machinery rather than the driver's teardown.
    """
    import fused_memory.config.schema as schema_mod  # noqa: PLC0415
    import fused_memory.services.memory_service as service_mod  # noqa: PLC0415

    _FakeMemoryService.instances = []
    _FakeMemoryService.search_raises_on = search_raises_on
    _FakeMemoryService.add_returns_no_id_on = add_returns_no_id_on
    _FakeMemoryService.add_ticks = add_ticks
    monkeypatch.setattr(schema_mod, 'FusedMemoryConfig', _driver_config)
    monkeypatch.setattr(service_mod, 'MemoryService', _FakeMemoryService)
    drops = _DropRecorder()
    monkeypatch.setattr(_mod(), 'drop_collections', drops)
    return drops


#: A deliberately small run: two clusters and a twelve-record slab.  The
#: wiring assertions below are about counts and identities, none of which
#: depend on corpus size — and a 20-cluster/300-distractor run through the
#: doubles would spend seconds proving nothing extra.
_SMALL_RUN = {'cluster_limit': 2, 'distractor_limit': 12, 'project_suffix': 'utest'}

#: The same run with the +1-re-emission probe OFF, for the assertions below
#: whose subject IS the six-arm pipeline's shape — "three collections", "one
#: fetch per arm", "this exact teardown set".  Those stay pinned to three
#: because that is what they are about; the probed counterparts live in
#: `TestRunBakeOffRegrowthWiring`, which owns the two injected passes.  The
#: assertions that are about a SEAM rather than a count (the queue repoint,
#: the api-key clear, the seeding bound, the serial fetch) deliberately keep
#: the probe on, so they cover its passes too.
_SIX_ARM_RUN = {**_SMALL_RUN, 'regrowth': False}


class TestEphemeralCollectionIdentity:
    """A collection nobody can reap is a leak, so its NAME is a contract."""

    def test_every_arm_collection_starts_with_the_reapable_prefix(self):
        mod = _mod()
        prefix = mod.load_cleanup_script().E2_BAKEOFF_PREFIX

        collections = mod.ephemeral_collections(suffix='utest')

        assert set(collections) == set(mod.ARM_SHAPES)
        for name in collections.values():
            assert name.startswith(prefix)

    def test_the_six_variants_map_onto_exactly_three_collections(self):
        """The pin is a READ-side transform: its variants reuse their shape's
        collection, so pin-on and pin-off compare identical stored state."""
        mod = _mod()

        collections = mod.ephemeral_collections(suffix='utest')

        assert len(mod.ARM_VARIANTS) == 6
        assert len(set(collections.values())) == 3

    def test_the_project_id_is_scoped_per_xdist_worker(self, monkeypatch):
        """Two workers sharing a collection would seed each other's arms."""
        mod = _mod()

        monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw7')
        assert mod.worker_suffix() == 'gw7'
        assert 'gw7' in mod.arm_project_id('c_peers')

        monkeypatch.delenv('PYTEST_XDIST_WORKER')
        assert mod.worker_suffix()  # never empty — an unsuffixed id would collide

    def test_the_collection_name_is_what_scope_would_really_build(self):
        """Derived through Scope, not string-formatted here: Scope canonicalizes
        the project_id (lowercase, '-'->'_'), so a hand-built name would differ
        from the collection mem0 actually writes to and the reap would miss."""
        from fused_memory.models.scope import Scope  # noqa: PLC0415

        mod = _mod()

        expected = Scope(
            project_id=mod.arm_project_id('b_grouped', suffix='utest'),
        ).mem0_collection_name(mod.ephemeral_collection_prefix())

        assert mod.ephemeral_collections(suffix='utest')['b_grouped'] == expected


class TestGuardProbeSelfDrop:
    """Pure functions only — no driver, no doubles, so this class is not
    asyncio-marked."""

    def test_every_arms_replay_window_is_at_least_GUARD_TOP_K_deep(self):
        """The two invariants — drop ALL of the probe's own records, and
        replay over 5 — are enforced in different places and drifted apart
        once already (the fixed +1).

        The arithmetic half is NOT what is asserted here: `guard_fetch_limit`
        is literally `GUARD_TOP_K + len(own)`, so `depth - len(own) >= 5` is
        an identity that holds for an empty or wrong `own` too, and asserting
        it would test nothing.  What can actually regress, and did, is the
        set: `probe_own_record_ids` resolves provenance, and an id-equality
        filter silently returns nothing on the two decomposed shapes
        (bake_off_storage_shape.py:2538-2551).  So assert the set against an
        independent re-derivation from the fixtures, that it is non-empty,
        that decomposition really does split one write across several records
        — the fact that makes a fixed +1 wrong — and that the arm still holds
        a full GUARD_TOP_K window of records the probe does not own.
        """
        mod = _mod()
        clusters = mod.load_calibration_clusters()
        claims = mod.load_arm_claims()
        topics = mod.load_registry_topics()
        # Re-derived from the CLAIMS, never from `seeded.records_by_source` —
        # otherwise this compares the index against itself.
        source_of = {claim.claim_id: claim.source_memory_id for claim in claims}

        own_sizes: dict[str, dict[str, int]] = {}
        for shape in ('status_quo', 'c_peers', 'b_grouped'):
            records = mod.materialize_arm(shape, clusters, claims, topics, [])
            seeded = mod._index_arm(shape, 'p', 'c', records, claims)
            own_sizes[shape] = {}
            for cluster_id in sorted(clusters):
                probe = mod.select_probing_write(clusters[cluster_id])
                if probe is None:
                    continue
                memory_id = probe['memory_id']
                own = mod.probe_own_record_ids(seeded, memory_id)
                expected = {memory_id} | {
                    record.record_id for record in records
                    if any(
                        source_of.get(claim_id) == memory_id
                        for claim_id in record.claim_ids
                    )
                }

                assert own == expected, (
                    f'{shape}/{cluster_id}: the self-drop set is {own}, but '
                    f'the probing write really occupies {expected} in this '
                    f'arm — the difference is a free self-match'
                )
                assert own, f'{shape}/{cluster_id}: nothing to drop'
                own_sizes[shape][cluster_id] = len(own)

                # The over-fetch is worthless if the arm cannot supply a full
                # window of records the probe does NOT own.
                non_own = [r for r in records if r.record_id not in own]
                assert len(non_own) >= mod.GUARD_TOP_K, (
                    f'{shape}/{cluster_id}: only {len(non_own)} records the '
                    f'probe does not own, so the replay cannot be '
                    f'{mod.GUARD_TOP_K} deep however far the fetch reaches'
                )

        # Why the floor had to become dynamic: a decomposed arm splits one
        # write across MORE than one record, so a fixed +1 would leave those
        # arms replaying over a shorter window than the baseline.
        assert any(
            own_sizes[shape][cluster_id] > own_sizes['status_quo'][cluster_id]
            for shape in ('c_peers', 'b_grouped')
            for cluster_id in own_sizes[shape]
        ), 'no arm splits a probing write across several records'


@pytest.mark.asyncio
class TestRunBakeOffWiring:
    """What the driver does to the world, measured through doubles."""

    async def test_it_seeds_exactly_three_collections_for_six_arms(self, monkeypatch):
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        report = await mod.run_bake_off(**_SIX_ARM_RUN)

        service = _FakeMemoryService.instances[-1]
        assert len(_FakeMemoryService.instances) == 1
        assert sorted(service.mem0._stored) == sorted(
            mod.arm_project_id(shape, suffix='utest') for shape in mod.ARM_SHAPES
        )
        assert list(report['arms']) == list(mod.ARM_VARIANTS)

    async def test_the_ephemeral_prefix_is_set_on_the_config_not_just_the_project_id(
        self, monkeypatch,
    ):
        """Collections are f'{collection_prefix}_{project_id}'.  A driver that
        scoped only the project_id would write under the default 'fused'
        prefix, which the reaper does not match — a permanent leak."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        config = _FakeMemoryService.instances[-1].config
        assert config.mem0.collection_prefix == mod.ephemeral_collection_prefix()

    async def test_it_clears_the_api_key_so_a_real_embedder_is_used(self, monkeypatch):
        """A stub constant vector would make every ranking in the report
        meaningless; nulling the config key makes mem0 fall back to the real
        OPENAI_API_KEY (the probe_config recipe)."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        config = _FakeMemoryService.instances[-1].config
        assert config.embedder.providers.openai.api_key is None

    async def test_it_never_attaches_to_the_shared_durable_write_queue(
        self, monkeypatch,
    ):
        """The data-loss guard at bake_off_storage_shape.py:2947-2967.

        `config.queue.data_dir` comes out of the schema RELATIVE
        (`./data/queue`), so run from the repo root — the natural cwd — an
        un-repointed run initialises the live server's own queue DB:
        `_recover_in_flight()` flips the other process's in-flight rows back
        to pending, workers drain real user writes through the backend whose
        prefix was just repointed at `_test_e2_bakeoff_*`, and the
        `drop_collections` in the driver's `finally` then destroys them.
        Silent data loss on the production memory store, invisible in the
        artifact and the logs.

        Deleting the two repoint lines broke no test in this suite until this
        one existed, which is what `_driver_config`'s `./data/queue` was put
        there for.
        """
        import tempfile  # noqa: PLC0415

        mod = _mod()
        _install_driver_doubles(monkeypatch)
        shared = _driver_config().queue.data_dir

        await mod.run_bake_off(**_SMALL_RUN)
        await mod.run_bake_off(**_SMALL_RUN)

        # Read at `initialize()`, not after the run: the repoint has to have
        # happened BEFORE the queue is built to be worth anything, and the
        # config is held by reference so an after-the-fact read cannot tell
        # the two apart.
        attached = [
            service.queue_data_dir_at_initialize
            for service in _FakeMemoryService.instances
        ]
        assert len(attached) == 2 and all(a is not None for a in attached)

        for a in attached:
            assert a != shared, (
                f'the run attached to the shared queue path {shared!r} — a '
                f'real MemoryService would have recovered and drained the '
                f"live server's in-flight writes into this experiment's "
                f'collections'
            )
            path = Path(a)
            assert path.is_absolute(), (
                f'{a!r} is relative, so it still resolves against whatever '
                f'cwd the run happens to have — the exact failure mode'
            )
            assert path.is_relative_to(Path(tempfile.gettempdir()))
            # Outside the checkout, so no run leaves queue state in a tree
            # nothing sweeps...
            assert not path.is_relative_to(SCRIPT_PATH.parent.parent)
            # ...and the run removes it rather than accumulating scratch.
            assert not path.exists()

        # Per-RUN, so two bake-offs cannot share a queue with each other
        # either — which a fixed path outside the checkout would still allow.
        assert attached[0] != attached[1]

    async def test_a_failed_initialize_still_tears_the_run_down(self, monkeypatch):
        """The queue temp dir and the service were acquired OUTSIDE the
        `try:`/`finally:` that cleans them up, so the one failure mode the
        temp dir exists for — `initialize()` raising, which is exactly what an
        unreachable Qdrant, a missing key, or a durable-queue recovery failure
        does — leaked the directory and skipped `close()`.

        The test above proves the happy path removes the dir; that is the
        cheap half.  The module comment above the `mkdtemp` argues at length
        that this queue must never outlive the run, and a teardown that only
        holds when nothing goes wrong does not carry that argument.

        `MemoryService.close()` is safe on a partially-initialised service —
        every sub-close goes through `_safe_close`, which is time-boxed and
        logs without re-raising (memory_service.py:1192) — so the finally can
        call it unguarded rather than swallowing close failures on the happy
        path too.
        """
        import tempfile  # noqa: PLC0415

        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)

        async def _initialize_explodes(self):
            raise RuntimeError('qdrant unreachable')

        monkeypatch.setattr(_FakeMemoryService, 'initialize', _initialize_explodes)

        with pytest.raises(RuntimeError, match='qdrant unreachable'):
            await mod.run_bake_off(**_SMALL_RUN)

        # The failure has to keep propagating: a driver that swallowed it
        # would publish an artifact measured against nothing.
        assert len(_FakeMemoryService.instances) == 1
        service = _FakeMemoryService.instances[-1]

        queue_dir = Path(service.config.queue.data_dir)
        assert queue_dir.is_relative_to(Path(tempfile.gettempdir()))
        assert not queue_dir.exists(), (
            f'{queue_dir} survived a failed run — the queue state the module '
            f'comment says must never outlive the run now outlives it, and '
            f'accumulates one directory per failed attempt'
        )
        assert service.closed, (
            'close() was skipped, so whatever initialize() managed to build '
            'before raising is never torn down'
        )
        # The ephemeral collections are dropped on the way out too: a failure
        # after the pre-clean drop must not leave a half-seeded arm behind for
        # the next run to measure.
        assert drops.calls, 'nothing was dropped on the failure path'

    async def test_it_copies_the_config_rather_than_mutating_the_callers(
        self, monkeypatch,
    ):
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        caller_config = _driver_config()

        await mod.run_bake_off(config=caller_config, **_SMALL_RUN)

        assert caller_config.mem0.collection_prefix == 'fused'
        assert caller_config.embedder.providers.openai.api_key == 'sk-fake-must-be-cleared'

    async def test_it_stubs_the_xdist_contended_history_writer(self, monkeypatch):
        """mem0's SQLite history writer is process-shared and contended (and
        read-only in the sandbox).  It is not the question under test, and its
        failure would mask the one that is."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        instances = _FakeMemoryService.instances[-1].mem0.instances
        # Every seeded project, the two INJECTED passes included: they go
        # through the same `seed_arm`, so an unstubbed instance there is the
        # same contended write, in a collection nobody was watching.
        assert len(instances) == len(mod.ARM_SHAPES) + len(mod.REGROWTH_MODES)
        for instance in instances.values():
            assert instance.db.add_history is not _FakeInstance._SENTINEL
            assert instance.db.add_history('anything', keyword=1) is None

    async def test_the_distractor_slab_is_seeded_into_every_arm(self, monkeypatch):
        """The contamination floor is a controlled variable: an arm that got a
        smaller slab would be ranking against a thinner field and would win for
        a reason the decision table does not name."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        slab = mod.load_distractor_slab()[:_SMALL_RUN['distractor_limit']]
        slab_contents = {d.content for d in slab}

        await mod.run_bake_off(**_SMALL_RUN)

        added = _FakeMemoryService.instances[-1].mem0.added
        by_project: dict[str, set[str]] = {}
        for project_id, content in added:
            by_project.setdefault(project_id, set()).add(content)

        # Including the two injected passes: `regrowth_corpus` is the ratified
        # arm's records PLUS the re-emissions, so a pass that lost the slab
        # would be ranking against a thinner field than its own baseline —
        # and the delta would report the missing contamination, not the
        # re-emission.
        assert len(by_project) == len(mod.ARM_SHAPES) + len(mod.REGROWTH_MODES)
        for contents in by_project.values():
            assert slab_contents <= contents

    async def test_seeding_is_bounded_rather_than_one_unbounded_gather(
        self, monkeypatch,
    ):
        """A gather over the whole slab opens hundreds of concurrent embedding
        requests at once — rate-limited into retries at best, and a run whose
        wall clock is set by the throttler rather than the experiment."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        mem0 = _FakeMemoryService.instances[-1].mem0
        smallest_arm = min(
            len(bucket) for bucket in mem0._stored.values()
        )
        assert smallest_arm > mod.SEED_CONCURRENCY  # the bound has to actually bite
        assert mem0.max_inflight <= mod.SEED_CONCURRENCY

    async def test_fetching_stays_serial_because_a_swallowed_read_timeout_is_silent(
        self, monkeypatch,
    ):
        """The read side is the one half that must NOT be made concurrent.

        Bounding it at SEED_CONCURRENCY is the obvious efficiency win — 753
        independent, order-insensitive round trips across the three arms — and
        it was implemented and measured. The live run aborted partway through
        the second arm with `MeasurementError: ... returned no results for a
        10-limit query`. `Mem0Client.search` SWALLOWS its read timeout (logs,
        returns `{}`), so on the read side concurrency converts a latency win
        into a silently-empty ranking; `add` propagates its timeout instead,
        which is why the write side keeps its bound.

        Asserted as the OBSERVABLE property — one search in flight at a time —
        rather than as "no semaphore exists", so a future re-attempt has to
        confront this test rather than route around it.
        """
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        mem0 = _FakeMemoryService.instances[-1].mem0
        per_arm = len(mem0.searches) / len(mod.ARM_SHAPES)
        assert per_arm > 1  # there is something to overlap, so this can fail
        assert mem0.max_search_inflight == 1

    async def test_a_failed_seed_leaves_no_write_in_flight_past_teardown(
        self, monkeypatch,
    ):
        """An orphan write can recreate the collection teardown just dropped.

        A bare `gather` propagates the first SeedingError IMMEDIATELY and
        leaves its siblings running.  Control unwinds to `run_bake_off`'s
        `finally`, which closes the service and then drops the ephemeral
        collections — and a write still in flight inside mem0's client can
        land AFTER the drop, recreating the collection under the
        `_test_e2_bakeoff` prefix and leaking exactly what the module's
        ISOLATION block says cannot leak.

        Asserted as "nothing is in flight when the raise arrives", which is
        the property that makes the ordering safe, rather than as "a
        TaskGroup was used", which is an implementation.  The exception type
        is pinned too: callers catch SeedingError, and an ExceptionGroup
        wrapper would be a silent contract change.
        """
        mod = _mod()
        drops = _install_driver_doubles(
            monkeypatch, add_returns_no_id_on=1, add_ticks=5,
        )

        with pytest.raises(mod.SeedingError):
            await mod.run_bake_off(**_SMALL_RUN)

        mem0 = _FakeMemoryService.instances[-1].mem0
        assert mem0.inflight == 0, (
            f'{mem0.inflight} seeding write(s) still in flight when teardown '
            f'dropped the collections'
        )
        # The bound had to actually bite, or "nothing in flight" is vacuous:
        # a run whose every write completed before the raise proves nothing.
        assert mem0.max_inflight > 1
        # Teardown really did run underneath the raise.
        assert len(drops.calls) == 2

    async def test_the_pin_variants_reuse_their_shapes_hits_instead_of_requerying(
        self, monkeypatch,
    ):
        """Pin-on and pin-off must run over the SAME ranked list, or the
        comparison is two ANN draws rather than a controlled A/B — and the run
        pays for six arms' worth of embeddings to answer a four-arm question."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        searches = _FakeMemoryService.instances[-1].mem0.searches
        issued = [(project_id, query) for project_id, query, _ in searches]
        assert len(issued) == len(set(issued))          # no query issued twice
        # One project per seeded corpus, the two injected passes included:
        # the probe's THREE read arms are scored off one fetch each for the
        # same reason the pin variants are — three ANN draws would make the
        # read-arm comparison uncontrolled and triple its cost.
        assert len({project_id for project_id, _ in issued}) == (
            len(mod.ARM_SHAPES) + len(mod.REGROWTH_MODES)
        )

    async def test_the_guard_probe_over_fetches_to_cover_its_own_removal(
        self, monkeypatch,
    ):
        """The probing write's own record has to leave its own guard window:
        in the real timeline that write had not landed when the guard ran, and
        arm (a) stores it verbatim — leaving it in would hand the BASELINE a
        free ~1.0 self-match the peer arms structurally cannot get, and the
        table would read that as arm (a) having the better guard.  Dropping it
        costs a slot, so the fetch is deeper — by the number of records the
        probe's own content occupies in THIS arm, which the decomposed arms
        split into several.  A fixed +1 would leave them replaying over a
        2-record window while the baseline replayed over 5.  The replay itself
        still happens at GUARD_TOP_K, which `guard_adequacy` enforces on its
        own (server/tools.py:1556 runs production's pre-check at limit=5)."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        limits = {limit for _, _, limit in _FakeMemoryService.instances[-1].mem0.searches}
        assert mod.GUARD_FETCH_LIMIT == mod.GUARD_TOP_K + 1     # the FLOOR
        assert mod.guard_fetch_limit(set()) == mod.GUARD_TOP_K
        assert mod.guard_fetch_limit({'a', 'b', 'c'}) == mod.GUARD_TOP_K + 3
        # Every probe limit clears the floor, and at least one arm fetched
        # DEEPER than the floor — otherwise the dynamic depth is untested.
        probe_limits = limits - {mod.DEFAULT_SEARCH_LIMIT}
        assert probe_limits
        assert min(probe_limits) >= mod.GUARD_FETCH_LIMIT
        assert max(probe_limits) > mod.GUARD_FETCH_LIMIT

    async def test_the_probing_write_is_never_its_own_guard_match(
        self, monkeypatch,
    ):
        """The over-fetched slot is worthless unless the self-hit is actually
        dropped, so assert the drop and not merely the fetch depth."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        clusters = mod.load_calibration_clusters()
        probe_by_cluster = {}
        for cluster_id in sorted(clusters)[:_SMALL_RUN['cluster_limit']]:
            probe = mod.select_probing_write(clusters[cluster_id])
            if probe is not None:
                probe_by_cluster[cluster_id] = probe['memory_id']
        assert probe_by_cluster  # the subset really does contain a probeable cluster

        captured: list[dict[str, list[str]]] = []
        real_fetch = mod.fetch_arm

        async def _capture(backend, seeded, queries, probes, **kwargs):
            fetched = await real_fetch(backend, seeded, queries, probes, **kwargs)
            captured.append({
                cluster_id: [hit.record.record_id for hit in hits]
                for cluster_id, hits in fetched['probes'].items()
            })
            return fetched

        monkeypatch.setattr(mod, 'fetch_arm', _capture)
        await mod.run_bake_off(**_SMALL_RUN)

        # Per seeded corpus, injected passes included: those carry the same
        # `status_quo`-free flat peers, but the drop is provenance-based and
        # has to hold for a corpus the arm loop never built.
        assert len(captured) == len(mod.ARM_SHAPES) + len(mod.REGROWTH_MODES)
        # status_quo stores the alpha record verbatim under its own memory_id,
        # so this is the arm where a self-hit is even possible.  Checked
        # per-cluster: another cluster's probe record is a legitimate hit.
        for arm in captured:
            for cluster_id, probe_id in probe_by_cluster.items():
                assert probe_id not in arm[cluster_id]

    async def test_every_collection_is_dropped_before_and_after_the_run(
        self, monkeypatch,
    ):
        """Before AND after: a swallowed teardown then self-heals on the next
        run instead of poisoning it with a half-seeded arm."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        expected = set(mod.ephemeral_collections(suffix='utest').values())

        await mod.run_bake_off(**_SIX_ARM_RUN)

        assert len(drops.calls) == 2
        assert set(drops.calls[0]) == expected
        assert set(drops.calls[-1]) == expected

    async def test_the_collections_are_dropped_even_when_a_query_raises(
        self, monkeypatch,
    ):
        """The failure that leaks is the mid-run one; a `finally` is the only
        thing that reaches it."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch, search_raises_on=2)
        expected = set(mod.ephemeral_collections(suffix='utest').values())

        with pytest.raises(RuntimeError, match='qdrant went away'):
            await mod.run_bake_off(**_SIX_ARM_RUN)

        assert set(drops.calls[-1]) == expected
        assert _FakeMemoryService.instances[-1].closed is True

    async def test_the_report_it_returns_is_the_one_build_report_validated(
        self, monkeypatch,
    ):
        """Not a second, looser assembly path: the completeness check that
        refuses to publish a partial decision table has to be on THIS road."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        report = await mod.run_bake_off(**_SMALL_RUN)

        assert report['schema_version'] == mod.REPORT_SCHEMA_VERSION
        assert set(report) == {
            'schema_version', 'protocol', 'arms', 'audit_recall',
            # v3: always present, an explicit None when the probe did not run.
            'regrowth',
        }
        for arm in mod.ARM_VARIANTS:
            for metric in mod._REQUIRED_ARM_METRICS:
                assert metric in report['arms'][arm]
        assert report['protocol']['distractor_slab_size'] == 12
        assert report['protocol']['embedder_model'] == 'text-embedding-3-small'


# ===========================================================================
# step-13 — the probe's driver, CLI, cache and teardown wiring
# ===========================================================================
#
# Everything the regrowth block needs that is NOT arithmetic: two more
# ephemeral collections that the SAME teardown has to reach, two more fetch
# passes the cache has to be able to describe and refuse when stale, and a
# CLI switch whose default is "probe".
#
# The failure this section exists to catch is a leak, not a wrong number.  A
# probe that seeds an injected corpus into a collection the `finally` does
# not name leaves a live collection behind on every run, and the report it
# returns looks perfect.
#
# Pins NO metric value (G6).


def _regrowth_collection_names(mod, *, suffix: str) -> set[str]:
    """The two injected passes' collections, derived exactly as the driver must."""
    return set(mod.ephemeral_collections(
        shapes=tuple(mod.regrowth_pass_key(mode) for mode in mod.REGROWTH_MODES),
        suffix=suffix,
    ).values())


class TestRegrowthPassKey:
    """The pass key names a COLLECTION, so it is a reapability contract."""

    def test_each_mode_gets_its_own_key_and_none_collides_with_an_arm(self):
        mod = _mod()

        keys = [mod.regrowth_pass_key(mode) for mode in mod.REGROWTH_MODES]

        assert len(set(keys)) == len(mod.REGROWTH_MODES)
        assert set(keys).isdisjoint(mod.ARM_SHAPES)

    def test_the_key_survives_scopes_canonicalization_unchanged(self):
        """`arm_project_id` interpolates this into a project id that `Scope`
        lowercases and `-`->`_`s.  A key that canonicalized DIFFERENTLY would
        name a collection under one spelling and be swept under another —
        i.e. a collection the teardown cannot find, which is the leak."""
        from fused_memory.models.scope import Scope  # noqa: PLC0415

        mod = _mod()

        for mode in mod.REGROWTH_MODES:
            key = mod.regrowth_pass_key(mode)
            assert key == key.lower()
            assert '-' not in key and '@' not in key
            project_id = mod.arm_project_id(key, suffix='utest')
            assert Scope(project_id=project_id).project_id == project_id

    def test_the_key_names_the_mode_it_measures(self):
        """Two collections whose names do not say which mode they hold make
        a leaked one unattributable in the reaper's output."""
        mod = _mod()

        for mode in mod.REGROWTH_MODES:
            assert mode in mod.regrowth_pass_key(mode)

    def test_the_two_pass_collections_are_distinct_and_disjoint_from_the_arms(self):
        mod = _mod()
        arms = set(mod.ephemeral_collections(suffix='utest').values())

        passes = _regrowth_collection_names(mod, suffix='utest')

        assert len(passes) == len(mod.REGROWTH_MODES)
        assert passes.isdisjoint(arms)
        prefix = mod.load_cleanup_script().E2_BAKEOFF_PREFIX
        for name in passes:
            # Under the reapable prefix or the sweep never finds it, which is
            # the same leak the arm collections' identity test guards.
            assert name.startswith(prefix)


@pytest.mark.asyncio
class TestRunBakeOffRegrowthWiring:
    """The probe's effect on the world, measured through the same doubles."""

    async def test_the_default_run_seeds_both_injected_passes(self, monkeypatch):
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        await mod.run_bake_off(**_SMALL_RUN)

        seeded_projects = set(_FakeMemoryService.instances[-1].mem0._stored)
        assert seeded_projects == {
            mod.arm_project_id(shape, suffix='utest') for shape in mod.ARM_SHAPES
        } | {
            mod.arm_project_id(mod.regrowth_pass_key(mode), suffix='utest')
            for mode in mod.REGROWTH_MODES
        }

    async def test_the_default_run_returns_a_complete_regrowth_block(
        self, monkeypatch,
    ):
        """Complete, not merely present: `_check_regrowth` is the gate, and
        this asserts the DRIVER hands it something that passes."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        report = await mod.run_bake_off(**_SMALL_RUN)

        regrowth = report['regrowth']
        assert regrowth is not None
        assert regrowth['shape'] == mod.REGROWTH_SHAPE
        assert list(regrowth['modes']) == list(mod.REGROWTH_MODES)
        assert list(regrowth['read_arms']) == list(mod.REGROWTH_READ_ARMS)
        for mode in mod.REGROWTH_MODES:
            for arm in mod.REGROWTH_READ_ARMS:
                assert set(regrowth['after'][mode][arm]) == set(
                    mod._regrowth_metric_keys()
                )
                assert set(regrowth['deltas'][mode][arm]) == set(
                    mod._regrowth_metric_keys()
                )

    async def test_the_probe_respects_the_cluster_subset(self, monkeypatch):
        """`--clusters N` has to filter the injections too, or a smoke run
        injects re-emissions for topics whose claims it never seeded — and
        cross-validation, which runs over the SUBSET, would reject them."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        report = await mod.run_bake_off(**_SMALL_RUN)

        assert report['regrowth']['topics_injected'] == 2
        assert report['protocol']['regrowth_injections_measured'] == 2

    async def test_the_probe_records_that_it_ran_in_the_protocol_block(
        self, monkeypatch,
    ):
        """Same reason `clusters_measured` is there: a reader holding the
        artifact must not have to infer the probe's coverage from whether a
        table looks populated."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        report = await mod.run_bake_off(**_SMALL_RUN)

        assert report['protocol']['regrowth_probed'] is True
        assert report['protocol']['regrowth_injections_measured'] > 0

    async def test_the_injected_collections_are_dropped_before_and_after(
        self, monkeypatch,
    ):
        """THE LEAK.  A probe that seeds into a collection the `finally` does
        not name leaves it live on every run, and the report looks perfect."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        expected = set(
            mod.ephemeral_collections(suffix='utest').values()
        ) | _regrowth_collection_names(mod, suffix='utest')

        await mod.run_bake_off(**_SMALL_RUN)

        assert len(drops.calls) == 2
        assert set(drops.calls[0]) == expected
        assert set(drops.calls[-1]) == expected

    async def test_the_injected_collections_are_dropped_when_a_query_raises(
        self, monkeypatch,
    ):
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch, search_raises_on=2)
        expected = set(
            mod.ephemeral_collections(suffix='utest').values()
        ) | _regrowth_collection_names(mod, suffix='utest')

        with pytest.raises(RuntimeError, match='qdrant went away'):
            await mod.run_bake_off(**_SMALL_RUN)

        assert set(drops.calls[-1]) == expected

    async def test_disabling_the_probe_creates_no_collection_and_reads_no_fixture(
        self, monkeypatch,
    ):
        """`regrowth=False` is a real skip, not a measured-then-discarded
        pass: it must cost no seeding, name no extra collection, and open no
        injection fixture."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        opened: list = []
        real_loader = mod.load_regrowth_injections
        monkeypatch.setattr(mod, 'load_regrowth_injections', lambda *a, **k: (
            opened.append(a), real_loader(*a, **k))[1])
        passes = _regrowth_collection_names(mod, suffix='utest')

        report = await mod.run_bake_off(regrowth=False, **_SMALL_RUN)

        assert opened == []
        assert drops.dropped.isdisjoint(passes)
        assert set(_FakeMemoryService.instances[-1].mem0._stored) == {
            mod.arm_project_id(shape, suffix='utest') for shape in mod.ARM_SHAPES
        }
        assert report['regrowth'] is None
        assert report['protocol']['regrowth_probed'] is False
        assert report['protocol']['regrowth_injections_measured'] == 0

    async def test_a_subset_that_retains_no_injection_is_not_published_as_probed(
        self, monkeypatch,
    ):
        """The live and replay paths must decide this from the SAME value.

        `--clusters N` filters the injection slab, so a subset can in
        principle retain no injected topic.  The live driver used to publish
        `regrowth_probed: true` for that run, with a block reading
        `topics_injected: 0, injections_per_topic: 1` — a hard-coded 1
        describing zero topics — while `_replay_bake_off`, which builds a
        block only when the post-subset list is non-empty, published
        `regrowth_probed: false` and a null block for the same run.  Both
        paths now key on the post-subset list.

        The loader and its cross-validator are doubled because the case's
        SUBJECT is the driver's predicate, not the committed fixture: the
        real slab covers every topic by construction (the validator enforces
        exactly one per topic over the full claim set), so the state under
        test is unreachable through it.
        """
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        monkeypatch.setattr(mod, 'load_regrowth_injections', lambda *a, **k: [])
        monkeypatch.setattr(
            mod, 'cross_validate_regrowth_injections', lambda **k: None,
        )
        passes = _regrowth_collection_names(mod, suffix='utest')

        report = await mod.run_bake_off(**_SMALL_RUN)

        assert report['regrowth'] is None
        assert report['protocol']['regrowth_probed'] is False
        assert report['protocol']['regrowth_injections_measured'] == 0
        # And nothing was seeded or reaped for a pass that measured nothing.
        assert drops.dropped.isdisjoint(passes)
        assert set(_FakeMemoryService.instances[-1].mem0._stored) == {
            mod.arm_project_id(shape, suffix='utest') for shape in mod.ARM_SHAPES
        }

    async def test_a_probe_less_run_claims_no_provenance_for_the_injection_fixture(
        self, monkeypatch,
    ):
        """Provenance for a file the run never opened is a false audit trail."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        relative = mod._repo_relative(mod.DEFAULT_REGROWTH_INJECTION_PATH)

        skipped = await mod.run_bake_off(regrowth=False, **_SMALL_RUN)
        probed = await mod.run_bake_off(**_SMALL_RUN)

        assert relative not in [
            row['path'] for row in skipped['protocol']['fixtures']
        ]
        assert relative in [
            row['path'] for row in probed['protocol']['fixtures']
        ]

    async def test_the_probe_does_not_disturb_the_six_arm_rows(self, monkeypatch):
        """The decision table is the ratified artifact; the probe rides
        alongside it and must not move a single one of its cells."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        without = await mod.run_bake_off(regrowth=False, **_SMALL_RUN)
        with_probe = await mod.run_bake_off(**_SMALL_RUN)

        assert with_probe['arms'] == without['arms']


class TestRegrowthFetchCache:
    """The two injected passes are cacheable, and refusably stale."""

    def _cache_inputs(self, mod):
        """Two hand-built pass corpora and their `SeededArm`s, keyed by pass.

        Built through `_index_arm` with `seeded.shape` left at
        `REGROWTH_SHAPE`: the pass key names the CACHE slot and the
        collection, never the read behaviour, so `read_path` must still see
        `c_peers`.
        """
        seeded, records = {}, {}
        for mode in mod.REGROWTH_MODES:
            key = mod.regrowth_pass_key(mode)
            corpus = list(_regrowth_arm(mode))
            records[key] = corpus
            seeded[key] = mod._index_arm(
                mod.REGROWTH_SHAPE, f'proj_{key}', f'coll_{key}',
                corpus, _committed_inputs()['claims'],
            )
        return seeded, records

    def _dump(self, mod, path, records, *, fixtures=None):
        hit = mod.ScoredHit
        arms = {
            key: {
                'queries': {'q1': [
                    hit(record=corpus[0], relevance_score=0.9),
                    hit(record=corpus[-1], relevance_score=0.5),
                ]},
                'probes': {},
            }
            for key, corpus in records.items()
        }
        return mod.dump_fetches(path, arms, provenance=mod.fetch_cache_provenance(
            records_by_shape=records,
            fixtures=list(fixtures if fixtures is not None else [
                ALPHA_FIXTURE_PATH, REGISTRY_PATH, ARM_CLAIMS_PATH,
                QUERY_SET_PATH, DISTRACTOR_SLAB_PATH,
                mod.DEFAULT_REGROWTH_INJECTION_PATH,
            ]),
            search_limit=10, guard_threshold=0.85,
            embedder_model='text-embedding-3-small',
        ))

    def test_a_document_carries_the_arm_keys_and_both_pass_keys(self, tmp_path):
        mod = _mod()
        seeded, records = self._cache_inputs(mod)
        arm_records = {shape: list(_arm(shape)) for shape in mod.ARM_SHAPES}

        path = self._dump(mod, tmp_path / 'cache.json', {**arm_records, **records})

        doc = json.loads(path.read_text())
        assert set(doc['arms']) == set(mod.ARM_SHAPES) | set(records)
        assert set(doc['provenance']['corpus_fingerprints']) == (
            set(mod.ARM_SHAPES) | set(records)
        )
        loaded = mod.load_fetches(path, seeded)
        assert set(loaded) == set(records)

    def test_a_caller_asking_only_for_the_arms_still_loads(self, tmp_path):
        """`load_fetches` iterates the CALLER's shapes, so the extra pass keys
        are merely a wider cache.  This is what keeps 4004's committed
        `e2_fetch_cache.json` — read by `read_transform_selection` — loadable
        after this task widens the dump."""
        mod = _mod()
        _, pass_records = self._cache_inputs(mod)
        arm_records = {shape: list(_arm(shape)) for shape in mod.ARM_SHAPES}
        path = self._dump(
            mod, tmp_path / 'cache.json', {**arm_records, **pass_records},
        )
        arm_seeded = {
            shape: mod._index_arm(
                shape, f'p_{shape}', f'c_{shape}', arm_records[shape],
                _committed_inputs()['claims'],
            )
            for shape in mod.ARM_SHAPES
        }

        loaded = mod.load_fetches(path, arm_seeded)

        assert set(loaded) == set(mod.ARM_SHAPES)

    def test_a_pass_replayed_over_the_other_modes_corpus_is_refused_by_name(
        self, tmp_path,
    ):
        """The two injected corpora differ ONLY in a metadata key, so a
        crossed cache still loads and every ranking still joins.  The
        fingerprint is the only thing standing between that and a stamped
        measurement published as an unstamped one."""
        mod = _mod()
        seeded, records = self._cache_inputs(mod)
        unstamped = mod.regrowth_pass_key('unstamped')
        stamped = mod.regrowth_pass_key('stamped')
        # Dumped with the two corpora SWAPPED under each other's key.
        path = self._dump(mod, tmp_path / 'cache.json', {
            unstamped: records[stamped], stamped: records[unstamped],
        })

        with pytest.raises(mod.FetchCacheError) as excinfo:
            mod.load_fetches(path, {unstamped: seeded[unstamped]})

        assert unstamped in str(excinfo.value)

    def _injection_fixture_copy(self, tmp_path) -> Path:
        """A writable copy of the committed injection slab.

        Copied rather than edited in place for the reason every other
        digest-drift test in this repo copies: mutating the real fixture
        would fail every other test in this module and leave the tree dirty.
        """
        target = tmp_path / REGROWTH_INJECTION_PATH.name
        target.write_bytes(REGROWTH_INJECTION_PATH.read_bytes())
        return target

    def test_replaying_a_pass_verifies_the_injection_fixtures_digest(
        self, tmp_path,
    ):
        """The corpus fingerprint cannot catch this.

        It is taken over the MATERIALIZED records, and the driver decides
        which injections exist by reading the fixture — so an edit that
        rewrites an injection's `text` while keeping its `injection_id`
        leaves every id intact and the cached rankings still join cleanly.
        The digest is the only thing between that and a measurement of a
        re-emission nobody wrote.
        """
        mod = _mod()
        seeded, records = self._cache_inputs(mod)
        fixture = self._injection_fixture_copy(tmp_path)
        path = self._dump(
            mod, tmp_path / 'cache.json', records, fixtures=[fixture],
        )
        rows = [
            json.loads(line)
            for line in fixture.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        rows[0]['text'] = rows[0]['text'] + ' and something else entirely'
        fixture.write_text(
            '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8',
        )

        # The premise, asserted rather than described: without the guard the
        # stale cache loads clean, because nothing else in the pipeline can
        # see the edit.
        assert mod.load_fetches(path, seeded)

        with pytest.raises(mod.FetchCacheError) as excinfo:
            mod.load_fetches(path, seeded, expect_fixtures=[fixture])

        assert REGROWTH_INJECTION_PATH.name in str(excinfo.value)

    def test_an_unedited_injection_slab_replays_clean(self, tmp_path):
        """The converse, so the refusal above is not vacuously always-on."""
        mod = _mod()
        seeded, records = self._cache_inputs(mod)
        fixture = self._injection_fixture_copy(tmp_path)
        path = self._dump(
            mod, tmp_path / 'cache.json', records, fixtures=[fixture],
        )

        loaded = mod.load_fetches(path, seeded, expect_fixtures=[fixture])

        assert set(loaded) == set(records)

    def test_the_five_original_fixtures_are_checked_separately_from_the_sixth(
        self, tmp_path,
    ):
        """A cache dumped WITHOUT the injection digest — 4004's committed one —
        must still satisfy a five-fixture check.  Folding the sixth into the
        same list would make `e2_fetch_cache.json` unloadable for the E2 arms
        and break `read_transform_selection`, which this task does not touch."""
        mod = _mod()
        arm_records = {shape: list(_arm(shape)) for shape in mod.ARM_SHAPES}
        five = [ALPHA_FIXTURE_PATH, REGISTRY_PATH, ARM_CLAIMS_PATH,
                QUERY_SET_PATH, DISTRACTOR_SLAB_PATH]
        path = self._dump(
            mod, tmp_path / 'cache.json', arm_records, fixtures=five,
        )
        arm_seeded = {
            shape: mod._index_arm(
                shape, f'p_{shape}', f'c_{shape}', arm_records[shape],
                _committed_inputs()['claims'],
            )
            for shape in mod.ARM_SHAPES
        }

        loaded = mod.load_fetches(path, arm_seeded, expect_fixtures=five)

        assert set(loaded) == set(mod.ARM_SHAPES)
        with pytest.raises(mod.FetchCacheError):
            mod.load_fetches(
                path, arm_seeded,
                expect_fixtures=[*five, mod.DEFAULT_REGROWTH_INJECTION_PATH],
            )


# ===========================================================================
# step-26 — the equal-window discipline
# ===========================================================================
#
# THE BUG THIS SECTION PINS.  `apply_topic_anchor` APPENDS: a full 5-hit
# window plus one pinned canonical is six records.  `measure_arm` then scored
# every metric at `k = len(window)`, so the +pin variants were measured over a
# SIX-record window while their pin-off twins were measured over five.  The
# pin column did not report "the pin helped"; it reported "the pin was given
# a bigger budget".  In the first committed artifact that showed up as
# c_peers 0.504 -> c_peers+pin 0.992 on canonical-in-top-5 and 1180 -> 1290
# tokens/query — a discoverability "win" bought entirely with extra results.
#
# The fix is a post-transform truncation in `read_path` plus literal-k
# scoring in `measure_arm`.  It is deliberately NOT a change to
# `apply_topic_anchor`, which stays additive-and-never-subtractive per PRD D1
# — the window budget is the READER's, so it belongs at the read path.
#
# Grouping is untouched by this: it SHRINKS the window, so it keeps its
# legitimate token win, and a pin that lands in the headroom grouping freed
# still survives.  That is the pin's real win, and it is the one the decision
# table should show.


def _rec(record_id, **kwargs):
    """Just the `ArmRecord` half of `_hit` — this section never needs the flag."""
    record, _ = _hit(record_id, **kwargs)
    return record


def _sh(record, score=0.5):
    return _mod().ScoredHit(record=record, relevance_score=score)


def _seeded(shape, records, *, canonical_by_topic=None, contested_ids=None,
            canonical_by_cluster=None, siblings_by_cluster=None,
            records_by_source=None):
    """A `SeededArm` over hand-built records — no store, no network."""
    mod = _mod()
    return mod.SeededArm(
        shape=shape,
        project_id=f'e2_{shape}_utest',
        collection=f'e2_{shape}_utest',
        records=list(records),
        by_stored_id={},
        records_by_id={record.record_id: record for record in records},
        canonical_by_topic=canonical_by_topic or {},
        contested_ids=contested_ids or set(),
        canonical_by_cluster=canonical_by_cluster or {},
        siblings_by_cluster=siblings_by_cluster or {},
        records_by_source=records_by_source or {},
    )


def _full_window_arm(shape='c_peers', *, n=12, topic='t'):
    """An arm whose fetch fills any window, with a pinnable canonical OUTSIDE it.

    The canonical is deliberately never a hit: that is the case where the pin
    has something to add, and therefore the case where an unequal window would
    show up as a fake win.
    """
    canonical = _rec('canon', topic=topic, canonical=True, claim_ids=['k-canon'])
    hits = [
        _sh(_rec(f'p{i}', topic=topic, claim_ids=[f'k{i}']), 0.9 - i / 100.0)
        for i in range(n)
    ]
    seeded = _seeded(
        shape,
        [canonical, *(hit.record for hit in hits)],
        canonical_by_topic={topic: canonical},
        canonical_by_cluster={'c1': 'canon'},
        siblings_by_cluster={'c1': {hit.record.record_id for hit in hits}},
    )
    return seeded, hits


def _query(query_id='q1', *, topic='t', expects=('k0',), cluster_id='c1',
           kind='claim', held_out=False):
    return _mod().Query(
        query_id=query_id,
        kind=kind,
        text='does the pin change the window?',
        topic=topic,
        cluster_id=cluster_id,
        expects_claim_ids=list(expects),
        held_out=held_out,
    )


#: A `(name, encode)` estimator that is trivially checkable by hand.
_CHARS = ('injected:chars', len)


def _measure(seeded, hits, *, pin, queries=None, probes=(), limit=10, **kwargs):
    return _mod().measure_arm(
        seeded,
        {'queries': {'q1': hits}, 'probes': {'c1': hits}},
        pin=pin,
        **kwargs,
        queries=list(queries if queries is not None else [_query()]),
        probes=list(probes),
        estimator=_CHARS,
        guard_threshold=0.92,
        limit=limit,
    )


class TestByQueryKindSplitsTheQueriesItClaimsToSplit:
    """`by_query_kind` is a REQUIRED report block that had no behavioral test.

    It is why the report can say anything at all about the distinction
    eval-design §5 E2 draws: pooled into one mean, a shape that wins on claim
    queries while losing on topic phrasings is indistinguishable from one that
    ties on both.  It is required (`_REQUIRED_ARM_METRICS`), rendered as its
    own markdown table, and computed by a real split in `measure_arm` — but
    it appeared in this file only inside the synthetic `_arm_measurement()`
    builder, where it was constructed and never asserted on.
    """

    @staticmethod
    def _by_kind(queries):
        seeded, hits = _full_window_arm('status_quo')
        return _mod().measure_arm(
            seeded,
            {'queries': {q.query_id: hits for q in queries}, 'probes': {'c1': hits}},
            pin=False,
            queries=list(queries),
            probes=[],
            estimator=_CHARS,
            guard_threshold=0.92,
            limit=10,
        )['by_query_kind']

    def test_each_kinds_query_count_is_the_input_split(self):
        by_kind = self._by_kind([
            _query('q1', kind='claim'),
            _query('q2', kind='claim'),
            _query('q3', kind='topic_phrasing'),
            _query('q4', kind='topic_phrasing', held_out=True),
        ])

        assert by_kind['claim']['queries'] == 2
        assert by_kind['topic_phrasing']['queries'] == 2
        assert by_kind['held_out']['queries'] == 1

    def test_held_out_is_a_subset_of_its_kind_and_not_a_third_kind(self):
        """The markdown states this in prose, and the split has to honour it.

        A held-out query is still a `topic_phrasing` query: if the split
        moved it out of its kind instead of also counting it under
        `held_out`, the topic row would silently shed the very phrasings that
        measure generalisation, and the two rows would no longer sum to
        anything a reader could reason about.
        """
        held_out_only = self._by_kind([
            _query('q1', kind='topic_phrasing', held_out=True),
        ])

        assert held_out_only['topic_phrasing']['queries'] == 1
        assert held_out_only['held_out']['queries'] == 1
        # Same single query on both rows, so the metrics must agree exactly.
        assert held_out_only['held_out'] == held_out_only['topic_phrasing']

    def test_an_unasked_subset_reports_none_rather_than_a_measured_zero(self):
        """`queries: 0` with `0.0` beside it would read as "measured, scored
        nothing" — the same lie the `—` cell exists to prevent in the table."""
        by_kind = self._by_kind([_query('q1', kind='claim')])

        empty = by_kind['topic_phrasing']
        assert empty['queries'] == 0
        assert empty['claim_recall']['at_5'] is None
        assert empty['claim_recall']['at_10'] is None
        assert empty['discoverability']['canonical_in_top_5_rate'] is None
        assert empty['discoverability']['median_canonical_rank'] is None
        # The kind that WAS asked is scored, so the None above is the empty
        # subset talking and not the whole block failing to compute.
        assert by_kind['claim']['queries'] == 1
        assert by_kind['claim']['claim_recall']['at_5'] is not None


# ---------------------------------------------------------------------------
# The transform-blind discoverability sub-metric
# ---------------------------------------------------------------------------
#
# The mechanism is stated ONCE, in the script's module docstring under
# "Rank-based is not transform-blind" — a transform can materialise a record
# wearing the canonical's `record_id`, so `canonical in top-5` is a property
# of the READ TRANSFORM and not purely of retrieval, and the `stored_*` trio
# is the transform-blind counterpart that discloses the gap.
#
# This section covers the AGGREGATION half only.  `measure_arm` populating
# the rows — and the b_grouped-vs-c_peers divergence itself — is the next.


def _agg_row(*, canonical_in_5: float | None = 1.0,
             canonical_rank: int | None = 1,
             stored_in_5: float | None = 1.0,
             stored_rank: int | None = 1,
             has_canonical=True, kind='claim', held_out=False,
             recall_5=0.5, recall_10=0.5):
    """One `measure_arm` row, in the shape `_aggregate_queries` consumes.

    Hand-built: no store, no network, no embedder — so every expectation
    below is exact.

    The four canonical params are `| None`-typed because `measure_arm`
    genuinely emits None for each: the RANK pair when the canonical never
    ranked, the RATE pair when the cluster had no canonical to look for at
    all (`bake_off_storage_shape.py:3111-3112`).  Both are non-observations
    `_mean` must skip rather than average in as 0.0 — the distinction
    several tests below exist to pin, so the defaults must not narrow it
    away.
    """
    return {
        'kind': kind,
        'held_out': held_out,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'canonical_in_5': canonical_in_5,
        'canonical_rank': canonical_rank,
        'stored_canonical_in_5': stored_in_5,
        'stored_canonical_rank': stored_rank,
        'has_canonical': has_canonical,
        'tokens': 10.0,
    }


class TestTransformBlindDiscoverabilityAggregation:
    """`_aggregate_queries` reports the canonical twice: credited, and raw."""

    def test_the_stored_trio_is_computed_from_the_stored_fields(self):
        """The two halves must come from DIFFERENT inputs, not be aliased.

        Every row here has the transformed canonical at rank 1 and the
        stored canonical either deep or absent — the divergence a grouped
        read produces.  If the new keys ever read the transformed fields,
        they would print the transformed answer and the disclosure would be
        a duplicated column.
        """
        block = _mod()._aggregate_queries([
            _agg_row(canonical_in_5=1.0, canonical_rank=1,
                     stored_in_5=0.0, stored_rank=9),
            _agg_row(canonical_in_5=1.0, canonical_rank=1,
                     stored_in_5=1.0, stored_rank=3),
            _agg_row(canonical_in_5=1.0, canonical_rank=1,
                     stored_in_5=0.0, stored_rank=None),
        ], limit=10)['discoverability']

        # Transformed: every row found it in the top 5, at rank 1.
        assert block['canonical_in_top_5_rate'] == 1.0
        assert block['median_canonical_rank'] == 1.0
        assert block['canonical_found_count'] == 3
        # Transform-blind: one row in five, ranks 9 and 3, one never found.
        assert block['stored_canonical_in_top_5_rate'] == pytest.approx(1 / 3)
        assert block['stored_canonical_median_rank'] == 6.0
        assert block['stored_canonical_found_count'] == 2

    def test_an_unasked_subset_reports_none_rather_than_a_measured_zero(self):
        """`queries: 0` is "not asked".  A `0.00` would claim it was asked.

        Same discipline the existing keys already hold to — a new column that
        printed a measured zero for an empty subset would put a lie in the
        by-kind table's held-out row on any arm that had none.
        """
        block = _mod()._aggregate_queries([], limit=10)

        assert block['queries'] == 0
        disc = block['discoverability']
        assert disc['stored_canonical_in_top_5_rate'] is None
        assert disc['stored_canonical_median_rank'] is None
        # A COUNT of successes over an empty subset is honestly zero — it is
        # a denominator, not a rate, exactly as `canonical_found_count` is.
        assert disc['stored_canonical_found_count'] == 0
        assert disc['canonical_found_count'] == 0

    def test_a_query_with_no_canonical_is_excluded_not_averaged_in_as_zero(self):
        """`_mean`'s contract: `None` is a non-observation, never a 0.0.

        A claim query whose cluster has no canonical was never asked the
        discoverability question.  Averaging it in as zero would drag the
        rate down for a question the arm was never posed.
        """
        block = _mod()._aggregate_queries([
            _agg_row(stored_in_5=1.0, stored_rank=1),
            _agg_row(stored_in_5=None, stored_rank=None, canonical_in_5=None,
                     canonical_rank=None, has_canonical=False),
        ], limit=10)['discoverability']

        # 1.0 over the ONE row that had a canonical — not 0.5 over both.
        assert block['stored_canonical_in_top_5_rate'] == 1.0
        assert block['stored_canonical_found_count'] == 1
        assert block['canonical_candidates'] == 1

    def test_the_found_count_is_the_denominator_the_median_is_censored_over(self):
        """Mirrors the `canonical_found_count`/`canonical_candidates` pairing.

        The median is over the queries where the stored canonical surfaced AT
        ALL.  Without the count beside it, an arm whose stored canonical
        almost never ranks prints the best stored median in the table —
        scored on the handful of queries where it did.  That is the exact
        trap the module already documents for the transformed column, and it
        must not reopen one column over.
        """
        block = _mod()._aggregate_queries([
            _agg_row(stored_in_5=1.0, stored_rank=1),
            _agg_row(stored_in_5=0.0, stored_rank=None),
            _agg_row(stored_in_5=0.0, stored_rank=None),
        ], limit=10)['discoverability']

        # A flawless-looking median...
        assert block['stored_canonical_median_rank'] == 1.0
        # ...taken over exactly one of three queries that HAD a canonical.
        assert block['stored_canonical_found_count'] == 1
        assert block['canonical_candidates'] == 3

    def test_no_existing_discoverability_key_changed_name_or_value(self):
        """The new trio is purely ADDITIVE.

        The headline decision table is read by gate η off these keys; a
        rename or a shifted value would silently re-point the whole artifact.
        """
        rows = [
            _agg_row(canonical_in_5=1.0, canonical_rank=2,
                     stored_in_5=0.0, stored_rank=7),
            _agg_row(canonical_in_5=0.0, canonical_rank=None,
                     stored_in_5=0.0, stored_rank=None),
        ]

        block = _mod()._aggregate_queries(rows, limit=25)['discoverability']

        assert block['canonical_in_top_5_rate'] == 0.5
        assert block['median_canonical_rank'] == 2.0
        assert block['canonical_found_count'] == 1
        assert block['canonical_candidates'] == 2
        assert block['canonical_rank_window'] == 25


def _credit_arm(shape, *, canonical_at=None, fillers=2):
    """One canonical, one CHILD of it, and filler hits on another topic.

    The canonical always exists in the arm (so a grouped read can resolve the
    child upward into it) but appears among the fetched hits only when
    *canonical_at* names its 1-based position.  ``canonical_at=None`` is the
    case the credit mechanism turns on: the store never returned the
    canonical's own record, and only the read transform can put it in the
    window.
    """
    canonical = _rec(PARENT, topic='t', canonical=True, claim_ids=['k-canon'])
    child = _rec('child-1', parent_id=PARENT, kind='amendment', topic='t',
                 claim_ids=['k0'])
    filler = [_rec(f'f{i}', topic='other', claim_ids=[f'kf{i}'])
              for i in range(fillers)]
    ranked = [child, *filler]
    if canonical_at is not None:
        ranked.insert(canonical_at - 1, canonical)
    hits = [_sh(record, 0.9 - i / 100.0) for i, record in enumerate(ranked)]
    seeded = _seeded(
        shape,
        [canonical, child, *filler],
        canonical_by_topic={'t': canonical},
        canonical_by_cluster={'c1': PARENT},
        siblings_by_cluster={'c1': {'child-1'}},
    )
    return seeded, hits


class TestGroupedReadCanonicalCreditIsDisclosed:
    """The mechanism behind `b_grouped`'s headline discoverability number.

    Stated once in the script's module docstring, "Rank-based is not
    transform-blind"; the tests below are that statement made executable.  Not
    a bug to be corrected — the numbers are recorded as measured (G6/D10
    assert no threshold) — a mechanism to be DISCLOSED, and the
    transform-blind column is the disclosure.
    """

    def test_a_child_folding_upward_is_credited_as_the_canonical(self):
        """The conflation, stated as an executable difference.

        The canonical's own record is absent from the fetch entirely; only
        its child ranked.  The transformed column says "found, at rank 1".
        The transform-blind column says "never returned" — and both are true
        statements about different questions.
        """
        seeded, hits = _credit_arm('b_grouped')

        disc = _measure(seeded, hits, pin=False)['discoverability']

        assert disc['canonical_in_top_5_rate'] == 1.0
        assert disc['median_canonical_rank'] == 1.0
        assert disc['canonical_found_count'] == 1
        assert disc['stored_canonical_in_top_5_rate'] == 0.0
        assert disc['stored_canonical_found_count'] == 0
        assert disc['stored_canonical_median_rank'] is None
        # The query DID have a canonical to look for, so the 0.0 above is a
        # measured miss and not a non-observation.
        assert disc['canonical_candidates'] == 1

    @pytest.mark.parametrize('shape', ['status_quo', 'c_peers'])
    @pytest.mark.parametrize('canonical_at', [None, 2])
    def test_without_grouping_the_two_columns_agree(self, shape, canonical_at):
        """No grouping transform, so there is nothing for the credit to come
        from and the two columns must report the same thing — whether the
        canonical ranked (``canonical_at=2``) or never did (``None``).

        This is what makes the divergence above attributable to grouping
        specifically rather than to the new column being wired to a different
        window, and it is what keeps the column from being always-zero.
        """
        seeded, hits = _credit_arm(shape, canonical_at=canonical_at)

        disc = _measure(seeded, hits, pin=False)['discoverability']

        assert disc['stored_canonical_in_top_5_rate'] == \
            disc['canonical_in_top_5_rate']
        assert disc['stored_canonical_median_rank'] == \
            disc['median_canonical_rank']
        assert disc['stored_canonical_found_count'] == \
            disc['canonical_found_count']
        # Not vacuous: a canonical that DID rank reads as found, at its rank.
        expected_rate = 0.0 if canonical_at is None else 1.0
        assert disc['stored_canonical_in_top_5_rate'] == expected_rate

    @pytest.mark.parametrize('shape', ['status_quo', 'c_peers', 'b_grouped'])
    def test_the_stored_column_is_blind_to_the_pin_as_well(self, shape):
        """Transform-blind means blind to EVERY read-side transform.

        `apply_topic_anchor` also injects the canonical into the window, so a
        column blind only to grouping would still credit the pin for a
        retrieval the ANN never performed — the same defect one transform
        over.  Measured over the raw hits, the trio cannot depend on the pin,
        which is what makes it comparable across all six arms.
        """
        seeded, hits = _credit_arm(shape)

        off = _measure(seeded, hits, pin=False)['discoverability']
        on = _measure(seeded, hits, pin=True)['discoverability']

        for key in ('stored_canonical_in_top_5_rate',
                    'stored_canonical_median_rank',
                    'stored_canonical_found_count'):
            assert on[key] == off[key], key

    def test_the_pin_moves_the_transformed_column_and_not_the_stored_one(self):
        """The anti-vacuity half of the test above.

        Equality across pin variants is only meaningful if the pin actually
        FIRED, so this pins a fixture where it does: the window has headroom,
        the pin appends the canonical, and the transformed rate moves 0 -> 1
        while the transform-blind rate stays where retrieval left it.
        """
        seeded, hits = _credit_arm('c_peers')

        off = _measure(seeded, hits, pin=False)
        on = _measure(seeded, hits, pin=True)

        assert on['pin']['window_changed_rate'] > 0.0
        assert off['discoverability']['canonical_in_top_5_rate'] == 0.0
        assert on['discoverability']['canonical_in_top_5_rate'] == 1.0
        assert on['discoverability']['stored_canonical_in_top_5_rate'] == 0.0

    def test_the_stored_rank_is_not_censored_at_the_read_window(self):
        """"Outside the top 5" and "absent entirely" stay different findings.

        Exactly the contract the transformed rank already holds — a stored
        rank censored at 5 would collapse a canonical that came NEARLY there
        into the same None as one the store never returned.
        """
        deep_seeded, deep_hits = _credit_arm('c_peers', canonical_at=7,
                                             fillers=7)
        absent_seeded, absent_hits = _credit_arm('c_peers', fillers=7)

        deep = _measure(deep_seeded, deep_hits, pin=False)['discoverability']
        absent = _measure(absent_seeded, absent_hits,
                          pin=False)['discoverability']

        assert deep['stored_canonical_median_rank'] == 7.0
        assert deep['stored_canonical_found_count'] == 1
        assert deep['stored_canonical_in_top_5_rate'] == 0.0
        assert absent['stored_canonical_median_rank'] is None
        assert absent['stored_canonical_found_count'] == 0
        assert absent['stored_canonical_in_top_5_rate'] == 0.0

    def test_the_new_fields_add_keys_and_change_no_existing_measurement(self):
        """Purely additive at the `measure_arm` seam, not only at aggregation.

        Every block's key set is pinned by equality and every pre-existing
        discoverability/claim_recall/pin value is hand-computed, so a change
        that re-pointed an existing column while adding the new one cannot
        pass.  The fixture is three plain records of `'body'` (4 chars) under
        `c_peers`, with the canonical itself ranked first.
        """
        seeded, hits = _credit_arm('c_peers', canonical_at=1, fillers=1)

        measurement = _measure(seeded, hits, pin=False)

        assert set(measurement) == {
            'pin', 'claim_recall', 'discoverability', 'by_query_kind',
            'tokens_per_query', 'guard_adequacy',
        }
        assert measurement['pin'] == {'enabled': False,
                                      'window_changed_rate': None}
        assert measurement['claim_recall'] == {'at_5': 1.0, 'at_10': 1.0}
        assert measurement['discoverability'] == {
            'canonical_in_top_5_rate': 1.0,
            'median_canonical_rank': 1.0,
            'canonical_found_count': 1,
            'canonical_candidates': 1,
            'canonical_rank_window': 10,
            'stored_canonical_in_top_5_rate': 1.0,
            'stored_canonical_median_rank': 1.0,
            'stored_canonical_found_count': 1,
        }
        assert measurement['tokens_per_query'] == {
            'mean': 12.0, 'estimator': 'injected:chars', 'window': 10,
        }
        assert set(measurement['guard_adequacy']) == {
            'candidate_present_rate', 'guard_matched_rate', 'threshold_replay',
            'threshold', 'max_observed_score', 'probes', 'guard_covered_probes',
            'guard_covered_category',
        }


# ===========================================================================
# 4012 step-5 — `read_path` can select 4004's PROMOTING pin
# ===========================================================================
#
# The regrowth probe's third read arm is the transform 4004's selection table
# picked: `read_transform_selection.apply_promoting_topic_anchor`.  It is
# reached lazily through the sibling-script loader, never reimplemented here —
# two copies could drift, and the point of the probe is to measure the
# transform that was actually selected.  Pure: hand-built `ScoredHit` lists
# with exactly-known answers, no embedding.


class TestReadPathPromote:

    def test_promote_defaults_to_false_on_read_path_and_measure_arm(self):
        """Every existing caller and test must be unchanged by this addition.

        The default is asserted BEHAVIOURALLY — omitting `promote` produces
        the same result as passing `promote=False` — rather than off
        `inspect.signature(...).parameters['promote'].default`.  A default
        declared in the signature and then ignored in the body satisfies the
        introspective form while breaking every existing caller, which is the
        property this test is actually for.

        The keyword-only half stays introspective: "cannot be passed
        positionally" is an API-surface contract with no behavioural shadow to
        assert, and it is what stops a future positional argument from
        silently re-pointing an existing call site.
        """
        import inspect  # noqa: PLC0415
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)

        assert mod.read_path(seeded, hits, 5, pin=True) == mod.read_path(
            seeded, hits, 5, pin=True, promote=False,
        )
        assert _measure(seeded, hits, pin=True) == _measure(
            seeded, hits, pin=True, promote=False,
        )

        for func in (mod.read_path, mod.measure_arm):
            parameter = inspect.signature(func).parameters['promote']
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_a_full_window_promote_puts_the_canonical_first_and_keeps_k(self):
        """A FULL window is what distinguishes the two transforms.

        `read_path` truncates AFTER the transforms, so at `len(hits) == k` the
        additive pin's appended canonical is cut straight back off and the
        window is unchanged.  Only a promoting transform can place it inside
        the budget.
        """
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)

        window = mod.read_path(seeded, hits, 5, pin=True, promote=True)

        assert len(window) == 5
        assert window[0].record_id == 'canon'
        assert 'canon' not in [hit.record.record_id for hit in hits]

    def test_at_a_full_window_the_additive_pin_changes_nothing(self):
        """The contrast the probe's third arm exists to measure."""
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)

        additive = mod.read_path(seeded, hits, 5, pin=True, promote=False)

        assert [r.record_id for r in additive] == [h.record.record_id for h in hits[:5]]

    def test_pin_on_promote_off_is_byte_identical_to_todays_additive_behaviour(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=12)

        for k in (5, 10, 12):
            expected = mod.apply_topic_anchor(
                [hit.record for hit in hits[:k]], seeded.canonical_by_topic,
            )[:k]
            got = mod.read_path(seeded, hits, k, pin=True, promote=False)
            assert [r.record_id for r in got] == [r.record_id for r in expected]

    def test_pin_off_promote_off_is_byte_identical_to_todays_flat_behaviour(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=12)

        for k in (5, 10, 12):
            got = mod.read_path(seeded, hits, k, pin=False, promote=False)
            assert [r.record_id for r in got] == [h.record.record_id for h in hits[:k]]

    def test_promote_without_pin_raises_naming_both_keywords(self):
        """Promotion is a variant of the pin's firing rule, not a fourth mode.

        A caller asking for `promote=True, pin=False` has a bug, and a silent
        fourth behaviour would let it reach the decision table unnoticed.
        """
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)

        with pytest.raises(ValueError) as excinfo:
            mod.read_path(seeded, hits, 5, pin=False, promote=True)

        message = str(excinfo.value)
        assert 'promote' in message
        assert 'pin' in message

    def test_the_promoting_transform_is_reached_by_a_lazy_module_load(self):
        """A rename in the sibling must fail HERE, by name.

        Without this, `apply_promoting_topic_anchor` disappearing surfaces as
        an `AttributeError` twenty minutes into a live run, after the seeding
        has already been paid for.
        """
        mod = _mod()

        script = mod.load_read_transform_script()

        assert hasattr(script, 'apply_promoting_topic_anchor'), (
            'read_transform_selection.apply_promoting_topic_anchor is gone — '
            'the regrowth probe measures the transform 4004 selected, so a '
            'rename there is a contract break here'
        )
        assert callable(script.apply_promoting_topic_anchor)

    def test_read_path_routes_to_the_sibling_transform_not_a_local_copy(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)
        script = mod.load_read_transform_script()

        expected = script.apply_promoting_topic_anchor(
            [hit.record for hit in hits[:5]], seeded.canonical_by_topic,
        )[:5]
        got = mod.read_path(seeded, hits, 5, pin=True, promote=True)

        assert [r.record_id for r in got] == [r.record_id for r in expected]


class TestMeasureArmPromote:

    def test_the_pin_diagnostic_is_scored_against_the_pin_off_baseline(self):
        """NOT against a promote=True baseline.

        `_window`'s counterfactual call must stay `pin=False` (and therefore
        promote-free by construction).  If promotion leaked into the baseline,
        the two would agree and the diagnostic would report 0.00 for an arm
        that reorders every window.

        `n=12` so that EVERY window `_window` measures is genuinely full —
        the k=5 pair, the k=10 pair and the guard probe's GUARD_TOP_K pair.
        At n=5 the k=10 window has headroom, the additive pin legitimately
        fires there, and the additive rate is 1/3 rather than 0 for a reason
        that has nothing to do with promotion leaking.
        """
        seeded, hits = _full_window_arm(n=12)

        promoting = _measure(seeded, hits, pin=True, promote=True, probes=[('c1', {})])
        additive = _measure(seeded, hits, pin=True, promote=False, probes=[('c1', {})])

        assert promoting['pin']['enabled'] is True
        assert promoting['pin']['window_changed_rate'] == 1.0
        # The same windows, under the additive pin, are unchanged at a full
        # window — so the two rates cannot both be an artifact of the closure.
        assert additive['pin']['window_changed_rate'] == 0.0

    def test_a_promoting_measurement_returns_the_full_required_metric_set(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=5)

        measurement = _measure(seeded, hits, pin=True, promote=True, probes=[('c1', {})])

        assert set(measurement) >= set(mod._REQUIRED_ARM_METRICS)
        for block, keys in mod._REQUIRED_ARM_METRICS.items():
            for key in keys:
                assert key in measurement[block], f'{block}.{key}'

    def test_the_transform_blind_stored_trio_is_unmoved_by_promotion(self):
        """`stored_*` is measured over the RAW hits, before any `read_path`.

        3560's correction is exactly this: the SCORED discoverability must be
        blind to the read transform, or the probe reports a placement property
        as a retrieval one.
        """
        seeded, hits = _full_window_arm(n=5)

        promoting = _measure(seeded, hits, pin=True, promote=True, probes=[('c1', {})])
        flat = _measure(seeded, hits, pin=False, probes=[('c1', {})])

        for key in ('stored_canonical_in_top_5_rate', 'stored_canonical_median_rank',
                    'stored_canonical_found_count'):
            assert promoting['discoverability'][key] == flat['discoverability'][key]

    def test_the_credited_column_is_the_one_promotion_moves(self):
        """Reported BESIDE the stored one, never alone — the pair's semantics.

        Promotion places the canonical inside the window, so the credited
        rate is a PLACEMENT property under this arm, exactly the way
        `apply_grouped_read`'s record-id aliasing was under `b_grouped`.
        Asserted as "the two columns disagree", not as a magnitude.
        """
        seeded, hits = _full_window_arm(n=5)

        promoting = _measure(seeded, hits, pin=True, promote=True, probes=[('c1', {})])

        assert promoting['discoverability']['canonical_in_top_5_rate'] != (
            promoting['discoverability']['stored_canonical_in_top_5_rate']
        )


# ===========================================================================
# 4012 step-7 — the per-arm measurement fan-out and the delta arithmetic
# ===========================================================================
#
# `measure_regrowth_arms` is fed a `SeededArm` and a hand-built `fetched`
# dict; `regrowth_deltas` / `regrowth_stamping_value` are fed synthetic
# measurement dicts, so every expected number below was written by the test
# itself.  NO threshold, bound or magnitude is asserted on any MEASURED
# quantity anywhere in this section (gate G6).


def _plucked(**overrides) -> dict:
    """A flat metric projection with every `REGROWTH_METRICS` key present."""
    values = {
        'claim_recall.at_5': 0.5,
        'claim_recall.at_10': 0.6,
        'discoverability.stored_canonical_in_top_5_rate': 0.7,
        'discoverability.stored_canonical_median_rank': 3.0,
        'discoverability.stored_canonical_found_count': 12.0,
        'discoverability.canonical_in_top_5_rate': 0.8,
        'tokens_per_query.mean': 100.0,
    }
    values.update(overrides)
    return values


def _arms(**per_arm) -> dict:
    """`{arm: plucked}` for all three read arms, overridable per arm."""
    return {
        arm: per_arm.get(arm, _plucked())
        for arm in _mod().REGROWTH_READ_ARMS
    }


class TestRegrowthReadArmsAndMetricsArePinned:

    def test_the_three_read_arms_are_pinned_by_equality_in_order(self):
        assert _mod().REGROWTH_READ_ARMS == ('flat', 'additive_pin', 'promoting_pin')

    def test_the_reported_metrics_are_pinned_by_equality(self):
        """A metric dropped from a delta table is a metric dropped from the
        decision — asserted by equality, exactly as `DECISION_TABLE_COLUMNS`.

        The credited `canonical_in_top_5_rate` travels BESIDE the stored trio
        and never alone: `apply_promoting_topic_anchor` injects the canonical
        into the window, so under `promoting_pin` that column is a PLACEMENT
        property in exactly the way `apply_grouped_read`'s was under
        `b_grouped`.  Dropping it would hide the transform's contribution;
        printing it alone would repeat the misreading 3560 and 4004 each had
        to correct after publication.
        """
        assert _mod().REGROWTH_METRICS == (
            ('claim_recall', 'at_5'),
            ('claim_recall', 'at_10'),
            ('discoverability', 'stored_canonical_in_top_5_rate'),
            ('discoverability', 'stored_canonical_median_rank'),
            ('discoverability', 'stored_canonical_found_count'),
            ('discoverability', 'canonical_in_top_5_rate'),
            ('tokens_per_query', 'mean'),
        )

    def test_the_probe_is_scoped_to_the_ratified_write_shape(self):
        """esc-3200-3 was a SPLIT ratification: C's write shape, no transform."""
        assert _mod().REGROWTH_SHAPE == 'c_peers'
        assert _mod().REGROWTH_SHAPE in _mod().ARM_SHAPES


class TestMeasureRegrowthArms:
    """One measurement block per read arm, routed to the right (pin, promote)."""

    def _fanout(self, seeded, hits):
        return _mod().measure_regrowth_arms(
            seeded,
            {'queries': {'q1': hits}, 'probes': {'c1': hits}},
            queries=[_query()],
            probes=[('c1', {})],
            estimator=_CHARS,
            guard_threshold=0.92,
            limit=10,
        )

    def test_one_block_per_read_arm_in_order(self):
        seeded, hits = _full_window_arm(n=12)

        blocks = self._fanout(seeded, hits)

        assert list(blocks) == list(_mod().REGROWTH_READ_ARMS)

    def test_each_block_carries_the_full_required_metric_set(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=12)

        for block in self._fanout(seeded, hits).values():
            for metric, keys in mod._REQUIRED_ARM_METRICS.items():
                for key in keys:
                    assert key in block[metric], f'{metric}.{key}'

    def test_the_arms_are_routed_to_flat_additive_and_promoting(self):
        """Asserted by BEHAVIOUR, not by a flag readback.

        The arm gives the canonical a rank only a promoting pin can move into
        the window — a full k=5 window with the canonical outside it — so the
        three arms are distinguishable by what they measured.
        """
        seeded, hits = _full_window_arm(n=12)

        blocks = self._fanout(seeded, hits)

        assert blocks['flat']['pin']['enabled'] is False
        assert blocks['flat']['pin']['window_changed_rate'] is None
        assert blocks['additive_pin']['pin']['enabled'] is True
        # A full window leaves the additive pin nowhere to put anything...
        assert blocks['additive_pin']['pin']['window_changed_rate'] == 0.0
        # ...while the promoting one reorders every window it touches.
        assert blocks['promoting_pin']['pin']['enabled'] is True
        assert blocks['promoting_pin']['pin']['window_changed_rate'] == 1.0

    def test_the_transform_blind_trio_agrees_across_all_three_arms(self):
        seeded, hits = _full_window_arm(n=12)

        blocks = self._fanout(seeded, hits)

        for key in ('stored_canonical_in_top_5_rate', 'stored_canonical_median_rank',
                    'stored_canonical_found_count'):
            values = {
                arm: block['discoverability'][key] for arm, block in blocks.items()
            }
            assert len(set(values.values())) == 1, values


class TestPluckRegrowthMetrics:

    def test_it_projects_exactly_the_pinned_metrics_to_a_flat_dict(self):
        mod = _mod()
        seeded, hits = _full_window_arm(n=12)
        measurement = _measure(seeded, hits, pin=False, probes=[('c1', {})])

        plucked = mod._pluck_regrowth_metrics(measurement)

        assert list(plucked) == [f'{b}.{k}' for b, k in mod.REGROWTH_METRICS]
        for block, key in mod.REGROWTH_METRICS:
            assert plucked[f'{block}.{key}'] == measurement[block][key]


class TestRegrowthDeltas:
    """`after - baseline`, exactly, with None propagated rather than zeroed."""

    def test_deltas_are_after_minus_baseline_per_arm_per_metric(self):
        mod = _mod()
        baseline = _arms()
        after = _arms(flat=_plucked(**{'claim_recall.at_5': 0.25}))

        deltas = mod.regrowth_deltas(baseline, after)

        assert list(deltas) == list(mod.REGROWTH_READ_ARMS)
        assert deltas['flat']['claim_recall.at_5'] == pytest.approx(-0.25)
        assert deltas['flat']['claim_recall.at_10'] == 0.0
        assert deltas['additive_pin']['claim_recall.at_5'] == 0.0

    def test_a_none_on_either_side_propagates_as_none_never_as_zero(self):
        """The `_NO_MEASUREMENT` discipline, one layer down.

        A delta table that prints "no measurement" as a measured zero says the
        injection changed nothing — which is a finding, not an absence.
        """
        mod = _mod()
        baseline = _arms(flat=_plucked(**{'claim_recall.at_5': None}))
        after = _arms(additive_pin=_plucked(**{'claim_recall.at_10': None}))

        deltas = mod.regrowth_deltas(baseline, after)

        assert deltas['flat']['claim_recall.at_5'] is None
        assert deltas['additive_pin']['claim_recall.at_10'] is None
        assert deltas['promoting_pin']['claim_recall.at_5'] == 0.0

    def test_an_arm_present_on_one_side_only_raises_naming_it(self):
        mod = _mod()
        baseline = _arms()
        after = dict(_arms())
        after.pop('promoting_pin')

        with pytest.raises(mod.MeasurementError) as excinfo:
            mod.regrowth_deltas(baseline, after)

        assert 'promoting_pin' in str(excinfo.value)

    def test_a_metric_present_on_one_side_only_raises_naming_it(self):
        mod = _mod()
        baseline = _arms()
        after = _arms()
        after['flat'] = {
            k: v for k, v in after['flat'].items() if k != 'tokens_per_query.mean'
        }

        with pytest.raises(mod.MeasurementError) as excinfo:
            mod.regrowth_deltas(baseline, after)

        assert 'tokens_per_query.mean' in str(excinfo.value)


class TestRegrowthStampingValue:
    """`stamped delta - unstamped delta` — the number task 4006 is owed."""

    def test_it_subtracts_the_unstamped_delta_from_the_stamped_one(self):
        mod = _mod()
        deltas = {
            'unstamped': _arms(flat=_plucked(**{'claim_recall.at_5': -0.10})),
            'stamped': _arms(flat=_plucked(**{'claim_recall.at_5': -0.02})),
        }

        value = mod.regrowth_stamping_value(deltas)

        assert list(value) == list(mod.REGROWTH_READ_ARMS)
        assert value['flat']['claim_recall.at_5'] == pytest.approx(0.08)

    def test_two_equal_measured_deltas_give_zero_not_none(self):
        """Zero here is a measurement: stamping bought nothing, and said so."""
        mod = _mod()
        deltas = {'unstamped': _arms(), 'stamped': _arms()}

        value = mod.regrowth_stamping_value(deltas)

        assert value['flat']['claim_recall.at_5'] == 0.0
        assert value['flat']['claim_recall.at_5'] is not None

    def test_none_propagates_through_the_stamping_value_too(self):
        mod = _mod()
        deltas = {
            'unstamped': _arms(flat=_plucked(**{'claim_recall.at_5': None})),
            'stamped': _arms(),
        }

        value = mod.regrowth_stamping_value(deltas)

        assert value['flat']['claim_recall.at_5'] is None

    def test_a_missing_mode_raises_naming_it(self):
        mod = _mod()

        with pytest.raises(mod.MeasurementError) as excinfo:
            mod.regrowth_stamping_value({'unstamped': _arms()})

        assert 'stamped' in str(excinfo.value)


class TestBuildRegrowthBlock:

    def _block(self):
        mod = _mod()
        deltas = {mode: _arms() for mode in mod.REGROWTH_MODES}
        return mod.build_regrowth_block(
            baseline=_arms(),
            after_by_mode={mode: _arms() for mode in mod.REGROWTH_MODES},
            injections=list(_injections()),
            fixture_path=REGROWTH_INJECTION_PATH,
        ), deltas

    def test_the_block_carries_its_descriptors_and_all_four_tables(self):
        mod = _mod()
        block, _ = self._block()

        assert block['shape'] == mod.REGROWTH_SHAPE
        assert block['read_arms'] == list(mod.REGROWTH_READ_ARMS)
        assert block['modes'] == list(mod.REGROWTH_MODES)
        assert block['topics_injected'] == 20
        assert block['injections_per_topic'] == 1
        assert block['injection_fixture'] == (
            'fused-memory/tests/fixtures/e2_regrowth_injection.jsonl'
        )
        assert list(block['after']) == list(mod.REGROWTH_MODES)
        assert list(block['deltas']) == list(mod.REGROWTH_MODES)
        for mode in mod.REGROWTH_MODES:
            assert list(block['after'][mode]) == list(mod.REGROWTH_READ_ARMS)
            assert list(block['deltas'][mode]) == list(mod.REGROWTH_READ_ARMS)
        assert list(block['stamping_value']) == list(mod.REGROWTH_READ_ARMS)

    def _descriptors(self, injections):
        mod = _mod()
        block = mod.build_regrowth_block(
            baseline=_arms(),
            after_by_mode={mode: _arms() for mode in mod.REGROWTH_MODES},
            injections=list(injections),
            fixture_path=REGROWTH_INJECTION_PATH,
        )
        return block['topics_injected'], block['injections_per_topic']

    def test_injections_per_topic_is_derived_from_the_slab_not_typed(self):
        """A hard-coded `1` is a descriptor that can contradict its own table.

        It was literally `1`, so a slab that injected twice per topic — or
        none at all, which a `--clusters N` subset can produce — still
        published "1 injection each" beside the count it disagrees with.  The
        `+1` in this probe's name is the independent variable, so the
        artifact has to REPORT it rather than assert it.
        """
        injections = list(_injections())

        assert self._descriptors(injections) == (20, 1)
        assert self._descriptors(injections * 2) == (20, 2)

    def test_an_empty_slab_describes_nothing_rather_than_describing_one(self):
        """`None`, explicitly — the `protocol['replayed_from']` convention.

        Zero topics with "1 injection each" is a sentence about a table that
        is not there.  Both drivers now decline to build a block at all for
        an empty post-subset slab, so this is the belt to that suspenders:
        a caller that builds one anyway gets an honest descriptor.
        """
        assert self._descriptors([]) == (0, None)

    def test_the_deltas_it_carries_are_the_arithmetic_not_a_restatement(self):
        mod = _mod()
        block = mod.build_regrowth_block(
            baseline=_arms(),
            after_by_mode={
                'unstamped': _arms(flat=_plucked(**{'claim_recall.at_5': 0.25})),
                'stamped': _arms(),
            },
            injections=list(_injections()),
            fixture_path=REGROWTH_INJECTION_PATH,
        )

        assert block['deltas']['unstamped']['flat']['claim_recall.at_5'] == pytest.approx(-0.25)
        assert block['deltas']['stamped']['flat']['claim_recall.at_5'] == 0.0
        assert block['stamping_value']['flat']['claim_recall.at_5'] == pytest.approx(0.25)


# ===========================================================================
# 4012 step-9 — `build_report` carries the regrowth block, and refuses a
#               partial one
# ===========================================================================


def _regrowth_block(**overrides) -> dict:
    """A COMPLETE regrowth block, before whatever a test removes from it."""
    mod = _mod()
    block = mod.build_regrowth_block(
        baseline=_arms(),
        after_by_mode={mode: _arms() for mode in mod.REGROWTH_MODES},
        injections=list(_injections()),
        fixture_path=REGROWTH_INJECTION_PATH,
    )
    block.update(overrides)
    return block


class TestReportSchemaVersionBump:

    def test_the_schema_version_is_three(self):
        """A v2 artifact carries no regrowth block.

        Diffing a v2 and a v3 artifact as if they answered the same questions
        is exactly the misreading the version exists to prevent.
        """
        assert _mod().REPORT_SCHEMA_VERSION == 3


class TestBuildReportCarriesRegrowth:

    def test_a_report_built_without_the_probe_emits_an_explicit_none(self):
        """`'regrowth' in report`, not merely falsy.

        An ABSENT key would make "this build predates the probe" and "the
        probe was skipped" the same reading — structurally the failure that
        let 3199 reach done, whose delivered-check tested only that the
        report exists.  The explicit `None` follows the convention
        `protocol['replayed_from']` already set in this file.
        """
        report = _mod().build_report(
            arms=_all_arms(), audit_recall=_audit_recall(), protocol=_protocol(),
        )

        assert 'regrowth' in report
        assert report['regrowth'] is None

    def test_a_complete_block_is_carried_through_unchanged_and_last(self):
        block = _regrowth_block()

        report = _mod().build_report(
            arms=_all_arms(), audit_recall=_audit_recall(), protocol=_protocol(),
            regrowth=block,
        )

        assert report['regrowth'] == block
        # Key order is stable so two runs stay diffable.
        assert list(report) == [
            'schema_version', 'protocol', 'arms', 'audit_recall', 'regrowth',
        ]


class TestCheckRegrowth:
    """Every incompleteness is NAMED, never discovered as a bare KeyError."""

    def _build(self, block):
        return _mod().build_report(
            arms=_all_arms(), audit_recall=_audit_recall(), protocol=_protocol(),
            regrowth=block,
        )

    def test_it_is_not_invoked_at_all_when_regrowth_is_none(self, monkeypatch):
        mod = _mod()
        called = []
        monkeypatch.setattr(
            mod, '_check_regrowth', lambda block: called.append(block),
        )

        mod.build_report(
            arms=_all_arms(), audit_recall=_audit_recall(), protocol=_protocol(),
        )

        assert called == []

    @pytest.mark.parametrize(
        'descriptor', ['shape', 'injection_fixture', 'topics_injected'],
    )
    def test_a_missing_top_level_descriptor_is_named(self, descriptor):
        block = _regrowth_block()
        block.pop(descriptor)

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        assert descriptor in str(excinfo.value)

    def test_a_missing_mode_is_named(self):
        block = _regrowth_block()
        block['deltas'].pop('stamped')

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        assert 'stamped' in str(excinfo.value)

    def test_an_unknown_mode_is_named(self):
        block = _regrowth_block()
        block['after']['stamped_and_pinned'] = _arms()

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        assert 'stamped_and_pinned' in str(excinfo.value)

    def test_a_missing_read_arm_is_named(self):
        block = _regrowth_block()
        block['baseline'].pop('promoting_pin')

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        assert 'promoting_pin' in str(excinfo.value)

    def test_an_unknown_read_arm_is_named(self):
        block = _regrowth_block()
        block['stamping_value']['grouped_pin'] = _plucked()

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        assert 'grouped_pin' in str(excinfo.value)

    @pytest.mark.parametrize('table', ['baseline', 'stamping_value'])
    def test_a_missing_metric_in_a_flat_table_is_named(self, table):
        block = _regrowth_block()
        block[table]['flat'].pop('tokens_per_query.mean')

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        message = str(excinfo.value)
        assert 'tokens_per_query.mean' in message
        assert 'flat' in message

    @pytest.mark.parametrize('table', ['after', 'deltas'])
    def test_a_missing_metric_in_a_per_mode_table_is_named(self, table):
        block = _regrowth_block()
        block[table]['unstamped']['flat'].pop('claim_recall.at_5')

        with pytest.raises(_mod().IncompleteReportError) as excinfo:
            self._build(block)

        message = str(excinfo.value)
        assert 'claim_recall.at_5' in message
        assert 'unstamped' in message
        assert 'flat' in message

    def test_a_none_VALUE_does_not_raise(self):
        """`None` is a legitimate "measured, no denominator" in this pipeline.

        The renderer prints it as `—`.  Only an ABSENT key means the run
        broke.  Both directions are pinned so the check cannot drift into
        rejecting a real measurement.
        """
        block = _regrowth_block()
        block['baseline']['flat']['claim_recall.at_5'] = None
        block['deltas']['unstamped']['flat']['claim_recall.at_5'] = None

        report = self._build(block)

        assert report['regrowth']['baseline']['flat']['claim_recall.at_5'] is None


# ===========================================================================
# 4012 step-11 — the rendered `## Regrowth deltas` section: the two tables,
#                the per-arm bullets, the credited-vs-stored disclosure and
#                the NOT-blind-authored disclosure in `## Protocol`.
#
# LANE: pure.  Every report rendered below is one the test itself built, so
# every number asserted here was written by the test.  NO metric magnitude,
# rate, threshold or bound is asserted anywhere in this section (gate G6);
# what is pinned is COLUMNS, ORDER, COMPLETENESS and the None-vs-zero
# distinction.
# ===========================================================================


def _report_with_regrowth(block=None, protocol=None):
    """A full report carrying a regrowth block."""
    mod = _mod()
    return mod.build_report(
        arms=_all_arms(),
        audit_recall=_audit_recall(),
        protocol=_protocol() if protocol is None else protocol,
        regrowth=_regrowth_block() if block is None else block,
    )


def _moved_block(**after_unstamped):
    """A block whose `unstamped`/`flat` arm MOVED, so deltas are nonzero.

    Every value here is the test's own: baseline comes from `_plucked`, and
    the override is whatever the caller passed.
    """
    mod = _mod()
    return mod.build_regrowth_block(
        baseline=_arms(),
        after_by_mode={
            'unstamped': _arms(flat=_plucked(**after_unstamped)),
            'stamped': _arms(),
        },
        injections=list(_injections()),
        fixture_path=REGROWTH_INJECTION_PATH,
    )


def _section(rendered: str, heading: str) -> list[str]:
    """One `## ` section's lines — heading included, next heading excluded.

    Located by the renderer-emitted heading rather than by a line index, for
    the same reason `_by_kind_table` is: a paragraph added above must not
    move the assertion off its section.
    """
    lines = rendered.splitlines()
    start = lines.index(heading)
    end = next(
        (i for i, line in enumerate(lines[start + 1:], start + 1)
         if line.startswith('## ')),
        len(lines),
    )
    return lines[start:end]


def _rows_under(lines: list[str], header: str) -> list[str]:
    """The data rows of the table whose header row is exactly `header`."""
    at = lines.index(header)
    rows = []
    for line in lines[at + 2:]:  # skip the header and its `| --- |`
        if not line.startswith('| '):
            break
        rows.append(line)
    return rows


def _header_row(columns) -> str:
    return '| ' + ' | '.join(columns) + ' |'


# ===========================================================================
# step-17 — the signal sentence's two derived phrases
# ===========================================================================
#
# `_absorption_phrase` and `_cost_phrase` are the only two places in this
# renderer that turn a measured delta into a DIRECTIONAL CLAIM, and they sit
# in the probe's headline sentence.  Both had zero direct coverage, and both
# are wrong on the committed data: `_absorption_phrase` compared `abs()`, so
# a measured GAIN of 0.0042 and a measured LOSS of 0.0042 read as "the same
# distance from baseline", and the sentence hard-typed the verb "costs"
# around `_gap_cell`, so a measured GAIN rendered as a cost.
#
# Pure — every input below is a literal this file writes, so NO measured
# magnitude, threshold or bound is pinned (gate G6).  What is pinned is the
# SENTENCE'S LOGIC, which stays valid whatever the next run measures.


class TestAbsorptionPhrase:
    """Does 4004's selected transform absorb the flat read's cost?

    The question is directional, so the answer has to be computed from the
    SIGNED cost `delta * cost_sign` — not from distance-from-baseline, which
    cannot tell a gain from a loss.
    """

    def test_every_metric_declares_a_cost_direction(self):
        """By EQUALITY, and covering every key `REGROWTH_METRICS` projects.

        A metric with no declared direction must not be silently defaulted
        to "higher is better": that is true of five of these seven columns
        and false of the other two, and a default would get the exceptions
        wrong in the one sentence a gate reads.
        """
        mod = _mod()
        assert mod._REGROWTH_COST_SIGN == {
            'claim_recall.at_5': -1,
            'claim_recall.at_10': -1,
            'discoverability.stored_canonical_in_top_5_rate': -1,
            'discoverability.stored_canonical_median_rank': +1,
            'discoverability.stored_canonical_found_count': -1,
            'discoverability.canonical_in_top_5_rate': -1,
            'tokens_per_query.mean': +1,
        }

    def test_its_keys_are_exactly_the_regrowth_metric_keys(self):
        """So adding a metric without declaring its direction fails loudly."""
        mod = _mod()
        assert set(mod._REGROWTH_COST_SIGN) == set(mod._regrowth_metric_keys())

    @pytest.mark.parametrize('flat,selected,cost_sign,expected', [
        # Either side never measured — unchanged behaviour.
        (None, -0.10, -1, 'not comparable'),
        (-0.10, None, -1, 'not comparable'),
        (None, None, +1, 'not comparable'),
        # Neither arm moved at all.
        (0.0, 0.0, -1, 'neither arm moved'),
        (0.0, 0.0, +1, 'neither arm moved'),
        # Both moved AGAINST the cost direction: nobody paid anything.
        (+0.10, +0.20, -1, 'neither arm paid a cost'),
        (+0.10, 0.0, -1, 'neither arm paid a cost'),
        (-0.10, -0.20, +1, 'neither arm paid a cost'),
        # The flat read paid; the selected transform paid nothing.
        (-0.10, 0.0, -1, 'absorbs all of it'),
        (+3.13, 0.0, +1, 'absorbs all of it'),
        # The flat read paid; the selected transform GAINED — a sign flip.
        (-0.10, +0.30, -1, 'more than absorbs it'),
        (+3.13, -1.00, +1, 'more than absorbs it'),
        # The flat read paid more than the selected transform did.
        (-0.20, -0.10, -1, 'absorbs part of it'),
        (+3.13, +1.00, +1, 'absorbs part of it'),
        # Both paid the same.
        (-0.10, -0.10, -1, 'does not change it'),
        (+3.13, +3.13, +1, 'does not change it'),
        # The selected transform paid MORE.
        (-0.10, -0.20, -1, 'costs more than the flat read'),
        (+1.00, +3.13, +1, 'costs more than the flat read'),
        # The flat read GAINED while the selected transform paid.  This is
        # the committed-data case, and the one the `abs()` form got wrong.
        # Strictly `cost_flat < 0` — a flat read that did not move is the
        # separate case below, not this one.
        (+0.10, -0.30, -1, 'opposite directions'),
        (-1.00, +3.13, +1, 'opposite directions'),
        # The flat read did not move AT ALL while the selected transform
        # paid.  One arm standing still is not two arms moving in opposite
        # directions, and the sentence must not say so beside a `0.00` cell.
        (0.0, -0.30, -1, 'had no cost to absorb'),
        (0.0, +3.13, +1, 'had no cost to absorb'),
    ])
    def test_the_branch_table(self, flat, selected, cost_sign, expected):
        phrase = _mod()._absorption_phrase(flat, selected, cost_sign=cost_sign)
        assert expected in phrase, phrase

    @pytest.mark.parametrize('flat,selected,cost_sign', [
        (+0.10, +0.20, -1),          # neither paid — name both, not "nothing"
        (-0.10, +0.30, -1),          # the sign flip
        (+0.0042372881355934, -0.0042372881355934, -1),   # committed data
        (-1.00, +3.13, +1),          # opposite directions
        (0.0, -0.30, -1),            # flat did not move; selected paid
    ])
    def test_it_names_both_signed_values_wherever_the_two_arms_disagree(
        self, flat, selected, cost_sign,
    ):
        """A reader told "nothing happened" when both arms GAINED, or told
        one arm's number when the two moved in opposite directions, cannot
        reconstruct the table from the sentence.
        """
        mod = _mod()
        phrase = mod._absorption_phrase(flat, selected, cost_sign=cost_sign)
        assert mod._gap_cell(flat) in phrase, phrase
        assert mod._gap_cell(selected) in phrase, phrase

    def test_the_committed_data_case_is_not_reported_as_no_change(self):
        """The regression anchor, pinned by VALUE.

        These two floats are the committed
        `deltas.unstamped.{flat,promoting_pin}.claim_recall.at_5`: the flat
        read GAINED 0.0042 while `promoting_pin` LOST 0.0042.  The `abs()`
        form called that 'does not change it (-<0.01, the same distance from
        baseline as the flat read)' — two opposite findings declared
        equivalent in the probe's headline sentence.

        They are literals this test carries, not a threshold on a
        re-measurement: the assertion is about the sentence's LOGIC and
        stays valid whatever the next run measures.
        """
        mod = _mod()
        phrase = mod._absorption_phrase(
            +0.0042372881355934, -0.0042372881355934, cost_sign=-1,
        )
        assert 'does not change it' not in phrase, phrase
        assert mod._gap_cell(+0.0042372881355934) in phrase, phrase
        assert mod._gap_cell(-0.0042372881355934) in phrase, phrase

    @pytest.mark.parametrize('a,b,cost_sign', [
        (-0.10, 0.0, -1),
        (-0.10, -0.20, -1),
        (+0.10, +0.20, -1),
        (-0.10, +0.30, -1),
        (+1.00, +3.13, +1),
        (+0.0042372881355934, -0.0042372881355934, -1),
    ])
    def test_the_two_arguments_are_not_interchangeable(self, a, b, cost_sign):
        """Symmetry is NOT claimed, so a transposed call site cannot be
        silent: the flat read and the selected transform play different
        roles in the sentence and swapping them must change what it says.
        """
        mod = _mod()
        assert (
            mod._absorption_phrase(a, b, cost_sign=cost_sign)
            != mod._absorption_phrase(b, a, cost_sign=cost_sign)
        )


class TestCostPhrase:
    """One delta, rendered as a cost or a gain — never hard-typed.

    The committed artifact reads 'one unstamped re-emission per topic costs
    <0.01 claim recall@5' for a measured GAIN of +0.0042, because the
    sentence typed the word "costs" around `_gap_cell`.  The verb has to be
    derived from the same `cost_sign` the absorption phrase branches on.
    """

    @pytest.mark.parametrize('delta,cost_sign,expected', [
        # cost_sign -1: a NEGATIVE delta is the cost.
        (-0.10, -1, 'costs'),
        (+0.10, -1, 'gains'),
        (-0.0042372881355934, -1, 'costs'),
        (+0.0042372881355934, -1, 'gains'),
        # cost_sign +1: a POSITIVE delta is the cost (tokens, rank).
        (+3.13, +1, 'costs'),
        (-3.13, +1, 'gains'),
    ])
    def test_the_verb_follows_the_cost_direction(
        self, delta, cost_sign, expected,
    ):
        phrase = _mod()._cost_phrase(delta, cost_sign=cost_sign)
        assert phrase.startswith(expected), phrase

    @pytest.mark.parametrize('cost_sign', [-1, +1])
    def test_an_exact_zero_did_not_move(self, cost_sign):
        phrase = _mod()._cost_phrase(0.0, cost_sign=cost_sign)
        assert 'does not move' in phrase, phrase
        assert 'costs' not in phrase, phrase
        assert 'gains' not in phrase, phrase

    @pytest.mark.parametrize('cost_sign', [-1, +1])
    def test_a_never_measured_delta_says_so(self, cost_sign):
        """`None` is "no measurement", which is a different finding from
        "measured zero" everywhere else in this renderer and stays different
        here.
        """
        phrase = _mod()._cost_phrase(None, cost_sign=cost_sign)
        assert 'never measured' in phrase, phrase
        assert 'costs' not in phrase, phrase
        assert 'gains' not in phrase, phrase
        assert 'does not move' not in phrase, phrase

    @pytest.mark.parametrize('delta,cost_sign', [
        (-0.10, -1), (+0.10, -1), (+3.13, +1), (-3.13, +1),
        (+0.0042372881355934, -1),
    ])
    def test_it_carries_the_magnitude_through_the_shared_formatter(
        self, delta, cost_sign,
    ):
        """`_gap_cell` stays the number formatter — this changes the VERB,
        not the formatting, so `<0.01` (moved, too little to round up) can
        never come out as `0.00` (measured, did not move).
        """
        mod = _mod()
        phrase = mod._cost_phrase(delta, cost_sign=cost_sign)
        assert mod._gap_cell(abs(delta)) in phrase, phrase


# ===========================================================================
# step-19 — the stamping table's zeros are STRUCTURALLY FORCED
# ===========================================================================
#
# THE DEFECT.  `### What topic-stamping buys` publishes `stamped delta -
# unstamped delta` per read arm, and the prose attaches a causal claim to it
# ("the difference is what the campaign buys against regrowth").  On the
# committed artifact every `after.unstamped.flat` value is bit-identical to
# `after.stamped.flat`, `stamping_value['flat']` and
# `stamping_value['additive_pin']` are all-`0.0`, and only
# `promoting_pin.tokens_per_query.mean` is nonzero.  A reader takes away
# "stamping buys 0.00 everywhere" — the wrong input for task 4006.
#
# Those zeros are not measurements.  Three separate ceilings force them, and
# each is mechanically checkable, so each gets a test: if a ceiling stops
# being true the disclosure that names it must fail loudly rather than go on
# explaining away a number that has become real.
#
# Pure — every number below is a literal this file wrote or a structural
# equality between two runs of the same pure function over hand-built data.
# NO threshold or bound on any measured quantity (G6).


def _ceiling_arm(mode: str, *, k: int = 5):
    """A hand-built `c_peers` arm with ONE injected re-emission, in `mode`.

    The two modes differ in EXACTLY the way the real injections do — a
    `topic` key on the injected record and nothing else — so any measured
    difference between them is a difference a read arm expressed from
    metadata alone.

    The canonical sits LAST in the ranking, outside a `k`-window: that is
    both what makes the additive pin have something to add (ceiling 2) and
    what keeps every discoverability metric measurable rather than `None`
    (`topic_discoverability` scores the RAW hits, so a canonical outside the
    read window is still found).
    """
    mod = _mod()
    canonical = _rec('canon', topic='t', canonical=True, claim_ids=['k-canon'])
    peers = [_rec(f'p{i}', topic='t', claim_ids=[f'k{i}']) for i in range(k - 1)]
    metadata: dict = {'category': 'procedural_knowledge'}
    if mode == 'stamped':
        metadata['topic'] = 't'
    injection = mod.ArmRecord(
        record_id='regrow-1',
        content='the canonical claim, restated in different words',
        metadata=metadata,
        cluster_id='c1',
        claim_ids=['k-canon'],
        role=mod.REGROWTH_ROLE,
    )
    records = [canonical, *peers, injection]
    seeded = _seeded(
        'c_peers', records,
        canonical_by_topic={'t': canonical},
        canonical_by_cluster={'c1': 'canon'},
        siblings_by_cluster={'c1': {r.record_id for r in (*peers, injection)}},
    )
    ranked = [*peers, injection, canonical]
    hits = [_sh(record, 0.9 - i / 100.0) for i, record in enumerate(ranked)]
    return seeded, hits


def _ceiling_base(*, k: int = 5):
    """The same arm with NO injection — the baseline the deltas subtract."""
    canonical = _rec('canon', topic='t', canonical=True, claim_ids=['k-canon'])
    peers = [_rec(f'p{i}', topic='t', claim_ids=[f'k{i}']) for i in range(k - 1)]
    seeded = _seeded(
        'c_peers', [canonical, *peers],
        canonical_by_topic={'t': canonical},
        canonical_by_cluster={'c1': 'canon'},
        siblings_by_cluster={'c1': {r.record_id for r in peers}},
    )
    hits = [_sh(record, 0.9 - i / 100.0) for i, record in enumerate([*peers, canonical])]
    return seeded, hits


def _ceiling_arms(seeded, hits, *, limit: int = 5) -> dict:
    """`measure_regrowth_arms` over one hand-built ranking, projected flat."""
    mod = _mod()
    return mod._plucked_regrowth_arms(mod.measure_regrowth_arms(
        seeded,
        {'queries': {'q1': hits}, 'probes': {'c1': hits}},
        queries=[_query()],
        probes=[],
        estimator=_CHARS,
        guard_threshold=0.92,
        limit=limit,
    ))


@functools.cache
def _ceiling_stamping_value() -> dict:
    """`stamping_value` end to end over the hand-built pair.

    Same arithmetic the artifact's table is built from — `measure_arm` is
    pure over `(SeededArm, fetched)`, so the whole chain runs with no store.
    """
    mod = _mod()
    baseline = _ceiling_arms(*_ceiling_base())
    deltas = {
        mode: mod.regrowth_deltas(baseline, _ceiling_arms(*_ceiling_arm(mode)))
        for mode in mod.REGROWTH_MODES
    }
    return mod.regrowth_stamping_value(deltas)


class TestTheReseedingControlPhrase:
    """The artifact used to CLAIM a delta carried no cross-seeding noise.

    It does not: `after` is measured over `c_peers_regrowth_<mode>` while the
    baseline is measured over `c_peers`, so a baseline-vs-after delta spans
    two seedings.  Only the three read arms within one mode are seeding-free.
    What the artifact reports instead is the noise floor it actually has —
    the `flat` stamping row, whose two sides are two separately seeded
    injected collections differing only in a metadata key a flat read never
    consults.
    """

    @staticmethod
    def _flat(**overrides):
        return {
            metric: overrides.get(metric, 0.0)
            for metric in _mod()._regrowth_metric_keys()
        }

    def test_an_all_zero_row_reports_that_re_seeding_moved_nothing(self):
        phrase = _mod()._reseeding_control_phrase(self._flat())

        assert 'exactly `0.00` on every column' in phrase, phrase

    def test_a_nonzero_row_says_so_and_names_every_column_that_moved(self):
        """The point of DERIVING it.  A typed "re-seeding contributes
        nothing" would still read as a guarantee on the first run where it
        stopped being true, three lines above the table that disagrees.

        Every moved cell is named, not just one: the floor is per column, so
        a reader needs the cell belonging to the column they are reading.
        """
        mod = _mod()
        flat = self._flat(**{'claim_recall.at_5': -0.25,
                             'tokens_per_query.mean': +0.10})

        phrase = mod._reseeding_control_phrase(flat)

        assert 'did NOT come out flat' in phrase, phrase
        assert mod._gap_cell(-0.25) in phrase, phrase
        assert mod._gap_cell(+0.10) in phrase, phrase
        assert 'claim recall@5' in phrase, phrase
        assert 'tokens/query' in phrase, phrase

    def test_the_floor_is_per_column_not_a_cross_unit_maximum(self):
        """The seven metrics are in four different units, so the row must
        never be collapsed to one scalar and published as a floor on every
        column.  A `tokens/query` cell of `-3.13` beside an otherwise flat
        row must not be rendered as a bound on a claim-recall delta: the
        sentence names the ONE column that moved, says the floor is per
        column, and does not claim that much of "any delta below" is noise.
        """
        mod = _mod()
        flat = self._flat(**{'tokens_per_query.mean': -3.13})

        phrase = mod._reseeding_control_phrase(flat)

        assert mod._gap_cell(-3.13) in phrase, phrase
        assert 'tokens/query' in phrase, phrase
        assert 'PER COLUMN' in phrase, phrase
        # The old cross-unit overclaim, in the two spellings it took.
        assert 'largest cell' not in phrase, phrase
        assert 'any delta below' not in phrase, phrase
        # A column that did not move is not bounded by another one's cell.
        assert 'claim recall@5' not in phrase, phrase

    def test_every_moved_column_is_named_with_the_tables_own_label(self):
        """The floor is only usable if the reader can match a cell to the
        column it bounds, and the stamping table's headers are
        `REGROWTH_METRIC_LABELS` — so the sentence must use those spellings
        rather than the internal `<block>.<key>` metric keys."""
        mod = _mod()
        metrics = mod._regrowth_metric_keys()
        labels = dict(zip(metrics, mod.REGROWTH_METRIC_LABELS, strict=True))
        flat = self._flat(**{metric: 0.5 for metric in metrics})

        phrase = mod._reseeding_control_phrase(flat)

        for metric in metrics:
            assert labels[metric] in phrase, (metric, phrase)
            assert metric not in phrase, (metric, phrase)

    def test_an_unmeasured_row_says_so_rather_than_reporting_a_zero(self):
        """`None` is "never asked", and reporting it as a clean control is
        exactly the overclaim this function exists to stop making."""
        mod = _mod()
        flat = {metric: None for metric in mod._regrowth_metric_keys()}

        assert 'not measured' in mod._reseeding_control_phrase(flat)

    def test_the_rendered_section_carries_it_rather_than_the_old_guarantee(self):
        """The published claim, narrowed.

        The section must not tell a gate reader that a delta contains the
        re-emission's contribution "and not ANN noise between two seedings" —
        a methodological guarantee the two-collection design does not
        provide.  What it may say is which comparison IS seeding-free, and
        what the measured noise floor came out as.
        """
        mod = _mod()
        block = _regrowth_block()
        section = '\n'.join(_section(
            mod.render_markdown(_report_with_regrowth(block)),
            '## Regrowth deltas',
        ))

        assert mod._reseeding_control_phrase(
            block['stamping_value']['flat']
        ) in section
        assert 'not ANN noise between two seedings' not in section


class TestTheStampingTableCeilings:
    """Why two of the three rows in `### What topic-stamping buys` are zero.

    Not "stamping is worth nothing" — "this arm could not express a
    difference".  The disclosure the renderer emits states these three
    premises in prose; these tests are the premises themselves.
    """

    # --- ceiling 1: a metadata-blind read arm ---------------------------

    def test_the_two_modes_emit_byte_identical_record_text(self):
        """mem0 embeds the record CONTENT, so the two modes are the same
        corpus to the embedder.  Whatever a stamping campaign changes, it is
        not what the vectors look like.
        """
        assert (
            [r.content for r in _regrowth_records('unstamped')]
            == [r.content for r in _regrowth_records('stamped')]
        )

    def test_the_two_modes_differ_in_exactly_the_topic_key(self):
        for unstamped, stamped in zip(
            _regrowth_records('unstamped'), _regrowth_records('stamped'),
            strict=True,
        ):
            assert (set(unstamped.metadata) ^ set(stamped.metadata)) == {'topic'}
            assert {
                key: value for key, value in stamped.metadata.items()
                if key != 'topic'
            } == dict(unstamped.metadata)

    def test_a_metadata_blind_read_arm_cannot_express_a_stamping_difference(self):
        """`read_path(pin=False)` slices `hits[:k]` and consults no metadata.

        So the same ranking measured under the two modes returns EQUAL
        metrics, and it could not have been otherwise.
        """
        mod = _mod()
        measured = {
            mode: mod._pluck_regrowth_metrics(
                _measure(*_ceiling_arm(mode), pin=False, promote=False, limit=5)
            )
            for mode in mod.REGROWTH_MODES
        }

        assert measured['unstamped'] == measured['stamped']

    def test_the_flat_rows_stamping_value_is_zero_on_every_metric(self):
        """The property the disclosure claims, asserted end to end.

        A structural zero, not a measurement: it is the difference between
        two runs of the same pure function over inputs that differ only in a
        key that function never reads.
        """
        flat = _ceiling_stamping_value()['flat']

        assert set(flat) == set(_mod()._regrowth_metric_keys())
        assert all(value == 0.0 for value in flat.values()), flat

    # --- ceiling 2: a full window has no room for an additive pin -------

    def test_a_full_window_additive_pin_cannot_express_one_either(self):
        """`apply_topic_anchor` APPENDS and `read_path` truncates AFTER.

        At `len(window) == k` the pinned canonical is taken straight back
        off — the same thing `pin changed window = 0.00` reports elsewhere
        in this artifact — so the additive arm's window is the flat arm's
        window in BOTH modes.
        """
        mod = _mod()
        for mode in mod.REGROWTH_MODES:
            seeded, hits = _ceiling_arm(mode)
            assert len(hits) > 5, 'the window must be full for this ceiling'

            flat = mod.read_path(seeded, hits, 5, pin=False, promote=False)
            pinned = mod.read_path(seeded, hits, 5, pin=True, promote=False)

            assert [r.record_id for r in pinned] == [r.record_id for r in flat]
            assert 'canon' not in [r.record_id for r in pinned]

    def test_the_additive_pin_rows_stamping_value_is_zero_on_every_metric(self):
        additive = _ceiling_stamping_value()['additive_pin']

        assert set(additive) == set(_mod()._regrowth_metric_keys())
        assert all(value == 0.0 for value in additive.values()), additive

    # --- ceiling 3: arm (c) stamps every peer ---------------------------

    def test_arm_c_stamps_every_peer_so_the_pin_usually_fires_from_a_sibling(self):
        """`_materialize_c_peers` sets `topic` unconditionally.

        So the injection's OWN stamp can only change the pin's firing when
        the injection is the sole windowed record for its topic.  The probe
        therefore measures stamping value against a 100%-stamped corpus —
        the exact inverse of the live corpus esc-3200-3 documents, where
        `count_memories_by_metadata(topic=...)` returned 1.  That bounds
        `promoting_pin`'s stamping value from ABOVE; the inference is what
        the disclosure states in prose, this asserts the arm-c property it
        rests on.
        """
        records = _knowledge(_arm('c_peers'))

        assert records
        for record in records:
            assert 'topic' in record.metadata, record.record_id

    def test_every_injected_record_shares_its_topic_with_the_arm_c_peers(self):
        """The other half of the same ceiling: the injection is never alone
        in its topic, because arm (c) already stamped every peer of the
        cluster it belongs to.
        """
        stamped_topics = {
            record.metadata['topic'] for record in _knowledge(_arm('c_peers'))
        }

        for injection in _injections():
            assert injection.topic in stamped_topics


class TestRenderMarkdownRegrowthSection:
    """The operator-facing half of the probe.

    The JSON block is the machine-readable record; this section is what a
    reader of the artifact actually reads, and esc-3200-3's finding was that
    a probe with no rendered section is indistinguishable from a probe that
    never ran.
    """

    def test_the_section_sits_between_by_query_kind_and_d10(self):
        """Asserted by relative index, so a section that MOVES fails here.

        The operator's reading order is part of the artifact: the deltas
        qualify the arm tables above them and are qualified by nothing in
        D10, so a section that silently relocated would change what the
        reader takes the numbers to mean.
        """
        lines = _mod().render_markdown(_report_with_regrowth()).splitlines()

        assert (
            lines.index('## By query kind')
            < lines.index('## Regrowth deltas')
            < lines.index('## D10 — audit-recall over the labeled fixture')
        )

    def test_the_metric_labels_line_up_one_for_one_with_the_metrics(self):
        """A label tuple shorter than the metric tuple drops a column."""
        mod = _mod()

        assert len(mod.REGROWTH_METRIC_LABELS) == len(mod.REGROWTH_METRICS)
        # Sliced rather than rebuilt as `(*prefix, *labels)` because that
        # spelling trips ruff's SIM300 (a SCREAMING_SNAKE attribute on the
        # left of a non-literal tuple reads to it as a Yoda condition).  The
        # slice form says the same thing more directly anyway: each table is
        # its own row-key prefix followed by the labels, in label order.
        assert mod.REGROWTH_TABLE_COLUMNS[:2] == ('mode', 'read arm')
        assert mod.REGROWTH_TABLE_COLUMNS[2:] == mod.REGROWTH_METRIC_LABELS
        assert mod.REGROWTH_STAMPING_COLUMNS[:1] == ('read arm',)
        assert mod.REGROWTH_STAMPING_COLUMNS[1:] == mod.REGROWTH_METRIC_LABELS

    def test_the_delta_table_columns_are_pinned_by_equality(self):
        """Exactly as `DECISION_TABLE_COLUMNS` is, and for the same reason:
        a column quietly dropped from a delta table is a metric quietly
        dropped from the decision."""
        mod = _mod()

        assert mod.REGROWTH_TABLE_COLUMNS == (
            'mode',
            'read arm',
            'claim recall@5',
            'claim recall@10',
            'canonical in top-5 (stored)',
            'median canonical rank (stored)',
            'canonical found (stored)',
            'canonical in top-5 (credited)',
            'tokens/query',
        )

    def test_the_rendered_header_row_is_built_from_the_pinned_columns(self):
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        assert _header_row(mod.REGROWTH_TABLE_COLUMNS) in section
        assert _header_row(mod.REGROWTH_STAMPING_COLUMNS) in section

    def test_there_is_one_row_per_mode_and_arm_in_pinned_order(self):
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        rows = _rows_under(section, _header_row(mod.REGROWTH_TABLE_COLUMNS))

        assert len(rows) == len(mod.REGROWTH_MODES) * len(mod.REGROWTH_READ_ARMS)
        assert [tuple(_cells(row)[:2]) for row in rows] == [
            (mode, arm)
            for mode in mod.REGROWTH_MODES
            for arm in mod.REGROWTH_READ_ARMS
        ]

    def test_every_row_carries_a_cell_for_every_metric(self):
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        for row in _rows_under(section, _header_row(mod.REGROWTH_TABLE_COLUMNS)):
            assert len(_cells(row)) == len(mod.REGROWTH_TABLE_COLUMNS)

    def test_a_metric_cell_carries_the_baseline_the_after_and_the_delta(self):
        """All three, in the one cell the reader is looking at.

        The numbers are the test's own: `_plucked` puts claim recall@5 at
        0.5 and the override moves the injected pass to 0.25, so the cell
        must read `0.50 → 0.25 (-0.25)` and nothing else.
        """
        mod = _mod()
        block = _moved_block(**{'claim_recall.at_5': 0.25})
        section = _section(
            mod.render_markdown(_report_with_regrowth(block)),
            '## Regrowth deltas',
        )

        row = next(
            r for r in _rows_under(
                section, _header_row(mod.REGROWTH_TABLE_COLUMNS))
            if _cells(r)[:2] == ['unstamped', 'flat']
        )

        recall_at_5 = _cells(row)[mod.REGROWTH_TABLE_COLUMNS.index('claim recall@5')]
        assert recall_at_5 == '0.50 → 0.25 (-0.25)'

    def test_an_unmoved_metric_renders_a_measured_zero_delta(self):
        """`0.00` here means "measured, and it did not move" — the finding
        the None rendering below has to stay distinguishable from."""
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        row = next(
            r for r in _rows_under(
                section, _header_row(mod.REGROWTH_TABLE_COLUMNS))
            if _cells(r)[:2] == ['stamped', 'flat']
        )

        assert _cells(row)[2] == '0.50 → 0.50 (0.00)'

    def test_a_none_metric_renders_as_no_measurement_never_zero(self):
        """On BOTH the value and the delta, in the same cell.

        A delta table that prints "never measured" as `0.00` says the
        injection changed nothing, which is a finding rather than an
        absence — the same discipline `_NO_MEASUREMENT` exists for.
        """
        mod = _mod()
        block = _moved_block(**{'claim_recall.at_5': None})
        section = _section(
            mod.render_markdown(_report_with_regrowth(block)),
            '## Regrowth deltas',
        )

        row = next(
            r for r in _rows_under(
                section, _header_row(mod.REGROWTH_TABLE_COLUMNS))
            if _cells(r)[:2] == ['unstamped', 'flat']
        )

        cell = _cells(row)[2]
        assert cell == f'0.50 → {mod._NO_MEASUREMENT} ({mod._NO_MEASUREMENT})'
        assert '0.00' not in cell

    def test_the_stamping_table_carries_one_row_per_read_arm(self):
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        rows = _rows_under(
            section, _header_row(mod.REGROWTH_STAMPING_COLUMNS))

        assert [_cells(row)[0] for row in rows] == list(mod.REGROWTH_READ_ARMS)
        for row in rows:
            assert len(_cells(row)) == len(mod.REGROWTH_STAMPING_COLUMNS)

    def test_the_stamping_table_renders_stamped_minus_unstamped(self):
        """The number task 4006's stamping campaign is owed.

        Baseline and the stamped pass are identical here and the unstamped
        pass moved by -0.25, so the stamping value is +0.25 — arithmetic
        over values this test wrote, not a measurement.
        """
        mod = _mod()
        block = _moved_block(**{'claim_recall.at_5': 0.25})
        section = _section(
            mod.render_markdown(_report_with_regrowth(block)),
            '## Regrowth deltas',
        )

        row = next(
            r for r in _rows_under(
                section, _header_row(mod.REGROWTH_STAMPING_COLUMNS))
            if _cells(r)[0] == 'flat'
        )

        assert _cells(row)[1] == '0.25'

    def test_the_section_cites_the_task_its_stamping_table_informs(self):
        """The task id is an IDENTIFIER a consumer keys on; the surrounding
        prose is free to be reworded.  Pinning the word "stamp" pinned
        nothing executable — a section gutted to the literal string
        'stamp 4006' would have passed it.
        """
        mod = _mod()
        section = '\n'.join(_section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )).lower()

        assert '4006' in section

    def test_each_read_arm_gets_exactly_one_regrowth_bullet(self):
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        for arm in mod.REGROWTH_READ_ARMS:
            anchor = mod.regrowth_bullet_prefix(arm)
            assert sum(
                1 for line in section if line.startswith(anchor)
            ) == 1, f'expected exactly one {anchor!r} bullet'

    def test_the_regrowth_anchor_is_distinct_from_the_other_two(self):
        """Three bullet lists share this document.

        `pin_bullet_prefix` and `stored_gap_bullet_prefix` each assert
        "exactly one" over their own anchor, so a regrowth bullet that
        collided with either would break a test in a distant section rather
        than here.
        """
        mod = _mod()

        anchor = mod.regrowth_bullet_prefix('flat')
        assert anchor != mod.stored_gap_bullet_prefix('flat')
        assert anchor != mod.pin_bullet_prefix('flat')

    def test_the_bullets_are_derived_from_the_block_not_typed(self):
        """Same rule as the pin bullets: a hand-typed number about a previous
        run silently becomes a false sentence beside the table that
        contradicts it."""
        mod = _mod()
        block = _moved_block(**{'claim_recall.at_5': 0.25})
        section = _section(
            mod.render_markdown(_report_with_regrowth(block)),
            '## Regrowth deltas',
        )

        bullet = next(
            line for line in section
            if line.startswith(mod.regrowth_bullet_prefix('flat'))
        )

        assert '-0.25' in bullet

    def test_the_section_discloses_credited_versus_stored_semantics(self):
        """Neither column can be quoted without its semantics.

        Under `promoting_pin` the credited column is a PLACEMENT property —
        the transform injects the canonical into the window — exactly as
        `apply_grouped_read`'s was under `b_grouped`.

        VERBATIM identity on the module constant, not substrings of the
        rendered prose.  The substring form this replaced asserted `'stored'`
        and `'promoting_pin'` were somewhere in the section — both of which
        the delta TABLE supplies on its own (`canonical in top-5 (stored)` is
        a column header, `promoting_pin` labels three rows), so deleting the
        whole paragraph left it green.  A disclosure test that survives the
        disclosure's deletion is not a disclosure test.
        """
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        assert mod.REGROWTH_CREDITED_SEMANTICS_DISCLOSURE in section

    def test_the_credited_semantics_sit_between_the_delta_table_and_the_next(self):
        """Asserted by relative index, like the stamping ceiling below.

        It qualifies the two `canonical in top-5` columns of the delta table,
        so it belongs after that table and before the next heading takes the
        reader somewhere else.  A qualifier a section away is not a qualifier
        on this number.
        """
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        table_at = section.index(_header_row(mod.REGROWTH_TABLE_COLUMNS))
        disclosure_at = section.index(mod.REGROWTH_CREDITED_SEMANTICS_DISCLOSURE)
        next_heading_at = section.index('### What topic-stamping buys')

        assert table_at < disclosure_at < next_heading_at

    def test_the_credited_semantics_are_not_emitted_with_no_table_to_qualify(self):
        """A `--no-regrowth` run has no columns for this to be about."""
        mod = _mod()

        assert mod.REGROWTH_CREDITED_SEMANTICS_DISCLOSURE not in (
            mod.render_markdown(_report())
        )

    def test_the_section_says_what_was_injected_and_what_the_modes_mean(self):
        mod = _mod()
        section = '\n'.join(_section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )).lower()

        for mode in mod.REGROWTH_MODES:
            assert mode in section
        assert mod.REGROWTH_SHAPE in section

    def test_a_run_without_the_probe_still_emits_the_heading(self):
        """An ABSENT section is how this probe went missing the first time.

        A reader of a probe-less artifact must be able to tell "skipped"
        from "this build predates the probe", and a heading that disappears
        makes those two identical — which is precisely what esc-3200-3 could
        not read off the previous artifact.

        `_section` locates by the heading and raises if it is absent, so the
        emission contract is carried by the call itself.  What the skipped
        section SAYS is operator-facing prose and free to be reworded; a
        substring pin on it would have passed for a section gutted to the
        literal words it pinned.  What is checked instead is that neither
        table is there to be misread as a measurement.
        """
        mod = _mod()

        section = _section(
            mod.render_markdown(_report()), '## Regrowth deltas',
        )

        assert _header_row(mod.REGROWTH_TABLE_COLUMNS) not in section
        assert _header_row(mod.REGROWTH_STAMPING_COLUMNS) not in section

    def test_rendering_is_byte_identical_for_identical_input(self):
        mod = _mod()

        assert (
            mod.render_markdown(_report_with_regrowth())
            == mod.render_markdown(_report_with_regrowth())
        )


    # --- step-19: the stamping table's forced zeros, disclosed -----------

    def test_the_stamping_ceiling_disclosure_appears_verbatim_in_the_section(self):
        """VERBATIM identity on a module constant, not substrings of prose.

        The convention `test_the_disclosure_appears_verbatim_in_the_protocol_
        section` already established: the constant's TEXT is operator-facing
        prose and free to change; what is pinned is that it is there and
        where it is.
        """
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        assert mod.REGROWTH_STAMPING_CEILING_DISCLOSURE in section

    def test_it_sits_between_the_stamping_heading_and_the_stamping_table(self):
        """Asserted by relative index: a disclosure that drifts away from the
        number it qualifies is not a disclosure.
        """
        mod = _mod()
        section = _section(
            mod.render_markdown(_report_with_regrowth()), '## Regrowth deltas',
        )

        heading_at = section.index('### What topic-stamping buys')
        disclosure_at = section.index(mod.REGROWTH_STAMPING_CEILING_DISCLOSURE)
        header_at = section.index(_header_row(mod.REGROWTH_STAMPING_COLUMNS))

        assert heading_at < disclosure_at < header_at

    def test_it_is_not_emitted_when_there_is_no_stamping_table_to_qualify(self):
        """Unlike the blind-authoring disclosure, which describes the probe's
        AUTHORING and is emitted always, this one qualifies a specific table.
        On a `--no-regrowth` run there is no table and nothing to qualify.
        """
        mod = _mod()

        assert mod.REGROWTH_STAMPING_CEILING_DISCLOSURE not in mod.render_markdown(
            _report())


class TestTheNotBlindAuthoredDisclosure:
    """This probe does NOT carry the protection the six arms above it do."""

    def test_the_disclosure_appears_verbatim_in_the_protocol_section(self):
        mod = _mod()
        rendered = mod.render_markdown(_report_with_regrowth())

        assert mod.REGROWTH_BLIND_AUTHORING_DISCLOSURE in rendered
        assert mod.REGROWTH_BLIND_AUTHORING_DISCLOSURE in '\n'.join(
            _section(rendered, '## Protocol'))

    def test_it_is_emitted_even_when_the_probe_was_skipped(self):
        """The disclosure describes the probe's authoring, not its run.

        A reader of a `--no-regrowth` artifact still needs to know what the
        probe's protocol is when they go looking for its numbers.
        """
        mod = _mod()

        assert mod.REGROWTH_BLIND_AUTHORING_DISCLOSURE in mod.render_markdown(
            _report())

    def test_the_injection_fixture_renders_as_a_row_in_the_fixture_table(self):
        mod = _mod()
        protocol = _protocol()
        protocol['fixtures'].append({
            'path': 'fused-memory/tests/fixtures/e2_regrowth_injection.jsonl',
            'commit': 'cafe123',
        })

        rendered = mod.render_markdown(
            _report_with_regrowth(protocol=protocol))

        assert (
            '| `fused-memory/tests/fixtures/e2_regrowth_injection.jsonl` '
            '| cafe123 |'
        ) in rendered


class TestTheInjectionFixtureIsClaimedOnlyWhenItWasRead:
    """Provenance for a fixture the run never opened is a false audit trail."""

    def test_it_is_appended_when_the_probe_ran(self):
        mod = _mod()

        paths = mod._protocol_fixture_paths(['a', 'b'], regrowth={})

        assert paths[:2] == ['a', 'b']
        assert paths[-1] == mod.DEFAULT_REGROWTH_INJECTION_PATH

    def test_the_gate_is_is_not_none_rather_than_truthiness(self):
        """An empty block is still a block that read the fixture."""
        mod = _mod()

        assert mod.DEFAULT_REGROWTH_INJECTION_PATH in mod._protocol_fixture_paths(
            [], regrowth={})

    def test_it_is_absent_when_the_probe_was_skipped(self):
        mod = _mod()

        assert mod._protocol_fixture_paths(['a', 'b'], regrowth=None) == ['a', 'b']


class TestReadPathHoldsTheWindowBudget:
    """Pin-on and pin-off must be scored over equal-size windows."""

    @pytest.mark.parametrize('shape', ['status_quo', 'c_peers', 'b_grouped'])
    @pytest.mark.parametrize('k', [5, 10])
    def test_a_pinned_window_is_never_wider_than_k(self, shape, k):
        mod = _mod()
        seeded, hits = _full_window_arm(shape)

        assert len(mod.read_path(seeded, hits, k, pin=True)) <= k

    @pytest.mark.parametrize('shape', ['status_quo', 'c_peers', 'b_grouped'])
    @pytest.mark.parametrize('k', [5, 10])
    def test_a_full_fetch_gives_both_variants_the_same_size_window(self, shape, k):
        """The A/B is only controlled if the two arms get the same budget."""
        mod = _mod()
        seeded, hits = _full_window_arm(shape)

        off = mod.read_path(seeded, hits, k, pin=False)
        on = mod.read_path(seeded, hits, k, pin=True)

        assert len(on) == len(off) == k

    def test_the_pre_transform_truncation_still_stands(self):
        """A record the store never returned must not enter the window, so the
        transforms still act on `hits[:k]` and not on the whole fetch."""
        mod = _mod()
        seeded, hits = _full_window_arm('c_peers', n=12)

        window = mod.read_path(seeded, hits, 5, pin=False)

        assert [r.record_id for r in window] == ['p0', 'p1', 'p2', 'p3', 'p4']


class TestGroupingKeepsItsAdvantage:
    """The fix must not take back grouping's legitimate win."""

    def _grouped_arm(self):
        """Four children of one parent, plus one hit on a second topic.

        The grouped read collapses the four into one document, taking a
        5-record window down to 2 — real headroom, freed by the storage shape
        rather than by a wider budget.
        """
        parent = _rec(PARENT, topic='t1', canonical=True, claim_ids=['k-par'])
        children = [
            _rec(f'child-{i}', parent_id=PARENT, kind='amendment', topic='t1',
                 claim_ids=[f'k{i}'])
            for i in range(4)
        ]
        other = _rec('other-1', topic='t2', claim_ids=['k-other'])
        anchor = _rec('canon-t2', topic='t2', canonical=True, claim_ids=['k-t2'])
        hits = [_sh(record, 0.9 - i / 100.0)
                for i, record in enumerate([*children, other])]
        seeded = _seeded(
            'b_grouped',
            [parent, *children, other, anchor],
            canonical_by_topic={'t1': parent, 't2': anchor},
            canonical_by_cluster={'c1': PARENT},
            siblings_by_cluster={'c1': {c.record_id for c in children}},
        )
        return seeded, hits, anchor

    def test_a_collapsed_window_stays_short_it_is_not_padded_back_to_k(self):
        mod = _mod()
        seeded, hits, _ = self._grouped_arm()

        window = mod.read_path(seeded, hits, 5, pin=False)

        assert len(window) == 2  # the four children became one document
        assert {r.record_id for r in window} == {PARENT, 'other-1'}

    def test_a_pin_landing_in_freed_headroom_survives_the_truncation(self):
        """Grouping frees a slot; the pin fills it.  THAT is the pin's real
        win, and truncating at k must not delete it."""
        mod = _mod()
        seeded, hits, anchor = self._grouped_arm()

        window = mod.read_path(seeded, hits, 5, pin=True)

        assert len(window) == 3
        assert anchor.record_id in {r.record_id for r in window}


class TestTopicDiscoverabilityRespectsTheLiteralK:
    """A rank-6 record is not "in the top 5", whatever the caller passes."""

    def test_a_canonical_past_k_is_reported_absent_but_its_rank_survives(self):
        hits = [_rec(f'r{i}', topic='t') for i in range(5)]
        hits.append(_rec('canon', topic='t', canonical=True))

        found = _mod().topic_discoverability(hits, 't', 'canon', 5)

        assert found['canonical_in_top_k'] is False
        assert found['canonical_rank'] == 6  # "nearly there" is its own finding

    def test_the_boundary_rank_equal_to_k_is_inside_the_window(self):
        hits = [_rec(f'r{i}', topic='t') for i in range(4)]
        hits.append(_rec('canon', topic='t', canonical=True))

        found = _mod().topic_discoverability(hits, 't', 'canon', 5)

        assert found['canonical_in_top_k'] is True
        assert found['canonical_rank'] == 5


class TestTheSelfHitFilterIsProvenanceBasedNotIdBased:
    """Pure — the bias is a property of the committed fixtures, not the store."""

    def test_the_self_hit_filter_bites_in_the_PEER_arms_not_only_the_baseline(
        self,
    ):
        """An id-equality filter is a NO-OP outside `status_quo`.

        `record_id == memory_id` holds only in `_materialize_status_quo`; the
        peer arms derive every id through `_derive_record_id`.  So filtering
        on id removes the self-match for the BASELINE alone — while the peer
        arms keep searching a corpus that still contains the probing write's
        own claims, decomposed into peers.  That biases the guard column in
        the opposite direction from the one GUARD_FETCH_LIMIT's rationale
        describes, and the guard column is one of E2's four named metrics.

        Asserted over the REAL fixtures, because the bias is a property of how
        the committed decomposition assigns `source_memory_id` — a hand-built
        double could be written so the question never arises.
        """
        mod = _mod()
        clusters = mod.load_calibration_clusters()
        claims = mod.load_arm_claims()
        topics = mod.load_registry_topics()

        bit_in: dict[str, int] = {}
        for shape in mod.ARM_SHAPES:
            records = mod.materialize_arm(shape, clusters, claims, topics, [])
            seeded = mod._index_arm(shape, 'p', 'c', records, claims)
            removed = 0
            for cluster_id in sorted(clusters):
                probe = mod.select_probing_write(clusters[cluster_id])
                if probe is None:
                    continue
                own = mod.probe_own_record_ids(seeded, probe['memory_id'])
                removed += len(own & set(seeded.records_by_id))
            bit_in[shape] = removed

        # Every arm — not just the one whose ids happen to collide.
        for shape in mod.ARM_SHAPES:
            assert bit_in[shape] > 0, (
                f'{shape}: the probing write leaves nothing behind, so its own '
                f'content stayed in its own guard window'
            )

    def test_a_peer_arm_drops_MORE_than_one_record_per_probing_write(self):
        """The decomposition splits one original into several peers, so the
        filter has to remove all of them, not one representative."""
        mod = _mod()
        clusters = mod.load_calibration_clusters()
        claims = mod.load_arm_claims()
        topics = mod.load_registry_topics()

        for shape in ('c_peers', 'b_grouped'):
            records = mod.materialize_arm(shape, clusters, claims, topics, [])
            seeded = mod._index_arm(shape, 'p', 'c', records, claims)
            widest = max(
                len(mod.probe_own_record_ids(seeded, probe['memory_id']) - {probe['memory_id']})
                for probe in (
                    mod.select_probing_write(clusters[c]) for c in sorted(clusters)
                )
                if probe is not None
            )
            assert widest > 1, (
                f'{shape}: no probing write decomposed into more than one peer, '
                f'so this test can no longer tell an id filter from a '
                f'provenance filter'
            )


@pytest.mark.asyncio
class TestAFailedQueryIsNotAMeasuredZero:
    """`Mem0Client.search` swallows `TimeoutError` and returns `{}`.

    So a network failure and a shape that ranked nothing arrive at `_search`
    as the same empty list — which this module's own rule ("an empty hits list
    with a real expectation IS a measured zero") then scores as 0.0 recall,
    0.0 discoverability and 0.0 guard, indistinguishable in the artifact from
    a real result.  Every arm seeds the SHARED distractor slab, so an empty
    ranking is not a possible outcome here: it is unambiguously detectable and
    must be loud, per the no-silent-fail-soft design invariant.
    """

    class _Backend:
        def __init__(self, response):
            self._response = response

        async def search(self, **kwargs):
            return self._response

    async def _search_against(self, response):
        mod = _mod()
        seeded = _seeded('status_quo', [_rec('r0')])
        return await mod._search(
            self._Backend(response), seeded, 'anything', limit=10,
        )

    @pytest.mark.parametrize('response', [{}, None, {'results': []}, {'results': None}])
    async def test_an_empty_ranking_raises_rather_than_scoring_zero(self, response):
        with pytest.raises(_mod().MeasurementError, match='no results'):
            await self._search_against(response)

    async def test_a_scoreless_hit_raises_rather_than_defaulting_to_zero(self):
        """A 0.0 default feeds the THRESHOLD replay a value the store never
        produced, and `guard matched (replay) = 0.00` then reads as a shape
        finding rather than as missing score plumbing."""
        mod = _mod()
        record = _rec('r0')
        seeded = _seeded('status_quo', [record])
        seeded.by_stored_id['stored-0'] = record

        with pytest.raises(mod.MeasurementError, match='no score'):
            await mod._search(
                self._Backend({'results': [{'id': 'stored-0'}]}),
                seeded, 'anything', limit=10,
            )

    async def test_a_real_ranking_still_passes_through(self):
        """The guard must not fire on the healthy path."""
        mod = _mod()
        record = _rec('r0')
        seeded = _seeded('status_quo', [record])
        seeded.by_stored_id['stored-0'] = record

        hits = await mod._search(
            self._Backend({'results': [{'id': 'stored-0', 'score': 0.42}]}),
            seeded, 'anything', limit=10,
        )

        assert [(h.record.record_id, h.relevance_score) for h in hits] == [('r0', 0.42)]


class TestTheGuardColumnSaysWhatTheReplayActuallySaw:
    """Pure — the diagnostic half of the same no-silent-zero concern."""

    def test_the_guard_column_carries_the_best_score_the_replay_actually_saw(self):
        """`guard_matched_rate: 0.00` across every arm is ambiguous on its own.
        "the best candidate scored 0.71 against a 0.92 threshold" is a shape
        finding; "the best candidate scored 0.00" is a bug report."""
        seeded, hits = _full_window_arm('status_quo')

        measured = _measure(seeded, hits, pin=False, probes=[('c1', {'memory_id': 'x'})])

        assert measured['guard_adequacy']['max_observed_score'] == pytest.approx(0.9)


class TestRescoreAgreesWithTheRankTheGroupedReadGave:
    """Score and rank are two views of one ordering. They cannot disagree.

    `apply_grouped_read` gives a group the BEST rank among its members
    (`group_rank[parent] = min(...)`).  If `rescore` gave it the canonical's
    OWN score whenever the canonical was itself a hit, a group could rank 1st
    on a child's 0.95 and then replay into the threshold guard at the
    canonical's 0.30 — a guard false negative attributable to this helper
    rather than to the storage shape, which is exactly the materialization
    artifact `_claim_categories` warns about.
    """

    def test_a_group_takes_its_best_childs_score_even_when_the_canonical_hit(self):
        mod = _mod()
        canonical = _rec('canon', topic='t')
        child = _rec('kid', topic='t', parent_id='canon')
        hits = [_sh(child, 0.95), _sh(canonical, 0.30)]

        rescored = mod.rescore([canonical], hits)

        assert rescored[0].relevance_score == 0.95

    def test_a_canonical_that_outscores_its_children_keeps_its_own_score(self):
        """max(), not "children always win" — the rule is best-available."""
        mod = _mod()
        canonical = _rec('canon', topic='t')
        child = _rec('kid', topic='t', parent_id='canon')
        hits = [_sh(canonical, 0.95), _sh(child, 0.30)]

        rescored = mod.rescore([canonical], hits)

        assert rescored[0].relevance_score == 0.95

    def test_the_guard_replay_sees_the_rank_one_score_not_the_canonicals(self):
        """The consequence, at the seam that matters: a group the grouped read
        ranked first must not fail a threshold its best member cleared."""
        mod = _mod()
        canonical = _rec('canon', topic='t', claim_ids=['k0'])
        child = _rec('kid', topic='t', parent_id='canon', claim_ids=['k1'])
        hits = [_sh(child, 0.95), _sh(canonical, 0.30)]
        seeded = _seeded(
            'b_grouped', [canonical, child],
            canonical_by_cluster={'c1': 'canon'},
            siblings_by_cluster={'c1': {'canon', 'kid'}},
        )

        window = mod.read_path(seeded, hits, 5, pin=False)
        verdict = mod.guard_adequacy(
            mod.rescore(window, hits), {'canon', 'kid'}, 0.92,
        )

        assert verdict['guard_matched'] is True

    def test_a_child_outside_the_replay_window_cannot_donate_its_score(self):
        """The mirror of the test above, and the boundary between them.

        `rescore` builds `best_child` by iterating everything it is HANDED, so
        handing it the full fetched list while the window was truncated to 5
        lets hit #6 donate its score to a group inside the window.  The arm
        then books a `guard_matched` that production — whose pre-check runs at
        limit=5 — structurally could not produce.  `guard_adequacy`'s own
        defensive window cannot catch this: the leak happens upstream of it.
        """
        mod = _mod()
        canonical = _rec('canon', topic='t', claim_ids=['k0'])
        fillers = [_rec(f'f{i}', topic='other', claim_ids=[f'f{i}']) for i in range(4)]
        # Rank 6 — one past the k=5 replay window — and scoring high enough to
        # clear the 0.92 threshold if it were ever allowed to count.
        outsider = _rec('kid', topic='t', parent_id='canon', claim_ids=['k1'])
        hits = (
            [_sh(canonical, 0.30)]
            + [_sh(filler, 0.29) for filler in fillers]
            + [_sh(outsider, 0.95)]
        )
        seeded = _seeded(
            'b_grouped', [canonical, *fillers, outsider],
            canonical_by_cluster={'c1': 'canon'},
            siblings_by_cluster={'c1': {'canon', 'kid'}},
        )

        window = mod.read_path(seeded, hits, mod.GUARD_TOP_K, pin=False)
        leaky = mod.guard_adequacy(
            mod.rescore(window, hits), {'canon', 'kid'}, 0.92,
        )
        honest = mod.guard_adequacy(
            mod.rescore(window, hits[:mod.GUARD_TOP_K]), {'canon', 'kid'}, 0.92,
        )

        # The fixture is only meaningful if the leak is reachable at all.
        assert leaky['guard_matched'] is True
        assert honest['guard_matched'] is False


class TestCanonicalRankIsNotCensoredByTheReadWindow:
    """"Outside top-5" and "absent entirely" are different findings.

    `topic_discoverability` documents that its rank survives past the window.
    Scoring it against an ALREADY-truncated k=5 window makes that contract
    dead in the live path: rank can never exceed 5, so a near-miss shape and a
    shape that never surfaces the canonical both report `None`, and the median
    — taken over successes only — prints BEST for the arm that found it least.
    """

    def test_measure_arm_reports_a_rank_beyond_the_five_record_read_window(self):
        seeded, hits = _full_window_arm('status_quo', n=12)
        # Put the canonical at rank 8 — outside the k=5 read window, but well
        # inside the fetch depth the arm was actually measured over.
        canonical = seeded.records_by_id['canon']
        hits = [*hits[:7], _sh(canonical, 0.5), *hits[7:]]

        measured = _measure(seeded, hits, pin=False)
        disco = measured['discoverability']

        assert disco['canonical_in_top_5_rate'] == 0.0   # genuinely not in the read
        assert disco['median_canonical_rank'] == 8.0     # but NOT "absent entirely"
        assert disco['canonical_found_count'] == 1

    def test_the_censored_denominator_travels_with_the_median(self):
        """A median over successes only is uninterpretable without its n."""
        seeded, hits = _full_window_arm('status_quo')

        disco = _measure(seeded, hits, pin=False)['discoverability']

        assert disco['canonical_candidates'] == 1
        assert disco['canonical_found_count'] == 0       # never surfaced at all
        assert disco['median_canonical_rank'] is None

    def test_the_markdown_prints_the_denominator_next_to_the_median(self):
        """The JSON already carried `canonical_found_count`; the OPERATOR
        making the shape decision reads the markdown."""
        mod = _mod()

        cell = mod._rank_cell({
            'median_canonical_rank': 2.0,
            'canonical_found_count': 119,
            'canonical_candidates': 236,
        })

        assert cell == '2.00 (n=119/236)'

    def test_the_disclosed_rank_window_is_the_depth_THIS_run_fetched_at(self):
        """`canonical_rank_window` is what stops the uncensoring fix from just
        moving the censoring point somewhere undisclosed.

        Rank is no longer censored at the k=5 read window, but it is still
        censored at the FETCH depth — and that depth is `--limit`, a CLI flag,
        not a constant.  Reporting `DEFAULT_SEARCH_LIMIT` would mean a
        `--limit 25` run publishes a field claiming 10: the artifact would
        understate how far it actually looked, which is the same class of
        defect as the censored median it replaces.
        """
        seeded, hits = _full_window_arm('status_quo', n=12)

        measured = _measure(seeded, hits, pin=False, limit=25)

        assert measured['discoverability']['canonical_rank_window'] == 25

    def test_run_arm_threads_its_own_fetch_depth_into_the_report(
        self, monkeypatch
    ):
        """The seam that matters: the number in the artifact has to come from
        the same `limit` the fetch was actually issued at."""
        mod = _mod()
        seeded, hits = _full_window_arm('status_quo', n=12)
        issued_at: list[int] = []

        async def _fake_fetch(backend, s, queries, probes, *, limit):
            issued_at.append(limit)
            return {'queries': {'q1': hits}, 'probes': {}}

        monkeypatch.setattr(mod, 'fetch_arm', _fake_fetch)
        rows = asyncio.run(mod.run_arm(
            None, seeded, queries=[_query()], probes=[],
            limit=25, estimator=_CHARS, guard_threshold=0.92,
        ))

        assert issued_at == [25]
        for row in rows.values():
            assert row['discoverability']['canonical_rank_window'] == 25


class TestMeasureArmScoresBothVariantsAtTheSameK:
    """The inflated column, asserted directly on `measure_arm`'s output."""

    def test_a_full_window_arm_measures_identically_with_and_without_the_pin(self):
        """The committed artifact's c_peers/c_peers+pin divergence, in one
        assertion: at a fixed result budget an append-only pin has nowhere to
        put anything, so every column must be unchanged."""
        seeded, hits = _full_window_arm('c_peers')

        off = _measure(seeded, hits, pin=False)
        on = _measure(seeded, hits, pin=True)

        assert on['discoverability']['canonical_in_top_5_rate'] == (
            off['discoverability']['canonical_in_top_5_rate']
        )
        assert on['claim_recall']['at_5'] == off['claim_recall']['at_5']
        assert on['tokens_per_query']['mean'] == off['tokens_per_query']['mean']

    def test_every_block_is_unchanged_not_merely_the_three_headline_numbers(self):
        seeded, hits = _full_window_arm('c_peers')

        off = _measure(seeded, hits, pin=False)
        on = _measure(seeded, hits, pin=True)

        for metric in ('claim_recall', 'discoverability', 'tokens_per_query',
                       'guard_adequacy'):
            assert on[metric] == off[metric], metric

    def test_a_claim_only_the_pinned_record_realizes_is_not_credited_at_5(self):
        """The mechanism, isolated: `k-canon` lives on the appended canonical
        alone, so crediting it at 5 is exactly the off-by-one-window bug."""
        seeded, hits = _full_window_arm('c_peers')

        on = _measure(seeded, hits, pin=True, queries=[_query(expects=('k-canon',))])

        assert on['claim_recall']['at_5'] == 0.0

    def test_the_token_column_charges_for_ten_payloads_not_eleven(self):
        """tokens/query is the cost half of the decision.  An eleventh payload
        billed to the pin is a cost the reader would never have paid."""
        seeded, hits = _full_window_arm('c_peers')

        on = _measure(seeded, hits, pin=True)

        # `_CHARS` counts characters, and every hand-built body is `'body'`.
        assert on['tokens_per_query']['mean'] == 10 * len('body')


class TestPinDiagnostic:
    """`pin` says whether the pin was on, and whether it did anything."""

    def test_pin_off_reports_not_applicable_rather_than_a_measured_zero(self):
        seeded, hits = _full_window_arm('c_peers')

        off = _measure(seeded, hits, pin=False)

        assert off['pin'] == {'enabled': False, 'window_changed_rate': None}

    def test_a_full_window_arm_reports_a_zero_change_rate(self):
        """Enabled, asked, and it changed nothing — a measured zero, which is
        a different statement from `None`."""
        seeded, hits = _full_window_arm('c_peers')

        on = _measure(seeded, hits, pin=True)

        assert on['pin']['enabled'] is True
        assert on['pin']['window_changed_rate'] == 0.0

    def test_a_grouped_arm_with_headroom_reports_a_positive_change_rate(self):
        mod = _mod()
        parent = _rec(PARENT, topic='t1', canonical=True, claim_ids=['k-par'])
        children = [
            _rec(f'child-{i}', parent_id=PARENT, kind='amendment', topic='t1',
                 claim_ids=[f'k{i}'])
            for i in range(4)
        ]
        other = _rec('other-1', topic='t2', claim_ids=['k-other'])
        anchor = _rec('canon-t2', topic='t2', canonical=True, claim_ids=['k-t2'])
        hits = [_sh(record, 0.9 - i / 100.0)
                for i, record in enumerate([*children, other])]
        seeded = _seeded(
            'b_grouped',
            [parent, *children, other, anchor],
            canonical_by_topic={'t1': parent, 't2': anchor},
            canonical_by_cluster={'c1': PARENT},
            siblings_by_cluster={'c1': {c.record_id for c in children}},
        )

        on = mod.measure_arm(
            seeded,
            {'queries': {'q1': hits}, 'probes': {}},
            pin=True, queries=[_query(topic='t2')], probes=[],
            estimator=_CHARS, guard_threshold=0.92, limit=10,
        )

        assert on['pin']['window_changed_rate'] > 0


class TestReportCarriesThePinDiagnostic:
    """A column that explains the other columns has to be in the artifact."""

    def test_build_report_refuses_an_arm_with_no_pin_block(self):
        mod = _mod()
        arms = _all_arms()
        del arms['c_peers+pin']['pin']

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'c_peers+pin' in str(excinfo.value)
        assert 'pin' in str(excinfo.value)

    @pytest.mark.parametrize('key', ['enabled', 'window_changed_rate'])
    def test_build_report_refuses_a_pin_block_missing_either_key(self, key):
        mod = _mod()
        arms = _all_arms()
        del arms['b_grouped+pin']['pin'][key]

        with pytest.raises(mod.IncompleteReportError) as excinfo:
            mod.build_report(
                arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
            )

        assert 'b_grouped+pin' in str(excinfo.value)
        assert key in str(excinfo.value)

    def test_the_decision_table_surfaces_the_diagnostic(self):
        """Without it, two identical rows read as "the pin is useless" rather
        than "the pin never fired"."""
        mod = _mod()
        arms = _all_arms()
        arms['c_peers+pin']['pin']['window_changed_rate'] = 0.375

        rendered = mod.render_markdown(mod.build_report(
            arms=arms, audit_recall=_audit_recall(), protocol=_protocol(),
        ))

        row = next(
            r for r in _decision_table_rows(rendered) if r.startswith('| c_peers+pin |')
        )
        assert '0.38' in row

    def test_a_pin_off_row_renders_no_measurement_not_a_zero(self):
        mod = _mod()

        rendered = mod.render_markdown(_report())

        row = next(
            r for r in _decision_table_rows(rendered) if r.startswith('| c_peers |')
        )
        assert '—' in row


class TestBuildParser:
    """The CLI surface, pinned by equality."""

    def test_the_defaults_are_the_committed_fixtures_and_artifact_paths(self):
        mod = _mod()

        args = mod.build_parser().parse_args([])

        assert Path(args.arm_claims) == mod.DEFAULT_ARM_CLAIMS_PATH
        assert Path(args.query_set) == mod.DEFAULT_QUERY_SET_PATH
        assert Path(args.distractor_slab) == mod.DEFAULT_DISTRACTOR_SLAB_PATH
        assert Path(args.json_out) == mod.DEFAULT_REPORT_JSON
        assert Path(args.md_out) == mod.DEFAULT_REPORT_MD

    def test_the_default_run_is_the_whole_fixture_not_a_sample(self):
        """A silently-sampled default would publish a decision table over a
        subset while reading as a full run."""
        args = _mod().build_parser().parse_args([])

        assert args.clusters is None
        assert args.distractors is None
        assert args.limit == _mod().DEFAULT_SEARCH_LIMIT

    def test_the_default_run_probes_regrowth_and_skipping_it_is_explicit(self):
        """Asserted by ATTRIBUTE, like every other default here.

        The default is ON because the artifact this script writes is gate
        leaf eta's input and esc-3200-3 asked for these deltas by name: a
        probe that had to be opted INTO would go missing again exactly the
        way it went missing the first time, and the artifact would carry no
        trace of the omission.
        """
        parser = _mod().build_parser()

        assert parser.parse_args([]).regrowth is True
        assert parser.parse_args(['--no-regrowth']).regrowth is False

    def test_the_probe_switch_does_not_disturb_the_cache_flags(self):
        """`--dump-fetches`/`--replay-fetches` and their exit codes are
        4004's contract; this task widens what they carry, not what they
        default to."""
        args = _mod().build_parser().parse_args(['--no-regrowth'])

        assert args.dump_fetches is None
        assert args.replay_fetches is None

    def test_the_mutual_exclusion_guard_still_fires_with_the_probe_off(
        self, tmp_path, capsys,
    ):
        """Exit 2, named on stderr, unchanged: a run that both dumps and
        replays has provenance the artifact cannot describe, and turning the
        probe off does not make that ambiguity readable."""
        mod = _mod()

        code = mod.main([
            '--no-regrowth',
            '--dump-fetches', str(tmp_path / 'a.json'),
            '--replay-fetches', str(tmp_path / 'b.json'),
        ])

        assert code == 2
        assert '--dump-fetches' in capsys.readouterr().err


class TestMain:
    """`main(argv)` driven directly — no subprocess."""

    def test_a_successful_run_writes_both_artifacts_and_returns_zero(
        self, monkeypatch, tmp_path,
    ):
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        json_out, md_out = tmp_path / 'report.json', tmp_path / 'report.md'

        code = mod.main([
            '--clusters', '2', '--distractors', '12', '--project-suffix', 'utest',
            '--json-out', str(json_out), '--md-out', str(md_out),
        ])

        assert code == 0
        assert list(json.loads(json_out.read_text())['arms']) == list(mod.ARM_VARIANTS)

    def test_a_missing_fixture_exits_2_before_a_single_collection_exists(
        self, monkeypatch, tmp_path,
    ):
        """Ordering matters as much as the code: a fixture checked AFTER
        seeding would leave three live collections behind on every typo."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        json_out, md_out = tmp_path / 'report.json', tmp_path / 'report.md'

        code = mod.main([
            '--arm-claims', str(tmp_path / 'does-not-exist.jsonl'),
            '--project-suffix', 'utest',
            '--json-out', str(json_out), '--md-out', str(md_out),
        ])

        assert code == 2
        assert drops.calls == []
        assert _FakeMemoryService.instances == []
        assert not json_out.exists()
        assert not md_out.exists()

    def test_an_inconsistent_fixture_set_exits_2_with_no_artifact(
        self, monkeypatch, tmp_path,
    ):
        """Cross-validation is part of the pre-flight, not a later surprise:
        a claim pointing at a missing cluster surfaces deep in the metrics as
        a silently-zero recall indistinguishable from a retrieval miss."""
        mod = _mod()
        drops = _install_driver_doubles(monkeypatch)
        bad_claims = tmp_path / 'claims.jsonl'
        bad_claims.write_text(json.dumps({
            'claim_id': 'ghost-01', 'cluster_id': 'no-such-cluster',
            'topic': 'ghost', 'text': 'a claim about a cluster that is not there',
            'source_memory_id': 'nobody', 'canonical': True,
            'b_arm_role': 'canonical', 'contested': False,
        }) + '\n', encoding='utf-8')
        json_out = tmp_path / 'report.json'

        code = mod.main([
            '--arm-claims', str(bad_claims), '--project-suffix', 'utest',
            '--json-out', str(json_out), '--md-out', str(tmp_path / 'report.md'),
        ])

        assert code == 2
        assert drops.calls == []
        assert not json_out.exists()

    def test_it_reports_the_fixture_failure_on_stderr_not_silently(
        self, monkeypatch, tmp_path, capsys,
    ):
        mod = _mod()
        _install_driver_doubles(monkeypatch)

        code = mod.main([
            '--query-set', str(tmp_path / 'absent.jsonl'),
            '--project-suffix', 'utest',
            '--json-out', str(tmp_path / 'r.json'), '--md-out', str(tmp_path / 'r.md'),
        ])

        assert 'absent.jsonl' in capsys.readouterr().err
        # Reporting the failure on stderr but still returning 0 is the worse
        # half of "silently": the run looks successful and a stale or absent
        # artifact gets published as if it were a fresh measurement.
        assert code == 2

    @pytest.mark.parametrize(
        ('argv', 'expected'), [([], True), (['--no-regrowth'], False)],
    )
    def test_the_switch_reaches_run_bake_off_rather_than_stopping_at_the_parser(
        self, monkeypatch, tmp_path, argv, expected,
    ):
        """A flag the driver never receives is a flag that silently does
        nothing — and the artifact would then say `regrowth_probed: true`
        for a run the operator asked to skip."""
        mod = _mod()
        _install_driver_doubles(monkeypatch)
        seen: list = []
        real = mod.run_bake_off

        async def _spy(**kwargs):
            seen.append(kwargs.get('regrowth'))
            return await real(**kwargs)

        monkeypatch.setattr(mod, 'run_bake_off', _spy)

        code = mod.main([
            *argv, '--clusters', '2', '--distractors', '12',
            '--project-suffix', 'utest',
            '--json-out', str(tmp_path / 'r.json'),
            '--md-out', str(tmp_path / 'r.md'),
        ])

        assert code == 0
        assert seen == [expected]


# ===========================================================================
# step-20 — the ONE live end-to-end test
# ===========================================================================
#
# Everything above measures this script's arithmetic against injected hit
# lists and doubled backends.  This measures it against a real Qdrant, a real
# embedder and a real MemoryService, and asserts the one thing a double can
# never prove: that a full seed/query/measure/teardown cycle produces a
# COMPLETE report and leaves NOTHING behind.
#
# It pins no metric value (G6).  A live embedding ranking is exactly the kind
# of number eval-design §1 says moves wholesale with wording and config drift
# — asserting one here would make this test fail for a reason that is not a
# defect.
#
# Marked PER-TEST, never via a module `pytestmark`: fused-memory's
# `addopts = -m 'not integration'` would otherwise deselect the ~200 pure
# tests above from the merge lane along with this one.

import os  # noqa: E402

from _fm_helpers import QDRANT_URL, qdrant_skipif  # noqa: E402


@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.asyncio
@qdrant_skipif()
@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='the seeded bake-off needs a real embedder',
)
async def test_a_live_two_cluster_run_reports_completely_and_leaves_nothing(
    worker_id,
):
    """A 2-cluster / 12-distractor subset: enough to exercise every seam
    (three arm collections, six variants, both read transforms, the guard
    replay, the D10 block, the two INJECTED regrowth passes and both
    artifacts) at a fraction of a full run's wall clock.

    The timeout covers five seed+fetch passes rather than three: the probe
    adds one live seeding and one live fetch per injection mode.
    """
    from qdrant_client import QdrantClient  # noqa: PLC0415

    mod = _mod()
    suffix = f'live_{worker_id}'
    collections = set(mod.ephemeral_collections(suffix=suffix).values())
    # The probe's two collections are part of the teardown contract, not a
    # separate one: a leaked injected pass is exactly as unreapable as a
    # leaked arm, and is the failure the widened disjointness below catches.
    collections |= _regrowth_collection_names(mod, suffix=suffix)

    report = await mod.run_bake_off(
        cluster_limit=2, distractor_limit=12, project_suffix=suffix,
    )

    assert list(report['arms']) == list(mod.ARM_VARIANTS)
    for arm in mod.ARM_VARIANTS:
        for metric, required in mod._REQUIRED_ARM_METRICS.items():
            for key in required:
                assert key in report['arms'][arm][metric]
    assert report['audit_recall']['true_dup']['pairs'] > 0
    assert report['protocol']['distractor_slab_size'] == 12

    # COMPLETE, not merely present.  A live run that produced an empty or
    # half-filled block would still render a `## Regrowth deltas` section, and
    # its blank cells read as "the injection changed nothing".
    regrowth = report['regrowth']
    assert regrowth is not None
    assert report['protocol']['regrowth_probed'] is True
    assert report['protocol']['regrowth_injections_measured'] == 2
    for mode in mod.REGROWTH_MODES:
        for arm in mod.REGROWTH_READ_ARMS:
            for key in mod._regrowth_metric_keys():
                assert key in regrowth['after'][mode][arm]
                assert key in regrowth['deltas'][mode][arm]
                assert key in regrowth['stamping_value'][arm]
                assert key in regrowth['baseline'][arm]

    # The teardown is the half that leaks silently: a report that looks right
    # while five collections survive is the failure this asserts against.
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        live = {col.name for col in client.get_collections().collections}
    finally:
        client.close()
    assert live.isdisjoint(collections)


# ===========================================================================
# step-23 — the committed artifact, asserted as DATA
# ===========================================================================
#
# `plans/e2-storage-shape-bakeoff-report.{json,md}` is this task's
# user-observable signal and gate leaf η's input, so it is guarded as data
# rather than described in prose: a decision table nobody parses is a
# decision table that can rot in place.
#
# Pins NO metric value, rate or bound (G6).  What it pins is COMPLETENESS —
# every arm present, every metric measured rather than `—`, both artifacts
# agreeing — because that is the difference between "the run said C wins"
# and "the run half-failed and the blank cells read as a tie".
#
# Pure file reads.  No network, no Qdrant, no key: this runs in the merge
# lane on every commit, which is the point.


@functools.cache
def _committed_report() -> dict:
    path = _mod().DEFAULT_REPORT_JSON
    assert path.exists(), (
        f'{path} is missing. It is the artifact gate eta reads; regenerate '
        f'it with `uv run python fused-memory/scripts/bake_off_storage_shape.py`.'
    )
    return json.loads(path.read_text(encoding='utf-8'))


class TestCommittedReportJson:
    """The machine-readable half."""

    def test_it_parses_and_declares_its_schema_version(self):
        report = _committed_report()

        assert report['schema_version'] == _mod().REPORT_SCHEMA_VERSION

    def test_every_arm_variant_has_a_row(self):
        report = _committed_report()

        assert list(report['arms']) == list(_mod().ARM_VARIANTS)

    @pytest.mark.parametrize('metric,keys', [
        ('claim_recall', ('at_5', 'at_10')),
        # Both forms of the same column: the one measured over the read
        # window, and the transform-blind one measured over the raw store
        # hits.  A committed run missing the second leaves `b_grouped`'s
        # headline rate readable only as though grouping were retrieval.
        ('discoverability', ('canonical_in_top_5_rate',
                             'stored_canonical_in_top_5_rate',
                             'stored_canonical_found_count')),
        ('tokens_per_query', ('mean',)),
        # Both halves: the rank-based one and the flagged threshold replay.
        ('guard_adequacy', ('candidate_present_rate', 'guard_matched_rate')),
    ])
    def test_every_arm_measured_every_metric(self, metric, keys):
        """`None` is a legitimate value in the pipeline — "measured, no
        denominator" — but in a FULL committed run it means the arm was never
        asked, and the markdown renders it as `—` next to real numbers."""
        report = _committed_report()

        for arm, measurement in report['arms'].items():
            for key in keys:
                assert measurement[metric][key] is not None, f'{arm}.{metric}.{key}'

    def test_every_arm_measured_every_query_kind(self):
        """The by-kind table is the block above split the way eval-design §5
        E2 says the metrics differ, so an unasked subset there is the same
        defect as an unasked metric — and in the committed run all three
        subsets are non-empty (176 claim / 60 topic_phrasing / 20 held_out),
        so `queries: 0` means the split broke, not that a subset was empty."""
        report = _committed_report()

        for arm, measurement in report['arms'].items():
            by_kind = measurement['by_query_kind']
            for kind in ('claim', 'topic_phrasing', 'held_out'):
                assert by_kind[kind]['queries'] > 0, f'{arm}.{kind}'
                for key in ('at_5', 'at_10'):
                    assert by_kind[kind]['claim_recall'][key] is not None, (
                        f'{arm}.{kind}.claim_recall.{key}'
                    )
                # The by-kind table renders the transform-blind column too,
                # and `held_out` — the only subset that measures
                # generalisation rather than recall of the derivation input
                # — is the row where a transform-credited rate is least safe
                # to read alone.
                assert by_kind[kind]['discoverability'][
                    'stored_canonical_in_top_5_rate'] is not None, (
                    f'{arm}.{kind}.stored_canonical_in_top_5_rate'
                )
            # held_out is a SUBSET of topic_phrasing, never a third kind, and
            # a strict one — the whole point is that some phrasings WERE the
            # registry's derivation input and so cannot measure generalisation.
            assert by_kind['held_out']['queries'] < by_kind['topic_phrasing']['queries']

    def test_the_transform_blind_column_is_identical_across_a_shapes_pin_twins(self):
        """Measured over the raw store hits, so it CANNOT depend on the pin.

        The artifact-level statement of the unit-level property, and the one
        that makes the column readable as retrieval rather than as any
        read-side transform: a difference here would mean the column is
        measuring something downstream of `read_path` after all.
        """
        report = _committed_report()

        for shape in _mod().ARM_SHAPES:
            off = report['arms'][shape]['discoverability']
            on = report['arms'][f'{shape}+pin']['discoverability']
            for key in ('stored_canonical_in_top_5_rate',
                        'stored_canonical_median_rank',
                        'stored_canonical_found_count'):
                assert on[key] == off[key], f'{shape}.{key}'

    def test_at_least_one_arm_reads_differently_before_and_after_the_transforms(self):
        """Anti-vacuity, in the spirit of the `assert checked` guard below.

        Pins NO rate, bound, direction or magnitude (G6) — only that a
        difference EXISTS somewhere in the table.  Two columns that always
        agreed would mean the disclosure column had been aliased to the one
        it is supposed to qualify, and a future regeneration that quietly did
        that would otherwise publish a duplicated number and pass.
        """
        report = _committed_report()

        assert any(
            measurement['discoverability']['stored_canonical_in_top_5_rate']
            != measurement['discoverability']['canonical_in_top_5_rate']
            for measurement in report['arms'].values()
        ), 'no arm distinguishes the stored rate from the transform-credited one'

    def test_every_arm_carries_the_pin_diagnostic(self):
        """Both keys on all six rows. A `+pin` row identical to its twin is
        unreadable without it: "the pin never fired" and "the pin does not
        help" are different findings, and the artifact has to say which."""
        report = _committed_report()

        for arm, measurement in report['arms'].items():
            assert 'pin' in measurement, arm
            assert measurement['pin']['enabled'] is arm.endswith('+pin'), arm
            assert 'window_changed_rate' in measurement['pin'], arm

    def test_a_pin_off_row_reports_no_rate_rather_than_a_measured_zero(self):
        """The question was never asked there. A 0.0 would claim it was."""
        report = _committed_report()

        for arm, measurement in report['arms'].items():
            if not arm.endswith('+pin'):
                assert measurement['pin']['window_changed_rate'] is None, arm

    @pytest.mark.parametrize('shape', ['status_quo', 'c_peers', 'b_grouped'])
    def test_a_pin_column_differs_from_its_twin_only_where_the_pin_fired(self, shape):
        """The reviewer's finding, stated as an invariant over the artifact.

        A `+pin` variant is measured over a window of the SAME size as its
        pin-off twin, so it can only score differently when the pin actually
        CHANGED that window. If it changed nothing, every metric block must be
        byte-identical to the twin's; and wherever any block differs, the
        diagnostic must show the pin firing. Anything else is a column bought
        with a bigger budget rather than earned.

        Pins no value, rate or bound (G6/D10) — only the two-way agreement
        between the diagnostic and the metrics it explains.

        Stated as ONE unconditional implication rather than as a branch per
        case, deliberately: `rate > 0` beside identical blocks is a state the
        artifact's own reading guide calls a legitimate third finding ("it
        fired and moved nothing these metrics measure"), so a guarded form
        executes zero assertions on exactly that input and passes vacuously.
        Here every parametrized shape runs both assertions whatever the data
        does, so the test cannot silently degrade to a no-op.
        """
        arms = _committed_report()['arms']
        off, on = arms[shape], arms[f'{shape}+pin']
        blocks = ('claim_recall', 'discoverability', 'tokens_per_query',
                  'guard_adequacy')
        differing = [block for block in blocks if on[block] != off[block]]
        rate = on['pin']['window_changed_rate']

        # A `+pin` arm always measured the pin, so the diagnostic that makes
        # its row readable is never absent.
        assert rate is not None, f'{shape}+pin carries no pin diagnostic'
        # The invariant itself: a column may differ ONLY where the pin fired.
        # (The converse is NOT asserted — a pin that fired and moved nothing
        # measurable is a real finding, not a failure.)
        assert not (differing and rate == 0.0), (
            f'{shape}+pin differs from its twin in {differing} while the pin '
            f'changed no window — the difference is an unequal measurement, '
            f'not a result'
        )

    def test_the_guard_column_is_flagged_as_a_threshold_replay_on_every_arm(self):
        """Carried per-arm so a reader who copies one row out of the table
        cannot lose the flag (eval-design §1's one sanctioned exception)."""
        report = _committed_report()

        for measurement in report['arms'].values():
            assert measurement['guard_adequacy']['threshold_replay'] is True
            assert measurement['guard_adequacy']['threshold'] is not None

    def test_the_d10_block_carries_its_recall_and_its_band_split(self):
        """D10 is a second, independent deliverable, not a footnote of E2."""
        audit = _committed_report()['audit_recall']
        true_dup = audit['true_dup']

        assert true_dup['recall'] is not None
        assert true_dup['pairs'] == (
            true_dup['lexical_band']['pairs'] + true_dup['paraphrase_band']['pairs']
        )
        assert audit['hard_negative']['pairs'] > 0
        assert audit['unrelated']['pairs'] > 0

    def test_the_protocol_block_says_how_the_numbers_were_produced(self):
        """An arbitration artifact whose provenance is not in it cannot be
        re-read in six months by somebody who was not in the room."""
        mod = _mod()
        protocol = _committed_report()['protocol']

        for key in mod._REQUIRED_PROTOCOL_KEYS:
            assert protocol[key] is not None
        assert protocol['token_estimator'] in (
            mod.TIKTOKEN_ESTIMATOR_NAME, mod.CHAR_PROXY_ESTIMATOR_NAME,
        )
        assert protocol['distractor_slab_size'] > 0

    def test_it_records_the_fixture_commits_the_blind_protocol_rests_on(self):
        """The claim "no metric code existed when the arms were authored" is
        only checkable if the artifact says which commits to go and look at."""
        fixtures = _committed_report()['protocol']['fixtures']

        paths = {entry['path'] for entry in fixtures}
        assert any(path.endswith('e2_arm_claims.jsonl') for path in paths)
        assert any(path.endswith('e2_query_set.jsonl') for path in paths)
        for entry in fixtures:
            assert not entry['path'].startswith('/')  # repo-relative, reproducible
            assert entry['commit'], f"{entry['path']} is not committed"

    def test_the_run_measured_the_whole_fixture_not_a_smoke_subset(self):
        """A `--clusters 2` artifact would read exactly like a full one."""
        protocol = _committed_report()['protocol']

        assert protocol['clusters_measured'] == len(_mod().load_calibration_clusters())
        assert protocol['queries_measured'] == len(_mod().load_query_set())
        assert protocol['distractor_slab_size'] == len(_mod().load_distractor_slab())

    # --- the +1-re-emission probe (task 4012) -----------------------------
    #
    # esc-3200-3 asked for regrowth deltas and read an artifact that had no
    # such section.  Nothing in the tree could have caught that: 3199's
    # delivered-check asserted only that the report EXISTS.  These are the
    # gate it did not have — a committed run whose probe was skipped now
    # fails in the merge lane instead of being published as complete.

    def test_the_committed_run_actually_probed_regrowth(self):
        """The absence esc-3200-3 could not read, made a test failure.

        `--no-regrowth` is a legitimate flag for a smoke run, and the
        renderer says so in words.  What it must not do is reach the
        COMMITTED pair, because a skipped probe and a probe that ran and
        found nothing read identically to the operator holding the artifact.
        """
        report = _committed_report()

        assert report['regrowth'] is not None, (
            'the committed report carries no regrowth block — the +1-'
            're-emission probe was skipped. Regenerate with `cd fused-memory '
            '&& uv run python scripts/bake_off_storage_shape.py` (probe on by '
            'default; `--no-regrowth` must not reach the committed pair).'
        )
        assert report['protocol']['regrowth_probed'] is True

    def test_the_delta_tables_cover_every_mode_and_every_read_arm(self):
        """Pinned by EQUALITY, like `test_every_arm_variant_has_a_row` above.

        A mode or a read arm quietly missing from the committed tables is a
        row quietly missing from the decision the probe informs, and the
        rendered table would simply be shorter — which nothing about the
        artifact makes visible.
        """
        mod = _mod()
        regrowth = _committed_report()['regrowth']

        assert list(regrowth['deltas']) == list(mod.REGROWTH_MODES)
        assert list(regrowth['after']) == list(mod.REGROWTH_MODES)
        assert list(regrowth['baseline']) == list(mod.REGROWTH_READ_ARMS)
        for mode in mod.REGROWTH_MODES:
            assert list(regrowth['deltas'][mode]) == list(mod.REGROWTH_READ_ARMS), mode
            assert list(regrowth['after'][mode]) == list(mod.REGROWTH_READ_ARMS), mode

    def test_every_regrowth_table_measured_every_metric(self):
        """The same completeness bar the arms are held to, for the same reason.

        `None` is a legitimate value in the pipeline — "measured, no
        denominator" — but in a FULL committed run it means the arm was never
        asked, and `_regrowth_cell` renders it as `—` beside real numbers.

        Keyed through `_regrowth_metric_keys()` rather than by re-spelling the
        `<block>.<key>` join here: that helper is the ONE spelling the
        projection, the completeness check and the renderer all share, so a
        test that spelled it independently could pass over a table the
        renderer cannot read.
        """
        mod = _mod()
        regrowth = _committed_report()['regrowth']
        metrics = mod._regrowth_metric_keys()

        tables = [('baseline', regrowth['baseline'])]
        for mode in mod.REGROWTH_MODES:
            tables.append((f'after.{mode}', regrowth['after'][mode]))
            tables.append((f'deltas.{mode}', regrowth['deltas'][mode]))
        tables.append(('stamping_value', regrowth['stamping_value']))

        for where, table in tables:
            for arm in mod.REGROWTH_READ_ARMS:
                assert set(table[arm]) == set(metrics), f'{where}.{arm}'
                for metric in metrics:
                    assert table[arm][metric] is not None, f'{where}.{arm}.{metric}'

    def test_the_plus_one_in_the_probes_name_is_a_property_of_the_artifact(self):
        """The `+1` is checkable here, not a claim in the artifact's prose.

        Joined to `clusters_measured` rather than to a literal 20, so a
        `--clusters 2` smoke artifact cannot satisfy it by carrying two
        injections either: the probe covered every topic the run measured, or
        it did not.
        """
        report = _committed_report()

        assert report['regrowth']['topics_injected'] == (
            report['protocol']['clusters_measured']
        )
        assert report['regrowth']['injections_per_topic'] == 1
        assert report['protocol']['regrowth_injections_measured'] == (
            report['regrowth']['topics_injected']
        )

    def test_it_records_the_injection_fixtures_commit(self):
        """The disclosure's own checkable claim.

        `REGROWTH_BLIND_AUTHORING_DISCLOSURE` says the injection fixture was
        committed on its own ahead of every line of probe code and that "its
        commit is in the fixture table below".  A disclosure pointing at a row
        that is not there is worse than no disclosure: it reads as a trail
        that can be followed.
        """
        mod = _mod()
        report = _committed_report()

        wanted = mod._repo_relative(mod.DEFAULT_REGROWTH_INJECTION_PATH)
        # The block and the fixture table name the same file, through the one
        # `_repo_relative` spelling, so the audit trail cannot point at a path
        # the probe did not read.
        assert report['regrowth']['injection_fixture'] == wanted
        rows = [
            entry for entry in report['protocol']['fixtures']
            if entry['path'] == wanted
        ]
        assert len(rows) == 1, f'{wanted}: {len(rows)} rows in the fixture table'
        assert rows[0]['commit'], f'{wanted} is not committed'


class TestCommittedReportMarkdown:
    """The operator-facing half, and its agreement with the JSON."""

    def test_the_decision_table_has_exactly_one_row_per_arm_in_the_json(self):
        """The two artifacts are written from one `build_report` result; this
        is what makes it impossible for them to disagree unnoticed."""
        rendered = _mod().DEFAULT_REPORT_MD.read_text(encoding='utf-8')

        rows = _decision_table_rows(rendered)
        arms = list(_committed_report()['arms'])
        assert len(rows) == len(arms)
        for arm in arms:
            assert sum(1 for row in rows if row.startswith(f'| {arm} |')) == 1

    def test_the_table_header_is_the_pinned_column_set(self):
        mod = _mod()

        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        header = next(
            line for line in rendered.splitlines() if line.startswith('| arm ')
        )

        assert header == '| ' + ' | '.join(mod.DECISION_TABLE_COLUMNS) + ' |'

    def test_no_measurement_cell_in_the_committed_table_is_missing(self):
        """`—` means "never measured", and next to real numbers it reads as a
        tie.  A committed table must not contain one.

        Scoped to the MEASUREMENT columns.  `pin changed window` is a
        diagnostic, not a measurement of the arm, and on a pin-off row the
        question genuinely was never asked — see the test below, which pins
        that column's `—` exactly where it belongs rather than allowing it
        anywhere.
        """
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        pin_column = mod.DECISION_TABLE_COLUMNS.index('pin changed window')

        for row in _decision_table_rows(rendered):
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            assert '—' not in cells[:pin_column], row

    def test_the_pin_column_is_not_applicable_exactly_on_the_pin_off_rows(self):
        """The stronger form of the rule above: `—` in that column is correct
        on a pin-off row and a broken run anywhere else."""
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        pin_column = mod.DECISION_TABLE_COLUMNS.index('pin changed window')

        for row in _decision_table_rows(rendered):
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            arm, pin_cell = cells[0], cells[pin_column]
            assert (pin_cell == '—') is not arm.endswith('+pin'), row

    def test_a_pin_that_fired_never_renders_as_the_never_fired_value(self):
        """`0.00` in this column is a CLAIM, and the artifact makes it.

        The reading guide reads `0.00` as "the pin never fired" — one of the
        two findings this diagnostic exists to separate.  A rate of 0.0041 (2
        of 487 windows) rounded to `0.00` at 2 decimals therefore makes the
        deliverable state the opposite of what was measured, and steers the
        gate-η reader to the wrong finding.  Asserted against the COMMITTED
        pair, because that is the artifact the operator reads.
        """
        mod = _mod()
        report = _committed_report()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        pin_column = mod.DECISION_TABLE_COLUMNS.index('pin changed window')

        by_arm = {}
        for row in _decision_table_rows(rendered):
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            by_arm[cells[0]] = cells[pin_column]

        checked = 0
        for arm, measurement in report['arms'].items():
            rate = measurement['pin']['window_changed_rate']
            if rate is None or rate == 0:
                continue
            checked += 1
            assert by_arm[arm] != '0.00', (
                f'{arm}: window_changed_rate={rate} rendered as `0.00`, which '
                f"this artifact's own reading guide defines as \"the pin never "
                f'fired"'
            )
        assert checked, 'no arm fired the pin, so this test asserted nothing'

    def test_the_prose_bullet_restates_the_same_rate_as_the_table_cell(self):
        """The bullet and the table cell are INDEPENDENT lookups.

        ``render_markdown`` reads ``arms[f'{shape}+pin']`` for the bullet
        (bake_off_storage_shape.py:2039) but formats the row from whichever
        arm the table loop is on (:2013).  Cross-wiring the bullet to the
        pin-OFF arm would make the artifact's prose contradict the column it
        summarizes, and byte-identity with the committed JSON could not
        notice: both sides render through the same function, so the wrong
        value would be written to the committed markdown too.

        Compared as extracted VALUES rather than as a sentence, and the bullet
        is LOCATED by `pin_bullet_prefix(shape)` — the anchor the renderer
        itself emits — rather than by English wording.  Selecting on "starts
        with `- `, names the shape, and contains the word 'pin'" would fail
        on a reworded bullet with a count mismatch, i.e. break on the one
        thing this test is supposed to leave free.  Only the number has to
        agree.
        """
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        pin_column = mod.DECISION_TABLE_COLUMNS.index('pin changed window')

        by_arm = {}
        for row in _decision_table_rows(rendered):
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            by_arm[cells[0]] = cells[pin_column]

        for shape in mod.ARM_SHAPES:
            bullets = [
                line for line in rendered.splitlines()
                if line.startswith(mod.pin_bullet_prefix(shape))
            ]
            # Exactly one, so a reworded bullet fails loudly here rather than
            # silently reducing the check below to a no-op.
            assert len(bullets) == 1, f'{shape}: {len(bullets)} pin bullets'

            cell = by_arm[f'{shape}+pin']
            assert cell != mod._NO_MEASUREMENT, (
                f'{shape}+pin rendered as unmeasured, so this assertion '
                f'cannot tell the two arms apart'
            )
            assert cell in bullets[0].split(), (
                f'{shape}: table says {cell!r} for {shape}+pin, but the '
                f'prose bullet does not restate it: {bullets[0]!r}'
            )

    def test_the_stored_gap_bullets_agree_with_the_committed_table(self):
        """The run-specific paragraph, checked against the table it sits above.

        Extracted from the MARKDOWN on both sides — the bullet's own text
        versus the table's own cells — so this fails on exactly the defect the
        derivation was introduced to prevent: an artifact whose prose states a
        relationship the table three lines below contradicts.  A regeneration
        that reverted the paragraph to a typed sentence would keep passing
        `test_it_renders_byte_identically_from_the_committed_json` (both sides
        render through the same function) but not this.
        """
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')
        stored_column = mod.DECISION_TABLE_COLUMNS.index(
            'canonical in top-5 (stored)')
        credited_column = mod.DECISION_TABLE_COLUMNS.index('canonical in top-5')

        cells = {}
        for row in _decision_table_rows(rendered):
            parsed = _cells(row)
            cells[parsed[0]] = (parsed[stored_column], parsed[credited_column])

        assert cells
        for arm, (stored, credited) in cells.items():
            prefix = mod.stored_gap_bullet_prefix(arm)
            bullets = [
                line for line in rendered.splitlines()
                if line.startswith(prefix)
            ]
            assert len(bullets) == 1, f'{arm}: {len(bullets)} bullets'
            assert f'{stored} vs {credited}' in bullets[0], (
                f'{arm}: table says stored={stored} credited={credited}, the '
                f'prose says: {bullets[0]!r}'
            )
            assert ('identical' in bullets[0]) is (stored == credited), (
                f'{arm}: the bullet and the table disagree on whether the two '
                f'columns match: {bullets[0]!r}'
            )

    # --- the +1-re-emission probe (task 4012) -----------------------------
    #
    # The JSON half above pins that the probe RAN and measured everything.
    # This half pins that the operator can read it: the section is where the
    # reading order puts it, its tables carry no `—`, and the disclosure that
    # qualifies every number in it is present verbatim.
    #
    # `TestTheCommittedPairAgrees` below does NOT subsume these.  Both sides
    # of that comparison go through `render_markdown`, so a renderer that
    # dropped the section, or a JSON half that never carried it, agrees with
    # itself perfectly.

    def test_the_regrowth_section_sits_between_by_query_kind_and_d10(self):
        """The committed artifact's reading order, not just the renderer's.

        `TestRenderMarkdownRegrowthSection` asserts this over a synthetic
        report; here it is asserted over the file the operator actually
        opens, which is also what makes the heading's PRESENCE a merge-lane
        gate rather than a unit-test property.
        """
        lines = _mod().DEFAULT_REPORT_MD.read_text(
            encoding='utf-8').splitlines()

        assert (
            lines.index('## By query kind')
            < lines.index('## Regrowth deltas')
            < lines.index('## D10 — audit-recall over the labeled fixture')
        )

    def test_the_regrowth_table_headers_are_the_pinned_column_sets(self):
        """Same contract as `test_the_table_header_is_the_pinned_column_set`.

        Both tables, because both carry metric columns and a column dropped
        from either is a metric dropped from the decision.
        """
        mod = _mod()
        section = _section(
            mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8'),
            '## Regrowth deltas',
        )

        assert _header_row(mod.REGROWTH_TABLE_COLUMNS) in section
        assert _header_row(mod.REGROWTH_STAMPING_COLUMNS) in section

    def test_there_is_one_regrowth_bullet_per_read_arm(self):
        """Located by `regrowth_bullet_prefix` — the renderer's own anchor.

        Same discipline as the pin and stored-gap bullets: the prose stays
        free to change, and what is pinned is that every read arm the tables
        report also gets a sentence a reader can find.
        """
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')

        for arm in mod.REGROWTH_READ_ARMS:
            bullets = [
                line for line in rendered.splitlines()
                if line.startswith(mod.regrowth_bullet_prefix(arm))
            ]
            assert len(bullets) == 1, f'{arm}: {len(bullets)} regrowth bullets'

    def test_the_not_blind_authored_disclosure_is_carried_verbatim(self):
        """The probe's numbers are not protected the way the six arms' are.

        Verbatim rather than by keyword: the disclosure is load-bearing prose
        — it states WHY the protection is unrecoverable and what the fixture
        commit does and does not prove — and a paraphrase that lost the
        "partial audit trail and nothing more" qualifier would let the
        section be read as a blind measurement.
        """
        mod = _mod()
        rendered = mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8')

        assert mod.REGROWTH_BLIND_AUTHORING_DISCLOSURE in rendered

    def test_no_regrowth_cell_in_the_committed_tables_is_unmeasured(self):
        """`—` beside real numbers reads as a tie, exactly as in the decision
        table above.  In a FULL committed run every cell was measured.

        Substring-tested per cell, not by cell equality: a delta cell is
        `base → after (Δ)`, so an unmeasured after or delta hides INSIDE an
        otherwise populated-looking cell.
        """
        mod = _mod()
        section = _section(
            mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8'),
            '## Regrowth deltas',
        )
        rows = [
            *_rows_under(section, _header_row(mod.REGROWTH_TABLE_COLUMNS)),
            *_rows_under(section, _header_row(mod.REGROWTH_STAMPING_COLUMNS)),
        ]

        # Anti-vacuity: a section whose tables failed to render would make
        # every assertion below run zero times and pass.
        assert len(rows) == (
            len(mod.REGROWTH_MODES) * len(mod.REGROWTH_READ_ARMS)
            + len(mod.REGROWTH_READ_ARMS)
        ), f'{len(rows)} regrowth rows'
        for row in rows:
            for cell in _cells(row):
                assert mod._NO_MEASUREMENT not in cell, row


class TestTheCommittedPairAgrees:
    """The two committed halves must describe the SAME run.

    `render_markdown` is byte-deterministic for identical input, and
    `write_artifacts` renders the markdown BEFORE either atomic replace
    precisely so the pair can never describe different runs.  Nothing
    checked that property against the COMMITTED pair, though — so a
    renderer edit could, and did, leave a stale markdown claiming things
    the current code no longer says while every other committed-artifact
    test stayed green.  Those tests assert table headers, bullet anchors,
    a disclosure constant and the absence of `—` cells; a derived SENTENCE
    that went stale is invisible to all of them.

    This is the mechanical gate for that.  It is a pure file read, so it
    runs in the merge lane on every commit, which is the point: it goes red
    the moment a renderer change lands without the artifact being
    re-rendered, rather than leaving a wrong sentence in the artifact for a
    downstream reader to find.

    Pins NO metric value, rate or bound.  What it pins is that the two
    halves agree.
    """

    #: The step-23 re-render.  Named in the failure message because the fix
    #: for this test is never "edit the markdown" — it is "re-render it from
    #: the JSON", and a reader who hand-edits the prose to match reintroduces
    #: exactly the drift this test exists to catch.
    RE_RENDER = (
        "cd fused-memory && uv run python -c \"\n"
        "import json, importlib.util, sys, pathlib\n"
        "p = pathlib.Path('scripts/bake_off_storage_shape.py')\n"
        "spec = importlib.util.spec_from_file_location("
        "'bake_off_storage_shape', p)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['bake_off_storage_shape'] = m\n"
        "spec.loader.exec_module(m)\n"
        "report = json.loads(m.DEFAULT_REPORT_JSON.read_text(encoding='utf-8'))\n"
        "m._atomic_write_text(m.DEFAULT_REPORT_MD, m.render_markdown(report))\n"
        "\""
    )

    def test_the_committed_markdown_is_what_the_renderer_produces_today(self):
        mod = _mod()

        assert mod.DEFAULT_REPORT_MD.read_text(encoding='utf-8') == (
            mod.render_markdown(_committed_report())
        ), (
            f'{mod.DEFAULT_REPORT_MD.name} is stale: it is not what the '
            f'current renderer produces from '
            f'{mod.DEFAULT_REPORT_JSON.name}.  The measurements have not '
            f'changed — re-render the markdown, do NOT hand-edit it and do '
            f'NOT re-run the bake-off:\n\n{self.RE_RENDER}'
        )

    def test_rendering_the_committed_json_twice_is_byte_identical(self):
        """The determinism contract, re-anchored on the real committed data
        rather than a synthetic report.

        The test above is only a gate if re-rendering is reproducible: a
        renderer that varied run to run would make it flap, and the fix it
        names would not fix anything.
        """
        mod = _mod()
        report = _committed_report()

        assert mod.render_markdown(report) == mod.render_markdown(report)
