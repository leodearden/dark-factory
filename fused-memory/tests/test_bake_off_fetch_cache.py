"""Tests for the fetch-replay cache in bake_off_storage_shape.py (task 4004).

`fetch_arm` is the only part of the E2 bake-off that costs an embedder call
and a live Qdrant collection; everything downstream of it (`read_path`,
`measure_arm`, `rescore`, `build_report`, `render_markdown`) is already pure.
Dumping its return value and replaying it makes every read-side variant free
and — the point of this module — makes the metric code unit-testable against
REAL rankings rather than only hand-built ones.

The script is loaded via importlib so it can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily (via ``_mod()``)
rather than bound at import time.

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

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'

#: MEASUREMENT ANCHOR — recorded BEFORE a line of task-4004 code was written,
#: so a later reader can tell "the cache replays a different corpus" from "the
#: cache is truncated" without re-deriving either.
#:
#: Anchor commit: ff303320c7c3d90b093076965992dac246db062a
#: Live-run environment confirmed available at that commit:
#:   - Qdrant  http://localhost:6333/collections -> HTTP 200
#:   - OPENAI_API_KEY set
#: If either is unavailable when the measurement run happens, the run
#: ESCALATES (category='infra_issue').  No number is ever estimated, fabricated
#: or hand-edited into the report artifacts: every measured cell comes from a
#: real run, or renders as the no-measurement em dash.
#:
#: sha256 of the five committed E2 fixtures at the anchor commit.  These are
#: the inputs `materialize_arm` is deterministic over, so a fetch cache dumped
#: against them is replayable exactly as long as they hash to these values.
#: Asserted here rather than only inside the cache so a fixture edit that
#: silently invalidates the committed cache fails a PURE test in the merge
#: lane, not a live run nobody reruns.
ANCHOR_COMMIT = 'ff303320c7c3d90b093076965992dac246db062a'

ANCHOR_FIXTURE_SHA256: dict[str, str] = {
    'write_triage_calibration.jsonl':
        'fa5958f3634ace98b846ac398cdfe28f2e105a746f0348fe48fb5ed08cd03fe3',
    'memory_eval_topic_registry.json':
        '23b5ba77d59b10854a000fe57c2ef4766033bedfd51335de45bcec467ae3ae30',
    'e2_arm_claims.jsonl':
        '0b09c7de1c30c38570543f1705f01c5b4ac5970618f64545facb486e6991c257',
    'e2_query_set.jsonl':
        'c0c4872d2bb76e5e28a3e6660cf80d4b838712fdbf938624d02a0217e12c26d0',
    'e2_distractor_slab.jsonl':
        '8663a11024d14fb7201591a191f33a628d26f44d2449298db9182fef66b57e57',
}


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
# pre-2 — the measurement anchor, asserted rather than merely commented
# ===========================================================================

import hashlib  # noqa: E402

import pytest  # noqa: E402


class TestMeasurementAnchor:
    """The committed fetch cache is only replayable against THESE fixtures.

    `materialize_arm` is deterministic over the five committed fixtures, and
    the fetch cache stores `(shape, query_id) -> [(record_id, score)]` keyed
    on the uuid5 `record_id` those fixtures derive.  Edit a fixture and the
    cache still LOADS — it just describes a corpus that no longer exists, and
    the replayed report would publish a stale ranking as a fresh measurement.

    This is the cheap, pure, merge-lane half of that guard: it fails the
    moment a fixture moves, in a test that actually runs, instead of only
    inside a live run nobody reruns.  The expensive half — the per-shape
    corpus fingerprint carried in the dump — is step-3.
    """

    @pytest.mark.parametrize('name', sorted(ANCHOR_FIXTURE_SHA256))
    def test_fixture_still_hashes_to_the_anchor(self, name: str) -> None:
        path = FIXTURES_DIR / name
        assert path.exists(), f'{name} vanished since {ANCHOR_COMMIT}'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == ANCHOR_FIXTURE_SHA256[name], (
            f'{name} changed since the task-4004 measurement anchor '
            f'{ANCHOR_COMMIT}. The committed fetch cache '
            f'(tests/fixtures/e2_fetch_cache.json) was dumped against the old '
            f'bytes, so replaying it now would measure a corpus that no longer '
            f'exists. Re-run the seeding pass with --dump-fetches and update '
            f'this anchor in the same commit — do NOT just edit the expected '
            f'digest.'
        )

    def test_anchor_covers_every_fixture_the_bake_off_defaults_to(self) -> None:
        """No fixture may drift out of the anchor's coverage unnoticed."""
        mod = _mod()
        defaults = [
            mod.DEFAULT_ALPHA_FIXTURE_PATH,
            mod.DEFAULT_REGISTRY_PATH,
            mod.DEFAULT_ARM_CLAIMS_PATH,
            mod.DEFAULT_QUERY_SET_PATH,
            mod.DEFAULT_DISTRACTOR_SLAB_PATH,
        ]
        assert {Path(p).name for p in defaults} == set(ANCHOR_FIXTURE_SHA256)


# ===========================================================================
# step-1 — the round-trip contract
# ===========================================================================
#
# Every hit list here is HAND-BUILT with exactly-known answers, so each
# assertion is exact and tolerance-free — the same discipline the rest of the
# bake-off's pure tests keep.

import json  # noqa: E402

#: Fake mem0 point ids.  Mem0 mints these fresh on EVERY seeding run
#: (`Mem0Backend.add` pins `infer=False`, which routes to the uuid4
#: `_create_memory` path), and `_search` joins its results back through
#: `seeded.by_stored_id[item['id']]`.  A cache keyed on them is therefore
#: structurally unreplayable — which is what `test_dump_never_contains_a_mem0_point_id`
#: exists to prevent.
_POINT_IDS = ('mem0-point-aaaa', 'mem0-point-bbbb', 'mem0-point-cccc')


def _record(record_id: str, *, content: str = 'body', topic: str = 't',
            claim_ids: tuple[str, ...] = ()) -> object:
    """One arm record, in the shape `materialize_arm` would emit it."""
    return _mod().ArmRecord(
        record_id=record_id,
        content=content,
        metadata={'category': 'procedural_knowledge', 'topic': topic},
        cluster_id='c1',
        claim_ids=list(claim_ids),
        role='peer',
    )


def _seeded(shape: str, records: list) -> object:
    """A `SeededArm` with `by_stored_id` populated the way `seed_arm` would.

    The fake point ids are deliberately UNRELATED to the record ids: that is
    the real relationship (mem0 mints its own uuid), and a test whose point
    ids happened to equal its record ids could not tell the two apart.
    """
    seeded = _mod()._index_arm(shape, f'proj_{shape}', f'coll_{shape}', records, [])
    for point_id, record in zip(_POINT_IDS, records, strict=False):
        seeded.by_stored_id[point_id] = record
    return seeded


def _fetched(queries: dict, probes: dict) -> dict:
    """A `fetch_arm` return value, built from (record, score) pairs."""
    hit = _mod().ScoredHit
    return {
        'queries': {
            qid: [hit(record=r, relevance_score=s) for r, s in pairs]
            for qid, pairs in queries.items()
        },
        'probes': {
            cid: [hit(record=r, relevance_score=s) for r, s in pairs]
            for cid, pairs in probes.items()
        },
    }


def _provenance(mod, records_by_shape: dict, *, search_limit: int = 10,
                guard_threshold: float = 0.85,
                embedder_model: str = 'text-embedding-3-small') -> dict:
    """A provenance block for hand-built arms, as the live driver would build.

    Required on every dump: a cache with no per-shape corpus fingerprint can
    never be loaded, so `dump_fetches` refuses to produce one rather than
    letting a full seeding run be spent on an unusable file.  Its contents are
    pinned by `TestFetchCacheProvenance` below.
    """
    return mod.fetch_cache_provenance(
        records_by_shape=records_by_shape,
        fixtures=[FIXTURES_DIR / name for name in sorted(ANCHOR_FIXTURE_SHA256)],
        search_limit=search_limit,
        guard_threshold=guard_threshold,
        embedder_model=embedder_model,
    )


class TestFetchCacheRoundTrip:
    """`dump_fetches`/`load_fetches` preserve a `fetch_arm` result exactly.

    The cache exists so every read-side variant costs zero dollars and
    seconds rather than a reseed, forever, and so the metric code is
    unit-testable against REAL rankings.  Both properties require the
    round trip to be lossless in the ways the metrics actually read:
    membership, RANK ORDER, and the float score the threshold replay
    consumes.
    """

    def test_round_trips_queries_and_probes_to_scored_hits(self, tmp_path):
        mod = _mod()
        r1, r2, r3 = _record('rec-1'), _record('rec-2'), _record('rec-3')
        seeded = _seeded('c_peers', [r1, r2, r3])
        fetched = _fetched(
            queries={'q1': [(r1, 0.9), (r2, 0.5)], 'q2': [(r3, 0.25)]},
            probes={'cl1': [(r2, 0.7), (r3, 0.1)]},
        )

        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'c_peers': fetched},
            provenance=_provenance(mod, {'c_peers': [r1, r2, r3]}),
        )
        loaded = mod.load_fetches(path, {'c_peers': seeded})

        assert set(loaded) == {'c_peers'}
        arm = loaded['c_peers']
        assert set(arm) == {'queries', 'probes'}
        assert set(arm['queries']) == {'q1', 'q2'}
        assert set(arm['probes']) == {'cl1'}
        for hit in arm['queries']['q1']:
            assert isinstance(hit, mod.ScoredHit)
        # Rehydrated through `records_by_id`, so the records are the ARM's own
        # objects — not reconstructions that would compare equal but carry a
        # different metadata dict for a transform to mutate.
        assert arm['queries']['q1'][0].record is r1
        assert arm['probes']['cl1'][1].record is r3

    def test_rank_order_is_preserved_exactly(self, tmp_path):
        """Every metric in this experiment is rank-based; order IS the data."""
        mod = _mod()
        records = [_record(f'rec-{i}') for i in range(6)]
        seeded = _seeded('b_grouped', records)
        # Deliberately NOT sorted by id, and NOT monotonic in score order of
        # id: a dump that round-tripped through a dict-of-id would silently
        # re-sort this and every rank assertion downstream would still "pass".
        ranked = [(records[4], 0.9), (records[0], 0.8), (records[3], 0.7),
                  (records[1], 0.6), (records[5], 0.5), (records[2], 0.4)]
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'b_grouped': _fetched({'q1': ranked}, {})},
            provenance=_provenance(mod, {'b_grouped': records}),
        )

        loaded = mod.load_fetches(path, {'b_grouped': seeded})
        assert [h.record.record_id for h in loaded['b_grouped']['queries']['q1']] == \
            [r.record_id for r, _ in ranked]

    def test_scores_survive_as_floats_bit_for_bit(self, tmp_path):
        """The guard-threshold replay is the one sanctioned absolute-score
        metric (`ScoredHit`'s docstring), so a rounded score would move a
        column that reads as a shape finding."""
        mod = _mod()
        awkward = [0.1 + 0.2, 1 / 3, 0.30000000000000004, 1e-17, 0.9999999999999999]
        records = [_record(f'rec-{i}') for i in range(len(awkward))]
        seeded = _seeded('status_quo', records)
        pairs = list(zip(records, awkward, strict=True))
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'status_quo': _fetched({'q1': pairs}, {})},
            provenance=_provenance(mod, {'status_quo': records}),
        )

        loaded = mod.load_fetches(path, {'status_quo': seeded})
        scores = [h.relevance_score for h in loaded['status_quo']['queries']['q1']]
        assert scores == awkward
        assert all(isinstance(s, float) for s in scores)

    def test_dump_never_contains_a_mem0_point_id(self, tmp_path):
        """THE cache landmine.

        `_search` (bake_off_storage_shape.py:2699) joins hits via
        `seeded.by_stored_id[item['id']]`, and `Mem0Backend.add` pins
        `infer=False` — which routes to mem0's fresh-uuid `_create_memory`
        path.  So `item['id']` is newly minted on every seeding run and a
        cache keyed on it rehydrates to nothing on the next run, silently.
        Only the deterministic uuid5 `record_id` (`_derive_record_id`) may
        appear.
        """
        mod = _mod()
        r1, r2 = _record('rec-1'), _record('rec-2')
        seeded = _seeded('c_peers', [r1, r2])
        # The premise, made explicit: these records really ARE reachable in
        # this run under mem0 point ids, which is the join `_search` uses.
        assert set(seeded.by_stored_id) == set(_POINT_IDS[:2])

        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'c_peers': _fetched({'q1': [(r1, 0.9)]}, {'cl1': [(r2, 0.5)]})},
            provenance=_provenance(mod, {'c_peers': [r1, r2]}),
        )

        raw = path.read_text(encoding='utf-8')
        for point_id in _POINT_IDS:
            assert point_id not in raw, (
                f'{point_id!r} leaked into the fetch cache. mem0 mints a fresh '
                f'point id on every seeding run, so this cache is replayable '
                f'exactly once — and would then rehydrate to an empty ranking '
                f'that measures as a zero.'
            )
        assert 'rec-1' in raw and 'rec-2' in raw

    def test_the_serialized_ranking_is_record_id_and_score_only(self, tmp_path):
        """A ranking map, not a corpus copy.

        Record CONTENT is deliberately absent: it is fully reconstructible
        from the committed fixtures via `materialize_arm`, so storing it would
        make the cache both large and capable of DISAGREEING with the fixtures
        it claims to describe.
        """
        mod = _mod()
        r1 = _record('rec-1', content='SOME DISTINCTIVE BODY')
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'c_peers': _fetched({'q1': [(r1, 0.875)]}, {})},
            provenance=_provenance(mod, {'c_peers': [r1]}),
        )

        doc = json.loads(path.read_text(encoding='utf-8'))
        assert doc['arms']['c_peers']['queries']['q1'] == [['rec-1', 0.875]]
        assert 'SOME DISTINCTIVE BODY' not in path.read_text(encoding='utf-8')

    def test_probe_lists_round_trip_post_self_filter(self, tmp_path):
        """`fetch_arm` stores probes AFTER dropping the probing write's own
        records (:2795-2797), so the cache holds the filtered list and replay
        must not re-filter — that would drop a second round of records."""
        mod = _mod()
        r1, r2 = _record('rec-1'), _record('rec-2')
        seeded = _seeded('c_peers', [r1, r2])
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path,
            {'c_peers': _fetched({'q1': [(r1, 0.9)]},
                                 {'cl1': [(r2, 0.4)], 'cl2': []})},
            provenance=_provenance(mod, {'c_peers': [r1, r2]}),
        )

        loaded = mod.load_fetches(path, {'c_peers': seeded})
        probes = loaded['c_peers']['probes']
        assert [h.record.record_id for h in probes['cl1']] == ['rec-2']
        # An EMPTY probe list is a real outcome (every hit was the probe's own
        # record) and must survive as an empty list, not vanish as a missing
        # key — `guard_adequacy` scores "the guard never fired" from it.
        assert probes['cl2'] == []

    def test_dumping_the_same_fetch_twice_is_byte_identical(self, tmp_path):
        """The cache is a COMMITTED artifact; a re-dump must diff cleanly."""
        mod = _mod()
        records = [_record(f'rec-{i}') for i in range(3)]
        # Insertion order deliberately reversed between the two dumps: JSON
        # object key order follows insertion order, so only a sorted dump is
        # stable here.
        forward = _fetched(
            {'q2': [(records[1], 0.4)], 'q1': [(records[0], 0.9)]},
            {'clB': [(records[2], 0.2)], 'clA': [(records[0], 0.3)]},
        )
        backward = _fetched(
            {'q1': [(records[0], 0.9)], 'q2': [(records[1], 0.4)]},
            {'clA': [(records[0], 0.3)], 'clB': [(records[2], 0.2)]},
        )
        first, second = tmp_path / 'a.json', tmp_path / 'b.json'
        prov = _provenance(mod, {'c_peers': records, 'status_quo': records})
        mod.dump_fetches(
            first, {'c_peers': forward, 'status_quo': forward}, provenance=prov,
        )
        mod.dump_fetches(
            second, {'status_quo': backward, 'c_peers': backward}, provenance=prov,
        )

        assert first.read_text(encoding='utf-8') == second.read_text(encoding='utf-8')

    def test_dump_is_pure_over_its_input(self, tmp_path):
        """Dumping must not mutate the live fetch result the caller still
        measures from — `run_arm` calls `measure_arm` twice off one fetch."""
        mod = _mod()
        r1, r2 = _record('rec-1'), _record('rec-2')
        fetched = _fetched({'q1': [(r1, 0.9), (r2, 0.5)]}, {'cl1': [(r2, 0.5)]})
        before = {
            'queries': {k: list(v) for k, v in fetched['queries'].items()},
            'probes': {k: list(v) for k, v in fetched['probes'].items()},
        }

        mod.dump_fetches(
            tmp_path / 'cache.json', {'c_peers': fetched},
            provenance=_provenance(mod, {'c_peers': [r1, r2]}),
        )

        assert fetched['queries']['q1'] == before['queries']['q1']
        assert fetched['probes']['cl1'] == before['probes']['cl1']
        assert fetched['queries']['q1'][0].record is r1

    def test_a_cached_record_id_the_arm_never_seeded_is_refused(self, tmp_path):
        """Mirrors `_search`'s own refusal (:2701): dropping an unjoinable hit
        would silently shrink the measured ranking."""
        mod = _mod()
        r1 = _record('rec-1')
        seeded = _seeded('c_peers', [r1])
        path = tmp_path / 'cache.json'
        doc = json.loads(
            (mod.dump_fetches(
                path, {'c_peers': _fetched({'q1': [(r1, 0.9)]}, {})},
                provenance=_provenance(mod, {'c_peers': [r1]}),
             )
             ).read_text(encoding='utf-8'),
        )
        doc['arms']['c_peers']['queries']['q1'].append(['rec-nope', 0.1])
        path.write_text(json.dumps(doc), encoding='utf-8')

        with pytest.raises(Exception) as excinfo:  # noqa: PT011 — typed in step-3
            mod.load_fetches(path, {'c_peers': seeded})
        assert 'rec-nope' in str(excinfo.value)


# ===========================================================================
# step-3 — the cache must refuse to silently measure a stale corpus
# ===========================================================================
#
# A replayed report is INDISTINGUISHABLE from a live one in the artifact.
# That is the whole value of the cache and also its whole danger: a cache
# replayed against drifted fixtures publishes a stale ranking as a fresh
# measurement, and nobody reading `plans/read-transform-selection-report.md`
# can tell.  So every way the cache can be wrong is loud at LOAD time.
#
# This is the same posture `_search` already takes at :2688 — "an empty
# ranking must never be published as a measured zero" — extended to the
# replay path, and the repo's no-silent-fail-soft norm requires it.


class TestCorpusFingerprint:
    """`corpus_fingerprint` answers: is this the corpus the cache describes?"""

    def test_is_stable_for_the_same_records(self):
        mod = _mod()
        records = [_record('rec-1', content='a'), _record('rec-2', content='b')]
        again = [_record('rec-1', content='a'), _record('rec-2', content='b')]
        assert mod.corpus_fingerprint(records) == mod.corpus_fingerprint(again)

    @pytest.mark.parametrize(
        ('field', 'mutated'),
        [
            ('content', lambda: [_record('rec-1', content='CHANGED')]),
            ('record_id', lambda: [_record('rec-CHANGED', content='a')]),
            ('metadata', lambda: [_record('rec-1', content='a', topic='CHANGED')]),
        ],
    )
    def test_changes_when_any_measured_field_changes(self, field, mutated):
        """Content, id and metadata all feed retrieval, so all three count."""
        mod = _mod()
        baseline = mod.corpus_fingerprint([_record('rec-1', content='a')])
        assert mod.corpus_fingerprint(mutated()) != baseline, field

    def test_changes_when_a_record_is_added_or_dropped(self):
        mod = _mod()
        one = mod.corpus_fingerprint([_record('rec-1')])
        two = mod.corpus_fingerprint([_record('rec-1'), _record('rec-2')])
        assert one != two

    def test_is_order_independent(self):
        """PINNED CHOICE, not an accident.

        `materialize_arm` is already deterministically ordered, so this is
        never exercised by the real pipeline.  The fingerprint sorts anyway
        because its question is "does the cache describe THIS corpus" — a set
        question — and answering it set-wise means a reordering cannot produce
        a scary false alarm while every content/id/metadata change still does.
        """
        mod = _mod()
        a, b = _record('rec-1', content='a'), _record('rec-2', content='b')
        assert mod.corpus_fingerprint([a, b]) == mod.corpus_fingerprint([b, a])


class TestFetchCacheProvenance:
    """The dump carries everything a replayed report cannot reconstruct."""

    def test_carries_a_per_shape_corpus_fingerprint(self, tmp_path):
        mod = _mod()
        peers = [_record('rec-1'), _record('rec-2')]
        quo = [_record('rec-3')]
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path,
            {'c_peers': _fetched({'q1': [(peers[0], 0.9)]}, {}),
             'status_quo': _fetched({'q1': [(quo[0], 0.9)]}, {})},
            provenance=_provenance(
                mod, {'c_peers': peers, 'status_quo': quo},
            ),
        )

        doc = json.loads(path.read_text(encoding='utf-8'))
        prints = doc['provenance']['corpus_fingerprints']
        assert set(prints) == {'c_peers', 'status_quo'}
        assert prints['c_peers'] == mod.corpus_fingerprint(peers)
        assert prints['status_quo'] == mod.corpus_fingerprint(quo)
        # Per SHAPE, not one global digest: the three arms materialize
        # different corpora from the same fixtures, and a single digest could
        # not say WHICH arm drifted.
        assert prints['c_peers'] != prints['status_quo']

    def test_carries_the_five_fixture_sha256s(self, tmp_path):
        """`fixture_provenance` stamps the last-touching COMMIT, which does not
        change when a fixture is edited in the working tree. The cache needs a
        content hash, so it carries both semantics rather than conflating them."""
        mod = _mod()
        path = tmp_path / 'cache.json'
        records = [_record('rec-1')]
        mod.dump_fetches(
            path, {'c_peers': _fetched({'q1': [(records[0], 0.9)]}, {})},
            provenance=_provenance(mod, {'c_peers': records}),
        )

        doc = json.loads(path.read_text(encoding='utf-8'))
        digests = {row['path']: row['sha256'] for row in doc['provenance']['fixtures']}
        assert len(digests) == 5
        for name, expected in ANCHOR_FIXTURE_SHA256.items():
            key = next(p for p in digests if p.endswith(name))
            assert digests[key] == expected
            # Repo-relative, never an absolute home-directory path — the same
            # rule `fixture_provenance` documents at :1852.
            assert not key.startswith('/')

    def test_dumping_without_provenance_is_refused_at_dump_time(self, tmp_path):
        """Fail BEFORE the run is spent, not after.

        A cache with no per-shape fingerprint can never be loaded
        (`_check_corpus_fingerprint` refuses it), so allowing the dump would
        only move the failure to load time — i.e. to after a full live seeding
        run had already produced an unusable file.
        """
        mod = _mod()
        with pytest.raises(mod.FetchCacheError, match='provenance'):
            mod.dump_fetches(
                tmp_path / 'cache.json',
                {'c_peers': _fetched({'q1': [(_record('rec-1'), 0.9)]}, {})},
                provenance={},
            )
        assert not (tmp_path / 'cache.json').exists()

    def test_carries_the_live_only_values_a_replay_cannot_reconstruct(self, tmp_path):
        """`resolve_guard_threshold(memory)` (:3264) navigates a live
        MemoryService and `config.embedder.model` (:3297) comes off the live
        config. Both reach the report's protocol block, so without them a
        replayed report is unbuildable — not merely less detailed."""
        mod = _mod()
        records = [_record('rec-1')]
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'c_peers': _fetched({'q1': [(records[0], 0.9)]}, {})},
            provenance=_provenance(
                mod, {'c_peers': records},
                search_limit=7, guard_threshold=0.83,
                embedder_model='text-embedding-3-large',
            ),
        )

        prov = mod.load_fetch_provenance(path)
        assert prov['search_limit'] == 7
        assert prov['guard_threshold'] == 0.83
        assert prov['embedder_model'] == 'text-embedding-3-large'


class TestFetchCacheRefusesAStaleCorpus:
    """Three named refusals, so an operator can tell them apart WITHOUT
    reading code: the fixtures moved, the cache is truncated, or the cache is
    too shallow for the requested depth."""

    def _dump(self, mod, tmp_path, records, *, queries, search_limit=10):
        path = tmp_path / 'cache.json'
        mod.dump_fetches(
            path, {'c_peers': _fetched(queries, {})},
            provenance=_provenance(
                mod, {'c_peers': records}, search_limit=search_limit,
            ),
        )
        return path

    def test_a_drifted_corpus_fingerprint_is_refused(self, tmp_path):
        mod = _mod()
        records = [_record('rec-1', content='AS DUMPED')]
        path = self._dump(
            mod, tmp_path, records, queries={'q1': [(records[0], 0.9)]},
        )
        # The fixtures moved: the same record id now materializes to different
        # content, so the cached ranking describes a corpus that is gone.
        drifted = _seeded('c_peers', [_record('rec-1', content='AFTER A FIXTURE EDIT')])

        with pytest.raises(mod.FetchCacheError) as excinfo:
            mod.load_fetches(path, {'c_peers': drifted})
        message = str(excinfo.value)
        assert 'c_peers' in message
        # BOTH fingerprints, so the operator can tell "the fixtures moved"
        # from "the cache is truncated" without reading code.
        assert mod.corpus_fingerprint(records)[:16] in message
        assert mod.corpus_fingerprint(drifted.records)[:16] in message

    def test_an_unfingerprinted_cache_is_refused(self, tmp_path):
        """A cache with no fingerprint cannot be checked, so it cannot be
        trusted — accepting it would reintroduce exactly the silent staleness
        the fingerprint exists to catch."""
        mod = _mod()
        records = [_record('rec-1')]
        path = self._dump(
            mod, tmp_path, records, queries={'q1': [(records[0], 0.9)]},
        )
        doc = json.loads(path.read_text(encoding='utf-8'))
        del doc['provenance']['corpus_fingerprints']['c_peers']
        path.write_text(json.dumps(doc), encoding='utf-8')

        with pytest.raises(mod.FetchCacheError, match='c_peers'):
            mod.load_fetches(path, {'c_peers': _seeded('c_peers', records)})

    def test_a_query_id_absent_from_the_cache_is_refused(self, tmp_path):
        """A PARTIAL cache measured as complete is the "empty ranking published
        as a measured zero" failure, one layer up."""
        mod = _mod()
        records = [_record('rec-1')]
        path = self._dump(
            mod, tmp_path, records, queries={'q1': [(records[0], 0.9)]},
        )

        with pytest.raises(mod.FetchCacheError) as excinfo:
            mod.load_fetches(
                path, {'c_peers': _seeded('c_peers', records)},
                expect_query_ids=['q1', 'q2-never-cached'],
            )
        assert 'q2-never-cached' in str(excinfo.value)

    def test_the_full_requested_query_set_loads(self, tmp_path):
        """The converse, so the refusal above is not vacuously always-on."""
        mod = _mod()
        records = [_record('rec-1'), _record('rec-2')]
        path = self._dump(
            mod, tmp_path, records,
            queries={'q1': [(records[0], 0.9)], 'q2': [(records[1], 0.4)]},
        )
        loaded = mod.load_fetches(
            path, {'c_peers': _seeded('c_peers', records)},
            expect_query_ids=['q1', 'q2'],
        )
        assert set(loaded['c_peers']['queries']) == {'q1', 'q2'}

    def test_a_cache_shallower_than_the_requested_limit_is_refused(self, tmp_path):
        """A limit-5 cache replayed at --limit 10 measures every arm over a
        5-deep window while the report claims 10 — a silently different
        experiment, not a smaller one."""
        mod = _mod()
        records = [_record('rec-1')]
        path = self._dump(
            mod, tmp_path, records, queries={'q1': [(records[0], 0.9)]},
            search_limit=5,
        )

        with pytest.raises(mod.FetchCacheError) as excinfo:
            mod.load_fetches(
                path, {'c_peers': _seeded('c_peers', records)}, expect_limit=10,
            )
        message = str(excinfo.value)
        assert '5' in message and '10' in message

    @pytest.mark.parametrize('requested', [5, 3])
    def test_a_cache_at_or_deeper_than_the_requested_limit_is_accepted(
        self, tmp_path, requested,
    ):
        """Over-fetching is fine — `read_path` truncates to the reader's budget
        anyway — so only UNDER-fetching is a refusal."""
        mod = _mod()
        records = [_record('rec-1')]
        path = self._dump(
            mod, tmp_path, records, queries={'q1': [(records[0], 0.9)]},
            search_limit=5,
        )
        loaded = mod.load_fetches(
            path, {'c_peers': _seeded('c_peers', records)}, expect_limit=requested,
        )
        assert list(loaded['c_peers']['queries']) == ['q1']
