"""Tests for calibrate_write_triage.py and its labeled curator fixture.

The script is loaded via importlib so it can be tested without sys.path
pollution — mirrors the pattern in test_audit_duplicate_memories.py.

Unlike that module, the loader here is invoked lazily (via ``_mod()``)
rather than bound at import time: the fixture-contract tests below are
about the committed data file alone and must stay runnable independently
of the script's existence. Every test in this file is free of
OPENAI_API_KEY, network and Qdrant dependencies — all similarity scores
and retrievals are injected.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import re
import types
from collections import Counter, defaultdict
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'calibrate_write_triage.py'
FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

VALID_LABELS = {'duplicate', 'canonical', 'distinct', 'pseudo_contradiction'}

# Canonical UUID prefixes the curator's adjudications are pinned to.
# esc-5606: three scoped techniques the curator ruled are "not three
# competing answers to one question" — the hard-negative triple.
CANONICAL_5606 = 'af367f55'
DISTINCT_5606 = ('e218437a', '29463217', '669fadfa')
# esc-5557 / esc-5626: entries that read as contradictory but were
# adjudicated both-correct ("the contradiction was an omission, not a
# disagreement").
CANONICAL_5557 = 'df6ff45d'
CANONICAL_5626 = '70fd0700'
# Deliberately excluded from the fixture: an intra-session canonical
# (created then superseded inside the curation run) and a spent Stage-1
# flag marker (a meta-record about a memory, not a topical entry).
EXCLUDED_IDS = ('8d79e0e4', '43a47400')


def _load_module() -> types.ModuleType:
    """Load calibrate_write_triage.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'calibrate_write_triage'
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


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def records() -> list[dict]:
    """Parse the committed fixture line-by-line.

    Deliberately parsed here with the stdlib rather than through the
    script's own load_fixture(), so a bug in the loader cannot mask a
    defect in the data (and vice versa).
    """
    assert FIXTURE_PATH.exists(), f'fixture missing: {FIXTURE_PATH}'
    out: list[dict] = []
    for lineno, line in enumerate(FIXTURE_PATH.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - data defect
            raise AssertionError(f'{FIXTURE_PATH}:{lineno} is not valid JSON: {exc}') from exc
    return out


def _by_prefix(records: list[dict], prefix: str) -> list[dict]:
    return [r for r in records if r['memory_id'].startswith(prefix)]


def _cluster(records: list[dict], canonical_prefix: str) -> list[dict]:
    canon = _by_prefix(records, canonical_prefix)
    assert len(canon) == 1, f'expected exactly one record with id prefix {canonical_prefix}'
    return [r for r in records if r['cluster_id'] == canon[0]['memory_id']]


class TestFixtureSchema:
    """Per-record shape. No total-count assertion — see fixtures/README.md."""

    def test_fixture_is_non_empty(self, records: list[dict]) -> None:
        assert records, 'fixture must carry at least one record'

    def test_every_record_has_a_full_uuid_memory_id(self, records: list[dict]) -> None:
        bad = [r.get('memory_id') for r in records if not UUID_RE.match(str(r.get('memory_id')))]
        assert not bad, f'memory_id must be a full 36-char UUID; got {bad[:5]}'

    def test_every_record_has_non_empty_string_content(self, records: list[dict]) -> None:
        bad = [
            r['memory_id'] for r in records
            if not isinstance(r.get('content'), str) or not r['content'].strip()
        ]
        assert not bad, (
            'content must be a non-empty str — an unrecovered record is EXCLUDED '
            f'and tallied, never emitted with a placeholder; got {bad[:5]}'
        )

    def test_every_record_has_a_string_category(self, records: list[dict]) -> None:
        bad = [r['memory_id'] for r in records if not isinstance(r.get('category'), str)]
        assert not bad, f'category must be a str (it lives at metadata.category); got {bad[:5]}'

    def test_every_record_has_a_full_uuid_cluster_id(self, records: list[dict]) -> None:
        bad = [r.get('cluster_id') for r in records if not UUID_RE.match(str(r.get('cluster_id')))]
        assert not bad, (
            'cluster_id must be a full 36-char UUID (the CANONICAL memory id, '
            f'never the gate id); got {bad[:5]}'
        )

    def test_every_record_has_a_known_label(self, records: list[dict]) -> None:
        bad = {r['memory_id']: r.get('label') for r in records if r.get('label') not in VALID_LABELS}
        assert not bad, f'label must be one of {sorted(VALID_LABELS)}; got {bad}'

    def test_every_record_has_a_provenance_block(self, records: list[dict]) -> None:
        for r in records:
            prov = r.get('provenance')
            assert isinstance(prov, dict), f'{r["memory_id"]}: provenance must be a dict'
            for key in ('gate_id', 'transcript_line', 'source'):
                assert key in prov, f'{r["memory_id"]}: provenance missing {key!r}'
            assert isinstance(prov['transcript_line'], int), (
                f'{r["memory_id"]}: provenance.transcript_line must be an int'
            )
            assert str(prov['gate_id']).strip(), f'{r["memory_id"]}: provenance.gate_id is empty'
            assert str(prov['source']).strip(), f'{r["memory_id"]}: provenance.source is empty'

    def test_optional_fields_are_present_and_correctly_typed(self, records: list[dict]) -> None:
        for r in records:
            for key in ('agent_id', 'created_at'):
                assert key in r, f'{r["memory_id"]}: missing {key!r} (may be null, must be present)'
                assert r[key] is None or isinstance(r[key], str), (
                    f'{r["memory_id"]}: {key} must be a str or null'
                )


class TestClusterReferentialIntegrity:
    """Clusters are keyed by the CANONICAL memory UUID, never by the gate id.

    Gates esc-5534, esc-5547, esc-5561 and esc-5610 each produced TWO
    canonicals. Keying clusters by gate id would fuse two unrelated
    canonicals' member sets into one cluster, injecting pairs that are not
    duplicates into the positive class — dragging the derived T_high down
    and manufacturing the very false positives this calibration measures.
    """

    def test_memory_ids_are_unique(self, records: list[dict]) -> None:
        dupes = [mid for mid, n in Counter(r['memory_id'] for r in records).items() if n > 1]
        assert not dupes, f'duplicate memory_id rows: {dupes}'

    def test_every_cluster_id_resolves_to_a_present_canonical(self, records: list[dict]) -> None:
        canonical_ids = {r['memory_id'] for r in records if r['label'] == 'canonical'}
        dangling = sorted({
            r['cluster_id'] for r in records if r['cluster_id'] not in canonical_ids
        })
        assert not dangling, (
            f'cluster_id values with no canonical record in the file: {dangling}'
        )

    def test_every_cluster_contains_exactly_one_canonical(self, records: list[dict]) -> None:
        per_cluster: dict[str, list[str]] = defaultdict(list)
        for r in records:
            if r['label'] == 'canonical':
                per_cluster[r['cluster_id']].append(r['memory_id'])
        offenders = {c: ids for c, ids in per_cluster.items() if len(ids) != 1}
        assert not offenders, f'clusters without exactly one canonical: {offenders}'

    def test_each_canonical_is_a_member_of_its_own_cluster(self, records: list[dict]) -> None:
        offenders = [
            r['memory_id'] for r in records
            if r['label'] == 'canonical' and r['cluster_id'] != r['memory_id']
        ]
        assert not offenders, (
            f'a canonical must key its own cluster (cluster_id == memory_id); got {offenders}'
        )

    def test_at_least_one_cluster_has_two_or_more_duplicates(self, records: list[dict]) -> None:
        counts = Counter(r['cluster_id'] for r in records if r['label'] == 'duplicate')
        assert counts, 'fixture carries no duplicate-labeled records at all'
        assert max(counts.values()) >= 2, (
            'the positive class needs at least one cluster with >=2 duplicates, '
            'otherwise there are no duplicate-to-duplicate pairs to measure'
        )


class TestCuratorAdjudications:
    """The specific human rulings the hard-negative class depends on."""

    def test_esc_5606_cluster_carries_exactly_three_distinct_records(
        self, records: list[dict],
    ) -> None:
        cluster = _cluster(records, CANONICAL_5606)
        distinct = [r for r in cluster if r['label'] == 'distinct']
        assert len(distinct) == 3, (
            'the esc-5606 cluster must carry exactly 3 records labeled distinct — '
            'the curator ruled they are "not three competing answers to one question"; '
            f'got {[(r["memory_id"], r["label"]) for r in cluster]}'
        )
        got = sorted(r['memory_id'][:8] for r in distinct)
        assert got == sorted(DISTINCT_5606), (
            f'expected the scoped-technique triple {sorted(DISTINCT_5606)}, got {got}'
        )

    def test_esc_5606_distinct_records_are_not_labeled_duplicates(
        self, records: list[dict],
    ) -> None:
        for prefix in DISTINCT_5606:
            found = _by_prefix(records, prefix)
            assert len(found) == 1, f'{prefix} must appear exactly once; got {len(found)}'
            assert found[0]['label'] == 'distinct', (
                f'{prefix} was curator-ruled distinct, not a duplicate; '
                f'got label={found[0]["label"]!r}'
            )

    @pytest.mark.parametrize(
        ('canonical_prefix', 'gate'),
        [(CANONICAL_5557, 'esc-5557'), (CANONICAL_5626, 'esc-5626')],
    )
    def test_pseudo_contradiction_clusters_are_labeled(
        self, records: list[dict], canonical_prefix: str, gate: str,
    ) -> None:
        cluster = _cluster(records, canonical_prefix)
        pseudo = [r for r in cluster if r['label'] == 'pseudo_contradiction']
        assert pseudo, (
            f'the {gate} cluster (canonical {canonical_prefix}) must carry '
            'pseudo_contradiction records — both entries were adjudicated CORRECT, '
            'the contradiction was an omission rather than a disagreement; '
            f'got {[(r["memory_id"][:8], r["label"]) for r in cluster]}'
        )

    @pytest.mark.parametrize('prefix', EXCLUDED_IDS)
    def test_excluded_records_are_absent(self, records: list[dict], prefix: str) -> None:
        assert not _by_prefix(records, prefix), (
            f'{prefix} must be excluded from the fixture: it is either an intra-session '
            'canonical (created then superseded within the curation run) or a spent '
            'Stage-1 flag marker (a meta-record about a memory, not a topical entry)'
        )


# ---------------------------------------------------------------------------
# load_fixture / build_pair_sets
# ---------------------------------------------------------------------------

def _rec(mid: str, cluster: str, label: str, *, gate: str = 'esc-0000', content: str = 'x') -> dict:
    """A minimal hand-built record (deliberately not from the real fixture)."""
    return {
        'memory_id': mid,
        'content': content,
        'category': 'procedural_knowledge',
        'agent_id': None,
        'created_at': None,
        'cluster_id': cluster,
        'label': label,
        'provenance': {'gate_id': gate, 'transcript_line': 1, 'source': 'test'},
    }


def _keys(pairs) -> set[tuple[str, str]]:
    """Normalise a pair class to a set of sorted memory_id tuples."""
    out = set()
    for p in pairs:
        a, b = (p['a'], p['b']) if isinstance(p, dict) else (p[0], p[1])
        out.add(tuple(sorted((a, b))))
    return out


class TestLoadFixture:
    def test_reads_every_record(self, tmp_path: Path) -> None:
        path = tmp_path / 'f.jsonl'
        path.write_text(
            json.dumps(_rec('a', 'a', 'canonical')) + '\n'
            + json.dumps(_rec('b', 'a', 'duplicate')) + '\n',
        )
        got = _mod().load_fixture(path)
        assert [r['memory_id'] for r in got] == ['a', 'b']

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / 'f.jsonl'
        path.write_text(json.dumps(_rec('a', 'a', 'canonical')) + '\n\n   \n')
        assert len(_mod().load_fixture(path)) == 1

    def test_malformed_line_raises_with_its_line_number(self, tmp_path: Path) -> None:
        """Loud over silent: a corrupt line must never be silently skipped.

        Skipping would shrink the measured population without saying so,
        yielding a report whose thresholds look fine but were computed on
        a subset.
        """
        path = tmp_path / 'f.jsonl'
        path.write_text(
            json.dumps(_rec('a', 'a', 'canonical')) + '\n'
            + '{not json\n'
            + json.dumps(_rec('c', 'a', 'duplicate')) + '\n',
        )
        with pytest.raises(ValueError, match='2'):
            _mod().load_fixture(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            _mod().load_fixture(tmp_path / 'nope.jsonl')


class TestBuildPairSets:
    """Three disjoint pair classes keyed on cluster_id (the canonical UUID)."""

    def test_returns_the_three_classes(self) -> None:
        pairs = _mod().build_pair_sets([
            _rec('c1', 'c1', 'canonical'),
            _rec('d1', 'c1', 'duplicate'),
        ])
        for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs'):
            assert key in pairs, f'missing pair class {key!r}'

    def test_duplicate_to_canonical_and_duplicate_to_duplicate_are_true_dups(self) -> None:
        """The curator-confirmed genuine rediscoveries — the positive class."""
        got = _mod().build_pair_sets([
            _rec('c1', 'c1', 'canonical'),
            _rec('d1', 'c1', 'duplicate'),
            _rec('d2', 'c1', 'duplicate'),
        ])
        assert _keys(got['true_dup_pairs']) == {('c1', 'd1'), ('c1', 'd2'), ('d1', 'd2')}
        assert _keys(got['unrelated_pairs']) == set()
        assert _keys(got['hard_negative_pairs']) == set()

    def test_members_of_different_clusters_are_unrelated(self) -> None:
        """The measured negative class.

        The corpus is domain-homogeneous, so unrelated scores have to be
        measured rather than assumed low.
        """
        got = _mod().build_pair_sets([
            _rec('c1', 'c1', 'canonical'),
            _rec('d1', 'c1', 'duplicate'),
            _rec('c2', 'c2', 'canonical'),
            _rec('d2', 'c2', 'duplicate'),
        ])
        assert _keys(got['unrelated_pairs']) == {
            ('c1', 'c2'), ('c1', 'd2'), ('c2', 'd1'), ('d1', 'd2'),
        }
        assert _keys(got['true_dup_pairs']) == {('c1', 'd1'), ('c2', 'd2')}

    def test_three_distinct_records_yield_three_hard_negatives_and_no_true_dups(self) -> None:
        """esc-5606: "not three competing answers to one question"."""
        got = _mod().build_pair_sets([
            _rec('x1', 'k', 'distinct'),
            _rec('x2', 'k', 'distinct'),
            _rec('x3', 'k', 'distinct'),
        ])
        assert len(_keys(got['hard_negative_pairs'])) == 3
        assert _keys(got['hard_negative_pairs']) == {('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')}
        assert _keys(got['true_dup_pairs']) == set()

    def test_a_distinct_record_never_pairs_with_its_own_canonical_as_a_true_dup(self) -> None:
        got = _mod().build_pair_sets([
            _rec('k', 'k', 'canonical'),
            _rec('x1', 'k', 'distinct'),
        ])
        assert _keys(got['true_dup_pairs']) == set()
        assert _keys(got['hard_negative_pairs']) == {('k', 'x1')}

    def test_pseudo_contradiction_pairs_are_hard_negatives(self) -> None:
        """esc-5557 / esc-5626: both entries adjudicated correct."""
        got = _mod().build_pair_sets([
            _rec('k', 'k', 'canonical'),
            _rec('p1', 'k', 'pseudo_contradiction'),
            _rec('p2', 'k', 'pseudo_contradiction'),
        ])
        assert _keys(got['hard_negative_pairs']) == {('k', 'p1'), ('k', 'p2'), ('p1', 'p2')}
        assert _keys(got['true_dup_pairs']) == set()

    def test_same_gate_different_clusters_are_unrelated_not_true_dups(self) -> None:
        """The two-canonicals-per-gate case (esc-5534/5547/5561/5610).

        Keying on gate_id instead of cluster_id would fuse these into one
        cluster and inject non-duplicate pairs into the positive class.
        """
        got = _mod().build_pair_sets([
            _rec('c1', 'c1', 'canonical', gate='esc-5547'),
            _rec('c2', 'c2', 'canonical', gate='esc-5547'),
        ])
        assert _keys(got['unrelated_pairs']) == {('c1', 'c2')}
        assert _keys(got['true_dup_pairs']) == set()

    def test_no_self_pairs(self) -> None:
        records = [
            _rec('c1', 'c1', 'canonical'), _rec('d1', 'c1', 'duplicate'),
            _rec('x1', 'c1', 'distinct'), _rec('c2', 'c2', 'canonical'),
        ]
        got = _mod().build_pair_sets(records)
        for name, pairs in got.items():
            for a, b in _keys(pairs):
                assert a != b, f'{name} contains a self-pair on {a}'

    def test_each_unordered_pair_appears_exactly_once(self) -> None:
        records = [
            _rec('c1', 'c1', 'canonical'), _rec('d1', 'c1', 'duplicate'),
            _rec('d2', 'c1', 'duplicate'), _rec('x1', 'c1', 'distinct'),
            _rec('c2', 'c2', 'canonical'), _rec('p1', 'c2', 'pseudo_contradiction'),
        ]
        got = _mod().build_pair_sets(records)
        raw: list[tuple[str, str]] = []
        for pairs in got.values():
            for p in pairs:
                a, b = (p['a'], p['b']) if isinstance(p, dict) else (p[0], p[1])
                raw.append(tuple(sorted((a, b))))
        dupes = [k for k, n in Counter(raw).items() if n > 1]
        assert not dupes, f'pairs emitted more than once: {dupes}'
        n = len(records)
        assert len(raw) == n * (n - 1) // 2, (
            'every unordered pair must be classified exactly once — '
            f'expected {n * (n - 1) // 2}, got {len(raw)}'
        )

    def test_the_three_classes_are_mutually_disjoint(self) -> None:
        records = [
            _rec('c1', 'c1', 'canonical'), _rec('d1', 'c1', 'duplicate'),
            _rec('x1', 'c1', 'distinct'), _rec('c2', 'c2', 'canonical'),
            _rec('p1', 'c2', 'pseudo_contradiction'),
        ]
        got = _mod().build_pair_sets(records)
        t, u, h = (_keys(got['true_dup_pairs']), _keys(got['unrelated_pairs']),
                   _keys(got['hard_negative_pairs']))
        assert t & u == set(), f'true_dup overlaps unrelated: {t & u}'
        assert t & h == set(), f'true_dup overlaps hard_negative: {t & h}'
        assert u & h == set(), f'unrelated overlaps hard_negative: {u & h}'

    def test_empty_and_single_record_inputs_yield_no_pairs(self) -> None:
        for records in ([], [_rec('c1', 'c1', 'canonical')]):
            got = _mod().build_pair_sets(records)
            assert all(len(v) == 0 for v in got.values()), f'{records} produced pairs'


# ---------------------------------------------------------------------------
# Per-category partition
# ---------------------------------------------------------------------------

def _cat_rec(mid: str, cluster: str, label: str, category: str | None) -> dict:
    """A minimal record with an explicit category (possibly absent/empty)."""
    record = _rec(mid, cluster, label)
    if category is None:
        record.pop('category')
    else:
        record['category'] = category
    return record


class TestPartitionPairsByCategory:
    """Per-category buckets using build_pair_sets' own classification rule.

    A per-category cutoff has to be measured on the population the consumer
    can actually form. fetch_ann_neighbors pushes the querying record's own
    category into the Qdrant query as a payload filter, so a cross-category
    ANN pair is structurally impossible — including such pairs would
    calibrate a threshold against pairs it will never be applied to.
    """

    def test_returns_the_three_classes_per_category(self) -> None:
        got = _mod().partition_pairs_by_category([
            _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
            _cat_rec('d1', 'c1', 'duplicate', 'procedural_knowledge'),
        ])
        assert 'procedural_knowledge' in got['by_category']
        for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs'):
            assert key in got['by_category']['procedural_knowledge'], f'missing {key!r}'

    def test_classification_matches_build_pair_sets_within_a_category(self) -> None:
        """Same rule, applied per bucket — not a second, divergent one."""
        records = [
            _cat_rec('c1', 'c1', 'canonical', 'preferences_and_norms'),
            _cat_rec('d1', 'c1', 'duplicate', 'preferences_and_norms'),
            _cat_rec('x1', 'c1', 'distinct', 'preferences_and_norms'),
            _cat_rec('c2', 'c2', 'canonical', 'preferences_and_norms'),
            _cat_rec('p1', 'c2', 'pseudo_contradiction', 'preferences_and_norms'),
        ]
        bucket = _mod().partition_pairs_by_category(records)['by_category'][
            'preferences_and_norms'
        ]
        pooled = _mod().build_pair_sets(records)
        for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs'):
            assert _keys(bucket[key]) == _keys(pooled[key]), key

    def test_a_cross_category_pair_lands_in_no_bucket(self) -> None:
        """fetch_ann_neighbors filters by category, so this pair cannot form."""
        got = _mod().partition_pairs_by_category([
            _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
            _cat_rec('o1', 'c1', 'duplicate', 'observations_and_summaries'),
        ])
        for bucket in got['by_category'].values():
            for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs'):
                assert _keys(bucket[key]) == set(), (
                    'a cross-category pair must reach no bucket'
                )
        assert got['cross_category_dropped'] == 1, (
            'the exclusion must be a disclosed number, not invisible attrition'
        )

    def test_a_mixed_cluster_keeps_only_its_same_category_pairs(self) -> None:
        """2 of the 20 committed clusters are category-mixed."""
        got = _mod().partition_pairs_by_category([
            _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
            _cat_rec('d1', 'c1', 'duplicate', 'procedural_knowledge'),
            _cat_rec('o1', 'c1', 'duplicate', 'observations_and_summaries'),
        ])
        assert _keys(got['by_category']['procedural_knowledge']['true_dup_pairs']) == {
            ('c1', 'd1'),
        }
        assert got['cross_category_dropped'] == 2
        obs = got['by_category']['observations_and_summaries']
        assert all(len(obs[k]) == 0 for k in obs), 'a lone record forms no pair'

    def test_a_record_with_no_or_empty_category_contributes_no_pairs(self) -> None:
        for missing in (None, ''):
            got = _mod().partition_pairs_by_category([
                _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
                _cat_rec('d1', 'c1', 'duplicate', 'procedural_knowledge'),
                _cat_rec('u1', 'c1', 'duplicate', missing),
            ])
            formed = {
                tuple(sorted((p['a'], p['b'])))
                for bucket in got['by_category'].values()
                for key in bucket for p in bucket[key]
            }
            assert formed == {('c1', 'd1')}, f'category={missing!r} formed {formed}'
            assert got['cross_category_dropped'] == 2

    def test_the_partition_is_total_and_loses_nothing_silently(self) -> None:
        records = [
            _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
            _cat_rec('d1', 'c1', 'duplicate', 'procedural_knowledge'),
            _cat_rec('x1', 'c1', 'distinct', 'procedural_knowledge'),
            _cat_rec('c2', 'c2', 'canonical', 'observations_and_summaries'),
            _cat_rec('p1', 'c2', 'pseudo_contradiction', 'observations_and_summaries'),
            _cat_rec('u1', 'c3', 'duplicate', None),
        ]
        got = _mod().partition_pairs_by_category(records)
        bucketed = sum(
            len(bucket[key])
            for bucket in got['by_category'].values()
            for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs')
        )
        pooled = sum(len(v) for v in _mod().build_pair_sets(records).values())
        assert bucketed + got['cross_category_dropped'] == pooled, (
            'buckets plus disclosed drops must equal the pooled pair total'
        )

    def test_every_category_present_in_the_input_appears_in_the_output(self) -> None:
        """Even a single-record category — absence would read as 'not measured'."""
        got = _mod().partition_pairs_by_category([
            _cat_rec('c1', 'c1', 'canonical', 'procedural_knowledge'),
            _cat_rec('d1', 'c1', 'duplicate', 'procedural_knowledge'),
            _cat_rec('o1', 'o1', 'canonical', 'observations_and_summaries'),
        ])
        assert set(got['by_category']) == {
            'procedural_knowledge', 'observations_and_summaries',
        }

    def test_empty_input_yields_no_categories_and_no_drops(self) -> None:
        got = _mod().partition_pairs_by_category([])
        assert got['by_category'] == {}
        assert got['cross_category_dropped'] == 0


class TestCommittedFixtureIsDerivablePerCategory:
    """Pin the per-category derivability facts this task turns on.

    Measured from the committed fixture, not assumed. Pinning them makes a
    future fixture edit force a re-measurement rather than silently
    invalidating the recorded per-category calibration.
    """

    EXPECTED = {
        'procedural_knowledge': (242, 3316, 12),
        'observations_and_summaries': (4, 45, 6),
        'preferences_and_norms': (28, 0, 0),
    }

    def test_measured_pair_counts_per_category(self, records: list[dict]) -> None:
        got = _mod().partition_pairs_by_category(records)['by_category']
        assert set(got) == set(self.EXPECTED)
        for category, (dup, unrelated, hard) in self.EXPECTED.items():
            bucket = got[category]
            assert (
                len(bucket['true_dup_pairs']),
                len(bucket['unrelated_pairs']),
                len(bucket['hard_negative_pairs']),
            ) == (dup, unrelated, hard), category

    def test_preferences_and_norms_has_zero_negatives(self, records: list[dict]) -> None:
        """All 8 records sit in ONE cluster, so no cutoff is derivable.

        The refusal is itself the calibration finding — this is the fact
        that makes the task's second acceptance disjunct unreachable.
        """
        clusters = {
            r['cluster_id'] for r in records
            if r.get('category') == 'preferences_and_norms'
        }
        assert len(clusters) == 1
        bucket = _mod().partition_pairs_by_category(records)['by_category'][
            'preferences_and_norms'
        ]
        assert len(bucket['unrelated_pairs']) == 0
        assert len(bucket['hard_negative_pairs']) == 0

    def test_cross_category_pairs_are_dropped_and_counted(self, records: list[dict]) -> None:
        got = _mod().partition_pairs_by_category(records)
        assert got['cross_category_dropped'] == 1703
        pooled = sum(len(v) for v in _mod().build_pair_sets(records).values())
        bucketed = sum(
            len(bucket[key])
            for bucket in got['by_category'].values()
            for key in ('true_dup_pairs', 'unrelated_pairs', 'hard_negative_pairs')
        )
        assert bucketed + got['cross_category_dropped'] == pooled


# ---------------------------------------------------------------------------
# cosine_similarity / summarize_distribution
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Assertions are mathematical identities over injected vectors.

    Nothing here calls an embedder, so the suite needs no OPENAI_API_KEY.
    """

    def test_identical_vectors_score_one(self) -> None:
        v = [0.3, -0.7, 1.2, 0.0]
        assert _mod().cosine_similarity(v, list(v)) == pytest.approx(1.0, abs=1e-12)

    def test_orthogonal_unit_vectors_score_zero(self) -> None:
        assert _mod().cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-12)

    def test_antiparallel_vectors_score_minus_one(self) -> None:
        assert _mod().cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0, abs=1e-12)

    def test_is_symmetric_in_its_arguments(self) -> None:
        a, b = [0.1, 0.9, -0.4], [0.5, -0.2, 0.8]
        assert _mod().cosine_similarity(a, b) == pytest.approx(_mod().cosine_similarity(b, a), abs=1e-12)

    def test_magnitude_does_not_affect_the_score(self) -> None:
        a, b = [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        assert _mod().cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(
        ('a', 'b'),
        [([0.0, 0.0], [1.0, 0.0]), ([1.0, 0.0], [0.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])],
    )
    def test_zero_norm_raises_rather_than_returning_nan(self, a: list, b: list) -> None:
        """Loud over silent: a NaN would propagate into the distributions and
        quietly corrupt every derived statistic."""
        with pytest.raises(ValueError):
            _mod().cosine_similarity(a, b)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _mod().cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestSummarizeDistribution:
    SAMPLE = [round(i / 10, 1) for i in range(11)]  # 0.0 .. 1.0 in fixed steps

    FIELDS = ('n', 'min', 'max', 'mean', 'median', 'p05', 'p25', 'p75', 'p95')

    def test_reports_every_field(self) -> None:
        got = _mod().summarize_distribution(self.SAMPLE)
        for field in self.FIELDS:
            assert field in got, f'missing statistic {field!r}'

    def test_n_equals_the_input_length(self) -> None:
        assert _mod().summarize_distribution(self.SAMPLE)['n'] == len(self.SAMPLE)

    def test_min_max_and_median_are_the_expected_order_statistics(self) -> None:
        got = _mod().summarize_distribution(self.SAMPLE)
        assert got['min'] == pytest.approx(0.0)
        assert got['max'] == pytest.approx(1.0)
        assert got['median'] == pytest.approx(0.5)
        assert got['mean'] == pytest.approx(0.5)

    def test_quantiles_are_measured_values_never_interpolated(self) -> None:
        """Every reported quantile must be a value actually observed.

        derive_bands picks T_high from these, so an interpolated quantile
        would be a threshold that no measurement supports.
        """
        got = _mod().summarize_distribution(self.SAMPLE)
        for field in ('min', 'max', 'median', 'p05', 'p25', 'p75', 'p95'):
            assert got[field] in self.SAMPLE, (
                f'{field}={got[field]} is not a member of the measured sample'
            )

    def test_quantiles_are_ordered(self) -> None:
        got = _mod().summarize_distribution(self.SAMPLE)
        ordered = [got[f] for f in ('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max')]
        assert ordered == sorted(ordered), f'quantiles out of order: {ordered}'

    def test_input_order_does_not_matter(self) -> None:
        shuffled = list(reversed(self.SAMPLE))
        assert _mod().summarize_distribution(shuffled) == _mod().summarize_distribution(self.SAMPLE)

    def test_single_value_sample(self) -> None:
        got = _mod().summarize_distribution([0.42])
        assert got['n'] == 1
        for field in ('min', 'max', 'median', 'p05', 'p95'):
            assert got[field] == pytest.approx(0.42)

    def test_empty_sample_reports_none_not_zero(self) -> None:
        """An empty class must never be mistakable for a measured zero."""
        got = _mod().summarize_distribution([])
        assert got['n'] == 0
        for field in self.FIELDS:
            if field == 'n':
                continue
            assert got[field] is None, f'{field} must be None for an empty sample, got {got[field]!r}'


# ---------------------------------------------------------------------------
# derive_bands
# ---------------------------------------------------------------------------

class TestDeriveBands:
    """Every assertion is relational or derived from the injected sample.

    Not one pins a constant — that is the whole point of the leaf.
    """

    # A realistic sample: the negative class overlaps the duplicate class's
    # lower tail, so the bands are non-trivially separated.
    DUPS = [0.5, 0.6, 0.7, 0.8, 0.9]
    NEGS = [0.1, 0.2, 0.55]

    def test_t_high_strictly_exceeds_every_measured_negative(self) -> None:
        """The deterministic band must admit ZERO measured false positives."""
        t_high, _, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        assert t_high is not None
        assert t_high > max(self.NEGS), (
            f't_high={t_high} does not clear the highest measured negative {max(self.NEGS)}'
        )

    def test_t_high_is_a_measured_order_statistic_of_the_duplicate_class(self) -> None:
        """Traceable to data, never interpolated out of thin air."""
        t_high, _, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        assert t_high in self.DUPS, f't_high={t_high} is not a value observed in dup_scores'

    def test_t_high_is_the_smallest_such_order_statistic(self) -> None:
        """Picking a higher one would needlessly shrink the deterministic band."""
        t_high, _, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        smaller = [s for s in self.DUPS if s > max(self.NEGS) and s < t_high]
        assert not smaller, f'a smaller separating dup score exists: {smaller}'

    def test_t_low_comes_from_the_duplicate_lower_tail(self) -> None:
        _, t_low, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        assert t_low is not None
        assert t_low in self.DUPS, f't_low={t_low} is not a value observed in dup_scores'
        assert t_low <= _mod().summarize_distribution(self.DUPS)['median'], (
            't_low must come from the lower tail of the duplicate distribution'
        )

    def test_t_low_is_strictly_below_t_high(self) -> None:
        t_high, t_low, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        assert t_low < t_high, f'expected t_low < t_high, got {t_low} !< {t_high}'

    def test_both_bands_are_within_the_cosine_unit_range(self) -> None:
        t_high, t_low, _ = _mod().derive_bands(self.DUPS, self.NEGS)
        for name, value in (('t_high', t_high), ('t_low', t_low)):
            assert 0.0 <= value <= 1.0, f'{name}={value} outside [0.0, 1.0]'

    # -- refusal paths: what stops an uncalibrated number reaching config ---

    @pytest.mark.parametrize(
        ('dups', 'negs'),
        [([], [0.1, 0.2]), ([0.8, 0.9], []), ([], [])],
    )
    def test_an_empty_class_refuses_with_a_reason(self, dups: list, negs: list) -> None:
        t_high, t_low, reason = _mod().derive_bands(dups, negs)
        assert t_high is None and t_low is None, (
            f'an empty class must yield no thresholds, got t_high={t_high} t_low={t_low}'
        )
        assert isinstance(reason, str) and reason.strip(), (
            'a refusal must carry a machine-readable reason'
        )

    def test_fully_overlapping_distributions_refuse_rather_than_inventing_a_threshold(
        self,
    ) -> None:
        """max(negative) >= max(duplicate): no measured value separates them.

        The honest output is no threshold plus a reason. Interpolating one
        here would produce a number that looks calibrated and is not — the
        exact failure this leaf exists to prevent.
        """
        t_high, _, reason = _mod().derive_bands([0.4, 0.5, 0.6], [0.1, 0.6, 0.95])
        assert t_high is None, f'expected refusal, got t_high={t_high}'
        assert isinstance(reason, str) and reason.strip()

    def test_the_refusal_reason_is_machine_readable(self) -> None:
        """Distinguishable causes, so a caller can branch without prose matching."""
        _, _, empty_reason = _mod().derive_bands([], [0.1])
        _, _, overlap_reason = _mod().derive_bands([0.4, 0.5], [0.6, 0.7])
        assert empty_reason != overlap_reason, (
            'an empty class and a non-separable overlap are different findings '
            'and must not share one reason code'
        )

    def test_perfect_separation_yields_no_judge_band_rather_than_a_fake_one(self) -> None:
        """Every duplicate already clears every negative.

        t_high is then the duplicate class's own minimum, so no measured
        value can sit strictly below it — the judge band is not derivable
        from this sample, and saying so beats inventing a floor.
        """
        t_high, t_low, reason = _mod().derive_bands([0.8, 0.85, 0.9], [0.1, 0.2])
        assert t_high == 0.8
        assert t_low is None, f'expected no derivable judge band, got t_low={t_low}'
        assert isinstance(reason, str) and reason.strip()


# ---------------------------------------------------------------------------
# derive_bands_per_category
# ---------------------------------------------------------------------------

def _classes(dups: list, unrelated: list, hard: list) -> dict:
    """One category's three measured score samples, in PAIR_CLASSES keys."""
    return {'true_dup': dups, 'unrelated': unrelated, 'hard_negative': hard}


class TestDeriveBandsPerCategory:
    """Per-category bands: the SAME derivation, applied per bucket.

    Nothing here re-derives a threshold. Each category delegates to
    derive_bands, so a per-category cutoff is exactly as evidence-bound as
    the pooled one — and refuses on the same terms.

    Both refusal paths are reachable from the REAL corpus, not hypothetical:
    preferences_and_norms has zero negative pairs (one cluster), and
    observations_and_summaries is thin enough that non-separability is a
    live outcome. The machinery handles both without predicting either.
    """

    POOLED_T_HIGH = 0.75
    POOLED_T_LOW = 0.5

    def _derive(self, by_category: dict, pooled_t_high=POOLED_T_HIGH) -> dict:
        return _mod().derive_bands_per_category(
            by_category, pooled_t_high, self.POOLED_T_LOW,
        )

    def test_a_populated_category_yields_exactly_what_derive_bands_yields(self) -> None:
        """No second derivation path — delegation, so the two cannot drift."""
        dups, unrelated, hard = [0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2], [0.55]
        got = self._derive({'procedural_knowledge': _classes(dups, unrelated, hard)})
        expected = _mod().derive_bands(dups, unrelated + hard)
        entry = got['procedural_knowledge']
        assert (entry['t_high'], entry['t_low'], entry['reason']) == expected

    def test_the_negative_class_is_unrelated_plus_hard_negative(self) -> None:
        """Same pooling of negatives run_calibration already uses."""
        got = self._derive({'c': _classes([0.5, 0.9], [0.1], [0.95])})
        assert got['c']['t_high'] is None, (
            'a hard negative above every duplicate must block derivation; '
            'ignoring the hard-negative class would fabricate a cutoff'
        )

    def test_a_category_with_zero_negatives_refuses_with_empty_class(self) -> None:
        """The measured preferences_and_norms case: one cluster, no negatives.

        No arithmetic produces a cutoff here. The refusal IS the calibration
        finding, and it must be machine-readable so the consumer can branch.
        """
        got = self._derive({'preferences_and_norms': _classes([0.8, 0.9, 0.95], [], [])})
        entry = got['preferences_and_norms']
        assert entry['t_high'] is None and entry['t_low'] is None
        assert entry['reason'].startswith(_mod().REASON_EMPTY_CLASS), entry['reason']

    def test_a_non_separable_category_refuses_with_not_separable(self) -> None:
        """The reviewer's observations_and_summaries hypothesis, if it fires.

        Formulaic recaps scoring at or above every confirmed duplicate means
        no measured value separates the classes. Interpolating one would
        manufacture the very unevidenced cutoff this task exists to remove.
        """
        got = self._derive({
            'observations_and_summaries': _classes([0.80, 0.85], [0.90], [0.88]),
        })
        entry = got['observations_and_summaries']
        assert entry['t_high'] is None
        assert entry['reason'].startswith(_mod().REASON_NOT_SEPARABLE), entry['reason']

    def test_an_uncalibrated_category_is_present_with_a_null_t_high_never_omitted(
        self,
    ) -> None:
        """Omission would read as 'not measured' rather than 'measured, refused'.

        That ambiguity is exactly what this task exists to remove, so the
        report keeps every category and distinguishes the two by reason.
        """
        got = self._derive({
            'procedural_knowledge': _classes([0.5, 0.6, 0.7, 0.8, 0.9], [0.1], [0.55]),
            'preferences_and_norms': _classes([0.8, 0.9], [], []),
            'observations_and_summaries': _classes([0.4], [0.9], []),
        })
        assert set(got) == {
            'procedural_knowledge', 'preferences_and_norms', 'observations_and_summaries',
        }
        assert got['procedural_knowledge']['t_high'] is not None
        for category, entry in got.items():
            if entry['t_high'] is None:
                assert (entry['reason'] or '').strip(), (
                    f'{category}: a null t_high must never be silent — it carries the '
                    'reason code that distinguishes "refused" from "not measured"'
                )
            if entry['reason']:
                # derive_bands also reasons on a SUCCESS (REASON_NO_JUDGE_BAND:
                # a derived t_high with no derivable judge band), so a reason
                # does not imply refusal — only that a caller can branch on it.
                assert entry['reason'].split(':')[0] in {
                    _mod().REASON_EMPTY_CLASS,
                    _mod().REASON_NOT_SEPARABLE,
                    _mod().REASON_NO_JUDGE_BAND,
                }, f'{category}: unbranchable reason {entry["reason"]!r}'

    def test_every_entry_carries_its_measured_distributions_and_pair_counts(self) -> None:
        """The evidence that justifies the number — or the refusal."""
        got = self._derive({'c': _classes([0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2], [0.55])})
        entry = got['c']
        assert set(entry['distributions']) == set(_mod().PAIR_CLASSES)
        assert entry['pair_counts'] == {'true_dup': 5, 'unrelated': 2, 'hard_negative': 1}
        assert entry['distributions']['true_dup'] == _mod().summarize_distribution(
            [0.5, 0.6, 0.7, 0.8, 0.9],
        )

    def test_an_empty_class_reports_n_zero_not_a_measured_zero(self) -> None:
        got = self._derive({'preferences_and_norms': _classes([0.8, 0.9], [], [])})
        entry = got['preferences_and_norms']
        assert entry['pair_counts']['unrelated'] == 0
        assert entry['distributions']['unrelated']['max'] is None, (
            'an empty pair class must not report a measured 0.0'
        )

    # -- the second acceptance disjunct's direct evidence --------------------

    def test_each_entry_counts_its_negatives_admitted_by_the_POOLED_t_high(self) -> None:
        """Does the pooled cutoff separate THIS category's classes?

        A non-zero count is a measured false-positive rate for the pooled
        t_high in that category — the direct evidence for (or against) the
        claim that one cutoff serves all three.
        """
        got = self._derive({
            'observations_and_summaries': _classes([0.6], [0.80, 0.90], [0.99, 0.10]),
        })
        assert got['observations_and_summaries']['pooled_t_high_negatives_admitted'] == 3

    def test_negatives_admitted_counts_only_negatives_never_true_duplicates(self) -> None:
        """A duplicate in the deterministic band is the band working."""
        got = self._derive({'c': _classes([0.95, 0.96], [0.10], [0.20])})
        assert got['c']['pooled_t_high_negatives_admitted'] == 0

    def test_a_category_with_no_negatives_admits_none_of_them(self) -> None:
        got = self._derive({'preferences_and_norms': _classes([0.9], [], [])})
        assert got['preferences_and_norms']['pooled_t_high_negatives_admitted'] == 0

    def test_an_uncalibrated_pooled_t_high_admits_no_measurement_not_zero(self) -> None:
        """With no pooled band there is nothing to admit against.

        0 would read as 'measured, and safe' — the same distinction
        build_report already draws for its false-positive tally.
        """
        got = self._derive({'c': _classes([0.9], [0.8], [])}, pooled_t_high=None)
        assert got['c']['pooled_t_high_negatives_admitted'] is None

    def test_empty_input_yields_no_entries(self) -> None:
        assert self._derive({}) == {}


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

def _retrieval(mid: str, canonical: str, candidates: list[str], *, present: bool = True) -> dict:
    """One duplicate's ground-truth canonical + the ranked ids search returned.

    ``canonical_present`` is resolved by a direct lookup at the live edge,
    NOT by whether search happened to return it — that distinction is what
    separates a corpus gap from a retrieval failure.
    """
    return {
        'memory_id': mid,
        'canonical_id': canonical,
        'canonical_present': present,
        'candidates': candidates,
    }


def _recall_by_k(result: dict) -> dict[int, float | None]:
    return {row['k']: row['recall'] for row in result['per_k']}


class TestComputeRecallAtK:
    """Retrievals are injected, so no live Qdrant or embedder is needed."""

    RETRIEVALS = [
        _retrieval('d1', 'c1', ['c1', 'z', 'y']),            # rank 1
        _retrieval('d2', 'c2', ['z', 'y', 'x', 'w', 'c2']),  # rank 5
        _retrieval('d3', 'c3', ['z', 'y']),                  # never returned
    ]

    def test_reports_a_row_per_requested_k(self) -> None:
        got = _mod().compute_recall_at_k(self.RETRIEVALS, [1, 5, 10])
        assert [row['k'] for row in got['per_k']] == [1, 5, 10]

    def test_hits_and_total_are_reported_so_the_denominator_is_auditable(self) -> None:
        got = _mod().compute_recall_at_k(self.RETRIEVALS, [1, 5])
        for row in got['per_k']:
            assert 'hits' in row and 'total' in row, f'row {row} hides its counts'
            assert row['total'] == 3, 'all three canonicals are present in the corpus'
            assert row['recall'] == pytest.approx(row['hits'] / row['total'])

    def test_a_canonical_at_rank_one_counts_at_every_k(self) -> None:
        got = _recall_by_k(_mod().compute_recall_at_k([self.RETRIEVALS[0]], [1, 5, 10]))
        assert got == {1: pytest.approx(1.0), 5: pytest.approx(1.0), 10: pytest.approx(1.0)}

    def test_a_canonical_below_k_does_not_count(self) -> None:
        got = _recall_by_k(_mod().compute_recall_at_k([self.RETRIEVALS[1]], [1, 5]))
        assert got[1] == pytest.approx(0.0), 'rank 5 must not count at k=1'
        assert got[5] == pytest.approx(1.0), 'rank 5 must count at k=5'

    def test_recall_is_monotonically_non_decreasing_in_k(self) -> None:
        """A structural property of the measure, not a target number."""
        got = _mod().compute_recall_at_k(self.RETRIEVALS, [1, 3, 5, 10])
        recalls = [row['recall'] for row in got['per_k']]
        assert recalls == sorted(recalls), f'recall@k must not decrease as k grows: {recalls}'

    def test_an_absent_canonical_is_excluded_from_the_denominator_not_scored_a_miss(
        self,
    ) -> None:
        """The duplicates were deleted, so an absent canonical is a CORPUS GAP.

        Scoring it as a retrieval miss would understate recall and could
        push T_low lower than the evidence supports.
        """
        retrievals = [
            _retrieval('d1', 'c1', ['c1']),
            _retrieval('d2', 'gone', ['z', 'y'], present=False),
        ]
        got = _mod().compute_recall_at_k(retrievals, [1])
        row = got['per_k'][0]
        assert row['total'] == 1, f'the absent canonical must leave the denominator: {row}'
        assert row['hits'] == 1
        assert row['recall'] == pytest.approx(1.0)

    def test_absent_canonicals_are_reported_separately(self) -> None:
        retrievals = [
            _retrieval('d1', 'c1', ['c1']),
            _retrieval('d2', 'gone', ['z'], present=False),
        ]
        got = _mod().compute_recall_at_k(retrievals, [1])
        assert 'canonical_absent' in got, 'a corpus gap must be visible in the report'
        reported = json.dumps(got['canonical_absent'])
        assert 'd2' in reported or 'gone' in reported, (
            f'the absent-canonical record must be identifiable: {got["canonical_absent"]}'
        )
        assert len(got['canonical_absent']) == 1

    def test_empty_input_reports_none_ratios_not_zero(self) -> None:
        """No measurement is not the same as a measured zero."""
        got = _mod().compute_recall_at_k([], [1, 5])
        for row in got['per_k']:
            assert row['total'] == 0
            assert row['recall'] is None, f'expected None recall for an empty sample, got {row}'

    def test_all_canonicals_absent_reports_none_ratios(self) -> None:
        got = _mod().compute_recall_at_k(
            [_retrieval('d1', 'gone', ['z'], present=False)], [1],
        )
        row = got['per_k'][0]
        assert row['total'] == 0
        assert row['recall'] is None
        assert len(got['canonical_absent']) == 1


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

PAIR_CLASSES = ('true_dup', 'unrelated', 'hard_negative')

_PROVENANCE = {
    'fixture_path': 'tests/fixtures/write_triage_calibration.jsonl',
    'record_count': 6,
    'cluster_count': 2,
    'embedder_model': 'text-embedding-3-small',
    'embedder_dimensions': 1536,
}

# Cleanly separated: every negative sits below every duplicate.
CLEAN_SCORES = {
    'true_dup': [0.70, 0.80, 0.90],
    'unrelated': [0.10, 0.20, 0.30],
    'hard_negative': [0.40, 0.50],
}
# Deliberately overlapping: two negatives sit at/above the duplicate lower tail.
OVERLAP_SCORES = {
    'true_dup': [0.70, 0.80, 0.90],
    'unrelated': [0.10, 0.85],
    'hard_negative': [0.40, 0.95],
}


def _report(scores: dict, t_high, t_low, reason=None, recall=None, **kwargs) -> dict:
    return _mod().build_report(
        scores_by_class=scores,
        t_high=t_high,
        t_low=t_low,
        reason=reason,
        recall=recall if recall is not None else {'per_k': [], 'canonical_absent': []},
        provenance=dict(_PROVENANCE),
        **kwargs,
    )


def _per_category(pooled_t_high=0.80, pooled_t_low=0.70) -> dict:
    """A realistic per-category block: one calibrated, two refused.

    Mirrors the shape the committed fixture actually produces — a category
    with enough evidence to derive a cutoff, one with zero negatives, and one
    too thin to separate.
    """
    return _mod().derive_bands_per_category(
        {
            'procedural_knowledge': {
                'true_dup': [0.70, 0.80, 0.90], 'unrelated': [0.10], 'hard_negative': [0.55],
            },
            'preferences_and_norms': {
                'true_dup': [0.80, 0.90], 'unrelated': [], 'hard_negative': [],
            },
            'observations_and_summaries': {
                'true_dup': [0.60], 'unrelated': [0.95], 'hard_negative': [],
            },
        },
        pooled_t_high,
        pooled_t_low,
    )


class TestBuildReport:
    def test_carries_a_distribution_summary_per_pair_class(self) -> None:
        got = _report(CLEAN_SCORES, 0.70, 0.60)
        for name in PAIR_CLASSES:
            assert name in got['distributions'], f'missing distribution for {name}'
            assert got['distributions'][name]['n'] == len(CLEAN_SCORES[name])

    def test_echoes_the_chosen_thresholds(self) -> None:
        got = _report(CLEAN_SCORES, 0.70, 0.60)
        assert got['chosen_t_high'] == pytest.approx(0.70)
        assert got['chosen_t_low'] == pytest.approx(0.60)

    def test_carries_the_recall_block(self) -> None:
        recall = {'per_k': [{'k': 1, 'hits': 2, 'total': 3, 'recall': 2 / 3}],
                  'canonical_absent': []}
        got = _report(CLEAN_SCORES, 0.70, 0.60, recall=recall)
        assert got['recall_at_k'] == recall

    def test_carries_run_provenance(self) -> None:
        got = _report(CLEAN_SCORES, 0.70, 0.60)
        prov = got['provenance']
        for key in ('fixture_path', 'record_count', 'cluster_count',
                    'embedder_model', 'embedder_dimensions'):
            assert key in prov, f'provenance missing {key!r}'
        assert prov['pair_counts'] == {name: len(CLEAN_SCORES[name]) for name in PAIR_CLASSES}, (
            'per-class pair counts must be recorded so the report is self-describing'
        )

    def test_counts_every_pair_class_across_the_three_bands(self) -> None:
        got = _report(CLEAN_SCORES, 0.70, 0.60)
        for name in PAIR_CLASSES:
            bands = got['per_band'][name]
            for band in ('deterministic', 'judge', 'store'):
                assert band in bands, f'{name} missing band count {band!r}'

    def test_band_counts_sum_to_each_class_n(self) -> None:
        """Accounting invariant: no pair is silently dropped from the report."""
        for scores, t_high, t_low in (
            (CLEAN_SCORES, 0.70, 0.60), (OVERLAP_SCORES, 0.90, 0.70),
        ):
            got = _report(scores, t_high, t_low)
            for name in PAIR_CLASSES:
                bands = got['per_band'][name]
                total = bands['deterministic'] + bands['judge'] + bands['store']
                assert total == len(scores[name]), (
                    f'{name}: band counts {bands} sum to {total}, expected {len(scores[name])}'
                )

    def test_bands_are_assigned_at_the_documented_boundaries(self) -> None:
        """s >= t_high deterministic; t_low <= s < t_high judge; s < t_low store."""
        got = _report({'true_dup': [0.60, 0.70, 0.80], 'unrelated': [], 'hard_negative': []},
                      0.70, 0.60)
        bands = got['per_band']['true_dup']
        assert bands['deterministic'] == 2, 't_high is inclusive (0.70 and 0.80)'
        assert bands['judge'] == 1, 't_low is inclusive (0.60)'
        assert bands['store'] == 0

    def test_deterministic_band_false_positives_is_zero_for_a_clean_separation(self) -> None:
        got = _report(CLEAN_SCORES, 0.70, 0.60)
        assert got['deterministic_band_false_positives'] == 0

    def test_deterministic_band_false_positives_counts_negatives_at_or_above_t_high(self) -> None:
        """The task's headline risk figure: unrelated PLUS hard_negative pairs
        that the deterministic band would restate without asking a judge."""
        got = _report(OVERLAP_SCORES, 0.80, 0.70)
        # unrelated 0.85 and hard_negative 0.95 both clear t_high=0.80.
        assert got['deterministic_band_false_positives'] == 2

    def test_false_positives_equal_the_negative_classes_deterministic_counts(self) -> None:
        got = _report(OVERLAP_SCORES, 0.80, 0.70)
        expected = (got['per_band']['unrelated']['deterministic']
                    + got['per_band']['hard_negative']['deterministic'])
        assert got['deterministic_band_false_positives'] == expected
        # True duplicates DO reach the deterministic band — that is the band
        # working. They must never be tallied as false positives.
        assert got['per_band']['true_dup']['deterministic'] > 0, 'sample sanity'
        assert got['deterministic_band_false_positives'] < (
            expected + got['per_band']['true_dup']['deterministic']
        )

    def test_an_uncalibrated_run_still_emits_the_distributions(self) -> None:
        """t_high=None must not raise: the refusal IS the finding, and the
        measured distributions are what justify it."""
        got = _report(CLEAN_SCORES, None, None, reason='not_separable: ...')
        assert got['chosen_t_high'] is None
        assert got['chosen_t_low'] is None
        assert got['reason'] == 'not_separable: ...'
        for name in PAIR_CLASSES:
            assert got['distributions'][name]['n'] == len(CLEAN_SCORES[name])

    def test_an_uncalibrated_run_reports_no_band_counts_rather_than_zeroes(self) -> None:
        got = _report(CLEAN_SCORES, None, None, reason='not_separable: ...')
        assert got['deterministic_band_false_positives'] is None, (
            'with no t_high there is no deterministic band; 0 would read as '
            '"measured, and safe"'
        )

    def test_a_missing_judge_band_still_accounts_for_every_pair(self) -> None:
        """Perfect separation: t_high derived, t_low None."""
        got = _report(CLEAN_SCORES, 0.70, None, reason='no_judge_band: ...')
        for name in PAIR_CLASSES:
            bands = got['per_band'][name]
            assert bands['judge'] == 0
            total = bands['deterministic'] + bands['judge'] + bands['store']
            assert total == len(CLEAN_SCORES[name]), f'{name}: {bands}'

    def test_the_report_is_json_serializable(self) -> None:
        got = _report(OVERLAP_SCORES, 0.80, 0.70)
        assert json.loads(json.dumps(got)) == got


class TestBuildReportPerCategory:
    """The per-category section is ADDITIVE — the pooled report is untouched.

    The pooled t_high remains the calibration of record; this section is the
    evidence about whether it is warranted per category. Any drift in the
    pooled numbers would mean this task changed a measurement it was only
    supposed to explain, so that is asserted directly.
    """

    POOLED_KEYS = (
        'chosen_t_high', 'chosen_t_low', 'reason',
        'deterministic_band_false_positives', 'distributions', 'per_band',
        'recall_at_k',
    )

    def test_carries_a_per_category_section(self) -> None:
        per_category = _per_category()
        got = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=per_category)
        assert got['per_category'] == per_category

    def test_the_pooled_measurement_is_byte_identical_with_and_without_it(self) -> None:
        """Additive means additive: same inputs, same pooled numbers."""
        without = _report(OVERLAP_SCORES, 0.80, 0.70)
        with_ = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        for key in self.POOLED_KEYS:
            assert with_[key] == without[key], f'{key} drifted when per_category was added'
        for key, value in without['provenance'].items():
            assert with_['provenance'][key] == value, f'provenance[{key!r}] drifted'

    def test_defaults_to_an_empty_section_so_existing_callers_are_unaffected(self) -> None:
        """Present-and-empty, not missing: 'measured nothing' is still a state."""
        got = _report(OVERLAP_SCORES, 0.80, 0.70)
        assert got['per_category'] == {}

    def test_every_category_keeps_its_derived_entry_verbatim(self) -> None:
        got = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        assert set(got['per_category']) == {
            'procedural_knowledge', 'preferences_and_norms', 'observations_and_summaries',
        }
        for category, entry in got['per_category'].items():
            for key in ('distributions', 't_high', 't_low', 'reason', 'pair_counts',
                        'pooled_t_high_negatives_admitted'):
                assert key in entry, f'{category} missing {key!r}'

    def test_an_uncalibrated_category_is_reported_not_dropped(self) -> None:
        got = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        refused = got['per_category']['preferences_and_norms']
        assert refused['t_high'] is None
        assert refused['reason'].startswith(_mod().REASON_EMPTY_CLASS)

    def test_an_uncalibrated_POOLED_run_still_emits_the_section(self) -> None:
        """The pooled refusal does not erase the per-category evidence.

        Those distributions are part of what justifies the pooled refusal.
        """
        got = _report(
            CLEAN_SCORES, None, None, reason='not_separable: ...',
            per_category=_per_category(pooled_t_high=None, pooled_t_low=None),
        )
        assert set(got['per_category']) == {
            'procedural_knowledge', 'preferences_and_norms', 'observations_and_summaries',
        }
        assert got['per_category']['procedural_knowledge'][
            'pooled_t_high_negatives_admitted'
        ] is None, 'with no pooled band there is nothing to admit against'

    def test_provenance_records_the_per_category_pair_counts(self) -> None:
        """Self-describing, exactly as the pooled pair_counts already are."""
        per_category = _per_category()
        got = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=per_category)
        assert got['provenance']['per_category_pair_counts'] == {
            category: entry['pair_counts'] for category, entry in per_category.items()
        }

    def test_the_report_is_json_serializable_with_the_section(self) -> None:
        got = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        assert json.loads(json.dumps(got)) == got


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdownPerCategory:
    """The human-readable form of the finding.

    Assertions are on the numbers and the machine-readable reason codes, not
    on prose wording — a reader must be able to see, without parsing JSON,
    which categories are calibrated and what the pooled cutoff would admit
    in the ones that are not.
    """

    def _md(self, pooled_t_high=0.80, pooled_t_low=0.70) -> str:
        return _mod().render_markdown(_report(
            OVERLAP_SCORES, pooled_t_high, pooled_t_low,
            per_category=_per_category(pooled_t_high, pooled_t_low),
        ))

    def test_emits_one_row_per_category(self) -> None:
        md = self._md()
        for category in ('procedural_knowledge', 'preferences_and_norms',
                         'observations_and_summaries'):
            assert len([ln for ln in md.splitlines()
                        if ln.startswith(f'| {category} |')]) == 1, (
                f'expected exactly one table row for {category}'
            )

    def test_a_calibrated_row_shows_its_derived_t_high_and_n(self) -> None:
        report = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        entry = report['per_category']['procedural_knowledge']
        row = next(ln for ln in _mod().render_markdown(report).splitlines()
                   if ln.startswith('| procedural_knowledge |'))
        assert str(entry['t_high']) in row
        assert str(entry['pair_counts']['true_dup']) in row

    def test_a_refused_row_shows_its_reason_code_instead_of_a_number(self) -> None:
        row = next(ln for ln in self._md().splitlines()
                   if ln.startswith('| preferences_and_norms |'))
        assert _mod().REASON_EMPTY_CLASS in row, (
            'a refusal must be readable as a refusal, not as a blank cell'
        )

    def test_every_row_shows_what_the_POOLED_cutoff_would_admit(self) -> None:
        """The evidence for or against one cutoff serving every category."""
        report = _report(OVERLAP_SCORES, 0.80, 0.70, per_category=_per_category())
        md = _mod().render_markdown(report)
        entry = report['per_category']['observations_and_summaries']
        assert entry['pooled_t_high_negatives_admitted'] == 1, 'sample sanity'
        row = next(ln for ln in md.splitlines()
                   if ln.startswith('| observations_and_summaries |'))
        assert str(entry['pooled_t_high_negatives_admitted']) in row

    def test_a_report_with_no_per_category_section_still_renders(self) -> None:
        """Older artifacts predate the section; rendering must not crash."""
        assert _mod().render_markdown(_report(CLEAN_SCORES, 0.70, 0.60))


# ---------------------------------------------------------------------------
# write_triage_config_block
# ---------------------------------------------------------------------------

import yaml  # noqa: E402

BASE_YAML = """\
# Leading comment that must survive.
server:
  host: localhost  # inline comment
  port: 8080

reconciliation:
  # An explanatory comment operators rely on.
  stale_run_recovery_seconds: 900

curator:
  enabled: true
"""

WITH_BLOCK = """\
# Leading comment that must survive.
server:
  host: localhost

write_triage:
  # stale annotation
  t_high: 0.11
  t_low: 0.05
  calibration_report_path: old/report.json

curator:
  # A comment AFTER the write_triage block.
  enabled: true
"""

# config/config.yaml's actual convention: a run of COLUMN-0 comment lines,
# preceded by a blank separator, is the header for the section that FOLLOWS
# it -- see the 6-line block before `summary_rebuild:` at config.yaml:213-218,
# and the same shape before `taskmaster:`, `task_metadata:` and `task_status:`.
WITH_BLOCK_THEN_COLUMN0_HEADER = """\
# Leading comment that must survive.
server:
  host: localhost

write_triage:
  t_high: 0.11
  t_low: 0.05
  calibration_report_path: old/report.json

# Periodic entity-summary rebuild -- scheduled staleness backstop.
# Disabled by default; costs nothing until an operator opts in.
summary_rebuild:
  enabled: false
  interval_seconds: 3600
"""

WITH_BLOCK_AT_EOF = """\
# Leading comment that must survive.
server:
  host: localhost

write_triage:
  t_high: 0.11
  t_low: 0.05
  calibration_report_path: old/report.json
"""


def _without_write_triage_block(text: str) -> list[str]:
    """Every line NOT belonging to a ``write_triage:`` block.

    The block is the ``write_triage:`` line plus its INDENTED body; a blank
    or column-0 line ends it. What survives is exactly the surrounding
    config, so two such lists are comparable across a replacement.
    """
    kept: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.startswith('write_triage:'):
            in_block = True
            continue
        if in_block:
            if line[:1].isspace():
                continue
            in_block = False
        kept.append(line)
    return kept


def _call(text: str, t_high=0.87, t_low=0.61, path='calibration/r.json') -> str:
    return _mod().write_triage_config_block(text, t_high, t_low, path)


class TestWriteTriageConfigBlock:
    def test_appends_the_block_when_absent(self) -> None:
        got = yaml.safe_load(_call(BASE_YAML))['write_triage']
        assert got == {
            't_high': 0.87, 't_low': 0.61, 'calibration_report_path': 'calibration/r.json',
        }

    def test_preserves_every_other_line_byte_for_byte_when_appending(self) -> None:
        """config.yaml's explanatory comments are load-bearing for operators,
        which is why this is a surgical text edit and not a safe_dump
        round-trip (pyyaml is the only YAML dep; it would strip them)."""
        out_lines = _call(BASE_YAML).splitlines()
        in_block = False
        kept = []
        for line in out_lines:
            if line.startswith('write_triage:'):
                in_block = True
                continue
            if in_block:
                if line and not line[0].isspace() and not line.startswith('#'):
                    in_block = False
                else:
                    continue
            kept.append(line)
        assert [ln for ln in kept if ln.strip()] == [
            ln for ln in BASE_YAML.splitlines() if ln.strip()
        ]

    def test_replaces_only_the_existing_block(self) -> None:
        got = yaml.safe_load(_call(WITH_BLOCK))
        assert got['write_triage'] == {
            't_high': 0.87, 't_low': 0.61, 'calibration_report_path': 'calibration/r.json',
        }
        assert got['server'] == {'host': 'localhost'}

    def test_a_section_declared_after_the_block_survives(self) -> None:
        """Guards against a regex eating to end-of-file."""
        out = _call(WITH_BLOCK)
        assert 'curator:' in out
        assert '# A comment AFTER the write_triage block.' in out
        assert yaml.safe_load(out)['curator'] == {'enabled': True}

    def test_the_stale_block_body_is_gone(self) -> None:
        out = _call(WITH_BLOCK)
        assert '0.11' not in out and 'old/report.json' not in out
        assert '# stale annotation' not in out

    def test_comments_outside_the_block_survive_a_replacement(self) -> None:
        assert '# Leading comment that must survive.' in _call(WITH_BLOCK)

    def test_result_parses_and_round_trips(self) -> None:
        for text in (BASE_YAML, WITH_BLOCK):
            parsed = yaml.safe_load(_call(text, 0.5, 0.25, 'x/y.json'))
            assert parsed['write_triage']['t_high'] == pytest.approx(0.5)
            assert parsed['write_triage']['t_low'] == pytest.approx(0.25)
            assert parsed['write_triage']['calibration_report_path'] == 'x/y.json'

    @pytest.mark.parametrize(('t_high', 't_low'), [(None, 0.6), (0.9, None), (None, None)])
    def test_refuses_to_write_an_uncalibrated_threshold(self, t_high, t_low) -> None:
        """An uncalibrated run must never put a null threshold into config."""
        with pytest.raises(ValueError):
            _mod().write_triage_config_block(BASE_YAML, t_high, t_low, 'r.json')

    def test_a_refused_write_leaves_no_partial_block(self) -> None:
        with pytest.raises(ValueError):
            _mod().write_triage_config_block(WITH_BLOCK, None, None, 'r.json')
        assert 'write_triage' in WITH_BLOCK and '0.11' in WITH_BLOCK, 'input untouched'


class TestWriteTriageConfigBlockSpanScan:
    """Byte-exact regression guard on the span scan's boundaries.

    Every assertion here compares line LISTS, never ``yaml.safe_load`` and
    never an ``in`` substring check. Both of those are blind to the two
    defects this class exists to pin: a swallowed column-0 comment header
    parses fine and still contains every section, and a duplicated blank
    line is invisible to both. Only same-content-same-order-same-COUNT
    equality catches them.
    """

    def test_a_column0_comment_header_for_the_next_section_is_not_swallowed(self) -> None:
        """A run of column-0 comments introduces the FOLLOWING section.

        config.yaml uses this shape throughout, so treating it as
        block-internal deletes an operator-facing section header on every
        recalibration run.
        """
        out = _call(WITH_BLOCK_THEN_COLUMN0_HEADER)
        assert _without_write_triage_block(out) == _without_write_triage_block(
            WITH_BLOCK_THEN_COLUMN0_HEADER,
        )

    def test_the_indented_comment_shape_still_round_trips_byte_exact(self) -> None:
        """An indented comment genuinely belongs to the block it sits in.

        Pins the blank-line accounting too: one blank separator in must be
        exactly one blank separator out.
        """
        out = _call(WITH_BLOCK)
        assert _without_write_triage_block(out) == _without_write_triage_block(WITH_BLOCK)

    def test_a_block_at_end_of_file_is_unaffected(self) -> None:
        """Nothing follows the block, so the scan has nothing to reclaim."""
        out = _call(WITH_BLOCK_AT_EOF)
        assert _without_write_triage_block(out) == _without_write_triage_block(
            WITH_BLOCK_AT_EOF,
        )

    @pytest.mark.parametrize(
        'text',
        [WITH_BLOCK_THEN_COLUMN0_HEADER, WITH_BLOCK, WITH_BLOCK_AT_EOF],
        ids=['column0-header', 'indented-comment', 'at-eof'],
    )
    def test_the_block_is_actually_replaced(self, text: str) -> None:
        """Guards the preservation assertions above from passing vacuously:
        an implementation that returned its input unchanged would satisfy
        every byte-exact check while writing no threshold at all."""
        parsed = yaml.safe_load(_call(text))
        assert parsed['write_triage'] == {
            't_high': 0.87, 't_low': 0.61, 'calibration_report_path': 'calibration/r.json',
        }
        assert '0.11' not in _call(text), 'the stale body must be gone'


# ---------------------------------------------------------------------------
# run_calibration (end-to-end, injected edges)
# ---------------------------------------------------------------------------

def _e2e_records() -> list[dict]:
    return [
        _rec('c1', 'c1', 'canonical'), _rec('d1', 'c1', 'duplicate'),
        _rec('c2', 'c2', 'canonical'), _rec('d2', 'c2', 'duplicate'),
    ]


def _embed_fn(separable: bool = True):
    """Cluster-1 and cluster-2 vectors are near-orthogonal, so same-cluster
    pairs score high and cross-cluster pairs score low."""
    vectors = {
        'c1': [1.0, 0.0], 'd1': [0.99, 0.14],
        'c2': [0.0, 1.0], 'd2': [0.14, 0.99],
    }
    if not separable:
        vectors['d2'] = [0.99, 0.14]  # collapses the classes together
    calls: list[str] = []

    def embed(mid: str, content: str) -> list[float]:
        calls.append(mid)
        return vectors[mid]

    embed.calls = calls  # type: ignore[attr-defined]
    return embed


def _search_fn(hits: dict[str, list[str]] | None = None, present: set[str] | None = None):
    hits = hits if hits is not None else {'d1': ['c1'], 'd2': ['c2']}
    present = present if present is not None else {'c1', 'c2'}

    def search(record: dict, k: int) -> dict:
        return {
            'candidates': hits.get(record['memory_id'], []),
            'canonical_present': record['cluster_id'] in present,
        }

    return search


def _run(tmp_path: Path, records=None, embed=None, search=None, ks=(1, 5)):
    return _mod().run_calibration(
        records=records if records is not None else _e2e_records(),
        embed_fn=embed if embed is not None else _embed_fn(),
        search_fn=search if search is not None else _search_fn(),
        report_path=tmp_path / 'report.json',
        ks=list(ks),
        provenance={'fixture_path': 'x.jsonl', 'embedder_model': 'text-embedding-3-small',
                    'embedder_dimensions': 1536},
    )


class TestRunCalibration:
    def test_writes_the_json_report(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        written = json.loads((tmp_path / 'report.json').read_text())
        assert written == result['report'], 'the JSON on disk must be the report built'

    def test_writes_a_markdown_sibling(self, tmp_path: Path) -> None:
        _run(tmp_path)
        assert (tmp_path / 'report.md').exists(), 'a human-readable sibling must be written'

    def test_the_markdown_carries_the_band_table_and_false_positive_figure(
        self, tmp_path: Path,
    ) -> None:
        """Assert on the numbers and structure, not on prose wording."""
        result = _run(tmp_path)
        md = (tmp_path / 'report.md').read_text()
        for name in PAIR_CLASSES:
            assert name in md, f'the band table must cover {name}'
        fps = result['report']['deterministic_band_false_positives']
        assert str(fps) in md, 'the deterministic-band false-positive count must be readable'

    def test_embeds_each_distinct_record_exactly_once(self, tmp_path: Path) -> None:
        """Pair-wise embedding would multiply API cost by O(n)."""
        embed = _embed_fn()
        records = _e2e_records()
        _run(tmp_path, embed=embed)
        assert len(embed.calls) == len(records), (
            f'expected {len(records)} embed calls (one per record), got {len(embed.calls)}'
        )
        assert sorted(embed.calls) == sorted(r['memory_id'] for r in records)

    def test_the_returned_bands_match_derive_bands_over_the_measured_scores(
        self, tmp_path: Path,
    ) -> None:
        result = _run(tmp_path)
        scores = result['scores_by_class']
        negatives = list(scores['unrelated']) + list(scores['hard_negative'])
        expected = _mod().derive_bands(scores['true_dup'], negatives)
        assert (result['t_high'], result['t_low']) == (expected[0], expected[1])

    def test_an_embed_failure_propagates_rather_than_silently_shrinking_the_sample(
        self, tmp_path: Path,
    ) -> None:
        """A partial distribution would yield a thresholds-look-fine artifact
        computed on a subset."""
        def boom(mid: str, content: str) -> list[float]:
            if mid == 'd2':
                raise RuntimeError('embedding failed')
            return [1.0, 0.0]

        with pytest.raises(RuntimeError):
            _run(tmp_path, embed=boom)

    def test_a_search_failure_surfaces_rather_than_scoring_a_miss(
        self, tmp_path: Path,
    ) -> None:
        def boom(record: dict, k: int) -> dict:
            raise RuntimeError('search failed')

        with pytest.raises(RuntimeError):
            _run(tmp_path, search=boom)

    def test_a_refusal_still_writes_the_report(self, tmp_path: Path) -> None:
        result = _run(tmp_path, embed=_embed_fn(separable=False))
        assert (tmp_path / 'report.json').exists(), 'the refusal IS the finding — record it'
        assert result['report']['reason'], 'the refusal must carry its reason'

    def test_a_refusal_does_not_attempt_the_config_write(self, tmp_path: Path) -> None:
        result = _run(tmp_path, embed=_embed_fn(separable=False))
        assert result['t_high'] is None or result['t_low'] is None
        assert result.get('config_written') is not True, (
            'an uncalibrated run must never reach the config write'
        )


def _mixed_category_records() -> list[dict]:
    """Two categories, one of them sharing a cluster with the other.

    The cross-category pair is the one fetch_ann_neighbors can never form.
    """
    records = _e2e_records()
    records.append(_cat_rec('o1', 'c1', 'duplicate', 'observations_and_summaries'))
    return records


def _mixed_embed():
    vectors = {
        'c1': [1.0, 0.0], 'd1': [0.99, 0.14],
        'c2': [0.0, 1.0], 'd2': [0.14, 0.99],
        'o1': [0.97, 0.24],
    }
    calls: list[str] = []

    def embed(mid: str, content: str) -> list[float]:
        calls.append(mid)
        return vectors[mid]

    embed.calls = calls  # type: ignore[attr-defined]
    return embed


class TestRunCalibrationPerCategory:
    """End-to-end wiring: the same embeddings, measured a second way.

    The per-category evidence must come out of the SAME single embedding
    pass — re-embedding per category would multiply a real API bill by the
    number of categories for no new information.
    """

    def _run_mixed(self, tmp_path: Path, embed=None):
        return _run(
            tmp_path,
            records=_mixed_category_records(),
            embed=embed if embed is not None else _mixed_embed(),
            search=_search_fn(hits={'d1': ['c1'], 'd2': ['c2'], 'o1': ['c1']},
                              present={'c1', 'c2'}),
        )

    def test_the_report_carries_a_per_category_entry_for_every_category(
        self, tmp_path: Path,
    ) -> None:
        report = self._run_mixed(tmp_path)['report']
        assert set(report['per_category']) == {
            'procedural_knowledge', 'observations_and_summaries',
        }

    def test_the_embed_budget_is_unchanged_at_one_call_per_record(
        self, tmp_path: Path,
    ) -> None:
        embed = _mixed_embed()
        records = _mixed_category_records()
        self._run_mixed(tmp_path, embed=embed)
        assert sorted(embed.calls) == sorted(r['memory_id'] for r in records), (
            'per-category measurement must reuse the single embedding pass'
        )

    def test_a_categorys_cutoff_is_bound_to_that_categorys_own_measurements(
        self, tmp_path: Path,
    ) -> None:
        """Derived from its own pairs — not inherited, not interpolated.

        Asserted against the distributions the report itself records, so a
        report that does not describe the derivation that produced it fails.
        """
        entry = self._run_mixed(tmp_path)['report']['per_category'][
            'procedural_knowledge'
        ]
        dup, unrelated = entry['distributions']['true_dup'], entry['distributions']['unrelated']
        assert dup['n'] == entry['pair_counts']['true_dup']
        assert entry['t_high'] is not None, 'this category is separable in the sample'
        assert dup['min'] <= entry['t_high'] <= dup['max'], (
            't_high must be an order statistic of this category\'s duplicate class'
        )
        assert entry['t_high'] > unrelated['max'], (
            't_high must clear this category\'s OWN highest measured negative'
        )

    def test_a_category_with_no_pairs_is_present_and_refused(self, tmp_path: Path) -> None:
        """A lone record forms no pair — 'not derivable' is still a measurement."""
        entry = self._run_mixed(tmp_path)['report']['per_category'][
            'observations_and_summaries'
        ]
        assert entry['pair_counts'] == {'true_dup': 0, 'unrelated': 0, 'hard_negative': 0}
        assert entry['t_high'] is None
        assert entry['reason'].startswith(_mod().REASON_EMPTY_CLASS)

    def test_provenance_records_the_per_category_record_counts(
        self, tmp_path: Path,
    ) -> None:
        prov = self._run_mixed(tmp_path)['report']['provenance']
        assert prov['per_category_record_counts'] == {
            'procedural_knowledge': 4, 'observations_and_summaries': 1,
        }

    def test_provenance_records_the_cross_category_pairs_dropped(
        self, tmp_path: Path,
    ) -> None:
        """A disclosed number, not invisible attrition."""
        records = _mixed_category_records()
        prov = self._run_mixed(tmp_path)['report']['provenance']
        expected = _mod().partition_pairs_by_category(records)['cross_category_dropped']
        assert expected == 4, 'sample sanity: o1 pairs with each of the four others'
        assert prov['cross_category_dropped'] == expected

    def test_the_markdown_carries_the_per_category_table(self, tmp_path: Path) -> None:
        self._run_mixed(tmp_path)
        md = (tmp_path / 'report.md').read_text()
        for category in ('procedural_knowledge', 'observations_and_summaries'):
            assert f'| {category} |' in md, f'{category} missing from the markdown table'


class TestCommittedCalibrationIsTraceable:
    """The committed thresholds must be traceable to the run that produced them.

    This is the task's headline signal, and the reason
    ``calibration_report_path`` exists at all: a reader must be able to get
    from a number in config.yaml back to the measured distributions that
    justify it, without taking anyone's word that it was not hand-picked.
    These tests read only committed artifacts — no network, no Qdrant.
    """

    CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

    # Decorator order is load-bearing: ``staticmethod`` must be OUTERMOST so
    # ``self._committed()`` resolves through the descriptor and calls the
    # cached function with zero arguments. Stacking them the other way round
    # leaves ``functools.cache``'s plain-function wrapper as the class
    # attribute, which binds ``self`` as its first argument and raises
    # TypeError on every access.
    @staticmethod
    @functools.cache
    def _committed():
        import yaml  # noqa: PLC0415

        block = yaml.safe_load(
            TestCommittedCalibrationIsTraceable.CONFIG_PATH.read_text()
        ).get('write_triage') or {}
        report_path = block.get('calibration_report_path')
        report = None
        if report_path is not None:
            resolved = Path(__file__).parent.parent / report_path
            if resolved.exists():
                report = json.loads(resolved.read_text())
        return block, report

    def test_the_report_path_resolves_to_a_committed_report(self):
        block, report = self._committed()
        if block.get('t_high') is None:
            pytest.skip('config is uncalibrated — nothing to trace')
        assert block.get('calibration_report_path'), (
            'calibrated thresholds without a report path are untraceable numbers'
        )
        assert not Path(block['calibration_report_path']).is_absolute(), (
            'the report path must not bake in the checkout it was produced in — '
            'this script runs in per-task worktrees that get reset'
        )
        assert report is not None, (
            f'calibration_report_path {block["calibration_report_path"]!r} does not '
            f'resolve to a committed report'
        )

    def test_the_committed_thresholds_are_exactly_the_reports(self):
        block, report = self._committed()
        if block.get('t_high') is None or report is None:
            pytest.skip('config is uncalibrated — nothing to trace')
        assert block['t_high'] == report['chosen_t_high']
        assert block['t_low'] == report['chosen_t_low'], (
            'a config value that differs from the report is a hand-edited '
            'threshold, which is exactly what this leaf forbids'
        )

    def test_the_committed_thresholds_separate_the_measured_classes(self):
        block, report = self._committed()
        if block.get('t_high') is None or report is None:
            pytest.skip('config is uncalibrated — nothing to trace')
        dists = report['distributions']
        worst_negative = max(
            dists[cls]['max'] for cls in ('unrelated', 'hard_negative')
            if dists.get(cls, {}).get('n')
        )
        assert block['t_high'] > worst_negative, (
            'the deterministic band must sit strictly above every measured '
            'negative, or it fires on pairs the curator ruled distinct'
        )
        assert block['t_low'] < block['t_high']
        assert report['deterministic_band_false_positives'] == 0
