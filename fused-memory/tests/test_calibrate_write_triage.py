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
