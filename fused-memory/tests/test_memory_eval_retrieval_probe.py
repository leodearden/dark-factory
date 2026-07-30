"""Tests for memory_eval_retrieval_probe.py — the E1 retrieval-health probe.

The script is loaded via importlib so it can be tested without sys.path
pollution — mirrors the pattern in test_calibrate_write_triage.py and
test_audit_duplicate_memories.py. The loader is invoked lazily (``_mod()``)
so the committed-fixture contract tests stay runnable independently.

**Lane discipline.** Every test in this file except the single seeded
induced-regression test is free of network, Qdrant, OPENAI_API_KEY and any
live store: the probe's metric computations are pure functions over
already-fetched result lists, precisely so the merge lane (which runs under
``addopts = -m 'not integration'``) covers all of them. The one integration
test carries ``@pytest.mark.integration`` PER-TEST rather than as a module
``pytestmark``, so marking it never deselects the pure tests here.

**No thresholds.** Per the plan's G6 decision, no test in this file asserts a
rate, tolerance, bound or pass/fail limit. Assertions are on structure, on
booleans and on flips. ``k`` appears only as a metric parameterisation.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_retrieval_probe.py'
REGISTRY_PATH = Path(__file__).parent / 'fixtures' / 'memory_eval_topic_registry.json'


def _load_module() -> types.ModuleType:
    """Load memory_eval_retrieval_probe.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'memory_eval_retrieval_probe'
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
# Registry payload builders (in-memory; no fixture file needed)
# ---------------------------------------------------------------------------

def _entry_payload(topic: str = 'sample-topic', **overrides) -> dict:
    """A minimal well-formed registry entry, overridable per test."""
    payload = {
        'topic': topic,
        'project_id': 'dark_factory',
        'derived_from': 'hand',
        'canonical': {
            'content_hash': 'a' * 16,
            'content_prefix': 'The sample canonical entry says something.',
            'last_known_id': '0b746438-6ce8-435c-885c-b3ac82666764',
        },
        'phrasings': [
            {'text': 'what does the sample topic say', 'held_out': False},
            {'text': 'sample topic summary', 'held_out': False},
            {'text': 'freshly authored held-out phrasing for sample', 'held_out': True},
        ],
        'claim_queries': [
            {'query': 'sample topic claim', 'needles': ['something']},
        ],
        'members': ['b' * 16],
        'supersedes_pairs': [],
    }
    payload.update(overrides)
    return payload


def _registry_payload(*entries: dict) -> dict:
    return {
        'schema_version': 1,
        'entries': list(entries) or [_entry_payload()],
    }


def _write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# step-1: registry schema + loader
# ---------------------------------------------------------------------------

class TestLoadTopicRegistry:
    """`load_topic_registry(path)` — required-strict, additive-tolerant."""

    def test_well_formed_payload_exposes_every_field(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(
            _entry_payload('briefing-assembler-queries', project_id='reify'),
        ))
        registry = _mod().load_topic_registry(path)

        assert len(registry.entries) == 1
        entry = registry.entries[0]
        assert entry.topic == 'briefing-assembler-queries'
        assert entry.project_id == 'reify'
        assert entry.derived_from == 'hand'
        assert entry.canonical.content_hash == 'a' * 16
        assert entry.canonical.content_prefix.startswith('The sample canonical')
        assert entry.canonical.last_known_id == '0b746438-6ce8-435c-885c-b3ac82666764'
        assert [p.text for p in entry.phrasings][-1].startswith('freshly authored')
        assert [p.held_out for p in entry.phrasings] == [False, False, True]
        assert entry.claim_queries[0].query == 'sample topic claim'
        # Tuples, not lists: the registry models are frozen dataclasses, so a
        # loaded entry cannot be mutated out from under a probe mid-run.
        assert entry.claim_queries[0].needles == ('something',)
        assert entry.members == ('b' * 16,)
        assert entry.supersedes_pairs == ()

    def test_registry_exposes_topic_lookup(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(
            _entry_payload('alpha-topic'), _entry_payload('beta-topic'),
        ))
        registry = _mod().load_topic_registry(path)

        assert registry.topics == {'alpha-topic', 'beta-topic'}
        assert registry.by_topic['alpha-topic'].topic == 'alpha-topic'

    def test_supersedes_pairs_are_parsed(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(_entry_payload(
            supersedes_pairs=[
                {'superseded_hash': 'c' * 16, 'successor_hash': 'd' * 16},
            ],
        )))
        entry = _mod().load_topic_registry(path).entries[0]

        assert len(entry.supersedes_pairs) == 1
        assert entry.supersedes_pairs[0].superseded_hash == 'c' * 16
        assert entry.supersedes_pairs[0].successor_hash == 'd' * 16

    # -- rejection matrix: every message must NAME the offending topic slug --

    @pytest.mark.parametrize('missing', ['project_id', 'canonical', 'phrasings', 'derived_from'])
    def test_missing_required_field_names_the_topic(self, tmp_path, missing):
        bad = _entry_payload('offending-slug')
        del bad[missing]
        path = _write_registry(tmp_path, _registry_payload(bad))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'offending-slug' in str(exc.value)
        assert missing in str(exc.value)

    def test_missing_topic_key_is_rejected(self, tmp_path):
        bad = _entry_payload()
        del bad['topic']
        path = _write_registry(tmp_path, _registry_payload(bad))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'topic' in str(exc.value)

    def test_duplicate_topic_slug_names_the_topic(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(
            _entry_payload('dup-slug'), _entry_payload('dup-slug'),
        ))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'dup-slug' in str(exc.value)
        assert 'duplicate' in str(exc.value).lower()

    def test_zero_phrasings_names_the_topic(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(
            _entry_payload('empty-phrasings', phrasings=[]),
        ))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'empty-phrasings' in str(exc.value)

    def test_no_held_out_phrasing_names_the_topic(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(_entry_payload(
            'no-heldout',
            phrasings=[
                {'text': 'one', 'held_out': False},
                {'text': 'two', 'held_out': False},
            ],
        )))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'no-heldout' in str(exc.value)
        assert 'held_out' in str(exc.value)

    def test_malformed_canonical_names_the_topic(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(_entry_payload(
            'bad-canonical', canonical={'content_prefix': 'no hash here'},
        )))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'bad-canonical' in str(exc.value)
        assert 'content_hash' in str(exc.value)

    def test_unknown_derived_from_names_the_topic(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(
            _entry_payload('bad-provenance', derived_from='telepathy'),
        ))

        with pytest.raises(_mod().RegistryError) as exc:
            _mod().load_topic_registry(path)
        assert 'bad-provenance' in str(exc.value)
        assert 'telepathy' in str(exc.value)

    def test_missing_file_raises_registry_error(self, tmp_path):
        with pytest.raises(_mod().RegistryError):
            _mod().load_topic_registry(tmp_path / 'does-not-exist.json')

    # -- forward-compat: additive-tolerant (3201 widens derivation later) --

    def test_unknown_entry_key_loads_successfully(self, tmp_path):
        path = _write_registry(tmp_path, _registry_payload(_entry_payload(
            'forward-compat', future_field_from_3201={'kind': 'gotcha'},
        )))
        entry = _mod().load_topic_registry(path).entries[0]

        assert entry.topic == 'forward-compat'
        assert entry.extra['future_field_from_3201'] == {'kind': 'gotcha'}

    def test_unknown_top_level_key_loads_successfully(self, tmp_path):
        payload = _registry_payload()
        payload['derived_at'] = '2026-07-30T00:00:00Z'
        path = _write_registry(tmp_path, payload)

        assert len(_mod().load_topic_registry(path).entries) == 1


# ---------------------------------------------------------------------------
# step-3: the committed registry fixture's data contract
#
# Loaded through load_topic_registry, so the fixture is validated by the
# PRODUCTION loader rather than a test-local parser — a fixture that only a
# bespoke test parser accepts is not a fixture the runner can read.
# ---------------------------------------------------------------------------

BRIEFING_QUERIES = (
    'project overview architecture goals',
    'coding conventions and project norms',
    'recent decisions and rationale',
)
"""The three literal briefing-assembler queries (briefing.py:978-1013).

The fourth is templated — ``f'task {task_id} context and related decisions'``
— so it is asserted separately: a literal ``{id}`` is never a real query.
"""

BRIEFING_TASK_QUERY_PREFIX = 'task '
BRIEFING_TASK_QUERY_SUFFIX = ' context and related decisions'

MIN_TOPICS = 20
"""A structural floor on fixture BREADTH, not a metric threshold.

eval-design:296 targets ~20-40 topics. Nothing about a probe RESULT is
compared to this number; it exists so a fixture silently shrinking to three
topics — which would report a clean run because there was almost nothing to
probe — fails the suite instead.
"""

MIN_PHRASINGS_PER_TOPIC = 3
"""Also structural: one held-out plus at least two ordinary phrasings, so a
topic's pooled rate is never a single coin flip."""


@pytest.fixture(scope='module')
def registry():
    return _mod().load_topic_registry(REGISTRY_PATH)


def _guard_cluster_phrases() -> set[str]:
    from fused_memory.config.schema import _default_topic_guard_clusters  # noqa: PLC0415

    return {
        phrase.strip().lower()
        for cluster in _default_topic_guard_clusters()
        for phrase in cluster.phrases
    }


class TestCommittedRegistryFixture:
    """The committed registry is the probe's input — its shape is a contract."""

    def test_fixture_exists_and_parses(self, registry):
        assert REGISTRY_PATH.is_file()
        assert registry.schema_version == _mod().REGISTRY_SCHEMA_VERSION
        assert registry.entries

    def test_topic_slugs_are_unique_and_slug_shaped(self, registry):
        topics = [e.topic for e in registry.entries]
        assert len(topics) == len(set(topics))
        for topic in topics:
            assert _mod()._SLUG_RE.match(topic), f'{topic!r} is not slug-shaped'

    def test_breadth_floor(self, registry):
        assert len(registry.entries) >= MIN_TOPICS

    def test_every_entry_has_enough_phrasings_including_a_held_out_one(self, registry):
        for entry in registry.entries:
            assert len(entry.phrasings) >= MIN_PHRASINGS_PER_TOPIC, entry.topic
            assert entry.held_out_phrasings, entry.topic

    def test_every_entry_has_a_claim_query_with_needles(self, registry):
        for entry in registry.entries:
            assert entry.claim_queries, entry.topic
            for claim in entry.claim_queries:
                assert claim.needles, f'{entry.topic}: {claim.query!r} has no needles'

    def test_canonical_hashes_match_content_key_shape(self, registry):
        for entry in registry.entries:
            digest = entry.canonical.content_hash
            assert len(digest) == 16, entry.topic
            assert all(c in '0123456789abcdef' for c in digest), entry.topic

    def test_member_hashes_match_content_key_shape(self, registry):
        for entry in registry.entries:
            for member in entry.members:
                assert len(member) == 16, f'{entry.topic}: {member!r}'
                assert all(c in '0123456789abcdef' for c in member), entry.topic

    def test_project_ids_are_known(self, registry):
        for entry in registry.entries:
            assert entry.project_id in {'dark_factory', 'reify'}, entry.topic

    def test_derived_from_is_in_the_closed_set(self, registry):
        for entry in registry.entries:
            assert entry.derived_from in _mod().DERIVED_FROM_VALUES, entry.topic

    def test_supersedes_pairs_reference_distinct_hashes(self, registry):
        for entry in registry.entries:
            for pair in entry.supersedes_pairs:
                assert pair.superseded_hash != pair.successor_hash, entry.topic

    # -- the briefing-assembler query surface (eval-design:297) --

    def test_literal_briefing_queries_appear_verbatim(self, registry):
        all_phrasings = {p.text for e in registry.entries for p in e.phrasings}
        for query in BRIEFING_QUERIES:
            assert query in all_phrasings, f'briefing query {query!r} is not probed'

    def test_templated_briefing_query_is_instantiated_not_literal(self, registry):
        all_phrasings = {p.text for e in registry.entries for p in e.phrasings}
        matches = [
            text for text in all_phrasings
            if text.startswith(BRIEFING_TASK_QUERY_PREFIX)
            and text.endswith(BRIEFING_TASK_QUERY_SUFFIX)
        ]
        assert matches, 'the templated briefing query is not probed'
        for text in matches:
            middle = text[len(BRIEFING_TASK_QUERY_PREFIX):-len(BRIEFING_TASK_QUERY_SUFFIX)]
            assert '{' not in middle and '}' not in middle, (
                f'{text!r} carries a literal template placeholder; a probe must issue '
                'the query a caller would actually issue, with a concrete task id.'
            )
            assert middle.strip(), text

    # -- the Goodhart guard, made checkable --

    def test_no_held_out_phrasing_duplicates_a_tuned_phrasing(self, registry):
        tuned = {
            p.text.strip().lower()
            for e in registry.entries for p in e.phrasings if not p.held_out
        }
        for entry in registry.entries:
            for phrasing in entry.held_out_phrasings:
                assert phrasing.text.strip().lower() not in tuned, (
                    f'{entry.topic}: held-out phrasing {phrasing.text!r} is also used as '
                    'a tuned phrasing somewhere in the registry, so it is not held out.'
                )

    def test_no_held_out_phrasing_reuses_a_topic_guard_phrase(self, registry):
        guard_phrases = _guard_cluster_phrases()
        for entry in registry.entries:
            for phrasing in entry.held_out_phrasings:
                assert phrasing.text.strip().lower() not in guard_phrases, (
                    f'{entry.topic}: held-out phrasing {phrasing.text!r} is one of the '
                    'topic-guard phrases those entries were BUILT from — reusing it '
                    'defeats the guard it exists to be.'
                )

    def test_held_out_phrasings_are_unique_across_the_registry(self, registry):
        seen: dict[str, str] = {}
        for entry in registry.entries:
            for phrasing in entry.held_out_phrasings:
                key = phrasing.text.strip().lower()
                assert key not in seen, (
                    f'held-out phrasing {phrasing.text!r} is shared by '
                    f'{seen.get(key)!r} and {entry.topic!r}'
                )
                seen[key] = entry.topic


# ---------------------------------------------------------------------------
# step-5: offline registry derivation
#
# Every source here is COMMITTED, so derivation needs no Qdrant, no embedder
# and no OPENAI_API_KEY — a reviewer can re-run it to audit any fixture entry.
# ---------------------------------------------------------------------------

CALIBRATION_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'
CENSUS_PATH = Path(__file__).parents[2] / 'plans' / 'memory-metadata-census-report.json'


@pytest.fixture(scope='module')
def calibration_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in CALIBRATION_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


@pytest.fixture(scope='module')
def census_report() -> dict:
    return json.loads(CENSUS_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def guard_clusters():
    from fused_memory.config.schema import _default_topic_guard_clusters  # noqa: PLC0415

    return _default_topic_guard_clusters()


@pytest.fixture(scope='module')
def derived(calibration_rows, census_report, guard_clusters):
    return _mod().derive_registry_candidates(calibration_rows, census_report, guard_clusters)


def _of(result, source: str) -> list[dict]:
    return [c for c in result.candidates if c['derived_from'] == source]


class TestDeriveFromCuratorGates:
    """The curator-gate census — 'each gate is a free labeled cluster'."""

    def test_one_candidate_per_cluster(self, derived, calibration_rows):
        clusters = {r['cluster_id'] for r in calibration_rows}
        assert len(_of(derived, 'curator_gate')) == len(clusters)

    def test_canonical_hash_is_content_key_of_the_canonical_row(
        self, derived, calibration_rows,
    ):
        content_key = _mod().content_key
        canonical_by_cluster = {
            r['cluster_id']: r for r in calibration_rows if r['label'] == 'canonical'
        }
        for candidate in _of(derived, 'curator_gate'):
            row = canonical_by_cluster[candidate['provenance']['cluster_id']]
            assert candidate['canonical']['content_hash'] == content_key(row['content'])
            assert candidate['canonical']['last_known_id'] == row['memory_id']

    def test_members_are_the_duplicate_rows(self, derived, calibration_rows):
        content_key = _mod().content_key
        dups = {}
        for row in calibration_rows:
            if row['label'] == 'duplicate':
                dups.setdefault(row['cluster_id'], set()).add(content_key(row['content']))
        for candidate in _of(derived, 'curator_gate'):
            expected = dups.get(candidate['provenance']['cluster_id'], set())
            assert set(candidate['members']) == expected

    def test_distinct_and_pseudo_contradiction_are_not_members(
        self, derived, calibration_rows,
    ):
        """The curator adjudicated these as SEPARATE claims.

        Folding them in would poison the contamination metric with entries that
        legitimately answer a different question.
        """
        content_key = _mod().content_key
        excluded = {
            content_key(r['content']) for r in calibration_rows
            if r['label'] in ('distinct', 'pseudo_contradiction')
        }
        assert excluded, 'fixture no longer carries the adjudicated non-duplicates'
        all_members = {m for c in _of(derived, 'curator_gate') for m in c['members']}
        assert not (all_members & excluded)

    def test_gate_id_is_carried_as_provenance(self, derived):
        for candidate in _of(derived, 'curator_gate'):
            assert candidate['provenance']['gate_ids']
            for gate in candidate['provenance']['gate_ids']:
                assert gate.startswith('esc-')

    def test_supersedes_pairs_point_duplicates_at_the_canonical(self, derived):
        for candidate in _of(derived, 'curator_gate'):
            members = set(candidate['members'])
            for pair in candidate['supersedes_pairs']:
                assert pair['superseded_hash'] in members
                assert pair['successor_hash'] == candidate['canonical']['content_hash']


class TestDeriveFromCensus:
    """Multi-entry census topics, with the skipped long tail DISCLOSED."""

    def test_emits_multi_entry_topics_only(self, derived, census_report):
        multi = {
            e['value'] for e in census_report['grand_total']['topic']['entries']
            if e['count'] > 1
        }
        emitted = {c['topic'] for c in _of(derived, 'census_topic')}
        assert emitted
        assert emitted <= multi

    def test_skipped_singletons_are_disclosed_not_silently_dropped(
        self, derived, census_report,
    ):
        singletons = [
            e for e in census_report['grand_total']['topic']['entries'] if e['count'] <= 1
        ]
        assert singletons, 'census no longer has a count-1 tail'
        assert derived.disclosures['census_topics_skipped_singleton'] == len(singletons)

    def test_census_forward_compat_extra_key(self, calibration_rows, guard_clusters):
        payload = {
            'grand_total': {
                'topic': {
                    'distinct_total': 2,
                    'entries': [{'value': 'multi-topic', 'count': 3}],
                },
                'unrecognised_key_from_3201': {'anything': True},
            },
            'another_new_top_level_key': 1,
        }
        result = _mod().derive_registry_candidates(
            calibration_rows, payload, guard_clusters,
        )
        assert {c['topic'] for c in _of(result, 'census_topic')} == {'multi-topic'}


class TestDeriveFromGuardClusters:
    """Guard phrases seed ORDINARY phrasings only — never the held-out one."""

    def test_emits_one_candidate_per_guard_slug(self, derived, guard_clusters):
        emitted = {c['topic'] for c in _of(derived, 'topic_guard_cluster')}
        assert emitted == {c.topic_id for c in guard_clusters}

    def test_guard_phrases_seed_phrasings(self, derived, guard_clusters):
        by_slug = {c.topic_id: c for c in guard_clusters}
        for candidate in _of(derived, 'topic_guard_cluster'):
            texts = {p['text'] for p in candidate['phrasings']}
            assert texts & set(by_slug[candidate['topic']].phrases)

    def test_no_derived_phrasing_is_marked_held_out(self, derived):
        """Held-out phrasings are the part machines cannot regenerate.

        A guard phrase was used to BUILD the entries it would retrieve, so
        marking one held out would defeat the guard it exists to be.
        """
        for candidate in derived.candidates:
            assert not any(p['held_out'] for p in candidate['phrasings']), candidate['topic']


class TestDerivationIsPure:
    def test_running_twice_is_byte_identical(
        self, calibration_rows, census_report, guard_clusters,
    ):
        first = _mod().derive_registry_candidates(
            calibration_rows, census_report, guard_clusters,
        )
        second = _mod().derive_registry_candidates(
            calibration_rows, census_report, guard_clusters,
        )
        assert json.dumps(first.as_registry_payload(), sort_keys=True) == json.dumps(
            second.as_registry_payload(), sort_keys=True,
        )

    def test_no_live_store_is_constructed(self, monkeypatch, calibration_rows,
                                          census_report, guard_clusters):
        """Derivation must not reach for Qdrant, an embedder or MemoryService."""
        import socket  # noqa: PLC0415

        def _boom(*a, **kw):
            raise AssertionError('derivation opened a socket')

        monkeypatch.setattr(socket.socket, 'connect', _boom)
        monkeypatch.setattr(socket.socket, 'connect_ex', _boom)
        result = _mod().derive_registry_candidates(
            calibration_rows, census_report, guard_clusters,
        )
        assert result.candidates

    def test_payload_is_shaped_like_the_registry(
        self, derived, tmp_path,
    ):
        """The deriver writes what the loader reads — schema drift is impossible.

        Loaded through load_topic_registry after filling in only the parts a
        machine cannot regenerate, so the two halves are proven to agree rather
        than assumed to. Three fields are completed here:

        - a held-out phrasing and a claim query (hand-authored by design), and
        - the canonical content_hash for census/guard candidates, which is NOT
          offline-derivable: the census records topic VALUES and counts, and a
          guard cluster records match phrases — neither carries the canonical's
          content. Only the curator-gate 20 hash offline, and the assertion
          below pins exactly that split so a future deriver silently emitting
          an empty hash for a curator cluster fails here.
        """
        payload = derived.as_registry_payload()
        hashless = set()
        for entry in payload['entries']:
            entry['phrasings'].append({'text': f'held out for {entry["topic"]}',
                                       'held_out': True})
            entry.setdefault('claim_queries', []).append(
                {'query': f'{entry["topic"]} claim', 'needles': ['x']},
            )
            if not entry['canonical']['content_hash']:
                hashless.add(entry['derived_from'])
                entry['canonical']['content_hash'] = _mod().content_key(entry['topic'])

        assert 'curator_gate' not in hashless
        assert hashless == {'census_topic', 'topic_guard_cluster'}
        path = tmp_path / 'derived.json'
        path.write_text(json.dumps(payload), encoding='utf-8')

        registry = _mod().load_topic_registry(path)
        assert len(registry.entries) == len(payload['entries'])


# ---------------------------------------------------------------------------
# step-7: canonical-in-top-k, pure over synthetic result lists
#
# No assertion below is on a RATE. Every one is on a hit/miss boolean, a rank,
# or a matched_by string — G6: this runner measures, leaf alpha sets limits.
# ---------------------------------------------------------------------------

class _R:
    """A stand-in for MemoryResult: id, content, metadata, relevance_score."""

    def __init__(self, content='', id='', metadata=None, relevance_score=0.0):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.relevance_score = relevance_score


def _entry(topic='t', content='the canonical text', last_known_id='ID-1', **kw):
    """Build a real RegistryEntry whose canonical hashes *content*."""
    m = _mod()
    return m.RegistryEntry(
        topic=topic,
        project_id=kw.pop('project_id', 'dark_factory'),
        derived_from=kw.pop('derived_from', 'hand'),
        canonical=m.Canonical(
            content_hash=kw.pop('content_hash', m.content_key(content)),
            content_prefix=content[:80],
            last_known_id=last_known_id,
        ),
        phrasings=kw.pop('phrasings', (m.Phrasing('q', False), m.Phrasing('h', True))),
        **kw,
    )


CANON = 'the canonical text'


def _filler(n, start=0):
    return [_R(content=f'unrelated filler {i}', id=f'F{i}') for i in range(start, start + n)]


class TestCanonicalHit:
    def test_rank_one_and_rank_k_both_hit(self):
        entry = _entry(content=CANON)
        at_one = [_R(content=CANON, id='ID-1'), *_filler(9)]
        at_five = [*_filler(4), _R(content=CANON, id='ID-1'), *_filler(5, start=4)]

        assert _mod().canonical_hit(at_one, entry, 5).hit
        assert _mod().canonical_hit(at_one, entry, 5).rank == 1
        assert _mod().canonical_hit(at_five, entry, 5).hit
        assert _mod().canonical_hit(at_five, entry, 5).rank == 5

    def test_rank_k_plus_one_does_not_hit(self):
        entry = _entry(content=CANON)
        results = [*_filler(5), _R(content=CANON, id='ID-1')]
        outcome = _mod().canonical_hit(results, entry, 5)

        assert not outcome.hit
        assert outcome.rank == 6, 'the true rank is still reported, for the report'

    def test_k_is_honored_independently(self):
        """The parameterisation, exercised: rank 7 misses at k=5 and hits at k=10."""
        entry = _entry(content=CANON)
        results = [*_filler(6), _R(content=CANON, id='ID-1'), *_filler(3, start=6)]

        assert not _mod().canonical_hit(results, entry, 5).hit
        assert _mod().canonical_hit(results, entry, 10).hit

    def test_content_hash_matches_even_when_the_id_rotated(self):
        entry = _entry(content=CANON, last_known_id='OLD-UUID')
        results = [_R(content=CANON, id='BRAND-NEW-UUID')]
        outcome = _mod().canonical_hit(results, entry, 5)

        assert outcome.hit
        assert outcome.matched_by == 'content_hash'

    def test_id_fallback_hits_and_flags_a_hash_repair(self):
        entry = _entry(content=CANON, last_known_id='ID-1')
        results = [_R(content='the canonical text, lightly reworded', id='ID-1')]
        outcome = _mod().canonical_hit(results, entry, 5)

        assert outcome.hit
        assert outcome.matched_by == 'last_known_id'
        assert outcome.needs_hash_repair

    def test_content_hash_wins_when_both_could_match(self):
        entry = _entry(content=CANON, last_known_id='ID-1')
        results = [_R(content='something else', id='ID-1'), _R(content=CANON, id='OTHER')]
        outcome = _mod().canonical_hit(results, entry, 5)

        assert outcome.matched_by == 'content_hash'
        assert outcome.rank == 2
        assert not outcome.needs_hash_repair

    def test_matching_neither_is_recorded_as_unmatched(self):
        entry = _entry(content=CANON, last_known_id='ID-1')
        outcome = _mod().canonical_hit(_filler(5), entry, 5)

        assert not outcome.hit
        assert outcome.matched_by is None
        assert outcome.unmatched
        assert outcome.rank is None

    def test_hash_matching_survives_whitespace_churn(self):
        entry = _entry(content=CANON)
        results = [_R(content='  the   canonical\n text  ', id='X')]

        assert _mod().canonical_hit(results, entry, 5).matched_by == 'content_hash'

    def test_absent_last_known_id_never_matches_an_empty_result_id(self):
        """A registry entry with no id must not match a result with no id."""
        entry = _entry(content=CANON, last_known_id=None)
        outcome = _mod().canonical_hit([_R(content='other', id='')], entry, 5)

        assert not outcome.hit
        assert outcome.unmatched


class TestObservationBuilder:
    def test_records_topic_phrasing_and_k(self):
        m = _mod()
        entry = _entry(topic='my-topic', content=CANON)
        obs = m.observe_phrasing(
            [_R(content=CANON, id='ID-1')], entry, m.Phrasing('some query', False), 5,
        )

        assert obs.topic == 'my-topic'
        assert obs.phrasing == 'some query'
        assert obs.k == 5
        assert obs.hit
        assert obs.matched_by == 'content_hash'

    def test_held_out_phrasings_are_tagged(self):
        m = _mod()
        entry = _entry(content=CANON)
        tuned = m.observe_phrasing([], entry, m.Phrasing('tuned', False), 5)
        held = m.observe_phrasing([], entry, m.Phrasing('fresh', True), 5)

        assert not tuned.held_out
        assert held.held_out

    def test_observation_is_not_degraded_by_default(self):
        m = _mod()
        obs = m.observe_phrasing([], _entry(content=CANON), m.Phrasing('q', False), 5)
        assert not obs.degraded


# ---------------------------------------------------------------------------
# step-9: superseded-above-successor, pure over synthetic result lists
# ---------------------------------------------------------------------------

def _pair_entry(topic='sup', old='old text', new='new text'):
    m = _mod()
    base = _entry(topic=topic, content=new)
    return m.RegistryEntry(
        topic=base.topic,
        project_id=base.project_id,
        derived_from=base.derived_from,
        canonical=base.canonical,
        phrasings=base.phrasings,
        supersedes_pairs=(
            m.SupersedesPair(
                superseded_hash=m.content_key(old),
                successor_hash=m.content_key(new),
            ),
        ),
    )


class TestSupersededInversions:
    def test_superseded_ranked_above_successor_is_one_inversion(self):
        entry = _pair_entry()
        results = [_R(content='old text', id='A'), _R(content='new text', id='B')]
        inversions = _mod().superseded_inversions(results, entry)

        assert len(inversions) == 1
        record = inversions[0]
        assert record.superseded_hash == _mod().content_key('old text')
        assert record.successor_hash == _mod().content_key('new text')
        assert record.superseded_rank == 1
        assert record.successor_rank == 2
        assert record.topic == 'sup'

    def test_correct_order_is_no_inversion(self):
        entry = _pair_entry()
        results = [_R(content='new text', id='B'), _R(content='old text', id='A')]

        assert _mod().superseded_inversions(results, entry) == []

    @pytest.mark.parametrize('present', ['old text', 'new text'])
    def test_only_one_member_present_is_no_inversion(self, present):
        """An absent successor is a findability question, not an inversion.

        canonical-in-top-k already measures it; counting it here too would
        charge one defect against two metrics.
        """
        entry = _pair_entry()
        results = [*_filler(2), _R(content=present, id='X')]

        assert _mod().superseded_inversions(results, entry) == []

    def test_neither_present_is_no_inversion(self):
        assert _mod().superseded_inversions(_filler(3), _pair_entry()) == []

    def test_several_pairs_are_counted_independently(self):
        m = _mod()
        base = _entry(topic='multi', content='c1')
        entry = m.RegistryEntry(
            topic=base.topic, project_id=base.project_id,
            derived_from=base.derived_from, canonical=base.canonical,
            phrasings=base.phrasings,
            supersedes_pairs=(
                m.SupersedesPair(m.content_key('o1'), m.content_key('c1')),
                m.SupersedesPair(m.content_key('o2'), m.content_key('c2')),
            ),
        )
        results = [
            _R(content='o1'), _R(content='o2'), _R(content='c1'), _R(content='c2'),
        ]

        assert len(m.superseded_inversions(results, entry)) == 2

    def test_ties_resolve_by_returned_list_order_deterministically(self):
        """Equal relevance_score must not let the count flap between runs."""
        entry = _pair_entry()
        tied_old_first = [
            _R(content='old text', relevance_score=0.5),
            _R(content='new text', relevance_score=0.5),
        ]
        tied_new_first = [
            _R(content='new text', relevance_score=0.5),
            _R(content='old text', relevance_score=0.5),
        ]
        run = _mod().superseded_inversions

        assert len(run(tied_old_first, entry)) == 1
        assert run(tied_old_first, entry) == run(tied_old_first, entry)
        assert run(tied_new_first, entry) == []

    def test_malformed_supersedes_metadata_is_never_read(self):
        """INV-5: the supersedes-pointer parser is task 3196's, not this one's.

        A deliberately malformed metadata['supersedes'] must not change the
        count — if it did, a second pointer parser would have grown here.
        """
        entry = _pair_entry()
        run = _mod().superseded_inversions
        clean = [_R(content='old text'), _R(content='new text')]
        poisoned = [
            _R(content='old text', metadata={'supersedes': {'nonsense': [1, 2, None]}}),
            _R(content='new text', metadata={'supersedes': 'not-a-list-or-dict'}),
        ]

        assert run(poisoned, entry) == run(clean, entry)
        assert len(run(poisoned, entry)) == 1

    def test_entry_without_pairs_yields_nothing(self):
        assert _mod().superseded_inversions(_filler(3), _entry()) == []


class TestContentKey:
    """`content_key(text)` — whitespace-normalized sha256[:16]."""

    def test_stable_across_surrounding_whitespace(self):
        content_key = _mod().content_key
        assert content_key('  hello world  ') == content_key('hello world')
        assert content_key('hello world\n') == content_key('hello world')

    def test_stable_across_collapsed_internal_whitespace(self):
        content_key = _mod().content_key
        assert content_key('hello   world') == content_key('hello world')
        assert content_key('hello\n\tworld') == content_key('hello world')

    def test_differs_for_different_text(self):
        content_key = _mod().content_key
        assert content_key('hello world') != content_key('goodbye world')

    def test_shape_is_sixteen_hex_chars(self):
        key = _mod().content_key('anything at all')
        assert len(key) == 16
        assert all(c in '0123456789abcdef' for c in key)

    def test_empty_and_whitespace_only_agree(self):
        content_key = _mod().content_key
        assert content_key('') == content_key('   \n\t ')


# ---------------------------------------------------------------------------
# step-11: claim recall, pure over synthetic result lists
#
# Claim recall is deliberately WEAKER than canonical identity: it asks whether
# the claim comes back AT ALL, from any returned entry. That is the
# Goodhart-resistant question — a consolidation that moved a claim into a
# different (or merged) entry has not lost the knowledge, and a metric that
# insisted on canonical identity would score exactly the 3111/3112
# consolidation work this eval exists to measure as a regression.
#
# As everywhere else in this file: no assertion below is on a rate.
# ---------------------------------------------------------------------------

def _claim(query, *needles):
    return _mod().ClaimQuery(query=query, needles=tuple(needles))


CLAIM_TEXT = (
    'The merge lane is strictly serial: one task lands at a time, and '
    'the queue advances only after the previous merge commit is reachable.'
)


class TestClaimRecalled:
    """`claim_recalled(results, claim_query, k)` — all needles, normalized, top-k."""

    def test_needles_in_a_non_canonical_entry_still_count(self):
        """(a) Recall asks 'did the claim come back', never 'from which entry'."""
        results = [
            _R(content='an unrelated preamble', id='X'),
            _R(content=CLAIM_TEXT, id='SOME-OTHER-ENTRY'),
        ]
        outcome = _mod().claim_recalled(
            results, _claim('how does the merge lane advance', 'strictly serial'), 5,
        )

        assert outcome.recalled
        assert outcome.missing_needles == ()
        assert outcome.matched_rank == 2

    def test_needles_absent_from_every_entry_is_not_recalled(self):
        outcome = _mod().claim_recalled(
            _filler(5), _claim('how does the merge lane advance', 'strictly serial'), 5,
        )

        assert not outcome.recalled
        assert outcome.missing_needles == ('strictly serial',)
        assert outcome.matched_rank is None

    def test_matching_is_whitespace_normalized_like_content_key(self):
        """(c) Re-wrapping a stored line must not read as knowledge loss."""
        results = [_R(content='the merge   lane is\n\tstrictly    serial', id='X')]
        outcome = _mod().claim_recalled(
            results, _claim('merge lane', 'merge lane is strictly serial'), 5,
        )

        assert outcome.recalled

    def test_matching_is_case_insensitive(self):
        """(c) Casing is presentation churn in prose — 'WRITE-SET' vs 'write-set'."""
        results = [_R(content='the plan files list is a WRITE-SET, not a read-set', id='X')]
        outcome = _mod().claim_recalled(
            results, _claim('what does the plan files list mean', 'write-set'), 5,
        )

        assert outcome.recalled

    def test_all_needles_are_required(self):
        """(d) A partial match is not recall."""
        results = [_R(content=CLAIM_TEXT, id='X')]
        outcome = _mod().claim_recalled(
            results,
            _claim('merge lane', 'strictly serial', 'never rolls back a landed merge'),
            5,
        )

        assert not outcome.recalled

    def test_missing_needles_name_what_did_not_come_back(self):
        """(d) The report must say WHAT was missing, not merely that something was."""
        results = [_R(content=CLAIM_TEXT, id='X')]
        outcome = _mod().claim_recalled(
            results,
            _claim('merge lane', 'strictly serial', 'never rolls back a landed merge'),
            5,
        )

        assert outcome.missing_needles == ('never rolls back a landed merge',)
        assert 'strictly serial' not in outcome.missing_needles

    def test_all_needles_must_come_from_one_entry(self):
        """Two entries jointly satisfying a claim neither one makes is not recall.

        Pooling needles across the result set would let an entry mentioning
        'strictly serial' and an unrelated entry mentioning 'rolls back'
        manufacture a claim the corpus never stated.
        """
        split = [
            _R(content='the merge lane is strictly serial', id='A'),
            _R(content='the queue never rolls back a landed merge', id='B'),
        ]
        outcome = _mod().claim_recalled(
            split,
            _claim('merge lane', 'strictly serial', 'never rolls back a landed merge'),
            5,
        )

        assert not outcome.recalled

    def test_missing_needles_come_from_the_closest_entry(self):
        """The best partial match is the informative one to diff against."""
        results = [
            _R(content='wholly unrelated filler', id='A'),
            _R(content='the merge lane is strictly serial', id='B'),
        ]
        outcome = _mod().claim_recalled(
            results,
            _claim('merge lane', 'strictly serial', 'never rolls back a landed merge'),
            5,
        )

        assert outcome.missing_needles == ('never rolls back a landed merge',)

    def test_recall_is_confined_to_the_top_k_slice(self):
        """(e) Same parameterised k as canonical-in-top-k."""
        results = [*_filler(5), _R(content=CLAIM_TEXT, id='X')]
        run = _mod().claim_recalled
        claim = _claim('merge lane', 'strictly serial')

        assert not run(results, claim, 5).recalled
        assert run(results, claim, 10).recalled

    def test_a_claim_query_with_no_needles_is_unscorable_not_recalled(self):
        """Vacuous truth over an empty needle set would be a silent free pass.

        'all of zero needles were found' is True, which would quietly inflate
        recall for a malformed registry entry. It is disclosed as unscorable
        instead, so leaf alpha's denominator can exclude it visibly.
        """
        outcome = _mod().claim_recalled(_filler(3), _claim('a claim with no needles'), 5)

        assert not outcome.scorable
        assert not outcome.recalled

    def test_a_needled_claim_is_scorable(self):
        outcome = _mod().claim_recalled(
            [_R(content=CLAIM_TEXT, id='X')], _claim('merge lane', 'strictly serial'), 5,
        )

        assert outcome.scorable

    def test_empty_results_are_not_recalled(self):
        outcome = _mod().claim_recalled([], _claim('merge lane', 'strictly serial'), 5)

        assert not outcome.recalled
        assert outcome.missing_needles == ('strictly serial',)
