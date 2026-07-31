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


_UNSET = object()


def _registry_payload(*entries: dict, entries_override=_UNSET) -> dict:
    """Build a registry payload.

    ``entries_override`` is the door to shapes the varargs form cannot
    express — notably an explicitly EMPTY entries list, which the default
    (substitute one entry when none is passed) would otherwise hide.
    """
    if entries_override is not _UNSET:
        return {'schema_version': 1, 'entries': entries_override}
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
    """A stand-in for MemoryResult: id, content, metadata, relevance_score.

    ``source_store`` is DECLARED but deliberately left unset by ``__init__``:
    a plain ``_R`` must lack the attribute at runtime so the store-disclosure
    tests still exercise the probe's ``getattr(..., 'source_store', '')``
    fallback path. ``_stored`` sets it when a test needs a served store.
    """

    source_store: str

    def __init__(self, content='', id='', metadata=None, relevance_score=0.0):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.relevance_score = relevance_score


def _entry(topic='t', content='the canonical text', last_known_id: str | None = 'ID-1', **kw):
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


# ---------------------------------------------------------------------------
# step-13: contamination, pure over synthetic result lists
#
# Contamination has to be WELL-POSED TODAY, before 3195/3201 widen
# `metadata.topic` coverage: the census measured 491 of 49,628 entries carrying
# a topic at all, so most results have none. A result is FOREIGN only when it
# carries a topic that is IN the registry and is not the probed topic.
# Everything else — no topic, or a topic the registry does not know — is
# counted as UNTOPICED and disclosed, never folded into the numerator. When
# 3201's retro stamping lands, the same code strictly widens the numerator's
# reach with no rewrite: that is the forward-compat property D5 asks for, and
# the test below asserts it by adding a topic to the registry rather than by
# changing any code.
# ---------------------------------------------------------------------------

def _mini_registry(*topics):
    """A TopicRegistry over bare topic slugs (only the slugs matter here)."""
    m = _mod()
    return m.TopicRegistry(
        schema_version=1,
        entries=tuple(_entry(topic=t, content=f'canonical for {t}') for t in topics),
    )


def _topiced(topic, content='some content', id='X'):
    return _R(content=content, id=id, metadata={'topic': topic})


class TestClassifyContamination:
    """`classify_contamination(results, entry, registry, k)`."""

    def test_a_different_registry_topic_is_foreign_and_named(self):
        """(a) The record must name WHICH foreign topic bled in."""
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_topiced('escalation-ladder', id='F1')],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.foreign_count == 1
        assert len(outcome.foreign_records) == 1
        assert outcome.foreign_records[0].foreign_topic == 'escalation-ladder'
        assert outcome.foreign_records[0].rank == 1

    def test_the_probed_topic_itself_is_not_foreign(self):
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_topiced('merge-lane')], registry.by_topic['merge-lane'], registry, 5,
        )

        assert outcome.foreign_count == 0
        assert outcome.foreign_records == ()

    def test_a_result_with_no_topic_is_untopiced_not_foreign(self):
        """(c) Pre-3201 well-posedness: absence of a stamp is not contamination."""
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_R(content='no topic key at all', id='U1')],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.foreign_count == 0
        assert outcome.untopiced_count == 1

    def test_an_unregistered_topic_is_untopiced_not_foreign(self):
        """(c) 352 distinct topic values live; the registry knows ~32 of them."""
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_topiced('some-topic-the-registry-has-never-heard-of', id='U1')],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.foreign_count == 0
        assert outcome.untopiced_count == 1

    def test_the_untopiced_disclosure_is_non_zero_for_a_mostly_unstamped_list(self):
        """(d) No silent caps: the narrowing must be visible in the artifact.

        Most of the live corpus carries no topic, so if this count could
        silently read zero the contamination share would look authoritative
        while being computed over a handful of stamped results.
        """
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        results = [*_filler(4), _topiced('escalation-ladder', id='F1')]
        outcome = m.classify_contamination(
            results, registry.by_topic['merge-lane'], registry, 5,
        )

        assert outcome.untopiced_count == 4
        assert outcome.foreign_count == 1
        assert outcome.scored_total == 5

    def test_hyphen_and_underscore_spellings_are_the_same_topic(self):
        """(e) The guard slug and the live metadata value differ by - vs _."""
        m = _mod()
        registry = _mini_registry(
            'architect-report-task-already-done-main-reachability', 'merge-lane',
        )
        probed = registry.by_topic['architect-report-task-already-done-main-reachability']
        outcome = m.classify_contamination(
            [_topiced('architect_report_task_already_done_main_reachability')],
            probed,
            registry,
            5,
        )

        assert outcome.foreign_count == 0, 'the same topic, spelled the live way'
        assert outcome.untopiced_count == 0, 'it IS registered — just underscored'

    def test_underscored_foreign_topic_is_still_foreign(self):
        """(e) The same fold must not make a genuinely foreign topic vanish."""
        m = _mod()
        registry = _mini_registry(
            'architect-report-task-already-done-main-reachability', 'merge-lane',
        )
        outcome = m.classify_contamination(
            [_topiced('architect_report_task_already_done_main_reachability')],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.foreign_count == 1

    def test_classification_is_confined_to_the_top_k_slice(self):
        """(f) Same parameterised k as every other metric here."""
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        probed = registry.by_topic['merge-lane']
        results = [*_filler(5), _topiced('escalation-ladder', id='F1')]

        at_five = m.classify_contamination(results, probed, registry, 5)
        at_ten = m.classify_contamination(results, probed, registry, 10)

        assert at_five.foreign_count == 0
        assert at_five.scored_total == 5
        assert at_ten.foreign_count == 1
        assert at_ten.scored_total == 6

    def test_widening_the_registry_reclassifies_without_a_code_change(self):
        """(g) The D5 forward-compat property, asserted as a flip.

        The ONLY thing that changes between the two calls is the registry —
        3201's retro stamping widens exactly this set, and the same code then
        sees contamination it could not previously name.
        """
        m = _mod()
        results = [_topiced('escalation-ladder', id='F1')]

        narrow = _mini_registry('merge-lane')
        before = m.classify_contamination(
            results, narrow.by_topic['merge-lane'], narrow, 5,
        )

        wide = _mini_registry('merge-lane', 'escalation-ladder')
        after = m.classify_contamination(results, wide.by_topic['merge-lane'], wide, 5)

        assert before.foreign_count == 0
        assert before.untopiced_count == 1
        assert after.foreign_count == 1
        assert after.untopiced_count == 0

    def test_foreign_records_carry_the_probed_topic_for_the_report(self):
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_topiced('escalation-ladder', id='F1')],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.foreign_records[0].topic == 'merge-lane'

    def test_an_empty_result_list_scores_nothing(self):
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [], registry.by_topic['merge-lane'], registry, 5,
        )

        assert outcome.scored_total == 0
        assert outcome.foreign_count == 0
        assert outcome.untopiced_count == 0

    def test_a_non_string_topic_value_is_untopiced_not_a_crash(self):
        """Live metadata is not schema-enforced; a list-valued topic must not
        take the run down, nor be silently read as the probed topic."""
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        outcome = m.classify_contamination(
            [_R(content='x', id='B1', metadata={'topic': ['merge-lane']})],
            registry.by_topic['merge-lane'],
            registry,
            5,
        )

        assert outcome.untopiced_count == 1
        assert outcome.foreign_count == 0

    def test_the_outcome_unpacks_as_the_documented_four_tuple(self):
        m = _mod()
        registry = _mini_registry('merge-lane', 'escalation-ladder')
        foreign_records, foreign_count, untopiced_count, scored_total = (
            m.classify_contamination(
                [_topiced('escalation-ladder', id='F1'), *_filler(2)],
                registry.by_topic['merge-lane'],
                registry,
                5,
            )
        )

        assert len(foreign_records) == foreign_count == 1
        assert untopiced_count == 2
        assert scored_total == 3


# ---------------------------------------------------------------------------
# step-15: M1 series assembly
#
# The artifact is the ONLY surface leaf alpha's evaluator and the dashboard
# read, and they join a run to its baseline window BY metric_id. A wrong
# spelling here does not crash anything — it makes the metric invisible, which
# is strictly worse. Hence the vocabulary is asserted exactly, including the
# negative: gamma's E4 metrics must not appear.
#
# Still no rate assertions: every number below is a count, a denominator, or a
# structural identity (value x denominator is a whole number of successes).
# ---------------------------------------------------------------------------

def _phrasing_obs(topic, phrasing='q', *, k=5, hit=True, held_out=False, degraded=False):
    m = _mod()
    return m.PhrasingObservation(
        topic=topic,
        phrasing=phrasing,
        held_out=held_out,
        k=k,
        hit=hit,
        rank=1 if hit else None,
        matched_by='content_hash' if hit else None,
        degraded=degraded,
    )


def _claim_obs(topic, query='c', *, k=5, recalled=True, scorable=True, degraded=False):
    m = _mod()
    return m.ClaimObservation(
        topic=topic,
        query=query,
        k=k,
        recalled=recalled,
        missing_needles=() if recalled else ('a needle',),
        scorable=scorable,
        degraded=degraded,
    )


def _contam_obs(topic, *, foreign=0, untopiced=0, scored=5, degraded=False):
    m = _mod()
    return m.ContaminationObservation(
        topic=topic,
        phrasing='q',
        k=5,
        foreign_count=foreign,
        untopiced_count=untopiced,
        scored_total=scored,
        foreign_records=(),
        degraded=degraded,
    )


def _inversion_obs(topic, *, pairs=2, inversions=0, degraded=False):
    m = _mod()
    return m.InversionObservation(
        topic=topic,
        phrasing='q',
        pairs_examined=pairs,
        inversions=tuple(
            m.InversionRecord(
                topic=topic, phrasing='q',
                superseded_hash='a' * 16, successor_hash='b' * 16,
                superseded_rank=1, successor_rank=2,
            )
            for _ in range(inversions)
        ),
        degraded=degraded,
    )


def _observations(*, topics=('alpha-topic', 'beta-topic'), ks=(5, 10), miss_held_out=()):
    """A complete, self-consistent observation set over *topics*.

    Each topic gets two tuned phrasings and one held-out phrasing at every k,
    one claim query, one contamination sample and one inversion opportunity.
    Topics named in *miss_held_out* miss on their held-out phrasing only — the
    Goodhart guard's failure mode, and the only way a topic in this fixture can
    fail its tripwire.
    """
    m = _mod()
    phrasings = []
    for topic in topics:
        for k in ks:
            phrasings.append(_phrasing_obs(topic, 'tuned one', k=k))
            phrasings.append(_phrasing_obs(topic, 'tuned two', k=k))
            phrasings.append(_phrasing_obs(
                topic, 'held out', k=k, held_out=True, hit=topic not in miss_held_out,
            ))
    return m.ProbeObservations(
        phrasings=phrasings,
        claims=[_claim_obs(t) for t in topics],
        contamination=[_contam_obs(t, foreign=1, untopiced=3, scored=5) for t in topics],
        inversions=[_inversion_obs(t) for t in topics],
    )


def _build(observations=None, *, counts=None, project_id='dark_factory', stamp='20260730T101500Z',
           ks=(5, 10)):
    return _mod().build_series(
        observations if observations is not None else _observations(),
        counts if counts is not None else {'topics': 2, 'phrasings': 6},
        project_id,
        stamp,
        ks,
    )


class TestBuildSeries:
    """`build_series(observations, corpus_counts, project_id, stamp, ks)`."""

    def test_the_series_validates_unchanged(self):
        """(a) The artifact must satisfy the shared schema as built."""
        from shared.memory_eval_metrics import validate_metric_series  # noqa: PLC0415

        validate_metric_series(_build())

    def test_it_carries_exactly_the_seven_metric_ids_this_leaf_owns(self):
        """(b) The join key. A wrong spelling is invisible, not loud."""
        ids = {metric.metric_id for metric in _build().metrics}

        assert ids == {
            'topic-canonical-present',
            'canonical-in-top-5',
            'canonical-in-top-10',
            'canonical-in-top-5-held-out',
            'claim-recall',
            'contamination-share',
            'superseded-above-successor',
        }

    def test_it_does_not_carry_gammas_e4_metrics(self):
        """(b) E4 is leaf gamma's, and it depends on 3196 — not this runner."""
        ids = {metric.metric_id for metric in _build().metrics}

        assert 'dangling-pointers' not in ids
        assert 'successor-pointer-present' not in ids

    def test_the_series_is_stamped_with_this_evals_id(self):
        series = _build(stamp='20260730T101500Z')

        assert series.eval_id == 'e1-retrieval-health'
        assert series.run_stamp == '20260730T101500Z'
        assert series.schema_version == 1

    def test_kinds_and_directions(self):
        """(c) The shared validator rejects a missing or misplaced direction."""
        by_id = {m.metric_id: m for m in _build().metrics}

        assert by_id['topic-canonical-present'].kind == 'tripwire'
        assert by_id['topic-canonical-present'].direction is None
        for metric_id in (
            'canonical-in-top-5', 'canonical-in-top-10',
            'canonical-in-top-5-held-out', 'claim-recall',
        ):
            assert by_id[metric_id].kind == 'proportion'
            assert by_id[metric_id].direction == 'lower_is_worse'
        assert by_id['contamination-share'].kind == 'proportion'
        assert by_id['contamination-share'].direction == 'higher_is_worse'
        assert by_id['superseded-above-successor'].kind == 'count'
        assert by_id['superseded-above-successor'].direction == 'higher_is_worse'

    def test_tripwire_items_are_one_per_topic_keyed_by_slug(self):
        """(d) The item_key shape alpha's grandfather set persists."""
        tripwire = {m.metric_id: m for m in _build().metrics}['topic-canonical-present']
        keys = {item.item_key for item in tripwire.items}

        assert keys == {'t-alpha-topic', 't-beta-topic'}
        assert tripwire.n == len(tripwire.items) == 2

    def test_a_topic_fails_its_tripwire_when_only_the_held_out_phrasing_misses(self):
        """(d) Every phrasing INCLUDING the held-out one must find the canonical.

        This is the Goodhart guard as a predicate: tuning the two known
        phrasings until they pass cannot make the topic pass.
        """
        series = _build(_observations(miss_held_out=('beta-topic',)))
        tripwire = {m.metric_id: m for m in series.metrics}['topic-canonical-present']
        passed = {item.item_key: item.passed for item in tripwire.items}

        assert passed['t-alpha-topic']
        assert not passed['t-beta-topic']
        assert tripwire.value == 1, 'a tripwire value IS its failure count'

    def test_tripwire_is_evaluated_at_k_five(self):
        """(d) Pinned at TRIPWIRE_K, matching alpha's committed exemplar."""
        m = _mod()
        observations = m.ProbeObservations(
            phrasings=[
                # Hits at k=10 only: rank 7 for every phrasing.
                _phrasing_obs('alpha-topic', 'p', k=5, hit=False),
                _phrasing_obs('alpha-topic', 'h', k=5, hit=False, held_out=True),
                _phrasing_obs('alpha-topic', 'p', k=10, hit=True),
                _phrasing_obs('alpha-topic', 'h', k=10, hit=True, held_out=True),
            ],
        )
        tripwire = {
            metric.metric_id: metric for metric in _build(observations).metrics
        }['topic-canonical-present']

        assert tripwire.value == 1, 'k=10 hits must not rescue a k=5 tripwire'

    def test_proportion_denominators_are_their_pair_counts(self):
        """(e) n == denominator == the number of trials that were scored."""
        by_id = {m.metric_id: m for m in _build().metrics}

        # 2 topics x 3 phrasings at each k.
        assert by_id['canonical-in-top-5'].denominator == 6
        assert by_id['canonical-in-top-5'].n == 6
        assert by_id['canonical-in-top-10'].denominator == 6
        # 2 topics x 1 held-out phrasing at k=5.
        assert by_id['canonical-in-top-5-held-out'].denominator == 2
        # 2 topics x 1 claim query.
        assert by_id['claim-recall'].denominator == 2
        # 2 contamination samples x 5 scored results each.
        assert by_id['contamination-share'].denominator == 10

    def test_every_proportion_is_a_whole_number_of_successes(self):
        """(e) The shared validator's rule, asserted on the built artifact."""
        for metric in _build().metrics:
            if metric.kind != 'proportion':
                continue
            successes = metric.value * metric.denominator
            assert abs(successes - round(successes)) < 1e-9, metric.metric_id

    def test_the_count_metric_carries_its_exposure_as_n(self):
        series = _build(_mod().ProbeObservations(
            phrasings=[_phrasing_obs('alpha-topic', 'h', held_out=True)],
            inversions=[
                _inversion_obs('alpha-topic', pairs=3, inversions=2),
                _inversion_obs('beta-topic', pairs=1, inversions=0),
            ],
        ))
        count = {m.metric_id: m for m in series.metrics}['superseded-above-successor']

        assert count.value == 2
        assert count.n == 4, 'n is the pairs examined — the exposure the rate is per'

    def test_corpus_carries_the_project_and_the_counts(self):
        """(f) The per-category mapping, passed through."""
        series = _build(counts={'topics': 32, 'entries_probed': 96}, project_id='reify')

        assert series.corpus.project_id == 'reify'
        assert series.corpus.counts['topics'] == 32
        assert series.corpus.counts['entries_probed'] == 96

    def test_the_untopiced_disclosure_reaches_the_artifact(self):
        """The step-13 disclosure must survive into the machine-readable surface.

        A contamination share whose unclassifiable remainder is only mentioned
        in prose is a silent cap for every consumer that reads the JSON.
        """
        counts = _build().corpus.counts

        assert counts['contamination_untopiced_results'] == 6
        assert counts['contamination_scored_results'] == 10

    def test_degraded_observations_are_excluded_from_denominators(self):
        """A store outage must not be charged as a findability failure."""
        m = _mod()
        series = _build(m.ProbeObservations(
            phrasings=[
                _phrasing_obs('alpha-topic', 'ok', hit=True),
                _phrasing_obs('alpha-topic', 'h', hit=True, held_out=True),
                _phrasing_obs('alpha-topic', 'down', hit=False, degraded=True),
            ],
        ))
        by_id = {metric.metric_id: metric for metric in series.metrics}

        assert by_id['canonical-in-top-5'].denominator == 2
        tripwire = by_id['topic-canonical-present']
        assert [item.passed for item in tripwire.items] == [True]

    def test_a_metric_with_no_scored_trials_is_absent_not_zero(self):
        """An absent proportion is honest; a 0/0 one would be a fabricated trial."""
        series = _build(_mod().ProbeObservations(
            phrasings=[_phrasing_obs('alpha-topic', 'h', held_out=True)],
        ))
        ids = {metric.metric_id for metric in series.metrics}

        assert 'claim-recall' not in ids
        assert 'contamination-share' not in ids
        assert 'canonical-in-top-5' in ids

    def test_unscorable_claims_are_disclosed_and_excluded(self):
        series = _build(_mod().ProbeObservations(
            phrasings=[_phrasing_obs('alpha-topic', 'h', held_out=True)],
            claims=[
                _claim_obs('alpha-topic', 'scorable', recalled=True),
                _claim_obs('alpha-topic', 'needle-less', recalled=False, scorable=False),
            ],
        ))
        by_id = {metric.metric_id: metric for metric in series.metrics}

        assert by_id['claim-recall'].denominator == 1
        assert series.corpus.counts['claim_queries_unscorable'] == 1

    def test_the_tripwire_points_at_the_report_beside_it(self):
        """An operator following an artifact must reach the prose."""
        from shared.memory_eval_metrics import report_artifact_path  # noqa: PLC0415

        series = _build(stamp='20260730T101500Z')
        tripwire = {m.metric_id: m for m in series.metrics}['topic-canonical-present']

        assert tripwire.details_path == report_artifact_path(
            '.', 'e1-retrieval-health', '20260730T101500Z',
        ).name

    def test_an_inconsistent_tripwire_is_rejected_at_emit_time(self, tmp_path):
        """(g) M1: malformed metrics are rejected by the producer, not the reader.

        By the time the dashboard reads the artifact there is nobody left to
        tell, so the tamper below must not be writable to disk.
        """
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            MetricSchemaError,
            serialize_metric_series,
            write_metric_series,
        )

        payload = json.loads(serialize_metric_series(_build(
            _observations(miss_held_out=('beta-topic',)),
        )))
        for metric in payload['metrics']:
            if metric['metric_id'] == 'topic-canonical-present':
                metric['value'] = 0  # it has one failing item

        with pytest.raises(MetricSchemaError):
            write_metric_series(payload, tmp_path)
        assert not (tmp_path / 'e1-retrieval-health').exists(), 'no partial artifact'

    def test_build_series_validates_its_own_output(self):
        """A bug in aggregation must surface here, not in the evaluator."""
        from shared.memory_eval_metrics import validate_metric_series  # noqa: PLC0415

        for observations in (
            _observations(),
            _observations(miss_held_out=('alpha-topic', 'beta-topic')),
            _mod().ProbeObservations(phrasings=[_phrasing_obs('a', 'h', held_out=True)]),
        ):
            validate_metric_series(_build(observations))

    def test_round_trip_through_the_artifact_is_lossless(self, tmp_path):
        """(h) Written where M1 says, and read back identical."""
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            load_metric_series,
            write_metric_series,
        )

        series = _build(stamp='20260730T101500Z')
        metrics_path, report_path = write_metric_series(series, tmp_path)

        assert metrics_path == (
            tmp_path / 'e1-retrieval-health' / 'metrics-20260730T101500Z.json'
        )
        assert report_path.parent == metrics_path.parent
        assert report_path.exists()
        assert load_metric_series(metrics_path) == series


# ---------------------------------------------------------------------------
# step-17: degraded-search disclosure
#
# MemoryService.search returns a SearchResults (a list subclass) carrying
# degraded / failed_stores / failure_diagnostics, and its own docstring warns
# that those attributes do NOT survive slicing, sorted(), concatenation or a
# list comprehension — such a transform returns a plain list and drops them
# silently. A probe that sliced to top-k first would therefore see a healthy
# empty result set during a Qdrant outage and report a corpus-wide canonical
# collapse. These tests use the REAL SearchResults so that failure mode is
# exercised against the real semantics, not a double's imitation of them.
# ---------------------------------------------------------------------------

from fused_memory.services.memory_service import SearchResults  # noqa: E402


def _degraded(results=(), *, stores=('mem0',), diagnostics=None):
    return SearchResults(
        results,
        degraded=True,
        failed_stores=list(stores),
        failure_diagnostics=list(diagnostics or [
            {'store': 'mem0', 'error_type': 'TimeoutError', 'project_id': 'dark_factory'},
        ]),
    )


def _healthy(results=()):
    return SearchResults(results)


def _probe_entry(topic='alpha-topic', content=CANON, **kw):
    """A registry entry with one tuned and one held-out phrasing plus a claim."""
    m = _mod()
    base = _entry(topic=topic, content=content, **kw)
    return m.RegistryEntry(
        topic=base.topic,
        project_id=base.project_id,
        derived_from=base.derived_from,
        canonical=base.canonical,
        phrasings=(m.Phrasing('tuned query', False), m.Phrasing('held out query', True)),
        claim_queries=(m.ClaimQuery(query='claim query', needles=('canonical text',)),),
    )


def _search_returning(by_query, default_factory=_healthy):
    """An async search double: query -> SearchResults."""
    async def search(query, limit):
        if query in by_query:
            return by_query[query]
        return default_factory()

    return search


class TestDegradedSearchDisclosure:
    """The per-query probe band must never charge an outage as a regression."""

    @pytest.mark.asyncio
    async def test_a_degraded_query_marks_every_observation_it_produced(self):
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning({'tuned query': _degraded()})

        await m.probe_topic(search, entry, registry, (5,), observations)

        by_phrasing = {o.phrasing: o for o in observations.phrasings}
        assert by_phrasing['tuned query'].degraded
        assert not by_phrasing['held out query'].degraded

    @pytest.mark.asyncio
    async def test_degraded_queries_are_recorded_with_stores_and_diagnostics(self):
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning({'tuned query': _degraded(
            stores=('mem0', 'graphiti'),
            diagnostics=[{'store': 'mem0', 'error_type': 'TimeoutError'}],
        )})

        await m.probe_topic(search, entry, registry, (5,), observations)

        assert len(observations.degraded_queries) == 1
        record = observations.degraded_queries[0]
        assert record.topic == 'alpha-topic'
        assert record.query == 'tuned query'
        assert set(record.failed_stores) == {'mem0', 'graphiti'}
        assert record.diagnostics[0]['error_type'] == 'TimeoutError'

    @pytest.mark.asyncio
    async def test_a_degraded_query_is_excluded_from_every_denominator(self):
        """(a) The exclusion, end to end through build_series."""
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning(
            {'tuned query': _degraded()},
            default_factory=lambda: _healthy([_R(content=CANON, id='ID-1')]),
        )

        await m.probe_topic(search, entry, registry, (5,), observations)
        series = m.build_series(observations, {}, 'dark_factory', '20260730T101500Z', (5,))
        by_id = {metric.metric_id: metric for metric in series.metrics}

        assert by_id['canonical-in-top-5'].denominator == 1, 'only the healthy query'

    def test_degrade_metadata_is_read_before_any_slicing(self):
        """(b) The documented memory_service.py:706-712 footgun, exercised.

        Slicing a SearchResults returns a plain list and silently drops the
        degrade attributes. Reading them off the object first is the only
        correct order, and this test fails for an implementation that slices
        first — the sliced value genuinely has no `degraded` to read.
        """
        m = _mod()
        raw = _degraded([_R(content='x') for _ in range(20)], stores=('graphiti',))

        assert not hasattr(raw[:5], 'degraded'), 'the footgun this test guards'
        info = m.read_degrade_metadata(raw)

        assert info.degraded
        assert info.failed_stores == ('graphiti',)
        assert len(info.results) == 20

    def test_a_plain_list_is_read_as_healthy(self):
        """Not every caller hands back a SearchResults; absence is not degradation."""
        info = _mod().read_degrade_metadata([_R(content='x')])

        assert not info.degraded
        assert info.failed_stores == ()

    @pytest.mark.asyncio
    async def test_a_wholly_degraded_topic_is_not_measured_not_failed(self):
        """(c) A store outage must not manufacture a tripwire failure.

        Leaf epsilon reads tripwire failures as candidate findings against the
        3111 lineage. An outage that presented as 32 failing topics would be a
        fabricated corpus-wide collapse.
        """
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()

        await m.probe_topic(
            search=_search_returning({}, default_factory=_degraded),
            entry=entry, registry=registry, ks=(5,), observations=observations,
        )
        series = m.build_series(observations, {}, 'dark_factory', '20260730T101500Z', (5,))
        ids = {metric.metric_id for metric in series.metrics}

        assert 'topic-canonical-present' not in ids, 'no item, so no tripwire'
        assert m.not_measured_topics(observations) == ['alpha-topic']

    @pytest.mark.asyncio
    async def test_a_wholly_degraded_run_still_emits_a_valid_artifact(self, tmp_path):
        """(d) Silence is never a healthy signal."""
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            load_metric_series,
            validate_metric_series,
        )

        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()

        await m.probe_topic(
            search=_search_returning({}, default_factory=_degraded),
            entry=entry, registry=registry, ks=(5,), observations=observations,
        )
        series = m.build_series(observations, {}, 'dark_factory', '20260730T101500Z', (5,))
        validate_metric_series(series)
        metrics_path, _ = m.emit_series(series, tmp_path)

        assert load_metric_series(metrics_path) == series
        assert series.corpus.counts['degraded_observations'] > 0

    @pytest.mark.asyncio
    async def test_the_report_names_the_failed_stores_and_the_not_measured_topics(self):
        """(a)+(c)+(d): the prose has to say what the numbers cannot."""
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()

        await m.probe_topic(
            search=_search_returning({}, default_factory=lambda: _degraded(
                stores=('qdrant',),
                diagnostics=[{'store': 'qdrant', 'error_type': 'ConnectionRefusedError'}],
            )),
            entry=entry, registry=registry, ks=(5,), observations=observations,
        )
        series = m.build_series(observations, {}, 'dark_factory', '20260730T101500Z', (5,))
        report = m.render_probe_report(series, observations)

        assert 'qdrant' in report
        assert 'ConnectionRefusedError' in report
        assert 'alpha-topic' in report
        assert 'not measured' in report.lower()

    @pytest.mark.asyncio
    async def test_a_healthy_run_produces_observations_for_every_family(self):
        """The band's happy path: phrasings, claims, contamination, inversions."""
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning(
            {}, default_factory=lambda: _healthy([_R(content=CANON, id='ID-1')]),
        )

        await m.probe_topic(search, entry, registry, (5, 10), observations)

        assert {o.k for o in observations.phrasings} == {5, 10}
        assert len(observations.phrasings) == 4, 'two phrasings x two k values'
        assert [o.query for o in observations.claims] == ['claim query']
        assert observations.claims[0].recalled
        assert [c.topic for c in observations.contamination] == ['alpha-topic'] * 2
        assert [i.topic for i in observations.inversions] == ['alpha-topic'] * 2
        assert observations.degraded_queries == []

    @pytest.mark.asyncio
    async def test_a_degraded_claim_query_is_excluded_too(self):
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning(
            {'claim query': _degraded()},
            default_factory=lambda: _healthy([_R(content=CANON, id='ID-1')]),
        )

        await m.probe_topic(search, entry, registry, (5,), observations)
        series = m.build_series(observations, {}, 'dark_factory', '20260730T101500Z', (5,))
        ids = {metric.metric_id for metric in series.metrics}

        assert observations.claims[0].degraded
        assert 'claim-recall' not in ids, 'no scorable claim trial remains'

    @pytest.mark.asyncio
    async def test_the_search_is_asked_for_the_widest_k(self):
        """One search per query, sliced per k — probing twice would double the
        embedding spend and could return two different result lists."""
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        seen: list[tuple[str, int]] = []

        async def search(query, limit):
            seen.append((query, limit))
            return _healthy([_R(content=CANON, id='ID-1')])

        await m.probe_topic(search, entry, registry, (5, 10), m.ProbeObservations())

        assert [limit for _, limit in seen] == [10, 10, 10]
        assert len(seen) == 3, 'two phrasings + one claim query, once each'


# ---------------------------------------------------------------------------
# step-19: the D1 initial-state / known-bad report
#
# D1: the first run is a BASELINE SNAPSHOT, not a day-one alarm source. The
# 3111/3112 fix lineage has been rewriting how the corpus is written for
# months, so whatever this probe finds on run one is inherited state, not a
# regression anybody introduced. The report says so in words and attributes it,
# and it stops there: turning that list into a ratchet baseline is leaf
# epsilon's job and every bound is leaf alpha's (G6). A runner that emitted
# either would put the limits in two places.
# ---------------------------------------------------------------------------

# The "this runner adjudicates nothing" invariant is STRUCTURAL and is pinned
# structurally: `test_it_carries_exactly_the_seven_metric_ids_this_leaf_owns`
# fixes the emitted vocabulary by equality, and
# `test_the_pinned_metric_set_is_a_superset_on_every_run` holds it across every
# --k. A banned-substring sweep over report prose was tried here and removed:
# it constrained wording rather than behaviour (it would have failed the very
# sentence "limits are set by leaf alpha, not here", which DISCLAIMS
# adjudication) while a report computing a pass rate under any other name would
# have sailed through it. Do not reintroduce it as a tightened list or a regex.


def _report_observations():
    """Observations exercising every disclosure the report must carry."""
    m = _mod()
    return m.ProbeObservations(
        phrasings=[
            _phrasing_obs('alpha-topic', 'tuned', hit=True),
            _phrasing_obs('alpha-topic', 'held out', hit=False, held_out=True),
            m.PhrasingObservation(
                topic='beta-topic', phrasing='reworded', held_out=False, k=5,
                hit=True, rank=2, matched_by='last_known_id',
            ),
            m.PhrasingObservation(
                topic='beta-topic', phrasing='held out', held_out=True, k=5,
                hit=False, rank=None, matched_by=None,
            ),
        ],
        claims=[_claim_obs('alpha-topic', 'a claim', recalled=False)],
        contamination=[_contam_obs('alpha-topic', foreign=1, untopiced=3, scored=5)],
        inversions=[m.InversionObservation(
            topic='alpha-topic', phrasing='tuned', pairs_examined=1,
            inversions=(m.InversionRecord(
                topic='alpha-topic', phrasing='tuned',
                superseded_hash='dead' * 4, successor_hash='beef' * 4,
                superseded_rank=1, successor_rank=4,
            ),),
        )],
    )


def _report_series(observations=None):
    return _build(observations if observations is not None else _report_observations())


class TestInitialRunDetection:
    """`is_initial_run(root)` — glob the eval dir through the shared helpers."""

    def test_an_empty_root_is_the_initial_run(self, tmp_path):
        assert _mod().is_initial_run(tmp_path)

    def test_a_root_with_a_prior_artifact_is_not(self, tmp_path):
        m = _mod()
        m.emit_series(_report_series(), tmp_path)

        assert not m.is_initial_run(tmp_path)

    def test_another_evals_artifacts_do_not_count(self, tmp_path):
        """Leaves beta/gamma/delta share one artifact root."""
        other = tmp_path / 'e4-pointer-integrity'
        other.mkdir(parents=True)
        (other / 'metrics-20260101T000000Z.json').write_text('{}', encoding='utf-8')

        assert _mod().is_initial_run(tmp_path)

    def test_a_report_without_its_metrics_does_not_count(self, tmp_path):
        """The metrics artifact is the series; the report is its companion."""
        eval_dir = tmp_path / 'e1-retrieval-health'
        eval_dir.mkdir(parents=True)
        (eval_dir / 'report-20260101T000000Z.txt').write_text('x', encoding='utf-8')

        assert _mod().is_initial_run(tmp_path)


def _count_line(report: str, label: str) -> str:
    """The single `  <label>...: <n>` line of a report breakdown.

    Keys on the metric term (`foreign`, `untopiced`) and reads the NUMBER off
    the end, so the assertion is about the count rather than the sentence the
    count sits in.
    """
    matches = [
        line.rstrip() for line in report.splitlines()
        if line.strip().startswith(label) and line.rstrip()[-1].isdigit()
    ]
    assert len(matches) == 1, f'expected one {label!r} count line, got {matches!r}'
    return matches[0]


def _caveat_anchor():
    """The routing caveat AS RENDERED, derived from the module's own constant.

    Deriving the needle instead of hand-copying a phrase is what keeps these
    tests about placement and scoping rather than about wording: rewording
    `_KNOWN_BAD_ROUTING_CAVEAT` moves the anchor with it.
    """
    m = _mod()
    return m._wrap(m._KNOWN_BAD_ROUTING_CAVEAT)[0]


class TestProbeReport:
    """`render_probe_report(series, observations, *, is_initial_run)`."""

    def test_the_initial_run_enumerates_known_bad_items(self):
        """(a) Every failing tripwire item key is named in the section."""
        report = _mod().render_probe_report(
            _report_series(), _report_observations(), is_initial_run=True,
        )

        assert 'initial state' in report.lower()
        assert 't-alpha-topic' in report
        assert 't-beta-topic' in report

    def test_the_known_bad_list_carries_the_routing_caveat_inline(self):
        """(a) esc-3208-1: the misreading happens at the headline, so the
        caveat has to be AT the known-bad list, not only in the store
        breakdown far below it. The first live baseline was 72/78 observations
        served by Graphiti against 72/78 unmatched — a reader who stops at the
        rate takes a router property for a findability collapse.

        The assertion is on POSITION: the caveat precedes the item keys it
        qualifies, so a reader cannot reach the list without passing it.
        """
        m = _mod()
        report = m.render_probe_report(
            _report_series(), _report_observations(), is_initial_run=True,
        )

        anchor = _caveat_anchor()
        preamble = m._wrap(m._KNOWN_BAD_PREAMBLE)[0]
        assert anchor in report

        # Inside the initial-state section (after its preamble), and above the
        # item keys it qualifies — the item keys also appear in the per-metric
        # block far above, so the check is on what follows the caveat.
        assert report.index(preamble) < report.index(anchor)
        after_caveat = report[report.index(anchor):]
        assert '  - t-alpha-topic' in after_caveat
        assert '  - t-beta-topic' in after_caveat

    def test_the_routing_caveat_is_scoped_to_the_initial_run(self):
        """It explains an inherited snapshot; on run fifty it would be noise."""
        report = _mod().render_probe_report(
            _report_series(), _report_observations(), is_initial_run=False,
        )

        assert _caveat_anchor() not in report

    def test_a_later_run_has_no_initial_state_section(self):
        """(b) Inherited state is only inherited once."""
        report = _mod().render_probe_report(
            _report_series(), _report_observations(), is_initial_run=False,
        )

        assert 'initial state' not in report.lower()
        assert 'known-bad' not in report.lower()

    def test_initial_state_defaults_off(self):
        """A caller that forgot the flag must not fabricate a first run."""
        report = _mod().render_probe_report(_report_series(), _report_observations())

        assert 'initial state' not in report.lower()

    def test_the_report_carries_the_matched_by_breakdown(self):
        """(d) step-7's dual matcher, made visible — with its COUNTS.

        The fixture scores four phrasings: one matched by content_hash, one by
        last_known_id, two unmatched. Asserting the tallies is what makes this
        cover the breakdown; the header alone renders whatever the numbers say.
        """
        report = _mod().render_probe_report(_report_series(), _report_observations())

        assert 'content_hash: 1' in report
        assert 'last_known_id: 1' in report
        assert 'unmatched: 2' in report

    def test_the_report_carries_the_untopiced_disclosure(self):
        """(d) step-13: the share's unclassifiable remainder, with its counts.

        The fixture's single contamination observation scored 5 results: 1
        foreign, 3 untopiced. Untopiced is reported SEPARATELY from foreign —
        folding it in would measure stamping coverage, not contamination — so
        the two numbers have to come back distinct.
        """
        report = _mod().render_probe_report(_report_series(), _report_observations())

        assert 'scored results: 5' in report
        foreign_line = _count_line(report, 'foreign')
        untopiced_line = _count_line(report, 'untopiced')

        assert foreign_line.endswith(': 1')
        assert untopiced_line.endswith(': 3')

    def test_the_report_carries_the_registry_composition(self):
        """(d) step-5: what derivation covered, and what it left out."""
        m = _mod()
        registry = m.TopicRegistry(
            schema_version=1,
            entries=(_entry(topic='alpha-topic'), _entry(topic='beta-topic')),
            disclosures={'census_topics_skipped_singleton': 41},
        )
        report = m.render_probe_report(
            _report_series(), _report_observations(), registry=registry,
        )

        assert 'census_topics_skipped_singleton' in report
        assert '41' in report
        assert 'hand' in report, 'the derived_from composition'

    def test_the_report_names_each_inversion(self):
        """(d) step-9: a bare count names no pair to go and look at."""
        report = _mod().render_probe_report(_report_series(), _report_observations())

        assert 'dead' * 4 in report
        assert 'beef' * 4 in report

    def test_the_shared_report_format_is_included_not_reimplemented(self):
        """(e) The M1 companion stays the shared module's format."""
        from shared.memory_eval_metrics import render_report  # noqa: PLC0415

        series = _report_series()
        shared_text = render_report(series)
        report = _mod().render_probe_report(series, _report_observations())

        assert report.startswith(shared_text.rstrip('\n'))

    def test_the_report_still_names_degraded_queries(self):
        """The step-17 section must survive the step-19 extension."""
        m = _mod()
        observations = _report_observations()
        observations.degraded_queries.append(m.DegradedQuery(
            topic='alpha-topic', query='tuned',
            failed_stores=('qdrant',),
            diagnostics=({'store': 'qdrant', 'error_type': 'ConnectionRefusedError'},),
        ))
        report = m.render_probe_report(_report_series(observations), observations)

        assert 'qdrant' in report
        assert 'ConnectionRefusedError' in report


# ---------------------------------------------------------------------------
# step-21: the read-only guarantee and the CLI band
#
# The read-only claim is the load-bearing one in this whole leaf: an eval that
# writes to the corpus it measures is not an eval. Asserting it in a docstring
# proves nothing, so it is asserted as BEHAVIOUR — the probe is driven, end to
# end through argparse and _run, against a MemoryService double whose every
# write method raises. A run that completes is a run that never wrote.
#
# Still no thresholds: every assertion below is on a call, a flag, an exit
# code or a filename.
# ---------------------------------------------------------------------------

def _canned(*results):
    """A healthy SearchResults carrying *results*."""
    return SearchResults(list(results))


class _ReadOnlyViolation(AssertionError):
    """Raised by the double when the probe touches a write path."""


class _ServiceDouble:
    """A MemoryService stand-in that cannot be written to.

    Every mutating method raises. ``search`` replays canned results and
    ``count_memories_by_metadata`` records its calls so the corpus-counting
    band can be asserted on rather than guessed at.
    """

    def __init__(self, by_query=None, counts=None, default=None):
        self._by_query = dict(by_query or {})
        self._counts = dict(counts or {})
        self._default = default
        self.searches: list[tuple[str, str, int]] = []
        self.count_calls: list[tuple[str, dict]] = []
        self.initialized = False
        self.closed = False

    # -- the two read paths the probe is allowed to use --------------------
    async def search(self, query, project_id='main', limit=10, **kwargs):
        self.searches.append((query, project_id, limit))
        if query in self._by_query:
            return self._by_query[query]
        return self._default() if self._default else _canned()

    async def count_memories_by_metadata(self, project_id, filters):
        self.count_calls.append((project_id, dict(filters)))
        return self._counts.get(filters.get('category'), 0)

    # -- lifecycle ---------------------------------------------------------
    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.closed = True

    # -- every write path is a tripwire ------------------------------------
    async def add_memory(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called add_memory')

    async def add_episode(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called add_episode')

    async def add_system_record(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called add_system_record')

    async def delete_memory(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called delete_memory')

    async def delete_episode(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called delete_episode')

    async def update_edge(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called update_edge')

    async def merge_entities(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called merge_entities')

    async def delete_entity(self, *a, **kw):
        raise _ReadOnlyViolation('the probe called delete_entity')


def _as_payload(registry) -> dict:
    """Serialize a TopicRegistry back to its on-disk shape.

    Round-tripping through the real loader is what makes the CLI tests
    end-to-end: a registry built in memory and one read off disk must be the
    same object, or `main()` is being tested against a shape `--registry`
    can never actually produce.
    """
    return {
        'schema_version': registry.schema_version,
        'entries': [
            {
                'topic': e.topic,
                'project_id': e.project_id,
                'derived_from': e.derived_from,
                'canonical': {
                    'content_hash': e.canonical.content_hash,
                    'content_prefix': e.canonical.content_prefix,
                    'last_known_id': e.canonical.last_known_id,
                },
                'phrasings': [
                    {'text': p.text, 'held_out': p.held_out} for p in e.phrasings
                ],
                'claim_queries': [
                    {'query': c.query, 'needles': list(c.needles)}
                    for c in e.claim_queries
                ],
                'members': list(e.members),
                'supersedes_pairs': [
                    {'superseded_hash': s.superseded_hash,
                     'successor_hash': s.successor_hash}
                    for s in e.supersedes_pairs
                ],
            }
            for e in registry.entries
        ],
    }


def _scoped_entry(topic, *, project_id='dark_factory'):
    """A registry entry whose query texts are unique to its topic.

    The step-17 helpers deliberately share query strings across topics; here
    they must not, because these tests assert WHICH project each individual
    query was scoped to.
    """
    m = _mod()
    content = f'the {topic} canonical text'
    return m.RegistryEntry(
        topic=topic,
        project_id=project_id,
        derived_from='hand',
        canonical=m.Canonical(
            content_hash=m.content_key(content),
            content_prefix=content,
            last_known_id=f'ID-{topic}',
        ),
        phrasings=(
            m.Phrasing(f'{topic} tuned', False),
            m.Phrasing(f'{topic} held out', True),
        ),
        claim_queries=(m.ClaimQuery(query=f'{topic} claim', needles=('canonical text',)),),
    )


def _probe_registry(*, project_id='dark_factory'):
    """A two-topic registry whose canonicals are findable in the canned results."""
    m = _mod()
    return m.TopicRegistry(schema_version=1, entries=(
        _scoped_entry('alpha-topic', project_id=project_id),
        _scoped_entry('beta-topic', project_id=project_id),
    ))


def _canned_hits(registry):
    """Canned SearchResults returning each topic's canonical for every query."""
    by_query = {}
    for entry in registry.entries:
        hit = _R(content=entry.canonical.content_prefix, id=entry.canonical.last_known_id)
        for phrasing in entry.phrasings:
            by_query[phrasing.text] = _canned(hit, *_filler(4))
        for claim in entry.claim_queries:
            by_query[claim.query] = _canned(hit, *_filler(4))
    return by_query


def _install_double(monkeypatch, double):
    """Point the lazily-imported MemoryService at *double*.

    No test-only seam in the script: `_run` imports MemoryService inside the
    function (the D8 pattern), so patching the module attribute is enough to
    drive the real argparse/_run/emit path end to end.
    """
    import fused_memory.services.memory_service as ms  # noqa: PLC0415

    monkeypatch.setattr(ms, 'MemoryService', lambda config: double)


class TestReadOnlyGuarantee:
    """(a) The probe completes against a service that cannot be written to."""

    def test_a_full_run_never_touches_a_write_path(self, monkeypatch, tmp_path):
        m = _mod()
        registry = _probe_registry()
        registry_path = tmp_path / 'registry.json'
        registry_path.write_text(json.dumps(_as_payload(registry)), encoding='utf-8')
        double = _ServiceDouble(by_query=_canned_hits(registry))
        _install_double(monkeypatch, double)
        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', '20260730T090000Z')

        code = m.main([
            '--registry', str(registry_path),
            '--out-root', str(tmp_path / 'out'),
            '--project-id', 'dark_factory',
        ])

        assert code == 0
        assert double.initialized and double.closed
        artifact = tmp_path / 'out' / 'e1-retrieval-health' / 'metrics-20260730T090000Z.json'
        assert artifact.exists()

    def test_the_double_would_have_caught_a_write(self, monkeypatch, tmp_path):
        """The guarantee is only worth what the double's tripwires are worth."""
        double = _ServiceDouble()

        import asyncio  # noqa: PLC0415

        for name in (
            'add_memory', 'add_episode', 'add_system_record', 'delete_memory',
            'delete_episode', 'update_edge', 'merge_entities', 'delete_entity',
        ):
            with pytest.raises(_ReadOnlyViolation):
                asyncio.run(getattr(double, name)())

    def test_the_emitted_artifact_validates_and_the_report_lands(
        self, monkeypatch, tmp_path,
    ):
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            load_metric_series,
        )

        m = _mod()
        registry = _probe_registry()
        double = _ServiceDouble(by_query=_canned_hits(registry))

        import asyncio  # noqa: PLC0415

        outcome = asyncio.run(m.run_probe(
            double, registry,
            project_ids=('dark_factory',),
            ks=(5, 10),
            out_root=tmp_path,
            stamp='20260730T091500Z',
        ))

        series = load_metric_series(outcome.metrics_path)
        assert series.eval_id == m.EVAL_ID
        assert outcome.report_path.exists()
        assert outcome.report_path.read_text(encoding='utf-8').strip()

    def test_the_probed_project_id_is_recorded_on_the_corpus(self, tmp_path):
        """An ephemeral run must never be mistakable for a live one."""
        m = _mod()
        registry = _probe_registry(project_id='_test_mem0_qdrant_integration_gw0')
        double = _ServiceDouble(by_query=_canned_hits(registry))

        import asyncio  # noqa: PLC0415

        outcome = asyncio.run(m.run_probe(
            double, registry,
            project_ids=('_test_mem0_qdrant_integration_gw0',),
            ks=(5,),
            out_root=tmp_path,
            stamp='20260730T092000Z',
        ))

        assert outcome.series.corpus.project_id == '_test_mem0_qdrant_integration_gw0'

    def test_the_search_is_scoped_to_each_entrys_project(self, tmp_path):
        m = _mod()
        registry = m.TopicRegistry(schema_version=1, entries=(
            _scoped_entry('alpha-topic', project_id='dark_factory'),
            _scoped_entry('beta-topic', project_id='reify'),
        ))
        double = _ServiceDouble(by_query=_canned_hits(registry))

        import asyncio  # noqa: PLC0415

        asyncio.run(m.run_probe(
            double, registry,
            project_ids=('dark_factory', 'reify'),
            ks=(5,),
            out_root=tmp_path,
            stamp='20260730T092500Z',
        ))

        scoped = {q: pid for q, pid, _ in double.searches}
        assert scoped['alpha-topic tuned'] == 'dark_factory'
        assert scoped['alpha-topic held out'] == 'dark_factory'
        assert scoped['alpha-topic claim'] == 'dark_factory'
        assert scoped['beta-topic tuned'] == 'reify'
        assert set(pid for _, pid, _ in double.searches) == {'dark_factory', 'reify'}

    def test_entries_outside_the_selected_projects_are_skipped_and_disclosed(
        self, tmp_path,
    ):
        """No silent caps: a narrowed run must say what it did not probe."""
        m = _mod()
        registry = m.TopicRegistry(schema_version=1, entries=(
            _scoped_entry('alpha-topic', project_id='dark_factory'),
            _scoped_entry('beta-topic', project_id='reify'),
        ))
        double = _ServiceDouble(by_query=_canned_hits(registry))

        import asyncio  # noqa: PLC0415

        outcome = asyncio.run(m.run_probe(
            double, registry,
            project_ids=('dark_factory',),
            ks=(5,),
            out_root=tmp_path,
            stamp='20260730T093000Z',
        ))

        assert outcome.skipped_topics == ('beta-topic',)
        assert 'beta-topic' in outcome.report
        assert set(pid for _, pid, _ in double.searches) == {'dark_factory'}


class TestCorpusCounting:
    """(b) One count per category, with the category list derived not restated."""

    def _run_counts(self, tmp_path, counts):
        m = _mod()
        registry = _probe_registry()
        double = _ServiceDouble(by_query=_canned_hits(registry), counts=counts)

        import asyncio  # noqa: PLC0415

        outcome = asyncio.run(m.run_probe(
            double, registry,
            project_ids=('dark_factory',),
            ks=(5,),
            out_root=tmp_path,
            stamp='20260730T094000Z',
        ))
        return double, outcome

    def test_the_category_list_is_the_stores_own(self):
        from fused_memory.models.enums import (  # noqa: PLC0415
            GRAPHITI_PRIMARY,
            MEM0_PRIMARY,
        )

        expected = {c.value for c in (GRAPHITI_PRIMARY | MEM0_PRIMARY)}

        assert set(_mod().corpus_categories()) == expected

    def test_one_count_call_per_category_keyed_by_category_only(self, tmp_path):
        double, _ = self._run_counts(tmp_path, {})

        assert [pid for pid, _ in double.count_calls] == (
            ['dark_factory'] * len(_mod().corpus_categories())
        )
        assert all(set(f) == {'category'} for _, f in double.count_calls)
        assert (
            sorted(f['category'] for _, f in double.count_calls)
            == sorted(_mod().corpus_categories())
        )

    def test_the_counts_reach_the_corpus(self, tmp_path):
        categories = _mod().corpus_categories()
        counts = {c: i + 1 for i, c in enumerate(categories)}
        _, outcome = self._run_counts(tmp_path, counts)

        assert counts.items() <= outcome.series.corpus.counts.items()

    def test_the_step_16_disclosures_still_ride_alongside(self, tmp_path):
        """A superset, deliberately: corpus.counts is free-form by design.

        step-16 folds this run's narrowings (untopiced results, unscorable
        claims, degraded observations) into the same mapping so they reach
        every consumer that reads the JSON rather than only the one who reads
        the prose. Asserting equality here would force that disclosure back
        out into prose-only — a silent cap dressed as a tidy test.
        """
        _, outcome = self._run_counts(tmp_path, {})

        assert 'contamination_untopiced_results' in outcome.series.corpus.counts
        assert 'degraded_observations' in outcome.series.corpus.counts


class TestArgparseBand:
    """(c) The flags this runner exposes — and the ones it must never expose."""

    def test_the_defaults_are_the_census_precedent(self):
        args = _mod().build_parser().parse_args([])

        assert tuple(args.project_id) == ('dark_factory', 'reify')
        assert tuple(args.k) == (5, 10)
        assert args.config is None
        assert args.derive_registry is False

    def test_the_default_registry_is_the_committed_fixture(self):
        args = _mod().build_parser().parse_args([])

        assert Path(args.registry).resolve() == REGISTRY_PATH.resolve()

    def test_the_default_out_root_is_the_gitignored_data_dir(self):
        args = _mod().build_parser().parse_args([])
        out_root = Path(args.out_root).resolve()

        assert out_root.name == 'memory-evals'
        assert out_root.parent.name == 'data'
        assert out_root.parent.parent.name == 'fused-memory'

    def test_an_explicit_project_id_replaces_the_default(self):
        """append+default is the classic argparse footgun; it must not bite."""
        args = _mod().build_parser().parse_args(['--project-id', 'reify'])

        assert tuple(args.project_id) == ('reify',)

    def test_an_explicit_k_replaces_the_default(self):
        args = _mod().build_parser().parse_args(['--k', '7'])

        assert tuple(args.k) == (7,)

    def test_project_id_and_k_are_repeatable(self):
        args = _mod().build_parser().parse_args(
            ['--project-id', 'a', '--project-id', 'b', '--k', '3', '--k', '20'],
        )

        assert tuple(args.project_id) == ('a', 'b')
        assert tuple(args.k) == (3, 20)

    def test_there_is_no_apply_band(self):
        with pytest.raises(SystemExit):
            _mod().build_parser().parse_args(['--apply'])

    def test_no_mutating_flag_exists_at_all(self):
        """The read-only guarantee has to be unreachable from the CLI too."""
        parser = _mod().build_parser()
        flags = {opt for action in parser._actions for opt in action.option_strings}

        assert flags == {
            '-h', '--help',
            '--project-id', '--registry', '--out-root', '--k', '--config',
            '--derive-registry',
        }


class TestRegistryLoadFailureIsFatal:
    """(d) An unloadable registry exits non-zero — never an empty artifact."""

    def test_a_broken_registry_exits_non_zero(self, monkeypatch, tmp_path, capsys):
        m = _mod()
        bad = tmp_path / 'bad.json'
        bad.write_text(json.dumps({'schema_version': 1, 'entries': [
            {'topic': 'no-canonical', 'project_id': 'dark_factory',
             'derived_from': 'hand', 'phrasings': [{'text': 'q'}]},
        ]}), encoding='utf-8')
        _install_double(monkeypatch, _ServiceDouble())

        code = m.main([
            '--registry', str(bad), '--out-root', str(tmp_path / 'out'),
        ])

        assert code != 0
        assert 'no-canonical' in capsys.readouterr().err

    def test_a_broken_registry_emits_no_artifact(self, monkeypatch, tmp_path):
        """Zero topics is indistinguishable from a healthy corpus."""
        m = _mod()
        bad = tmp_path / 'bad.json'
        bad.write_text('{"schema_version": 1, "entries": "not a list"}', encoding='utf-8')
        _install_double(monkeypatch, _ServiceDouble())
        out_root = tmp_path / 'out'

        code = m.main(['--registry', str(bad), '--out-root', str(out_root)])

        assert code != 0
        assert not list(out_root.rglob('metrics-*.json'))

    def test_an_empty_entries_list_fails_the_load(self, tmp_path):
        """A well-formed object carrying zero entries is still unloadable.

        A stale, truncated, or half-written fixture decodes cleanly as
        ``{"entries": []}``. Accepting it produces `"metrics": []` at exit 0 —
        the exact silent artifact RegistryError exists to prevent.
        """
        m = _mod()
        path = _write_registry(tmp_path, _registry_payload(entries_override=[]))

        with pytest.raises(m.RegistryError) as exc:
            m.load_topic_registry(path)

        assert 'zero entries' in str(exc.value)

    def test_an_empty_registry_emits_no_artifact(self, monkeypatch, tmp_path):
        """And it burns no initial-state snapshot on the way out."""
        m = _mod()
        _install_double(monkeypatch, _ServiceDouble())
        path = _write_registry(tmp_path, _registry_payload(entries_override=[]))
        out_root = tmp_path / 'out'

        code = m.main(['--registry', str(path), '--out-root', str(out_root)])

        assert code != 0
        assert not list(out_root.rglob('metrics-*.json'))
        assert m.is_initial_run(out_root)

    def test_a_missing_registry_exits_non_zero(self, monkeypatch, tmp_path):
        m = _mod()
        _install_double(monkeypatch, _ServiceDouble())

        code = m.main([
            '--registry', str(tmp_path / 'absent.json'),
            '--out-root', str(tmp_path / 'out'),
        ])

        assert code != 0

    def test_the_store_is_never_reached_for_a_broken_registry(
        self, monkeypatch, tmp_path,
    ):
        """Fail fast: no embedder spin-up to discover a fixture typo."""
        m = _mod()
        double = _ServiceDouble()
        _install_double(monkeypatch, double)
        bad = tmp_path / 'bad.json'
        bad.write_text('not json at all', encoding='utf-8')

        m.main(['--registry', str(bad), '--out-root', str(tmp_path / 'out')])

        assert not double.initialized


class TestEmptyProjectSelectionIsFatal:
    """(d) The same hazard as an unloadable registry, entered via --project-id.

    A mistyped project id selects no entry, so every metric family measures
    nothing and the run would emit ``"metrics": []`` at exit 0. Downstream that
    is not inert: leaf alpha's evaluator joins by metric_id and would simply
    stop trending the seven pinned metrics, and the junk file counts for
    `is_initial_run`, permanently suppressing the D1 initial-state snapshot for
    the next genuine first run. So it aborts before emission, like a bad load.
    """

    def _registry_file(self, tmp_path):
        registry = _probe_registry()
        path = tmp_path / 'registry.json'
        path.write_text(json.dumps(_as_payload(registry)), encoding='utf-8')
        return registry, path

    def test_run_probe_raises_rather_than_measuring_nothing(self, tmp_path):
        import asyncio  # noqa: PLC0415

        m = _mod()
        registry = _probe_registry()

        with pytest.raises(m.EmptySelectionError):
            asyncio.run(m.run_probe(
                _ServiceDouble(), registry,
                project_ids=('typo_project',),
                ks=m.DEFAULT_KS,
                out_root=tmp_path,
            ))

    def test_a_hand_built_empty_registry_aborts_too(self, tmp_path):
        """Both doors converge on one abort.

        ``load_topic_registry`` rejects an empty entries list, but a caller
        constructing a TopicRegistry directly bypasses it — so the run_probe
        guard keys on "nothing to probe", not on "a selection filtered
        everything out".
        """
        import asyncio  # noqa: PLC0415

        m = _mod()
        empty = m.TopicRegistry(schema_version=1, entries=())

        with pytest.raises(m.EmptySelectionError):
            asyncio.run(m.run_probe(
                _ServiceDouble(), empty,
                project_ids=('dark_factory',),
                ks=m.DEFAULT_KS,
                out_root=tmp_path,
            ))

        assert not list(tmp_path.rglob('metrics-*.json'))

    def test_the_message_names_what_was_asked_and_what_exists(self, tmp_path):
        import asyncio  # noqa: PLC0415

        m = _mod()

        with pytest.raises(m.EmptySelectionError) as excinfo:
            asyncio.run(m.run_probe(
                _ServiceDouble(), _probe_registry(),
                project_ids=('typo_project',),
                ks=m.DEFAULT_KS,
                out_root=tmp_path,
            ))

        message = str(excinfo.value)
        assert 'typo_project' in message
        assert 'dark_factory' in message

    def test_an_unmatched_project_id_exits_non_zero(self, monkeypatch, tmp_path, capsys):
        m = _mod()
        _, path = self._registry_file(tmp_path)
        _install_double(monkeypatch, _ServiceDouble())

        code = m.main([
            '--registry', str(path), '--out-root', str(tmp_path / 'out'),
            '--project-id', 'typo_project',
        ])

        assert code != 0
        assert 'typo_project' in capsys.readouterr().err

    def test_an_unmatched_project_id_emits_no_artifact(self, monkeypatch, tmp_path):
        """The junk artifact would suppress the D1 snapshot forever."""
        m = _mod()
        _, path = self._registry_file(tmp_path)
        _install_double(monkeypatch, _ServiceDouble())
        out_root = tmp_path / 'out'

        code = m.main([
            '--registry', str(path), '--out-root', str(out_root),
            '--project-id', 'typo_project',
        ])

        assert code != 0
        assert not list(out_root.rglob('metrics-*.json'))
        assert m.is_initial_run(out_root), 'the next genuine run is still the first'

    def test_a_matching_project_id_still_runs(self, monkeypatch, tmp_path):
        """The guard fires on an EMPTY selection only — not on any selection."""
        m = _mod()
        registry, path = self._registry_file(tmp_path)
        _install_double(monkeypatch, _ServiceDouble(by_query=_canned_hits(registry)))
        out_root = tmp_path / 'out'

        code = m.main([
            '--registry', str(path), '--out-root', str(out_root),
            '--project-id', 'dark_factory',
        ])

        assert code == 0
        assert list(out_root.rglob('metrics-*.json'))


class TestRunStampOverride:
    """(e) MEMORY_EVAL_RUN_STAMP gives deterministic filenames, no frozen clock."""

    def test_the_env_stamp_names_the_artifact(self, monkeypatch, tmp_path):
        m = _mod()
        registry = _probe_registry()
        registry_path = tmp_path / 'registry.json'
        registry_path.write_text(json.dumps(_as_payload(registry)), encoding='utf-8')
        _install_double(monkeypatch, _ServiceDouble(by_query=_canned_hits(registry)))
        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', '20260731T000000Z')

        code = m.main([
            '--registry', str(registry_path),
            '--out-root', str(tmp_path / 'out'),
            '--project-id', 'dark_factory',
        ])

        assert code == 0
        eval_dir = tmp_path / 'out' / 'e1-retrieval-health'
        assert (eval_dir / 'metrics-20260731T000000Z.json').exists()
        assert (eval_dir / 'report-20260731T000000Z.txt').exists()

    def test_the_stamp_is_the_shared_modules_not_a_local_one(self, monkeypatch):
        from shared.memory_eval_metrics import run_stamp  # noqa: PLC0415

        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', '20260101T010101Z')

        assert run_stamp() == '20260101T010101Z'


# ---------------------------------------------------------------------------
# step-23: the user-observable signal, on a seeded ephemeral collection
#
# The ONE integration test in this file. Everything above measures the probe's
# arithmetic against synthetic lists; this measures the probe against a real
# Qdrant, a real embedder and a real MemoryService, and asserts the single
# thing a synthetic list can never prove: that removing a canonical from the
# store actually flips its tripwire item.
#
# The assertion is a BOOLEAN FLIP on a named item_key — never a rate, never a
# threshold. G6 keeps every limit in leaf alpha; all this test claims is that
# the signal moves when the world moves.
#
# Marked per-test rather than via a module `pytestmark`, because fused-memory's
# `addopts = -m 'not integration'` would otherwise deselect the ~170 pure tests
# above from the merge lane along with this one.
#
# Isolation, in the order it matters:
#   - collection_prefix is `_test_mem0_qdrant_integration`, the ONLY prefix
#     scripts/cleanup_test_collections.py reaps. A collection under the default
#     `fused` prefix would leak forever. Asserted against that script's own
#     PREFIX constant rather than a restated string.
#   - the collection is deleted BEFORE and AFTER, so a swallowed teardown
#     self-heals on the next run instead of poisoning it.
#   - project_id is per-xdist-worker, so concurrent workers cannot collide.
#   - queue.data_dir comes from mock_config's tmp_path, so the durable queue
#     never touches the live one.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import os  # noqa: E402

from _fm_helpers import QDRANT_URL, qdrant_skipif  # noqa: E402

EPHEMERAL_COLLECTION_PREFIX = '_test_mem0_qdrant_integration'

FLIP_TOPIC = 'e1-probe-flip-topic'
FOREIGN_TOPIC = 'e1-probe-foreign-topic'

FLIP_CANONICAL = (
    'The E1 retrieval probe seeds this sentence as the canonical entry for the '
    'ephemeral flip topic: canonical findability is measured by asking whether '
    'this exact entry comes back in the top k results.'
)
FLIP_MEMBERS = (
    'Canonical findability for the ephemeral flip topic is probed with several '
    'query phrasings, at least one of them held out.',
    'The ephemeral flip topic exists only inside a seeded test collection and '
    'is never part of the live corpus.',
)
FOREIGN_CANONICAL = (
    'Contamination distractor: this entry belongs to the foreign probe topic '
    'about unrelated fleet redeploy watchdog staleness backstops.'
)
FOREIGN_MEMBER = (
    'A second foreign-topic distractor about watchdog liveness probes reviving '
    'a wedged orchestrator unit.'
)


def _flip_registry(project_id: str, *, last_known_id: str | None = None):
    """A two-topic registry over the seeded content.

    Both keys are populated on the flip canonical — content_hash (primary) and
    last_known_id (the disclosed fallback) — so the run exercises the dual
    matcher. The test then asserts the hash is what actually fired, because a
    silent fall back to the id would mean search stopped returning content
    verbatim and every content-keyed metric had quietly gone blind.
    """
    m = _mod()

    def entry(topic, canonical, *, known_id=None):
        return m.RegistryEntry(
            topic=topic,
            project_id=project_id,
            derived_from='hand',
            canonical=m.Canonical(
                content_hash=m.content_key(canonical),
                content_prefix=canonical[:80],
                last_known_id=known_id,
            ),
            phrasings=(
                m.Phrasing(f'what does the {topic} say', False),
                m.Phrasing(f'summarise everything known about {topic}', True),
            ),
        )

    return m.TopicRegistry(schema_version=1, entries=(
        entry(FLIP_TOPIC, FLIP_CANONICAL, known_id=last_known_id),
        entry(FOREIGN_TOPIC, FOREIGN_CANONICAL),
    ))


def _tripwire(series):
    """The topic-canonical-present metric off an emitted series."""
    m = _mod()
    for metric in series.metrics:
        if metric.metric_id == m.METRIC_TOPIC_CANONICAL_PRESENT:
            return metric
    raise AssertionError('the series carries no topic-canonical-present metric')


def _item(series, item_key):
    for item in _tripwire(series).items or []:
        if item.item_key == item_key:
            return item
    raise AssertionError(f'no tripwire item {item_key!r} in {_tripwire(series).items!r}')


@pytest.fixture
def probe_project_id(worker_id):
    """Per-xdist-worker so concurrent workers cannot share a collection."""
    return f'probe_e1_{worker_id}'


@pytest.fixture
def probe_config(mock_config, probe_project_id):
    """mock_config pointed at an ephemeral collection with a REAL embedder.

    Clearing the fake api_key makes mem0's OpenAIEmbedding fall back to the
    real OPENAI_API_KEY. A stub constant vector would make every ranking in
    this test meaningless — the whole point is that real retrieval finds the
    seeded canonical and stops finding it once it is gone.
    """
    config = mock_config.model_copy(deep=True)
    config.mem0.collection_prefix = EPHEMERAL_COLLECTION_PREFIX
    config.embedder.providers.openai.api_key = None
    return config


@pytest.fixture
def clean_probe_collection(probe_config, probe_project_id):
    """Delete the seeded collection before AND after the test."""
    from qdrant_client import QdrantClient  # noqa: PLC0415

    from fused_memory.models.scope import Scope  # noqa: PLC0415

    collection = Scope(project_id=probe_project_id).mem0_collection_name(
        probe_config.mem0.collection_prefix,
    )
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


class TestSeededInducedRegression:
    """Delete the canonical; the tripwire item must flip. That is the signal."""

    def test_the_ephemeral_collection_is_one_the_reaper_can_reclaim(
        self, monkeypatch, probe_config, probe_project_id,
    ):
        """A leaked collection under the default prefix would live forever.

        Deliberately NOT via ``clean_probe_collection``: that fixture opens a
        real ``QdrantClient``, and this assertion is about a NAME. Taking the
        fixture would have dragged the one pure test in this class onto the
        network — two 10s timeouts on a machine with no Qdrant — contradicting
        this module's docstring and the merge lane's ``-m 'not integration'``
        selection. ``mem0_collection_name`` is pure, so ask it directly.
        """
        import importlib.util as _ilu  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415

        from fused_memory.models.scope import Scope  # noqa: PLC0415

        collection = Scope(project_id=probe_project_id).mem0_collection_name(
            probe_config.mem0.collection_prefix,
        )

        path = SCRIPT_PATH.parent / 'cleanup_test_collections.py'
        spec = _ilu.spec_from_file_location('cleanup_test_collections', path)
        assert spec is not None and spec.loader is not None
        cleanup = _ilu.module_from_spec(spec)
        # setitem, not a bare assignment: exec_module needs the module visible
        # in sys.modules, but leaving it there leaks into the rest of the
        # session. monkeypatch undoes it at teardown.
        monkeypatch.setitem(_sys.modules, 'cleanup_test_collections', cleanup)
        spec.loader.exec_module(cleanup)

        assert collection.startswith(cleanup.PREFIX)

    @pytest.mark.integration
    @pytest.mark.timeout(300)
    @pytest.mark.asyncio
    @qdrant_skipif()
    @pytest.mark.skipif(
        not os.environ.get('OPENAI_API_KEY'),
        reason='the seeded probe needs a real embedder',
    )
    async def test_deleting_the_canonical_flips_its_tripwire_item(
        self, probe_config, probe_project_id, clean_probe_collection, tmp_path,
    ):
        from fused_memory.models.scope import Scope  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        m = _mod()
        memory = MemoryService(probe_config)
        await memory.initialize()
        try:
            # mem0's SQLite history writer is process-shared and xdist-contended
            # (and read-only in the sandbox). Stubbed for the same reason
            # test_recon_dedup_premise.py:135 stubs it: it is not the question
            # under test, and its failure would mask the one that is.
            instance = await memory.mem0._get_instance(Scope(project_id=probe_project_id))
            instance.db.add_history = lambda *a, **kw: None

            seeded = await memory.add_memory(
                FLIP_CANONICAL, category='procedural_knowledge',
                project_id=probe_project_id, agent_id='e1-probe-seed',
                metadata={'topic': FLIP_TOPIC},
            )
            canonical_id = seeded.memory_ids[0]
            for text in FLIP_MEMBERS:
                await memory.add_memory(
                    text, category='procedural_knowledge',
                    project_id=probe_project_id, agent_id='e1-probe-seed',
                    metadata={'topic': FLIP_TOPIC},
                )
            for text in (FOREIGN_CANONICAL, FOREIGN_MEMBER):
                await memory.add_memory(
                    text, category='procedural_knowledge',
                    project_id=probe_project_id, agent_id='e1-probe-seed',
                    metadata={'topic': FOREIGN_TOPIC},
                )

            registry = _flip_registry(probe_project_id, last_known_id=canonical_id)
            item_key = registry.by_topic[FLIP_TOPIC].item_key

            before = await m.run_probe(
                memory, registry,
                project_ids=(probe_project_id,), ks=(5,),
                out_root=tmp_path, stamp='20260730T100000Z',
            )

            # The live corpus was never in scope: the artifact says so itself.
            assert before.series.corpus.project_id == probe_project_id
            assert _item(before.series, item_key).passed

            # The hash, not the id. A silent fall back to last_known_id would
            # mean search stopped returning content verbatim, which would blind
            # every content-keyed metric while still reporting a pass.
            assert all(
                obs.matched_by == m.MATCHED_BY_CONTENT_HASH
                for obs in before.observations.phrasings
                if obs.topic == FLIP_TOPIC and obs.hit
            )

            await memory.delete_memory(
                canonical_id, store='mem0', project_id=probe_project_id,
            )

            after = await m.run_probe(
                memory, registry,
                project_ids=(probe_project_id,), ks=(5,),
                out_root=tmp_path, stamp='20260730T101000Z',
            )

            assert not _item(after.series, item_key).passed
            assert _tripwire(after.series).value == _tripwire(before.series).value + 1

            # Two runs, two artifacts: the second must not overwrite the first,
            # or the baseline window leaf alpha reads would be one run short.
            assert before.metrics_path != after.metrics_path
            assert before.metrics_path.exists() and after.metrics_path.exists()
            assert before.is_initial_run and not after.is_initial_run
        finally:
            await memory.close()


# ---------------------------------------------------------------------------
# step-24: which store served the query
#
# The first live run against dark_factory made these tests necessary. It
# reported canonical-in-top-5 = 2/38 and 72 unmatched observations, which reads
# as a corpus-wide findability collapse. It is not one. MemoryService.search
# ROUTES: the read router picks a store set per query, and result lists come
# back homogeneous — one phrasing served entirely by Mem0 (canonical at rank 1,
# exact content-hash match), the next served entirely by Graphiti, whose edge
# facts are LLM-extracted sentences and can never contain a Mem0 entry's raw
# content no matter how healthy retrieval is.
#
# The probe deliberately does NOT pin stores: an agent's search is routed too,
# so "the router sent this query somewhere the canonical does not live" is a
# real retrieval-health fact, not a confound to engineer away. But a rate that
# is dominated by routing and does not SAY so is exactly the silent
# fail-soft this leaf exists to prevent — leaf alpha would end up computing
# limits over router coin-flips. So the served store set rides along with every
# observation, in the artifact and in the report.
#
# Still no thresholds: these assert on recorded facts and disclosure presence.
# ---------------------------------------------------------------------------

def _stored(content, store, id='X'):
    r = _R(content=content, id=id)
    r.source_store = store
    return r


class TestStoresServedDisclosure:
    """Which store answered is a recorded fact, not something to infer."""

    def test_the_observation_records_the_stores_that_served_it(self):
        m = _mod()
        entry = _entry(content=CANON)
        results = [_stored(CANON, 'mem0', 'ID-1'), _stored('other', 'graphiti', 'G1')]

        obs = m.observe_phrasing(results, entry, m.Phrasing('q', False), 5)

        assert obs.stores_served == ('graphiti', 'mem0')

    def test_a_result_without_a_store_is_recorded_as_unknown(self):
        """A shape this probe does not recognise must not silently vanish."""
        m = _mod()
        entry = _entry(content=CANON)

        obs = m.observe_phrasing([_R(content=CANON, id='ID-1')], entry, m.Phrasing('q'), 5)

        assert obs.stores_served == ('unknown',)

    def test_only_the_top_k_slice_is_credited(self):
        m = _mod()
        entry = _entry(content=CANON)
        results = [*[_stored('f', 'mem0', f'M{i}') for i in range(5)],
                   _stored('g', 'graphiti', 'G1')]

        assert m.observe_phrasing(results, entry, m.Phrasing('q'), 5).stores_served == ('mem0',)

    def test_the_report_breaks_observations_down_by_serving_store(self):
        m = _mod()
        observations = _report_observations()
        observations.phrasings.append(m.PhrasingObservation(
            topic='alpha-topic', phrasing='routed away', held_out=False, k=5,
            hit=False, rank=None, matched_by=None, stores_served=('graphiti',),
        ))
        report = m.render_probe_report(_report_series(observations), observations)

        assert 'which store served' in report.lower()
        assert 'graphiti' in report

    def test_the_unmatched_section_names_the_stores_that_answered(self):
        """An operator must not have to guess whether the canonical could
        have been returned at all."""
        m = _mod()
        observations = m.ProbeObservations(phrasings=[
            m.PhrasingObservation(
                topic='routed-away-topic', phrasing='q', held_out=False, k=5,
                hit=False, rank=None, matched_by=None, stores_served=('graphiti',),
            ),
            m.PhrasingObservation(
                topic='routed-away-topic', phrasing='h', held_out=True, k=5,
                hit=False, rank=None, matched_by=None, stores_served=('graphiti',),
            ),
        ])
        report = m.render_probe_report(_build(observations), observations)

        section = report.split('canonicals matched by NEITHER key')[1]
        assert 'routed-away-topic' in section
        assert 'graphiti' in section.split('\n\n')[0]

    def test_the_serving_store_counts_ride_in_the_machine_readable_artifact(self):
        """Prose-only disclosure is invisible to every consumer that reads JSON."""
        m = _mod()
        observations = m.ProbeObservations(phrasings=[
            m.PhrasingObservation(
                topic='a', phrasing='q', held_out=False, k=5, hit=True, rank=1,
                matched_by=m.MATCHED_BY_CONTENT_HASH, stores_served=('mem0',),
            ),
            m.PhrasingObservation(
                topic='a', phrasing='h', held_out=True, k=5, hit=False, rank=None,
                matched_by=None, stores_served=('graphiti',),
            ),
        ])
        counts = _build(observations).corpus.counts

        assert counts['observations_served_by_mem0'] == 1
        assert counts['observations_served_by_graphiti'] == 1

    def test_the_probe_band_threads_the_served_stores_through(self):
        """The wiring, not just the dataclass field."""
        import asyncio  # noqa: PLC0415

        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning({
            'tuned query': _healthy([_stored(CANON, 'mem0', 'ID-1')]),
            'held out query': _healthy([_stored('unrelated', 'graphiti', 'G1')]),
        })

        asyncio.run(m.probe_topic(search, entry, registry, (5,), observations))

        by_phrasing = {o.phrasing: o for o in observations.phrasings}
        assert by_phrasing['tuned query'].stores_served == ('mem0',)
        assert by_phrasing['held out query'].stores_served == ('graphiti',)


class TestCorpusCountScopeIsDisclosed:
    """count_memories_by_metadata is a Mem0 count; the report must say so."""

    def test_the_disclosure_names_every_graphiti_primary_category(self):
        """The live run reported entities_and_relations: 0 against a graph
        holding thousands. The zero is honest about what was counted and
        misleading about what exists — so the disclosure has to NAME the
        categories whose zero is an artifact of a Mem0-side count, not just
        say the counts are Mem0-side. The expected names are read off the
        module's own helper, so adding a Graphiti-primary category fails here
        until the report covers it."""
        m = _mod()
        report = m.render_probe_report(_report_series(), _report_observations())

        categories = m.graphiti_primary_categories()
        assert categories
        for category in categories:
            assert category in report


# ---------------------------------------------------------------------------
# step-25: a non-default --k must never be able to DELETE a pinned metric
#
# `--k` is advertised as a repeatable parameterisation, so `--k 7` is a legal
# operator request. Today it is also a silent amputation: `probe_topic(ks=(7,))`
# + `build_series(ks=(7,))` emits only canonical-in-top-7 / claim-recall /
# contamination-share / superseded-above-successor. The flagship
# `topic-canonical-present` tripwire and `canonical-in-top-5-held-out` are
# simply GONE, and nothing anywhere in the artifact or the report says so —
# `not_measured_topics()` also defaults to k=5, so with no k=5 observation to
# read it returns [] no matter how badly the run went.
#
# Since leaf alpha's evaluator joins a run to its baseline window BY metric_id,
# a missing metric does not fail loudly — it just stops being trended. That is
# the "worse than a crash" shape this module's own docstring warns about.
#
# Still no thresholds: every assertion below is on which metric_ids are
# present and on which depth an observation recorded.
# ---------------------------------------------------------------------------

class TestNormaliseKs:
    """(a)-(c) The contract-pinned depth is folded in, the caller's order kept."""

    def test_a_k_that_omits_the_tripwire_depth_gains_it(self):
        m = _mod()

        assert m.TRIPWIRE_K in m.normalise_ks((7,))

    def test_the_callers_order_is_preserved_and_the_pin_appended(self):
        """Requested depths first: the operator asked for 7, and got 7 AND 5."""
        m = _mod()

        assert m.normalise_ks((7,)) == (7, m.TRIPWIRE_K)
        assert m.normalise_ks((20, 7)) == (20, 7, m.TRIPWIRE_K)

    def test_an_already_pinned_k_is_returned_unchanged(self):
        """The default path's metric vocabulary must not churn."""
        m = _mod()

        assert m.normalise_ks((5, 10)) == (5, 10)
        assert m.normalise_ks(m.DEFAULT_KS) == m.DEFAULT_KS

    def test_duplicates_collapse(self):
        """A doubled --k would otherwise emit the same metric_id twice."""
        m = _mod()

        assert m.normalise_ks((7, 7, 5)) == (7, 5)

    def test_an_empty_k_yields_the_pin_alone(self):
        m = _mod()

        assert m.normalise_ks(()) == (m.TRIPWIRE_K,)


class TestNonDefaultKKeepsThePinnedMetrics:
    """(d) The review's exact repro, pinned end to end as a regression test."""

    def _run(self, tmp_path, ks, *, double=None, registry=None):
        m = _mod()
        registry = registry if registry is not None else _probe_registry()
        double = double if double is not None else _ServiceDouble(
            by_query=_canned_hits(registry),
        )

        import asyncio  # noqa: PLC0415

        return double, asyncio.run(m.run_probe(
            double, registry,
            project_ids=('dark_factory',),
            ks=ks,
            out_root=tmp_path,
            stamp='20260731T100000Z',
        ))

    def test_k_seven_still_emits_the_flagship_tripwire(self, tmp_path):
        m = _mod()
        _, outcome = self._run(tmp_path, (7,))

        ids = [metric.metric_id for metric in outcome.series.metrics]
        assert m.METRIC_TOPIC_CANONICAL_PRESENT in ids

    def test_k_seven_still_emits_the_held_out_proportion(self, tmp_path):
        """The Goodhart guard cannot be switched off by a CLI flag."""
        m = _mod()
        _, outcome = self._run(tmp_path, (7,))

        ids = [metric.metric_id for metric in outcome.series.metrics]
        assert m.METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=m.TRIPWIRE_K) in ids

    def test_the_requested_depth_is_still_measured(self, tmp_path):
        """Normalising ADDS the pin; it never drops what the operator asked for."""
        m = _mod()
        _, outcome = self._run(tmp_path, (7,))

        ids = [metric.metric_id for metric in outcome.series.metrics]
        assert m.METRIC_CANONICAL_IN_TOP_K.format(k=7) in ids
        assert m.METRIC_CANONICAL_IN_TOP_K.format(k=m.TRIPWIRE_K) in ids

    def test_the_pinned_metric_set_is_a_superset_on_every_run(self, tmp_path):
        """Whatever --k is passed, alpha's join keys are all still there."""
        m = _mod()
        pinned = {
            m.METRIC_TOPIC_CANONICAL_PRESENT,
            m.METRIC_CANONICAL_IN_TOP_K.format(k=m.TRIPWIRE_K),
            m.METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=m.TRIPWIRE_K),
            m.METRIC_CLAIM_RECALL,
            m.METRIC_CONTAMINATION_SHARE,
            m.METRIC_SUPERSEDED_ABOVE_SUCCESSOR,
        }
        for ks in ((7,), (3,), (20, 7), (5, 10)):
            _, outcome = self._run(tmp_path, ks)
            ids = {metric.metric_id for metric in outcome.series.metrics}
            assert pinned <= ids, f'--k {ks} amputated {sorted(pinned - ids)}'

    def test_the_not_measured_disclosure_is_no_longer_vacuous(self, tmp_path):
        """not_measured_topics() reads k=5 observations; with --k 7 there were
        none, so a fully-degraded topic was reported as [] — silence that reads
        exactly like a healthy run."""
        m = _mod()
        registry = _probe_registry()
        healthy = _canned_hits(registry)
        for phrasing in registry.entries[1].phrasings:
            healthy[phrasing.text] = _degraded()

        _, outcome = self._run(tmp_path, (7,), double=_ServiceDouble(by_query=healthy))

        assert m.not_measured_topics(outcome.observations) == ['beta-topic']

    def test_the_operators_actual_path_through_the_cli(self, monkeypatch, tmp_path):
        """(e) --k 7 is a supported flag, so the fix has to hold through main()."""
        from shared.memory_eval_metrics import load_metric_series  # noqa: PLC0415

        m = _mod()
        registry = _probe_registry()
        registry_path = tmp_path / 'registry.json'
        registry_path.write_text(json.dumps(_as_payload(registry)), encoding='utf-8')
        _install_double(monkeypatch, _ServiceDouble(by_query=_canned_hits(registry)))
        monkeypatch.setenv('MEMORY_EVAL_RUN_STAMP', '20260731T101500Z')

        code = m.main([
            '--registry', str(registry_path),
            '--out-root', str(tmp_path / 'out'),
            '--project-id', 'dark_factory',
            '--k', '7',
        ])

        assert code == 0
        series = load_metric_series(
            tmp_path / 'out' / 'e1-retrieval-health' / 'metrics-20260731T101500Z.json',
        )
        ids = {metric.metric_id for metric in series.metrics}
        assert m.METRIC_TOPIC_CANONICAL_PRESENT in ids
        assert m.METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=m.TRIPWIRE_K) in ids
        assert m.METRIC_CANONICAL_IN_TOP_K.format(k=7) in ids

    def test_the_added_depth_is_disclosed_rather_than_inferred(self, tmp_path):
        """Measuring a depth the operator did not ask for is a narrowing like
        any other: it gets said out loud, not left to be reverse-engineered
        from the metric list."""
        _, outcome = self._run(tmp_path, (7,))

        assert 'measurement depth' in outcome.report.lower()
        assert 'requested 7' in outcome.report
        assert 'measured 7, 5' in outcome.report

    def test_the_disclosure_says_why_the_pin_was_added(self, tmp_path):
        """A depth appearing unbidden is only legible if the reason is beside
        it: both pinned metrics are DEFINED at k=5."""
        m = _mod()
        _, outcome = self._run(tmp_path, (7,))
        section = outcome.report.lower().split('measurement depth', 1)[1]

        assert m.METRIC_TOPIC_CANONICAL_PRESENT in section
        assert m.METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=m.TRIPWIRE_K) in section

    def test_the_default_run_discloses_no_added_depth(self, tmp_path):
        """Nothing was added, so there is nothing to disclose — a section that
        fired every run would train an operator to skip it."""
        _, outcome = self._run(tmp_path, (5, 10))

        assert 'measurement depth' not in outcome.report.lower()


class TestObservationDepthIsHonest:
    """(f) An observation never claims a depth deeper than what was fetched."""

    def _observe(self, ks):
        m = _mod()
        entry = _probe_entry()
        registry = m.TopicRegistry(schema_version=1, entries=(entry,))
        observations = m.ProbeObservations()
        search = _search_returning(
            {}, default_factory=lambda: _healthy(_filler(10)),
        )

        import asyncio  # noqa: PLC0415

        asyncio.run(m.probe_topic(search, entry, registry, ks, observations))
        return observations

    def test_a_shallow_direct_call_records_the_depth_it_fetched(self):
        """probe_topic is public and called directly by this file's own tests;
        with ks=(3,) it searched at limit=3 but stamped k=5 on every claim and
        contamination observation — an artifact mislabelling its own depth."""
        m = _mod()
        observations = self._observe((3,))
        expected = min(m.TRIPWIRE_K, 3)

        assert {o.k for o in observations.contamination} == {expected}
        assert {o.k for o in observations.claims} == {expected}

    def test_a_deep_call_still_scores_at_the_pinned_depth(self):
        """min(), not max(): contamination and claim recall are DEFINED at the
        tripwire's depth, so a deeper fetch must not silently widen them."""
        m = _mod()
        observations = self._observe((10,))

        assert {o.k for o in observations.contamination} == {m.TRIPWIRE_K}
        assert {o.k for o in observations.claims} == {m.TRIPWIRE_K}

    def test_the_default_path_is_unchanged(self):
        m = _mod()
        observations = self._observe(m.DEFAULT_KS)

        assert {o.k for o in observations.contamination} == {m.TRIPWIRE_K}
        assert {o.k for o in observations.claims} == {m.TRIPWIRE_K}
