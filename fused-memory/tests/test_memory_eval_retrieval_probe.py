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
