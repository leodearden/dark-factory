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
        assert entry.claim_queries[0].needles == ['something']
        assert entry.members == ['b' * 16]
        assert entry.supersedes_pairs == []

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
