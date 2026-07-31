"""Tests for scripts/memory_eval_transcript_corpus.py — the retro replay corpus.

PRD ``docs/prds/memory-eval-program.md`` §5 leaf θ, decision D9: a one-shot,
READ-ONLY mining of every memory search out of the already-archived agent
transcripts, plus a coverage report that discloses what it could not read.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — the same ``_load_module()`` helper as
test_census_memory_metadata.py.

Every test here runs against in-memory record lists or a temp archive tree,
except the mini-fixture round-trip, which reads the committed archive under
``tests/fixtures/transcript_corpus/``. Nothing touches the live memory store:
the script only reads transcripts off disk.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_transcript_corpus.py'
FIXTURE_ARCHIVE = Path(__file__).parent / 'fixtures' / 'transcript_corpus'


def _load_module() -> types.ModuleType:
    """Load memory_eval_transcript_corpus.py from its file path."""
    mod_name = 'memory_eval_transcript_corpus'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()

SEARCH = 'mcp__fused-memory__search'


# ===========================================================================
# Record builders — the shapes measured in the live archive.
# ===========================================================================


def _tool_use(tool_use_id: str, query: str, *, name: str = SEARCH, **params):
    """An assistant record carrying one search ``tool_use`` block.

    Note the block's own ``"caller": {"type": "direct"}``: a Claude Code
    harness field present on every measured call, uniform, and NOT agent
    identity. Reproduced here so the extractor is exercised against the
    name collision rather than around it.
    """
    return {
        'type': 'assistant',
        'message': {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': tool_use_id,
                    'name': name,
                    'input': {'query': query, **params},
                    'caller': {'type': 'direct'},
                }
            ],
        },
    }


def _tool_result(tool_use_id: str, payload, *, is_error: bool = False):
    """A user record carrying the answer. ``content`` is a JSON STRING."""
    block = {
        'type': 'tool_result',
        'tool_use_id': tool_use_id,
        'content': payload if isinstance(payload, str) else json.dumps(payload),
    }
    if is_error:
        block['is_error'] = True
    return {'type': 'user', 'message': {'role': 'user', 'content': [block]}}


def _result(memory_id: str, score: float, *, content: str = 'x', **over):
    entry = {
        'id': memory_id,
        'content': content,
        'category': 'procedural_knowledge',
        'source_store': 'mem0',
        'relevance_score': score,
        'provenance': [],
        'temporal': None,
        'entities': [],
        'metadata': {},
        'created_at': '2026-07-01T00:00:00+00:00',
    }
    entry.update(over)
    return entry


# ===========================================================================
# step-3 — the pure extraction core
# ===========================================================================


class TestModuleLoads:
    """The importlib load itself proves the sys.path bootstrap works.

    ``legibility`` is a namespace package under repo-root ``scripts/``, which
    fused-memory's pytest ``pythonpath`` does not include — so if the module's
    bootstrap were wrong, importing it here would raise ImportError.
    """

    def test_exposes_schema_version(self):
        assert _mod.SCHEMA_VERSION == 1

    def test_exposes_search_tool_names(self):
        assert SEARCH in _mod.SEARCH_TOOL_NAMES


class TestExtractSearches:
    """``extract_searches(records)`` — one pass, tool_use order, pure."""

    RECORDS = [
        # Interleaved non-message record: really present mid-session, and
        # carrying no 'message' key at all.
        {'type': 'queue-operation', 'operation': 'enqueue'},
        _tool_use('toolu_A', 'archive layout and gz naming',
                  project_id='dark_factory', limit=5),
        _tool_result('toolu_A', {'results': [
            _result('aaaaaaaa-1111-4aaa-8aaa-aaaaaaaaaaaa', 0.87, content='first'),
            _result('bbbbbbbb-2222-4bbb-8bbb-bbbbbbbbbbbb', 0.42, content='second one',
                    category='observations_and_summaries', source_store='graphiti'),
        ]}),
    ]

    def _only(self):
        records = _mod.extract_searches(self.RECORDS)
        assert len(records) == 1
        return records[0]

    def test_emits_exactly_one_record(self):
        assert len(_mod.extract_searches(self.RECORDS)) == 1

    def test_query_is_full_and_untruncated(self):
        assert self._only()['query'] == 'archive layout and gz naming'

    def test_tool_use_id_and_name(self):
        record = self._only()
        assert record['tool_use_id'] == 'toolu_A'
        assert record['tool_name'] == SEARCH

    def test_params_carry_non_query_input_keys(self):
        # The query has its own field; params is everything else the caller
        # passed, which is what makes a limit= or project_id= effect legible.
        assert self._only()['params'] == {'project_id': 'dark_factory', 'limit': 5}

    def test_result_status_and_count(self):
        record = self._only()
        assert record['result_status'] == 'ok'
        assert record['result_count'] == 2

    def test_results_are_rank_ordered_projections(self):
        assert self._only()['results'] == [
            {
                'id': 'aaaaaaaa-1111-4aaa-8aaa-aaaaaaaaaaaa',
                'score': 0.87,
                'source_store': 'mem0',
                'category': 'procedural_knowledge',
                'content_chars': len('first'),
                'rank': 1,
            },
            {
                'id': 'bbbbbbbb-2222-4bbb-8bbb-bbbbbbbbbbbb',
                'score': 0.42,
                'source_store': 'graphiti',
                'category': 'observations_and_summaries',
                'content_chars': len('second one'),
                'rank': 2,
            },
        ]

    def test_result_content_text_is_never_copied(self):
        # Decision 3: ids are stable handles the consumer resolves against the
        # live store. Copying the text would fork the memory corpus into an
        # eval artifact that silently stales.
        for entry in self._only()['results']:
            assert 'content' not in entry

    def test_schema_version_stamped_on_every_record(self):
        assert self._only()['schema_version'] == _mod.SCHEMA_VERSION

    def test_non_message_records_contribute_nothing(self):
        assert _mod.extract_searches([{'type': 'queue-operation'}]) == []
        assert _mod.extract_searches([{'type': 'attachment', 'message': None}]) == []

    def test_non_search_tool_uses_contribute_nothing(self):
        records = [
            _tool_use('toolu_R', 'x', name='Read'),
            _tool_use('toolu_M', 'x', name='mcp__fused-memory__add_memory'),
            _tool_use('toolu_T', 'x', name='mcp__fused-memory__search_tasks'),
        ]
        assert _mod.extract_searches(records) == []

    def test_empty_input(self):
        assert _mod.extract_searches([]) == []

    def test_emits_in_tool_use_order(self):
        records = _mod.extract_searches([
            _tool_use('toolu_1', 'first query'),
            _tool_use('toolu_2', 'second query'),
            _tool_result('toolu_2', {'results': []}),
            _tool_result('toolu_1', {'results': []}),
        ])
        # Answered second, but issued first: the corpus is ordered by the
        # search, not by when its answer happened to stream back.
        assert [r['query'] for r in records] == ['first query', 'second query']

    def test_source_is_stamped(self):
        records = _mod.extract_searches(self.RECORDS, source='some/relative/path.jsonl.gz')
        assert records[0]['transcript'] == 'some/relative/path.jsonl.gz'

    def test_tool_names_is_overridable(self):
        records = _mod.extract_searches(
            [_tool_use('toolu_X', 'q', name='custom__search')],
            tool_names=frozenset({'custom__search'}),
        )
        assert len(records) == 1
        assert records[0]['tool_name'] == 'custom__search'

    def test_string_message_content_is_not_scanned_for_tool_use(self):
        # A user record's content is often a plain string; it must not blow up.
        assert _mod.extract_searches([
            {'type': 'user', 'message': {'role': 'user', 'content': 'plain text'}}
        ]) == []
