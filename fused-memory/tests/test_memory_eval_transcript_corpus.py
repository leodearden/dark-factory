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

import gzip
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


# ===========================================================================
# step-5 — caller identity
# ===========================================================================


def _briefing(agent_id: str) -> dict:
    """The first user record's briefing, in the measured live shape."""
    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': (
                '# Context\n\n(elided)\n\n## Agent Identity\n\n'
                f'- **agent_id:** `{agent_id}`\n'
                '- **project_id:** `dark_factory`\n'
            ),
        },
    }


class TestCallerIdentity:
    """Who issued the search — recovered from the briefing, never from the
    ``tool_use`` block's same-named harness field.

    PRD §3: the going-forward telemetry seam left 99.7% of journal rows
    unattributed because agent_id was overloaded as a FILTER param, so retro
    attribution cannot be read off the search call itself. The dispatched
    briefing does carry it.

    Every field degrades to None rather than blocking emission: dropping an
    unattributed search would silently bias the corpus toward whichever agent
    roles happen to carry a parseable briefing.
    """

    def _caller_for(self, agent_id: str) -> dict:
        records = _mod.extract_searches([_briefing(agent_id), _tool_use('toolu_A', 'q')])
        assert len(records) == 1
        return records[0]['caller']

    def test_stamps_agent_id_task_and_role(self):
        # The measured live shape.
        assert self._caller_for('claude-task-2280-architect') == {
            'agent_id': 'claude-task-2280-architect',
            'task_id': '2280',
            'role': 'architect',
        }

    def test_hyphenated_role_survives(self):
        # Role is the remainder after the numeric id, so a multi-segment role
        # is not truncated at its first hyphen.
        assert self._caller_for('claude-task-3214-code-reviewer') == {
            'agent_id': 'claude-task-3214-code-reviewer',
            'task_id': '3214',
            'role': 'code-reviewer',
        }

    def test_unrecognised_shape_preserves_agent_id_verbatim(self):
        # An interactive session's agent_id is not claude-task-<id>-<role>.
        # What we DID observe is kept; what we cannot derive is None.
        assert self._caller_for('claude-interactive') == {
            'agent_id': 'claude-interactive',
            'task_id': None,
            'role': None,
        }

    def test_no_identity_block_leaves_all_fields_none(self):
        records = _mod.extract_searches([_tool_use('toolu_A', 'q')])
        assert len(records) == 1, 'an unattributed search must still be emitted'
        assert records[0]['caller'] == {'agent_id': None, 'task_id': None, 'role': None}

    def test_first_match_wins(self):
        # A later record quoting a different agent_id (an agent discussing
        # another agent's briefing) must not overwrite the real one.
        records = _mod.extract_searches([
            _briefing('claude-task-2280-architect'),
            {'type': 'user', 'message': {'role': 'user',
                                         'content': 'agent_id:** `claude-task-9999-imposter`'}},
            _tool_use('toolu_A', 'q'),
        ])
        assert records[0]['caller']['agent_id'] == 'claude-task-2280-architect'

    def test_identity_resolved_before_and_after_a_search_alike(self):
        # The briefing precedes any search in a real transcript, but the
        # extractor must not depend on that ordering accident.
        records = _mod.extract_searches([
            _tool_use('toolu_A', 'q'),
            _briefing('claude-task-2280-architect'),
        ])
        assert records[0]['caller']['task_id'] == '2280'


# ===========================================================================
# step-7 — result resolution status (INV-2)
# ===========================================================================


class TestResultStatus:
    """A search whose answer could not be recovered is DISCLOSED, never
    silently dropped and never mistaken for a search that found nothing.

    Those are different facts about the corpus: "the store returned nothing"
    is a retrieval measurement, "we could not read what the store returned" is
    a gap in our own instrument. Collapsing them would let a decoding bug read
    as a recall collapse.
    """

    def _one(self, records) -> dict:
        emitted = _mod.extract_searches(records)
        assert len(emitted) == 1, 'the search must be emitted whatever its answer'
        return emitted[0]

    def test_no_tool_result_anywhere_is_missing(self):
        # The truncated-transcript case: archiving raced the session's end.
        record = self._one([_tool_use('toolu_A', 'q')])
        assert record['result_status'] == 'missing'
        assert record['results'] == []
        assert record['result_count'] == 0

    def test_undecodable_content_is_unparsed(self):
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', 'not json at all {['),
        ])
        assert record['result_status'] == 'unparsed'
        assert record['results'] == []

    def test_is_error_flag_is_error(self):
        # The tool itself failed. The search was still issued, and that it
        # failed is a fact about the corpus worth keeping.
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', 'MCP error -32603: backend unavailable', is_error=True),
        ])
        assert record['result_status'] == 'error'
        assert record['results'] == []

    def test_is_error_wins_over_a_decodable_payload(self):
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', {'results': [_result('a', 0.5)]}, is_error=True),
        ])
        assert record['result_status'] == 'error'

    def test_missing_results_key_is_unparsed(self):
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', {'something_else': 1}),
        ])
        assert record['result_status'] == 'unparsed'

    def test_non_list_results_is_unparsed(self):
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', {'results': 'nope'}),
        ])
        assert record['result_status'] == 'unparsed'

    def test_genuine_zero_hit_search_is_ok(self):
        # THE distinction this class exists for: an empty answer is a
        # measurement, and it is distinguishable from an unrecovered one.
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', {'results': []}),
        ])
        assert record['result_status'] == 'ok'
        assert record['result_count'] == 0
        assert record['results'] == []

    def test_zero_hit_and_unrecovered_are_distinguishable(self):
        # Asserted as a property, not inferred from the two tests above: all
        # four zero-result cases share result_count == 0, so ONLY the status
        # separates them. A consumer that filters on count alone would merge
        # a decoding bug into the recall measurement.
        cases = {
            'ok': [_tool_use('t', 'q'), _tool_result('t', {'results': []})],
            'missing': [_tool_use('t', 'q')],
            'unparsed': [_tool_use('t', 'q'), _tool_result('t', 'garbage{')],
            'error': [_tool_use('t', 'q'), _tool_result('t', 'boom', is_error=True)],
        }
        observed = {name: self._one(records) for name, records in cases.items()}
        assert all(r['result_count'] == 0 for r in observed.values())
        assert {name: r['result_status'] for name, r in observed.items()} == {
            'ok': 'ok', 'missing': 'missing', 'unparsed': 'unparsed', 'error': 'error',
        }

    def test_status_vocabulary_is_closed(self):
        assert set(_mod._RESULT_STATUS) == {'ok', 'error', 'unparsed', 'missing'}

    def test_result_entry_missing_fields_degrades_without_losing_siblings(self):
        record = self._one([
            _tool_use('toolu_A', 'q'),
            _tool_result('toolu_A', {'results': [
                {'content': 'no id and no score here'},
                _result('bbbbbbbb-2222-4bbb-8bbb-bbbbbbbbbbbb', 0.42, content='intact'),
            ]}),
        ])
        assert record['result_status'] == 'ok'
        assert record['result_count'] == 2
        degraded, intact = record['results']
        assert degraded['id'] is None
        assert degraded['score'] is None
        assert degraded['rank'] == 1
        assert degraded['content_chars'] == len('no id and no score here')
        assert intact['id'] == 'bbbbbbbb-2222-4bbb-8bbb-bbbbbbbbbbbb'
        assert intact['score'] == 0.42
        assert intact['rank'] == 2

    def test_block_list_content_shape_is_decoded(self):
        # Some tool results arrive as [{'type': 'text', 'text': '<json>'}]
        # rather than a bare string.
        payload = json.dumps({'results': [_result('aaaa', 0.9, content='hi')]})
        record = self._one([
            _tool_use('toolu_A', 'q'),
            {'type': 'user', 'message': {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'toolu_A',
                 'content': [{'type': 'text', 'text': payload}]},
            ]}},
        ])
        assert record['result_status'] == 'ok'
        assert record['results'][0]['id'] == 'aaaa'


# ===========================================================================
# step-9 — archive-path provenance
# ===========================================================================

MAIN_REL = '2280/-home-leo-src-dark-factory--worktrees-2280/775be5c7-539a-4071-ad3c-89422cf3317a.jsonl.gz'
SUB_REL = (
    '3006/-home-leo-src-dark-factory--worktrees-3006/'
    '2f747640-1111-2222-3333-444444444444/subagents/agent-a2af78e35bd434081.jsonl.gz'
)


class TestParseArchivePath:
    """Task/session/subagent identity comes from the PATH, because that is
    where ``shared.transcript_archive`` puts it.

    Both live layouts are covered. A path that does not conform degrades every
    field to None rather than raising: an archive can acquire a stray file, and
    one of those must not abort a whole-archive scan.
    """

    def test_main_session_layout(self):
        assert _mod.parse_archive_path(MAIN_REL) == {
            'task_id': '2280',
            'session_id': '775be5c7-539a-4071-ad3c-89422cf3317a',
            'is_subagent': False,
            'subagent_id': None,
        }

    def test_subagent_layout(self):
        assert _mod.parse_archive_path(SUB_REL) == {
            'task_id': '3006',
            'session_id': '2f747640-1111-2222-3333-444444444444',
            'is_subagent': True,
            'subagent_id': 'agent-a2af78e35bd434081',
        }

    def test_plain_jsonl_suffix_also_parses(self):
        parsed = _mod.parse_archive_path('2280/-enc-cwd/session-abc.jsonl')
        assert parsed['task_id'] == '2280'
        assert parsed['session_id'] == 'session-abc'

    def test_too_shallow_path_degrades_without_raising(self):
        assert _mod.parse_archive_path('loose-file.jsonl.gz') == {
            'task_id': None, 'session_id': None,
            'is_subagent': False, 'subagent_id': None,
        }

    def test_unexpected_shape_degrades_without_raising(self):
        parsed = _mod.parse_archive_path('a/b/c/d/e/f.jsonl.gz')
        assert parsed['task_id'] == 'a'
        assert parsed['session_id'] is None
        assert parsed['is_subagent'] is False

    def test_accepts_a_path_object(self):
        assert _mod.parse_archive_path(Path(MAIN_REL))['task_id'] == '2280'


class TestProvenanceThreadedIntoRecords:
    """Emitted records carry the provenance and an archive-RELATIVE path."""

    def _record(self, rel: str) -> dict:
        records = _mod.extract_searches(
            [_tool_use('toolu_A', 'q')],
            source=rel,
            provenance=_mod.parse_archive_path(rel),
        )
        assert len(records) == 1
        return records[0]

    def test_main_session_provenance(self):
        record = self._record(MAIN_REL)
        assert record['task_id'] == '2280'
        assert record['session_id'] == '775be5c7-539a-4071-ad3c-89422cf3317a'
        assert record['is_subagent'] is False

    def test_subagent_provenance(self):
        record = self._record(SUB_REL)
        assert record['task_id'] == '3006'
        assert record['is_subagent'] is True
        assert record['subagent_id'] == 'agent-a2af78e35bd434081'

    def test_transcript_path_is_archive_relative(self):
        # An absolute machine path would make the corpus unreadable on any
        # other machine and would leak the operator's home directory into a
        # shared artifact.
        record = self._record(MAIN_REL)
        assert record['transcript'] == MAIN_REL
        assert not record['transcript'].startswith('/')


# ===========================================================================
# step-11 — archive scan + coverage counters
# ===========================================================================


def _write_transcript(root: Path, rel: str, records: list[dict]) -> Path:
    """Write a gzipped transcript at *rel* under *root*."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(r) for r in records) + '\n')
    return path


def _write_corrupt(root: Path, rel: str) -> Path:
    """A file under a .jsonl.gz name that is not gzip at all."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'this is not gzip\n')
    return path


class TestScanArchive:
    """Walk an archive tree, extract, and account for what could not be read.

    The counters are the operator-facing half of this script: a scan that
    under-reports is indistinguishable from a scan of a smaller archive unless
    it says what it skipped.
    """

    GOOD_REL = '2280/-enc-a/aaaa-1111.jsonl.gz'
    EMPTY_REL = '2280/-enc-a/bbbb-2222.jsonl.gz'
    BAD_REL = '3006/-enc-b/cccc-3333.jsonl.gz'

    def _tree(self, tmp_path: Path) -> Path:
        root = tmp_path / 'archive'
        _write_transcript(root, self.GOOD_REL, [
            _briefing('claude-task-2280-architect'),
            _tool_use('toolu_1', 'first'),
            _tool_result('toolu_1', {'results': [_result('a', 0.9)]}),
            _tool_use('toolu_2', 'second'),
            _tool_result('toolu_2', {'results': []}),
        ])
        # Parses fine, holds no searches — must count as READ, not as failed.
        _write_transcript(root, self.EMPTY_REL, [{'type': 'queue-operation'}])
        _write_corrupt(root, self.BAD_REL)
        return root

    def test_returns_the_good_files_records(self, tmp_path):
        records, _ = _mod.scan_archive(self._tree(tmp_path))
        assert [r['query'] for r in records] == ['first', 'second']

    def test_counters(self, tmp_path):
        _, coverage = _mod.scan_archive(self._tree(tmp_path))
        assert coverage['tasks_scanned'] == 2
        assert coverage['transcripts_found'] == 3
        assert coverage['transcripts_read'] == 2
        assert coverage['searches_extracted'] == 2

    def test_parse_failure_is_counted_and_exemplified(self, tmp_path):
        _, coverage = _mod.scan_archive(self._tree(tmp_path))
        failures = coverage['parse_failures']
        assert failures['count'] == 1
        assert len(failures['examples']) == 1
        example = failures['examples'][0]
        # The archive-relative path, so an operator can find the file, and a
        # reason, so they know whether to care.
        assert example['transcript'] == self.BAD_REL
        assert example['reason']
        assert not example['transcript'].startswith('/')

    def test_searches_unresolved_counts_non_ok_statuses(self, tmp_path):
        root = tmp_path / 'archive'
        _write_transcript(root, self.GOOD_REL, [
            _tool_use('toolu_1', 'answered'),
            _tool_result('toolu_1', {'results': [_result('a', 0.9)]}),
            _tool_use('toolu_2', 'never answered'),
        ])
        _, coverage = _mod.scan_archive(root)
        assert coverage['searches_extracted'] == 2
        assert coverage['searches_unresolved'] == 1

    def test_provenance_recovered_from_the_tree(self, tmp_path):
        records, _ = _mod.scan_archive(self._tree(tmp_path))
        assert records[0]['task_id'] == '2280'
        assert records[0]['session_id'] == 'aaaa-1111'
        assert records[0]['is_subagent'] is False
        assert records[0]['transcript'] == self.GOOD_REL
        assert records[0]['caller']['role'] == 'architect'

    def test_subagent_transcripts_are_scanned(self, tmp_path):
        root = tmp_path / 'archive'
        _write_transcript(root, '3006/-enc-b/sess-1/subagents/agent-ff00.jsonl.gz', [
            _tool_use('toolu_1', 'from a subagent'),
            _tool_result('toolu_1', {'results': []}),
        ])
        records, coverage = _mod.scan_archive(root)
        assert coverage['transcripts_found'] == 1
        assert len(records) == 1
        assert records[0]['is_subagent'] is True
        assert records[0]['subagent_id'] == 'agent-ff00'

    def test_plain_jsonl_transcripts_are_scanned_too(self, tmp_path):
        root = tmp_path / 'archive'
        path = root / '2280/-enc-a/plain.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_tool_use('toolu_1', 'uncompressed')) + '\n')
        records, coverage = _mod.scan_archive(root)
        assert coverage['transcripts_found'] == 1
        assert [r['query'] for r in records] == ['uncompressed']

    def test_missing_root_reports_zero_without_raising(self, tmp_path):
        records, coverage = _mod.scan_archive(tmp_path / 'does-not-exist')
        assert records == []
        assert coverage['transcripts_found'] == 0
        assert coverage['parse_failures']['count'] == 0

    def test_deterministic_order_across_runs(self, tmp_path):
        root = self._tree(tmp_path)
        first, _ = _mod.scan_archive(root)
        second, _ = _mod.scan_archive(root)
        # Byte-identical corpora from the same tree — otherwise two runs of an
        # unchanged archive would diff, and a real change would be unreadable.
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_tool_names_is_threaded_through(self, tmp_path):
        root = tmp_path / 'archive'
        _write_transcript(root, self.GOOD_REL, [_tool_use('t', 'q', name='legacy__search')])
        records, _ = _mod.scan_archive(root, tool_names=frozenset({'legacy__search'}))
        assert len(records) == 1


class TestFailureExamplesAreCapped:
    """The example list is bounded — and says so when it truncates.

    An unbounded list could be thousands of entries; a silently truncated one
    would understate a systemic failure as a handful of stragglers.
    """

    def _two_failures(self, tmp_path: Path) -> Path:
        root = tmp_path / 'archive'
        _write_corrupt(root, '1/-enc/a.jsonl.gz')
        _write_corrupt(root, '2/-enc/b.jsonl.gz')
        return root

    def test_count_is_never_capped(self, tmp_path):
        _, coverage = _mod.scan_archive(self._two_failures(tmp_path), max_failure_examples=1)
        assert coverage['parse_failures']['count'] == 2

    def test_examples_are_capped(self, tmp_path):
        _, coverage = _mod.scan_archive(self._two_failures(tmp_path), max_failure_examples=1)
        assert len(coverage['parse_failures']['examples']) == 1

    def test_truncation_is_disclosed(self, tmp_path):
        _, coverage = _mod.scan_archive(self._two_failures(tmp_path), max_failure_examples=1)
        failures = coverage['parse_failures']
        assert failures['examples_truncated'] is True
        assert failures['examples_omitted'] == 1

    def test_no_truncation_is_disclosed_as_such(self, tmp_path):
        _, coverage = _mod.scan_archive(self._two_failures(tmp_path), max_failure_examples=10)
        failures = coverage['parse_failures']
        assert failures['examples_truncated'] is False
        assert failures['examples_omitted'] == 0
        assert len(failures['examples']) == 2


# ===========================================================================
# step-13 — THE INV-2/INV-4 DISTINGUISHABILITY TEST
# ===========================================================================


def _coverage(found: int, read: int, failed: int = 0, searches: int = 0) -> dict:
    coverage = _mod._new_coverage()
    coverage['transcripts_found'] = found
    coverage['transcripts_read'] = read
    coverage['searches_extracted'] = searches
    coverage['parse_failures']['count'] = failed
    return coverage


class TestCoverageStatus:
    """The task's headline requirement.

    "0 searches extracted" must not read the same way for an empty archive and
    for a run where every transcript failed. A count alone cannot carry that —
    both are 0. So the status STRING makes it legible to a human reading the
    report, and the EXIT CODE makes it legible to a scheduler or wrapper that
    never opens the artifact.
    """

    def test_all_read_no_failures_is_ok(self):
        assert _mod.coverage_status(_coverage(found=3, read=3, searches=7)) == 'ok'

    def test_partial_failure_is_degraded(self):
        assert _mod.coverage_status(_coverage(found=3, read=2, failed=1, searches=5)) == 'degraded'

    def test_missing_or_empty_archive_is_no_input(self):
        assert _mod.coverage_status(_coverage(found=0, read=0)) == 'no_input'

    def test_everything_failed_is_total_failure(self):
        assert _mod.coverage_status(_coverage(found=3, read=0, failed=3)) == 'total_failure'

    def test_exit_codes(self):
        assert _mod.EXIT_CODES == {
            'ok': 0, 'degraded': 0, 'no_input': 2, 'total_failure': 3,
        }

    def test_degraded_is_still_a_usable_corpus(self):
        # Exit 0 on purpose: a partial failure that is FULLY disclosed (count
        # plus a bounded, self-disclosing example list) still yields a corpus
        # worth having. A wholesale failure does not.
        assert _mod.EXIT_CODES[_mod.coverage_status(_coverage(3, 2, failed=1))] == 0

    def test_status_is_stamped_into_the_coverage_mapping(self):
        # The artifact must carry it, not only the process exit code — a
        # coverage file read later has no exit code attached.
        coverage = _coverage(found=3, read=0, failed=3)
        assert _mod.stamp_status(coverage) is coverage
        assert coverage['status'] == 'total_failure'

    def test_zero_search_cases_are_distinguishable(self):
        # THE property, asserted directly rather than inferred from the four
        # tests above. An empty archive and an every-transcript-failed run BOTH
        # extract zero searches; if they agreed on status or exit code, a
        # silently broken extractor could masquerade as a clean empty result.
        empty = _coverage(found=0, read=0, searches=0)
        all_failed = _coverage(found=5, read=0, failed=5, searches=0)

        assert empty['searches_extracted'] == all_failed['searches_extracted'] == 0

        empty_status = _mod.coverage_status(empty)
        failed_status = _mod.coverage_status(all_failed)
        assert empty_status != failed_status
        assert _mod.EXIT_CODES[empty_status] != _mod.EXIT_CODES[failed_status]
        assert _mod.EXIT_CODES[empty_status] != 0
        assert _mod.EXIT_CODES[failed_status] != 0

    def test_a_real_archive_holding_no_searches_is_a_third_outcome(self):
        # Read everything, found nothing to extract. That is a measurement
        # ('this archive has no searches'), not a failure — and it must be
        # separately identifiable from both cases above.
        healthy_but_empty = _coverage(found=5, read=5, searches=0)
        status = _mod.coverage_status(healthy_but_empty)
        assert status == 'ok'
        assert _mod.EXIT_CODES[status] == 0
        assert status not in {
            _mod.coverage_status(_coverage(0, 0)),
            _mod.coverage_status(_coverage(5, 0, failed=5)),
        }

    def test_status_vocabulary_matches_the_exit_code_table(self):
        # No status can exist without a declared exit code, and vice versa —
        # otherwise a new status would KeyError at the worst moment, on the
        # failure path.
        for found, read, failed in [(3, 3, 0), (3, 2, 1), (0, 0, 0), (3, 0, 3)]:
            assert _mod.coverage_status(_coverage(found, read, failed)) in _mod.EXIT_CODES


# ===========================================================================
# step-15 — artifact emission
# ===========================================================================

STAMP = '20260731T120000Z'


class TestWriteCorpus:
    """Three artifacts under ``<out_root>/transcript-corpus/``.

    The layout is delegated to ``shared.memory_eval_metrics``, never restated:
    this leaf and its siblings must land in one place so a single memory-eval
    run correlates across them.
    """

    RECORDS = [
        {'schema_version': _mod.SCHEMA_VERSION, 'query': 'first', 'result_status': 'ok',
         'results': [{'id': 'a', 'score': 0.9, 'rank': 1}], 'transcript': 'x/y/z.jsonl.gz'},
        {'schema_version': _mod.SCHEMA_VERSION, 'query': 'secönd', 'result_status': 'missing',
         'results': [], 'transcript': 'x/y/z.jsonl.gz'},
    ]

    def _write(self, out_root: Path, coverage=None):
        coverage = coverage or _mod.stamp_status(_coverage(found=2, read=2, searches=2))
        return _mod.write_corpus(self.RECORDS, coverage, out_root, stamp=STAMP)

    def test_returns_three_paths(self, tmp_path):
        corpus, cov, report = self._write(tmp_path)
        assert corpus.name == f'corpus-{STAMP}.jsonl'
        assert cov.name == f'coverage-{STAMP}.json'
        assert report.name == f'report-{STAMP}.txt'

    def test_report_path_is_the_shared_helper_verbatim(self, tmp_path):
        # Pin the REUSE, not the literal layout: report_artifact_path already
        # produces <root>/<eval_id>/report-<STAMP>.txt, so it is CALLED rather
        # than re-derived — and stays correct if that helper ever moves.
        from shared.memory_eval_metrics import report_artifact_path
        _, _, report = self._write(tmp_path)
        assert report == report_artifact_path(tmp_path, _mod.EVAL_ID, STAMP)

    def test_all_three_share_the_eval_directory(self, tmp_path):
        corpus, cov, report = self._write(tmp_path)
        assert corpus.parent == cov.parent == report.parent
        assert corpus.parent == tmp_path / _mod.EVAL_ID

    def test_creates_the_directory(self, tmp_path):
        corpus, _, _ = self._write(tmp_path / 'nested' / 'deeper')
        assert corpus.exists()

    def test_corpus_is_strict_json_lines(self, tmp_path):
        corpus, _, _ = self._write(tmp_path)
        text = corpus.read_text(encoding='utf-8')
        assert text.endswith('\n')
        lines = text.splitlines()
        assert len(lines) == len(self.RECORDS)
        # One object per PHYSICAL line — the one place the shared
        # serialization idiom must be adapted rather than copied, since
        # indent=2 would break the format outright.
        for line in lines:
            assert json.loads(line)
            assert '\n' not in line

    def test_corpus_lines_are_key_sorted_and_non_ascii_preserved(self, tmp_path):
        corpus, _, _ = self._write(tmp_path)
        lines = corpus.read_text(encoding='utf-8').splitlines()
        parsed = [json.loads(line) for line in lines]
        assert [json.dumps(p, sort_keys=True, ensure_ascii=False) for p in parsed] == lines
        assert 'secönd' in lines[1], 'ensure_ascii=False keeps queries readable'

    def test_every_corpus_line_carries_the_schema_version(self, tmp_path):
        corpus, _, _ = self._write(tmp_path)
        for line in corpus.read_text(encoding='utf-8').splitlines():
            assert json.loads(line)['schema_version'] == _mod.SCHEMA_VERSION

    def test_coverage_json_is_indented_and_stable(self, tmp_path):
        _, cov, _ = self._write(tmp_path)
        text = cov.read_text(encoding='utf-8')
        assert text.endswith('\n')
        payload = json.loads(text)
        assert text == json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n'

    def test_coverage_json_carries_the_status(self, tmp_path):
        _, cov, _ = self._write(tmp_path)
        assert json.loads(cov.read_text(encoding='utf-8'))['status'] == 'ok'

    def test_stamp_defaults_to_the_shared_run_stamp(self, tmp_path, monkeypatch):
        # Reusing run_stamp/RUN_STAMP_ENV_VAR is what lets one logical
        # memory-eval run be correlated across leaves.
        from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, '20200101T000000Z')
        coverage = _mod.stamp_status(_coverage(found=1, read=1))
        corpus, cov, report = _mod.write_corpus(self.RECORDS, coverage, tmp_path)
        assert corpus.name == 'corpus-20200101T000000Z.jsonl'
        assert cov.name == 'coverage-20200101T000000Z.json'
        assert report.name == 'report-20200101T000000Z.txt'

    def test_two_identical_runs_are_byte_identical(self, tmp_path):
        first = self._write(tmp_path / 'a')
        second = self._write(tmp_path / 'b')
        for lhs, rhs in zip(first, second, strict=True):
            assert lhs.read_bytes() == rhs.read_bytes()


class TestRenderReport:
    """The operator-facing coverage report — what a human reads when pointed
    at a run. It must state what was skipped, in words."""

    def _report(self, coverage) -> str:
        return _mod.render_report(_mod.stamp_status(coverage))

    def test_states_the_status(self):
        assert 'total_failure' in self._report(_coverage(found=3, read=0, failed=3))

    def test_states_tasks_and_searches(self):
        text = self._report(_coverage(found=9, read=9, searches=42))
        assert '42' in text
        assert 'searches' in text.lower()
        assert 'transcripts' in text.lower()

    def test_states_the_parse_failure_count(self):
        text = self._report(_coverage(found=3, read=2, failed=1))
        assert 'parse' in text.lower() or 'failure' in text.lower()
        assert '1' in text

    def test_names_the_failure_examples(self):
        coverage = _coverage(found=2, read=1, failed=1)
        coverage['parse_failures']['examples'] = [
            {'transcript': '3006/-enc/bad.jsonl.gz', 'reason': 'BadGzipFile: nope'}
        ]
        text = self._report(coverage)
        assert '3006/-enc/bad.jsonl.gz' in text
        assert 'BadGzipFile' in text

    def test_discloses_example_truncation_in_words(self):
        coverage = _coverage(found=100, read=0, failed=100)
        coverage['parse_failures']['examples_truncated'] = True
        coverage['parse_failures']['examples_omitted'] = 80
        text = self._report(coverage)
        assert '80' in text

    def test_no_input_report_says_so_rather_than_reporting_zero(self):
        # The whole point: an operator reading this must not have to infer
        # 'no archive' from a zero.
        text = self._report(_coverage(found=0, read=0))
        assert 'no_input' in text


# ===========================================================================
# step-17 — THE END-TO-END SIGNAL TESTS
#
# Both halves of the capability manifest's `coverage-report-discloses-failures`
# manual check: the committed mini-fixture round-trip PLUS an all-unparseable-
# input case — driven through main() at the CLI boundary, not only at the pure
# -function boundary TestCoverageStatus covers.
# ===========================================================================


def _run_cli(archive_root, out_root, *extra) -> tuple[int, dict, list[dict], str]:
    """Drive main() in-process; return (exit_code, coverage, records, report)."""
    code = _mod.main([
        '--archive-root', str(archive_root),
        '--out-root', str(out_root),
        '--stamp', STAMP,
        *extra,
    ])
    directory = Path(out_root) / _mod.EVAL_ID
    coverage = json.loads((directory / f'coverage-{STAMP}.json').read_text(encoding='utf-8'))
    corpus_text = (directory / f'corpus-{STAMP}.jsonl').read_text(encoding='utf-8')
    records = [json.loads(line) for line in corpus_text.splitlines()]
    report = (directory / f'report-{STAMP}.txt').read_text(encoding='utf-8')
    return code, coverage, records, report


class TestMiniFixtureRoundTrip:
    """(a) The committed mini-fixture, end to end through the CLI.

    The fixture's authored facts are asserted verbatim — queries, ids, scores,
    identity, provenance — because that is what makes it a fixture rather than
    a smoke test. See tests/fixtures/README.md.
    """

    FIXTURE_TASK = '4242'
    FIXTURE_SESSION = '11111111-2222-3333-4444-555555555555'

    def _run(self, tmp_path):
        return _run_cli(FIXTURE_ARCHIVE, tmp_path)

    def test_exits_zero(self, tmp_path):
        code, _, _, _ = self._run(tmp_path)
        assert code == 0

    def test_extracts_all_three_searches(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        assert len(records) == 3

    def test_main_session_answered_search(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        [record] = [r for r in records if r['query'] == 'transcript archive layout and gz naming']
        assert record['result_status'] == 'ok'
        assert record['result_count'] == 2
        assert [(r['id'], r['score'], r['rank']) for r in record['results']] == [
            ('aaaaaaaa-1111-4aaa-8aaa-aaaaaaaaaaaa', 0.87, 1),
            ('bbbbbbbb-2222-4bbb-8bbb-bbbbbbbbbbbb', 0.42, 2),
        ]
        assert record['params'] == {'project_id': 'dark_factory', 'limit': 5}

    def test_main_session_identity_and_provenance(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        [record] = [r for r in records if r['query'] == 'transcript archive layout and gz naming']
        assert record['caller'] == {
            'agent_id': 'claude-task-4242-architect',
            'task_id': '4242',
            'role': 'architect',
        }
        assert record['task_id'] == self.FIXTURE_TASK
        assert record['session_id'] == self.FIXTURE_SESSION
        assert record['is_subagent'] is False

    def test_unanswered_search_is_emitted_as_missing(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        [record] = [r for r in records if r['query'] == 'search whose answer never arrived']
        assert record['result_status'] == 'missing'
        assert record['results'] == []

    def test_subagent_search(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        [record] = [r for r in records if r['is_subagent']]
        assert record['query'] == 'coverage report discloses parse failures'
        assert record['subagent_id'] == 'agent-abc123def4567890'
        assert record['caller']['role'] == 'code-reviewer'
        assert record['results'][0]['id'] == 'cccccccc-3333-4ccc-8ccc-cccccccccccc'

    def test_non_search_tool_uses_are_absent(self, tmp_path):
        _, _, records, _ = self._run(tmp_path)
        assert all(r['tool_name'] == SEARCH for r in records)

    def test_coverage_counts_match_the_fixture(self, tmp_path):
        _, coverage, _, _ = self._run(tmp_path)
        assert coverage['status'] == 'ok'
        assert coverage['tasks_scanned'] == 1
        assert coverage['transcripts_found'] == 2
        assert coverage['transcripts_read'] == 2
        assert coverage['searches_extracted'] == 3
        assert coverage['searches_unresolved'] == 1

    def test_blank_and_corrupt_lines_are_not_parse_failures(self, tmp_path):
        # The fixture's main transcript carries one blank and one corrupt
        # LINE. Those are reader-level skips; parse_failures counts unreadable
        # FILES. Conflating them would make every truncated transcript in the
        # live archive look like a broken one.
        _, coverage, _, _ = self._run(tmp_path)
        assert coverage['parse_failures']['count'] == 0

    def test_report_names_the_counts(self, tmp_path):
        _, _, _, report = self._run(tmp_path)
        assert 'ok' in report
        assert '3' in report


class TestZeroSearchCasesAreDistinguishableEndToEnd:
    """(b) The all-unparseable-input case, paired against an empty archive and
    an absent one — the manifest's literal pin, at the CLI boundary.

    All three extract zero searches. If they agreed on status or exit code, a
    silently broken extractor could ship as a clean empty result.
    """

    def _all_corrupt(self, tmp_path) -> Path:
        root = tmp_path / 'corrupt-archive'
        for rel in ('1/-enc/a.jsonl.gz', '2/-enc/b.jsonl.gz', '3/-enc/c.jsonl.gz'):
            _write_corrupt(root, rel)
        return root

    def test_all_unparseable_exits_three(self, tmp_path):
        code, coverage, records, report = _run_cli(
            self._all_corrupt(tmp_path), tmp_path / 'out')
        assert code == 3
        assert coverage['status'] == 'total_failure'
        assert records == []

    def test_all_unparseable_discloses_every_failure(self, tmp_path):
        _, coverage, _, _ = _run_cli(self._all_corrupt(tmp_path), tmp_path / 'out')
        assert coverage['parse_failures']['count'] == 3
        assert coverage['parse_failures']['examples']

    def test_all_unparseable_report_says_so_in_words(self, tmp_path):
        # Not merely 'searches: 0' — the report must state that nothing could
        # be read, or an operator reads a broken run as an empty archive.
        _, _, _, report = _run_cli(self._all_corrupt(tmp_path), tmp_path / 'out')
        assert 'total_failure' in report
        assert 'NONE could be read' in report or 'none could be read' in report.lower()

    def test_empty_archive_exits_two(self, tmp_path):
        empty = tmp_path / 'empty-archive'
        empty.mkdir()
        code, coverage, _, report = _run_cli(empty, tmp_path / 'out')
        assert code == 2
        assert coverage['status'] == 'no_input'
        assert 'no_input' in report

    def test_absent_archive_exits_two(self, tmp_path):
        # The real default when the script runs from a git worktree, which has
        # no data/ dir at all.
        code, coverage, _, _ = _run_cli(tmp_path / 'nope', tmp_path / 'out')
        assert code == 2
        assert coverage['status'] == 'no_input'

    def test_the_three_zero_search_runs_are_distinguishable(self, tmp_path):
        empty = tmp_path / 'empty'
        empty.mkdir()
        runs = {
            'corrupt': _run_cli(self._all_corrupt(tmp_path), tmp_path / 'out-corrupt'),
            'empty': _run_cli(empty, tmp_path / 'out-empty'),
            'absent': _run_cli(tmp_path / 'nope', tmp_path / 'out-absent'),
        }
        codes = {name: run[0] for name, run in runs.items()}
        coverages = {name: run[1] for name, run in runs.items()}

        # Same observable count, different verdicts.
        assert all(c['searches_extracted'] == 0 for c in coverages.values())
        assert coverages['corrupt']['status'] != coverages['empty']['status']
        assert codes['corrupt'] != codes['empty']
        assert codes['corrupt'] != 0
        assert codes['empty'] != 0
        # An absent archive and an empty one are the same fact, deliberately.
        assert coverages['absent']['status'] == coverages['empty']['status']
        assert codes['absent'] == codes['empty']
