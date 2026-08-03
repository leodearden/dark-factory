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
import io
import json
import sys
import types
import zlib
from pathlib import Path
from typing import Any

import pytest

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
    block: dict[str, Any] = {
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


def _gz_payload(n_records: int = 200) -> bytes:
    """A multi-record JSONL body for the two damaged-archive helpers below.
    Padded to many records so a cut at the halfway mark lands mid-stream."""
    return ''.join(
        json.dumps({'type': 'user', 'seq': i, 'pad': 'x' * 200}) + '\n'
        for i in range(n_records)
    ).encode('utf-8')


def _write_truncated(root: Path, rel: str) -> Path:
    """Half a valid gz stream — an archive write interrupted by a killed unit.

    Raw ``gzip`` signals this with ``EOFError``, which is not an ``OSError``;
    the readers normalize it so this script's ``except OSError`` counts it.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(_gz_payload())
    path.write_bytes(blob[: len(blob) // 2])
    return path


def _write_corrupt_body(root: Path, rel: str) -> Path:
    """A gz whose DEFLATE body is damaged — raw ``gzip`` raises ``zlib.error``.

    Probes for the first single-byte flip producing that shape (most flips
    only trip the trailing checksum, which is already an ``OSError``), using
    stdlib gzip directly so the helper is independent of the readers.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(_gz_payload())
    for index in range(10, len(blob)):
        candidate = bytearray(blob)
        candidate[index] ^= 0xFF
        try:
            gzip.GzipFile(fileobj=io.BytesIO(bytes(candidate))).read()
        except zlib.error:
            path.write_bytes(bytes(candidate))
            return path
        except Exception:
            continue  # a different corruption shape — keep probing
    raise AssertionError('no single-byte flip produced a zlib.error body')


def _write_undecodable(root: Path, rel: str) -> Path:
    """A structurally VALID gz whose payload is not valid UTF-8.

    A fourth shape, and the one the helpers above structurally cannot reach:
    they damage the gzip CONTAINER, and the ``_write_corrupt_body`` probe
    decompresses in binary mode, so none of them exercises the text layer.
    This file decompresses cleanly and fails one level up, where the readers'
    ``encoding='utf-8'`` wrapper meets byte 0xFF — ``UnicodeDecodeError``,
    a ``ValueError`` subclass, so it escapes ``except OSError`` unless the
    readers normalize it too. A flipped byte in stored archive data or a
    ``.jsonl`` half-written by a killed unit produces it.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(
        b'{"type": "user", "seq": 0}\n{"type": "user", "t": "\xff\xfe"}\n'))
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


# ===========================================================================
# step-19 — two readers, ONE core (INV-5 made checkable at runtime)
# ===========================================================================

FIXTURE_MAIN = (
    FIXTURE_ARCHIVE / '4242' / '-home-leo-src-dark-factory--worktrees-4242'
    / '11111111-2222-3333-4444-555555555555.jsonl'
)


class TestReaderBindings:
    """The module's readers ARE the imported ones — no local parser.

    INV-5 / PRD D9 names both readers and forbids a third. Asserting the
    BINDINGS, not just the behaviour, is what makes a future regression that
    quietly reintroduces a local parser fail a test rather than pass review.
    """

    def test_load_transcript_is_the_imported_function(self):
        import legibility.digest  # type: ignore[reportMissingImports]
        assert _mod.load_transcript is legibility.digest.load_transcript

    def test_iter_json_lines_is_the_imported_function(self):
        import legibility.inventory  # type: ignore[reportMissingImports]
        assert _mod.iter_json_lines is legibility.inventory.iter_json_lines


class TestSingleTranscriptMode:
    """``--transcript`` drives the SAME core through the OTHER reader."""

    def _cli(self, path, out_root):
        code = _mod.main([
            '--transcript', str(path), '--out-root', str(out_root), '--stamp', STAMP,
        ])
        directory = Path(out_root) / _mod.EVAL_ID
        coverage = json.loads(
            (directory / f'coverage-{STAMP}.json').read_text(encoding='utf-8'))
        corpus = (directory / f'corpus-{STAMP}.jsonl').read_text(encoding='utf-8')
        return code, coverage, [json.loads(line) for line in corpus.splitlines()]

    def test_exits_zero_and_emits_coverage(self, tmp_path):
        code, coverage, records = self._cli(FIXTURE_MAIN, tmp_path)
        assert code == 0
        assert coverage['status'] == 'ok'
        assert coverage['transcripts_found'] == 1
        assert coverage['transcripts_read'] == 1
        assert len(records) == 2

    def test_matches_what_the_scan_path_produced_for_the_same_file(self, tmp_path):
        # THE reuse assertion: load_transcript (slurp) and iter_json_lines
        # (stream) feed ONE core, so for the same file they must agree
        # exactly. Any divergence means a second parser has appeared.
        #
        # This used to exclude the five path-derived fields, because
        # --transcript mode passed only `path.name` to parse_archive_path and
        # so reported them all null. That exclusion hid a real defect
        # (reviewer_comprehensive/correctness): the debug mode's provenance
        # was not merely absent, it was WRONG for a subagent transcript, which
        # came out labelled is_subagent=False. Now that
        # archive_relative_slice recovers the slice, the two modes must agree
        # on EVERY field — and the comparison is the whole record.
        _, _, via_slurp = self._cli(FIXTURE_MAIN, tmp_path / 'single')
        _, _, scan_records, _ = _run_cli(FIXTURE_ARCHIVE, tmp_path / 'scan')
        rel = FIXTURE_MAIN.relative_to(FIXTURE_ARCHIVE).as_posix()
        via_scan = [r for r in scan_records if r['transcript'] == rel]

        assert via_slurp == via_scan
        assert len(via_slurp) == 2

    def test_a_subagent_transcript_is_not_reported_as_a_main_session(self, tmp_path):
        # The specific wrong answer the old `path.name` slice produced. A
        # subagent transcript mined through the debug path claimed
        # is_subagent=False with a null subagent_id — confidently wrong, and
        # exactly what an operator comparing modes would trust.
        subagent = (
            FIXTURE_ARCHIVE / '4242' / '-home-leo-src-dark-factory--worktrees-4242'
            / '11111111-2222-3333-4444-555555555555' / 'subagents'
            / 'agent-abc123def4567890.jsonl'
        )
        _, coverage, records = self._cli(subagent, tmp_path / 'single')

        assert records, 'the fixture subagent transcript holds a search'
        for record in records:
            assert record['is_subagent'] is True
            assert record['subagent_id'] == 'agent-abc123def4567890'
            assert record['task_id'] == '4242'
            assert record['session_id'] == '11111111-2222-3333-4444-555555555555'
        # The guard at the tasks_scanned line is now reachable, where before
        # task_id was structurally always None and the branch was dead code.
        assert coverage['tasks_scanned'] == 1

    def test_a_transcript_outside_an_archive_recovers_nothing(self, tmp_path):
        # The other half of the contract: recovering nothing is correct for a
        # file that is genuinely not in an archive tree. Inventing provenance
        # from whatever directories happen to be above it would be worse than
        # the null it replaces.
        loose = tmp_path / 'scratch' / 'somewhere' / 'session.jsonl.gz'
        loose.parent.mkdir(parents=True)
        with gzip.open(loose, 'wt', encoding='utf-8') as f:
            f.write(json.dumps(_briefing('claude-task-9001-implementer')) + '\n')
            f.write(json.dumps(_tool_use('toolu_loose', 'a loose search')) + '\n')

        _, coverage, records = self._cli(loose, tmp_path / 'out')

        assert [r['task_id'] for r in records] == [None]
        assert records[0]['session_id'] is None
        assert records[0]['is_subagent'] is False
        assert records[0]['transcript'] == 'session.jsonl.gz'
        assert coverage['tasks_scanned'] == 0
        # The caller identity read from the briefing is unaffected — it comes
        # from the record bodies, not the path.
        assert records[0]['caller']['task_id'] == '9001'

    def test_unreadable_file_is_total_failure_not_a_traceback(self, tmp_path):
        bad = tmp_path / 'broken.jsonl.gz'
        bad.write_bytes(b'not gzip at all\n')
        code, coverage, records = self._cli(bad, tmp_path / 'out')
        assert code == 3
        assert coverage['status'] == 'total_failure'
        assert coverage['parse_failures']['count'] == 1
        assert records == []

    def test_absent_file_is_total_failure(self, tmp_path):
        code, coverage, _ = self._cli(tmp_path / 'nope.jsonl.gz', tmp_path / 'out')
        assert code == 3
        assert coverage['status'] == 'total_failure'


# ===========================================================================
# step-27 — a PARTIALLY-WRITTEN archived transcript is counted, not fatal
#
# The `except OSError` handlers in this script are only as good as the
# contract the readers keep. Measured, an unreadable archived transcript
# arrives as FOUR different types, and only bad magic is already an OSError:
#
#     bad magic         -> gzip.BadGzipFile     (an OSError subclass)
#     truncated stream  -> EOFError             (not an OSError)
#     corrupt body      -> zlib.error           (not an OSError)
#     undecodable byte  -> UnicodeDecodeError   (a ValueError, not an OSError)
#
# The last is the one that survives a fix aimed only at the gzip layer: the
# container decompresses fine and the fault is at the readers' strict
# `encoding='utf-8'` text wrapper, so it is reachable on a plain `.jsonl`
# too, and being a ValueError it escapes `except OSError` however the three
# gzip shapes are handled.
#
# Before the readers normalized them, ONE truncated file among the live
# archive's ~2814 aborted the entire run with a traceback: no corpus, no
# coverage file, no report, and an exit status outside the documented
# ok/degraded/no_input/total_failure vocabulary. That is precisely the
# ambiguity this module exists to make impossible, so it is pinned here at
# the CLI boundary rather than only at the reader.
#
# The archive is fire-and-forget runtime state written by the running fleet:
# a unit killed mid-write and a file read while still being compressed both
# produce exactly these shapes. No handler in the script changes — these
# tests are what prove `except OSError` is now sufficient rather than
# assuming it.
# ===========================================================================


class TestPartiallyWrittenTranscriptsAreCounted:
    GOOD_REL = '5150/-enc-a/good-session.jsonl.gz'
    TRUNCATED_REL = '5150/-enc-a/truncated-session.jsonl.gz'
    CORRUPT_BODY_REL = '5151/-enc-b/corrupt-body-session.jsonl.gz'
    UNDECODABLE_REL = '5152/-enc-c/undecodable-session.jsonl.gz'

    def _good(self, root: Path) -> Path:
        _write_transcript(root, self.GOOD_REL, [
            _briefing('claude-task-5150-implementer'),
            _tool_use('toolu_ok', 'the search that must survive'),
            _tool_result('toolu_ok', {'results': [_result('mem-1', 0.87)]}),
        ])
        return root

    def _one_bad_one_good(self, tmp_path: Path) -> Path:
        root = self._good(tmp_path / 'archive')
        _write_truncated(root, self.TRUNCATED_REL)
        return root

    def _one_undecodable_one_good(self, tmp_path: Path) -> Path:
        root = self._good(tmp_path / 'undecodable-archive')
        _write_undecodable(root, self.UNDECODABLE_REL)
        return root

    def _all_bad(self, tmp_path: Path) -> Path:
        root = tmp_path / 'all-bad-archive'
        _write_truncated(root, self.TRUNCATED_REL)
        _write_corrupt_body(root, self.CORRUPT_BODY_REL)
        _write_undecodable(root, self.UNDECODABLE_REL)
        return root

    # -- (a) partial corruption: degraded, and the run survives -------------

    def test_one_truncated_file_does_not_cost_the_others(self, tmp_path):
        # The whole point. One bad file among many must cost exactly one
        # counted failure, not the entire corpus.
        code, coverage, records, _ = _run_cli(
            self._one_bad_one_good(tmp_path), tmp_path / 'out')

        assert code == 0
        assert coverage['status'] == 'degraded'
        assert [r['query'] for r in records] == ['the search that must survive']
        assert records[0]['results'][0]['id'] == 'mem-1'

    def test_the_truncated_file_is_named_in_the_disclosed_failures(self, tmp_path):
        _, coverage, _, report = _run_cli(
            self._one_bad_one_good(tmp_path), tmp_path / 'out')

        failures = coverage['parse_failures']
        assert failures['count'] == 1
        assert coverage['transcripts_found'] == 2
        assert coverage['transcripts_read'] == 1
        assert [ex['transcript'] for ex in failures['examples']] == [self.TRUNCATED_REL]
        assert self.TRUNCATED_REL in report

    # -- (b) wholesale corruption: total_failure, said in words -------------

    def test_every_file_unreadable_exits_three(self, tmp_path):
        code, coverage, records, _ = _run_cli(self._all_bad(tmp_path), tmp_path / 'out')

        assert code == 3
        assert coverage['status'] == 'total_failure'
        assert coverage['parse_failures']['count'] == 3
        assert records == []

    def test_every_file_unreadable_report_says_so_in_words(self, tmp_path):
        # Not merely 'searches: 0' — an operator must not read a wholly
        # unreadable archive as an empty one.
        _, _, _, report = _run_cli(self._all_bad(tmp_path), tmp_path / 'out')

        assert 'total_failure' in report
        assert 'none could be read' in report.lower()

    # -- (c) both shapes reach the counted path, distinguishably ------------

    def test_the_three_corruption_shapes_report_different_reasons(self, tmp_path):
        # All are normalized to OSError so one handler catches them, but the
        # disclosed reason must still say WHICH — an operator triaging the
        # archive writer needs to tell a half-written file from a damaged one
        # from one whose bytes are not text, since those are three different
        # things to go and fix.
        _, coverage, _, _ = _run_cli(self._all_bad(tmp_path), tmp_path / 'out')

        reasons = {ex['transcript']: ex['reason'] for ex in
                   coverage['parse_failures']['examples']}
        assert set(reasons) == {
            self.TRUNCATED_REL, self.CORRUPT_BODY_REL, self.UNDECODABLE_REL}
        assert len(set(reasons.values())) == 3
        assert 'end-of-stream' in reasons[self.TRUNCATED_REL]
        assert 'decompressing' in reasons[self.CORRUPT_BODY_REL]
        assert '0xff' in reasons[self.UNDECODABLE_REL].lower()

    # -- (b2) the decode shape, on its own, at the archive boundary ---------

    def test_one_undecodable_file_does_not_cost_the_others(self, tmp_path):
        # The regression this class exists to prevent, for the shape that was
        # still leaking: UnicodeDecodeError is a ValueError, not an OSError,
        # so before the readers normalized it a single non-UTF-8 byte anywhere
        # in the ~2814-file live archive aborted scan_archive with a traceback
        # — no corpus, no coverage.json, no report, and an exit status outside
        # the documented ok/degraded/no_input/total_failure vocabulary. That
        # is precisely the "silently broken extractor masquerading as a clean
        # result" failure the module's contract forbids, so it is pinned at
        # the CLI boundary and not only at the reader.
        code, coverage, records, report = _run_cli(
            self._one_undecodable_one_good(tmp_path), tmp_path / 'out')

        assert code == 0
        assert coverage['status'] == 'degraded'
        assert [r['query'] for r in records] == ['the search that must survive']
        assert coverage['transcripts_found'] == 2
        assert coverage['transcripts_read'] == 1
        assert coverage['parse_failures']['count'] == 1
        assert [ex['transcript'] for ex in
                coverage['parse_failures']['examples']] == [self.UNDECODABLE_REL]
        assert self.UNDECODABLE_REL in report

    # -- (d) the other reader, via --transcript ----------------------------

    def test_single_transcript_mode_on_a_truncated_file(self, tmp_path):
        # --transcript reads through load_transcript, the OTHER reader, and
        # must answer the same way rather than raising through main().
        truncated = _write_truncated(tmp_path, 'lone/truncated.jsonl.gz')
        out_root = tmp_path / 'out'

        code = _mod.main([
            '--transcript', str(truncated),
            '--out-root', str(out_root), '--stamp', STAMP,
        ])

        coverage = json.loads(
            (out_root / _mod.EVAL_ID / f'coverage-{STAMP}.json').read_text(
                encoding='utf-8'))
        assert code == 3
        assert coverage['status'] == 'total_failure'
        assert coverage['parse_failures']['count'] == 1

    # -- (e) distinguishability, extended to this failure class ------------

    def test_a_corrupt_archive_still_reads_differently_from_an_empty_one(self, tmp_path):
        # Mirrors TestZeroSearchCasesAreDistinguishableEndToEnd for the shapes
        # that used to escape entirely: normalizing them into the counted path
        # must not have quietly converted a wholly unreadable archive into
        # something that looks like a clean empty one.
        empty = tmp_path / 'empty'
        empty.mkdir()
        corrupt_code, corrupt_cov, _, _ = _run_cli(
            self._all_bad(tmp_path), tmp_path / 'out-corrupt')
        empty_code, empty_cov, _, _ = _run_cli(empty, tmp_path / 'out-empty')

        assert corrupt_cov['searches_extracted'] == empty_cov['searches_extracted'] == 0
        assert corrupt_cov['status'] != empty_cov['status']
        assert corrupt_code != empty_code
        assert corrupt_code != 0
        assert empty_code != 0


# ===========================================================================
# Amendment pass — the gaps a code review found after verification passed.
# ===========================================================================


class TestArchiveRelativeSlice:
    """``--transcript`` recovers the archive-relative tail of an absolute path.

    The unit half of the correctness fix pinned end-to-end in
    ``TestSingleTranscriptMode``. Operators hand this mode a path from their
    shell — usually absolute, so the archive root is still on the front. The
    scan path has already stripped that prefix; this is what lets the debug
    path arrive at the same string, and therefore at the same provenance.
    """

    ENC = '-home-leo-src-dark-factory'

    def test_recovers_a_main_session_slice_from_an_absolute_path(self):
        absolute = f'/home/leo/src/df/data/orchestrator/agent-transcripts/3214/{self.ENC}/sess.jsonl.gz'

        assert _mod.archive_relative_slice(absolute) == f'3214/{self.ENC}/sess.jsonl.gz'

    def test_recovers_a_subagent_slice(self):
        absolute = (
            f'/var/archive/3214/{self.ENC}/sess/subagents/agent-ab12.jsonl.gz'
        )

        assert _mod.archive_relative_slice(absolute) == (
            f'3214/{self.ENC}/sess/subagents/agent-ab12.jsonl.gz'
        )

    def test_an_already_relative_slice_is_returned_unchanged(self):
        rel = f'3214/{self.ENC}/sess.jsonl.gz'

        assert _mod.archive_relative_slice(rel) == rel

    def test_a_non_numeric_task_id_still_matches(self):
        # Measured on the live archive: 4 of the 409 task dirs are
        # `df_task_12`-style eval-run ids, not numbers. A digits-only guard on
        # the task component would silently drop them.
        rel = f'df_task_12/{self.ENC}/sess.jsonl.gz'

        assert _mod.archive_relative_slice(rel) == rel

    def test_an_arbitrary_three_component_path_is_not_relabelled(self):
        # The guard that keeps this recovery honest. Without the encoded-cwd
        # check, ANY path with three trailing components would be read as an
        # archive slice and its grandparent directory reported as a task id —
        # a confidently wrong answer, which is worse than the null it would
        # replace.
        assert _mod.archive_relative_slice('/tmp/pytest-42/scratch/sess.jsonl.gz') == (
            'sess.jsonl.gz'
        )

    def test_a_subagents_dir_without_an_encoded_cwd_is_not_relabelled(self):
        assert _mod.archive_relative_slice(
            '/tmp/a/b/c/subagents/agent-ab12.jsonl.gz'
        ) == 'agent-ab12.jsonl.gz'

    def test_a_bare_file_name_is_returned_as_is(self):
        assert _mod.archive_relative_slice('sess.jsonl.gz') == 'sess.jsonl.gz'

    def test_the_recovered_slice_round_trips_through_parse_archive_path(self):
        # The property that makes the whole fix work: what this returns is
        # exactly what the scan path would have passed, so both modes reach
        # parse_archive_path with the same string.
        absolute = f'/var/archive/3214/{self.ENC}/sess/subagents/agent-ab12.jsonl.gz'

        parsed = _mod.parse_archive_path(_mod.archive_relative_slice(absolute))

        assert parsed == {
            'task_id': '3214',
            'session_id': 'sess',
            'is_subagent': True,
            'subagent_id': 'agent-ab12',
        }


class TestStampIsValidated:
    """An operator-supplied ``--stamp`` gets the shared shape check.

    All three artifact names carry the stamp, and every declared consumer
    (leaf eta, the E3/E8 runners) picks a run by ordering those filenames as
    strings. A stamp of the wrong shape sorts outside the chronological
    sequence, so ``--stamp latest`` would make a window loader select the
    wrong run with nothing to signal it. ``shared.memory_eval_metrics`` treats
    a caller-supplied override as the one thing it must validate; this script
    was re-implementing the stamp-to-path step without that check
    (reviewer_comprehensive/robustness).
    """

    def _archive(self, tmp_path: Path) -> Path:
        root = tmp_path / 'archive'
        _write_transcript(root, f'7000/-enc/{"s" * 4}.jsonl.gz', [
            _briefing('claude-task-7000-implementer'),
            _tool_use('toolu_a', 'a search'),
            _tool_result('toolu_a', {'results': [_result('mem-1', 0.9)]}),
        ])
        return root

    def test_a_valid_stamp_is_accepted(self, tmp_path):
        code = _mod.main([
            '--archive-root', str(self._archive(tmp_path)),
            '--out-root', str(tmp_path / 'out'), '--stamp', STAMP,
        ])

        assert code == 0
        assert (tmp_path / 'out' / _mod.EVAL_ID / f'corpus-{STAMP}.jsonl').exists()

    def test_a_malformed_stamp_exits_with_the_documented_code(self, tmp_path, capsys):
        code = _mod.main([
            '--archive-root', str(self._archive(tmp_path)),
            '--out-root', str(tmp_path / 'out'), '--stamp', 'latest',
        ])

        assert code == _mod.EXIT_BAD_STAMP
        # The rejected value and the reason both reach stderr, so an operator
        # sees which string was refused rather than only a numeric code.
        stderr = capsys.readouterr().err
        assert 'latest' in stderr
        assert '--stamp' in stderr

    def test_a_malformed_stamp_writes_nothing(self, tmp_path):
        # Rejected BEFORE any path is built, so a bad run leaves no
        # half-named artifact and no eval directory to mislead a later reader.
        out_root = tmp_path / 'out'

        _mod.main([
            '--archive-root', str(self._archive(tmp_path)),
            '--out-root', str(out_root), '--stamp', 'latest',
        ])

        assert not (out_root / _mod.EVAL_ID).exists()

    def test_a_malformed_env_stamp_is_a_code_not_a_traceback(self, tmp_path, monkeypatch):
        # The other route an operator-supplied string enters by. run_stamp()
        # raises MetricSchemaError for it, which escaped main() as exit 1 —
        # outside this module's documented vocabulary entirely.
        monkeypatch.setenv(_mod.RUN_STAMP_ENV_VAR, 'not-a-stamp')

        code = _mod.main([
            '--archive-root', str(self._archive(tmp_path)),
            '--out-root', str(tmp_path / 'out'),
        ])

        assert code == _mod.EXIT_BAD_STAMP

    def test_write_corpus_rejects_the_override_directly(self, tmp_path):
        from shared.memory_eval_metrics import MetricSchemaError

        with pytest.raises(MetricSchemaError):
            _mod.write_corpus([], _mod._new_coverage(), tmp_path, stamp='2026-07-31')

    def test_the_bad_stamp_code_is_outside_the_status_table(self):
        # Deliberately not a fifth status: the four describe how much of an
        # archive a completed run covered, and a rejected stamp means no run
        # happened at all. Folding it into total_failure would assert
        # transcripts were found and none could be read.
        assert _mod.EXIT_BAD_STAMP not in _mod.EXIT_CODES.values()


class TestDefaultArchiveRoot:
    """The path a bare production invocation takes, and its own worst case.

    ``default_archive_root``'s docstring argues that getting this wrong is
    this script's worst failure mode — every worktree run of a correct script
    would report ``no_input``, which trains operators to read exit 2 as normal
    and hollows out the signal the script exists to create. It had no test
    (reviewer_comprehensive/test-coverage).
    """

    def _scope_module(self, monkeypatch, resolver):
        module = types.ModuleType('fused_memory.models.scope')
        module.resolve_main_checkout = resolver  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, 'fused_memory.models.scope', module)

    def test_resolves_against_the_main_checkout(self, tmp_path, monkeypatch):
        self._scope_module(monkeypatch, lambda _start: str(tmp_path / 'main'))

        assert _mod.default_archive_root() == (
            tmp_path / 'main' / 'data' / 'orchestrator' / 'agent-transcripts'
        )

    def test_falls_back_to_the_repo_root_when_resolution_fails(self, monkeypatch):
        # resolve_main_checkout raises ValueError outside a git working tree
        # or when git is unavailable (a hermetic temp-dir test, say). The
        # fallback still canonicalizes rather than raising.
        def _boom(_start):
            raise ValueError('not a git working tree')

        self._scope_module(monkeypatch, _boom)

        assert _mod.default_archive_root() == (
            Path(_mod._REPO_ROOT).resolve() / _mod.ARCHIVE_RELPATH
        )

    def test_it_is_the_default_main_uses(self, tmp_path, monkeypatch):
        # Wiring, not just the helper: with no --archive-root, main() must
        # actually scan what default_archive_root returns.
        archive = tmp_path / 'main' / _mod.ARCHIVE_RELPATH
        _write_transcript(archive, '8100/-enc/sess.jsonl.gz', [
            _briefing('claude-task-8100-implementer'),
            _tool_use('toolu_d', 'found via the default root'),
            _tool_result('toolu_d', {'results': [_result('mem-9', 0.5)]}),
        ])
        self._scope_module(monkeypatch, lambda _start: str(tmp_path / 'main'))

        code = _mod.main(['--out-root', str(tmp_path / 'out'), '--stamp', STAMP])

        corpus = (tmp_path / 'out' / _mod.EVAL_ID / f'corpus-{STAMP}.jsonl')
        records = [json.loads(line) for line in corpus.read_text().splitlines()]
        assert code == 0
        assert [r['query'] for r in records] == ['found via the default root']

    def test_an_absent_default_root_is_no_input_not_a_crash(self, tmp_path, monkeypatch):
        # The measured worktree case: a worktree has no data/ dir at all, so
        # the default resolves to a non-existent path. That must land in
        # no_input / exit 2 — loudly and by name — rather than in a silent
        # empty corpus or a traceback.
        self._scope_module(monkeypatch, lambda _start: str(tmp_path / 'no-such-checkout'))

        code = _mod.main(['--out-root', str(tmp_path / 'out'), '--stamp', STAMP])

        coverage = json.loads(
            (tmp_path / 'out' / _mod.EVAL_ID / f'coverage-{STAMP}.json').read_text())
        assert code == _mod.EXIT_CODES['no_input'] == 2
        assert coverage['status'] == 'no_input'


class TestFlagWiring:
    """``--tool-name`` and ``--max-failure-examples`` through ``main()``.

    Both were only ever exercised via the ``scan_archive(...)`` keyword
    arguments, so a typo in ``dest=``/``action='append'`` or in the
    ``frozenset(args.tool_names) if args.tool_names else SEARCH_TOOL_NAMES``
    fallback would have shipped green (reviewer_comprehensive/test-coverage).
    """

    LEGACY = 'legacy__memory_search'

    def _archive(self, tmp_path: Path) -> Path:
        root = tmp_path / 'archive'
        _write_transcript(root, '8200/-enc/sess.jsonl.gz', [
            _briefing('claude-task-8200-implementer'),
            _tool_use('toolu_new', 'current name'),
            _tool_result('toolu_new', {'results': [_result('mem-1', 0.9)]}),
            _tool_use('toolu_old', 'historical name', name=self.LEGACY),
            _tool_result('toolu_old', {'results': [_result('mem-2', 0.8)]}),
        ])
        return root

    def test_default_matches_only_the_current_tool_name(self, tmp_path):
        _, _, records, _ = _run_cli(self._archive(tmp_path), tmp_path / 'out')

        assert [r['query'] for r in records] == ['current name']

    def test_tool_name_flag_selects_a_historical_name(self, tmp_path):
        # The archive spans months of tool-name history; the flag exists so a
        # rename is minable without editing the script.
        _, _, records, _ = _run_cli(
            self._archive(tmp_path), tmp_path / 'out', '--tool-name', self.LEGACY)

        assert [r['query'] for r in records] == ['historical name']
        assert [r['tool_name'] for r in records] == [self.LEGACY]

    def test_tool_name_flag_is_repeatable(self, tmp_path):
        _, _, records, _ = _run_cli(
            self._archive(tmp_path), tmp_path / 'out',
            '--tool-name', self.LEGACY, '--tool-name', SEARCH)

        assert sorted(r['query'] for r in records) == ['current name', 'historical name']

    def test_max_failure_examples_flag_caps_the_disclosed_list(self, tmp_path):
        root = tmp_path / 'archive'
        for i in range(3):
            path = root / '8300' / '-enc' / f'bad-{i}.jsonl.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not gzip at all\n')

        _, coverage, _, report = _run_cli(
            root, tmp_path / 'out', '--max-failure-examples', '1')

        # The COUNT is never capped — only the example list is, and the
        # truncation discloses itself in both the mapping and the report.
        assert coverage['parse_failures']['count'] == 3
        assert len(coverage['parse_failures']['examples']) == 1
        assert coverage['parse_failures']['examples_truncated'] is True
        assert coverage['parse_failures']['examples_omitted'] == 2
        assert 'more' in report
