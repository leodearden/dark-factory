"""Tests for scripts/legibility/digest.py — the deterministic confusion-digest
extractor (Claude Code session transcript JSONL -> 5-15KB markdown digest).

Fixture helpers below construct synthetic transcript records that mirror the
real Claude Code JSONL schema (observed in
``~/.claude/projects/<enc>/*.jsonl``):
  - assistant records: ``message.content`` is a list of blocks
    (thinking | text | tool_use); a tool_use block is
    ``{type, id, name, input, caller}``.
  - user records: ``message.content`` is a str (human turn) OR a list of
    blocks (text | tool_result); a tool_result block is
    ``{type, tool_use_id, content, is_error}``.
  - ``isSidechain`` flags subagent turns; ``isMeta`` flags system-injected
    user turns (both observed as top-level record keys).

Every helper below returns a plain dict (or writes one) — no behavior, pure
scaffolding reused by every test class in this file. Import the target
module as `import digest as mod`, resolved via scripts/tests/conftest.py's
sys.path insertion (both scripts/ and scripts/legibility/ are on sys.path;
no package __init__ needed).
"""
from __future__ import annotations

import json

import digest as mod


def _assistant(*blocks):
    """Build a synthetic assistant record wrapping the given content blocks."""
    return {
        'type': 'assistant',
        'message': {'role': 'assistant', 'content': list(blocks)},
        'isSidechain': False,
    }


def _text(s):
    """Build an assistant/user 'text' content block."""
    return {'type': 'text', 'text': s}


def _thinking(s):
    """Build an assistant 'thinking' content block."""
    return {'type': 'thinking', 'thinking': s}


def _tool_use(name, input, id='tool-1'):
    """Build an assistant 'tool_use' content block."""
    return {
        'type': 'tool_use',
        'id': id,
        'name': name,
        'input': input,
        'caller': {'type': 'direct'},
    }


def _user_text(s):
    """Build a synthetic genuine (non-meta, non-sidechain) human user turn."""
    return {
        'type': 'user',
        'message': {'role': 'user', 'content': s},
        'isSidechain': False,
        'isMeta': False,
    }


def _tool_result(tool_use_id, content, is_error=False):
    """Build a synthetic user record wrapping a single tool_result block."""
    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': tool_use_id,
                'content': content,
                'is_error': is_error,
            }],
        },
        'isSidechain': False,
        'isMeta': False,
    }


def _meta_user(s):
    """Build a system-injected (isMeta=True) user turn."""
    rec = _user_text(s)
    rec['isMeta'] = True
    return rec


def _sidechain(rec):
    """Return a copy of *rec* flagged as a subagent (isSidechain=True) turn."""
    out = dict(rec)
    out['isSidechain'] = True
    return out


def _write_jsonl(tmp_path, records):
    """Serialize *records* to ``<tmp_path>/transcript.jsonl``; return its Path."""
    path = tmp_path / 'transcript.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r))
            f.write('\n')
    return path


# ---------------------------------------------------------------------------
# load_transcript — ordered parse; blank/malformed lines degrade (skip)
# rather than raise (mirrors analyze_speculation_depth.load_events).
# ---------------------------------------------------------------------------

class TestLoadTranscript:
    def test_parses_ordered_records(self, tmp_path):
        records = [_user_text('first'), _assistant(_text('second')), _user_text('third')]
        path = _write_jsonl(tmp_path, records)

        loaded = mod.load_transcript(path)

        assert [r['message']['content'] for r in loaded] == [
            'first', [{'type': 'text', 'text': 'second'}], 'third',
        ]

    def test_skips_blank_and_malformed_lines_without_raising(self, tmp_path):
        path = tmp_path / 'transcript.jsonl'
        with path.open('w', encoding='utf-8') as f:
            f.write(json.dumps(_user_text('one')) + '\n')
            f.write('\n')  # blank line
            f.write('{not valid json,,,\n')  # malformed line
            f.write(json.dumps(_user_text('two')) + '\n')

        loaded = mod.load_transcript(path)  # must not raise

        assert [r['message']['content'] for r in loaded] == ['one', 'two']


# ---------------------------------------------------------------------------
# iter_user_turns — genuine non-sidechain, non-meta human turns only.
# User corrections are gold (PRD Sec 5): this is the highest-priority section.
# ---------------------------------------------------------------------------

class TestIterUserTurns:
    def test_includes_string_content_turn(self):
        turns = mod.iter_user_turns([_user_text('hello there')])

        assert len(turns) == 1
        assert turns[0]['text'] == 'hello there'
        assert turns[0]['index'] == 0

    def test_includes_text_block_content_turn(self):
        rec = {
            'type': 'user',
            'message': {'role': 'user', 'content': [_text('typed message')]},
            'isSidechain': False,
            'isMeta': False,
        }

        turns = mod.iter_user_turns([rec])

        assert len(turns) == 1
        assert turns[0]['text'] == 'typed message'

    def test_excludes_tool_result_only_content(self):
        records = [_tool_result('tool-1', 'some result content')]

        assert mod.iter_user_turns(records) == []

    def test_excludes_meta_user_turns(self):
        records = [_meta_user('Continue from where you left off.')]

        assert mod.iter_user_turns(records) == []

    def test_excludes_sidechain_turns(self):
        records = [_sidechain(_user_text('subagent instructions'))]

        assert mod.iter_user_turns(records) == []

    def test_excludes_assistant_and_other_record_types(self):
        records = [_assistant(_text('hi')), {'type': 'attachment'}]

        assert mod.iter_user_turns(records) == []

    def test_preserves_order_and_position_across_mixed_records(self):
        records = [
            _user_text('first'),
            _assistant(_text('assistant reply')),
            _meta_user('system injected'),
            _user_text('second'),
        ]

        turns = mod.iter_user_turns(records)

        assert [t['text'] for t in turns] == ['first', 'second']
        assert [t['index'] for t in turns] == [0, 3]


# ---------------------------------------------------------------------------
# iter_error_neighborhoods — is_error tool_result blocks ONLY (never a
# substring "FAIL" match -- the core of the decoy-FAIL decision), paired
# with the preceding assistant attempt via tool_use_id.
# ---------------------------------------------------------------------------

class TestIterErrorNeighborhoods:
    def test_pairs_error_result_with_preceding_attempt(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'false'}, id='tu-1')),
            _tool_result('tu-1', 'Exit code 1', is_error=True),
        ]

        neighborhoods = mod.iter_error_neighborhoods(records)

        assert len(neighborhoods) == 1
        n = neighborhoods[0]
        assert n['attempt_tool'] == 'Bash'
        assert n['attempt_input_summary'] == json.dumps({'command': 'false'}, sort_keys=True)
        assert n['error_content'] == 'Exit code 1'
        assert n['index'] == 1

    def test_ignores_non_error_result(self):
        records = [
            _assistant(_tool_use('Read', {'file_path': '/tmp/x'}, id='tu-2')),
            _tool_result('tu-2', 'file contents', is_error=False),
        ]

        assert mod.iter_error_neighborhoods(records) == []

    def test_only_error_result_returned_among_mixed(self):
        records = [
            _assistant(
                _tool_use('Bash', {'command': 'false'}, id='tu-1'),
                _tool_use('Read', {'file_path': '/tmp/x'}, id='tu-2'),
            ),
            _tool_result('tu-2', 'file contents', is_error=False),
            _tool_result('tu-1', 'Exit code 1', is_error=True),
        ]

        neighborhoods = mod.iter_error_neighborhoods(records)

        assert len(neighborhoods) == 1
        assert neighborhoods[0]['attempt_tool'] == 'Bash'

    def test_unmatched_tool_use_id_does_not_crash(self):
        # No preceding tool_use carries this tool_use_id (e.g. the attempt
        # record was truncated off the front of the transcript window).
        records = [_tool_result('orphan-id', 'boom', is_error=True)]

        neighborhoods = mod.iter_error_neighborhoods(records)

        assert len(neighborhoods) == 1
        assert neighborhoods[0]['attempt_tool'] is None
        assert neighborhoods[0]['attempt_input_summary'] is None
        assert neighborhoods[0]['error_content'] == 'boom'
