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
