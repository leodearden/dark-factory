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


# ---------------------------------------------------------------------------
# iter_self_corrections — curated markers in assistant TEXT blocks only.
# Native-carrier scoping: the same phrase inside a tool_result or inside a
# Write/Edit tool_use input (an agent authoring test data) is not a signal.
# ---------------------------------------------------------------------------

class TestIterSelfCorrections:
    def test_detects_marker_in_assistant_text(self):
        records = [_assistant(_text("Wait, that's wrong — I need to redo this."))]

        hits = mod.iter_self_corrections(records)

        assert len(hits) == 1
        assert hits[0]['pattern'] == "that's wrong"
        assert "that's wrong" in hits[0]['context'].lower()

    def test_detects_multiple_distinct_patterns_across_records(self):
        records = [
            _assistant(_text('My mistake, I misread the file.')),
            _assistant(_text('Actually, the fix belongs elsewhere.')),
        ]

        hits = mod.iter_self_corrections(records)

        assert [h['pattern'] for h in hits] == ['my mistake', 'actually,']

    def test_ignores_marker_in_tool_result_content(self):
        records = [_tool_result('tu-1', "that's wrong, retry", is_error=False)]

        assert mod.iter_self_corrections(records) == []

    def test_ignores_marker_inside_write_tool_use_input(self):
        records = [
            _assistant(_tool_use(
                'Write', {'file_path': '/tmp/x.py', 'content': '# my mistake\n'},
            )),
        ]

        assert mod.iter_self_corrections(records) == []

    def test_ignores_thinking_blocks(self):
        # Only genuine assistant TEXT blocks are the self-correction
        # carrier -- thinking is internal deliberation, never shown to the
        # user, so it is not scoped in.
        records = [_assistant(_thinking('actually, this approach is wrong'))]

        assert mod.iter_self_corrections(records) == []


# ---------------------------------------------------------------------------
# iter_not_found / iter_df_guards / iter_interrupts — secondary scalar
# signals, each carrier-scoped.
# ---------------------------------------------------------------------------

class TestSecondarySignals:
    def test_not_found_detects_pattern_in_tool_result(self):
        records = [_tool_result('tu-1', 'bash: foo: command not found', is_error=True)]

        hits = mod.iter_not_found(records)

        assert len(hits) == 1
        assert hits[0]['pattern'] == 'command not found'

    def test_not_found_ignores_assistant_text(self):
        records = [_assistant(_text('The file does not exist yet, I will create it.'))]

        assert mod.iter_not_found(records) == []

    def test_not_found_ignores_write_tool_use_input(self):
        records = [_assistant(_tool_use(
            'Write', {'file_path': '/tmp/x', 'content': 'No such file or directory'},
        ))]

        assert mod.iter_not_found(records) == []

    def test_df_guard_detects_trip_context_in_tool_result(self):
        # Real PathGuardVerdict trip literal (fused_memory path_scope_guard.py)
        records = [_tool_result(
            'tu-1',
            "DarkFactoryPathScopeViolation: task references paths owned by "
            "another project ('foo/bar.py',) but was filed under project 'x'.",
            is_error=True,
        )]

        hits = mod.iter_df_guards(records)

        assert len(hits) == 1
        assert hits[0]['pattern'] == 'darkfactorypathscopeviolation'

    def test_df_guard_detects_phantom_done_gate_error_code(self):
        # Real phantom-done gate trip literal (task_interceptor.py/scheduler.py)
        records = [_tool_result(
            'tu-1', "error: done_gate_missing_files -- cannot mark done", is_error=True,
        )]

        hits = mod.iter_df_guards(records)

        assert len(hits) == 1
        assert hits[0]['pattern'] == 'done_gate_missing_files'

    def test_df_guard_ignores_bare_category_mention(self):
        # scope_violation/phantom-done are common CATEGORY NAMES mentioned in
        # prose (task descriptions, skill instructions) -- not a real trip.
        # The bare, lowercase-underscore label never matches: only the real
        # trip literals (the camelCase error_type, the gate's error code, or
        # premise-lint's fixed prefix) count.
        records = [_tool_result(
            'tu-1',
            'This task should file category="scope_violation" if the '
            'phantom-done gate is relevant.',
        )]

        assert mod.iter_df_guards(records) == []

    def test_interrupt_detects_marker_in_injected_user_text(self):
        records = [_user_text('[Request interrupted by user for tool use]')]

        hits = mod.iter_interrupts(records)

        assert len(hits) == 1

    def test_interrupt_ignores_sidechain(self):
        records = [_sidechain(_user_text('[Request interrupted by user for tool use]'))]

        assert mod.iter_interrupts(records) == []


# ---------------------------------------------------------------------------
# find_retry_loops — same tool name + canonical (sort_keys) input signature
# recurring >= RETRY_MIN (3) times across the session. Deterministic,
# dependency-free grouping -- no fuzzy string similarity.
# ---------------------------------------------------------------------------

class TestFindRetryLoops:
    def test_detects_group_repeated_at_least_three_times(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-1')),
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-2')),
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-3')),
        ]

        loops = mod.find_retry_loops(records)

        assert len(loops) == 1
        loop = loops[0]
        assert loop['tool'] == 'Bash'
        assert loop['signature'] == json.dumps({'command': 'pytest'}, sort_keys=True)
        assert loop['count'] == 3

    def test_two_occurrences_not_yet_a_loop(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-1')),
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-2')),
        ]

        assert mod.find_retry_loops(records) == []

    def test_single_call_is_not_a_loop(self):
        records = [_assistant(_tool_use('Read', {'file_path': '/tmp/x'}, id='tu-1'))]

        assert mod.find_retry_loops(records) == []

    def test_distinct_inputs_are_not_grouped(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'ls'}, id='tu-1')),
            _assistant(_tool_use('Bash', {'command': 'pwd'}, id='tu-2')),
        ]

        assert mod.find_retry_loops(records) == []

    def test_normalizes_input_key_order_for_grouping(self):
        # Same logical input, keys serialized in a different order each
        # time -- canonical sort_keys=True signature must still group them.
        records = [
            _assistant(_tool_use(
                'Edit', {'file_path': '/tmp/x', 'old': 'a', 'new': 'b'}, id='tu-1',
            )),
            _assistant(_tool_use(
                'Edit', {'new': 'b', 'old': 'a', 'file_path': '/tmp/x'}, id='tu-2',
            )),
            _assistant(_tool_use(
                'Edit', {'old': 'a', 'file_path': '/tmp/x', 'new': 'b'}, id='tu-3',
            )),
        ]

        loops = mod.find_retry_loops(records)

        assert len(loops) == 1
        assert loops[0]['count'] == 3

    def test_mixed_fixture_returns_only_the_loop_group(self):
        records = [
            # loop: same tool+input, 3x
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-1')),
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-2')),
            _assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-3')),
            # one-off, not a loop
            _assistant(_tool_use('Read', {'file_path': '/tmp/x'}, id='tu-4')),
            # two distinct inputs, not grouped together
            _assistant(_tool_use('Bash', {'command': 'ls'}, id='tu-5')),
            _assistant(_tool_use('Bash', {'command': 'pwd'}, id='tu-6')),
        ]

        loops = mod.find_retry_loops(records)

        assert len(loops) == 1
        assert loops[0]['tool'] == 'Bash'
        assert loops[0]['count'] == 3
