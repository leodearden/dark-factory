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

import yaml

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


# ---------------------------------------------------------------------------
# Decoy-FAIL suppression contract (PRD Sec 13.2, owned by alpha) -- a
# dedicated test locking the tactical decision explicitly, distinct from the
# per-detector native-carrier-scoping tests above:
#   (a) a literal "FAIL:"/"FAILED" string in an assistant text block or
#       inside a Write/Edit tool_use input never increments tool_error --
#       only a structured is_error=True tool_result does.
#   (b) a "not found"-shaped string planted inside a Write tool_use input
#       never increments not_found.
#   (c) an inline "# decoy-fail" sentinel on a line suppresses a would-be
#       match on THAT line only -- other lines/carriers are unaffected.
# ---------------------------------------------------------------------------

class TestDecoyFailSuppressionContract:
    def test_fail_text_in_assistant_and_write_input_does_not_inflate_tool_error(self):
        records = [
            _assistant(_text('FAIL: this is just narration, not a real error.')),
            _assistant(_tool_use(
                'Write',
                {'file_path': '/tmp/t.py', 'content': 'assert False  # FAILED case'},
                id='tu-decoy',
            )),
            # the one genuine tool_error signal:
            _assistant(_tool_use('Bash', {'command': 'false'}, id='tu-real')),
            _tool_result('tu-real', 'Exit code 1', is_error=True),
        ]

        neighborhoods = mod.iter_error_neighborhoods(records)

        assert len(neighborhoods) == 1
        assert neighborhoods[0]['attempt_tool'] == 'Bash'

    def test_not_found_text_in_write_input_does_not_inflate_not_found(self):
        records = [
            _assistant(_tool_use(
                'Write',
                {
                    'file_path': '/tmp/t.py',
                    'content': 'raise FileNotFoundError("No such file or directory")',
                },
                id='tu-decoy',
            )),
            # the one genuine not_found signal:
            _tool_result('tu-real', 'bash: foo: command not found', is_error=True),
        ]

        hits = mod.iter_not_found(records)

        assert len(hits) == 1
        assert hits[0]['pattern'] == 'command not found'

    def test_decoy_marker_suppresses_same_line_not_found_match(self):
        records = [_tool_result(
            'tu-1',
            'ModuleNotFoundError: shhh, this is a fixture literal  # decoy-fail\n'
            'No such file or directory: bar',
            is_error=True,
        )]

        hits = mod.iter_not_found(records)

        assert [h['pattern'] for h in hits] == ['no such file or directory']

    def test_decoy_marker_suppresses_same_line_df_guard_match(self):
        records = [_tool_result(
            'tu-1',
            'BLOCKED: fixture-only value, ignore me  # decoy-fail\n'
            'error: done_gate_missing_files -- cannot mark done',
            is_error=True,
        )]

        hits = mod.iter_df_guards(records)

        assert [h['pattern'] for h in hits] == ['done_gate_missing_files']

    def test_decoy_marker_suppresses_same_line_self_correction_match(self):
        records = [_assistant(_text(
            'Actually, this line is just fixture narration.  # decoy-fail\n'
            'Let me fix the real bug now.'
        ))]

        hits = mod.iter_self_corrections(records)

        assert [h['pattern'] for h in hits] == ['let me fix']

    def test_decoy_marker_does_not_suppress_other_lines_in_same_block(self):
        # The marker suppresses only its OWN line -- a real signal on a
        # different line in the same block/content still counts.
        records = [_tool_result(
            'tu-1',
            'command not found  # decoy-fail\n'
            'does not exist',
            is_error=True,
        )]

        hits = mod.iter_not_found(records)

        assert [h['pattern'] for h in hits] == ['does not exist']


# ---------------------------------------------------------------------------
# signal_counts / score_signals — the 5-key frontmatter tally (Sec 7.2) and
# the documented weighted-sum score (user turns are gold: PRD Sec 5).
# ---------------------------------------------------------------------------

class TestSignalCountsAndScore:
    def test_signal_counts_returns_exact_five_key_dict_for_mixed_fixture(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'false'}, id='tu-1')),
            _tool_result('tu-1', 'Exit code 1', is_error=True),
            _assistant(_text('My mistake, let me redo this.')),
            _tool_result('tu-2', 'command not found', is_error=False),
            _tool_result('tu-3', 'BLOCKED: real block', is_error=False),
            _user_text('[Request interrupted by user for tool use]'),
        ]

        counts = mod.signal_counts(records)

        assert counts == {
            'tool_error': 1,
            'self_correct': 1,
            'not_found': 1,
            'df_guard': 1,
            'interrupt': 1,
        }

    def test_signal_counts_reports_zero_for_absent_classes(self):
        records = [_user_text('a perfectly normal turn, no confusion here')]

        counts = mod.signal_counts(records)

        assert counts == {
            'tool_error': 0,
            'self_correct': 0,
            'not_found': 0,
            'df_guard': 0,
            'interrupt': 0,
        }

    def test_score_signals_is_monotonic_when_adding_a_signal(self):
        zero = {
            'tool_error': 0, 'self_correct': 0, 'not_found': 0,
            'df_guard': 0, 'interrupt': 0,
        }
        base_score = mod.score_signals(zero, n_user_turns=0)

        for key in zero:
            bumped = dict(zero)
            bumped[key] += 1
            assert mod.score_signals(bumped, n_user_turns=0) > base_score

        assert mod.score_signals(zero, n_user_turns=1) > base_score

    def test_score_signals_weights_user_turns_highest(self):
        zero = {
            'tool_error': 0, 'self_correct': 0, 'not_found': 0,
            'df_guard': 0, 'interrupt': 0,
        }
        one_user_turn_score = mod.score_signals(zero, n_user_turns=1)

        # A single user correction (gold) outweighs a single occurrence of
        # any one OTHER individual signal class.
        for key in zero:
            single_signal = dict(zero)
            single_signal[key] = 1
            assert one_user_turn_score > mod.score_signals(single_signal, n_user_turns=0)


# ---------------------------------------------------------------------------
# classify_agent_class — best-effort marker heuristics grounded in real
# system-prompt/injection literals, with an 'unknown' empty-transcript
# fallback and 'interactive' as the no-marker-matched default. A caller
# override always wins verbatim (beta owns the authoritative class).
# ---------------------------------------------------------------------------

class TestClassifyAgentClass:
    def test_empty_transcript_is_unknown(self):
        assert mod.classify_agent_class([]) == 'unknown'

    def test_task_id_worktree_injection_is_orchestrated_task(self):
        # Literal orchestrator task-briefing preamble
        # (orchestrator/src/orchestrator/dry_run_unblock.py).
        records = [_meta_user(
            'Task ID: 2572\nWorktree: /home/leo/src/dark-factory/.worktrees/2572\n'
        )]

        assert mod.classify_agent_class(records) == 'orchestrated-task'

    def test_recon_stage_marker_is_recon(self):
        # Shared phrase across all reconciliation stage system prompts
        # (fused_memory/reconciliation/prompts/stage{1,2,3}.py).
        records = [_assistant(_text(
            'You are a Memory Consolidator agent operating in sleep mode.'
        ))]

        assert mod.classify_agent_class(records) == 'recon'

    def test_watcher_marker_is_watcher(self):
        # Canonical auto-watcher identity string
        # (escalation/src/escalation/authority.py: _WATCHER_AUTO_IDENTITY).
        records = [_tool_result('tu-1', 'resolved_by=orchestrator-escalation-watcher-auto')]

        assert mod.classify_agent_class(records) == 'watcher'

    def test_curator_classifier_marker_is_curator_classifier(self):
        # Literal task-curator system-prompt fragment
        # (fused_memory/middleware/task_curator.py).
        records = [_assistant(_text(
            'You are the task curator for the dark-factory orchestrator.'
        ))]

        assert mod.classify_agent_class(records) == 'curator-classifier'

    def test_no_marker_falls_back_to_interactive(self):
        records = [_user_text('just a normal conversation, please fix this bug')]

        assert mod.classify_agent_class(records) == 'interactive'

    def test_override_returned_verbatim_even_with_a_marker_present(self):
        records = [_assistant(_text('operating in sleep mode'))]

        result = mod.classify_agent_class(records, override='beta-authoritative-class')

        assert result == 'beta-authoritative-class'

    def test_override_wins_even_on_empty_transcript(self):
        assert mod.classify_agent_class([], override='custom') == 'custom'


# ---------------------------------------------------------------------------
# render_frontmatter — hand-rendered '---'-delimited YAML block, PRD Sec 7.2
# key order verbatim (session, cwd, encoded_dir, agent_class, date,
# size_bytes, score, signal_counts), signal_counts nested in order
# (tool_error, self_correct, not_found, df_guard, interrupt). Must round-trip
# via yaml.safe_load -- NOT yaml.safe_dump (design decision: fixed key order,
# explicit formatting, deterministic byte-stable output).
# ---------------------------------------------------------------------------

def _frontmatter_meta():
    return {
        'session': 'sess-abc123',
        'cwd': '/home/leo/src/dark-factory',
        'encoded_dir': '-home-leo-src-dark-factory',
        'agent_class': 'interactive',
        'date': '2026-07-14',
        'size_bytes': 512,
        'score': 12.5,
        'signal_counts': {
            'tool_error': 1,
            'self_correct': 2,
            'not_found': 3,
            'df_guard': 4,
            'interrupt': 5,
        },
    }


class TestRenderFrontmatter:
    def test_starts_and_ends_with_dash_delimiter(self):
        block = mod.render_frontmatter(_frontmatter_meta())

        lines = block.splitlines()

        assert lines[0] == '---'
        assert lines[-1] == '---'

    def test_top_level_keys_in_contract_order(self):
        block = mod.render_frontmatter(_frontmatter_meta())

        body = block.splitlines()[1:-1]
        top_level_keys = [
            line.split(':', 1)[0] for line in body if not line.startswith(' ')
        ]

        assert top_level_keys == [
            'session', 'cwd', 'encoded_dir', 'agent_class', 'date',
            'size_bytes', 'score', 'signal_counts',
        ]

    def test_signal_counts_nested_in_contract_order(self):
        block = mod.render_frontmatter(_frontmatter_meta())

        body = block.splitlines()[1:-1]
        nested_keys = [
            line.strip().split(':', 1)[0] for line in body if line.startswith(' ')
        ]

        assert nested_keys == [
            'tool_error', 'self_correct', 'not_found', 'df_guard', 'interrupt',
        ]

    def test_round_trips_via_yaml_safe_load(self):
        meta = _frontmatter_meta()
        block = mod.render_frontmatter(meta)

        inner = '\n'.join(block.splitlines()[1:-1])
        loaded = yaml.safe_load(inner)

        assert loaded == meta

    def test_date_round_trips_as_string_not_a_yaml_date_object(self):
        # A bare unquoted 2026-07-14 is resolved by PyYAML's implicit
        # timestamp resolver to a datetime.date, not a string -- the
        # renderer must explicitly quote it (or otherwise dodge this) so
        # downstream consumers always see a plain string.
        block = mod.render_frontmatter(_frontmatter_meta())

        inner = '\n'.join(block.splitlines()[1:-1])
        loaded = yaml.safe_load(inner)

        assert loaded['date'] == '2026-07-14'
        assert isinstance(loaded['date'], str)


# ---------------------------------------------------------------------------
# render_digest — full markdown assembly: frontmatter + one section per
# NON-EMPTY signal class, and a byte-exact self-consistent size_bytes.
# session/cwd/date are derived straight from the records: real transcripts
# carry top-level cwd/sessionId/timestamp on every non-queue-operation
# record (grounded via ~/.claude/projects/-home-leo-src-dark-factory
# inspection); encoded_dir prefers the transcript file's parent dir name
# (ground truth) and falls back to the cwd-derived mirror encoding when no
# path is given.
# ---------------------------------------------------------------------------

_DIGEST_SESSION_ID = '000a8a42-3ca1-44f2-aa82-7f7ed46231fd'
_DIGEST_CWD = '/home/leo/src/dark-factory'
_DIGEST_TIMESTAMP = '2026-07-11T06:02:29.796Z'


def _with_session_meta(
    rec, *, cwd=_DIGEST_CWD, session_id=_DIGEST_SESSION_ID, timestamp=_DIGEST_TIMESTAMP,
):
    """Return a copy of *rec* with real-transcript-shaped top-level cwd/
    sessionId/timestamp fields merged in (observed on every non-
    queue-operation record in a real Claude Code transcript)."""
    out = dict(rec)
    out['cwd'] = cwd
    out['sessionId'] = session_id
    out['timestamp'] = timestamp
    return out


def _all_signals_records():
    """A fixture touching every one of the seven digest sections at least
    once: user correction, error neighborhood, self-correction, a 3x retry
    loop, not-found, a df-guard trip, and an interrupt marker."""
    return [
        _with_session_meta(_user_text('Please redo this, it is wrong.')),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'false'}, id='tu-err'))),
        _with_session_meta(_tool_result('tu-err', 'Exit code 1', is_error=True)),
        _with_session_meta(_assistant(_text('My mistake, let me redo this.'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-1'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-2'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-3'))),
        _with_session_meta(_tool_result('tu-nf', 'bash: foo: command not found', is_error=False)),
        _with_session_meta(_tool_result(
            'tu-guard', 'error: done_gate_missing_files -- cannot mark done', is_error=False,
        )),
        _with_session_meta(_user_text('[Request interrupted by user for tool use]')),
    ]


def _split_frontmatter(digest):
    """Split a rendered digest into (frontmatter_inner_yaml, body) — the
    inverse of render_frontmatter's '---'-delimiting, used to independently
    verify the frontmatter contents of a full digest via yaml.safe_load."""
    lines = digest.splitlines()
    assert lines[0] == '---'
    end = lines.index('---', 1)
    return '\n'.join(lines[1:end]), '\n'.join(lines[end + 1:])


class TestRenderDigest:
    def test_includes_heading_for_each_present_signal_class(self):
        digest = mod.render_digest(_all_signals_records(), agent_class='interactive')

        for heading in (
            'User Corrections', 'Error Neighborhoods', 'Self-Corrections',
            'Retry Loops', 'Not Found', 'Guard Trips', 'Interrupts',
        ):
            assert f'## {heading}' in digest, f'missing heading: {heading}'

    def test_omits_heading_for_absent_signal_class(self):
        records = [_with_session_meta(_user_text('just a normal turn, no confusion'))]

        digest = mod.render_digest(records, agent_class='interactive')

        assert '## User Corrections' in digest
        for heading in (
            'Error Neighborhoods', 'Self-Corrections', 'Retry Loops',
            'Not Found', 'Guard Trips', 'Interrupts',
        ):
            assert f'## {heading}' not in digest

    def test_frontmatter_fields_derived_from_records(self):
        digest = mod.render_digest(_all_signals_records(), agent_class='interactive')

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['session'] == _DIGEST_SESSION_ID
        assert meta['cwd'] == _DIGEST_CWD
        assert meta['date'] == '2026-07-11'
        assert meta['agent_class'] == 'interactive'
        # No transcript path given -> encoded_dir falls back to the
        # cwd.replace('/','-').replace('.','-') mirror encoding.
        assert meta['encoded_dir'] == '-home-leo-src-dark-factory'

    def test_encoded_dir_prefers_transcript_path_parent_dir_name(self, tmp_path):
        # An arbitrary dir name unrelated to cwd's mirror encoding, so a
        # match proves the value came from the path, not a recomputation.
        transcript_path = tmp_path / 'custom-encoded-dir-name' / f'{_DIGEST_SESSION_ID}.jsonl'

        digest = mod.render_digest(
            _all_signals_records(), agent_class='interactive', path=transcript_path,
        )

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['encoded_dir'] == 'custom-encoded-dir-name'

    def test_size_bytes_equals_actual_rendered_byte_length(self):
        digest = mod.render_digest(_all_signals_records(), agent_class='interactive')

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['size_bytes'] == len(digest.encode('utf-8'))


# ---------------------------------------------------------------------------
# render_digest — 15KB soft-cap priority truncation. Over max_bytes, the
# LOWEST-priority non-empty section is trimmed/dropped first (ascending
# SECTION_PRIORITY), with gold (user_corrections) surviving longest — PRD
# Sec 7.2 "truncate lowest-signal sections last".
# ---------------------------------------------------------------------------

def _oversized_not_found_records(n=600):
    """One gold user-correction plus *n* not-found hits -- enough bytes to
    exceed the default 15360-byte soft cap on its own."""
    records = [_with_session_meta(_user_text('This approach is wrong, please redo it properly.'))]
    for i in range(n):
        records.append(_with_session_meta(
            _tool_result(f'tu-nf-{i}', f'bash: cmd{i}: command not found', is_error=False)
        ))
    return records


def _gold_plus_retry_heavy_records():
    """One gold user-correction, a few not-found hits, and MANY retry-loop
    groups (lowest priority) -- oversized enough that dropping retry_loops
    alone comfortably satisfies a tight max_bytes, without needing to touch
    not_found or the gold section."""
    records = [_with_session_meta(_user_text('This is wrong, please fix it.'))]
    for i in range(3):
        records.append(_with_session_meta(_tool_result(f'tu-nf-{i}', 'command not found', is_error=False)))
    for g in range(60):
        for r in range(3):
            records.append(_with_session_meta(_assistant(_tool_use(
                'Bash', {'command': f'pytest-group-{g}'}, id=f'tu-loop-{g}-{r}',
            ))))
    return records


class TestRenderDigestTruncation:
    def test_default_cap_is_respected_and_gold_survives(self):
        digest = mod.render_digest(_oversized_not_found_records(), agent_class='interactive')

        assert len(digest.encode('utf-8')) <= 15360
        assert '## User Corrections' in digest
        # truncation genuinely trimmed the low-signal section, not a no-op
        assert digest.count('command not found') < 600

    def test_max_bytes_override_is_honored(self):
        digest = mod.render_digest(
            _oversized_not_found_records(), agent_class='interactive', max_bytes=2048,
        )

        assert len(digest.encode('utf-8')) <= 2048
        assert '## User Corrections' in digest

    def test_lowest_priority_section_dropped_before_higher_priority(self):
        digest = mod.render_digest(
            _gold_plus_retry_heavy_records(), agent_class='interactive', max_bytes=2048,
        )

        assert len(digest.encode('utf-8')) <= 2048
        assert '## User Corrections' in digest
        assert '## Not Found' in digest
        assert '## Retry Loops' not in digest
