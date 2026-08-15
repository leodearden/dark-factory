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
import logging

import digest as mod
import pytest
import yaml
from legibility import inventory as inventory_mod


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


def _write_archived_jsonl(tmp_path, records, name='archived.jsonl'):
    """Serialize *records* to ``<tmp_path>/<name>``; return its Path.
    Mirrors the on-disk form of an archived fleet session produced by
    ``shared.transcript_archive`` (``<sid>.jsonl``, a verbatim copy)."""
    path = tmp_path / name
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r))
            f.write('\n')
    return path


# ---------------------------------------------------------------------------
# Corruption scaffolding for the file-level contract tests below. A near-copy
# of the helper in test_legibility_inventory.py, deliberately: pytest runs here
# under --import-mode=importlib (pyproject.toml), which does not make sibling
# test modules importable by bare name, and scripts/tests/conftest.py is a
# sys.path bootstrap with no fixtures. The coupling that matters — that both
# readers answer the same way — is asserted directly by
# TestLoadTranscriptCorruptionShapes.test_both_readers_agree_on_an_undecodable_file
# rather than implied by a shared helper.
# ---------------------------------------------------------------------------

_UNDECODABLE_BODY = b'{"type": "user", "seq": 0}\n{"type": "user", "t": "\xff\xfe"}\n'
"""A JSONL body whose SECOND line carries a raw 0xFF — invalid UTF-8."""


def _write_undecodable_plain(path):
    """The one file-level corruption shape a plain ``.jsonl`` archive still has.

    The reader opens under strict ``encoding='utf-8'``, so a single bad byte
    raises ``UnicodeDecodeError`` — a ``ValueError``, not an ``OSError`` —
    and would escape every consumer's documented ``except OSError`` degrade
    path unless the reader normalizes it."""
    path.write_bytes(_UNDECODABLE_BODY)
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
# load_transcript — plain-corpus read. Archived fleet sessions land on disk as
# ``<sid>.jsonl`` (shared.transcript_archive), a verbatim copy with no added
# suffix, so load_transcript reads them with the same plain open it uses for a
# live transcript — census/nightly RENDER their digests rather than
# enumerate-then-drop the whole archive at build_digest time.
# ---------------------------------------------------------------------------

class TestLoadTranscriptPlain:
    def test_plain_jsonl_still_parses_as_before(self, tmp_path):
        # Byte-parity: the reader keeps the exact pre-existing read.
        records = [_user_text('one'), _user_text('two')]
        path = _write_jsonl(tmp_path, records)

        loaded = mod.load_transcript(path)

        assert [r['message']['content'] for r in loaded] == ['one', 'two']


# ---------------------------------------------------------------------------
# load_transcript — the FILE-level half of the degrade contract. This slurping
# reader has the byte-identical open + iterate body as its streaming
# sibling inventory.iter_json_lines, so it has the identical hole: an
# unreadable file surfaces as the identical shape. With the archive stored
# plain (task 3618) exactly one file-level shape survives: a non-UTF-8 byte
# raises UnicodeDecodeError, which no consumer's documented `except OSError`
# degrade path catches, since it is a ValueError subclass. It was always
# reachable on a plain `.jsonl` — the reader opens under strict
# encoding='utf-8' either way — which is why it outlives the gzip shapes.
#
# The corpus extractor's `--transcript` operator mode reads through THIS
# function under `except OSError`, so an archived transcript whose write was
# interrupted aborts that run with a traceback rather than being counted.
# ---------------------------------------------------------------------------

class TestLoadTranscriptCorruptionShapes:
    def test_undecodable_plain_jsonl_raises_oserror(self, tmp_path):
        # Same byte, no gzip layer: this reader's plain branch opens with the
        # same strict encoding, so the shape is reachable there too.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')
        with pytest.raises(OSError):
            mod.load_transcript(undecodable)

    def test_the_decode_shape_names_the_offending_byte(self, tmp_path):
        # One file-level shape survives, so the reason can no longer be
        # triaged by contrast — the actionable detail is the offending byte.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')

        with pytest.raises(OSError) as exc:
            mod.load_transcript(undecodable)

        assert 'gzip' not in str(exc.value)
        assert '0xff' in str(exc.value).lower()

    def test_both_readers_agree_on_an_undecodable_file(self, tmp_path):
        # The two-readers-one-contract property, asserted for the shape that
        # was leaking: whichever mode an operator reaches for, the same bad
        # byte must be reported the same way.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')

        with pytest.raises(OSError) as slurped:
            mod.load_transcript(undecodable)
        with pytest.raises(OSError) as streamed:
            list(inventory_mod.iter_json_lines(undecodable))

        assert type(slurped.value) is type(streamed.value)
        assert str(slurped.value) == str(streamed.value)

    def test_corrupt_line_in_a_valid_file_still_degrades_silently(self, tmp_path):
        # The line-level half, unchanged: a well-formed transcript whose LAST
        # line is a half-written record still yields every parseable record and
        # still does not raise. A fix that wrapped the parse loop too broadly
        # would collapse this into the file-level path and inflate a caller's
        # unreadable-file count with ordinary trailing debris.
        path = tmp_path / 'trailing-partial.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(_user_text('one')) + '\n')
            f.write('\n')
            f.write('{not valid json,,,\n')
            f.write(json.dumps(_user_text('two')) + '\n')
            f.write('{"type": "user", "message": {"content": "cut mid-writ')

        loaded = mod.load_transcript(path)  # must not raise

        assert [r['message']['content'] for r in loaded] == ['one', 'two']

    def test_the_two_readers_share_one_normalization(self):
        # The agreement assertions above verify the two readers BEHAVE the
        # same; this one verifies they cannot stop. Before task 3214's
        # amendment pass the normalization was copy-pasted into both readers
        # and only these tests held the copies in sync — a reviewer-flagged
        # duplication smell (a test pinning two copies together is not the
        # same as there being one). Now digest re-exports inventory's names
        # rather than defining its own, so an edit to one reader's failure
        # contract is an edit to both by construction.
        assert mod.as_unreadable_file_error is inventory_mod.as_unreadable_file_error
        assert mod.UNREADABLE_FILE_ERRORS is inventory_mod.UNREADABLE_FILE_ERRORS


class TestAsUnreadableFileError:
    """The one place that answers 'which exceptions mean an unreadable file'."""

    def test_decode_shape_gets_its_own_wording(self):
        # Not a "gzip stream" failure: this shape is reachable on a plain
        # .jsonl path too, so that label would misdirect an operator reading
        # a disclosed coverage reason.
        exc = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

        normalized = inventory_mod.as_unreadable_file_error(exc)

        assert isinstance(normalized, OSError)
        assert 'undecodable transcript bytes' in str(normalized)
        assert 'gzip' not in str(normalized)

    def test_an_oserror_passes_through_unwrapped(self):
        # An OSError is ALREADY the target type, so it needs no
        # normalization. Returning it unchanged keeps the helper idempotent —
        # a caller can hand it anything it caught without first classifying
        # it, and a bad-magic message never acquires a misleading second
        # "corrupt or truncated gzip stream" prefix.
        original = OSError('Not a readable file')

        assert inventory_mod.as_unreadable_file_error(original) is original

    def test_the_catch_tuple_omits_oserror_shapes(self):
        # UNREADABLE_FILE_ERRORS is exactly the set that is NOT already an
        # OSError. A shape wrongly added here would be caught and re-wrapped
        # rather than propagating, and an OSError in particular must
        # keep reaching callers by its own inheritance.
        assert (UnicodeDecodeError,) == inventory_mod.UNREADABLE_FILE_ERRORS
        assert not any(
            issubclass(exc_type, OSError)
            for exc_type in inventory_mod.UNREADABLE_FILE_ERRORS
        )


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
        assert n['exit_code'] == 1
        assert n['designed_outcome'] is None

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
        assert neighborhoods[0]['exit_code'] is None
        assert neighborhoods[0]['designed_outcome'] is None

    def test_every_neighborhood_carries_the_two_classification_keys(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'false'}, id='tu-1')),
            _tool_result('tu-1', 'boom, no code here', is_error=True),
            _assistant(_tool_use('Bash', {'command': 'watcher'}, id='tu-2')),
            _tool_result('tu-2', _CEILING_DECLARATION, is_error=True),
        ]

        for n in mod.iter_error_neighborhoods(records):
            assert set(n) == {
                'index', 'attempt_tool', 'attempt_input_summary',
                'error_content', 'exit_code', 'designed_outcome',
            }

    def test_ceiling_result_is_enriched_with_code_and_label(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'watcher-rearm.sh'}, id='tu-c')),
            _tool_result('tu-c', _CEILING_DECLARATION, is_error=True),
        ]

        n = mod.iter_error_neighborhoods(records)[0]

        assert n['exit_code'] == 124
        assert n['designed_outcome'] is not None

    def test_plain_error_without_a_code_is_unclassified(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'false'}, id='tu-p')),
            _tool_result('tu-p', 'something broke', is_error=True),
        ]

        n = mod.iter_error_neighborhoods(records)[0]

        assert n['exit_code'] is None
        assert n['designed_outcome'] is None


# ---------------------------------------------------------------------------
# iter_genuine_errors / iter_designed_outcomes — a TOTAL, LOSSLESS partition
# of iter_error_neighborhoods on `designed_outcome is None`. The scan itself
# stays single-source-of-truth; only the split is new.
# ---------------------------------------------------------------------------

def _mixed_error_records():
    """2 declared CEILINGs + 1 bare exit-124 + 1 genuine exit-2 failure."""
    return [
        _assistant(_tool_use('Bash', {'command': 'w1'}, id='tu-c1')),
        _tool_result('tu-c1', _CEILING_DECLARATION, is_error=True),
        _assistant(_tool_use('Bash', {'command': 'w2'}, id='tu-c2')),
        _tool_result('tu-c2', _CEILING_DECLARATION, is_error=True),
        _assistant(_tool_use('Bash', {'command': 'poll'}, id='tu-t')),
        _tool_result('tu-t', 'timed out after 600s, exit=124', is_error=True),
        _assistant(_tool_use('Bash', {'command': 'real'}, id='tu-g')),
        _tool_result('tu-g', 'DARK_FACTORY_ROOT unset, exit code 2', is_error=True),
    ]


class TestErrorNeighborhoodPartition:
    def test_genuine_errors_are_exactly_the_undesigned_ones(self):
        genuine = mod.iter_genuine_errors(_mixed_error_records())

        assert len(genuine) == 1
        assert genuine[0]['exit_code'] == 2
        assert genuine[0]['designed_outcome'] is None

    def test_designed_outcomes_are_exactly_the_rest(self):
        designed = mod.iter_designed_outcomes(_mixed_error_records())

        assert len(designed) == 3
        assert all(d['designed_outcome'] is not None for d in designed)

    def test_partition_is_disjoint(self):
        records = _mixed_error_records()

        genuine_ids = {id(n) for n in mod.iter_genuine_errors(records)}
        designed_ids = {id(n) for n in mod.iter_designed_outcomes(records)}

        # Compare by index -- the two calls build separate dicts.
        assert not (
            {n['index'] for n in mod.iter_genuine_errors(records)}
            & {n['index'] for n in mod.iter_designed_outcomes(records)}
        )
        assert genuine_ids and designed_ids

    def test_partition_is_total_and_lossless(self):
        records = _mixed_error_records()

        recombined = (
            mod.iter_designed_outcomes(records) + mod.iter_genuine_errors(records)
        )
        all_neighborhoods = mod.iter_error_neighborhoods(records)

        assert sorted(recombined, key=lambda n: n['index']) == all_neighborhoods


# ---------------------------------------------------------------------------
# _extract_exit_code / classify_designed_outcome — the 07-31 census cluster
# 1.3 finding (plans/confusion-census-2026-07-31.md:97): watcher-rearm.sh's
# bounded-wait CEILING is a DESIGNED loop-continuation outcome that the
# extractor counted as a tool_error, inflating session a189558e's error
# signal 4x and burying the one genuine failure among false positives.
# Recognition is two-tier: the machine-readable self-declaration first
# (scripts/watcher-rearm.sh:228-237), the bare exit code second.
# ---------------------------------------------------------------------------

_CEILING_DECLARATION = 'WATCHER_REARM_OUTCOME: CEILING exit=124'
"""The sighted line verbatim (plans/confusion-census-2026-07-31.md:97)."""


class TestDesignedOutcomeRecognition:
    def test_extracts_exit_equals_form(self):
        assert mod._extract_exit_code(_CEILING_DECLARATION) == 124

    def test_extracts_exit_code_prose_form(self):
        assert mod._extract_exit_code('command failed with exit code 2') == 2

    def test_extracts_exit_status_prose_form(self):
        assert mod._extract_exit_code('exit status 137') == 137

    def test_extraction_is_case_insensitive(self):
        assert mod._extract_exit_code('Exit Code 2') == 2

    def test_returns_none_when_no_code_present(self):
        assert mod._extract_exit_code('something went wrong') is None

    def test_first_match_wins_when_several_appear(self):
        text = 'exit=124\nlater the retry gave exit code 2\n'

        assert mod._extract_exit_code(text) == 124

    def test_declared_ceiling_is_designed(self):
        assert mod.classify_designed_outcome(_CEILING_DECLARATION, 124) is not None

    def test_declared_fired_is_designed(self):
        label = mod.classify_designed_outcome(
            'WATCHER_REARM_OUTCOME: FIRED exit=0', 0,
        )

        assert label is not None

    def test_declared_killed_is_a_genuine_failure(self):
        # scripts/watcher-rearm.sh:234 classes 137|143|144 as KILLED -- a
        # REAL failure the census should still see, not a designed outcome.
        assert mod.classify_designed_outcome(
            'WATCHER_REARM_OUTCOME: KILLED exit=137', 137,
        ) is None

    def test_declared_error_is_a_genuine_failure(self):
        # scripts/watcher-rearm.sh:237's catch-all ERROR branch.
        assert mod.classify_designed_outcome(
            'WATCHER_REARM_OUTCOME: ERROR exit=2', 2,
        ) is None

    def test_bare_exit_124_is_a_designed_bounded_wait(self):
        assert mod.classify_designed_outcome(
            'command timed out after 600s', 124,
        ) is not None

    def test_ordinary_failure_is_not_designed(self):
        assert mod.classify_designed_outcome('boom, exit code 2', 2) is None

    def test_empty_content_with_no_exit_code_is_not_designed(self):
        assert mod.classify_designed_outcome('', None) is None

    def test_declaration_and_bare_124_labels_are_distinct(self):
        # Which rule fired must stay legible to a digest reader.
        declared = mod.classify_designed_outcome(_CEILING_DECLARATION, 124)
        bare = mod.classify_designed_outcome('timed out', 124)

        assert declared != bare

    def test_bounded_wait_exit_code_constant_is_timeout_ceiling(self):
        assert mod.BOUNDED_WAIT_EXIT_CODE == 124


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

    def test_decoy_marker_suppresses_same_line_interrupt_match(self):
        # Parity with the other three text-pattern detectors: a same-line
        # "# decoy-fail" sentinel suppresses an otherwise-matching line. Two
        # SEPARATE records are needed to prove this (unlike the multi-pattern
        # detectors, iter_interrupts contributes at most one hit per record,
        # so a decoy+real line combined in a single record would still
        # yield exactly one hit regardless of suppression): the first record
        # is decoy-only and must vanish; the second is a genuine, unrelated
        # occurrence and must still be counted.
        records = [
            _user_text('Request interrupted by user for tool use  # decoy-fail'),
            _user_text('[Request interrupted by user for tool use]'),
        ]

        hits = mod.iter_interrupts(records)

        assert [h['index'] for h in hits] == [1]

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
    def test_signal_counts_returns_exact_key_dict_for_mixed_fixture(self):
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
            'designed_outcome': 0,
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
            'designed_outcome': 0,
        }

    def test_ceiling_only_session_reports_zero_tool_errors(self):
        # The 07-31 census cluster 1.3 headline: session a189558e's watcher
        # ceilings were each counted as a tool_error. A session whose ONLY
        # structured errors are declared CEILINGs has no errors at all.
        records = []
        for i in range(13):
            records.append(
                _assistant(_tool_use('Bash', {'command': f'w{i}'}, id=f'tu-{i}'))
            )
            records.append(_tool_result(f'tu-{i}', _CEILING_DECLARATION, is_error=True))

        counts = mod.signal_counts(records)

        assert counts['tool_error'] == 0
        assert counts['designed_outcome'] == 13

    def test_mixed_session_surfaces_the_one_genuine_failure(self):
        # census :97 verbatim: 3 benign ceilings burying 1 real exit-2 bug.
        records = []
        for i in range(3):
            records.append(
                _assistant(_tool_use('Bash', {'command': f'w{i}'}, id=f'tu-c{i}'))
            )
            records.append(_tool_result(f'tu-c{i}', _CEILING_DECLARATION, is_error=True))
        records.append(_assistant(_tool_use('Bash', {'command': 'real'}, id='tu-g')))
        records.append(
            _tool_result('tu-g', 'DARK_FACTORY_ROOT unset, exit code 2', is_error=True)
        )

        counts = mod.signal_counts(records)

        assert counts['tool_error'] == 1
        assert counts['designed_outcome'] == 3

    def test_designed_outcome_has_no_signal_weight(self):
        # Deliberate: score_signals iterates SIGNAL_WEIGHTS, so omitting the
        # entry reports the count without perturbing the sampler's ranking.
        assert 'designed_outcome' not in mod.SIGNAL_WEIGHTS

    def test_score_is_bit_identical_across_designed_outcome_counts(self):
        base = {
            'tool_error': 2, 'self_correct': 1, 'not_found': 0,
            'df_guard': 1, 'interrupt': 0, 'designed_outcome': 0,
        }
        noisy = dict(base, designed_outcome=13)

        assert mod.score_signals(noisy, 3) == mod.score_signals(base, 3)

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

    def test_bare_worktree_mention_alone_is_not_orchestrated_task(self):
        # 'Worktree:' alone (without its paired 'Task ID:' preamble line) is
        # ordinary interactive prose -- e.g. this very project's docs and
        # sessions mention worktrees constantly -- and must not by itself
        # trigger a false-positive orchestrated-task classification. Both
        # preamble lines must co-occur (PRD Sec 13.2-adjacent decoy-style
        # false-positive guard, owned here).
        records = [_assistant(_text(
            'The worktree: /home/leo/src/dark-factory/.worktrees/2572 is where this runs.'
        ))]

        assert mod.classify_agent_class(records) == 'interactive'

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
            'designed_outcome': 6,
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
            'designed_outcome',
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


# ---------------------------------------------------------------------------
# is_harness_injected_turn / iter_user_turns exclusion -- R1 (confusion
# census 2026-07-24 Sec 4): a harness-injected briefing turn (the
# orchestrator's '# Context' + '## Agent Identity' + '# Task' preamble, or
# the trickle coder's system-prompt preamble) is typed into the transcript
# as ordinary user-role text (isMeta=False -- it really is the first "user"
# turn Claude Code sees), so it was previously indistinguishable from a
# genuine human correction: it inflated both the gold user_corrections
# section and score's n_user_turns component. Exclusion is by CONTENT (all
# three headings must co-occur, line-anchored, mirroring
# ORCHESTRATED_TASK_MARKERS' all() guard) so an ordinary human turn that
# merely mentions one heading, or quotes one mid-sentence, is never
# over-excluded.
# ---------------------------------------------------------------------------

def _briefing_text(body_filler='Project overview and recent decisions go here.'):
    """Build the text of an orchestrator-briefing-shaped user turn -- the
    literal shape ``BriefingAssembler`` injects as the harness preamble of
    every dispatched-agent session
    (orchestrator/src/orchestrator/agents/briefing.py): a '# Context'
    heading (``_get_memory_context``, emits '# Context\\n\\n...'), a
    '## Agent Identity' heading with an agent_id bullet
    (``_agent_identity``), and a '# Task' heading with a task block (the
    per-role prompt templates, e.g. ``build_architect_prompt``)."""
    return (
        f'# Context\n\n{body_filler}\n\n'
        '## Agent Identity\n\n'
        '- **agent_id:** `claude-task-3278-implementer`\n'
        '- **project_id:** `dark_factory`\n\n'
        '# Task\n\n'
        '**Task:** Some task title\n'
        '**Description:** Do the thing.\n'
    )


_CENSUS_MEMORY_CONTEXT_TURN = (
    '# Context\n\n'
    '## Project Context\n\n'
    '{ "results": [ { "id": "ccf73ca4-8240-4f9a-8abf-32b210e5b35b", '
    '"content": "Task 1470 wired /audit into /review Phase-2 Architectural '
    'Coherence." } ] }\n'
)
"""The 07-31 census cluster 1.1(b) sighting verbatim
(plans/confusion-census-2026-07-31.md:85): a context-ONLY briefing injection
that the all-three-headings rule admitted into the gold section."""


_TRICKLE_CODER_PREAMBLE = (
    'You are the trickle coder for the dark-factory agent-confusion '
    'codebook (plans/confusion-reduction-prd.md §7.3). Read the '
    'session digest below and decide which existing codebook entries '
    'it matches.'
)
"""Literal prefix of scripts/legibility/coder.py:174 build_prompt's
harness-authored preamble."""


class TestHarnessInjectedTurnFilter:
    def test_briefing_shaped_turn_is_excluded_by_content(self):
        records = [_user_text(_briefing_text())]

        assert mod.iter_user_turns(records) == []

    def test_trickle_coder_preamble_turn_is_excluded(self):
        records = [_user_text(_TRICKLE_CODER_PREAMBLE)]

        assert mod.iter_user_turns(records) == []

    def test_ordinary_correction_retained_after_briefing_turn_index_preserved(self):
        records = [
            _user_text(_briefing_text()),
            _user_text('This is wrong, please redo it.'),
        ]

        turns = mod.iter_user_turns(records)

        assert [t['text'] for t in turns] == ['This is wrong, please redo it.']
        assert [t['index'] for t in turns] == [1]

    def test_census_sighted_memory_context_turn_is_excluded(self):
        # The 07-31 census cluster 1.1(b) sighting, verbatim
        # (plans/confusion-census-2026-07-31.md:85): a context-ONLY briefing
        # injection -- '# Context' + '## Project Context' + the memory-search
        # JSON dump. It carries NEITHER '## Agent Identity' NOR '# Task', so
        # the old all-three rule let it through into the gold section.
        text = _CENSUS_MEMORY_CONTEXT_TURN

        assert mod.is_harness_injected_turn(text) is True
        assert mod.iter_user_turns([_user_text(text)]) == []

    def test_context_plus_conventions_subheading_is_excluded(self):
        # briefing.py:1096 emits '## Conventions' under the same '# Context'.
        records = [_user_text(
            '# Context\n\n## Conventions\n\n{"results": []}\n'
        )]

        assert mod.iter_user_turns(records) == []

    def test_memory_unavailable_context_variant_is_excluded(self):
        # briefing.py:1113's exception path emits a '## Context' sub-block;
        # '## Context' is therefore an ANCHOR too, not just '# Context'.
        records = [_user_text(
            '## Context\n\n_Memory unavailable._\n\n## Project Context\n\nx\n'
        )]

        assert mod.iter_user_turns(records) == []

    def test_task_plus_action_is_excluded(self):
        # The role prompt templates emit '# Task' and '# Action'
        # (briefing.py:193/496/1038) as the two headings of one preamble.
        records = [_user_text('# Task\n\ndo the thing\n\n# Action\n\ngo\n')]

        assert mod.iter_user_turns(records) == []

    def test_lone_context_anchor_with_prose_is_retained(self):
        # The relaxation's boundary: ONE anchor and NO corroborating
        # briefing heading is not enough to classify as injected.
        records = [_user_text('# Context\n\nsome prose about the problem')]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1
        assert turns[0]['text'] == '# Context\n\nsome prose about the problem'

    def test_lone_subheading_without_an_anchor_is_retained(self):
        # A corroborator alone never classifies -- at least one ANCHOR is
        # required, so a human turn headed '## Conventions' stays gold.
        records = [_user_text('## Conventions\n\nwe always use tabs, fix this')]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1

    def test_single_heading_alone_is_not_excluded(self):
        # Only ONE of the three co-occurring headings -- must not alone
        # trigger exclusion (all()-not-any(), mirrors
        # test_bare_worktree_mention_alone_is_not_orchestrated_task).
        records = [_user_text('# Task\n\nplease do X')]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1
        assert turns[0]['text'] == '# Task\n\nplease do X'

    def test_context_heading_mentioned_mid_sentence_is_not_excluded(self):
        # Line-anchored matching: '# Context' appearing mid-sentence (not
        # as its own stripped line) never counts as the heading.
        records = [_user_text(
            'I read the # Context heading in the docs and want to update it.'
        )]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1

    def test_render_digest_excludes_briefing_turn_from_body_and_score(self):
        records = [
            _with_session_meta(_user_text(_briefing_text())),
            _with_session_meta(_user_text('This is wrong, please redo it.')),
        ]

        digest = mod.render_digest(records, agent_class='interactive')

        frontmatter_yaml, body = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert 'This is wrong, please redo it.' in body
        assert '# Context' not in body
        assert '## Agent Identity' not in body
        # The excluded turn no longer counts toward n_user_turns (today: 2).
        assert meta['score'] == mod.score_signals(meta['signal_counts'], 1)

    def test_classify_agent_class_still_detects_markers_inside_excluded_turn(self):
        # 'Task ID:'/'Worktree:' (a SEPARATE injected preamble --
        # dry_run_unblock.py -- from the '# Context'/'## Agent
        # Identity'/'# Task' briefing) live only inside this now-excluded
        # turn; classify_agent_class reads a different carrier
        # (_signal_text_sources, never filtered by iter_user_turns) so it
        # must be unaffected by the exclusion.
        text = (
            _briefing_text()
            + 'Task ID: 3278\nWorktree: /home/leo/src/dark-factory/.worktrees/3278\n'
        )
        records = [_user_text(text)]

        assert mod.iter_user_turns(records) == []
        assert mod.classify_agent_class(records) == 'orchestrated-task'

    def test_signal_counts_unaffected_by_signal_free_briefing_turn(self):
        # The filter must not leak into _signal_text_sources: adding a
        # briefing turn with no NOT_FOUND/DF_GUARD/INTERRUPT pattern must
        # not perturb any of the five signal_counts.
        base = _all_signals_records()
        with_briefing = [_user_text(_briefing_text())] + base

        assert mod.signal_counts(with_briefing) == mod.signal_counts(base)


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
        # No transcript path given -> encoded_dir falls back to the mirror of
        # session_registry.encode_cwd (every '/', '.' and '_' maps to '-').
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
        # Higher-priority not_found (only 3 hits, ~120 bytes) survives
        # completely untouched -- the trimmer never reaches it as long as
        # the lower-priority retry_loops section still has items to give.
        assert digest.count('command not found') == 3
        # Lowest-priority retry_loops (60 groups) is what absorbed the
        # truncation: fewer than all 60 groups survive.
        assert digest.count('Bash x3:') < 60


# ---------------------------------------------------------------------------
# _cap_item / _item_byte_cap -- R2 part 1 (confusion census 2026-07-24 Sec
# 4): a per-item byte cap applied at the single _build_sections choke
# point, so ONE oversized item (an enormous pasted user turn, a huge
# tool_result echo, ...) can never evict every sibling section before
# being popped itself. Capping is byte-wise and UTF-8-safe: the whole
# digest budget is measured in UTF-8 bytes (_resolve_size_bytes, the 15360
# soft cap), so a naive character-wise slice would not actually bound it
# for non-ASCII content, and a naive byte slice can split a multi-byte
# codepoint mid-sequence.
# ---------------------------------------------------------------------------

def _oversized_user_correction_records():
    """One ORDINARY (non-briefing-shaped) human user turn whose text alone
    exceeds the default 15360-byte soft cap, plus a genuine is_error
    tool_result error neighborhood and a 'command not found' not-found
    hit. Deliberately NOT briefing-shaped: R2 (whole-item truncation
    eviction) is a distinct pathology from R1 (harness-turn exclusion), so
    step-2's is_harness_injected_turn filter is never why this fixture's
    body would survive. Baseline today (pre per-item cap):
    _truncate_sections pops the WHOLE oversized item, which along the way
    empties every other section too -- the final digest is ~276 bytes of
    frontmatter with an entirely empty body, despite every detector having
    fired at least once."""
    huge_text = 'This is wrong, please redo it properly. ' * 500  # 20000 bytes
    return [
        _with_session_meta(_user_text(huge_text)),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'false'}, id='tu-err'))),
        _with_session_meta(_tool_result('tu-err', 'Exit code 1', is_error=True)),
        _with_session_meta(_tool_result('tu-nf', 'bash: foo: command not found', is_error=False)),
    ]


class TestPerItemByteCap:
    def test_cap_item_returns_short_line_byte_identical(self):
        line = '- (turn 0) short line, well under any cap'

        capped = mod._cap_item(line, cap=2048)

        assert capped == line
        assert mod.ITEM_TRUNCATION_MARKER not in capped

    def test_cap_item_truncates_oversized_line_with_marker(self):
        line = '- (turn 0) ' + ('x' * 5000)
        cap = 2048

        capped = mod._cap_item(line, cap)

        assert len(capped.encode('utf-8')) <= cap
        assert capped.endswith(mod.ITEM_TRUNCATION_MARKER)

    def test_cap_item_truncates_multibyte_text_without_mojibake(self):
        # The cap is a BYTE cap -- naive slicing on a multi-byte-character
        # string must not split a codepoint into a replacement character.
        line = '- (turn 0) ' + ('é→' * 2000)
        cap = 500

        capped = mod._cap_item(line, cap)  # must not raise

        assert len(capped.encode('utf-8')) <= cap
        assert '�' not in capped
        capped.encode('utf-8')  # re-encodes cleanly

    def test_item_byte_cap_respects_bounds_and_frontmatter_headroom(self):
        for max_bytes in (0, 1, 100, 512, 1000, 2048, 4096, 15360, 100_000):
            cap = mod._item_byte_cap(max_bytes)

            assert cap <= mod.MAX_ITEM_BYTES
            assert cap >= mod.MIN_ITEM_BYTES
            reserved = max_bytes - mod.FRONTMATTER_RESERVE_BYTES
            if reserved > mod.MIN_ITEM_BYTES:
                assert cap <= reserved

    def test_default_cap_retains_all_sections_with_oversized_user_turn(self):
        digest = mod.render_digest(_oversized_user_correction_records(), agent_class='interactive')

        _, body = _split_frontmatter(digest)
        # Baseline today (pre-step-4): the whole-item eviction pathology
        # empties the body entirely -- assert non-empty so this is
        # unambiguously RED before the per-item cap exists.
        assert body.strip() != ''

        assert '## User Corrections' in digest
        assert '## Error Neighborhoods' in digest
        assert '## Not Found' in digest
        assert len(digest.encode('utf-8')) <= 15360
        assert mod.ITEM_TRUNCATION_MARKER in digest

    def test_gold_section_still_precedes_error_neighborhoods_under_cap(self):
        digest = mod.render_digest(_oversized_user_correction_records(), agent_class='interactive')

        assert '## User Corrections' in digest
        assert '## Error Neighborhoods' in digest
        assert digest.index('## User Corrections') < digest.index('## Error Neighborhoods')


# ---------------------------------------------------------------------------
# Designed Outcomes section -- 07-31 census cluster 1.3 / R3. A declared
# bounded-poll ceiling is reported, but in its OWN section: it must never
# masquerade as an error neighborhood, and the error section must surface
# exit codes so a bare 124 stays distinguishable to a human reader.
# ---------------------------------------------------------------------------

def _mixed_designed_and_genuine_records():
    """3 declared CEILINGs plus 1 genuine exit-2 failure -- census :97."""
    records = [_with_session_meta(_user_text('kick off the bounded poll'))]
    for i in range(3):
        records.append(_assistant(_tool_use('Bash', {'command': f'poll{i}'}, id=f'tu-c{i}')))
        records.append(_tool_result(f'tu-c{i}', _CEILING_DECLARATION, is_error=True))
    records.append(_assistant(_tool_use('Bash', {'command': 'real'}, id='tu-g')))
    records.append(
        _tool_result('tu-g', 'DARK_FACTORY_ROOT unset, exit code 2', is_error=True)
    )
    return records


def _designed_outcomes_only_records():
    """A session whose ONLY nonzero signal class is designed outcomes."""
    records = []
    for i in range(3):
        records.append(_assistant(_tool_use('Bash', {'command': f'poll{i}'}, id=f'tu-{i}')))
        records.append(_tool_result(f'tu-{i}', _CEILING_DECLARATION, is_error=True))
    return records


class TestDesignedOutcomesSection:
    def test_registered_in_all_three_section_registries(self):
        # All three or none: _SECTION_RENDERERS alone would leave
        # _render_body/_truncate_sections unable to find the heading.
        assert 'designed_outcomes' in mod.SECTION_HEADINGS
        assert 'designed_outcomes' in mod.SECTION_PRIORITY
        assert 'designed_outcomes' in mod._SECTION_RENDERERS

    def test_ranks_lowest_in_section_priority(self):
        # Index 0 == trimmed FIRST under the soft cap, below retry_loops:
        # a designed outcome is explicitly NOT confusion.
        assert mod.SECTION_PRIORITY[0] == 'designed_outcomes'

    def test_genuine_error_alone_under_error_neighborhoods(self):
        digest = mod.render_digest(
            _mixed_designed_and_genuine_records(), agent_class='interactive',
        )
        _, body = _split_frontmatter(digest)

        error_block = body.split('## Error Neighborhoods', 1)[1].split('\n##', 1)[0]

        assert 'exit code 2' in error_block
        assert 'CEILING' not in error_block

    def test_designed_outcomes_render_under_their_own_heading(self):
        digest = mod.render_digest(
            _mixed_designed_and_genuine_records(), agent_class='interactive',
        )
        _, body = _split_frontmatter(digest)

        assert '## Designed Outcomes' in body
        designed_block = body.split('## Designed Outcomes', 1)[1].split('\n##', 1)[0]
        assert designed_block.count('CEILING') == 3

    def test_ceiling_text_is_never_rendered_in_both_sections(self):
        digest = mod.render_digest(
            _mixed_designed_and_genuine_records(), agent_class='interactive',
        )
        _, body = _split_frontmatter(digest)

        # 3 ceilings, each rendered exactly once -- the partition is
        # lossless, so the total must not double.
        assert body.count(_CEILING_DECLARATION) == 3

    def test_error_neighborhood_line_surfaces_the_exit_code(self):
        # 07-31 R3's "record exit codes" ask: the code is promoted to a
        # STRUCTURED marker on the line, not merely left buried in
        # whatever prose the tool happened to echo -- and the error
        # content can be byte-truncated away by the per-item cap.
        records = [
            _assistant(_tool_use('Bash', {'command': 'boom'}, id='tu-1')),
            _tool_result('tu-1', 'fatal: bad thing, exit status 137', is_error=True),
        ]

        digest = mod.render_digest(records, agent_class='interactive')
        _, body = _split_frontmatter(digest)

        error_block = body.split('## Error Neighborhoods', 1)[1].split('\n##', 1)[0]
        assert '[exit 137]' in error_block

    def test_error_neighborhood_line_omits_the_marker_when_no_code(self):
        records = [
            _assistant(_tool_use('Bash', {'command': 'boom'}, id='tu-1')),
            _tool_result('tu-1', 'something went wrong', is_error=True),
        ]

        digest = mod.render_digest(records, agent_class='interactive')
        _, body = _split_frontmatter(digest)

        error_block = body.split('## Error Neighborhoods', 1)[1].split('\n##', 1)[0]
        assert '[exit' not in error_block

    def test_designed_outcome_line_names_the_rule_that_fired(self):
        # Distinct labels per rule keep which-rule-fired legible: a
        # self-declared CEILING must not read the same as a bare 124.
        declared = mod.classify_designed_outcome(_CEILING_DECLARATION, 124)
        digest = mod.render_digest(
            _designed_outcomes_only_records(), agent_class='interactive',
        )
        _, body = _split_frontmatter(digest)

        designed_block = body.split('## Designed Outcomes', 1)[1].split('\n##', 1)[0]
        assert declared in designed_block
        assert '[exit 124]' in designed_block

    def test_designed_outcome_only_session_renders_a_body_without_warning(self, caplog):
        # The registration regression: signal_counts is now nonzero for a
        # ceiling-only session, so an unregistered section would leave the
        # body empty and trip _warn_if_body_evicted on EVERY such session.
        caplog.set_level(logging.WARNING, logger='legibility.digest')
        records = _designed_outcomes_only_records()

        assert mod.signal_counts(records)['designed_outcome'] == 3

        digest = mod.render_digest(records, agent_class='interactive')
        _, body = _split_frontmatter(digest)

        assert body.strip() != ''
        assert '## Designed Outcomes' in body
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# ---------------------------------------------------------------------------
# _warn_if_body_evicted / render_digest emit-time guard -- R2 part 2
# (confusion census 2026-07-24 Sec 4): belt-and-braces backstop for the
# invariant step-4's per-item cap makes structurally unreachable at any
# realistic max_bytes -- "nonzero signal_counts implies a non-empty body"
# is checked at emit time, and a violation is logged LOUDLY (never silent,
# never raised: census.py/nightly.py must keep receiving a plain string).
# ---------------------------------------------------------------------------

class TestEmitConsistencyGuard:
    def test_warns_once_when_nonzero_counts_and_every_section_empty(self, caplog):
        caplog.set_level(logging.WARNING, logger='legibility.digest')
        meta = {
            'session': 'sess-guard-1',
            'signal_counts': {
                'tool_error': 1, 'self_correct': 0, 'not_found': 0,
                'df_guard': 0, 'interrupt': 0,
            },
        }
        sections = {key: [] for key in mod.SECTION_PRIORITY}

        mod._warn_if_body_evicted(meta, sections, max_bytes=15360)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'sess-guard-1' in warnings[0]

    def test_no_warning_when_all_signal_counts_zero(self, caplog):
        caplog.set_level(logging.WARNING, logger='legibility.digest')
        meta = {
            'session': 'sess-guard-2',
            'signal_counts': {
                'tool_error': 0, 'self_correct': 0, 'not_found': 0,
                'df_guard': 0, 'interrupt': 0,
            },
        }
        sections = {key: [] for key in mod.SECTION_PRIORITY}

        mod._warn_if_body_evicted(meta, sections, max_bytes=15360)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_no_warning_when_a_section_still_has_lines(self, caplog):
        caplog.set_level(logging.WARNING, logger='legibility.digest')
        meta = {
            'session': 'sess-guard-3',
            'signal_counts': {
                'tool_error': 1, 'self_correct': 0, 'not_found': 0,
                'df_guard': 0, 'interrupt': 0,
            },
        }
        sections = {key: [] for key in mod.SECTION_PRIORITY}
        sections['not_found'] = ['- (turn 0) command not found']

        mod._warn_if_body_evicted(meta, sections, max_bytes=15360)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_render_digest_warns_and_still_returns_degenerate_digest_at_tiny_cap(self, caplog):
        # max_bytes=1 makes the trim loop pop every item regardless of
        # frontmatter length -- a deterministic violation path (not a
        # guessed numeric tolerance): the body ends up empty while
        # signal_counts stays nonzero.
        caplog.set_level(logging.WARNING, logger='legibility.digest')
        records = _all_signals_records()

        digest = mod.render_digest(records, agent_class='interactive', max_bytes=1)

        # Never silent, never raises: the degenerate digest is still
        # returned (fail-soft for census.py/nightly.py callers).
        _, body = _split_frontmatter(digest)
        assert body.strip() == ''
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert _DIGEST_SESSION_ID in warnings[0]

    def test_invariant_holds_at_real_cap_for_oversized_fixture(self, caplog):
        # Already green off steps 2/4 -- serves as part of the end-to-end
        # regression lock for the whole task.
        caplog.set_level(logging.WARNING, logger='legibility.digest')

        digest = mod.render_digest(_oversized_user_correction_records(), agent_class='interactive')

        frontmatter_yaml, body = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert any(meta['signal_counts'].values())
        assert body.strip() != ''
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_build_digest_end_to_end_on_archived_orchestrated_session(self, tmp_path):
        # Already green off steps 2/4 -- the end-to-end acceptance check:
        # a real orchestrated-looking archived transcript (briefing turn +
        # genuine human correction + tool_error + not_found) yields a
        # non-empty, briefing-free digest with an intact frontmatter
        # contract.
        records = [
            _with_session_meta(_user_text(_briefing_text('x' * 21000))),
            _with_session_meta(_user_text('This is wrong, please redo it.')),
            _with_session_meta(_assistant(_tool_use('Bash', {'command': 'false'}, id='tu-err'))),
            _with_session_meta(_tool_result('tu-err', 'Exit code 1', is_error=True)),
            _with_session_meta(_tool_result('tu-nf', 'bash: foo: command not found', is_error=False)),
        ]
        path = _write_archived_jsonl(tmp_path, records)

        digest = mod.build_digest(path)

        _, body = _split_frontmatter(digest)
        assert body.strip() != ''
        assert '## User Corrections' in digest
        assert 'This is wrong, please redo it.' in digest
        assert '## Error Neighborhoods' in digest
        assert '## Not Found' in digest
        assert '# Context' not in digest
        assert '## Agent Identity' not in digest

        frontmatter_yaml, _ = _split_frontmatter(digest)
        parsed = yaml.safe_load(frontmatter_yaml)
        assert set(parsed) == set(mod.FRONTMATTER_KEYS) | {'signal_counts'}
        assert set(parsed['signal_counts']) == set(mod.SIGNAL_COUNT_KEYS)

        lines = frontmatter_yaml.splitlines()
        top_level_keys = [line.split(':', 1)[0] for line in lines if not line.startswith(' ')]
        assert top_level_keys == list(mod.FRONTMATTER_KEYS) + ['signal_counts']
        nested_keys = [line.strip().split(':', 1)[0] for line in lines if line.startswith(' ')]
        assert nested_keys == list(mod.SIGNAL_COUNT_KEYS)


# ---------------------------------------------------------------------------
# build_digest — PRD Sec 8.1 boundary test, row 1, producer side. The
# headline acceptance test: load_transcript -> detectors -> classify ->
# render_digest, end to end from a real JSONL file on disk, with a decoy
# "FAIL:" planted inside a Write tool_use input that must NOT inflate
# tool_error (Sec 13.2 decoy-FAIL suppression, owned by alpha).
# ---------------------------------------------------------------------------

def _boundary_records():
    """Plants exactly the PRD Sec 8.1 row-1 fixture: a user-correction, an
    is_error tool_result with its preceding attempt, a self-correction
    marker, a 3x retry loop, and a decoy "FAIL:" string inside a Write
    tool_use input that must not be counted as a real tool_error."""
    return [
        _with_session_meta(_user_text('This is wrong, please redo it.')),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'false'}, id='tu-err'))),
        _with_session_meta(_tool_result('tu-err', 'Exit code 1', is_error=True)),
        _with_session_meta(_assistant(_text('My mistake, let me redo this.'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-1'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-2'))),
        _with_session_meta(_assistant(_tool_use('Bash', {'command': 'pytest'}, id='tu-loop-3'))),
        # decoy: a Write tool_use input containing a literal "FAIL:" string,
        # with no matching tool_result -- never a real signal of any kind.
        _with_session_meta(_assistant(_tool_use(
            'Write',
            {'file_path': '/tmp/fixture.py', 'content': 'assert False  # FAIL: decoy, not real'},
            id='tu-decoy',
        ))),
    ]


class TestBuildDigestBoundary:
    def test_boundary_fixture_four_sections_within_cap_decoy_suppressed(self, tmp_path):
        path = _write_jsonl(tmp_path, _boundary_records())

        digest = mod.build_digest(path)

        for heading in ('User Corrections', 'Error Neighborhoods', 'Self-Corrections', 'Retry Loops'):
            assert f'## {heading}' in digest, f'missing heading: {heading}'

        assert len(digest.encode('utf-8')) <= 15360

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)
        counts = meta['signal_counts']

        # Real planted signals reflected exactly.
        assert counts['tool_error'] == 1
        assert counts['self_correct'] == 1
        assert counts['not_found'] == 0
        assert counts['df_guard'] == 0
        assert counts['interrupt'] == 0

        # The decoy never surfaces anywhere in the digest: it has no
        # matching tool_result, so iter_error_neighborhoods (the only
        # detector that ever echoes a tool_use input) never touches it.
        assert 'FAIL:' not in digest


# ---------------------------------------------------------------------------
# main — thin argparse CLI over build_digest, mirroring
# analyze_speculation_depth.py's TestMainCLI (capsys end-to-end smoke tests).
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_main_prints_digest_to_stdout_and_returns_zero(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, _boundary_records())

        ret = mod.main([str(path)])

        captured = capsys.readouterr()
        assert ret == 0
        assert captured.out.startswith('---\n')
        assert '## User Corrections' in captured.out
        # print() appends exactly one trailing newline beyond the digest
        # itself -- strip it before checking the digest's own soft cap.
        assert len(captured.out.rstrip('\n').encode('utf-8')) <= 15360

    def test_agent_class_flag_overrides_frontmatter_class(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, _boundary_records())

        ret = mod.main([str(path), '--agent-class', 'recon'])

        captured = capsys.readouterr()
        assert ret == 0
        frontmatter_yaml, _ = _split_frontmatter(captured.out)
        assert yaml.safe_load(frontmatter_yaml)['agent_class'] == 'recon'

    def test_out_flag_writes_digest_to_file_instead_of_stdout(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, _boundary_records())
        out_path = tmp_path / 'digest.md'

        ret = mod.main([str(path), '--out', str(out_path)])

        captured = capsys.readouterr()
        assert ret == 0
        assert captured.out == ''  # --out redirects the digest away from stdout
        assert out_path.read_text(encoding='utf-8') == mod.build_digest(path)

    def test_nonexistent_path_returns_nonzero_and_writes_stderr(self, tmp_path, capsys):
        missing = tmp_path / 'does-not-exist.jsonl'

        ret = mod.main([str(missing)])

        captured = capsys.readouterr()
        assert ret != 0
        assert captured.out == ''
        assert str(missing) in captured.err
