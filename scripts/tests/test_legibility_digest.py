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

import gzip
import io
import json
import logging
import re
import zlib

import pytest
import yaml

import digest as mod
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


def _write_jsonl_gz(tmp_path, records, name='transcript.jsonl.gz'):
    """Serialize *records* gzip-compressed to ``<tmp_path>/<name>``; return its
    Path. Mirrors the on-disk form of an archived fleet session produced by
    ``shared.transcript_archive`` (``<sid>.jsonl.gz``)."""
    path = tmp_path / name
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r))
            f.write('\n')
    return path


# ---------------------------------------------------------------------------
# Corruption scaffolding for the file-level contract tests below. A near-copy
# of the pair in test_legibility_inventory.py, deliberately: pytest runs here
# under --import-mode=importlib (pyproject.toml), which does not make sibling
# test modules importable by bare name, and scripts/tests/conftest.py is a
# sys.path bootstrap with no fixtures. The coupling that matters — that both
# readers answer the same way — is asserted directly by
# TestLoadTranscriptCorruptionShapes.test_both_readers_agree_on_a_truncated_file
# rather than implied by a shared helper.
# ---------------------------------------------------------------------------

def _gz_payload(n_records: int = 200) -> bytes:
    """Serialize a multi-record JSONL body — the payload the helpers below
    compress and then damage. Many padded records so a cut at the halfway
    mark lands mid-stream rather than inside the 10-byte header."""
    return ''.join(
        json.dumps({'type': 'user', 'seq': i, 'pad': 'x' * 200}) + '\n'
        for i in range(n_records)
    ).encode('utf-8')


def _write_truncated_gz(path):
    """Write the first half of a valid gz stream — the interrupted-write
    shape, which decompresses to ``EOFError`` (NOT an ``OSError``)."""
    blob = gzip.compress(_gz_payload())
    path.write_bytes(blob[: len(blob) // 2])
    return path


def _write_corrupt_body_gz(path):
    """Write a gz whose DEFLATE body is damaged, so decompression raises
    ``zlib.error`` (also NOT an ``OSError``).

    Where a flipped byte lands decides which failure gzip reports — most
    flips still decode and only trip the trailing checksum
    (``gzip.BadGzipFile``, already an ``OSError``), a few truncate the bit
    stream (``EOFError``), and only a flip that makes the DEFLATE stream
    itself unparseable raises ``zlib.error``. So probe for the first flip
    position producing that shape rather than assuming a byte. The probe runs
    against STDLIB ``gzip``, never a reader under test, so it stays valid on
    both sides of the normalization.
    """
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


_UNDECODABLE_BODY = b'{"type": "user", "seq": 0}\n{"type": "user", "t": "\xff\xfe"}\n'
"""A JSONL body whose SECOND line carries a raw 0xFF — invalid UTF-8."""


def _write_undecodable_gz(path):
    """A structurally VALID gz whose payload is not valid UTF-8.

    The fourth shape, unreachable by the two helpers above: they damage the
    gzip container, and the ``_write_corrupt_body_gz`` probe decompresses in
    BINARY mode, so neither can surface a decode fault. This file
    decompresses cleanly and fails one layer up, at the reader's
    ``encoding='utf-8'`` text wrapper, as ``UnicodeDecodeError`` — a
    ``ValueError``, so it escapes ``except OSError`` however the gzip shapes
    are handled.
    """
    path.write_bytes(gzip.compress(_UNDECODABLE_BODY))
    return path


def _write_undecodable_plain(path):
    """The same bad byte with no gzip layer — the plain branch opens with the
    same strict encoding, so the shape is reachable on ``.jsonl`` too."""
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
# load_transcript — gz transparency. Archived fleet sessions land on disk as
# ``<sid>.jsonl.gz`` (shared.transcript_archive); load_transcript must read
# them identically to a plain .jsonl so census/nightly RENDER their digests
# rather than enumerate-then-drop the whole archive at build_digest time.
# ---------------------------------------------------------------------------

class TestLoadTranscriptGzAware:
    def test_gz_transcript_parses_identically_to_plain_jsonl(self, tmp_path):
        records = [_user_text('first'), _assistant(_text('second')), _user_text('third')]
        gz_path = _write_jsonl_gz(tmp_path, records)
        plain_path = _write_jsonl(tmp_path, records)

        loaded_gz = mod.load_transcript(gz_path)

        assert loaded_gz == mod.load_transcript(plain_path)
        assert [r['message']['content'] for r in loaded_gz] == [
            'first', [{'type': 'text', 'text': 'second'}], 'third',
        ]

    def test_plain_jsonl_still_parses_as_before(self, tmp_path):
        # Byte-parity: the non-gz branch keeps the exact pre-existing read.
        records = [_user_text('one'), _user_text('two')]
        path = _write_jsonl(tmp_path, records)

        loaded = mod.load_transcript(path)

        assert [r['message']['content'] for r in loaded] == ['one', 'two']


# ---------------------------------------------------------------------------
# load_transcript — the FILE-level half of the degrade contract. This slurping
# reader has the byte-identical gzip.open + iterate body as its streaming
# sibling inventory.iter_json_lines, so it has the identical hole: an
# unreadable file surfaces as four different exception types, and only bad
# magic (gzip.BadGzipFile) is already an OSError. A truncated stream raises
# EOFError, a corrupt body raises zlib.error, and a non-UTF-8 byte raises
# UnicodeDecodeError — none of which any consumer's documented
# `except OSError` degrade path catches. The last is reachable on a plain
# `.jsonl` as well, since both branches open under strict encoding='utf-8'.
#
# The corpus extractor's `--transcript` operator mode reads through THIS
# function under `except OSError`, so an archived transcript whose write was
# interrupted aborts that run with a traceback rather than being counted.
# ---------------------------------------------------------------------------

class TestLoadTranscriptCorruptionShapes:
    def test_truncated_gz_raises_oserror(self, tmp_path):
        truncated = _write_truncated_gz(tmp_path / 'truncated.jsonl.gz')
        with pytest.raises(OSError):
            mod.load_transcript(truncated)

    def test_corrupt_body_gz_raises_oserror(self, tmp_path):
        corrupt = _write_corrupt_body_gz(tmp_path / 'corrupt-body.jsonl.gz')
        with pytest.raises(OSError):
            mod.load_transcript(corrupt)

    def test_bad_magic_gz_raises_oserror(self, tmp_path):
        # Already true (BadGzipFile subclasses OSError) — pinned here so the
        # three shapes are asserted as one contract at this reader, and so a
        # regression in the newly-added wrap cannot quietly change it.
        corrupt = tmp_path / 'not-gzip.jsonl.gz'
        corrupt.write_bytes(b'this is not gzip at all\n')
        with pytest.raises(OSError):
            mod.load_transcript(corrupt)

    def test_the_two_shapes_are_distinguishable_in_the_message(self, tmp_path):
        # Normalizing to one TYPE must not flatten them to one MESSAGE: the
        # reason string is what a coverage-reporting caller discloses, and an
        # operator has to tell a half-written transcript from a corrupted one
        # without re-reading the file.
        truncated = _write_truncated_gz(tmp_path / 'truncated.jsonl.gz')
        corrupt = _write_corrupt_body_gz(tmp_path / 'corrupt-body.jsonl.gz')

        with pytest.raises(OSError) as truncated_exc:
            mod.load_transcript(truncated)
        with pytest.raises(OSError) as corrupt_exc:
            mod.load_transcript(corrupt)

        assert str(truncated_exc.value) != str(corrupt_exc.value)
        assert 'end-of-stream' in str(truncated_exc.value)
        assert 'decompressing' in str(corrupt_exc.value)

    def test_undecodable_gz_raises_oserror(self, tmp_path):
        # The FOURTH shape. The gz container is intact — the payload simply is
        # not UTF-8 — so this arrives as UnicodeDecodeError, a ValueError
        # subclass that no `except (EOFError, zlib.error)` tuple catches and no
        # `except OSError` consumer sees. Un-normalized, one flipped byte in
        # one archived transcript aborts the corpus extractor's whole run.
        undecodable = _write_undecodable_gz(tmp_path / 'undecodable.jsonl.gz')
        with pytest.raises(OSError):
            mod.load_transcript(undecodable)

    def test_undecodable_plain_jsonl_raises_oserror(self, tmp_path):
        # Same byte, no gzip layer: this reader's plain branch opens with the
        # same strict encoding, so the shape is reachable there too.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')
        with pytest.raises(OSError):
            mod.load_transcript(undecodable)

    def test_the_decode_shape_is_distinguishable_from_the_gzip_shapes(self, tmp_path):
        # Same rule as the two-shape test above, extended to the fourth: an
        # encoding fault must not be disclosed as a gzip-stream fault, which
        # would send an operator to audit the compressor instead of the bytes.
        truncated = _write_truncated_gz(tmp_path / 'truncated.jsonl.gz')
        undecodable = _write_undecodable_gz(tmp_path / 'undecodable.jsonl.gz')

        with pytest.raises(OSError) as truncated_exc:
            mod.load_transcript(truncated)
        with pytest.raises(OSError) as undecodable_exc:
            mod.load_transcript(undecodable)

        message = str(undecodable_exc.value)
        assert message != str(truncated_exc.value)
        assert 'gzip' not in message
        assert '0xff' in message.lower()

    def test_both_readers_agree_on_an_undecodable_file(self, tmp_path):
        # The two-readers-one-contract property, asserted for the shape that
        # was leaking: whichever mode an operator reaches for, the same bad
        # byte must be reported the same way.
        undecodable = _write_undecodable_gz(tmp_path / 'undecodable.jsonl.gz')

        with pytest.raises(OSError) as slurped:
            mod.load_transcript(undecodable)
        with pytest.raises(OSError) as streamed:
            list(inventory_mod.iter_json_lines(undecodable))

        assert type(slurped.value) is type(streamed.value)
        assert str(slurped.value) == str(streamed.value)

    def test_both_readers_agree_on_a_truncated_file(self, tmp_path):
        # The property the corpus extractor's design rests on: two readers,
        # one core, ONE failure contract. Its scan mode reads via
        # inventory.iter_json_lines and its --transcript mode via
        # load_transcript; if the two disagreed here, the same corrupt archive
        # would be reported differently depending on which mode an operator
        # reached for.
        truncated = _write_truncated_gz(tmp_path / 'truncated.jsonl.gz')

        with pytest.raises(OSError) as slurped:
            mod.load_transcript(truncated)
        with pytest.raises(OSError) as streamed:
            list(inventory_mod.iter_json_lines(truncated))

        assert type(slurped.value) is type(streamed.value)
        assert str(slurped.value) == str(streamed.value)

    def test_corrupt_line_in_a_valid_gz_still_degrades_silently(self, tmp_path):
        # The line-level half, unchanged: a well-formed gz whose LAST line is
        # a half-written record still yields every parseable record and still
        # does not raise. A fix that wrapped the parse loop too broadly would
        # collapse this into the file-level path and inflate a caller's
        # unreadable-file count with ordinary trailing debris.
        path = tmp_path / 'trailing-partial.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as f:
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

    def test_normalizes_the_gzip_stream_shapes(self):
        for exc in (EOFError('ended early'), zlib.error('bad block')):
            normalized = inventory_mod.as_unreadable_file_error(exc)

            assert isinstance(normalized, OSError)
            assert 'corrupt or truncated gzip stream' in str(normalized)
            assert str(exc) in str(normalized)

    def test_decode_shape_gets_its_own_wording(self):
        # Not a "gzip stream" failure: this shape is reachable on a plain
        # .jsonl path too, so that label would misdirect an operator reading
        # a disclosed coverage reason.
        exc = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

        normalized = inventory_mod.as_unreadable_file_error(exc)

        assert isinstance(normalized, OSError)
        assert 'undecodable transcript bytes' in str(normalized)
        assert 'gzip stream' not in str(normalized)

    def test_an_oserror_passes_through_unwrapped(self):
        # gzip.BadGzipFile is ALREADY an OSError, so it needs no
        # normalization. Returning it unchanged keeps the helper idempotent —
        # a caller can hand it anything it caught without first classifying
        # it, and a bad-magic message never acquires a misleading second
        # "corrupt or truncated gzip stream" prefix.
        original = gzip.BadGzipFile('Not a gzipped file')

        assert inventory_mod.as_unreadable_file_error(original) is original

    def test_the_catch_tuple_omits_oserror_shapes(self):
        # UNREADABLE_FILE_ERRORS is exactly the set that is NOT already an
        # OSError. A shape wrongly added here would be caught and re-wrapped
        # rather than propagating, and gzip.BadGzipFile in particular must
        # keep reaching callers by its own inheritance.
        assert inventory_mod.UNREADABLE_FILE_ERRORS == (
            EOFError, zlib.error, UnicodeDecodeError,
        )
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
# size_bytes, score, n_user_turns, signal_counts), signal_counts nested in
# order (tool_error, self_correct, not_found, df_guard, interrupt). Must
# round-trip via yaml.safe_load -- NOT yaml.safe_dump (design decision: fixed
# key order, explicit formatting, deterministic byte-stable output).
#
# n_user_turns is present because of confusion-census-2026-07-31 R2 / §1.2:
# score_signals adds SIGNAL_WEIGHTS['user_turn'] * n_user_turns on top of the
# weighted signal_counts, but the frontmatter used to render only the five
# counts -- so a session whose single "user turn" was a pasted report showed
# `score: 5.0` beside an all-zero `signal_counts` and a reader could not
# reconstruct the score from the rendered fields. It is a top-level key, NOT
# a sixth signal_counts entry: signal_counts is specifically signal_counts()'s
# five detector hit-counts (mirrored by sampling.SignalCounts and read by
# _warn_if_body_evicted's any(counts.values()) guard), and n_user_turns is
# neither produced by that function nor a detector hit count.
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
        'n_user_turns': 3,
        'truncated_items': 0,
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
            'size_bytes', 'score', 'n_user_turns', 'truncated_items',
            'signal_counts',
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

    def test_n_user_turns_renders_as_bare_numeric(self):
        # Mirrors the size_bytes/score treatment: numeric frontmatter fields
        # are emitted as BARE YAML scalars (every other top-level key is
        # double-quoted via _yaml_dquote), so they load back as numbers
        # rather than strings.
        block = mod.render_frontmatter(_frontmatter_meta())

        assert 'n_user_turns: 3' in block.splitlines()

        inner = '\n'.join(block.splitlines()[1:-1])
        loaded = yaml.safe_load(inner)

        assert loaded['n_user_turns'] == 3
        assert isinstance(loaded['n_user_turns'], int)

    def test_truncated_items_renders_as_bare_numeric(self):
        block = mod.render_frontmatter(_frontmatter_meta())

        assert 'truncated_items: 0' in block.splitlines()

        inner = '\n'.join(block.splitlines()[1:-1])
        loaded = yaml.safe_load(inner)

        assert loaded['truncated_items'] == 0
        assert isinstance(loaded['truncated_items'], int)

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


# ---------------------------------------------------------------------------
# is_pasted_report_turn / iter_user_turns exclusion -- R1 (confusion census
# 2026-07-31 §1.1 facet (b), :81/:85): 5 sightings across 5 sessions where
# the flagged "User Correction" is a full Reconciliation Run Review pasted
# into a turn for review/discussion -- report output, not anyone correcting
# the agent. The generator is a literal f-string,
# fused-memory/src/fused_memory/reconciliation/judge.py:455-475
# (``_build_prompt``), so the shape is greppable rather than guessed.
#
# This is a SEPARATE predicate from is_harness_injected_turn, not a widening
# of it: a briefing preamble is injected BY the harness into a user-role
# record (the human never typed it), whereas a pasted report IS a genuine
# human turn -- the human really did paste it, it just is not a correction.
# The census tracks the two as facet (a) vs facet (b) and §6 names "zero
# facet-(a) sightings next cycle" as the discriminating fix-confirmed
# measurement, which needs the two predicates individually testable.
#
# Exclusion is by line-anchored heading CO-OCCURRENCE (all(), not any()),
# the same false-positive guard HARNESS_BRIEFING_HEADINGS and
# ORCHESTRATED_TASK_MARKERS already use: user corrections are gold (PRD Sec
# 5) and the highest-priority digest section, so over-excluding a genuine
# human turn is strictly the worse error.
# ---------------------------------------------------------------------------

def _recon_run_review_text():
    """Build the text of a pasted-reconciliation-run-review user turn -- the
    literal shape ``ReconciliationJudge._build_prompt`` emits
    (fused-memory/src/fused_memory/reconciliation/judge.py:455-475): a
    '## Reconciliation Run Review' heading, a '### Run Metadata' block whose
    first bullet is '- Run ID: {run.id}', a '### Stage Reports' block of
    ``json.dumps`` output, and '### MCP Actions (N total)' / '### Journal
    Entries (N total)' blocks. The run id matches the one the census quotes
    verbatim at §1.1 (:85).

    Deliberately free of every detector literal (NOT_FOUND_PATTERNS,
    DF_GUARD_PATTERNS, INTERRUPT_PATTERN, SELF_CORRECTION_PATTERNS) so
    test_signal_counts_unaffected_by_pasted_report_turn isolates the
    iter_user_turns filter rather than an incidental pattern hit.
    """
    return (
        '## Reconciliation Run Review\n\n'
        '### Run Metadata\n'
        '- Run ID: 84e6ce03-1f2a-4c7b-9d84-6b0f1a2e35cc\n'
        '- Project: dark_factory\n'
        '- Type: targeted\n'
        '- Trigger: task_status_change\n'
        '- Events processed: 12\n'
        '- Status: completed\n\n'
        '### Stage Reports\n'
        '[\n  {\n    "stage": 1,\n    "status": "completed",\n'
        '    "summary": "consolidated 4 memories"\n  }\n]\n\n'
        '### MCP Actions (2 total)\n'
        '[\n  {\n    "tool": "add_memory",\n    "created_at": "2026-07-31T09:00:00Z"\n  }\n]\n\n'
        '### Journal Entries (5 total)\n'
        '[\n  {\n    "kind": "stage_start",\n    "created_at": "2026-07-31T09:00:00Z"\n  }\n]\n\n'
        'Review this run and provide your verdict as JSON.\n'
    )


class TestPastedReportTurnFilter:
    def test_recon_run_review_turn_is_excluded(self):
        records = [_user_text(_recon_run_review_text())]

        assert mod.iter_user_turns(records) == []

    def test_ordinary_correction_after_pasted_report_retained_with_index(self):
        records = [
            _user_text(_recon_run_review_text()),
            _user_text('This is wrong, please redo it.'),
        ]

        turns = mod.iter_user_turns(records)

        assert [t['text'] for t in turns] == ['This is wrong, please redo it.']
        assert [t['index'] for t in turns] == [1]

    def test_single_report_heading_alone_is_not_excluded(self):
        # Only ONE of the co-occurring headings, and the rest of the turn is
        # a genuine human question -- must not alone trigger exclusion
        # (all()-not-any(), mirroring test_single_heading_alone_is_not_excluded).
        text = '## Reconciliation Run Review\n\nwhat happened in this run?'
        records = [_user_text(text)]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1
        assert turns[0]['text'] == text

    def test_report_heading_mentioned_mid_sentence_is_not_excluded(self):
        # Line-anchored matching: the headings appearing mid-sentence (not
        # each as its own stripped line) never count as headings, even when
        # every one of them is mentioned.
        records = [_user_text(
            'The ## Reconciliation Run Review prompt emits ### Run Metadata '
            'and ### Stage Reports — please document that in the runbook.'
        )]

        turns = mod.iter_user_turns(records)

        assert len(turns) == 1

    def test_render_digest_excludes_pasted_report_from_body_and_n_user_turns(self):
        records = [
            _with_session_meta(_user_text(_recon_run_review_text())),
            _with_session_meta(_user_text('This is wrong, please redo it.')),
        ]

        digest = mod.render_digest(records, agent_class='interactive')

        frontmatter_yaml, body = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert 'This is wrong, please redo it.' in body
        assert 'Run ID:' not in body
        assert '### Stage Reports' not in body
        # Census §3.1: clusters 1.1(b) and 1.2 are one event seen from two
        # surfaces, so the single iter_user_turns filter must fix the body
        # AND the score together (today: n_user_turns would be 2).
        assert meta['n_user_turns'] == 1
        assert meta['score'] == mod.score_signals(meta['signal_counts'], 1)

    def test_signal_counts_unaffected_by_pasted_report_turn(self):
        # The filter must not leak into _signal_text_sources: prepending a
        # signal-free pasted report must not perturb any of the five counts.
        base = _all_signals_records()
        with_report = [_user_text(_recon_run_review_text())] + base

        assert mod.signal_counts(with_report) == mod.signal_counts(base)

    def test_pasted_report_is_not_classified_as_harness_injected(self):
        # The two predicates stay semantically distinct: a pasted report is
        # NOT harness-injected (a human really did paste it), which is why
        # it is a separate predicate rather than a widening of the other.
        # Keeping them individually testable is what lets a future census
        # tell facet (a) from facet (b) (census §6).
        text = _recon_run_review_text()

        assert mod.is_harness_injected_turn(text) is False
        assert mod.is_pasted_report_turn(text) is True
        assert mod.is_pasted_report_turn(_briefing_text()) is False


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

    def test_frontmatter_score_is_reconstructible_from_rendered_fields(self):
        # confusion-census-2026-07-31 R2 / §1.2: score_signals' dominant
        # component is SIGNAL_WEIGHTS['user_turn'] * n_user_turns, which the
        # frontmatter used to omit entirely -- so `score` could not be
        # checked against `signal_counts` by a reader of the digest alone.
        # With n_user_turns rendered, the score is fully reconstructible.
        records = _all_signals_records()

        digest = mod.render_digest(records, agent_class='interactive')

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['n_user_turns'] == len(mod.iter_user_turns(records))
        assert meta['score'] == mod.score_signals(
            meta['signal_counts'], meta['n_user_turns'],
        )

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
        # 'in', not endswith: the marker is a stable greppable PREFIX with
        # a variable quantified tail (see the R1 tests below).
        assert mod.ITEM_TRUNCATION_MARKER in capped

    def test_truncation_marker_is_a_stable_greppable_prefix(self):
        # confusion-census-2026-07-31 R1: the marker gains a quantified
        # tail, but the constant itself stays a fixed ASCII PREFIX so every
        # existing `MARKER in digest` grep/test keeps working unchanged.
        assert mod.ITEM_TRUNCATION_MARKER == '... [item truncated'

    def test_truncation_marker_reports_bytes_dropped_and_original_size(self):
        # census §3.4, "rendered surfaces that omit their own basis": a
        # fixed opaque marker says something was dropped but never how
        # much, so a 20KB pasted turn capped to 2KB looks identical to one
        # capped by 40 bytes. The two numbers must be internally
        # consistent, not decorative.
        line = '- (turn 0) ' + ('x' * 5000)
        cap = 2048

        capped = mod._cap_item(line, cap)

        match = re.search(
            r'\.\.\. \[item truncated: (\d+) of (\d+) bytes dropped\]$', capped,
        )
        assert match is not None, capped[-120:]

        original_bytes = len(line.encode('utf-8'))
        suffix_bytes = len(match.group(0).encode('utf-8'))
        kept_bytes = len(capped.encode('utf-8')) - suffix_bytes

        assert int(match.group(2)) == original_bytes
        assert int(match.group(1)) == original_bytes - kept_bytes

    def test_quantified_marker_still_respects_the_byte_cap(self):
        # The suffix's own byte length depends on the digit counts of the
        # numbers it reports, which depend on how many bytes the suffix
        # leaves room for -- the cap must hold anyway, for ASCII and
        # multi-byte alike, with no mojibake.
        for line in (
            '- (turn 0) ' + ('x' * 5000),
            '- (turn 0) ' + ('é→' * 2000),
        ):
            for cap in (256, 500, 1024, 2048):
                capped = mod._cap_item(line, cap)

                assert len(capped.encode('utf-8')) <= cap, (cap, line[:20])
                assert '�' not in capped
                assert mod.ITEM_TRUNCATION_MARKER in capped

    def test_truncation_suffix_fits_within_min_item_bytes(self):
        # Makes the `<= cap` guarantee structural rather than incidental:
        # _item_byte_cap never returns below MIN_ITEM_BYTES, so as long as
        # the longest possible suffix fits inside MIN_ITEM_BYTES there is
        # always room for it plus at least some retained content.
        longest = mod._truncation_suffix(999_999_999_999, 999_999_999_999)

        assert len(longest.encode('utf-8')) < mod.MIN_ITEM_BYTES

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
# truncated_items frontmatter key -- R1 (confusion census 2026-07-31, second
# truncation-marking surface): the per-item marker makes a truncated item
# legible only to someone reading that far into the body. The frontmatter
# reports the count up front, so a reader (or the downstream trickle coder)
# knows the body it is about to weigh has had substance capped out of it.
#
# The count describes the POST-trim body: _truncate_sections pops trailing
# items until the digest fits, so a truncated item can be removed entirely
# and a pre-trim count would claim the shipped body contains a truncated
# item it does not -- a fresh instance of the very defect being fixed
# (census §3.4).
# ---------------------------------------------------------------------------

def _multiple_oversized_items_records():
    """One oversized gold user turn plus three oversized error
    neighborhoods. Every rendered item exceeds the per-item cap, so all
    four carry a truncation marker BEFORE trimming; at a small max_bytes
    the three lower-priority error items (SECTION_PRIORITY index 2) are
    popped while the gold turn (index 6, trimmed last) survives. That gap
    is what makes test_truncated_items_never_counts_items_the_soft_cap_evicted
    discriminating: a pre-trim count would report 4 for a body containing 1.
    """
    records = [
        _with_session_meta(_user_text('This is wrong, please redo it properly. ' * 500)),
    ]
    for n in range(3):
        records.append(_with_session_meta(
            _assistant(_tool_use('Bash', {'command': 'false'}, id=f'tu-err-{n}')),
        ))
        records.append(_with_session_meta(
            _tool_result(f'tu-err-{n}', 'Exit code 1: ' + ('e' * 4000), is_error=True),
        ))
    return records


class TestTruncatedItemsFrontmatter:
    def test_truncated_items_is_zero_when_nothing_was_truncated(self):
        # _all_signals_records() is well under the 15360 default cap, so
        # nothing is truncated and the key must say so rather than be absent.
        digest = mod.render_digest(_all_signals_records(), agent_class='interactive')

        frontmatter_yaml, _ = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['truncated_items'] == 0
        assert mod.ITEM_TRUNCATION_MARKER not in digest

    def test_truncated_items_matches_marker_count_in_final_body(self):
        digest = mod.render_digest(
            _oversized_user_correction_records(), agent_class='interactive',
        )

        frontmatter_yaml, body = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        assert meta['truncated_items'] >= 1
        assert meta['truncated_items'] == body.count(mod.ITEM_TRUNCATION_MARKER)

    def test_truncated_items_never_counts_items_the_soft_cap_evicted(self):
        # Four truncated items pre-trim; at max_bytes=1200 the three
        # lower-priority ones are popped. The count must describe the body
        # that actually ships, never over-report what was evicted.
        max_bytes = 1200
        records = _multiple_oversized_items_records()

        digest = mod.render_digest(
            records, agent_class='interactive', max_bytes=max_bytes,
        )

        frontmatter_yaml, body = _split_frontmatter(digest)
        meta = yaml.safe_load(frontmatter_yaml)

        pre_trim = sum(
            line.count(mod.ITEM_TRUNCATION_MARKER)
            for lines in mod._build_sections(
                records, item_max_bytes=mod._item_byte_cap(max_bytes),
            ).values()
            for line in lines
        )

        assert pre_trim > meta['truncated_items'], (
            'fixture must actually evict a truncated item for this test to '
            f'discriminate (pre_trim={pre_trim})'
        )
        assert meta['truncated_items'] == body.count(mod.ITEM_TRUNCATION_MARKER)
        assert len(digest.encode('utf-8')) <= max_bytes


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

    def test_build_digest_end_to_end_on_gz_orchestrated_session(self, tmp_path):
        # Already green off steps 2/4 -- the end-to-end acceptance check:
        # a real orchestrated-looking gz transcript (briefing turn +
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
        path = _write_jsonl_gz(tmp_path, records)

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
