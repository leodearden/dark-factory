"""Tests for the toolcall-markup fixture corpus and its extractor.

PRD ``plans/toolcall-markup-containment-prd.md`` task alpha, boundary row B13.

TDD pair 4 (step 7/8): the extractor's pure functions, over SYNTHETIC plain
transcripts written to ``tmp_path``. Nothing here reads the live archive, so
these stay green after the retention window rotates it away (PRD section 11.4).

TDD pair 5 (step 9/10): the committed-corpus replay contract.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
``<``, exactly as ``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86
require. Writing ``<`` verbatim here would force any agent editing this file to
emit that literal inside its own tool-call envelope, reproducing the very defect
these tests pin. Leave it escaped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import toolcall_markup_corpus_extract as extract

# The D5 assertion helper is imported, NOT re-derived: it is the one check in
# the replay that is non-circular with respect to the machine-generated
# expectation column, so it must be the SAME code the hand-authored specimen
# tests run. Bare import works via conftest's tests/ sys.path insert (the
# capability_manifest_corpus.py precedent).
from test_toolcall_markup import assert_repair_invariants

from shared.toolcall_markup import INVOKE_CLOSER, repair

# ---------------------------------------------------------------------------
# Synthetic-transcript builders.
# ---------------------------------------------------------------------------


def _closer(name: str) -> str:
    return '\x3c/' + name + '>'


def _opener(name: str) -> str:
    return '\x3c' + name + '>'


def _canonical_opener(name: str) -> str:
    return '\x3cparameter name="' + name + '">'


def _tool_use(block_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    """One assistant record carrying one tool_use block.

    Shape re-confirmed against the live archive at revalidation: transcript
    lines are JSON objects, assistant records carry ``message.content`` as a
    LIST of blocks, and a tool_use block has ``{type, id, name, input, caller}``
    with a unique ``toolu_...`` id.
    """
    return {
        'type': 'assistant',
        'message': {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'some narration'},
                {
                    'type': 'tool_use',
                    'id': block_id,
                    'name': name,
                    'input': arguments,
                    'caller': 'agent',
                },
            ],
        },
    }


def _write_transcript(path: Path, records: list[dict[str, object]]) -> None:
    """Write *records* as a plain JSONL transcript at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def _archive_leaf(root: Path, task_id: str, uuid: str) -> Path:
    """A leaf at the REAL archive depth: root/<task_id>/<project-dir>/<uuid>.

    Mirrors the layout measured 2026-08-08 (prerequisite P3), with the plain
    suffix the archive has carried since task 3618:
    ``<root>/2474/-home-leo-src-dark-factory--worktrees-2474/<uuid>.jsonl``.
    """
    project_dir = '-home-leo-src-dark-factory--worktrees-' + task_id
    return root / task_id / project_dir / (uuid + '.jsonl')


# A total-drift value: the tail ends at the invoke closer (predicate alt 1).
_TOTAL_DRIFT = (
    'Investigate the divergence and report back.'
    + _closer('description') + '\n'
    + _opener('priority') + 'medium' + _closer('priority') + '\n'
    + INVOKE_CLOSER
)
# A partial-drift value: the tail ends in an unterminated canonical opener
# (predicate alt 2), so it never reaches an invoke closer.
_PARTIAL_DRIFT = (
    'The scheduler retries this automatically).'
    + _closer('description') + '\n'
    + _canonical_opener('priority') + 'low'
)


class TestCollectionPredicate:
    """PRD section 2 (line 35), verbatim — the numbers are reproducible or not.

    ``\\x3c/invoke>\\s*$`` OR
    ``\\x3c/[A-Za-z_]\\w*>\\s*\\x3cparameter\\s+name="[^"]+">``.
    """

    @pytest.mark.parametrize(
        'value',
        [
            _TOTAL_DRIFT,
            _PARTIAL_DRIFT,
            # alt 1 with trailing whitespace, which the \s* explicitly allows
            'a value that just ends ' + INVOKE_CLOSER + '  \n',
            # alt 2 with the canonical closer as the mis-close (specimen 4)
            'scoped).' + _closer('parameter') + '\n' + _canonical_opener('agent_id') + 'x',
        ],
    )
    def test_positive_rows(self, value):
        assert extract.matches_collection_predicate(value) is True

    @pytest.mark.parametrize(
        'value',
        [
            None,
            '',
            17,
            {'a': 'b'},
            'ordinary prose with no markup at all',
            # a LONE closer in prose: alt 1 needs end-of-string, alt 2 needs a
            # canonical opener after it. This is the row that keeps the corpus
            # from filling up with documentation that quotes a tag.
            'the audit quotes ' + _closer('content') + ' in the middle of a sentence',
            # a name-echoing closer followed by something that is NOT a
            # canonical opener
            'value' + _closer('description') + '\n' + _opener('priority') + 'medium',
            # an invoke closer that is not at the END of the string
            'lead ' + INVOKE_CLOSER + ' and then more prose',
            # a canonical opener with an EMPTY name — [^"]+ requires one char
            'value' + _closer('description') + '\n' + '\x3cparameter name="">low',
        ],
    )
    def test_negative_rows(self, value):
        assert extract.matches_collection_predicate(value) is False


class TestArchiveWalk:
    """The revalidation finding (P3): the archive is NESTED, not flat.

    A top-level ``*.jsonl`` glob returns ZERO files against the real archive,
    so without this test a silently-empty corpus would look like a clean run.
    """

    def test_finds_leaves_nested_two_directories_down(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        first = _archive_leaf(root, '2474', 'aaaa')
        second = _archive_leaf(root, '3688', 'bbbb')
        _write_transcript(first, [])
        _write_transcript(second, [])

        found = list(extract.iter_transcript_files(root))

        assert set(found) == {first, second}

    def test_ignores_a_decoy_beside_the_leaf(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        leaf = _archive_leaf(root, '2474', 'aaaa')
        _write_transcript(leaf, [])
        decoy = leaf.parent / 'metadata.json'
        decoy.write_text('{}', encoding='utf-8')
        other_decoy = leaf.parent / 'aaaa.jsonl'
        other_decoy.write_text('{}', encoding='utf-8')

        assert list(extract.iter_transcript_files(root)) == [leaf]

    def test_walk_order_is_deterministic(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        for task_id, uuid in (('30', 'c'), ('4', 'a'), ('200', 'b')):
            _write_transcript(_archive_leaf(root, task_id, uuid), [])

        first = list(extract.iter_transcript_files(root))
        second = list(extract.iter_transcript_files(root))

        assert first == second
        assert first == sorted(first)

    def test_missing_root_yields_nothing_without_raising(self, tmp_path):
        assert list(extract.iter_transcript_files(tmp_path / 'nope')) == []


class TestExtractRecords:
    """The record schema, dedup, and deterministic ordering."""

    def test_record_schema(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [
                _tool_use(
                    'toolu_01',
                    'submit_task',
                    {
                        'project_root': '/repo',
                        'title': 'a title',
                        'description': _TOTAL_DRIFT,
                    },
                ),
                # A CLEAN call of the same tool. It is what testifies that
                # `priority` exists: a dropped parameter is by definition absent
                # from the corrupted call's own key set, so without a clean
                # sibling somewhere in the archive the recovered name fails
                # schema validation and the specimen reads as unrepairable.
                _tool_use('toolu_clean', 'submit_task', {'title': 't', 'priority': 'low'}),
            ],
        )

        records = extract.extract(root)

        assert len(records) == 1
        record = records[0]
        assert set(record) == {
            'tool',
            'param',
            'supplied',
            'value',
            'expected_outcome',
            'expected_recovered',
            'schema_params',
            'tool_use_id',
            'truncated',
            'original_length',
        }
        assert record['tool'] == 'submit_task'
        assert record['param'] == 'description'
        assert record['supplied'] == ['description', 'project_root', 'title']
        assert record['tool_use_id'] == 'toolu_01'
        assert record['value'] == _TOTAL_DRIFT
        assert record['original_length'] == len(_TOTAL_DRIFT)
        assert record['truncated'] is False
        assert record['expected_outcome'] == 'repaired'
        assert record['expected_recovered'] == ['priority']

    def test_schema_params_is_the_union_observed_across_the_archive(self, tmp_path):
        """Self-contained: replay needs no live MCP schema (PRD section 11.4)."""
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [
                _tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT}),
                # a CLEAN call of the same tool, contributing more schema keys
                _tool_use(
                    'toolu_02',
                    'submit_task',
                    {'title': 't', 'priority': 'low', 'metadata': {}, 'project_root': '/repo'},
                ),
            ],
        )

        records = extract.extract(root)

        assert len(records) == 1
        assert records[0]['schema_params'] == [
            'description',
            'metadata',
            'priority',
            'project_root',
            'title',
        ]

    def test_dedupes_by_tool_use_id_across_files(self, tmp_path):
        """Resumed sessions and re-archival really do duplicate a block.

        Measured at revalidation: 30 raw hits across 29 unique tool_use ids in
        the first 900 archive files. An un-deduped extraction is not
        reproducible, so this is a correctness requirement, not a tidy-up.
        """
        root = tmp_path / 'agent-transcripts'
        record = _tool_use('toolu_dup', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})
        _write_transcript(_archive_leaf(root, '2474', 'aaaa'), [record])
        _write_transcript(_archive_leaf(root, '2474', 'bbbb'), [record])

        records = extract.extract(root)

        assert [r['tool_use_id'] for r in records] == ['toolu_dup']

    def test_ordering_is_a_stable_total_order(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '9', 'zzz'),
            [_tool_use('toolu_c', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})],
        )
        _write_transcript(
            _archive_leaf(root, '1', 'aaa'),
            [
                _tool_use('toolu_a', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT}),
                _tool_use('toolu_b', 'submit_task', {'title': 't', 'description': _PARTIAL_DRIFT}),
            ],
        )

        first = extract.extract(root)
        second = extract.extract(root)

        assert [r['tool_use_id'] for r in first] == ['toolu_a', 'toolu_b', 'toolu_c']
        assert first == second

    def test_skips_values_that_do_not_match_the_predicate(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [
                _tool_use('toolu_clean', 'add_memory', {'content': 'perfectly ordinary text'}),
                _tool_use('toolu_int', 'add_memory', {'content': 17}),
            ],
        )

        assert extract.extract(root) == []

    def test_unrepairable_value_is_recorded_as_such(self, tmp_path):
        """The ambiguous cases are the ones a future improvement is judged on."""
        root = tmp_path / 'agent-transcripts'
        # `nonesuch` is not a key this tool was ever seen with, so the recovered
        # name fails schema validation (boundary row B8) -> unrepairable.
        value = (
            'body text'
            + _closer('description') + '\n'
            + _opener('nonesuch') + 'x' + _closer('nonesuch') + '\n'
            + INVOKE_CLOSER
        )
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': value})],
        )

        records = extract.extract(root)

        assert len(records) == 1
        assert records[0]['expected_outcome'] == 'unrepairable'
        assert records[0]['expected_recovered'] == []
        assert records[0]['value'] == value
        assert records[0]['truncated'] is False


class TestTruncationRule:
    """PRD section 11.3 — repaired cases truncate by RULE, ambiguous ones never."""

    def test_repaired_value_keeps_lead_in_plus_everything_from_the_misclose(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        lead = 'z' * 5000
        value = (
            lead
            + _closer('description') + '\n'
            + _opener('priority') + 'medium' + _closer('priority') + '\n'
            + INVOKE_CLOSER
        )
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [
                _tool_use('toolu_01', 'submit_task', {'title': 't', 'description': value}),
                _tool_use('toolu_clean', 'submit_task', {'title': 't', 'priority': 'low'}),
            ],
        )

        record = extract.extract(root)[0]

        assert record['truncated'] is True
        assert record['original_length'] == len(value)
        assert len(record['value']) < len(value)
        # up-to-200 chars of lead-in, then EVERYTHING from the mis-close onward
        assert record['value'] == lead[-extract.LEAD_IN_CHARS:] + value[len(lead):]
        # and the pin still pins real behaviour
        assert record['expected_outcome'] == 'repaired'
        assert record['expected_recovered'] == ['priority']

    def test_short_repaired_value_is_not_truncated(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})],
        )

        record = extract.extract(root)[0]

        assert record['truncated'] is False
        assert record['value'] == _TOTAL_DRIFT

    def test_unrepairable_value_is_never_truncated(self, tmp_path):
        """PRD section 11.3: the ambiguous cases stay VERBATIM, however long."""
        root = tmp_path / 'agent-transcripts'
        value = (
            'y' * 5000
            + _closer('description') + '\n'
            + _opener('nonesuch') + 'x' + _closer('nonesuch') + '\n'
            + INVOKE_CLOSER
        )
        _write_transcript(
            _archive_leaf(root, '2474', 'aaaa'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': value})],
        )

        record = extract.extract(root)[0]

        assert record['expected_outcome'] == 'unrepairable'
        assert record['truncated'] is False
        assert record['value'] == value

    def test_refuses_to_truncate_when_the_repair_would_differ(self):
        """A truncated record that repairs DIFFERENTLY would pin a fiction.

        Direct unit test of the rule, because constructing an archive record
        that trips it is far less legible than calling it.
        """
        # A value whose mis-close is inside the first 200 chars is unaffected;
        # one whose repair depends on text the truncation would drop must fall
        # back to verbatim rather than emit a record that pins the wrong thing.
        value = 'w' * 5000 + _closer('content') + '\n' + INVOKE_CLOSER
        stored, truncated = extract.apply_truncation(
            value,
            param='content',
            schema_params=('content', 'project_id'),
            supplied=('content',),
        )
        assert truncated is True
        assert stored == 'w' * extract.LEAD_IN_CHARS + value[5000:]

        # An unrepairable value is returned verbatim and untruncated.
        unrepairable = 'w' * 5000 + _closer('rationale') + ' leftover prose'
        stored, truncated = extract.apply_truncation(
            unrepairable,
            param='content',
            schema_params=('content',),
            supplied=('content',),
        )
        assert truncated is False
        assert stored == unrepairable

    def test_a_stored_value_always_still_satisfies_the_predicate(self):
        """Truncation must never strip the substring the predicate matched on.

        Both predicate alternatives ``search`` the whole value, so a specimen
        can be collected on a mis-close-then-opener sequence sitting MID-value
        while ``repair`` accepts a later mis-close. Truncating to lead-in plus
        the accepted tail drops the middle, and the stored value is then one the
        corpus's own collection predicate would never have picked up — which
        would force the replay test's non-circular re-derivation to carry an
        exemption. Store it verbatim instead. Observed once in the 473-specimen
        2026-08-08 snapshot, so this is a real archive shape, not a hypothetical.

        Pinned as an INVARIANT OVER THE OUTPUT rather than as a branch of
        :func:`apply_truncation`, because WHICH refusal fires moved under review
        fix 2 and could move again. It used to be the predicate guard: ``repair``
        accepted the trailing mis-close, and truncation would have dropped the
        middle. It is now the earlier ``result is None`` branch — the prefix
        ``repair`` would have had to hand back still carries the canonical
        opener, so the prefix-clean post-condition declines the whole shape
        rather than return a partial repair. Either way the specimen is stored
        verbatim and the stored value is still collectable, which is the only
        thing the corpus actually depends on.
        """
        value = (
            'w' * 5000
            # collected on THIS: a mis-close then a canonical opener (alt 2) —
            # but the name is not in the schema, so repair refuses here...
            + _closer('rationale') + '\n'
            + _canonical_opener('nonesuch') + 'x' + _closer('parameter') + '\n'
            + 'w' * 500
            # ...and USED TO accept HERE, where the tail is whitespace only.
            # The clean_value that would have produced is not envelope-free.
            + _closer('rationale') + '\n'
        )
        assert extract.matches_collection_predicate(value)
        assert repair(value, 'decision', ('decision', 'how', 'rationale'), ('decision',)) is None

        stored, truncated = extract.apply_truncation(
            value,
            param='decision',
            schema_params=('decision', 'how', 'rationale'),
            supplied=('decision',),
        )

        assert truncated is False
        assert stored == value
        assert extract.matches_collection_predicate(stored)


class TestCorpusWriter:
    """Every '<' is written as its \\u003c JSON escape."""

    def test_emitted_text_carries_no_literal_opening_angle_bracket(self, tmp_path):
        out = tmp_path / 'corpus.jsonl'
        records = [
            {
                'tool': 'submit_task',
                'param': 'description',
                'supplied': ['description', 'title'],
                'value': _TOTAL_DRIFT,
                'expected_outcome': 'repaired',
                'expected_recovered': ['priority'],
                'schema_params': ['description', 'priority', 'title'],
                'tool_use_id': 'toolu_01',
                'truncated': False,
                'original_length': len(_TOTAL_DRIFT),
            }
        ]

        extract.write_corpus(records, out)

        text = out.read_text(encoding='utf-8')
        assert '\x3c' not in text
        assert '\\u003c' in text

    def test_round_trips_through_the_loader(self, tmp_path):
        out = tmp_path / 'corpus.jsonl'
        records = [
            {
                'tool': 'submit_task',
                'param': 'description',
                'supplied': ['description', 'title'],
                'value': _TOTAL_DRIFT,
                'expected_outcome': 'repaired',
                'expected_recovered': ['priority'],
                'schema_params': ['description', 'priority', 'title'],
                'tool_use_id': 'toolu_01',
                'truncated': False,
                'original_length': len(_TOTAL_DRIFT),
            }
        ]

        extract.write_corpus(records, out)

        assert extract.load_corpus(out) == records

    def test_writing_the_same_records_twice_is_byte_identical(self, tmp_path):
        records = [
            {
                'tool': 'add_memory',
                'param': 'content',
                'supplied': ['content'],
                'value': _PARTIAL_DRIFT,
                'expected_outcome': 'unrepairable',
                'expected_recovered': [],
                'schema_params': ['content'],
                'tool_use_id': 'toolu_02',
                'truncated': False,
                'original_length': len(_PARTIAL_DRIFT),
            }
        ]
        first = tmp_path / 'a.jsonl'
        second = tmp_path / 'b.jsonl'

        extract.write_corpus(records, first)
        extract.write_corpus(records, second)

        assert first.read_bytes() == second.read_bytes()


class TestExtractorRobustness:
    """The archive is 551 MB of live-written data; a bad line is not a crash."""

    def test_malformed_lines_are_skipped(self, tmp_path):
        root = tmp_path / 'agent-transcripts'
        leaf = _archive_leaf(root, '2474', 'aaaa')
        leaf.parent.mkdir(parents=True, exist_ok=True)
        good = json.dumps(
            _tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})
        )
        with open(leaf, 'w', encoding='utf-8') as handle:
            handle.write('{"type": "assistant", "tool_use": truncated mid-writ')
            handle.write('\n')
            handle.write('\n')
            handle.write('[1, 2, 3]\n')
            handle.write(json.dumps({'type': 'assistant', 'message': 'not a dict'}) + '\n')
            handle.write(json.dumps({'type': 'user', 'message': {'content': 'hi'}}) + '\n')
            handle.write(good + '\n')

        records = extract.extract(root)

        assert [r['tool_use_id'] for r in records] == ['toolu_01']


# ---------------------------------------------------------------------------
# Unreadable FILES (step 16/17). Distinct from the corrupt LINES above, and the
# split between the two is a contract, not an implementation detail.
# ---------------------------------------------------------------------------

#: A transcript carrying a raw non-UTF-8 byte INSIDE a tool_use argument. Under
#: ``errors='replace'`` this decodes to a U+FFFD and parses, so the specimen
#: enters the corpus with a replacement character standing in for real harness
#: output. Strict decoding is what makes it an unreadable FILE instead.
_UNDECODABLE_BYTES = (
    json.dumps(
        _tool_use('toolu_undecodable', 'submit_task', {'title': 'PLACEHOLDER', 'description': _TOTAL_DRIFT})
    ).encode('utf-8').replace(b'PLACEHOLDER', b't\xff') + b'\n'
)

#: The ONE shape ``legibility.inventory.as_unreadable_file_error`` still
#: normalizes: an undecodable byte (``UnicodeDecodeError``). It is what a
#: fire-and-forget writer produces when a unit is killed mid-write — the
#: archive is live-written AND concurrently pruned by
#: scripts/gc_agent_transcripts.py, so that is an expected state, not a
#: theoretical one.
#:
#: WHY THE OTHER THREE ARE GONE (task 3618 stored the archive plain). This
#: class originally parametrized over four shapes chosen to hit four DISTINCT
#: gzip-container exceptions. With no gzip container, re-measured against the
#: plain reader:
#:   - ``truncated_stream`` and ``corrupt_body`` were gzip byte-strings; read
#:     as plain UTF-8 they still raise ``UnicodeDecodeError``, i.e. they became
#:     redundant re-spellings of the case below rather than distinct shapes,
#:     and their names no longer describe what they test.
#:   - ``bad_magic`` was literal ASCII ("plain text that was never gzipped");
#:     read plain it is perfectly READABLE and merely yields zero records. It
#:     changed class entirely, so it moved to the readable-file test below
#:     rather than being deleted.
#: The contract task 3688 pinned — skip-and-report, never fatal — is unchanged
#: and is still exercised end-to-end by the surviving shape.
_UNREADABLE_SHAPES = {
    'undecodable_byte': _UNDECODABLE_BYTES,
}


class TestUnreadableFilesAreSkippedAndCounted:
    """An unreadable transcript must be SKIPPED and REPORTED, never fatal.

    The failure this guards is worse than a crash: under ``errors='replace'``
    a corrupt byte silently becomes U+FFFD and is committed to the corpus as
    if it were real harness output.

    One bad leaf must not cost the other 5688 files, and a run that loses files
    must be distinguishable from a clean one.
    """

    @pytest.mark.parametrize('shape', sorted(_UNREADABLE_SHAPES))
    def test_a_bad_leaf_does_not_cost_the_good_leaf_beside_it(self, tmp_path, shape):
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'good'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})],
        )
        bad = _archive_leaf(root, '2475', 'bad')
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(_UNREADABLE_SHAPES[shape])

        records = extract.extract(root)

        assert [r['tool_use_id'] for r in records] == ['toolu_01']

    def test_an_undecodable_byte_is_skipped_not_silently_replaced(self, tmp_path):
        """(d) A U+FFFD inside a stored specimen would be a fabricated value.

        The corpus is evidence about what the harness really emitted. A
        replacement character committed as if it were harness output is exactly
        the fabrication the D5 invariant exists to rule out on the repair side,
        so the honest answer is to count the FILE as unreadable and drop it.
        """
        root = tmp_path / 'agent-transcripts'
        bad = _archive_leaf(root, '2474', 'bad')
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(_UNDECODABLE_BYTES)
        _write_transcript(
            _archive_leaf(root, '2475', 'good'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})],
        )

        records = extract.extract(root)

        assert [r['tool_use_id'] for r in records] == ['toolu_01']
        assert not any('�' in json.dumps(record) for record in records)

    @pytest.mark.parametrize('shape', sorted(_UNREADABLE_SHAPES))
    def test_the_skip_is_counted_and_names_the_path(self, tmp_path, shape):
        """(e) LOUD, not silent — a run that loses files says which ones.

        A skip nobody can see is the silent degradation the shared ratchet and
        the project's loud-over-silent norm both exist to prevent: the corpus
        would just quietly shrink.
        """
        root = tmp_path / 'agent-transcripts'
        _write_transcript(
            _archive_leaf(root, '2474', 'good'),
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})],
        )
        bad = _archive_leaf(root, '2475', 'bad')
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(_UNREADABLE_SHAPES[shape])

        report = extract.WalkReport()
        extract.extract(root, report=report)

        assert report.files_walked == 2
        assert report.files_read == 1
        assert [path for path, _ in report.unreadable] == [str(bad)]
        # the normalized message survives, so the disclosed skip still says
        # WHICH shape it was rather than just that something went wrong
        assert report.unreadable[0][1]

    def test_the_skip_is_also_logged_at_warning(self, tmp_path, caplog):
        root = tmp_path / 'agent-transcripts'
        bad = _archive_leaf(root, '2474', 'bad')
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(_UNREADABLE_SHAPES['undecodable_byte'])

        with caplog.at_level('WARNING'):
            extract.extract(root)

        assert any(str(bad) in record.getMessage() for record in caplog.records)

    def test_a_run_with_zero_readable_files_is_loud_and_writes_nothing(self, tmp_path):
        """(e) An all-unreadable archive must not emit an empty corpus.

        Same failure shape the existing zero-files-walked guard covers, one
        level in: files were found, so that guard passes, and without this one
        the run would overwrite the committed corpus with an empty file and
        exit 0.
        """
        root = tmp_path / 'agent-transcripts'
        for index, data in enumerate(sorted(_UNREADABLE_SHAPES.values())):
            leaf = _archive_leaf(root, '247' + str(index), 'bad')
            leaf.parent.mkdir(parents=True, exist_ok=True)
            leaf.write_bytes(data)
        out = tmp_path / 'corpus.jsonl'

        assert extract.main(['--archive-root', str(root), '--out', str(out)]) == 1
        assert not out.exists()

    def test_a_readable_file_yielding_no_records_is_not_unreadable(self, tmp_path):
        """The successor to the old ``bad_magic`` shape, which changed class.

        Pre-task-3618 this content ("plain text that was never gzipped at all")
        was an UNREADABLE file: it hit gzip.BadGzipFile at the container layer.
        Stored plain it is valid UTF-8 that simply parses to zero records, so
        it is now the OPPOSITE case — and worth pinning from that side, because
        conflating "read fine, had nothing" with "could not be read" is exactly
        what would inflate the unreadable count until the alarm stops meaning
        anything.
        """
        root = tmp_path / 'agent-transcripts'
        leaf = _archive_leaf(root, '2474', 'aaaa')
        leaf.parent.mkdir(parents=True, exist_ok=True)
        leaf.write_bytes(b'plain text that was never gzipped at all\n')

        report = extract.WalkReport()
        records = extract.extract(root, report=report)

        assert records == []
        assert report.files_walked == 1
        assert report.files_read == 1
        assert report.unreadable == []

    def test_a_corrupt_trailing_line_is_not_an_unreadable_file(self, tmp_path):
        """(f) The line/file split, pinned from the side that must NOT alarm.

        Every fire-and-forget writer eventually produces a truncated trailing
        line. Counting that as an unreadable file would inflate the alarm until
        nobody reads it, which is how a real loss gets missed.
        """
        root = tmp_path / 'agent-transcripts'
        leaf = _archive_leaf(root, '2474', 'aaaa')
        leaf.parent.mkdir(parents=True, exist_ok=True)
        good = json.dumps(
            _tool_use('toolu_01', 'submit_task', {'title': 't', 'description': _TOTAL_DRIFT})
        )
        with open(leaf, 'w', encoding='utf-8') as handle:
            handle.write(good + '\n')
            handle.write('{"type": "assistant", "tool_use": truncated mid-writ')

        report = extract.WalkReport()
        records = extract.extract(root, report=report)

        assert [r['tool_use_id'] for r in records] == ['toolu_01']
        assert report.files_read == 1
        assert report.unreadable == []


# ---------------------------------------------------------------------------
# The committed-corpus replay contract (step 9/10, boundary row B13).
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).resolve().parent / 'fixtures' / 'toolcall_markup_corpus.jsonl'


@pytest.fixture(scope='module')
def corpus() -> list[dict]:
    assert CORPUS_PATH.is_file(), f'committed corpus missing at {CORPUS_PATH}'
    return extract.load_corpus(CORPUS_PATH)


class TestCommittedCorpus:
    """Replay the committed specimens (PRD boundary row B13).

    NO COUNT ASSERTIONS ANYWHERE IN THIS CLASS. PRD G6 is explicit that pinning
    the reference figures would make a BETTER repairer look like a regression,
    and the mirror-image trap is just as real: a floor on unrepairable records
    would punish an implementation that repairs all of them. The only
    quantitative assertion is non-emptiness. Aggregate figures live in the
    fixture README as provenance, flagged as such.

    What this class actually buys, honestly stated: (b) and (c) are REGRESSION
    pins over a machine-generated expectation column, so they detect a CHANGE in
    behaviour, not an error in it. (d) and (e) are non-circular by construction
    and are the ones that could catch the repairer being wrong.
    """

    def test_corpus_exists_and_is_non_empty(self, corpus):
        assert len(corpus) > 0

    def test_every_record_carries_the_full_schema(self, corpus):
        for record in corpus:
            assert set(record) == set(extract.RECORD_KEYS), record.get('tool_use_id')
            assert isinstance(record['value'], str)
            assert isinstance(record['supplied'], list)
            assert isinstance(record['schema_params'], list)
            assert isinstance(record['truncated'], bool)
            assert isinstance(record['original_length'], int)
            assert record['expected_outcome'] in (
                extract.OUTCOME_REPAIRED,
                extract.OUTCOME_UNREPAIRABLE,
            )

    def test_tool_use_ids_are_unique(self, corpus):
        ids = [record['tool_use_id'] for record in corpus]
        assert len(ids) == len(set(ids))

    def test_expectations_agree_with_replay(self, corpus):
        """(b) Each record's stored expectation is what repair() produces now."""
        disagreements = []
        for record in corpus:
            result = repair(
                record['value'],
                record['param'],
                record['schema_params'],
                record['supplied'],
            )
            outcome = (
                extract.OUTCOME_UNREPAIRABLE if result is None else extract.OUTCOME_REPAIRED
            )
            recovered = [] if result is None else sorted(result.recovered)
            if outcome != record['expected_outcome'] or recovered != record['expected_recovered']:
                disagreements.append(
                    (record['tool_use_id'], record['expected_outcome'], outcome)
                )
        assert not disagreements, (
            f'{len(disagreements)} record(s) disagree with the committed expectation. '
            'If the repairer legitimately improved, regenerate the corpus in the SAME '
            'commit (PRD G6 anticipates exactly this). First few: '
            f'{disagreements[:5]}'
        )

    def test_replay_is_deterministic(self, corpus):
        """(c) Two full replays serialize identically (boundary row B13)."""

        def replay() -> str:
            rows = []
            for record in corpus:
                result = repair(
                    record['value'],
                    record['param'],
                    record['schema_params'],
                    record['supplied'],
                )
                rows.append(
                    {
                        'id': record['tool_use_id'],
                        'outcome': None if result is None else 'repaired',
                        'clean': None if result is None else result.clean_value,
                        'recovered': None if result is None else sorted(result.recovered.items()),
                        'pattern': None if result is None else result.pattern,
                        'misclose': None if result is None else result.misclose,
                    }
                )
            return json.dumps(rows, sort_keys=True)

        assert replay() == replay()

    def test_d5_holds_for_every_repaired_record(self, corpus):
        """(d) NON-CIRCULAR: the same helper the hand-authored specimens use."""
        for record in corpus:
            result = repair(
                record['value'],
                record['param'],
                record['schema_params'],
                record['supplied'],
            )
            assert_repair_invariants(record['value'], result)

    def test_every_stored_value_still_satisfies_the_collection_predicate(self, corpus):
        """(e) NON-CIRCULAR re-derivation: no record drifted in via some other route.

        Truncation rewrites the stored value, so this also proves the truncation
        rule never produces something the corpus predicate would not have
        collected in the first place.
        """
        for record in corpus:
            assert extract.matches_collection_predicate(record['value']), record['tool_use_id']

    def test_committed_file_carries_no_literal_opening_angle_bracket(self):
        """(f) A future agent must be able to hand-edit a row safely (PRD G6)."""
        assert '\x3c' not in CORPUS_PATH.read_text(encoding='utf-8')
