"""Tests for the toolcall-markup fixture corpus and its extractor.

PRD ``plans/toolcall-markup-containment-prd.md`` task alpha, boundary row B13.

TDD pair 4 (step 7/8): the extractor's pure functions, over SYNTHETIC gzipped
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

import gzip
import json
from pathlib import Path

import pytest

import toolcall_markup_corpus_extract as extract
from shared.toolcall_markup import INVOKE_CLOSER

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
    """Write *records* as a gzipped JSONL transcript at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def _archive_leaf(root: Path, task_id: str, uuid: str) -> Path:
    """A leaf at the REAL archive depth: root/<task_id>/<project-dir>/<uuid>.

    Mirrors the layout measured 2026-08-08 (prerequisite P3):
    ``<root>/2474/-home-leo-src-dark-factory--worktrees-2474/<uuid>.jsonl.gz``.
    """
    project_dir = '-home-leo-src-dark-factory--worktrees-' + task_id
    return root / task_id / project_dir / (uuid + '.jsonl.gz')


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

    A top-level ``*.jsonl.gz`` glob returns ZERO files against the real archive,
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
                )
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
            [_tool_use('toolu_01', 'submit_task', {'title': 't', 'description': value})],
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
        with gzip.open(leaf, 'wt', encoding='utf-8') as handle:
            handle.write('not json at all ' + INVOKE_CLOSER + '\n')
            handle.write('\n')
            handle.write('[1, 2, 3]\n')
            handle.write(json.dumps({'type': 'assistant', 'message': 'not a dict'}) + '\n')
            handle.write(json.dumps({'type': 'user', 'message': {'content': 'hi'}}) + '\n')
            handle.write(good + '\n')

        records = extract.extract(root)

        assert [r['tool_use_id'] for r in records] == ['toolu_01']
