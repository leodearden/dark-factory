"""Tests for the transcript evidence detector and its session wrapper (task 3995).

``transcript_evidence`` summarises what a run did from its on-disk transcript
records: how many assistant turns it took, the StructuredOutput payload it
produced (if any), and which other tools it called. The curator reads it after
a killed run whose stdout never arrived.

Records are built in memory in the shape of the redacted curator transcripts
under ``fused-memory/tests/fixtures/curator_transcripts/``; that package's
fixtures are not read from here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from test_liveness_boundary_gate import _write_transcript

from shared.cli_invoke import (
    TranscriptEvidence,
    transcript_evidence,
    transcript_evidence_for_session,
)

_DECISION = {
    'action': 'drop',
    'justification': 'j',
    'target_id': '9001',
    'target_fingerprint': None,
    'rewritten_task': None,
}


def _assistant(*blocks: object, nested: bool = True) -> dict:
    if nested:
        return {'type': 'assistant', 'message': {'role': 'assistant', 'content': list(blocks)}}
    return {'type': 'assistant', 'content': list(blocks)}


def _tool_use(name: str, tool_input: object = None) -> dict:
    return {'type': 'tool_use', 'id': f'tu-{name}', 'name': name, 'input': tool_input or {}}


def _thinking() -> dict:
    return {'type': 'thinking', 'thinking': 't'}


def _preamble() -> list[dict]:
    return [
        {'type': 'queue-operation', 'operation': 'enqueue'},
        {'type': 'attachment', 'attachment': {'type': 'skill_listing'}},
        {'type': 'user', 'message': {'role': 'user', 'content': 'prompt'}},
        {'type': 'last-prompt', 'lastPrompt': 'prompt'},
    ]


def _tool_result() -> dict:
    return {'type': 'user', 'message': {
        'role': 'user', 'content': [{'type': 'tool_result', 'content': 'r'}],
    }}


def _evidence(turns: int, payload: dict | None, tools: tuple[str, ...]) -> TranscriptEvidence:
    return TranscriptEvidence(
        assistant_turns=turns, schema_payload=payload, other_tool_uses=tools,
    )


class TestTranscriptEvidence:
    def test_tool_wandering_run_lists_every_other_tool_in_order(self) -> None:
        """esc-curator-2: three tool calls, no StructuredOutput."""
        records = _preamble()
        for name in ('ToolSearch', 'TaskGet', 'ToolSearch'):
            records += [_assistant(_thinking()), _assistant(_tool_use(name)), _tool_result()]

        assert transcript_evidence(records) == _evidence(
            6, None, ('ToolSearch', 'TaskGet', 'ToolSearch'),
        )

    def test_completed_structured_output_is_the_payload(self) -> None:
        """esc-curator-33: the verdict was produced, then stdout never arrived."""
        records = _preamble() + [
            _assistant(_thinking()),
            _assistant(_tool_use('StructuredOutput', dict(_DECISION))),
            _tool_result(),
        ]

        assert transcript_evidence(records) == _evidence(2, _DECISION, ())

    def test_last_structured_output_wins(self) -> None:
        records = [
            _assistant(_tool_use('StructuredOutput', {'action': 'create'})),
            _assistant(_tool_use('StructuredOutput', {'action': 'drop'})),
        ]

        assert transcript_evidence(records).schema_payload == {'action': 'drop'}

    def test_last_dict_input_wins_over_a_later_malformed_one(self) -> None:
        records = [
            _assistant(_tool_use('StructuredOutput', {'action': 'drop'})),
            _assistant({'type': 'tool_use', 'name': 'StructuredOutput', 'input': 'garbled'}),
        ]

        assert transcript_evidence(records).schema_payload == {'action': 'drop'}

    @pytest.mark.parametrize('block', [
        {'type': 'tool_use', 'name': 'StructuredOutput'},
        {'type': 'tool_use', 'name': 'StructuredOutput', 'input': None},
        {'type': 'tool_use', 'name': 'StructuredOutput', 'input': ['action', 'drop']},
        {'type': 'tool_use', 'name': 'StructuredOutput', 'input': '{"action": "drop"}'},
    ], ids=['missing', 'none', 'list', 'string'])
    def test_structured_output_without_a_dict_input_is_no_payload(self, block: dict) -> None:
        assert transcript_evidence([_assistant(block)]) == _evidence(1, None, ())

    def test_pre_turn_stall_is_empty_evidence(self) -> None:
        """esc-curator-4: the run never produced an assistant record."""
        assert transcript_evidence(_preamble()) == _evidence(0, None, ())

    def test_no_records_is_empty_evidence(self) -> None:
        assert transcript_evidence([]) == _evidence(0, None, ())

    def test_malformed_records_and_blocks_are_skipped(self) -> None:
        records = [
            'not-a-record',
            None,
            {'type': 'assistant', 'message': {'content': 'plain text'}},
            {'type': 'assistant', 'message': 'not-a-dict'},
            _assistant('not-a-block', {'name': 'NoType'}, {'type': 'tool_use'},
                       {'type': 'tool_use', 'name': 7}, _tool_use('TaskGet')),
        ]

        assert transcript_evidence(records) == _evidence(3, None, ('TaskGet',))

    @pytest.mark.parametrize('nested', [True, False], ids=['message.content', 'flat content'])
    def test_both_content_nestings_are_read(self, nested: bool) -> None:
        records = [
            _assistant(_tool_use('ToolSearch'), nested=nested),
            _assistant(_tool_use('StructuredOutput', {'action': 'drop'}), nested=nested),
        ]

        assert transcript_evidence(records) == _evidence(2, {'action': 'drop'}, ('ToolSearch',))

    def test_tool_use_blocks_outside_assistant_records_are_ignored(self) -> None:
        tool_blocks = [_tool_use('ToolSearch'), _tool_use('StructuredOutput', {'action': 'drop'})]
        records = [
            {'type': 'attachment', 'content': tool_blocks},
            {'type': 'user', 'message': {'role': 'user', 'content': tool_blocks}},
        ]

        assert transcript_evidence(records) == _evidence(0, None, ())

    def test_evidence_is_immutable(self) -> None:
        evidence = transcript_evidence([])

        with pytest.raises(dataclasses.FrozenInstanceError):
            evidence.assistant_turns = 1  # type: ignore[misc]


class TestTranscriptEvidenceForSession:
    def test_absent_transcript_is_none(self, tmp_path: Path) -> None:
        """Absence of a transcript is never reported as empty evidence."""
        assert transcript_evidence_for_session(tmp_path, 'no-such-session') is None

    def test_reads_the_real_on_disk_layout(self, tmp_path: Path) -> None:
        _write_transcript(tmp_path, 'sess-1', n_assistant=3, last_tool='ToolSearch')

        assert transcript_evidence_for_session(tmp_path, 'sess-1') == _evidence(
            3, None, ('ToolSearch',),
        )

    def test_tolerates_a_truncated_final_line(self, tmp_path: Path) -> None:
        """A SIGKILLed CLI can leave its last JSONL line half-written."""
        project_dir = tmp_path / 'projects' / 'proj'
        project_dir.mkdir(parents=True)
        complete = _assistant(_tool_use('StructuredOutput', dict(_DECISION)))
        (project_dir / 'sess-2.jsonl').write_text(
            json.dumps(complete) + '\n' + '{"type": "assistant", "message": {"con',
        )

        assert transcript_evidence_for_session(tmp_path, 'sess-2') == _evidence(1, _DECISION, ())
