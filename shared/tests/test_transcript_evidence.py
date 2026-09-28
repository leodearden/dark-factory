"""Tests for the transcript evidence detector and its session wrapper (task 3995).

``transcript_evidence`` summarises what a run did from its on-disk transcript
records: how many assistant turns it took, the StructuredOutput verdict the CLI
ACCEPTED (if any), and which other tools it called. The curator reads it after
a killed run whose stdout never arrived.

The CLI records an accepted StructuredOutput call as a ``structured_output``
attachment carrying the validated ``data``, written before the success
tool_result. A rejected call (schema mismatch, unparseable input, permission
denial) gets an ``is_error`` tool_result and no attachment, so the model's
tool_use ``input`` alone is an attempt, never a verdict.

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

_SCHEMA_MISMATCH = 'Output does not match required schema: /rewritten_task: must be object,null'
_PERMISSION_DENIAL = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected "
    '(eg. if it was a file edit, the new_string was NOT written to the file).'
)


def _assistant(*blocks: object, nested: bool = True) -> dict:
    if nested:
        return {'type': 'assistant', 'message': {'role': 'assistant', 'content': list(blocks)}}
    return {'type': 'assistant', 'content': list(blocks)}


def _tool_use(name: str, tool_input: object = None, *, tool_use_id: str | None = None) -> dict:
    return {
        'type': 'tool_use',
        'id': tool_use_id or f'tu-{name}',
        'name': name,
        'input': tool_input or {},
    }


def _thinking() -> dict:
    return {'type': 'thinking', 'thinking': 't'}


def _preamble() -> list[dict]:
    return [
        {'type': 'queue-operation', 'operation': 'enqueue'},
        {'type': 'attachment', 'attachment': {'type': 'skill_listing'}},
        {'type': 'user', 'message': {'role': 'user', 'content': 'prompt'}},
        {'type': 'last-prompt', 'lastPrompt': 'prompt'},
    ]


def _tool_result(
    text: str = 'r', *, tool_use_id: str | None = None, is_error: bool = False,
) -> dict:
    block: dict = {'type': 'tool_result', 'content': text}
    if tool_use_id is not None:
        block['tool_use_id'] = tool_use_id
    if is_error:
        block['is_error'] = True
    return {'type': 'user', 'message': {'role': 'user', 'content': [block]}}


def _attachment(attachment_type: str, data: object, tool_use_id: str) -> dict:
    return {'type': 'attachment', 'attachment': {
        'type': attachment_type, 'data': data, 'toolUseID': tool_use_id,
    }}


def _schema_call(payload: object, tool_use_id: str, *, nested: bool = True) -> dict:
    return _assistant(
        _tool_use('StructuredOutput', payload, tool_use_id=tool_use_id), nested=nested,
    )


def _accepted(payload: object, tool_use_id: str = 'tu-accepted', *, nested: bool = True) -> list[dict]:
    """The CLI's acceptance sequence: tool_use, ``structured_output`` attachment, success."""
    return [
        _schema_call(payload, tool_use_id, nested=nested),
        {'type': 'last-prompt', 'lastPrompt': 'prompt'},
        _attachment('structured_output', payload, tool_use_id),
        _tool_result('Structured output provided successfully', tool_use_id=tool_use_id),
    ]


def _rejected(payload: object, reason: str, tool_use_id: str = 'tu-rejected') -> list[dict]:
    """The CLI's rejection sequence: tool_use, then an ``is_error`` tool_result, no attachment."""
    return [
        _schema_call(payload, tool_use_id),
        _tool_result(reason, tool_use_id=tool_use_id, is_error=True),
    ]


def _evidence(turns: int, payload: dict | None, tools: tuple[str, ...]) -> TranscriptEvidence:
    return TranscriptEvidence(
        assistant_turns=turns, accepted_schema_payload=payload, other_tool_uses=tools,
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

    def test_accepted_structured_output_is_the_payload(self) -> None:
        """esc-curator-33: the verdict was accepted, then stdout never arrived."""
        records = _preamble() + [_assistant(_thinking())] + _accepted(dict(_DECISION))

        assert transcript_evidence(records) == _evidence(2, _DECISION, ())

    def test_last_accepted_structured_output_wins(self) -> None:
        records = _accepted({'action': 'create'}, 'tu-1') + _accepted({'action': 'drop'}, 'tu-2')

        assert transcript_evidence(records).accepted_schema_payload == {'action': 'drop'}

    def test_last_dict_data_wins_over_a_later_malformed_attachment(self) -> None:
        records = _accepted({'action': 'drop'}) + [
            _attachment('structured_output', 'garbled', 'tu-later'),
        ]

        assert transcript_evidence(records).accepted_schema_payload == {'action': 'drop'}

    @pytest.mark.parametrize('attachment', [
        {'type': 'structured_output', 'toolUseID': 'tu-1'},
        {'type': 'structured_output', 'data': None, 'toolUseID': 'tu-1'},
        {'type': 'structured_output', 'data': ['action', 'drop'], 'toolUseID': 'tu-1'},
        {'type': 'structured_output', 'data': '{"action": "drop"}', 'toolUseID': 'tu-1'},
    ], ids=['missing', 'none', 'list', 'string'])
    def test_acceptance_without_dict_data_is_no_payload(self, attachment: dict) -> None:
        records = [
            _schema_call({'action': 'drop'}, 'tu-1'),
            {'type': 'attachment', 'attachment': attachment},
        ]

        assert transcript_evidence(records) == _evidence(1, None, ())

    @pytest.mark.parametrize('reason', [_SCHEMA_MISMATCH, _PERMISSION_DENIAL],
                             ids=['schema-mismatch', 'permission-denial'])
    def test_rejected_structured_output_is_no_payload(self, reason: str) -> None:
        """The model's input was a dict, but the CLI rejected the call."""
        records = _preamble() + [_assistant(_thinking())] + _rejected(dict(_DECISION), reason)

        assert transcript_evidence(records) == _evidence(2, None, ())

    def test_attempted_but_unvalidated_structured_output_is_no_payload(self) -> None:
        """Killed before the CLI validated the call: an attempt is not a verdict."""
        records = [_schema_call(dict(_DECISION), 'tu-1')]

        assert transcript_evidence(records) == _evidence(1, None, ())

    def test_rejected_then_accepted_yields_the_accepted_payload(self) -> None:
        records = (
            _rejected({'action': 'bogus'}, _SCHEMA_MISMATCH, 'tu-1')
            + _accepted({'action': 'drop'}, 'tu-2')
        )

        assert transcript_evidence(records) == _evidence(2, {'action': 'drop'}, ())

    def test_acceptance_without_a_following_tool_result_is_the_payload(self) -> None:
        """Killed between the acceptance record and the tool_result write."""
        records = [
            _schema_call(dict(_DECISION), 'tu-1'),
            _attachment('structured_output', dict(_DECISION), 'tu-1'),
        ]

        assert transcript_evidence(records) == _evidence(1, _DECISION, ())

    @pytest.mark.parametrize('attachment_type', [
        'hook_success', 'budget_usd', 'skill_listing', 'deferred_tools_delta',
    ])
    def test_other_attachment_types_are_never_a_payload(self, attachment_type: str) -> None:
        records = [_attachment(attachment_type, dict(_DECISION), 'tu-1')]

        assert transcript_evidence(records) == _evidence(0, None, ())

    def test_structured_output_is_never_another_tool(self) -> None:
        records = (
            _rejected({'action': 'bogus'}, _PERMISSION_DENIAL, 'tu-1')
            + [_assistant(_tool_use('ToolSearch'))]
            + _accepted({'action': 'drop'}, 'tu-2')
        )

        assert transcript_evidence(records).other_tool_uses == ('ToolSearch',)

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
            {'type': 'attachment'},
            {'type': 'attachment', 'attachment': 'not-a-dict'},
            {'type': 'attachment', 'attachment': {'data': {'action': 'drop'}}},
        ]

        assert transcript_evidence(records) == _evidence(3, None, ('TaskGet',))

    @pytest.mark.parametrize('nested', [True, False], ids=['message.content', 'flat content'])
    def test_both_content_nestings_are_read(self, nested: bool) -> None:
        records = [_assistant(_tool_use('ToolSearch'), nested=nested)] + _accepted(
            {'action': 'drop'}, nested=nested,
        )

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
        complete = [json.dumps(record) for record in _accepted(dict(_DECISION))]
        (project_dir / 'sess-2.jsonl').write_text(
            '\n'.join(complete) + '\n' + '{"type": "assistant", "message": {"con',
        )

        assert transcript_evidence_for_session(tmp_path, 'sess-2') == _evidence(1, _DECISION, ())
