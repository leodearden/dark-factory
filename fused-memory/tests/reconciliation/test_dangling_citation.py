"""Tests for ``reconciliation.dangling_citation`` — the handoff record (task 5749).

Stage 3 emits one finding per confirmed-absent cross-run citation and Stage 1
consumes it through ``repair_memory_citation``. These tests pin the record's
construction invariant, its journal round-trip, and that the line Stage 1 reads
names the ids under the repair tool's own parameter names.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any

import pytest

from fused_memory.models.reconciliation import FINDING_SEVERITY_SET
from fused_memory.reconciliation.cli_stage_runner import FINDING_ITEM_SCHEMA
from fused_memory.reconciliation.dangling_citation import (
    DANGLING_CITATION_KEY,
    DANGLING_CITATION_LINE_LABEL,
    DANGLING_CROSS_RUN_CITATION_FLAG_TYPE,
    DanglingCitation,
    build_dangling_citation_finding,
    dangling_citation_from_finding,
    dangling_citation_identity,
    render_dangling_citation_line,
)
from fused_memory.server.recon_report import get_recon_report_tool_signatures

RUN_ID = '06a4466d-cdc0-49ac-8e99-e6723be39392'
DANGLING = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
FINDING_ID = '5d0f3f53-6f43-4c1e-9d55-1f4f7f2e7c11'

_LOGGER = 'fused_memory.reconciliation.dangling_citation'


def _record() -> DanglingCitation:
    return DanglingCitation(
        target_run_id=RUN_ID, finding_id=FINDING_ID, memory_id=DANGLING, store='mem0',
    )


def _record_fields(**overrides: object) -> dict[str, Any]:
    fields: dict[str, Any] = {
        'target_run_id': RUN_ID,
        'finding_id': FINDING_ID,
        'memory_id': DANGLING,
        'store': 'mem0',
    }
    fields.update(overrides)
    return fields


class TestConstructionInvariant:
    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('target_run_id', 'not-a-uuid'),
            ('target_run_id', DANGLING[:8]),
            ('memory_id', 'deadbeef'),
            ('memory_id', ''),
            ('finding_id', ''),
            ('finding_id', '   '),
            ('finding_id', 42),
            ('store', ''),
            ('store', '  '),
        ],
    )
    def test_invalid_field_raises_naming_field_and_value(
        self, field: str, value: object,
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            DanglingCitation(**_record_fields(**{field: value}))
        message = str(excinfo.value)
        assert field in message
        assert repr(value) in message

    def test_valid_record_is_frozen(self) -> None:
        record: Any = _record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.memory_id = RUN_ID


class TestBuildFinding:
    def test_finding_shape(self) -> None:
        finding = build_dangling_citation_finding(_record())
        category_enum = FINDING_ITEM_SCHEMA['properties']['category']['enum']

        assert finding['actionable'] is True
        assert finding['severity'] in FINDING_SEVERITY_SET
        assert finding['category'] in category_enum
        assert finding['flag_type'] == DANGLING_CROSS_RUN_CITATION_FLAG_TYPE
        assert isinstance(finding['description'], str) and finding['description'].strip()
        assert (
            isinstance(finding['suggested_action'], str)
            and finding['suggested_action'].strip()
        )
        assert finding[DANGLING_CITATION_KEY] == _record_fields()

    def test_dangling_id_is_never_a_backing_citation(self) -> None:
        finding = build_dangling_citation_finding(_record())
        assert 'cited_memories' not in finding

    def test_description_names_all_three_ids(self) -> None:
        description = build_dangling_citation_finding(_record())['description']
        for value in (RUN_ID, FINDING_ID, DANGLING):
            assert value in description

    def test_text_is_deterministic(self) -> None:
        assert build_dangling_citation_finding(_record()) == build_dangling_citation_finding(
            _record()
        )


class TestRoundTrip:
    def test_journal_persistence_round_trip(self) -> None:
        record = _record()
        persisted = json.loads(json.dumps(build_dangling_citation_finding(record)))
        assert dangling_citation_from_finding(persisted) == record


class TestParseNeverRaises:
    def test_finding_without_key_is_none(self) -> None:
        assert dangling_citation_from_finding({'description': 'x'}) is None

    @pytest.mark.parametrize('finding', [None, 'a string', 42, ['list']])
    def test_non_dict_finding_is_none(self, finding: object) -> None:
        assert dangling_citation_from_finding(finding) is None

    @pytest.mark.parametrize(
        'record',
        [
            {k: v for k, v in _record_fields().items() if k != 'finding_id'},
            _record_fields(memory_id='not-a-uuid'),
            'not a mapping',
        ],
    )
    def test_malformed_record_is_none_and_warns(
        self, record: object, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert dangling_citation_from_finding({DANGLING_CITATION_KEY: record}) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, 'a malformed record must be logged, not silently dropped'
        assert any(DANGLING_CITATION_KEY in r.getMessage() for r in warnings)


class TestIdentity:
    def test_identity_of_built_finding(self) -> None:
        finding = build_dangling_citation_finding(_record())
        assert dangling_citation_identity(finding) == [RUN_ID, FINDING_ID, DANGLING]

    def test_identity_without_record_is_empty(self) -> None:
        assert dangling_citation_identity({'description': 'x'}) == []
        assert dangling_citation_identity(
            {DANGLING_CITATION_KEY: _record_fields(memory_id='bad')}
        ) == []


class TestRenderLine:
    def test_no_record_renders_nothing(self) -> None:
        assert render_dangling_citation_line({'description': 'x'}) is None

    def test_line_names_ids(self) -> None:
        line = render_dangling_citation_line(build_dangling_citation_finding(_record()))
        assert line is not None
        assert '\n' not in line
        assert line.startswith(DANGLING_CITATION_LINE_LABEL)
        for fragment in (
            f'target_run_id={RUN_ID}',
            f'finding_id={FINDING_ID}',
            f'memory_id={DANGLING}',
            'store=mem0',
        ):
            assert fragment in line

    def test_line_keys_are_live_repair_tool_parameters(self) -> None:
        line = render_dangling_citation_line(build_dangling_citation_finding(_record()))
        assert line is not None
        keys = re.findall(r'(\w+)=', line)
        assert keys, 'the rendered line must name its ids as key=value pairs'
        parameters = get_recon_report_tool_signatures()['repair_memory_citation'].parameters
        assert set(keys) <= set(parameters), (
            f'rendered keys {sorted(set(keys) - set(parameters))} are not '
            f'repair_memory_citation parameters {sorted(parameters)}'
        )
