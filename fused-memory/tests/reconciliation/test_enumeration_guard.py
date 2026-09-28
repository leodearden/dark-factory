"""Tests for ``reconciliation.enumeration_guard.enumeration_liveness_gate``.

The gate is the write-time filter a recon-stage ``submit_task`` /
``update_task`` payload passes through: enumerated memory ids are
shape-checked, re-resolved against the live store, and the enumeration is
stamped as verified. Every test drives the ONE public entry point with an
explicit ``AsyncMock`` memory service and a fixed clock.

The fixture ids are the real ones from reify gates 5533 and 5537, the two
incidents the gate exists to prevent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.enumeration_guard import (
    X_LIVENESS_EXCEPTIONS_KEY,
    EnumerationGateResult,
    enumeration_liveness_gate,
)
from fused_memory.reconciliation.predicate_contradiction import (
    PROVENANCE_METADATA_KEY,
)
from fused_memory.utils.validation import validate_full_uuid

FIXED_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RECON_AGENT = 'recon-stage-task_knowledge_sync'

# Gate 5537's enumeration at filing time: three live entries plus 0d542614,
# which had been deleted ~11 minutes before the gate shipped.
LIVE_B3F = 'b3f55b47-0879-49e0-a26d-53bf6ea9689a'
LIVE_21F = '21f78de1-a5f3-4259-b907-16c8f2328e2f'
LIVE_23E = '23eff0cd-8515-40fb-9b3e-9a5be19cbc0c'
DELETED_0D5 = '0d542614-874b-4003-b136-76dd9d8b40f8'

# Gate 5533's mem0_legacy_repeat_entries, verbatim: ten bare 8-char fragments
# with no full id beside them.
GATE_5533_FRAGMENTS = [
    '1d54d1d7', '72ad5fd0', 'ae208971', '6e67e041', 'a859a052',
    '46f01444', '73c387d3', 'dde9cd6f', '72c1b19d', 'c764bcdc',
]

# Gate 5533's scalar pointers — a Mem0 target, a Stage-1 finding and a run.
GATE_5533_SCALARS: dict[str, Any] = {
    'mem0_primary_target': 'd1e1b490-fe26-4a4e-8759-d07b5eb0a106',
    'mem0_amended_canonical': 'f71c92ff-08d5-4264-9a29-f3b1bfcaca61',
    'stage1_finding_id': '60ebe901-0821-48ec-a4c8-545c7b97cde9',
    '_causation_id': '7e7ac617-73e2-4d7d-9776-b6693a236756',
    'execution_class': 'operational',
    'operational_mode': 'gate',
}


def memory_service(
    *,
    live: tuple[str, ...] = (),
    raising: dict[str, Exception] | None = None,
) -> AsyncMock:
    """A memory service whose point read resolves *live* ids, raises for the
    ids in *raising*, and returns None (a genuine miss) for everything else."""
    raising = raising or {}

    async def _get(_project_id: str, memory_id: str) -> dict[str, Any] | None:
        if memory_id in raising:
            raise raising[memory_id]
        if memory_id in live:
            return {'id': memory_id, 'content': 'x', 'metadata': {}}
        return None

    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(side_effect=_get)
    return service


@pytest.fixture
def project_root(tmp_path) -> str:
    root = tmp_path / 'reify'
    root.mkdir()
    return str(root)


async def run_gate(
    project_root: str,
    metadata: Any,
    *,
    service: AsyncMock,
    description: str | None = None,
    agent_id: str | None = RECON_AGENT,
    is_filing: bool = False,
) -> EnumerationGateResult:
    return await enumeration_liveness_gate(
        agent_id=agent_id,
        description=description,
        metadata=metadata,
        memory_service=service,
        project_root=project_root,
        is_filing=is_filing,
        now=lambda: FIXED_NOW,
    )


def _as_dict(metadata: Any) -> Any:
    return json.loads(metadata) if isinstance(metadata, str) else metadata


# --------------------------------------------------------------------------- #
# Pass-through contract
# --------------------------------------------------------------------------- #


class TestPassThrough:
    """Anything the gate does not apply to comes back as the caller's own
    objects, with no lookup performed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('agent_id', [None, 'claude-interactive', 'orchestrator'])
    @pytest.mark.parametrize('is_filing', [True, False])
    async def test_non_recon_callers_are_never_touched(
        self, project_root, agent_id, is_filing,
    ):
        metadata = {
            'mem0_legacy_repeat_entries': ['1d54d1d7', '72ad5fd0'],
            'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5],
            **GATE_5533_SCALARS,
        }
        description = f'- {DELETED_0D5} likeliest deletion target'
        service = memory_service(live=(LIVE_B3F,))

        result = await run_gate(
            project_root, metadata, service=service, description=description,
            agent_id=agent_id, is_filing=is_filing,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.description is description
        assert result.stats is None
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata', [None, '', '["not", "a", "dict"]', '42', '{not json'],
    )
    async def test_non_dict_recon_metadata_passes_through_verbatim(
        self, project_root, metadata,
    ):
        description = 'a description'
        service = memory_service()

        result = await run_gate(
            project_root, metadata, service=service, description=description,
            is_filing=True,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.description is description
        assert result.stats is None
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('is_filing', [True, False])
    async def test_scalar_uuids_are_pointers_not_enumerations(
        self, project_root, is_filing,
    ):
        metadata = dict(GATE_5533_SCALARS)
        description = f'Stage 1 finding {metadata["stage1_finding_id"]}'
        service = memory_service(live=(metadata['mem0_primary_target'],))

        result = await run_gate(
            project_root, metadata, service=service, description=description,
            is_filing=is_filing,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.description is description
        assert result.stats is None
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_json_string_payload_is_processed_like_the_equivalent_dict(
        self, project_root,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
            'x_enumeration_scope': {
                'stores': ['mem0'],
                'categories': ['observations_and_summaries'],
                'exhaustive': False,
            },
            **GATE_5533_SCALARS,
        }
        description = f'- {DELETED_0D5} likeliest deletion target'
        live = (LIVE_B3F, LIVE_21F, LIVE_23E)
        dict_service = memory_service(live=live)
        string_service = memory_service(live=live)

        from_dict = await run_gate(
            project_root, metadata, service=dict_service, description=description,
        )
        from_string = await run_gate(
            project_root, json.dumps(metadata), service=string_service,
            description=description,
        )

        assert from_string.rejection == from_dict.rejection
        assert from_string.description == from_dict.description
        assert from_string.stats == from_dict.stats
        assert _as_dict(from_string.metadata) == _as_dict(from_dict.metadata)
        assert (
            string_service.get_memory_by_id.await_args_list
            == dict_service.get_memory_by_id.await_args_list
        )


# --------------------------------------------------------------------------- #
# CHANGE 1(a): the fragment shape check
# --------------------------------------------------------------------------- #

FULL_UUID_HINT = validate_full_uuid('21f78de1')['hint']

# (payload, [(path, fragment), ...]) — each a list-element fragment the shape
# check must name.
TRUNCATED_PAYLOADS = {
    'gate-5537': (
        {'mem0_falsified_entries': [LIVE_B3F, LIVE_23E, DELETED_0D5, '21f78de1']},
        [('mem0_falsified_entries[3]', '21f78de1')],
    ),
    'gate-5533': (
        {'mem0_legacy_repeat_entries': list(GATE_5533_FRAGMENTS)},
        [
            (f'mem0_legacy_repeat_entries[{i}]', fragment)
            for i, fragment in enumerate(GATE_5533_FRAGMENTS)
        ],
    ),
    'dashed-partial': (
        {'mem0_falsified_entries': [LIVE_B3F, '21f78de1-a5f3']},
        [('mem0_falsified_entries[1]', '21f78de1-a5f3')],
    ),
    'consolidation-gate-provenance': (
        {GATE_METADATA_KEY: {'topic': 't', 'provenance': {
            'observed_members': [LIVE_B3F, '72ad5fd0'], 'authoritative': False,
        }}},
        [(f'{GATE_METADATA_KEY}.provenance.observed_members[1]', '72ad5fd0')],
    ),
    'predicate-provenance': (
        {PROVENANCE_METADATA_KEY: {'cited_memory_ids': ['ae208971']}},
        [(f'{PROVENANCE_METADATA_KEY}.cited_memory_ids[0]', 'ae208971')],
    ),
    'list-of-lists': (
        {'x_pairs': [[LIVE_B3F, 'dde9cd6f']]},
        [('x_pairs[0][1]', 'dde9cd6f')],
    ),
    'uppercase': (
        {'mem0_falsified_entries': [LIVE_B3F, '21F78DE1']},
        [('mem0_falsified_entries[1]', '21F78DE1')],
    ),
}

# Shapes measured in the live stores that must never be read as fragments.
NOT_FRAGMENTS = {
    'ten-char-commit-list': {'landing_commits': ['fa99bddc35']},
    'forty-char-sha-list': {
        'suspect_ranges': ['509e950e4d6849814df6176387685a4cd2467335'],
    },
    'seven-hex-element': {'x_short': ['abcdef1']},
    'eight-hex-scalar': {'item4_tool_commit': '7074ed0e'},
    'fragment-in-dict-key': {'x_resolved_reference_a6f4a594': 'CLOSED 2026-07-25'},
    'prose-list-element': {'x_notes': ['see 21f78de1 for context']},
}


class TestTruncatedFragmentShape:
    """A UUID-prefix fragment in a metadata list is rejected when filing and
    recorded — never silently kept — when updating."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', sorted(TRUNCATED_PAYLOADS))
    async def test_filing_rejects_every_fragment_with_zero_lookups(
        self, project_root, case,
    ):
        metadata, offenders = TRUNCATED_PAYLOADS[case]
        service = memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E))

        result = await run_gate(
            project_root, metadata, service=service, is_filing=True,
        )

        rejection = result.rejection
        assert rejection is not None
        assert rejection['error_type'] == 'EnumeratedIdTruncated'
        assert rejection['offenders'] == [
            {'path': path, 'value': value} for path, value in offenders
        ]
        assert rejection['hint'] == FULL_UUID_HINT
        for path, value in offenders:
            assert validate_full_uuid(value, field_name=path)['error'] in (
                rejection['error']
            )
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', sorted(TRUNCATED_PAYLOADS))
    async def test_updating_keeps_the_write_and_records_each_fragment(
        self, project_root, case,
    ):
        metadata, offenders = TRUNCATED_PAYLOADS[case]
        original = json.loads(json.dumps(metadata))

        result = await run_gate(
            project_root, metadata, service=memory_service(), is_filing=False,
        )

        assert result.rejection is None
        rewritten = dict(result.metadata)
        assert rewritten.pop(X_LIVENESS_EXCEPTIONS_KEY) == [
            {'memory_id': value, 'path': path, 'outcome': 'truncated'}
            for path, value in offenders
        ]
        assert rewritten == original
        assert metadata == original
        assert result.stats is not None
        assert result.stats['truncated'] == len(offenders)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', sorted(NOT_FRAGMENTS))
    @pytest.mark.parametrize('is_filing', [True, False])
    async def test_measured_non_fragment_shapes_are_never_flagged(
        self, project_root, case, is_filing,
    ):
        metadata = dict(NOT_FRAGMENTS[case])
        service = memory_service()

        result = await run_gate(
            project_root, metadata, service=service, is_filing=is_filing,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.stats is None
