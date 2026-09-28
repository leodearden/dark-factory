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

from fused_memory.reconciliation.enumeration_guard import (
    EnumerationGateResult,
    enumeration_liveness_gate,
)

FIXED_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RECON_AGENT = 'recon-stage-task_knowledge_sync'

# Gate 5537's enumeration at filing time: three live entries plus 0d542614,
# which had been deleted ~11 minutes before the gate shipped.
LIVE_B3F = 'b3f55b47-0879-49e0-a26d-53bf6ea9689a'
LIVE_21F = '21f78de1-a5f3-4259-b907-16c8f2328e2f'
LIVE_23E = '23eff0cd-8515-40fb-9b3e-9a5be19cbc0c'
DELETED_0D5 = '0d542614-874b-4003-b136-76dd9d8b40f8'

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
