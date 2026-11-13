"""Tests for ``reconciliation.enumeration_guard``'s write-time gate.

The gate is the write-time filter a recon-stage ``submit_task`` /
``update_task`` payload passes through: enumerated memory ids are
shape-checked, re-resolved against the live store, and the enumeration is
stamped as verified. Every test drives one of the two public entry points,
``enumeration_gate_for_filing`` or ``enumeration_gate_for_update``, with an
explicit ``AsyncMock`` memory service and a fixed clock.

The fixture ids are the real ones from reify gates 5533 and 5537, the two
incidents the gate exists to prevent.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.enumeration_guard import (
    ENUMERATION_UPDATE_NOT_OVERWRITING,
    LIVENESS_LOOKUP_SKIPPED,
    MAX_CONCURRENT_LIVENESS_READS,
    MAX_CONSECUTIVE_LIVENESS_READ_ERRORS,
    MAX_ENUMERATED_IDS_PER_WRITE,
    X_ENUMERATION_SCOPE_KEY,
    X_LIVENESS_EXCEPTIONS_KEY,
    X_LIVENESS_REVERIFIED_AT_KEY,
    EnumeratedIdOutcome,
    EnumerationGateResult,
    enumeration_gate_for_filing,
    enumeration_gate_for_update,
    render_enumeration_guard_section,
)
from fused_memory.reconciliation.predicate_contradiction import (
    PROVENANCE_METADATA_KEY,
)
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.utils.validation import validate_full_uuid
from reconciliation.enumeration_fixtures import (
    DELETED_0D5,
    FINDING_IDS,
    LIVE_21F,
    LIVE_23E,
    LIVE_B3F,
    VALID_SCOPE,
    fake_memory_service,
)

FIXED_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RECON_AGENT = 'recon-stage-task_knowledge_sync'

# Graphiti edge uuids: canonical, but not Mem0 ids, so get_memory_by_id (a
# Mem0 point read) reports them as absent, like FINDING_IDS.
GRAPHITI_EDGE_IDS = [
    '4c1f0a57-6a8e-4d1b-9a53-2d7e1f0c9b11',
    '9e2b7d44-1f3c-4a6e-8b0d-5c7a3e9f2d22',
]

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
    metadata_mode: str | None = None,
    append: bool | None = None,
) -> EnumerationGateResult:
    """*is_filing* picks the entry point; only an update takes merge flags."""
    if is_filing:
        assert metadata_mode is None and append is None, 'a filing has no merge'
        return await enumeration_gate_for_filing(
            agent_id=agent_id,
            description=description,
            metadata=metadata,
            memory_service=service,
            project_root=project_root,
            now=lambda: FIXED_NOW,
        )
    return await enumeration_gate_for_update(
        agent_id=agent_id,
        description=description,
        metadata=metadata,
        memory_service=service,
        project_root=project_root,
        metadata_mode=metadata_mode,
        append=append,
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
        service = fake_memory_service(live=(LIVE_B3F,))

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
        service = fake_memory_service()

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
        service = fake_memory_service(live=(metadata['mem0_primary_target'],))

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
        dict_service = fake_memory_service(live=live)
        string_service = fake_memory_service(live=live)

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

def uuid_shape_error(value: str, **kwargs: str) -> dict[str, str]:
    error = validate_full_uuid(value, **kwargs)
    assert error is not None
    return error


FULL_UUID_HINT = uuid_shape_error('21f78de1')['hint']

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
    # Measured on dark-factory task 1141: an all-decimal fragment is still a
    # truncation, so an 8-digit list element is flagged (a named residual).
    'all-decimal': (
        {'memory_hints': {'memory_ids': ['12851553']}},
        [('memory_hints.memory_ids[0]', '12851553')],
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
        service = fake_memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E))

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
            assert uuid_shape_error(value, field_name=path)['error'] in (
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
            project_root, metadata, service=fake_memory_service(), is_filing=False,
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
        service = fake_memory_service()

        result = await run_gate(
            project_root, metadata, service=service, is_filing=is_filing,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.stats is None


# --------------------------------------------------------------------------- #
# CHANGE 1(b): liveness over top-level metadata lists
# --------------------------------------------------------------------------- #


def awaited_ids(service: AsyncMock) -> list[str]:
    return [c.args[1] for c in service.get_memory_by_id.await_args_list]


class TestLiveness:
    """A top-level list is a Mem0 enumeration iff a member resolves live;
    only then are its genuinely absent members dropped."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True])
    async def test_gate_5537_deleted_member_is_dropped_and_recorded(
        self, project_root, as_json,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
        }
        payload = json.dumps(metadata) if as_json else metadata

        result = await run_gate(
            project_root, payload,
            service=fake_memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E)),
        )

        assert result.rejection is None
        assert result.metadata['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E,
        ]
        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            {
                'memory_id': DELETED_0D5,
                'path': 'mem0_falsified_entries[3]',
                'outcome': 'absent',
            },
        ]
        assert result.stats is not None
        assert {k: result.stats[k] for k in (
            'mem0_enumerations', 'verified', 'dropped_absent', 'inconclusive',
            'unproven_lists',
        )} == {
            'mem0_enumerations': 1, 'verified': 3, 'dropped_absent': 1,
            'inconclusive': 0, 'unproven_lists': 0,
        }

    @pytest.mark.asyncio
    async def test_an_inconclusive_member_is_kept_recorded_and_logged(
        self, project_root, caplog,
    ):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, LIVE_21F]}
        service = fake_memory_service(
            live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('qdrant timeout')},
        )

        with caplog.at_level(logging.WARNING):
            result = await run_gate(project_root, metadata, service=service)

        assert result.rejection is None
        assert result.metadata['mem0_falsified_entries'] == [LIVE_B3F, LIVE_21F]
        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            {
                'memory_id': LIVE_21F,
                'path': 'mem0_falsified_entries[1]',
                'outcome': 'inconclusive',
                'error_type': 'TimeoutError',
            },
        ]
        assert result.stats is not None
        assert result.stats['verified'] == 1
        assert result.stats['dropped_absent'] == 0
        assert result.stats['inconclusive'] == 1
        assert any(
            record.levelno == logging.WARNING and LIVE_21F in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_lists_that_never_resolve_live_are_left_untouched(
        self, project_root,
    ):
        metadata = {
            'origin_finding_ids': list(FINDING_IDS),
            'cited_edges': list(GRAPHITI_EDGE_IDS),
            'x_deleted_before_gate_open': [DELETED_0D5],
            'x_unreachable': [LIVE_B3F, LIVE_23E],
        }
        service = fake_memory_service(raising={
            LIVE_B3F: RuntimeError('backend down'),
            LIVE_23E: TimeoutError('qdrant timeout'),
        })

        result = await run_gate(project_root, metadata, service=service)

        assert result.rejection is None
        assert result.metadata is metadata
        assert X_LIVENESS_EXCEPTIONS_KEY not in metadata
        assert result.stats is not None
        assert result.stats['unproven_lists'] == 4
        assert result.stats['mem0_enumerations'] == 0

    @pytest.mark.asyncio
    async def test_unproven_lists_beside_a_proven_one_stay_untouched(
        self, project_root,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5],
            'origin_finding_ids': list(FINDING_IDS),
            'x_deleted_before_gate_open': [DELETED_0D5],
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
        )

        assert result.metadata['mem0_falsified_entries'] == [LIVE_B3F]
        assert result.metadata['origin_finding_ids'] == FINDING_IDS
        assert result.metadata['x_deleted_before_gate_open'] == [DELETED_0D5]
        assert [r['path'] for r in result.metadata[X_LIVENESS_EXCEPTIONS_KEY]] == [
            'mem0_falsified_entries[1]',
        ]
        assert result.stats is not None
        assert result.stats['unproven_lists'] == 2

    @pytest.mark.asyncio
    async def test_only_top_level_list_uuids_are_looked_up(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F],
            'mem0_primary_target': LIVE_21F,
            'x_notes': [f'see {LIVE_23E} for context'],
            GATE_METADATA_KEY: {'topic': 't', 'provenance': {
                'observed_members': [DELETED_0D5], 'authoritative': False,
            }},
        }
        service = fake_memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E))

        result = await run_gate(project_root, metadata, service=service)

        assert awaited_ids(service) == [LIVE_B3F]
        assert result.metadata[GATE_METADATA_KEY] == metadata[GATE_METADATA_KEY]

    @pytest.mark.asyncio
    async def test_each_distinct_id_costs_one_lookup_with_the_root_project_id(
        self, project_root,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5],
            'x_related_memories': [DELETED_0D5, LIVE_21F, LIVE_B3F],
        }
        service = fake_memory_service(live=(LIVE_B3F, LIVE_21F))

        result = await run_gate(project_root, metadata, service=service)

        assert sorted(awaited_ids(service)) == sorted([LIVE_B3F, DELETED_0D5, LIVE_21F])
        assert {c.args[0] for c in service.get_memory_by_id.await_args_list} == {
            'reify',
        }
        assert result.metadata['x_related_memories'] == [LIVE_21F, LIVE_B3F]
        assert [r['path'] for r in result.metadata[X_LIVENESS_EXCEPTIONS_KEY]] == [
            'mem0_falsified_entries[1]', 'x_related_memories[0]',
        ]

    @pytest.mark.asyncio
    async def test_distinct_ids_are_resolved_concurrently(self, project_root):
        in_flight = 0
        peak = 0

        async def _get(_project_id: str, memory_id: str) -> dict[str, Any] | None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return None if memory_id == DELETED_0D5 else {'id': memory_id}

        service = AsyncMock()
        service.get_memory_by_id = AsyncMock(side_effect=_get)
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
        }

        result = await run_gate(project_root, metadata, service=service)

        assert peak == 4
        assert result.metadata['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E,
        ]

    @pytest.mark.asyncio
    async def test_the_input_is_never_mutated(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, DELETED_0D5],
            X_LIVENESS_EXCEPTIONS_KEY: [{'memory_id': 'old', 'outcome': 'absent'}],
        }
        snapshot = copy.deepcopy(metadata)

        await run_gate(
            project_root, metadata,
            service=fake_memory_service(
                live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('slow')},
            ),
        )

        assert metadata == snapshot

    @pytest.mark.asyncio
    @pytest.mark.parametrize('is_filing', [True, False])
    async def test_a_backend_failing_every_lookup_never_raises(
        self, project_root, is_filing,
    ):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]}
        service = AsyncMock()
        service.get_memory_by_id = AsyncMock(side_effect=RuntimeError('down'))

        result = await run_gate(
            project_root, metadata, service=service, is_filing=is_filing,
        )

        assert result.rejection is None
        assert result.metadata is metadata


def generated_ids(count: int) -> list[str]:
    return [f'{index:08x}-0000-4000-8000-{index:012x}' for index in range(1, count + 1)]


def skipped_record(memory_id: str, path: str) -> dict[str, str]:
    return {
        'memory_id': memory_id,
        'path': path,
        'outcome': EnumeratedIdOutcome.INCONCLUSIVE.value,
        'error_type': LIVENESS_LOOKUP_SKIPPED,
    }


class TestBoundedLookups:
    """The gate runs on a write path, so its point reads are bounded: an id it
    does not read is kept and recorded inconclusive, never treated as absent."""

    @pytest.mark.asyncio
    async def test_ids_past_the_lookup_budget_are_kept_unread_as_inconclusive(
        self, project_root,
    ):
        members = generated_ids(MAX_ENUMERATED_IDS_PER_WRITE + 2)
        service = fake_memory_service(live=tuple(members))

        result = await run_gate(
            project_root, {'x_related_memories': members}, service=service,
        )

        assert awaited_ids(service) == members[:MAX_ENUMERATED_IDS_PER_WRITE]
        assert result.metadata['x_related_memories'] == members
        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            skipped_record(member, f'x_related_memories[{index}]')
            for index, member in enumerate(members)
            if index >= MAX_ENUMERATED_IDS_PER_WRITE
        ]

    @pytest.mark.asyncio
    async def test_no_more_than_the_read_limit_is_ever_in_flight(self, project_root):
        in_flight = 0
        peak = 0

        async def _get(_project_id: str, memory_id: str) -> dict[str, Any] | None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return {'id': memory_id}

        service = AsyncMock()
        service.get_memory_by_id = AsyncMock(side_effect=_get)
        members = generated_ids(MAX_CONCURRENT_LIVENESS_READS * 3)

        await run_gate(project_root, {'x_related_memories': members}, service=service)

        assert peak == MAX_CONCURRENT_LIVENESS_READS
        assert sorted(awaited_ids(service)) == sorted(members)

    @pytest.mark.asyncio
    async def test_reads_stop_once_the_store_keeps_failing(self, project_root):
        failing = generated_ids(20)
        service = fake_memory_service(
            live=(LIVE_B3F,),
            raising=dict.fromkeys(failing, TimeoutError('qdrant timeout')),
        )
        members = [LIVE_B3F, *failing]

        result = await run_gate(
            project_root, {'x_related_memories': members}, service=service,
        )

        read = awaited_ids(service)
        assert len(read) <= 1 + MAX_CONCURRENT_LIVENESS_READS
        assert result.metadata['x_related_memories'] == members
        records = result.metadata[X_LIVENESS_EXCEPTIONS_KEY]
        assert len(records) == len(failing)
        timed_out = [r for r in records if r['error_type'] == 'TimeoutError']
        assert len(timed_out) == len(read) - 1 >= MAX_CONSECUTIVE_LIVENESS_READ_ERRORS
        assert [r for r in records if r not in timed_out] == [
            skipped_record(r['memory_id'], r['path'])
            for r in records if r not in timed_out
        ]


# --------------------------------------------------------------------------- #
# CHANGE 1(c): description annotation
# --------------------------------------------------------------------------- #

VERIFIED_AT = FIXED_NOW.isoformat()

# Gate 5537's description as filed: the deleted row is nominated by full id and
# again by its 8-char prefix, beside commit prefixes of the identical shape.
GATE_5537_DESCRIPTION = '\n'.join([
    'THE FOUR ENTRIES:',
    f'- {LIVE_B3F} (observations_and_summaries)',
    f'- {DELETED_0D5} (procedural_knowledge)',
    'Delete 0d542614 first: it ranks into how-to queries.',
    'main moved 7074ed0e -> 509e950e on 2026-07-24.',
])


class TestDescriptionAnnotation:
    """Only ids the METADATA pass proved absent or inconclusive are annotated
    in the description; the description itself is never a detection source."""

    @pytest.mark.asyncio
    async def test_lines_naming_an_absent_id_are_annotated(self, project_root):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]}

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
            description=GATE_5537_DESCRIPTION,
        )

        assert result.description is not None
        before = GATE_5537_DESCRIPTION.split('\n')
        after = result.description.split('\n')
        assert len(after) == len(before)
        for index in (2, 3):
            assert after[index].startswith(before[index])
            assert after[index].count(f'already deleted, verified {VERIFIED_AT}') == 1
        for index in (0, 1, 4):
            assert after[index] == before[index]
        assert result.stats is not None
        assert result.stats['annotated_lines'] == 2

    @pytest.mark.asyncio
    async def test_an_inconclusive_id_is_never_called_deleted(self, project_root):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, LIVE_21F]}
        description = f'- {LIVE_21F} the fuller twin'

        result = await run_gate(
            project_root, metadata,
            service=fake_memory_service(
                live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('slow')},
            ),
            description=description,
        )

        assert result.description is not None
        assert result.description.startswith(description)
        assert 'liveness inconclusive (TimeoutError)' in result.description
        assert 'deleted' not in result.description

    @pytest.mark.asyncio
    async def test_annotating_twice_adds_no_second_annotation(self, project_root):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]}
        service = fake_memory_service(live=(LIVE_B3F,))

        first = await run_gate(
            project_root, metadata, service=service,
            description=GATE_5537_DESCRIPTION,
        )
        second = await run_gate(
            project_root, metadata, service=service, description=first.description,
        )

        assert second.description == first.description
        assert second.stats is not None
        assert second.stats['annotated_lines'] == 0

    @pytest.mark.asyncio
    async def test_a_later_absent_verdict_rewrites_an_inconclusive_annotation(
        self, project_root,
    ):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, LIVE_21F]}
        description = f'- {LIVE_21F} the fuller twin'
        first = await run_gate(
            project_root, metadata,
            service=fake_memory_service(
                live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('slow')},
            ),
            description=description,
        )

        second = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
            description=first.description,
        )

        assert second.description is not None
        assert second.description.startswith(description)
        assert second.description.count('[21f78de1: ') == 1
        assert f'already deleted, verified {VERIFIED_AT}' in second.description
        assert 'inconclusive' not in second.description
        assert second.stats is not None
        assert second.stats['dropped_absent'] == 1
        assert second.stats['annotated_lines'] == 1

    @pytest.mark.asyncio
    async def test_a_none_description_stays_none(self, project_root):
        result = await run_gate(
            project_root, {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]},
            service=fake_memory_service(live=(LIVE_B3F,)), description=None,
        )

        assert result.description is None
        assert result.metadata['mem0_falsified_entries'] == [LIVE_B3F]

    @pytest.mark.asyncio
    async def test_a_miss_in_an_unproven_list_never_annotates(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F],
            'origin_finding_ids': list(FINDING_IDS),
            'x_deleted_before_gate_open': [DELETED_0D5],
        }
        description = '\n'.join([
            f'Filed from Stage 1 finding {FINDING_IDS[0]}.',
            f'{DELETED_0D5} was deleted before this gate opened.',
        ])

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
            description=description,
        )

        assert result.description is description

    @pytest.mark.asyncio
    async def test_the_description_is_never_a_detection_source(self, project_root):
        service = fake_memory_service(live=(LIVE_B3F,))
        description = f'- {DELETED_0D5} likeliest deletion target'

        result = await run_gate(
            project_root, {'mem0_falsified_entries': [LIVE_B3F]}, service=service,
            description=description,
        )

        assert awaited_ids(service) == [LIVE_B3F]
        assert result.description is description

    @pytest.mark.asyncio
    async def test_a_prefix_twin_is_not_mistaken_for_the_absent_id(
        self, project_root,
    ):
        twin = '0d542614-9999-4aaa-8bbb-000000000001'
        description = f'- {twin} a different entry sharing the prefix'

        result = await run_gate(
            project_root, {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]},
            service=fake_memory_service(live=(LIVE_B3F,)), description=description,
        )

        assert result.description == description


# --------------------------------------------------------------------------- #
# CHANGE 1(c): the gate-owned stamp and exceptions keys
# --------------------------------------------------------------------------- #

# Gate 5537's hand-written value: prose, copied forward from a remediation run.
GATE_5537_STAMP = (
    '2026-07-25 (remediation run 97762388-504c-4603-a57d-dd3c9e3bae02) — all 5 '
    'enumerated ids (b3f55b47, 21f78de1-a5f3-4259-b907-16c8f2328e2f, 0d542614, '
    '23eff0cd, 9a4e568b) re-resolved via get_memory_by_id at update time'
)
STALE_EXCEPTIONS = [
    {'memory_id': DELETED_0D5, 'path': 'x_old[0]', 'outcome': 'absent'},
]

# name -> (enumeration, memory service) for each way a list can be re-resolved.
REVERIFIED = {
    'all-live': (
        [LIVE_B3F, LIVE_21F],
        lambda: fake_memory_service(live=(LIVE_B3F, LIVE_21F)),
    ),
    'one-absent': (
        [LIVE_B3F, DELETED_0D5],
        lambda: fake_memory_service(live=(LIVE_B3F,)),
    ),
    'one-inconclusive': (
        [LIVE_B3F, LIVE_21F],
        lambda: fake_memory_service(
            live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('slow')},
        ),
    ),
}


class TestGateOwnedKeys:
    """The stamp says THIS write re-resolved an enumeration; nothing else may
    set it, and a copied-forward value never survives the gate."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', sorted(REVERIFIED))
    async def test_a_reverified_enumeration_is_stamped_in_utc(
        self, project_root, case,
    ):
        members, make_service = REVERIFIED[case]

        result = await run_gate(
            project_root, {'mem0_falsified_entries': members},
            service=make_service(),
        )

        stamp = result.metadata[X_LIVENESS_REVERIFIED_AT_KEY]
        assert stamp == FIXED_NOW.isoformat()
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == FIXED_NOW.utcoffset()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'incoming', [GATE_5537_STAMP, '2026-07-25T00:00:00+00:00'],
        ids=['gate-5537-prose', 'older-iso'],
    )
    async def test_an_incoming_stamp_is_overwritten(self, project_root, incoming):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F],
            X_LIVENESS_REVERIFIED_AT_KEY: incoming,
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
        )

        assert result.metadata[X_LIVENESS_REVERIFIED_AT_KEY] == FIXED_NOW.isoformat()

    @pytest.mark.asyncio
    async def test_incoming_exceptions_are_replaced_never_merged(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5],
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
        )

        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            {
                'memory_id': DELETED_0D5,
                'path': 'mem0_falsified_entries[1]',
                'outcome': 'absent',
            },
        ]

    @pytest.mark.asyncio
    async def test_exceptions_are_absent_when_this_write_produced_none(
        self, project_root,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F],
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
        )

        assert X_LIVENESS_EXCEPTIONS_KEY not in result.metadata
        assert result.metadata[X_LIVENESS_REVERIFIED_AT_KEY] == FIXED_NOW.isoformat()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('lists', [
        {},
        {'origin_finding_ids': list(FINDING_IDS)},
        {'x_deleted_before_gate_open': [DELETED_0D5]},
    ], ids=['no-lists', 'finding-ids', 'deliberately-deleted'])
    async def test_without_a_proven_enumeration_both_keys_are_stripped(
        self, project_root, lists,
    ):
        metadata = {
            **GATE_5533_SCALARS,
            **lists,
            X_LIVENESS_REVERIFIED_AT_KEY: GATE_5537_STAMP,
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(project_root, metadata, service=fake_memory_service())

        assert result.rejection is None
        assert result.metadata == {**GATE_5533_SCALARS, **lists}

    @pytest.mark.asyncio
    async def test_a_fragment_only_update_is_recorded_but_not_stamped(
        self, project_root,
    ):
        metadata = {
            'mem0_legacy_repeat_entries': list(GATE_5533_FRAGMENTS),
            X_LIVENESS_REVERIFIED_AT_KEY: GATE_5537_STAMP,
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(), is_filing=False,
        )

        assert X_LIVENESS_REVERIFIED_AT_KEY not in result.metadata
        assert [r['outcome'] for r in result.metadata[X_LIVENESS_EXCEPTIONS_KEY]] == (
            ['truncated'] * len(GATE_5533_FRAGMENTS)
        )


class TestReplaceKeepsEarlierExceptions:
    """``replace`` swaps in the whole blob, so the gate carries an earlier
    write's exception records forward rather than erase the only trace of a
    drop; ``merge`` leaves the stored key alone unless this write records."""

    @pytest.mark.asyncio
    async def test_records_for_ids_this_write_did_not_observe_are_carried(
        self, project_root,
    ):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F],
            X_LIVENESS_REVERIFIED_AT_KEY: GATE_5537_STAMP,
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(
            project_root, metadata,
            service=fake_memory_service(live=(LIVE_B3F, LIVE_21F)),
            metadata_mode='replace',
        )

        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == STALE_EXCEPTIONS
        assert result.metadata[X_LIVENESS_REVERIFIED_AT_KEY] == FIXED_NOW.isoformat()

    @pytest.mark.asyncio
    async def test_this_writes_verdict_supersedes_a_record_for_the_same_id(
        self, project_root,
    ):
        earlier_inconclusive = {
            'memory_id': LIVE_21F.upper(),
            'path': 'mem0_falsified_entries[1]',
            'outcome': 'inconclusive',
            'error_type': 'TimeoutError',
        }
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, DELETED_0D5],
            X_LIVENESS_EXCEPTIONS_KEY: [*STALE_EXCEPTIONS, earlier_inconclusive],
        }

        result = await run_gate(
            project_root, metadata,
            service=fake_memory_service(live=(LIVE_B3F, LIVE_21F)),
            metadata_mode='replace',
        )

        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            {
                'memory_id': DELETED_0D5,
                'path': 'mem0_falsified_entries[2]',
                'outcome': 'absent',
            },
        ]

    @pytest.mark.asyncio
    async def test_an_unproven_replace_keeps_the_records_but_not_the_stamp(
        self, project_root,
    ):
        metadata = {
            'origin_finding_ids': list(FINDING_IDS),
            X_LIVENESS_REVERIFIED_AT_KEY: GATE_5537_STAMP,
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(),
            metadata_mode='replace',
        )

        assert result.metadata == {
            'origin_finding_ids': FINDING_IDS,
            X_LIVENESS_EXCEPTIONS_KEY: STALE_EXCEPTIONS,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize('incoming', [
        'dropped 0d542614 on 2026-07-25',
        [[DELETED_0D5]],
        [{'memory_id': DELETED_0D5}],
        [{'memory_id': DELETED_0D5, 'outcome': 'gone'}],
        [{'memory_id': DELETED_0D5, 'outcome': ['absent']}],
        [{'memory_id': 7, 'outcome': 'absent'}],
    ], ids=['prose', 'bare-list', 'no-outcome', 'unknown-outcome',
            'unhashable-outcome', 'non-string-id'])
    async def test_a_malformed_record_is_never_carried(self, project_root, incoming):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F],
            X_LIVENESS_EXCEPTIONS_KEY: incoming,
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(live=(LIVE_B3F,)),
            metadata_mode='replace',
        )

        assert X_LIVENESS_EXCEPTIONS_KEY not in result.metadata


# --------------------------------------------------------------------------- #
# CHANGE 2: the structured x_enumeration_scope
# --------------------------------------------------------------------------- #

# Gate 5533's real value: the legacy prose form.
GATE_5533_SCOPE = (
    'mem0_legacy_repeat_entries was enumerated over category=procedural_knowledge '
    'ONLY; e9d9326a (preferences_and_norms) is a known cross-category sibling '
    'teaching the same now-false rule, tracked separately in '
    'mem0_cross_category_repeat_entries'
)

# name -> (scope value, fields the rejection must name)
MALFORMED_SCOPES = {
    'legacy-prose-string': (GATE_5533_SCOPE, {X_ENUMERATION_SCOPE_KEY}),
    'missing-exhaustive': (
        {k: v for k, v in VALID_SCOPE.items() if k != 'exhaustive'}, {'exhaustive'},
    ),
    'string-exhaustive': ({**VALID_SCOPE, 'exhaustive': 'false'}, {'exhaustive'}),
    'empty-stores': ({**VALID_SCOPE, 'stores': []}, {'stores'}),
    'unknown-store': ({**VALID_SCOPE, 'stores': ['mem0', 'qdrant']}, {'stores'}),
    'empty-categories': ({**VALID_SCOPE, 'categories': []}, {'categories'}),
    'unknown-category': (
        {**VALID_SCOPE, 'categories': ['procedural_knowledge', 'how_tos']},
        {'categories'},
    ),
    'non-string-note': ({**VALID_SCOPE, 'note': 7}, {'note'}),
    'every-field-at-once': (
        {'stores': 'mem0', 'categories': [], 'note': ['x']},
        {'stores', 'categories', 'exhaustive', 'note'},
    ),
}


def gate_5537_filing(**overrides: Any) -> dict[str, Any]:
    """Gate 5537 as Stage 2 filed it: a four-entry enumeration, no scope."""
    return {
        'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
        'stage1_finding_id': 'f11018d4-be96-456b-9ab9-cf6814155500',
        'execution_class': 'operational',
        'operational_mode': 'gate',
        **overrides,
    }


def gate_5537_service() -> AsyncMock:
    return fake_memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E))


# Every declaration the operational routing turns into an escalating gate,
# plus the already-routed stamp itself.
GATE_DECLARATIONS = {
    'operational-gate': {'execution_class': 'operational', 'operational_mode': 'gate'},
    'operational-mode-absent': {'execution_class': 'operational'},
    'operational-llm': {'execution_class': 'operational', 'operational_mode': 'llm'},
    'decision': {'execution_class': 'decision'},
    'routed-pure-gate': {'task_kind': 'deterministic', 'always_escalates': True},
}


class TestEnumerationScope:
    """Filing a gate that enumerates Mem0 entries must say what was searched."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('declaration', sorted(GATE_DECLARATIONS))
    async def test_filing_an_enumerating_gate_without_a_scope_is_rejected(
        self, project_root, declaration,
    ):
        metadata = {
            key: value for key, value in gate_5537_filing().items()
            if key not in ('execution_class', 'operational_mode')
        } | GATE_DECLARATIONS[declaration]

        result = await run_gate(
            project_root, metadata, service=gate_5537_service(), is_filing=True,
        )

        rejection = result.rejection
        assert rejection is not None
        assert rejection['error_type'] == 'EnumerationScopeMissing'
        assert rejection['enumerations'] == ['mem0_falsified_entries']
        assert set(rejection['expected_shape']) == {
            'stores', 'categories', 'exhaustive', 'note',
        }
        assert 'mem0_falsified_entries' in rejection['error']

    @pytest.mark.asyncio
    async def test_the_same_gate_with_a_valid_scope_is_accepted(self, project_root):
        metadata = gate_5537_filing(**{X_ENUMERATION_SCOPE_KEY: VALID_SCOPE})

        result = await run_gate(
            project_root, metadata, service=gate_5537_service(), is_filing=True,
        )

        assert result.rejection is None
        assert result.metadata[X_ENUMERATION_SCOPE_KEY] == VALID_SCOPE
        assert result.metadata['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E,
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', sorted(MALFORMED_SCOPES))
    @pytest.mark.parametrize('is_filing', [True, False])
    @pytest.mark.parametrize('enumerating', [True, False])
    async def test_a_present_scope_must_be_well_formed(
        self, project_root, case, is_filing, enumerating,
    ):
        scope, fields = MALFORMED_SCOPES[case]
        metadata = gate_5537_filing(**{X_ENUMERATION_SCOPE_KEY: scope})
        if not enumerating:
            del metadata['mem0_falsified_entries']
        service = gate_5537_service()

        result = await run_gate(
            project_root, metadata, service=service, is_filing=is_filing,
        )

        rejection = result.rejection
        assert rejection is not None
        assert rejection['error_type'] == 'EnumerationScopeMalformed'
        assert {problem['field'] for problem in rejection['problems']} == fields
        assert set(rejection['expected_shape']) == {
            'stores', 'categories', 'exhaustive', 'note',
        }
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('metadata', [
        pytest.param(gate_5537_filing(execution_class='code_tdd'), id='non-gate'),
        pytest.param(
            {
                'origin_finding_ids': list(FINDING_IDS),
                'x_deleted_before_gate_open': [DELETED_0D5],
                'execution_class': 'operational',
            },
            id='gate-with-only-unproven-lists',
        ),
    ])
    async def test_scope_is_not_required_outside_an_enumerating_gate_filing(
        self, project_root, metadata,
    ):
        result = await run_gate(
            project_root, metadata, service=gate_5537_service(), is_filing=True,
        )

        assert result.rejection is None

    @pytest.mark.asyncio
    async def test_scope_is_not_required_on_update(self, project_root):
        result = await run_gate(
            project_root, gate_5537_filing(), service=gate_5537_service(),
            is_filing=False,
        )

        assert result.rejection is None
        assert result.metadata['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E,
        ]

    @pytest.mark.asyncio
    async def test_a_truncated_fragment_is_rejected_first_with_zero_lookups(
        self, project_root,
    ):
        metadata = gate_5537_filing(
            mem0_falsified_entries=[LIVE_B3F, LIVE_23E, DELETED_0D5, '21f78de1'],
        )
        service = gate_5537_service()

        result = await run_gate(
            project_root, metadata, service=service, is_filing=True,
        )

        assert result.rejection is not None
        assert result.rejection['error_type'] == 'EnumeratedIdTruncated'
        service.get_memory_by_id.assert_not_awaited()


# --------------------------------------------------------------------------- #
# The update path: the gate's rewrite must be able to land
# --------------------------------------------------------------------------- #

GATE_5537_ENUMERATION = {
    'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
}

# (metadata_mode, append) pairs under which a stored top-level key survives the
# incoming one: the additive union, and the combinations the backend refuses.
NOT_OVERWRITING = [
    ('additive', None),
    ('additive', True),
    ('additive', False),
    (None, True),
    (None, False),
    ('bogus', None),
]

# Pairs under which every incoming top-level key replaces the stored one.
OVERWRITING = [
    (None, None),
    ('merge', None),
    ('replace', None),
    ('replace', True),
    ('replace', False),
]

STAGE2_MEMORY_HINTS_ATTACH = {
    'memory_hints': {'entities': ['E1'], 'queries': ['Q1']},
}


class TestUpdateMergeMode:
    """An update the backend would merge additively keeps every stored member
    and the stored stamp, so the gate refuses to report a drop or a fresh
    verification that would never persist."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('metadata_mode', 'append'), NOT_OVERWRITING)
    async def test_an_enumeration_under_a_non_overwriting_merge_is_rejected(
        self, project_root, metadata_mode, append,
    ):
        metadata = {**GATE_5537_ENUMERATION, 'origin_finding_ids': list(FINDING_IDS)}

        result = await run_gate(
            project_root, metadata, service=gate_5537_service(),
            metadata_mode=metadata_mode, append=append,
        )

        rejection = result.rejection
        assert rejection is not None
        assert rejection['error_type'] == 'EnumerationUpdateNotOverwriting'
        assert rejection['enumerations'] == ['mem0_falsified_entries']
        assert 'mem0_falsified_entries' in rejection['error']
        assert 'origin_finding_ids' not in rejection['error']
        assert rejection['metadata_mode'] == metadata_mode
        assert rejection['append'] == append
        assert rejection['accepted_modes'] == ['merge', 'replace']
        assert result.stats is None

    @pytest.mark.asyncio
    async def test_an_all_live_enumeration_under_append_is_rejected_too(
        self, project_root,
    ):
        result = await run_gate(
            project_root, {'mem0_falsified_entries': [LIVE_B3F, LIVE_21F]},
            service=fake_memory_service(live=(LIVE_B3F, LIVE_21F)), append=True,
        )

        assert result.rejection is not None
        assert result.rejection['error_type'] == 'EnumerationUpdateNotOverwriting'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('metadata_mode', 'append'), OVERWRITING)
    async def test_an_overwriting_merge_is_rewritten_as_before(
        self, project_root, metadata_mode, append,
    ):
        result = await run_gate(
            project_root, dict(GATE_5537_ENUMERATION), service=gate_5537_service(),
            metadata_mode=metadata_mode, append=append,
        )

        assert result.rejection is None
        assert result.metadata['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E,
        ]
        assert result.metadata[X_LIVENESS_REVERIFIED_AT_KEY] == FIXED_NOW.isoformat()
        assert result.metadata[X_LIVENESS_EXCEPTIONS_KEY] == [
            {
                'memory_id': DELETED_0D5,
                'path': 'mem0_falsified_entries[3]',
                'outcome': 'absent',
            },
        ]

    @pytest.mark.asyncio
    async def test_an_append_to_unproven_lists_is_accepted_unchanged(
        self, project_root,
    ):
        metadata = {
            'origin_finding_ids': list(FINDING_IDS),
            'cited_edges': list(GRAPHITI_EDGE_IDS),
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(), append=True,
        )

        assert result.rejection is None
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_a_memory_hints_attach_is_accepted_with_zero_lookups(
        self, project_root,
    ):
        metadata = copy.deepcopy(STAGE2_MEMORY_HINTS_ATTACH)
        service = fake_memory_service()

        result = await run_gate(project_root, metadata, service=service, append=True)

        assert result.rejection is None
        assert result.metadata == STAGE2_MEMORY_HINTS_ATTACH
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_fragment_only_append_is_recorded_but_not_stamped(
        self, project_root,
    ):
        metadata = {'mem0_legacy_repeat_entries': list(GATE_5533_FRAGMENTS)}

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(), append=True,
        )

        assert result.rejection is None
        assert X_LIVENESS_REVERIFIED_AT_KEY not in result.metadata
        assert [r['outcome'] for r in result.metadata[X_LIVENESS_EXCEPTIONS_KEY]] == (
            ['truncated'] * len(GATE_5533_FRAGMENTS)
        )

    @pytest.mark.asyncio
    async def test_an_unproven_append_still_strips_an_incoming_stamp(
        self, project_root,
    ):
        metadata = {
            'origin_finding_ids': list(FINDING_IDS),
            X_LIVENESS_REVERIFIED_AT_KEY: GATE_5537_STAMP,
        }

        result = await run_gate(
            project_root, metadata, service=fake_memory_service(), append=True,
        )

        assert result.rejection is None
        assert result.metadata == {'origin_finding_ids': FINDING_IDS}

    @pytest.mark.asyncio
    async def test_a_malformed_scope_is_rejected_first_with_zero_lookups(
        self, project_root,
    ):
        metadata = {**GATE_5537_ENUMERATION, X_ENUMERATION_SCOPE_KEY: GATE_5533_SCOPE}
        service = gate_5537_service()

        result = await run_gate(project_root, metadata, service=service, append=True)

        assert result.rejection is not None
        assert result.rejection['error_type'] == 'EnumerationScopeMalformed'
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_recon_append_is_never_touched(self, project_root):
        metadata = dict(GATE_5537_ENUMERATION)
        description = f'- {DELETED_0D5} likeliest deletion target'
        service = gate_5537_service()

        result = await run_gate(
            project_root, metadata, service=service, description=description,
            agent_id='claude-interactive', append=True,
        )

        assert result.rejection is None
        assert result.metadata is metadata
        assert result.description is description
        service.get_memory_by_id.assert_not_awaited()


# --------------------------------------------------------------------------- #
# The Stage-2 prompt section is rendered from this module's own constants
# --------------------------------------------------------------------------- #


class TestRenderEnumerationGuardSection:
    """Pins single-sourcing only — the names the gate enforces appear in the
    rendered text, and the text is what Stage 2 sees — never prose wording."""

    def test_names_every_key_field_and_error_type_the_gate_enforces(self):
        text = render_enumeration_guard_section()

        for token in (
            X_ENUMERATION_SCOPE_KEY,
            X_LIVENESS_REVERIFIED_AT_KEY,
            X_LIVENESS_EXCEPTIONS_KEY,
            'stores', 'categories', 'exhaustive',
            'EnumeratedIdTruncated', 'EnumerationScopeMissing',
            'EnumerationScopeMalformed', ENUMERATION_UPDATE_NOT_OVERWRITING,
        ):
            assert token in text, token

    def test_the_stage2_system_prompt_carries_the_section_verbatim(self):
        assert render_enumeration_guard_section() in STAGE2_SYSTEM_PROMPT
