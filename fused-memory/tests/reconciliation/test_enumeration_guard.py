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

import copy
import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.enumeration_guard import (
    X_LIVENESS_EXCEPTIONS_KEY,
    X_LIVENESS_REVERIFIED_AT_KEY,
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

# Canonical UUIDs that are NOT Mem0 ids: Stage-1 finding ids and Graphiti edge
# uuids, both of which get_memory_by_id (a Mem0 point read) reports as absent.
FINDING_IDS = [
    'f11018d4-be96-456b-9ab9-cf6814155500',
    '00004132-cd91-41dc-a059-bcd698bdd73c',
]
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
            service=memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E)),
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
        service = memory_service(
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
        service = memory_service(raising={
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
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
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
        service = memory_service(live=(LIVE_B3F, LIVE_21F, LIVE_23E))

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
        service = memory_service(live=(LIVE_B3F, LIVE_21F))

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
    async def test_the_input_is_never_mutated(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, DELETED_0D5],
            X_LIVENESS_EXCEPTIONS_KEY: [{'memory_id': 'old', 'outcome': 'absent'}],
        }
        snapshot = copy.deepcopy(metadata)

        await run_gate(
            project_root, metadata,
            service=memory_service(
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
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
            description=GATE_5537_DESCRIPTION,
        )

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
            service=memory_service(
                live=(LIVE_B3F,), raising={LIVE_21F: TimeoutError('slow')},
            ),
            description=description,
        )

        assert result.description.startswith(description)
        assert 'liveness inconclusive (TimeoutError)' in result.description
        assert 'deleted' not in result.description

    @pytest.mark.asyncio
    async def test_annotating_twice_adds_no_second_annotation(self, project_root):
        metadata = {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]}
        service = memory_service(live=(LIVE_B3F,))

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
    async def test_a_none_description_stays_none(self, project_root):
        result = await run_gate(
            project_root, {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]},
            service=memory_service(live=(LIVE_B3F,)), description=None,
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
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
            description=description,
        )

        assert result.description is description

    @pytest.mark.asyncio
    async def test_the_description_is_never_a_detection_source(self, project_root):
        service = memory_service(live=(LIVE_B3F,))
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
            service=memory_service(live=(LIVE_B3F,)), description=description,
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
        lambda: memory_service(live=(LIVE_B3F, LIVE_21F)),
    ),
    'one-absent': (
        [LIVE_B3F, DELETED_0D5],
        lambda: memory_service(live=(LIVE_B3F,)),
    ),
    'one-inconclusive': (
        [LIVE_B3F, LIVE_21F],
        lambda: memory_service(
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
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
        )

        assert result.metadata[X_LIVENESS_REVERIFIED_AT_KEY] == FIXED_NOW.isoformat()

    @pytest.mark.asyncio
    async def test_incoming_exceptions_are_replaced_never_merged(self, project_root):
        metadata = {
            'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5],
            X_LIVENESS_EXCEPTIONS_KEY: list(STALE_EXCEPTIONS),
        }

        result = await run_gate(
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
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
            project_root, metadata, service=memory_service(live=(LIVE_B3F,)),
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

        result = await run_gate(project_root, metadata, service=memory_service())

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
            project_root, metadata, service=memory_service(), is_filing=False,
        )

        assert X_LIVENESS_REVERIFIED_AT_KEY not in result.metadata
        assert [r['outcome'] for r in result.metadata[X_LIVENESS_EXCEPTIONS_KEY]] == (
            ['truncated'] * len(GATE_5533_FRAGMENTS)
        )
