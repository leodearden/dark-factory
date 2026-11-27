"""Unit tests for fused_memory.middleware.preserved_specimen_gate_guard (task 5456).

The guard refuses, at ``submit_task``, a recon-filed escalating gate that asks
for a deliberately preserved specimen to be moved out of ``in-progress`` —
the shape of tasks 5080 and 5104, which asked a human to reset specimen 3105.

The module does not exist until step-2, so every test imports the names it
needs inside its body (the RED-collection idiom of
tests/test_recurring_gate_guard.py): the file stays collectable at RED and
each test fails on its own import.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _preserved_specimen_fixtures import (
    OWN_NODE_ENTITY,
    SPECIMEN_EDGE_UUID,
    SPECIMEN_MEMORY_ID,
    corroboration_memory,
    corroboration_reads,
    reset_gate_description,
    reset_gate_metadata,
    reset_gate_title,
    specimen_memory,
)
from shared.testing_virtual_clock import virtual_clock_test

_RESET_TITLE = reset_gate_title()
_RESET_DESCRIPTION = reset_gate_description()

_DARK_FACTORY_ROOT = '/home/leo/src/dark-factory'
_RECON_AGENT = 'recon-stage-task_knowledge_sync'
_GUARD_LOGGER = 'fused_memory.middleware.preserved_specimen_gate_guard'

_DEGRADED_ENTITY: dict[str, Any] = {
    'nodes': [], 'edges': [], 'degraded': True, 'failed_stores': ['graphiti'],
}

_ABSENT = object()


def _reset_gate_metadata(**overrides: Any) -> dict[str, Any]:
    """:func:`reset_gate_metadata` for 3105; an override of ``_ABSENT`` removes that key."""
    meta = reset_gate_metadata()
    for key, value in overrides.items():
        if value is _ABSENT:
            meta.pop(key, None)
        else:
            meta[key] = value
    return meta


async def _guard(
    metadata: Any,
    memory: AsyncMock,
    *,
    agent_id: str | None = _RECON_AGENT,
    title: str | None = _RESET_TITLE,
    description: str | None = _RESET_DESCRIPTION,
    **kwargs: Any,
) -> dict[str, Any] | None:
    from fused_memory.middleware.preserved_specimen_gate_guard import (
        preserved_specimen_gate_error,
    )

    return await preserved_specimen_gate_error(
        metadata,
        agent_id,
        _DARK_FACTORY_ROOT,
        title=title,
        description=description,
        memory_service=memory,
        **kwargs,
    )


def _warned_proceeding(caplog: pytest.LogCaptureFixture, subject: str) -> bool:
    return any(
        record.levelno == logging.WARNING
        and record.name == _GUARD_LOGGER
        and subject in record.getMessage()
        and 'proceeds' in record.getMessage()
        for record in caplog.records
    )


class TestExemptions:
    """Every short-circuit ahead of the corroboration read costs zero I/O."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('agent_id', ['claude-interactive', None])
    async def test_non_recon_caller_is_exempt(self, agent_id):
        memory = specimen_memory()

        assert await _guard(_reset_gate_metadata(), memory, agent_id=agent_id) is None
        assert corroboration_reads(memory) == 0

    @pytest.mark.asyncio
    async def test_code_tdd_filing_is_not_a_gate(self):
        memory = specimen_memory()

        meta = _reset_gate_metadata(execution_class='code_tdd')
        assert await _guard(meta, memory) is None
        assert corroboration_reads(memory) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('title', 'description'),
        [
            pytest.param(
                'GATE: stranded task 3105 needs a human ruling',
                _RESET_DESCRIPTION,
                id='status-alone-in-description',
            ),
            pytest.param(
                "Human action needed: reset task 3105's priority override back to its default",
                'The override was set during an incident and never cleared.',
                id='reset-verb-without-in-progress',
            ),
            pytest.param(
                'Human action needed: reset the priority override on task 3105',
                _RESET_DESCRIPTION,
                id='reset-verb-aimed-elsewhere-beside-status',
            ),
            pytest.param(None, None, id='no-title-no-description'),
            pytest.param(
                'Human action needed: reopen the escalation pinning task 3105 so it '
                'is not reset out of in-progress',
                _RESET_DESCRIPTION,
                id='negated-verb-beside-status-in-title',
            ),
            pytest.param(
                'Human action needed: reopen the pinning escalation for task 3105',
                'The pinning escalation was closed by the reaper; reset it to open '
                'so task 3105 stays in-progress.',
                id='verb-aimed-at-the-escalation-beside-status-in-description',
            ),
            pytest.param(
                'Human action needed: do not reset task 3105.',
                _RESET_DESCRIPTION,
                id='negated-verb-aimed-at-the-task-across-fields',
            ),
            pytest.param(
                'Human gate: keep task 3105 in-progress without resetting it to pending',
                None,
                id='verb-after-without',
            ),
        ],
    )
    async def test_gate_that_does_not_request_a_reset_is_not_enforced(
        self, title, description
    ):
        memory = specimen_memory()

        result = await _guard(
            _reset_gate_metadata(), memory, title=title, description=description
        )
        assert result is None
        assert corroboration_reads(memory) == 0


class TestSubjectRequired:
    """A recon reset gate must name the task it asks to move."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param(_reset_gate_metadata(gate_subject=_ABSENT), id='no-subject'),
            pytest.param(
                _reset_gate_metadata(gate_subject=_ABSENT, related_task_id='3105'),
                id='related-task-id-is-not-a-subject',
            ),
        ],
    )
    async def test_reset_gate_without_a_subject_is_refused_without_io(self, metadata):
        memory = specimen_memory()

        err = await _guard(metadata, memory)

        assert err is not None
        assert err['error_type'] == 'ResetGateSubjectRequired'
        assert 'gate_subject' in err['error'] + err['hint']
        assert corroboration_reads(memory) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('subject', ['#3105', 'task-3105', 'Task 3105', '3105,task-5080', '0'])
    async def test_subject_the_store_cannot_key_on_is_refused_without_io(self, subject):
        memory = specimen_memory()

        err = await _guard(_reset_gate_metadata(gate_subject=subject), memory)

        assert err is not None
        assert err['error_type'] == 'ResetGateSubjectRequired'
        assert repr(subject) in err['error']
        assert "'3105'" in err['hint']
        assert corroboration_reads(memory) == 0


class TestPreservedSpecimenRefused:
    """A reset gate whose subject corroborates as a preserved specimen is refused."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'routing',
        [
            pytest.param({'operational_mode': _ABSENT}, id='operational-mode-absent-5080'),
            pytest.param({'operational_mode': 'gate'}, id='operational-gate-5104'),
            pytest.param({'operational_mode': 'llm'}, id='operational-llm'),
            pytest.param({'execution_class': 'decision'}, id='decision'),
        ],
    )
    async def test_every_escalating_gate_shape_is_enforced(self, routing):
        memory = specimen_memory()

        err = await _guard(_reset_gate_metadata(**routing), memory)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'
        assert memory.get_memories_by_metadata.await_count == 1
        assert memory.get_entity.await_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('title', 'description'),
        [
            pytest.param(
                'OPERATOR GATE — confirm no live process holds task 3105 before '
                'resetting in-progress → pending',
                None,
                id='source-and-destination-3944',
            ),
            pytest.param(
                'Human action needed: reset task 3105',
                _RESET_DESCRIPTION,
                id='verb-aimed-at-the-task-across-fields',
            ),
            pytest.param(
                'Human gate: task 3105 stranded in-progress — resolve its re-dispatch gap',
                'An operator should re-drive it (reset to pending) once no live '
                'process holds it.',
                id='destination-in-description-3463',
            ),
            pytest.param(
                'Task 3105 is stuck in-progress; reset it.',
                None,
                id='pronoun-object-ends-the-clause',
            ),
        ],
    )
    async def test_destructive_phrasings_are_enforced(self, title, description):
        memory = specimen_memory()

        err = await _guard(_reset_gate_metadata(), memory, title=title, description=description)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'

    @pytest.mark.asyncio
    async def test_mem0_citation_is_named_and_read_is_scoped(self):
        from fused_memory.models.scope import resolve_project_id
        from fused_memory.reconciliation.preservation_specimen_guard import (
            preservation_mem0_filters,
        )

        memory = specimen_memory()

        err = await _guard(_reset_gate_metadata(), memory)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'
        assert err['gate_subject'] == '3105'
        assert err['preservation_citation'] == SPECIMEN_MEMORY_ID
        memory.get_memories_by_metadata.assert_awaited_once_with(
            project_id=resolve_project_id(_DARK_FACTORY_ROOT),
            filters=preservation_mem0_filters('3105'),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param(
                _reset_gate_metadata(gate_subject=_ABSENT, stranded_task_id='3105'),
                id='stranded-task-id-alias',
            ),
            pytest.param(json.dumps(_reset_gate_metadata()), id='json-string-metadata'),
        ],
    )
    async def test_subject_spellings_are_honoured(self, metadata):
        memory = specimen_memory()

        err = await _guard(metadata, memory)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'
        assert err['gate_subject'] == '3105'
        assert err['preservation_citation'] == SPECIMEN_MEMORY_ID

    @pytest.mark.asyncio
    async def test_graphiti_citation_when_mem0_has_none(self):
        memory = corroboration_memory(entity=OWN_NODE_ENTITY)

        err = await _guard(_reset_gate_metadata(), memory)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'
        assert err['preservation_citation'] == SPECIMEN_EDGE_UUID
        assert memory.get_memories_by_metadata.await_count == 1
        assert memory.get_entity.await_count == 1


class TestCompositeSubject:
    """A comma-joined subject names several tasks, and every one is corroborated."""

    @pytest.mark.asyncio
    async def test_specimen_inside_a_composite_is_refused(self):
        memory = specimen_memory()

        err = await _guard(_reset_gate_metadata(gate_subject='5080, 3105'), memory)

        assert err is not None
        assert err['error_type'] == 'PreservedSpecimenResetViolation'
        assert err['gate_subject'] == '3105'
        assert err['preservation_citation'] == SPECIMEN_MEMORY_ID

    @pytest.mark.asyncio
    async def test_composite_of_ordinary_tasks_passes_after_reading_each(self):
        memory = specimen_memory()

        result = await _guard(_reset_gate_metadata(gate_subject='3560,3561'), memory)

        assert result is None
        assert memory.get_memories_by_metadata.await_count == 2
        assert memory.get_entity.await_count == 2


class TestOrdinarySubjectPasses:
    """A reset gate for a task nobody preserved reaches the interceptor."""

    @pytest.mark.asyncio
    async def test_another_tasks_node_does_not_lend_its_citation(self):
        """The esc-4223-1 binding: only 'Task 3105' came back for subject 5231."""
        memory = corroboration_memory(entity=OWN_NODE_ENTITY)

        result = await _guard(_reset_gate_metadata(gate_subject='5231'), memory)

        assert result is None
        assert memory.get_entity.await_count == 1

    @pytest.mark.asyncio
    async def test_ordinary_stranded_subject_with_nothing_recorded(self):
        memory = specimen_memory()

        result = await _guard(_reset_gate_metadata(gate_subject='3843'), memory)

        assert result is None
        assert memory.get_memories_by_metadata.await_count == 1
        assert memory.get_entity.await_count == 1


class TestFailsOpen:
    """An unreadable verdict lets the submission through, and says so."""

    @pytest.mark.asyncio
    async def test_degraded_reads_fail_open_with_a_warning(self, caplog):
        memory = corroboration_memory(entity=_DEGRADED_ENTITY)
        memory.get_memories_by_metadata.side_effect = RuntimeError('qdrant down')

        with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
            result = await _guard(_reset_gate_metadata(), memory)

        assert result is None
        assert _warned_proceeding(caplog, '3105')

    @virtual_clock_test
    async def test_timeout_fails_open_with_a_warning(self, caplog):
        never = asyncio.Event()

        async def _hang(**_: Any) -> list[dict[str, Any]]:
            await never.wait()
            return []

        memory = corroboration_memory()
        memory.get_memories_by_metadata.side_effect = _hang

        with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
            result = await _guard(_reset_gate_metadata(), memory)

        assert result is None
        assert _warned_proceeding(caplog, '3105')

