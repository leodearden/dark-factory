"""Tests for fused_memory.reconciliation.internal_writers (task 3138).

The registry answers exactly one question: "is this agent_id a recognised
house-operated writer?"  It is the provenance source for
``flag_dedup.filter_style_only_authorship_flags`` and the render source for the
Stage-1 ``## Authorship Provenance Before Injection Flags`` prompt section,
closing the reify esc-5564-1 incident in which Stage 1 flagged its own earlier
consolidator output (agent_id ``recon-stage-memory_consolidator``) as "possibly
injected/fabricated" purely because the imperative writing style looked foreign
— it never checked the stored agent_id.

RED until step-2 creates the module.
"""
from __future__ import annotations

import pytest


class TestIsInternalWriter:
    """Tests for ``is_internal_writer(agent_id) -> bool``.

    Fail direction matters: this predicate gates a DROP of a security-shaped
    flag, so anything it does not positively recognise must classify as NOT
    internal (False), never as internal.
    """

    @pytest.mark.parametrize(
        'agent_id',
        [
            # The esc-5564-1 writer itself — the whole reason this registry exists.
            'recon-stage-memory_consolidator',
            'recon-stage-task_knowledge_sync',
            # Defensive: the ``reconciliation-stage-N`` doc-convention spelling
            # (CLAUDE.md write-tagging examples) as well as the runtime one.
            'reconciliation-stage-1',
            # Orchestrator-dispatched task agents.
            'claude-task-3138-architect',
            'orchestrator-task-3138',
            # The documented interactive/operator agent id.
            'claude-interactive',
        ],
    )
    def test_house_writer_populations_are_internal(self, agent_id):
        """Every house-operated writer family the task names classifies internal."""
        from fused_memory.reconciliation.internal_writers import is_internal_writer

        assert is_internal_writer(agent_id) is True, (
            f'{agent_id!r} is a house-operated writer and must classify as '
            'internal — otherwise the esc-5564-1 self-flagging recurs'
        )

    @pytest.mark.parametrize('agent_id', ['attacker-xyz', 'unknown-writer-7'])
    def test_foreign_agent_ids_are_not_internal(self, agent_id):
        """A genuinely unrecognised writer must NOT be cleared as house output."""
        from fused_memory.reconciliation.internal_writers import is_internal_writer

        assert is_internal_writer(agent_id) is False, (
            f'{agent_id!r} matches no house writer prefix and must classify as '
            'foreign — clearing it would silently disable injection detection'
        )

    @pytest.mark.parametrize('agent_id', [None, '', 123, {}, [], 0.0, object()])
    def test_absent_or_non_string_agent_ids_are_not_internal(self, agent_id):
        """Missing / empty / non-string provenance is unknown, never internal."""
        from fused_memory.reconciliation.internal_writers import is_internal_writer

        assert is_internal_writer(agent_id) is False, (
            f'{agent_id!r} carries no usable provenance and must classify as '
            'NOT internal (unknown authorship stays flaggable)'
        )


class TestInternalWriterPrefixRegistry:
    """The prefix tuple is the single source for BOTH the code gate and the
    rendered Stage-1 prompt fragment, so a malformed value must fail loudly
    here rather than render a blank agent-facing policy (INV-5)."""

    def test_prefixes_is_a_non_empty_tuple_of_non_empty_strings(self):
        from fused_memory.reconciliation.internal_writers import (
            INTERNAL_WRITER_AGENT_ID_PREFIXES,
        )

        assert isinstance(INTERNAL_WRITER_AGENT_ID_PREFIXES, tuple), (
            'INTERNAL_WRITER_AGENT_ID_PREFIXES must be a tuple (immutable '
            'single-source registry), got '
            f'{type(INTERNAL_WRITER_AGENT_ID_PREFIXES).__name__}'
        )
        assert INTERNAL_WRITER_AGENT_ID_PREFIXES, (
            'INTERNAL_WRITER_AGENT_ID_PREFIXES must be non-empty — an empty '
            'registry renders a blank Stage-1 policy and makes the gate a no-op'
        )
        for prefix in INTERNAL_WRITER_AGENT_ID_PREFIXES:
            assert isinstance(prefix, str) and prefix, (
                f'every prefix must be a non-empty string; got {prefix!r}'
            )

    def test_population_note_names_every_prefix(self):
        """The prompt fragment is rendered FROM the tuple, never hand-transcribed."""
        from fused_memory.reconciliation.internal_writers import (
            INTERNAL_WRITER_AGENT_ID_PREFIXES,
            INTERNAL_WRITER_POPULATION_NOTE,
        )

        assert isinstance(INTERNAL_WRITER_POPULATION_NOTE, str)
        assert INTERNAL_WRITER_POPULATION_NOTE.strip(), (
            'INTERNAL_WRITER_POPULATION_NOTE must be non-empty — it is '
            'interpolated into STAGE1_SYSTEM_PROMPT'
        )
        for prefix in INTERNAL_WRITER_AGENT_ID_PREFIXES:
            assert prefix in INTERNAL_WRITER_POPULATION_NOTE, (
                f'prefix {prefix!r} is missing from INTERNAL_WRITER_POPULATION_NOTE '
                '— the rendered agent-facing population has drifted from the registry'
            )
