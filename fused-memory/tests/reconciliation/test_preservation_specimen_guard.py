"""Tests for the Stage-1 preservation-specimen corroboration guard (task 4223).

Task 3105 is ``in-progress`` with a null claimant and a null heartbeat *on
purpose*: it is the sole preserved live validation specimen for gate task
3546.  Stage 1's stranded-task heuristic cannot see that, so it re-emits a
moderate/actionable "stranded" finding on roughly every cycle, and twice that
finding became an operator-gate task asking for the specimen to be reset —
5080 (filed 2026-09-04T09:18:52Z) and 5104 (born-at-L2 critical, filed
2026-09-04T14:09:16Z).  Both were declined and cancelled by hand.

This module's guard derives the verdict from the preservation evidence that
already exists in both stores, so the recommendation is dropped before it can
reach Stage 2.

Covers:
- ``cites_preservation`` — the shared preservation-citation matcher, run
  against the VERBATIM live corroboration strings of both channels.
- ``flag_asserts_stranded`` — the stranded-class discriminator, run against
  all three flag_type namings this one false positive has actually worn.
- the DISTINCTIVENESS of both token families, checked behaviourally against
  live-but-unrelated flag_types and unrelated recon prose.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.preservation_specimen_guard import (
    MEM0_KIND_INVESTIGATION_OUTCOME,
    PRESERVATION_TOKEN_FAMILY,
    STRANDED_FLAG_TOKEN_FAMILY,
    cites_preservation,
    filter_preservation_specimen_flags,
    flag_asserts_stranded,
)

# ── Live corroboration strings, copied verbatim from the base branch ─────────
# Re-verify at any time:
#   get_entity('Task 3105', 'dark_factory')                      -> edge a8fd36a8
#   get_memories_by_metadata('dark_factory',
#       {'kind': 'investigation_outcome', 'task_id': '3105',
#        'actionable': False})                                   -> 5 rows

#: Graphiti edge a8fd36a8-46db-4ca8-a21c-554c38a918ee, verbatim.
LIVE_GRAPHITI_EDGE_FACT = (
    'Task 3105 is the sole preserved live validation specimen for gate task 3546.'
)

#: Mem0 d489942a-96b8-45a2-a0a2-fdad36741ff2 (2026-08-10, run c2e75c48), verbatim.
LIVE_MEM0_PRESERVATION_PROSE = (
    "Investigation outcome: Stage 1's 2026-08-10 (run c2e75c48) re-flag of "
    'dark_factory task 3105 as a stranded in-progress task, with the suggested '
    'action of filing an OPERATOR GATE task to reset it in-progress->pending, is '
    "NOT actionable. Task 3105's in-progress/null-claimant/null-heartbeat state "
    'is intentional: it is the sole remaining preserved live validation specimen '
    "for gate task 3546's soak/flip checklist (per 3546's SECOND and THIRD "
    'DEVIATION NOTICES), corroborated by Graphiti edge '
    'a8fd36a8-46db-4ca8-a21c-554c38a918ee. Filing a reset gate would risk '
    'repeating the hand-recovery evidence loss already suffered on task 3066 '
    '(via task 3463) and possibly task 3371 (under investigation by task 3914). '
    "No gate task filed; no status change made; a confirmation note was appended "
    "to task 3546's details field."
)

#: The remaining four live rows' load-bearing sentences, verbatim.  Included so
#: the family is checked against every phrasing the corpus actually carries, not
#: just the one it was written from.
LIVE_PRESERVATION_SENTENCES = (
    LIVE_GRAPHITI_EDGE_FACT,
    LIVE_MEM0_PRESERVATION_PROSE,
    # 92ebc941 (2026-08-25) — note "preserved dark-factory validation specimen",
    # a phrasing no 'preserved live validation specimen' literal would catch.
    "Task 3546's own text records that task 3105 is the SOLE remaining preserved "
    'dark-factory validation specimen for that soak/flip checklist',
    # 4dcf7de6 (2026-08-26)
    "Task 3105 is cited in gate task 3546's text as the sole preserved live "
    "validation specimen for that gate's soak/flip checklist",
    # 0397cf9d (2026-09-10) — "preserved SOLE live validation specimen".
    "Task 3105's in-progress/null-claimant/null-heartbeat state remains the "
    'deliberately preserved SOLE live validation specimen for gate task 3546'
    "'s mu-gate soak/flip checklist",
    # Graphiti node 95719d7c's summary, verbatim and WHOLE.  The Graphiti
    # channel matches against the entire summary string and each entire edge
    # fact — never a single line in isolation — so the whole blob is the
    # honest fixture.  Note only one of these seven lines carries the citation;
    # requiring each line to match on its own would force the family wider than
    # the live evidence justifies, which is the over-suppression the
    # under-suppression bias forbids.
    "Any urgent-intervention concern about task 3105 must be recorded in task "
    "3546's own description.\n"
    'Task 3105 is the sole preserved live validation specimen for gate task 3546.\n'
    'Task 3546 nominates task 3105 as the substitute traced specimen.\n'
    "Task 3105's state is keyed by task_ground_truth.py based on its status and "
    'other attributes.\n'
    "Task 3105's preservation is documented as a precedent due to the harm "
    'caused when twin specimen task 3371 lost its pin in a bulk close.\n'
    'The open escalation for task 3105 is the preservation mechanism, as it '
    'falls through to RecoveryAction.LEAVE.\n'
    'The preserved set for dark_factory task 3371 was recorded as SIZE-1, which '
    'includes task 3105 only.',
)

#: Recon prose with nothing to do with a preserved specimen.  A family member
#: that fires on any of these would DROP a genuine finding, which is the
#: failure the under-suppression bias forbids.
UNRELATED_RECON_PROSE = (
    'Task 4102 has been in-progress for 9 days with no claimant and no '
    'heartbeat; recommend an operator reset it to pending.',
    'Entity "Reconciliation Pipeline" carries 214 edges whose facts conflate '
    'several unrelated topics; consider splitting it.',
    'Memory a1b2c3d4 duplicates e5f6a7b8 at cosine 0.97; the later write should '
    'be consolidated into the canonical entry.',
    'The stage2_cycle_summary pool has grown to 1,412 entries, well past its '
    'configured cap, because GC used semantic search instead of a payload scroll.',
    'Investigation outcome: the flagged escalation esc-4102-1 was already '
    'resolved by the steward on 2026-09-02; no further action is needed.',
    'Task 3066 was hand-recovered by the operator after its worktree was '
    'reset, and the recovery is recorded in its details field.',
)

#: flag_types drawn from the live Stage-1 vocabulary that have NOTHING to do
#: with a stranded task.  ``flag_asserts_stranded`` firing on one of these would
#: suppress a real, unrelated finding for any preserved task.
UNRELATED_LIVE_FLAG_TYPES = (
    'task_blocked_stale_escalations',
    'task_absent',
    'phantom_tasks_created',
    'terminal_state_pre_check',
    'cross_project',
    'task_completed_not_reflected',
    'duplicate_procedural_knowledge',
    'oversized_entity',
    'topic_conflation',
    'recon_stale_task_count_snapshot',
    'scope_violation',
    'stale_priority_override',
)

#: Every flag_type naming this single false positive has actually worn, in the
#: order the corpus records them.  An enumerated (task_id, flag_type)
#: suppression list is exactly what could not keep up with this drift.
OBSERVED_STRANDED_FLAG_TYPES = (
    'task_stranded_no_claimant',
    'task_stranded_no_claimant_heartbeat',
    'stranded_merge_phase_liveness',
)


class TestCitesPreservation:
    """The preservation-citation matcher, checked against the live corpus."""

    @pytest.mark.parametrize('text', LIVE_PRESERVATION_SENTENCES)
    def test_live_corroboration_strings_match(self, text):
        """Every phrasing the live corpus carries is recognised as a citation."""
        assert cites_preservation(text) is True, (
            f'live preservation citation not recognised: {text!r}'
        )

    def test_the_graphiti_edge_fact_matches_verbatim(self):
        """The single edge both channels' investigations cite back to."""
        assert cites_preservation(LIVE_GRAPHITI_EDGE_FACT) is True

    def test_the_mem0_investigation_prose_matches_verbatim(self):
        """Mem0 d489942a, the first recorded adjudication of this false positive."""
        assert cites_preservation(LIVE_MEM0_PRESERVATION_PROSE) is True

    def test_matching_is_case_insensitive(self):
        """The corpus shouts ("SOLE", "PRESERVED"); matching must not care."""
        assert cites_preservation(LIVE_GRAPHITI_EDGE_FACT.upper()) is True

    @pytest.mark.parametrize('text', UNRELATED_RECON_PROSE)
    def test_unrelated_recon_prose_does_not_match(self, text):
        """Unrelated prose is NOT a preservation citation.

        This is the behavioural distinctiveness check
        ``standing_decision_constants.GROUNDS_TOKEN_FAMILIES``' comment demands:
        "distinctive" means "does not occur outside this class", and that is a
        checkable property, not a slogan.  A family member firing here would
        silently DROP a genuine stranded finding.
        """
        assert cites_preservation(text) is False, (
            f'over-broad preservation family matched unrelated prose: {text!r}'
        )

    @pytest.mark.parametrize('value', [None, 123, b'bytes', [], {}, ''])
    def test_total_over_malformed_input(self, value):
        """Non-``str``/empty input is False, never an exception."""
        assert cites_preservation(value) is False

    def test_family_carries_no_bare_generic_word(self):
        """No member is a single generic word.

        ``'preserved'``, ``'specimen'`` and ``'intentional'`` all occur
        throughout unrelated recon prose; a family holding one of them bare
        would suppress far past this class.  Multi-word phrases only.
        """
        for stem in PRESERVATION_TOKEN_FAMILY:
            assert ' ' in stem or '-' in stem, (
                f'{stem!r} is a bare single word — too generic for a substring '
                f'family (see the module docstring on distinctiveness)'
            )


class TestFlagAssertsStranded:
    """The stranded-class discriminator over free-form LLM flag dicts."""

    @pytest.mark.parametrize('flag_type', OBSERVED_STRANDED_FLAG_TYPES)
    def test_every_observed_naming_matches(self, flag_type):
        """All three namings this false positive has worn are recognised."""
        assert flag_asserts_stranded({'task_id': '3105', 'flag_type': flag_type}) is True

    def test_novel_flag_type_with_reset_action_matches(self):
        """A naming nobody has seen yet, recognised by its requested ACTION.

        flag_type is free-form LLM output, so the vocabulary channel alone will
        always lag.  A finding asking for a status reset out of ``in-progress``
        is making the stranded claim whatever it calls itself.
        """
        flag = {
            'task_id': '3105',
            'flag_type': 'liveness_anomaly_unattended_workflow',
            'suggested_action': (
                'File an operator gate task to reset task 3105 from in-progress '
                'to pending so it can be redispatched.'
            ),
        }
        assert flag_asserts_stranded(flag) is True

    def test_reset_verb_without_in_progress_target_does_not_match(self):
        """Both halves are required: a reset of something else is not this class."""
        flag = {
            'task_id': '3105',
            'flag_type': 'stale_priority_override',
            'suggested_action': 'Reset the priority override back to its default.',
        }
        assert flag_asserts_stranded(flag) is False

    @pytest.mark.parametrize('flag_type', UNRELATED_LIVE_FLAG_TYPES)
    def test_unrelated_live_flag_types_do_not_match(self, flag_type):
        """Behavioural distinctiveness guard, run against the live vocabulary.

        Every entry here is a flag_type Stage 1 actually emits.  A stem that
        fires on one of them does not merely add noise: for a preserved task it
        DROPS that unrelated finding.
        """
        assert flag_asserts_stranded({'task_id': '3105', 'flag_type': flag_type}) is False, (
            f'over-broad stranded family matched live flag_type {flag_type!r}'
        )

    @pytest.mark.parametrize(
        'flag',
        [
            {},
            {'task_id': '3105'},
            {'task_id': '3105', 'flag_type': None},
            {'task_id': '3105', 'flag_type': 123},
            {'task_id': '3105', 'flag_type': ''},
            {'task_id': '3105', 'flag_type': ['stranded']},
            {'task_id': '3105', 'flag_type': 'x', 'suggested_action': None},
            {'task_id': '3105', 'flag_type': 'x', 'suggested_action': 42},
        ],
    )
    def test_total_over_malformed_flags(self, flag):
        """``items_flagged`` entries are free-form LLM dicts; never raise on one."""
        assert flag_asserts_stranded(flag) is False

    def test_not_a_mapping_is_false(self):
        """A non-dict element in items_flagged is skipped, not fatal."""
        assert flag_asserts_stranded(None) is False
        assert flag_asserts_stranded('stranded') is False

    def test_matching_is_case_insensitive(self):
        """Casefolded, like ``_flag_type_in_grounds_family``."""
        assert flag_asserts_stranded({'flag_type': 'TASK_STRANDED_NO_CLAIMANT'}) is True

    def test_family_stem_covers_all_three_namings(self):
        """The vocabulary channel is ONE stem, not an enumerated list.

        ``'strand'`` is inside ``task_stranded_no_claimant``,
        ``task_stranded_no_claimant_heartbeat`` and
        ``stranded_merge_phase_liveness`` alike — which is the whole reason a
        stem family keeps up with naming drift where a (task_id, flag_type)
        list cannot.
        """
        assert 'strand' in STRANDED_FLAG_TOKEN_FAMILY
        for naming in OBSERVED_STRANDED_FLAG_TYPES:
            assert any(stem in naming for stem in STRANDED_FLAG_TOKEN_FAMILY)


# ── filter_preservation_specimen_flags ───────────────────────────────────────

#: A Mem0 row shaped exactly like the live Qdrant payload
#: ``get_memories_by_metadata`` returns: ``metadata`` is the FULL payload, so
#: the prose lives under ``metadata['data']`` and is reachable only through the
#: canonical ``_mem0_content`` key fallback.
LIVE_MEM0_ROW = {
    'id': 'd489942a-96b8-45a2-a0a2-fdad36741ff2',
    'created_at': '2026-08-10T17:22:17.388627+00:00',
    'metadata': {
        'kind': 'investigation_outcome',
        'entity_uuid': '95719d7c-3778-4687-8b50-10ec5ad46e75',
        'actionable': False,
        'run_id': 'c2e75c48-5cb5-427b-bc13-8964a00726ab',
        'task_id': '3105',
        'category': 'observations_and_summaries',
        'agent_id': 'recon-stage-task_knowledge_sync',
        'data': LIVE_MEM0_PRESERVATION_PROSE,
    },
}

#: The live Graphiti shape for ``get_entity('Task 3105', ...)``.
LIVE_GRAPHITI_ENTITY = {
    'nodes': [
        {
            'uuid': '95719d7c-3778-4687-8b50-10ec5ad46e75',
            'name': 'Task 3105',
            'summary': LIVE_PRESERVATION_SENTENCES[-1],
            'labels': ['Entity'],
        },
    ],
    'edges': [
        {
            'uuid': 'a8a68220-2a9c-418e-b358-cb39c554e067',
            'fact': 'Any urgent-intervention concern about task 3105 must be '
                    "recorded in task 3546's own description.",
        },
        {'uuid': 'a8fd36a8-46db-4ca8-a21c-554c38a918ee', 'fact': LIVE_GRAPHITI_EDGE_FACT},
    ],
}

PROJECT = 'dark_factory'

GUARD_LOGGER = 'fused_memory.reconciliation.preservation_specimen_guard'


@contextlib.contextmanager
def caplog_at_warning():
    """Yield a growing list of the guard's WARNING+ messages.

    A local handler rather than pytest's ``caplog`` fixture: these tests run
    under xdist and assert on one named logger, so attaching to that logger
    directly keeps the capture independent of global log-level state.
    """
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            # getMessage() already interpolates record.args.
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    log = logging.getLogger(GUARD_LOGGER)
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def _stranded_flag(task_id='3105', flag_type='task_stranded_no_claimant', **extra):
    """A Stage-1 stranded finding, shaped like the ones actually emitted."""
    flag = {
        'task_id': task_id,
        'flag_type': flag_type,
        'category': 'task_memory_mismatch',
        'severity': 'moderate',
        'actionable': True,
        'description': f'Task {task_id} is in-progress with no claimant and no heartbeat.',
        'suggested_action': f'File an operator gate task to reset task {task_id} to pending.',
    }
    flag.update(extra)
    return flag


def _make_memory_service(*, rows=None, entities=None, mem0_error=None, entity_error=None):
    """MagicMock memory_service with AsyncMock readers for both channels.

    Mirrors ``_make_memory_service`` in test_curator_gate_resolution_sweep.py.
    *rows* maps a str task_id to the Mem0 scroll result for it; *entities* maps
    an entity NAME (``'Task 3105'``) to the ``get_entity`` result.  Either
    channel can be made to raise by passing an exception instance.
    """
    rows = rows or {}
    entities = entities or {}

    def _scroll(project_id, filters):
        if mem0_error is not None:
            raise mem0_error
        return rows.get(str(filters.get('task_id')), [])

    def _entity(name, project_id, **kwargs):
        if entity_error is not None:
            raise entity_error
        return entities.get(name, {'nodes': [], 'edges': []})

    memory_service = MagicMock()
    memory_service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
    memory_service.get_entity = AsyncMock(side_effect=_entity)
    return memory_service


class TestFilterPreservationSpecimenFlagsMem0Channel:
    """The Mem0 corroboration channel — the concrete task-3105 repro."""

    @pytest.mark.asyncio
    async def test_corroborated_stranded_flag_is_suppressed(self):
        """The exact finding that twice became a destructive gate task is dropped."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})
        flag = _stranded_flag()

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {'3105': 1}
        assert result.unresolved_task_ids == ()

    @pytest.mark.asyncio
    async def test_suppression_cites_the_memory_it_relied_on(self):
        """A suppression is never anonymous — it names its evidence.

        Without this an operator reading "one flag suppressed" has no way to
        check whether the citation is real, current, or over-broad.
        """
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.citations_by_task['3105'] == 'd489942a-96b8-45a2-a0a2-fdad36741ff2'

    @pytest.mark.asyncio
    async def test_queries_the_deterministic_metadata_filter(self):
        """Exact payload equality, never semantic search.

        ``actionable: False`` is ANDed in deliberately: Qdrant ANDs equality
        conditions, so the term is what makes a retrieved row a RECORDED
        not-actionable adjudication rather than any passing mention of the task.
        """
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})

        await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        memory_service.get_memories_by_metadata.assert_awaited_once_with(
            project_id=PROJECT,
            filters={
                'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
                'task_id': '3105',
                'actionable': False,
            },
        )

    @pytest.mark.asyncio
    async def test_row_without_preservation_prose_does_not_suppress(self):
        """An investigation_outcome that says something else is not a citation."""
        row = {
            'id': 'aaaaaaaa-0000-0000-0000-000000000000',
            'metadata': {
                'kind': 'investigation_outcome',
                'task_id': '3105',
                'actionable': False,
                'data': 'Investigation outcome: the flagged escalation was already '
                        'resolved by the steward; no further action is needed.',
            },
        }
        memory_service = _make_memory_service(rows={'3105': [row]})
        flag = _stranded_flag()

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}
        assert result.citations_by_task == {}

    @pytest.mark.asyncio
    async def test_reads_prose_through_the_canonical_key_fallback(self):
        """A row storing its text under 'memory' rather than 'data' still matches.

        The raw-payload key order (``data`` -> ``memory`` -> ``content``) is
        owned by ``memory_service._MEM0_CONTENT_KEYS`` and imported, not
        re-spelled here — guessing one key is exactly the mistake that helper
        exists to prevent.
        """
        row = {
            'id': 'bbbbbbbb-0000-0000-0000-000000000000',
            'metadata': {'memory': LIVE_GRAPHITI_EDGE_FACT},
        }
        memory_service = _make_memory_service(rows={'3105': [row]})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.kept_flags == []
        assert result.citations_by_task['3105'] == 'bbbbbbbb-0000-0000-0000-000000000000'


class TestFilterPreservationSpecimenFlagsGraphitiChannel:
    """The Graphiti fallback — what protects a NEWLY documented specimen.

    The two channels are not redundant.  A Graphiti edge exists as soon as the
    preservation fact is recorded, whereas an ``investigation_outcome`` row only
    exists after some stage has ALREADY investigated a flag — so on cycle one,
    the cycle in which tasks 5080 and 5104 were filed, Graphiti is the only
    channel with anything in it.
    """

    @pytest.mark.asyncio
    async def test_falls_back_when_mem0_is_empty(self):
        """No investigation_outcome row yet, but the preservation fact is in the graph."""
        memory_service = _make_memory_service(entities={'Task 3105': LIVE_GRAPHITI_ENTITY})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {'3105': 1}
        assert result.unresolved_task_ids == ()
        memory_service.get_entity.assert_awaited_once_with('Task 3105', PROJECT)

    @pytest.mark.asyncio
    async def test_cites_the_edge_that_carries_the_fact(self):
        """The citation is the atomic edge, not the whole entity.

        An edge uuid is individually addressable (``get_edge``) and durable; a
        node summary is regenerated by ``refresh_entity_summary``.  Citing the
        edge is what lets a reader check the evidence rather than take the
        suppression on trust.
        """
        memory_service = _make_memory_service(entities={'Task 3105': LIVE_GRAPHITI_ENTITY})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.citations_by_task['3105'] == 'a8fd36a8-46db-4ca8-a21c-554c38a918ee'

    @pytest.mark.asyncio
    async def test_node_summary_alone_suffices(self):
        """A summary carrying the fact corroborates even with no matching edge."""
        entity = {
            'nodes': [
                {
                    'uuid': '95719d7c-3778-4687-8b50-10ec5ad46e75',
                    'name': 'Task 3105',
                    'summary': LIVE_PRESERVATION_SENTENCES[-1],
                },
            ],
            'edges': [{'uuid': 'a8a68220-2a9c-418e-b358-cb39c554e067', 'fact': 'unrelated'}],
        }
        memory_service = _make_memory_service(entities={'Task 3105': entity})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.kept_flags == []
        assert result.citations_by_task['3105'] == '95719d7c-3778-4687-8b50-10ec5ad46e75'

    @pytest.mark.asyncio
    async def test_graphiti_is_not_consulted_when_mem0_answered(self):
        """Cheapest-first: the fallback costs nothing on the common path.

        Mem0's metadata filter is both cheaper and the more authoritative
        signal, so a second round trip to Graphiti is pure waste once it has
        answered.
        """
        memory_service = _make_memory_service(
            rows={'3105': [LIVE_MEM0_ROW]},
            entities={'Task 3105': LIVE_GRAPHITI_ENTITY},
        )

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.kept_flags == []
        assert memory_service.get_entity.await_count == 0

    @pytest.mark.asyncio
    async def test_entity_without_preservation_text_keeps_the_flag(self):
        """A task with a graph node but no preservation fact is NOT corroborated."""
        entity = {
            'nodes': [{'uuid': 'n-1', 'name': 'Task 4102', 'summary': 'Task 4102 is blocked.'}],
            'edges': [{'uuid': 'e-1', 'fact': 'Task 4102 depends on task 4101.'}],
        }
        memory_service = _make_memory_service(entities={'Task 4102': entity})
        flag = _stranded_flag(task_id='4102')

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}
        assert result.unresolved_task_ids == ()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'entity',
        [
            None,
            'not a mapping',
            {},
            {'nodes': None, 'edges': None},
            {'nodes': 'oops', 'edges': 7},
            {'nodes': [None, 'x', 42], 'edges': [None, 'y']},
            {'nodes': [{'summary': LIVE_GRAPHITI_EDGE_FACT}]},
            {'edges': [{'fact': LIVE_GRAPHITI_EDGE_FACT}]},
        ],
    )
    async def test_total_over_malformed_graphiti_results(self, entity):
        """A malformed result yields no citation rather than raising.

        Includes the two uuid-less shapes: a matching node/edge carrying no
        uuid cannot be cited, and an uncitable suppression is refused for the
        same reason the Mem0 channel refuses an id-less row.
        """
        memory_service = _make_memory_service()
        memory_service.get_entity = AsyncMock(return_value=entity)
        flag = _stranded_flag()

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}


class TestBoundedScopeAndReadEconomy:
    """What the guard must NOT drop, and what it must not spend to decide.

    Over-suppression is the expensive direction: this guard runs on every
    cycle's whole flag batch, so a scope error hides real findings fleet-wide.
    The stranded-class discriminator is what bounds it.
    """

    @pytest.mark.asyncio
    async def test_unrelated_flag_for_a_preserved_task_is_kept(self):
        """(a) Preserved does not mean invisible.

        Task 3105 carries a preservation citation, but a finding that says
        something ELSE about it is not the adjudicated class and must still
        reach Stage 2.  Without this the guard would turn a preserved specimen
        into a blind spot.
        """
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})
        flag = {
            'task_id': '3105',
            'flag_type': 'task_completed_not_reflected',
            'description': 'Task 3105 has curator entries suggesting it is resolved.',
        }

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}

    @pytest.mark.asyncio
    async def test_unrelated_flag_costs_no_backend_read(self):
        """Narrowing happens BEFORE I/O, not after."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})

        await filter_preservation_specimen_flags(
            memory_service=memory_service,
            project_id=PROJECT,
            flags=[{'task_id': '3105', 'flag_type': 'task_completed_not_reflected'}],
        )

        assert memory_service.get_memories_by_metadata.await_count == 0
        assert memory_service.get_entity.await_count == 0

    @pytest.mark.asyncio
    async def test_stranded_flag_without_citation_is_kept(self):
        """(b) A genuinely stranded task is exactly what Stage 1 should report."""
        memory_service = _make_memory_service()
        flag = _stranded_flag(task_id='4102')

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}
        assert result.citations_by_task == {}
        assert result.unresolved_task_ids == ()

    @pytest.mark.asyncio
    async def test_batch_with_no_stranded_flag_short_circuits(self):
        """(c) A cycle with nothing in this class costs ZERO backend calls."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})
        flags = [
            {'task_id': '3105', 'flag_type': 'duplicate_procedural_knowledge'},
            {'task_id': '4102', 'flag_type': 'oversized_entity'},
            {'task_id': '4103', 'flag_type': 'task_absent'},
        ]

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=flags,
        )

        assert result.kept_flags == flags
        assert memory_service.get_memories_by_metadata.await_count == 0
        assert memory_service.get_entity.await_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('flags', [[], None])
    async def test_empty_input_returns_empty_result_with_no_reads(self, flags):
        """(d) Nothing in, nothing out, nothing spent."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=flags,
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {}
        assert result.citations_by_task == {}
        assert result.unresolved_task_ids == ()
        assert memory_service.get_memories_by_metadata.await_count == 0

    @pytest.mark.asyncio
    async def test_two_flags_for_one_task_cost_one_read(self):
        """(e) The per-task verdict is memoized; N flags cost one corroboration."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})
        flags = [
            _stranded_flag(flag_type='task_stranded_no_claimant'),
            _stranded_flag(flag_type='stranded_merge_phase_liveness'),
        ]

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=flags,
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {'3105': 2}
        assert memory_service.get_memories_by_metadata.await_count == 1

    @pytest.mark.asyncio
    async def test_uncorroborated_task_is_also_memoized(self):
        """A negative verdict is cached too, so a noisy task is not re-read per flag."""
        memory_service = _make_memory_service()
        flags = [_stranded_flag(task_id='4102'), _stranded_flag(task_id='4102')]

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=flags,
        )

        assert result.kept_flags == flags
        assert memory_service.get_memories_by_metadata.await_count == 1
        assert memory_service.get_entity.await_count == 1

    @pytest.mark.asyncio
    async def test_int_task_id_is_coerced_and_matches(self):
        """(f) Task ids arrive as ints straight off a task dict."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag(task_id=3105)],
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {'3105': 1}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'task_id', [None, '', 0, [], {'nested': 'dict'}],
    )
    async def test_unusable_task_id_is_kept_without_reads(self, task_id):
        """(f) A flag with no resolvable task is kept, never a junk backend query."""
        memory_service = _make_memory_service()
        flag = _stranded_flag(task_id=task_id)

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert memory_service.get_memories_by_metadata.await_count == 0

    @pytest.mark.asyncio
    async def test_absent_task_id_key_is_kept_without_reads(self):
        """(f) ``items_flagged`` entries may omit task_id entirely."""
        memory_service = _make_memory_service()
        flag = {'flag_type': 'task_stranded_no_claimant', 'description': 'no task id'}

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert memory_service.get_memories_by_metadata.await_count == 0

    @pytest.mark.asyncio
    async def test_input_order_is_preserved_among_survivors(self):
        """kept_flags is assigned straight back to items_flagged; order must hold."""
        memory_service = _make_memory_service(rows={'3105': [LIVE_MEM0_ROW]})
        a = {'task_id': '4101', 'flag_type': 'task_absent'}
        b = _stranded_flag()
        c = {'task_id': '4103', 'flag_type': 'oversized_entity'}
        d = _stranded_flag(task_id='4104')

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[a, b, c, d],
        )

        assert result.kept_flags == [a, c, d]


class TestDegradedCorroborationReads:
    """INV-11: fail OPEN on the drop, but never silently.

    The two failure directions are asymmetric. Suppressing on a backend blip
    would hide every stranded finding fleet-wide — the hidden-finding cost the
    under-suppression bias forbids — whereas KEEPING the flag costs at most one
    more cycle of a false positive the rotation has absorbed since August.

    But a bare fail-open makes "corroboration unreadable" byte-identical to "no
    citation exists", and then the destructive recommendation flows on
    unremarked. ``unresolved_task_ids`` is what stops that.
    """

    @pytest.mark.asyncio
    async def test_mem0_timeout_keeps_the_flag_and_discloses_it(self):
        """(a) TimeoutError is the real failure ``scroll_by_metadata`` propagates."""
        memory_service = _make_memory_service(mem0_error=TimeoutError('qdrant read timed out'))
        flag = _stranded_flag()

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.suppressed_by_task == {}
        assert result.unresolved_task_ids == ('3105',)

    @pytest.mark.asyncio
    async def test_graphiti_failure_keeps_the_flag_and_discloses_it(self):
        """(b) The fallback channel degrades the same way."""
        memory_service = _make_memory_service(entity_error=RuntimeError('falkordb down'))
        flag = _stranded_flag()

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[flag],
        )

        assert result.kept_flags == [flag]
        assert result.unresolved_task_ids == ('3105',)

    @pytest.mark.asyncio
    async def test_failed_channel_logs_a_warning_naming_the_task(self):
        """Loud over silent: the degradation is in the log stream, not only a field."""
        memory_service = _make_memory_service(mem0_error=TimeoutError('boom'))
        with caplog_at_warning() as records:
            await filter_preservation_specimen_flags(
                memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
            )
        blob = '\n'.join(records)
        assert '3105' in blob
        assert PROJECT in blob

    @pytest.mark.asyncio
    async def test_graphiti_still_answers_when_mem0_is_broken(self):
        """(c) One broken channel must not cost the OTHER channel's verdict.

        A citation found on the surviving channel suppresses AND leaves the task
        out of ``unresolved_task_ids`` — the corroboration WAS resolved, just not
        by the cheap path.
        """
        memory_service = _make_memory_service(
            mem0_error=TimeoutError('qdrant read timed out'),
            entities={'Task 3105': LIVE_GRAPHITI_ENTITY},
        )

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
        )

        assert result.kept_flags == []
        assert result.suppressed_by_task == {'3105': 1}
        assert result.citations_by_task['3105'] == 'a8fd36a8-46db-4ca8-a21c-554c38a918ee'
        assert result.unresolved_task_ids == ()

    @pytest.mark.asyncio
    async def test_both_channels_broken_is_unresolved_once(self):
        """A task is disclosed once, not once per failed channel."""
        memory_service = _make_memory_service(
            mem0_error=TimeoutError('down'), entity_error=RuntimeError('also down'),
        )
        flags = [_stranded_flag(), _stranded_flag(flag_type='stranded_merge_phase_liveness')]

        result = await filter_preservation_specimen_flags(
            memory_service=memory_service, project_id=PROJECT, flags=flags,
        )

        assert result.kept_flags == flags
        assert result.unresolved_task_ids == ('3105',)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc', [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()],
    )
    async def test_lifecycle_exceptions_propagate(self, exc):
        """(d) Shutdown is not a backend blip — never absorbed as best-effort."""
        memory_service = _make_memory_service(mem0_error=exc)

        with pytest.raises(type(exc)):
            await filter_preservation_specimen_flags(
                memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc', [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()],
    )
    async def test_lifecycle_exceptions_propagate_from_graphiti_too(self, exc):
        """(d) Both channels, so neither can quietly swallow a shutdown."""
        memory_service = _make_memory_service(entity_error=exc)

        with pytest.raises(type(exc)):
            await filter_preservation_specimen_flags(
                memory_service=memory_service, project_id=PROJECT, flags=[_stranded_flag()],
            )

    @pytest.mark.asyncio
    async def test_one_tasks_failure_does_not_abort_the_batch(self):
        """(e) A failure costs one task's verdict, not the cycle's.

        Without this, a single flaky lookup would leave every later flag
        unfiltered — and the corroborated specimen's destructive recommendation
        would sail through on a cycle that merely had bad luck earlier.
        """
        def _scroll(project_id, filters):
            if filters['task_id'] == '4102':
                raise TimeoutError('only this one')
            return {'3105': [LIVE_MEM0_ROW]}.get(filters['task_id'], [])

        memory_service = MagicMock()
        memory_service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
        memory_service.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})

        broken = _stranded_flag(task_id='4102')
        result = await filter_preservation_specimen_flags(
            memory_service=memory_service,
            project_id=PROJECT,
            flags=[broken, _stranded_flag(task_id='3105')],
        )

        assert result.kept_flags == [broken]
        assert result.suppressed_by_task == {'3105': 1}
        assert result.unresolved_task_ids == ('4102',)
