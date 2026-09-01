"""Tests for the resolved-curator-gate Mem0 source sweep (task 3084).

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that notices when
a human-curator gate task (``metadata.operational_mode == 'gate'``) has in
fact been resolved.  The resolution evidence is already deterministic and
already in Mem0 — the reify curator writes its ruling stamped
``metadata.source == f'curator_gate_{task_id}'`` (independently documented at
``fused-memory/tests/fixtures/README.md``) — but nothing reads it back, so
detection is an ad-hoc Stage-3 spot-check that misses roughly a quarter of
cases (reify run ec45eed0: gates 5561 and 5563 were resolved-but-stale and
went undetected).

Covers:
- curator_gate_source / CURATOR_GATE_SOURCE_TEMPLATE: the single owner of the
  ``curator_gate_{task_id}`` source-key spelling — the one load-bearing
  string of this task.
- extract_open_gate_task_ids: pure selector that reads
  ``FilteredTaskTree.active_tasks`` and returns the sorted, deduped str ids
  of tasks carrying ``metadata.operational_mode == 'gate'``.
- build_gate_resolution_flag: pure builder for the Stage-1 flag dict Stage 2
  consumes to close the resolved gate.
- sweep_resolved_curator_gates: best-effort async orchestrator that counts
  each gate's ``curator_gate_{id}`` Mem0 entries via a deterministic Qdrant
  payload filter and emits one flag per resolved gate.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.cli_stage_runner import FINDING_ITEM_SCHEMA
from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    CURATOR_GATE_SOURCE_TEMPLATE,
    build_gate_resolution_flag,
    curator_gate_source,
    extract_open_gate_task_ids,
    sweep_resolved_curator_gates,
)
from fused_memory.reconciliation.flag_dedup import compute_flag_signature
from fused_memory.reconciliation.task_filter import FilteredTaskTree


class TestCuratorGateSource:
    """curator_gate_source(task_id) owns the ``curator_gate_{task_id}`` spelling.

    Every Mem0 read this module performs filters on exactly this string, so a
    divergence here is silently a zero-recall sweep — the failure mode this
    task exists to fix.  These tests pin the spelling and the int coercion.

    They deliberately do NOT assert
    ``CURATOR_GATE_SOURCE_TEMPLATE.format(...) == curator_gate_source(...)``:
    the helper IS that format call, so the assertion is true by construction
    and can never fail while the implementation stands (reviewer finding
    "test-quality", amendment pass).  What is worth pinning is the literal
    wire format itself, once on the template and once on the helper's output.
    """

    def test_str_task_id_yields_curator_gate_key(self):
        """A str task id formats to the exact observed key (reify gate 5561)."""
        assert curator_gate_source('5561') == 'curator_gate_5561', (
            'source key must be the exact curator-written spelling; '
            f'got {curator_gate_source("5561")!r}'
        )

    def test_int_task_id_is_coerced_to_the_same_key(self):
        """An int task id must not silently produce a different key than the str form."""
        assert curator_gate_source(5561) == 'curator_gate_5561', (
            'int task ids must be coerced to str before formatting, else an '
            f'int-typed caller queries a different key; got {curator_gate_source(5561)!r}'
        )
        assert curator_gate_source(5561) == curator_gate_source('5561'), (
            'int and str spellings of the same task id must collapse to one key'
        )

    def test_template_pins_the_exported_wire_format(self):
        """The exported template is the wire format other readers would format against."""
        assert CURATOR_GATE_SOURCE_TEMPLATE == 'curator_gate_{task_id}', (
            f'template spelling drifted; got {CURATOR_GATE_SOURCE_TEMPLATE!r}'
        )


class TestExtractOpenGateTaskIds:
    """extract_open_gate_task_ids(tasks) selects operational_mode == 'gate' tasks.

    Inputs are built as ``FilteredTaskTree(active_tasks=[...])`` and the
    ``active_tasks`` field is passed, mirroring how
    ``extract_stalled_gate_backlog_task_ids`` is exercised — that helper is
    the live proof that ``task['metadata']`` is reachable as a dict on those
    dicts.  Restricting to ``active_tasks`` excludes done/cancelled gates for
    free.  The match is deliberately value-sensitive and exact, mirroring
    ``TaskInterceptor._is_gate_metadata``.
    """

    def test_selects_only_operational_mode_gate_tasks(self):
        """A gate task is selected; a non-gate task alongside it is not."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 5561, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
            {'id': 4242, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['5561'], (
            f'only operational_mode == "gate" tasks may be selected, got {result!r}'
        )

    def test_excludes_missing_empty_and_modeless_metadata(self):
        """No metadata key, empty metadata, and metadata without operational_mode all skip."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'status': 'blocked'},
            {'id': 2, 'status': 'blocked', 'metadata': {}},
            {'id': 3, 'status': 'blocked', 'metadata': {'execution_class': 'operational'}},
            {'id': 4, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['4'], (
            'absent/empty/mode-less metadata must never select a task, got '
            f'{result!r}'
        )

    def test_operational_mode_match_is_value_sensitive(self):
        """operational_mode='llm' is NOT a gate — the match is exact, not truthy."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 7, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
            {'id': 8, 'status': 'pending', 'metadata': {'operational_mode': 'GATE'}},
            {'id': 9, 'status': 'pending', 'metadata': {'operational_mode': None}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == [], (
            'the operational_mode comparison must be an exact == "gate" match '
            f'(mirroring TaskInterceptor._is_gate_metadata), got {result!r}'
        )

    def test_skips_non_dict_metadata_without_raising(self):
        """A str/list/None metadata value is skipped, not fed to .get()."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'metadata': 'operational_mode=gate'},
            {'id': 2, 'metadata': ['gate']},
            {'id': 3, 'metadata': None},
            {'id': 4, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['4'], (
            f'non-dict metadata must be skipped without raising, got {result!r}'
        )

    def test_skips_non_dict_tasks_and_none_ids(self):
        """A non-dict element, and a gate task whose id is None, contribute nothing."""
        tree = FilteredTaskTree(active_tasks=[
            'not-a-dict',  # type: ignore[list-item]
            None,  # type: ignore[list-item]
            {'id': None, 'metadata': {'operational_mode': 'gate'}},
            {'metadata': {'operational_mode': 'gate'}},
            {'id': 12, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['12'], (
            'non-dict tasks and gates with a missing/None id must be skipped — a '
            f'spurious id would query the wrong source key; got {result!r}'
        )

    def test_coerces_int_ids_and_returns_sorted_deduped(self):
        """int ids coerce to str; the result is sorted and deduped."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 5563, 'metadata': {'operational_mode': 'gate'}},
            {'id': '5561', 'metadata': {'operational_mode': 'gate'}},
            {'id': 5561, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_open_gate_task_ids(tree.active_tasks)

        assert result == ['5561', '5563'], (
            'int and str spellings of one id must collapse, and the result must '
            f'be sorted; got {result!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """No active tasks -> []."""
        assert extract_open_gate_task_ids(FilteredTaskTree().active_tasks) == []
        assert extract_open_gate_task_ids([]) == []


class TestBuildGateResolutionFlag:
    """build_gate_resolution_flag(task_id, memories) emits the Stage-1 flag dict.

    Stage 1 runs under DISALLOW_TASK_WRITES, so it cannot close the gate
    itself — it emits a flag and Stage 2 (which holds set_task_status) acts.
    The flag must therefore carry enough evidence for a Stage-2 reader to
    re-derive the finding deterministically, and must be dedupable so an
    un-actioned gate does not re-emit unmarked every cycle.
    """

    def test_carries_the_stage1_flag_contract_fields(self):
        """flag_type/task_id/category/severity/actionable are the agreed values."""
        flag = build_gate_resolution_flag('5561', [{'id': 'mem-a'}])

        assert flag['flag_type'] == 'task_completed_not_reflected', (
            f'flag_type must be the name the ASK specifies, got {flag["flag_type"]!r}'
        )
        assert flag['task_id'] == '5561' and isinstance(flag['task_id'], str), (
            f'task_id must be a str, got {flag["task_id"]!r}'
        )
        assert flag['category'] == 'task_memory_mismatch', (
            f'category must be a FINDING_ITEM_SCHEMA enum member, got {flag["category"]!r}'
        )
        assert flag['actionable'] is True, 'Stage 2 can act on this, so actionable is True'
        assert flag['severity'] in FINDING_ITEM_SCHEMA['properties']['severity']['enum'], (
            f'severity must be a schema enum member, got {flag["severity"]!r}'
        )
        assert flag['category'] in FINDING_ITEM_SCHEMA['properties']['category']['enum'], (
            f'category must be a schema enum member, got {flag["category"]!r}'
        )

    def test_int_task_id_is_coerced_to_str(self):
        """An int task id becomes a str task_id (dedup compares str-coerced values)."""
        flag = build_gate_resolution_flag(5563, [{'id': 'mem-a'}])

        assert flag['task_id'] == '5563' and isinstance(flag['task_id'], str)

    def test_description_names_the_source_key_and_hit_count(self):
        """A Stage-2 reader can re-derive the evidence from the description alone.

        The count is asserted as the RENDERED PHRASE, not as a bare '2'
        substring: any digit 2 anywhere in the string — a task id, some future
        count — satisfies the bare form, so it would not catch a wrong count
        being rendered (reviewer finding "test-quality", amendment pass).
        """
        flag = build_gate_resolution_flag('5561', [{'id': 'a'}, {'id': 'b'}])

        assert curator_gate_source('5561') in flag['description'], (
            'the description must name the exact metadata.source key so the '
            f'evidence is re-derivable; got {flag["description"]!r}'
        )
        assert '2 Mem0 entries' in flag['description'], (
            f'the description must name the hit count; got {flag["description"]!r}'
        )

    def test_description_uses_the_singular_form_for_one_hit(self):
        """One matching entry renders '1 Mem0 entry', not '1 Mem0 entries'."""
        flag = build_gate_resolution_flag('5561', [{'id': 'a'}])

        assert '1 Mem0 entry' in flag['description'], (
            f'expected the singular rendering; got {flag["description"]!r}'
        )
        assert 'entries' not in flag['description'], (
            f'the plural form must not leak into a single-hit description; got '
            f'{flag["description"]!r}'
        )

    def test_description_claims_only_what_was_observed(self):
        """The text asserts the entries EXIST — never that the gate is resolved.

        The key's producer lives outside this repo and may stamp it for merely
        CURATED (not ruled-on) clusters, and Stage 2 holds set_task_status — so
        an over-claiming description could get a still-open human decision gate
        closed (reviewer finding "correctness-risk", amendment pass).
        """
        flag = build_gate_resolution_flag('5561', [{'id': 'a'}, {'id': 'b'}])

        assert 'already ruled' not in flag['description'], (
            'the description must not assert that the curator ruled — it can only '
            f'report that entries carrying the key exist; got {flag["description"]!r}'
        )
        assert 'not proof' in flag['description'], (
            'the description must mark itself as evidence rather than proof; got '
            f'{flag["description"]!r}'
        )
        assert 'dismiss this flag' in flag['suggested_action'], (
            'the suggested action must offer the dismiss branch for a curated-but-'
            f'unruled cluster; got {flag["suggested_action"]!r}'
        )

    def test_description_names_the_task_title_when_task_given(self):
        """The optional *task* enriches the description with the gate's title."""
        flag = build_gate_resolution_flag(
            '5561', [{'id': 'a'}],
            task={'id': 5561, 'title': 'Gate: adopt the widget policy'},
        )

        assert 'adopt the widget policy' in flag['description'], (
            f'the task title should appear when a task is supplied; got {flag["description"]!r}'
        )

    def test_cited_memories_built_in_order_from_input(self):
        """cited_memories mirrors the input memory dicts in order, store='mem0'."""
        flag = build_gate_resolution_flag('5561', [{'id': 'mem-a'}, {'id': 'mem-b'}])

        assert flag['cited_memories'] == [
            {'memory_id': 'mem-a', 'store': 'mem0'},
            {'memory_id': 'mem-b', 'store': 'mem0'},
        ], f'unexpected cited_memories: {flag["cited_memories"]!r}'

    def test_memories_without_usable_id_contribute_no_citation(self):
        """A memory dict missing/None 'id' is skipped, not raised on."""
        flag = build_gate_resolution_flag(
            '5561', [{'id': None}, {'created_at': 'x'}, {'id': 'mem-c'}],
        )

        assert flag['cited_memories'] == [{'memory_id': 'mem-c', 'store': 'mem0'}], (
            f'unusable memory ids must be skipped silently, got {flag["cited_memories"]!r}'
        )

    def test_flag_has_a_computable_dedup_signature(self):
        """compute_flag_signature returns the (task_id, flag_type) tuple.

        This is what makes the flag dedupe across cycles — get a
        stage1_flag_marker ledger row and honour suppression — instead of
        re-emitting unmarked forever.
        """
        flag = build_gate_resolution_flag('5561', [{'id': 'mem-a'}])

        signature = compute_flag_signature(flag)

        assert signature is not None, (
            'a flag with no computable signature is passed through dedup_flags '
            'unchanged and never gains recurrence history'
        )
        assert signature == ('5561', 'task_completed_not_reflected'), (
            f'unexpected dedup signature: {signature!r}'
        )


def _raise(exc):
    """Raise *exc* from inside a lambda side_effect."""
    raise exc


def _make_memory_service(*, counts=None, memories=None) -> MagicMock:
    """MagicMock memory_service with AsyncMock metadata readers.

    Mirrors ``_make_memory_service`` in test_degenerate_task_node_sweep.py.
    *counts*/*memories* map a ``curator_gate_{id}`` source key to that key's
    count / memory list, defaulting to 0 / [].
    """
    counts = counts or {}
    memories = memories or {}
    memory_service = MagicMock()
    memory_service.count_memories_by_metadata = AsyncMock(
        side_effect=lambda project_id, filters: counts.get(filters.get('source'), 0),
    )
    memory_service.get_memories_by_metadata = AsyncMock(
        side_effect=lambda project_id, filters: memories.get(filters.get('source'), []),
    )
    memory_service.search = AsyncMock(return_value=[])
    return memory_service


class TestSweepResolvedCuratorGates:
    """sweep_resolved_curator_gates emits one flag per gate with curator evidence."""

    @pytest.mark.asyncio
    async def test_zero_count_yields_no_flag_and_no_fetch(self):
        """A gate with no curator entry is left alone — and costs only the count call."""
        memory_service = _make_memory_service(counts={'curator_gate_5561': 0})

        stats = await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

        assert stats['flags'] == [], f'no evidence must mean no flag, got {stats["flags"]!r}'
        assert stats['resolved'] == 0
        memory_service.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_positive_count_yields_one_flag_with_that_tasks_citations(self):
        """A gate with curator entries yields exactly one flag citing those memories."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 2},
            memories={'curator_gate_5561': [{'id': 'mem-a'}, {'id': 'mem-b'}]},
        )

        stats = await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

        assert len(stats['flags']) == 1, f'expected exactly one flag, got {stats["flags"]!r}'
        flag = stats['flags'][0]
        assert flag['task_id'] == '5561'
        assert flag['flag_type'] == 'task_completed_not_reflected'
        assert flag['cited_memories'] == [
            {'memory_id': 'mem-a', 'store': 'mem0'},
            {'memory_id': 'mem-b', 'store': 'mem0'},
        ]

    @pytest.mark.asyncio
    async def test_count_filter_is_the_exact_source_key_and_never_semantic_search(self):
        """The count is a deterministic payload-filter read, not a semantic search."""
        memory_service = _make_memory_service(counts={'curator_gate_5563': 0})

        await sweep_resolved_curator_gates(memory_service, 'reify', ['5563'])

        memory_service.count_memories_by_metadata.assert_awaited_once_with(
            project_id='reify', filters={'source': 'curator_gate_5563'},
        )
        memory_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filter_contains_only_the_source_key(self):
        """No task_id key is ANDed in — Qdrant ANDs conditions, so it can only lose recall.

        A curator entry whose writer omitted metadata.task_id would be missed,
        and the source key already encodes the id, so the extra condition buys
        nothing. Missing a resolved gate is the exact failure this sweep exists
        to fix.
        """
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 1},
            memories={'curator_gate_5561': [{'id': 'mem-a'}]},
        )

        await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

        for call in (
            *memory_service.count_memories_by_metadata.await_args_list,
            *memory_service.get_memories_by_metadata.await_args_list,
        ):
            filters = call.kwargs['filters']
            assert set(filters) == {'source'}, (
                f'the payload filter must contain ONLY the source key, got {filters!r}'
            )

    @pytest.mark.asyncio
    async def test_returns_counter_dict_with_flags_matching_resolved(self):
        """Stats shape is {'flags', 'scanned', 'resolved', 'errors'}; len(flags) == resolved."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 1, 'curator_gate_5563': 3, 'curator_gate_9999': 0},
            memories={
                'curator_gate_5561': [{'id': 'mem-a'}],
                'curator_gate_5563': [{'id': 'mem-b'}, {'id': 'mem-c'}, {'id': 'mem-d'}],
            },
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563', '9999'],
        )

        assert set(stats) == {'flags', 'scanned', 'resolved', 'errors'}, (
            f'unexpected stats shape: {sorted(stats)!r}'
        )
        assert stats['scanned'] == 3
        assert stats['resolved'] == 2
        assert stats['errors'] == 0
        assert len(stats['flags']) == stats['resolved']
        assert {f['task_id'] for f in stats['flags']} == {'5561', '5563'}

    @pytest.mark.asyncio
    async def test_tasks_by_id_puts_the_gate_title_in_the_emitted_flag(self):
        """The optional map threads the gate's title into the emitted description.

        Without it, build_gate_resolution_flag's title branch is unreachable in
        production — extract_open_gate_task_ids returns bare ids (reviewer
        finding "dead-code", amendment pass).
        """
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 1},
            memories={'curator_gate_5561': [{'id': 'mem-a'}]},
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561'],
            tasks_by_id={'5561': {'id': 5561, 'title': 'Gate: adopt the widget policy'}},
        )

        assert 'adopt the widget policy' in stats['flags'][0]['description'], (
            'the mapped task title must reach the emitted flag; got '
            f'{stats["flags"][0]["description"]!r}'
        )

    @pytest.mark.asyncio
    async def test_missing_or_absent_tasks_by_id_entry_still_flags(self):
        """A partial/None map only costs the title — it never changes what is flagged."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 1},
            memories={'curator_gate_5561': [{'id': 'mem-a'}]},
        )

        partial = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561'], tasks_by_id={'9999': {'title': 'other'}},
        )
        none_map = await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

        for stats in (partial, none_map):
            assert [f['task_id'] for f in stats['flags']] == ['5561'], (
                'the enrichment map must never gate flag emission; got '
                f'{stats["flags"]!r}'
            )
        assert 'other' not in partial['flags'][0]['description'], (
            'a non-matching map entry must not leak into the description; got '
            f'{partial["flags"][0]["description"]!r}'
        )

    @pytest.mark.asyncio
    async def test_empty_task_ids_short_circuits_with_no_backend_calls(self):
        """No gates -> zero stats, empty flag list, and not one backend round trip."""
        memory_service = _make_memory_service()

        stats = await sweep_resolved_curator_gates(memory_service, 'reify', [])

        assert stats == {'flags': [], 'scanned': 0, 'resolved': 0, 'errors': 0}
        memory_service.count_memories_by_metadata.assert_not_awaited()
        memory_service.get_memories_by_metadata.assert_not_awaited()


class TestSweepResolvedCuratorGatesFailSafe:
    """A backend failure is tallied, never mistaken for evidence either way.

    The fail-safe direction is asymmetric and deliberate: an errored read must
    never be recorded as "this gate is resolved", because acting on that flag
    would close a still-open human decision gate. It must also never abort the
    sweep for the remaining gates.
    """

    @pytest.mark.asyncio
    async def test_count_failure_tallies_error_and_loop_continues(self):
        """One gate's count raising leaves it unflagged; a later healthy gate still flags."""
        memory_service = _make_memory_service(
            memories={'curator_gate_5563': [{'id': 'mem-b'}]},
        )
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=lambda project_id, filters: (
                _raise(RuntimeError('qdrant down'))
                if filters['source'] == 'curator_gate_5561'
                else 1
            ),
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563'],
        )

        assert stats['errors'] == 1, f'the failed count must be tallied, got {stats!r}'
        assert stats['scanned'] == 2
        assert stats['resolved'] == 1, 'an errored read is never "resolved"'
        assert [f['task_id'] for f in stats['flags']] == ['5563'], (
            f'the loop must continue past the failure, got {stats["flags"]!r}'
        )

    @pytest.mark.asyncio
    async def test_count_timeout_is_not_read_as_no_curator_entry(self):
        """A Qdrant read-timeout PROPAGATES out of count_memories_by_metadata.

        It is not returned as 0, so it must be caught as an error here rather
        than silently recorded as "no curator entry" (or as "resolved").
        """
        memory_service = _make_memory_service(
            memories={'curator_gate_5563': [{'id': 'mem-b'}]},
        )
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=lambda project_id, filters: (
                _raise(TimeoutError('qdrant read timed out'))
                if filters['source'] == 'curator_gate_5561'
                else 1
            ),
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563'],
        )

        assert stats['errors'] == 1
        assert stats['resolved'] == 1
        assert [f['task_id'] for f in stats['flags']] == ['5563']

    @pytest.mark.asyncio
    async def test_fetch_failure_after_positive_count_emits_no_flag(self):
        """Uncertain evidence must never become a "this gate is resolved" claim."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 2, 'curator_gate_5563': 1},
            memories={'curator_gate_5563': [{'id': 'mem-b'}]},
        )
        memory_service.get_memories_by_metadata = AsyncMock(
            side_effect=lambda project_id, filters: (
                _raise(RuntimeError('scroll failed'))
                if filters['source'] == 'curator_gate_5561'
                else [{'id': 'mem-b'}]
            ),
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563'],
        )

        assert stats['errors'] == 1, f'the failed fetch must be tallied, got {stats!r}'
        assert [f['task_id'] for f in stats['flags']] == ['5563'], (
            'a gate whose citation fetch failed must not be flagged resolved, got '
            f'{stats["flags"]!r}'
        )

    @pytest.mark.asyncio
    async def test_count_scroll_divergence_emits_no_flag(self):
        """count > 0 but the scroll returns [] must NOT become a citation-less flag.

        The deletion/TOCTOU race (entry deleted or GC'd between the two reads, or
        a count answered from a stale segment) is not an exception, so without an
        explicit guard it falls through to a flag whose description reads "0 Mem0
        entries ... exist" with an EMPTY cited_memories — the same uncitable
        claim the fetch-failure branch exists to prevent, handed to a stage that
        holds set_task_status (reviewer finding "robustness", amendment pass).
        """
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 2, 'curator_gate_5563': 1},
            memories={'curator_gate_5563': [{'id': 'mem-b'}]},
        )

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563'],
        )

        assert [f['task_id'] for f in stats['flags']] == ['5563'], (
            'a gate is only flagged when at least one citable memory was actually '
            f'read; got {stats["flags"]!r}'
        )
        assert stats['resolved'] == 1, (
            f'a divergent read is never counted as resolved; got {stats!r}'
        )
        assert stats['errors'] == 1, (
            'the divergence is an anomalous read, not a clean "no evidence" — it '
            f'must be tallied so it is visible in report.stats; got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_divergence_never_emits_an_empty_citation_flag(self):
        """The single-gate case: divergence yields no flag at all, not an empty one."""
        memory_service = _make_memory_service(counts={'curator_gate_5561': 3})

        stats = await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

        assert stats['flags'] == [], f'expected no flag whatsoever, got {stats["flags"]!r}'
        assert stats['resolved'] == 0
        assert stats['errors'] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
    async def test_cancellation_from_count_is_reraised(self, exc):
        """CancelledError/KeyboardInterrupt/SystemExit are never swallowed as best-effort."""
        memory_service = _make_memory_service()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=exc)

        with pytest.raises(exc):
            await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
    async def test_cancellation_from_fetch_is_reraised(self, exc):
        """Same for the citation fetch after a positive count."""
        memory_service = _make_memory_service(counts={'curator_gate_5561': 1})
        memory_service.get_memories_by_metadata = AsyncMock(side_effect=exc)

        with pytest.raises(exc):
            await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'])


class TestSweepZeroRecallCanary:
    """A clean sweep that matched nothing must be greppable, not silent.

    ``scanned=N, flags_emitted=0`` is byte-identical whether no gate happened to
    be resolved or the ``curator_gate_{task_id}`` spelling does not match what
    the curator actually writes (its producer lives outside this repo).  The
    second case would make this sweep permanently zero-recall — exactly the
    silent miss it was built to fix — so it is logged (reviewer finding
    "correctness-risk", amendment pass).
    """

    @pytest.mark.asyncio
    async def test_scanned_but_matched_nothing_logs_the_probed_key_shape(self):
        """A clean, empty-handed sweep warns and names the key format it probed."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 0, 'curator_gate_5563': 0},
        )
        log = MagicMock()

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561', '5563'], log=log,
        )

        assert stats == {'flags': [], 'scanned': 2, 'resolved': 0, 'errors': 0}
        assert log.warning.call_count == 1, (
            'a scanned-but-matched-nothing cycle must emit exactly one canary '
            f'warning; got {log.warning.call_args_list!r}'
        )
        rendered = log.warning.call_args.args[0] % log.warning.call_args.args[1:]
        assert CURATOR_GATE_SOURCE_TEMPLATE in rendered, (
            'the canary must name the probed key shape so a format drift is '
            f'greppable; got {rendered!r}'
        )

    @pytest.mark.asyncio
    async def test_no_canary_when_something_matched(self):
        """A sweep that found evidence is not zero-recall — no canary."""
        memory_service = _make_memory_service(
            counts={'curator_gate_5561': 1},
            memories={'curator_gate_5561': [{'id': 'mem-a'}]},
        )
        log = MagicMock()

        await sweep_resolved_curator_gates(memory_service, 'reify', ['5561'], log=log)

        log.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_canary_when_the_sweep_errored(self):
        """An errored sweep matched nothing for a KNOWN reason already logged."""
        memory_service = _make_memory_service()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=lambda project_id, filters: _raise(RuntimeError('qdrant down')),
        )
        log = MagicMock()

        stats = await sweep_resolved_curator_gates(
            memory_service, 'reify', ['5561'], log=log,
        )

        assert stats['errors'] == 1
        log.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_canary_when_there_were_no_gates_to_scan(self):
        """Zero open gates is not evidence about the key format either way."""
        log = MagicMock()

        await sweep_resolved_curator_gates(_make_memory_service(), 'reify', [], log=log)

        log.warning.assert_not_called()
