"""End-to-end tests for the dispatched-agent briefing ``# Context`` block.

Task 3609 (census R5): memory recalled for a dispatch can carry facts from
another project. This file covers the assembled block: cross-project
filtering, the query table, rendering, degradation, the outage streak and the
standing provenance caveat that covers the untagged leak channel the filter
cannot reach on its own. The pure parse, filter and render functions are
tested directly in ``test_memory_recall.py``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _briefing_helpers import (
    _edge,
    _entity_envelope,
    _grouped_parent,
    _mcp_search_envelope,
    _node,
    _result,
    _search_arguments,
    briefing,  # noqa: F401 — re-export: pytest fixture used by test methods
)
from shared.briefing_queries import TASK_SEMANTIC, BriefingScope, queries_for

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.memory_recall import (
    MEMORY_CONTEXT_CAVEAT,
    MemoryFailure,
    MemoryQueryOutcome,
    render_memory_results,
)


def _task_scope(task_id: str = '3609') -> BriefingScope:
    """A scope standing in for a real dispatch — an id, a title and declared files.

    Every end-to-end test below drives ``_get_memory_context`` through this,
    because the query table asks what a dispatch is ABOUT: a scope carrying
    only an id yields no area terms and so fires the single generic
    conventions query instead of the two-query task-scoped set.
    """
    return BriefingScope.from_task({
        'id': task_id,
        'title': 'Project-scope the dispatched-agent briefing context block',
        'metadata': {'files': ['orchestrator/src/orchestrator/agents/briefing.py']},
    })


@pytest.mark.asyncio
class TestGetMemoryContextFiltersForeignFacts:
    """``_get_memory_context`` drops foreign-tagged results end-to-end.

    ``_mcp_search`` is called once per spec the query table selects for the
    scope — for a task-scoped dispatch, the area-conventions and
    task-semantic pair (task 3659: the four hardcoded queries are gone). The
    stub answers every call identically, so a single foreign result per query
    yields a filtered count of 2 in the assembled block, not 1 — the count
    must reflect both queries, not just one.
    """

    async def test_filters_foreign_facts_and_announces_the_drop(
        self, briefing: BriefingAssembler, caplog,
    ):
        envelope = _mcp_search_envelope([
            _result(
                '1',
                "A park on 'crates/reify-compiler/src' blocks acquire of "
                "'crates/reify-compiler'.",
                metadata={'project_id': 'reify'},
            ),
            _result(
                '2',
                'Own project fact about dark_factory.',
                metadata={'project_id': 'dark_factory'},
            ),
        ])
        foreign_per_query = 1
        queries_fired = 2  # area conventions + task-semantic (the scope declares files)
        expected_dropped = foreign_per_query * queries_fired

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert context.splitlines()[0] == '# Context'
        assert 'Own project fact about dark_factory.' in context
        assert 'crates/reify-compiler' not in context
        assert "A park on 'crates/reify-compiler/src'" not in context
        # The message names BOTH numbers (slots and queries) rather than
        # just `expected_dropped` — one distinct foreign fact matching every
        # query must not read as N distinct "memory results", which would
        # overstate the leak volume by the number of queries fired (task
        # 3609 amendment).
        assert (
            f'{expected_dropped} memory result slot(s) across {queries_fired} queries were '
            'tagged to another project and filtered out'
        ) in context
        assert 'dark_factory' in caplog.text
        assert 'filtered' in caplog.text.lower()
        assert any(r.levelno == logging.INFO for r in caplog.records)

    async def test_all_foreign_results_surface_drop_count_in_empty_context(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The most consequential outcome — every recalled result is
        foreign — must not degrade silently: the drop count is visible in
        both the rendered message and the INFO log, not just discarded
        along with the (empty) recalled sections.
        """
        envelope = _mcp_search_envelope([
            _result('1', 'Foreign fact.', metadata={'project_id': 'reify'}),
        ])

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert context.splitlines()[0] == '# Context'
        assert 'Foreign fact.' not in context
        assert '_No memory context available' in context
        assert '2 memory result slot(s) across 2 queries' in context
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert 'filtered' in caplog.text.lower()

    async def test_a_nested_only_drop_is_reported_as_nested_records(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The rendered note must not call a dropped CHILD a dropped result slot.

        Task 4008 amendment. Here every top-level result survives and only
        children inside them are removed, so a note reading "N memory result
        slot(s) ... filtered out" would tell an operator that N recalled facts
        vanished when in fact none did. The note is the ONLY signal a human
        gets that a leak was blocked, so it names the two quantities apart.
        """
        envelope = _mcp_search_envelope([_grouped_parent()])
        # One foreign amendment + one foreign pinned body per query, and the
        # stub answers both queries identically.
        expected_nested = 2 * 2

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'FOREIGN AMENDMENT BODY' not in context
        assert 'FOREIGN SIGHTING BODY' not in context
        assert 'Native canonical.' in context, (
            'The parent itself was never foreign and must still be recalled'
        )
        assert (
            f'{expected_nested} nested memory record(s) across 2 queries were '
            'tagged to another project and filtered out'
        ) in context, (
            f'a nested-only drop must be reported as nested records, got {context!r}'
        )
        assert 'memory result slot(s)' not in context, (
            'No top-level result was dropped, so the note must not claim a '
            f'result slot was vacated, got {context!r}'
        )
        assert any(r.levelno == logging.INFO for r in caplog.records)

    async def test_mixed_drops_name_both_quantities_separately(
        self, briefing: BriefingAssembler, caplog,
    ):
        """A foreign result AND a foreign child are reported as distinct counts."""
        envelope = _mcp_search_envelope([
            _result('f1', 'Foreign fact.', metadata={'project_id': 'reify'}),
            _grouped_parent(),
        ])

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'Foreign fact.' not in context
        assert 'FOREIGN AMENDMENT BODY' not in context
        assert (
            '2 memory result slot(s) and 4 nested memory record(s) across 2 '
            'queries were tagged to another project and filtered out'
        ) in context, (
            f'both quantities must be named, and named apart, got {context!r}'
        )


@pytest.mark.asyncio
class TestQueryTableComesFromTheSharedSpecs:
    """The queries fired are the ones ``shared.briefing_queries`` declares.

    Task 3659 (PRD lane β, D1/D2/D3/D8/D9). The assembler used to spell four
    queries inline at ``limit=5`` with no store, category or caller identity.
    Every assertion here reads the expected values back OUT of the shared
    specs rather than re-spelling them, so a template reworded in ``shared``
    cannot leave this file passing against a stale copy.
    """

    async def test_a_task_scoped_dispatch_fires_exactly_two_searches(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(_task_scope(), 'implementer')

        fired = [args['query'] for args in _search_arguments(mcp)]
        assert fired == [text for _spec, text in queries_for(_task_scope())]

    async def test_every_search_carries_its_spec_s_scoping(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(_task_scope(), 'implementer')

        by_query = {args['query']: args for args in _search_arguments(mcp)}
        for spec, text in queries_for(_task_scope()):
            args = by_query[text]
            assert args['limit'] == spec.limit
            assert tuple(args.get('stores', ())) == spec.stores
            assert tuple(args.get('categories', ())) == spec.categories
            assert args['project_id'] == briefing.project_id

    async def test_the_retired_queries_are_asked_by_nobody(
        self, briefing: BriefingAssembler,
    ):
        """D1 retires these two rather than rewording them, so neither may
        survive as a query OR as a rendered section heading."""
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        asked = ' '.join(args['query'] for args in _search_arguments(mcp)).lower()
        for retired in ('project overview architecture goals', 'recent decisions and rationale'):
            assert retired not in asked
        assert '## Project Context' not in context
        assert '## Recent Decisions' not in context

    async def test_every_search_declares_who_is_asking(
        self, briefing: BriefingAssembler,
    ):
        """D8: the journal records the caller, and the string it records is
        the same one the prompt's own ``## Agent Identity`` block declares."""
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(_task_scope(), 'implementer')

        arguments = _search_arguments(mcp)
        assert arguments
        for args in arguments:
            assert args['caller_agent_id'] == 'claude-task-3609-implementer'
            assert args['caller_task_id'] == '3609'
        assert 'claude-task-3609-implementer' in briefing._agent_identity('3609', 'implementer')

    # A task-less dispatch's query table, caller identity and absent
    # caller_task_id are pinned one level up, through the public builder
    # that actually has task-less callers, by
    # test_briefing.py::TestPerRoleMemoryTable::
    # test_the_reviewer_still_builds_without_a_task.


@pytest.mark.asyncio
class TestDistilledRendering:
    """Recalled memory renders as markdown bullets, never as raw JSON (D5).

    Task 3659. The block used to carry the search payload verbatim — braces,
    ids, relevance scores, provenance uuids — which measured ~85% envelope by
    token. Each surviving result now renders as one
    ``- [category · date · store] content`` bullet, with the content WHOLE:
    fidelity over budget, bounded by ``limit=5`` and by query scoping rather
    than by a per-entry cap.
    """

    async def _render(self, briefing: BriefingAssembler, results: list[dict]) -> str:
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value=_mcp_search_envelope(results)),
        ):
            return await briefing._get_memory_context(_task_scope(), 'implementer')

    async def test_a_mem0_result_renders_as_a_tagged_bullet(
        self, briefing: BriefingAssembler,
    ):
        entry = _result('1', 'Never run git stash in any dark-factory checkout.', source_store='mem0')
        entry['category'] = 'preferences_and_norms'
        entry['created_at'] = '2026-08-15T22:22:49+00:00'

        context = await self._render(briefing, [entry])

        assert (
            '- [preferences_and_norms · 2026-08-15 · mem0] '
            'Never run git stash in any dark-factory checkout.'
        ) in context

    async def test_no_raw_json_survives_anywhere_in_the_block(
        self, briefing: BriefingAssembler,
    ):
        entry = _result('1', 'A recalled fact.', source_store='mem0')
        entry['category'] = 'observations_and_summaries'
        entry['created_at'] = '2026-08-15T22:22:49+00:00'

        context = await self._render(briefing, [entry])

        assert '{' not in context and '}' not in context
        assert 'relevance_score' not in context
        assert 'provenance' not in context

    async def test_a_long_entry_renders_whole(self, briefing: BriefingAssembler):
        """D5 is explicit that entries are not capped: a canonical memory
        record is long precisely because its reasoning is the payload."""
        long_content = 'The measured rationale. ' * 200
        entry = _result('1', long_content.strip(), source_store='mem0')

        context = await self._render(briefing, [entry])

        assert long_content.strip() in context
        assert '...' not in context
        assert 'truncated' not in context

    async def test_a_graphiti_result_falls_back_to_its_valid_at_date(
        self, briefing: BriefingAssembler,
    ):
        """Graphiti-sourced results carry no category and no created_at, but
        may carry a temporal envelope — measured live, not assumed."""
        entry = _result('1', 'An edge fact about task 3659.', source_store='graphiti')
        entry['temporal'] = {'valid_at': '2026-09-14T07:58:01.179808+00:00', 'invalid_at': None}

        context = await self._render(briefing, [entry])

        assert (
            '- [uncategorized · 2026-09-14 · graphiti] An edge fact about task 3659.'
        ) in context

    async def test_the_category_falls_back_to_the_metadata_copy(
        self, briefing: BriefingAssembler,
    ):
        """Mem0 results carry the category on the result AND in metadata; a
        payload that only carries the metadata copy must still be tagged."""
        entry = _result('1', 'A convention.', metadata={'category': 'procedural_knowledge'},
                        source_store='mem0')
        entry['created_at'] = '2026-08-15T22:22:49+00:00'

        context = await self._render(briefing, [entry])

        assert '- [procedural_knowledge · 2026-08-15 · mem0] A convention.' in context

    async def test_an_untagged_undated_result_still_renders_its_content(
        self, briefing: BriefingAssembler,
    ):
        """The tag is best-effort; the content is not. A missing category or
        date renders as a named placeholder rather than as ``None``."""
        context = await self._render(
            briefing, [_result('1', 'A fact with no tags at all.', source_store='mem0')],
        )

        assert '- [uncategorized · undated · mem0] A fact with no tags at all.' in context
        assert 'None' not in context

    async def test_grouped_children_render_as_nested_bullets(
        self, briefing: BriefingAssembler,
    ):
        """An amendment digest reaches the prompt today as nested JSON; it
        must keep reaching it as a nested bullet, or the nested-drop note
        would announce blocking a leak of content nobody renders."""
        context = await self._render(briefing, [_grouped_parent()])

        assert '- [uncategorized · undated · mem0] Native canonical.' in context
        # The child is tagged with its PARENT's store: a nested digest has no
        # source_store of its own, having been collapsed into the parent hit.
        assert '  - [amendment · undated · mem0] NATIVE AMENDMENT BODY' in context
        assert 'FOREIGN AMENDMENT BODY' not in context

    async def test_a_native_matched_child_renders_as_a_nested_bullet(
        self, briefing: BriefingAssembler,
    ):
        """The other half of ``GROUPED_CHILD_KEYS``, and the untested one.

        ``matched_children`` is where a swallowed child's FULL body is pinned
        so its text stays reachable, and it carries ``content`` where an
        amendment digest carries ``digest``. The suite's only
        ``matched_children`` fixture is tagged FOREIGN and so is filtered out
        before rendering, which left dropping the key from the walk entirely
        undetectable.
        """
        parent = _grouped_parent(grouped={
            'matched_children': [
                {'id': 's1', 'content': 'NATIVE SIGHTING BODY', 'created_at': None,
                 'kind': 'sighting', 'matched': True,
                 'metadata': {'project_id': 'dark_factory'}},
            ],
            'sighting_count': 1,
        })

        context = await self._render(briefing, [parent])

        assert '- [uncategorized · undated · mem0] Native canonical.' in context
        assert '  - [sighting · undated · mem0] NATIVE SIGHTING BODY' in context

    async def test_the_section_headings_come_from_the_specs(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, [_result('1', 'A fact.', source_store='mem0')])

        assert '## Conventions & Gotchas' in context
        assert '## Task Context' in context
        for spec, _text in queries_for(_task_scope()):
            assert f'## {spec.section_title}' in context


@pytest.mark.asyncio
class TestTaskEntityChannel:
    """The second half of D3's dual-channel task context.

    ``get_entity`` is asked for ``Task <id>`` alongside the semantic search.
    Its documented fuzzy fallback answers a miss with a DIFFERENT entity —
    measured, a request for one task number returning a neighbouring one — so
    the reply is admitted only on exact name equality. The guard is
    deliberately client-side: the PRD puts server-side fuzzy-path changes out
    of scope.
    """

    async def _render(
        self, briefing: BriefingAssembler, entity_reply: dict,
    ) -> str:
        """Answer the searches with one fact and the entity call with *entity_reply*."""
        search_reply = _mcp_search_envelope([_result('1', 'A recalled fact.', source_store='mem0')])

        async def dispatch(_url, _method, params, **_kwargs):
            return entity_reply if params['name'] == 'get_entity' else search_reply

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(side_effect=dispatch)):
            return await briefing._get_memory_context(_task_scope(), 'implementer')

    async def test_an_exactly_named_node_renders_its_summary_and_edges(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, _entity_envelope(
            nodes=[_node('Task 3609', 'Project-scopes the briefing context block.')],
            edges=[_edge('Task 3609 is related to task 3212.')],
        ))

        assert 'Project-scopes the briefing context block.' in context
        assert 'Task 3609 is related to task 3212.' in context
        task_section = context.split('## Task Context')[1]
        assert 'Task 3609 is related to task 3212.' in task_section

    async def test_the_entity_is_asked_for_by_exact_task_name(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(_task_scope(), 'implementer')

        entity_calls = [
            call.args[2]['arguments']
            for call in mcp.await_args_list
            if call.args[2].get('name') == 'get_entity'
        ]
        assert len(entity_calls) == 1
        assert entity_calls[0]['name'] == 'Task 3609'
        assert entity_calls[0]['project_id'] == briefing.project_id

    async def test_a_fuzzy_neighbour_is_rendered_by_nothing(
        self, briefing: BriefingAssembler,
    ):
        """The wrong-neighbour hazard: a miss answers with another task's
        node, whose facts would otherwise be read as this task's own."""
        context = await self._render(briefing, _entity_envelope(
            nodes=[_node('Task 3212', 'A DIFFERENT TASK ENTIRELY.')],
            edges=[_edge('Task 3212 threads caller identity through search.')],
        ))

        assert 'A DIFFERENT TASK ENTIRELY.' not in context
        assert 'Task 3212 threads caller identity' not in context
        assert 'A recalled fact.' in context, 'the semantic channel is unaffected'

    async def test_an_empty_node_list_renders_nothing_and_does_not_raise(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, _entity_envelope(nodes=[], edges=[]))

        assert '## Task Context' in context
        assert 'A recalled fact.' in context

    async def test_a_dateless_edge_renders_its_fact_alone(
        self, briefing: BriefingAssembler,
    ):
        """Measured live: every edge of a queried task node carried
        ``temporal: null``, so a date must never be required."""
        context = await self._render(briefing, _entity_envelope(
            nodes=[_node('Task 3609')],
            edges=[_edge('Task 3609 is related to task 3212.')],
        ))

        assert 'Task 3609 is related to task 3212.' in context
        assert 'None' not in context

    async def test_a_dated_edge_renders_its_date_alongside_the_fact(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, _entity_envelope(
            nodes=[_node('Task 3609')],
            edges=[
                _edge('Task 3609 landed as commit abc123.', valid_at='2026-09-14T07:58:01.179808+00:00'),
                _edge('Task 3609 is related to task 3212.'),
            ],
        ))

        task_section = context.split('## Task Context')[1]
        assert '2026-09-14' in task_section
        assert 'Task 3609 landed as commit abc123.' in task_section
        assert 'Task 3609 is related to task 3212.' in task_section

    async def test_a_graph_outage_is_named_not_rendered_as_an_empty_graph(
        self, briefing: BriefingAssembler,
    ):
        """D6/INV-2 applied to the second channel.

        ``get_entity`` answers a Graphiti fault with the SAME fault-only
        ``degraded``/``failed_stores`` keys the search channel uses, and the
        rendered result of a fault is byte-identical to the rendered result
        of a graph that simply holds nothing about this task — the exact
        indistinguishability the search half of this change exists to end.
        """
        from orchestrator.agents.memory_recall import (
            ENTITY_CHANNEL_SUFFIX,
            MEMORY_DEGRADED_STORES_NOTICE,
        )

        degraded = _entity_envelope(nodes=[], edges=[])
        payload = json.loads(degraded['result']['content'][0]['text'])
        payload.update({'degraded': True, 'failed_stores': ['graphiti']})
        degraded['result']['content'][0]['text'] = json.dumps(payload)

        context = await self._render(briefing, degraded)
        healthy = await self._render(briefing, _entity_envelope(nodes=[], edges=[]))

        assert MEMORY_DEGRADED_STORES_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX, stores='graphiti',
        ) in context
        assert context != healthy, (
            'a failed graph and an empty graph must not render identically'
        )

    async def test_an_unreachable_graph_names_its_channel_and_reason(
        self, briefing: BriefingAssembler,
    ):
        """A raising ``get_entity`` used to return ``None`` with a bare
        WARNING, so a total entity-channel outage reached neither the prompt
        nor the section notices."""
        from orchestrator.agents.memory_recall import (
            ENTITY_CHANNEL_SUFFIX,
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        search_reply = _mcp_search_envelope([_result('1', 'A recalled fact.', source_store='mem0')])

        async def dispatch(_url, _method, params, **_kwargs):
            if params['name'] == 'get_entity':
                raise ConnectionError('graph unreachable')
            return search_reply

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(side_effect=dispatch)):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX,
            reason=MemoryFailure.TRANSPORT.value,
        ) in context
        assert 'A recalled fact.' in context, 'the semantic channel still renders'

    async def test_a_graph_tool_error_is_named_and_never_rendered(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The shared ``isError`` blind spot, on the channel that also has it."""
        from orchestrator.agents.memory_recall import (
            ENTITY_CHANNEL_SUFFIX,
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with caplog.at_level(logging.DEBUG):
            context = await self._render(briefing, {'result': {
                'isError': True,
                'content': [{'type': 'text', 'text': 'Error: graph unavailable'}],
            }})

        assert 'graph unavailable' not in context
        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX,
            reason=MemoryFailure.MALFORMED.value,
        ) in context
        assert any(
            'graph unavailable' in r.getMessage() and r.levelno >= logging.WARNING
            for r in caplog.records
        )

    async def test_the_two_channels_of_one_section_name_themselves_apart(
        self, briefing: BriefingAssembler,
    ):
        """Both answer ``## Task Context``, so an unqualified notice would
        print the same sentence twice and name neither corpus."""
        from orchestrator.agents.memory_recall import (
            ENTITY_CHANNEL_SUFFIX,
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=ConnectionError('everything is down')),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        semantic = MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context', reason='transport',
        )
        graph = MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX, reason='transport',
        )
        assert semantic in context and graph in context
        assert context.count(semantic) == 1, 'no duplicated, unattributable line'

    async def test_the_graph_channel_is_gated_on_the_spec_field(
        self, briefing: BriefingAssembler,
    ):
        """The loop asks ``wants_entity_block``, not "is this spec TASK_SEMANTIC".

        A slug comparison couples two independent dimensions: which question
        a spec asks, and which renderer the answer needs.
        """
        assert TASK_SEMANTIC.wants_entity_block
        fired = [spec.wants_entity_block for spec, _text in queries_for(_task_scope())]
        assert fired.count(True) == 1

        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))
        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(_task_scope(), 'implementer')

        assert len([
            call for call in mcp.await_args_list
            if call.args[2].get('name') == 'get_entity'
        ]) == 1

    async def test_a_task_less_dispatch_asks_for_no_entity(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_mcp_search_envelope([_result('1', 'A fact.')]))

        with patch('orchestrator.agents.briefing.mcp_call', new=mcp):
            await briefing._get_memory_context(BriefingScope(), 'reviewer')

        assert not [
            call for call in mcp.await_args_list
            if call.args[2].get('name') == 'get_entity'
        ]


@pytest.mark.asyncio
class TestDegradationIsLoud:
    """A memory outage is reported, not silently rendered as "nothing known".

    Task 3659 (PRD lane β, D6 / INV-2). ``_mcp_search`` used to swallow every
    exception at DEBUG and return ``None``, which is indistinguishable from
    "the corpus holds nothing" — so the honest outage branch below it was
    unreachable, and a live transient server error was observed being masked
    as an empty corpus across 234 briefings.
    """

    def _dispatch(self, *, failing_slug: str | None = None, payload: dict | None = None):
        """Answer every call with *payload*, except the query for *failing_slug*."""
        payload = payload if payload is not None else {
            'results': [_result('1', 'A recalled fact.', source_store='mem0')],
        }
        reply = {'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]}}
        failing = {
            text for spec, text in queries_for(_task_scope()) if spec.slug == failing_slug
        }

        async def dispatch(_url, _method, params, **_kwargs):
            if params['arguments'].get('query') in failing:
                raise ConnectionError('memory service unreachable')
            return reply

        return dispatch

    async def test_a_failed_query_names_its_missing_section_and_warns(
        self, briefing: BriefingAssembler, caplog,
    ):
        from orchestrator.agents.memory_recall import MEMORY_SECTION_FAILURE_NOTICE

        with caplog.at_level(logging.DEBUG), patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._dispatch(failing_slug='briefing-task-semantic')),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert '**Task Context**' in context
        assert MEMORY_SECTION_FAILURE_NOTICE.split('{')[0] in context
        assert 'transport' in context

        assert '## Conventions & Gotchas' in context, 'the healthy query still renders'
        assert 'A recalled fact.' in context

        failure_logs = [r for r in caplog.records if 'memory service unreachable' in r.getMessage()]
        assert failure_logs, 'the transport failure must reach the log'
        assert all(r.levelno >= logging.WARNING for r in failure_logs), (
            f'a swallowed search must not be a DEBUG line, got '
            f'{[(r.levelname, r.getMessage()) for r in failure_logs]}'
        )

    async def test_a_degraded_reply_names_the_stores_that_failed(
        self, briefing: BriefingAssembler,
    ):
        """The signal was already on the wire and nobody read it: 701
        briefings carried a ``degraded`` payload that rendered as ordinary
        (silently partial) recall."""
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._dispatch(payload={
                'results': [_result('1', 'A recalled fact.', source_store='mem0')],
                'degraded': True,
                'failed_stores': ['graphiti'],
            })),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        from orchestrator.agents.memory_recall import MEMORY_DEGRADED_STORES_NOTICE

        assert MEMORY_DEGRADED_STORES_NOTICE.split('{')[0] in context, (
            'the store name alone could be incidental prose inside a recalled '
            'memory; the notice is what this test exists to pin'
        )
        assert 'graphiti' in context
        assert 'A recalled fact.' in context
        assert '{' not in context and '}' not in context

    async def test_a_total_outage_reads_differently_from_an_empty_corpus(
        self, briefing: BriefingAssembler,
    ):
        """The defect this closes: both outcomes used to emit the same
        sentence, so an operator reading a briefing could not tell a dead
        memory service (234 briefings) from a corpus with nothing to say
        (77)."""
        from orchestrator.agents.memory_recall import MEMORY_EMPTY_NOTICE, MEMORY_OUTAGE_NOTICE

        assert MEMORY_EMPTY_NOTICE != MEMORY_OUTAGE_NOTICE

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=ConnectionError('memory service unreachable')),
        ):
            outage = await briefing._get_memory_context(_task_scope(), 'implementer')

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'result': {'content': []}}),
        ):
            empty = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert MEMORY_EMPTY_NOTICE in empty
        assert MEMORY_EMPTY_NOTICE not in outage
        assert MEMORY_OUTAGE_NOTICE.split('{')[0] in outage
        assert 'transport' in outage
        assert outage != empty

    async def test_a_drop_note_and_a_failure_notice_both_render(
        self, briefing: BriefingAssembler,
    ):
        """A blocked cross-project leak and a broken query are separate
        facts; neither may displace the other."""
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._dispatch(
                failing_slug='briefing-task-semantic',
                payload={'results': [
                    _result('1', 'Own fact.', metadata={'project_id': 'dark_factory'}),
                    _result('2', 'Foreign fact.', metadata={'project_id': 'reify'}),
                ]},
            )),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'tagged to another project and filtered out' in context
        assert '**Task Context**' in context
        assert 'Own fact.' in context
        assert 'Foreign fact.' not in context

    async def test_a_partial_failure_survives_having_nothing_left_to_render(
        self, briefing: BriefingAssembler,
    ):
        """The notices must not be thrown away just because no section rendered.

        The case the four tests above all miss: one query fails AND the other
        recalls nothing, so ``recalled_sections`` is empty and the early
        return fires. Measured before the fix, that return discarded every
        notice computed for this dispatch and emitted the byte-identical
        healthy-empty sentence — reporting a broken query and a degraded
        store as "the corpus has nothing to say".
        """
        from orchestrator.agents.memory_recall import (
            MEMORY_DEGRADED_STORES_NOTICE,
            MEMORY_EMPTY_NOTICE,
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._dispatch(
                failing_slug='briefing-task-semantic',
                payload={'results': [], 'degraded': True, 'failed_stores': ['graphiti']},
            )),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert MEMORY_SECTION_FAILURE_NOTICE.split('{')[0] in context, (
            'the query that broke must still name itself'
        )
        assert 'transport' in context, 'and its reason class must survive'

        assert MEMORY_DEGRADED_STORES_NOTICE.split('{')[0] in context, (
            'the store outage the server reported must still be named'
        )
        assert 'graphiti' in context

        assert context != f'# Context\n\n{MEMORY_EMPTY_NOTICE}', (
            'this is the measured defect: 234 broken dispatches rendered the '
            'same bytes as 77 genuinely-empty ones'
        )

    async def test_a_total_outage_keeps_its_per_section_notices(
        self, briefing: BriefingAssembler,
    ):
        """The family line and the per-section notices coexist.

        A dispatch where every query broke owes the reader both: the family
        line says "memory is unavailable", the notices say WHICH questions
        went unanswered. Neither may displace the other, and no ``degraded``
        reply was ever seen here — the notices come from the failures alone.
        """
        from orchestrator.agents.memory_recall import (
            MEMORY_DEGRADED_STORES_NOTICE,
            MEMORY_OUTAGE_NOTICE,
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=ConnectionError('memory service unreachable')),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert MEMORY_OUTAGE_NOTICE.split('{')[0] in context
        assert context.index(MEMORY_OUTAGE_NOTICE.split('{')[0]) < context.index(
            MEMORY_SECTION_FAILURE_NOTICE.split('{')[0]
        ), 'the family line leads the block, so the digest markers still match'

        for section in ('Conventions & Gotchas', 'Task Context'):
            assert MEMORY_SECTION_FAILURE_NOTICE.format(
                section=section, reason='transport',
            ) in context, f'{section} went unanswered and must say so'

        assert MEMORY_DEGRADED_STORES_NOTICE.split('{')[0] not in context, (
            'no store outage was reported on the wire, so none is claimed'
        )


    async def test_a_reply_with_no_tool_result_is_named_malformed(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The JSON-RPC error envelope: ``_raw_call`` returns ``{'error': ...}``
        with no ``result`` key at all, so nothing can be read out of it."""
        from orchestrator.agents.memory_recall import (
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with caplog.at_level(logging.DEBUG), patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'error': {'code': -32603, 'message': 'boom'}}),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        for section in ('Conventions & Gotchas', 'Task Context'):
            assert MEMORY_SECTION_FAILURE_NOTICE.format(
                section=section, reason=MemoryFailure.MALFORMED.value,
            ) in context
        assert all(
            r.levelno >= logging.WARNING
            for r in caplog.records if 'no tool result' in r.getMessage()
        )

    async def test_a_non_dict_tool_result_is_named_malformed(
        self, briefing: BriefingAssembler,
    ):
        """Same class, different shape: ``result`` present but not a dict."""
        from orchestrator.agents.memory_recall import (
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'result': ['not', 'a', 'dict']}),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in context

    async def test_a_malformed_dispatch_counts_toward_the_outage_streak(
        self, briefing: BriefingAssembler, caplog,
    ):
        """All three failure classes are outages; only the reason differs."""
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        with caplog.at_level(logging.ERROR), patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'error': {'code': -32603, 'message': 'boom'}}),
        ):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD):
                await briefing._get_memory_context(_task_scope(), 'implementer')

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, errors

    async def test_a_tool_error_is_a_failure_not_a_recalled_fact(
        self, briefing: BriefingAssembler, caplog,
    ):
        """FastMCP reports a tool-level failure IN the body, not by raising.

        The envelope is well-formed and its text block reads exactly like a
        result, so a reader that checks only the SHAPE renders ``Error: ...``
        into the agent's prompt as remembered fact — and counts it as a
        genuine recall, which resets the very outage streak this change adds.
        """
        from orchestrator.agents.memory_recall import (
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        with caplog.at_level(logging.DEBUG), patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'result': {
                'isError': True,
                'content': [{'type': 'text', 'text': 'Error: embeddings backend refused'}],
            }}),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'embeddings backend refused' not in context, (
            'the error prose must never render as recalled memory'
        )
        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in context
        assert briefing._memory_outage_streak == 1, (
            'a tool error recalled nothing, so it must not reset the streak'
        )
        assert any(
            'embeddings backend refused' in r.getMessage() and r.levelno >= logging.WARNING
            for r in caplog.records
        ), 'the error text is owed to the log even though it is kept out of the prompt'

    async def test_a_search_reply_without_a_results_list_is_malformed_not_a_recall(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The server answers a rejected search with a NORMAL, non-isError JSON dict.

        ``{"error": ..., "error_type": ...}`` parses cleanly but carries no
        ``results`` list, so it recalled nothing; rendered as text it would put
        raw JSON in the prompt and reset the outage streak.
        """
        import json

        from orchestrator.agents.memory_recall import (
            MEMORY_SECTION_FAILURE_NOTICE,
        )

        rejection = json.dumps({
            'error': "Invalid project_id 'foo-bar'", 'error_type': 'ValidationError',
        })
        with caplog.at_level(logging.DEBUG), patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value={'result': {
                'content': [{'type': 'text', 'text': rejection}],
            }}),
        ):
            prompt = await briefing.build_implementer_prompt(
                {'steps': []}, task_id='3609',
            )

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in prompt
        assert 'error_type' not in prompt
        assert '{' not in prompt
        assert briefing._memory_outage_streak == 1
        assert any(
            r.levelno >= logging.WARNING and 'results' in r.getMessage()
            for r in caplog.records
        )


@pytest.mark.asyncio
class TestOutageStreakEscape:
    """A sustained memory outage escalates once, above the per-dispatch noise.

    Task 3659 (PRD lane β, INV-4). The per-query WARNING and the in-block
    notice are the base layer: they report one dispatch. Neither says "this
    has now failed N dispatches running", which is the difference between a
    flaky call and an outage nobody has noticed.

    A consecutive STREAK, not a time window: it resets on any success and
    keeps reporting however slowly dispatches arrive (see the module
    constant's own note on why ``shared.storm_counter`` is the wrong shape).
    """

    async def _failing_dispatch(self, briefing: BriefingAssembler) -> str:
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=ConnectionError('memory service unreachable')),
        ):
            return await briefing._get_memory_context(_task_scope(), 'implementer')

    async def _healthy_dispatch(self, briefing: BriefingAssembler) -> str:
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(return_value=_mcp_search_envelope([
                _result('1', 'A recalled fact.', source_store='mem0'),
            ])),
        ):
            return await briefing._get_memory_context(_task_scope(), 'implementer')

    @staticmethod
    def _errors(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]

    async def test_a_short_run_of_outages_does_not_escalate(
        self, briefing: BriefingAssembler, caplog,
    ):
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
                await self._failing_dispatch(briefing)

        assert self._errors(caplog) == []

    async def test_crossing_the_threshold_escalates_exactly_once(
        self, briefing: BriefingAssembler, caplog,
    ):
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD):
                await self._failing_dispatch(briefing)

        errors = self._errors(caplog)
        assert len(errors) == 1, errors
        assert str(MEMORY_OUTAGE_STREAK_THRESHOLD) in errors[0], (
            f'the escalation must name the streak it is reporting, got {errors[0]!r}'
        )

    async def test_a_permanent_outage_re_alarms_on_each_further_crossing(
        self, briefing: BriefingAssembler, caplog,
    ):
        """Runs twice the threshold, which is what separates the implemented
        ``% THRESHOLD`` from ``== THRESHOLD`` and from ``>= THRESHOLD``.

        A single run of exactly N dispatches cannot tell the three apart —
        all emit one ERROR — so the re-alarm the docstring promises ("a
        permanent outage must keep saying so ... without one line per
        dispatch") is unpinned in BOTH directions: ``==`` goes permanently
        silent after the first crossing and ``>=`` shouts once per dispatch
        forever.
        """
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        with caplog.at_level(logging.ERROR):
            for _ in range(2 * MEMORY_OUTAGE_STREAK_THRESHOLD):
                await self._failing_dispatch(briefing)

        errors = self._errors(caplog)
        assert len(errors) == 2, (
            f'one line per crossing — not one per dispatch, and not one ever: {errors}'
        )
        assert f' {MEMORY_OUTAGE_STREAK_THRESHOLD} consecutive ' in errors[0]
        assert f' {2 * MEMORY_OUTAGE_STREAK_THRESHOLD} consecutive ' in errors[1]

    async def test_a_single_success_resets_the_streak(
        self, briefing: BriefingAssembler, caplog,
    ):
        """What makes this a streak and not a burst: recovery clears it, so
        the next run of failures must earn its own escalation from one."""
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
            await self._failing_dispatch(briefing)
        await self._healthy_dispatch(briefing)

        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
                await self._failing_dispatch(briefing)

        assert self._errors(caplog) == []

    async def test_a_dispatch_that_recalled_something_is_not_an_outage(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The loop can break AFTER a section was genuinely recalled — by a
        filter or renderer surprise rather than a per-query fault. That
        dispatch still has memory in its prompt, so it must not count toward
        a streak that means "the memory service is gone"."""
        from orchestrator.agents.memory_recall import MEMORY_OUTAGE_STREAK_THRESHOLD

        async def half_broken(spec, query, **_kwargs):
            if spec.slug == 'briefing-task-semantic':
                raise RuntimeError('the renderer surprised us')
            return MemoryQueryOutcome(rendered=render_memory_results([
                _result('1', 'A recalled fact.', source_store='mem0'),
            ]))

        with caplog.at_level(logging.ERROR), patch.object(
            briefing, '_scoped_search', new=AsyncMock(side_effect=half_broken),
        ):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD + 1):
                context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'A recalled fact.' in context
        assert self._errors(caplog) == []

    async def test_the_per_dispatch_layer_still_reports_every_failure(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The escalation is an added layer, not a replacement: every failing
        dispatch still warns and still says so in its own prompt."""
        from orchestrator.agents.memory_recall import (
            MEMORY_OUTAGE_NOTICE,
            MEMORY_OUTAGE_STREAK_THRESHOLD,
        )

        with caplog.at_level(logging.WARNING):
            blocks = [
                await self._failing_dispatch(briefing)
                for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD)
            ]

        assert all(MEMORY_OUTAGE_NOTICE.split('{')[0] in block for block in blocks)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 2 * MEMORY_OUTAGE_STREAK_THRESHOLD, (
            'each dispatch fires two queries and each failure warns'
        )


@pytest.mark.asyncio
class TestMemoryContextProvenanceCaveat:
    """A standing caveat covers the leak channel the tag filter cannot reach.

    ``filter_foreign_project_results`` only ever fires on Mem0-sourced
    results that carry a project tag. Every Graphiti-sourced result is
    untagged today (metadata == {}), so it survives the filter unclassified
    — the caveat is what converts "agent cd's/reads into a recalled
    'crates/reify-compiler' path" into "agent verifies the path first" for
    that untagged channel.
    """

    async def test_caveat_present_and_precedes_first_section(
        self, briefing: BriefingAssembler,
    ):
        envelope = _mcp_search_envelope([
            _result('1', 'Own project fact.', metadata={'project_id': 'dark_factory'}),
        ])

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert context.splitlines()[0] == '# Context'
        assert briefing.project_id in context

        # Identity check against the actual constant, rather than pinning a
        # handful of substrings from its prose: any rewording that keeps the
        # constant's own text intact still passes, and any drift in what's
        # actually rendered is caught exactly (task 3609 amendment).
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context

        # The caveat must precede the first '## ' section heading. Locate the
        # project_id mention that establishes the caveat is present (the
        # earliest one after the heading) and confirm it comes before that
        # heading — the JSON section body below also names the project, but
        # only as a later occurrence.
        caveat_mention = context.index(briefing.project_id, len('# Context'))
        first_section = context.index('\n## ')
        assert caveat_mention < first_section

    async def test_caveat_absent_when_no_sections_survive(self, briefing: BriefingAssembler):
        envelope = {'result': {'content': []}}

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert context == '# Context\n\n_No memory context available._'

    async def test_caveat_present_alongside_untagged_foreign_path_content(
        self, briefing: BriefingAssembler,
    ):
        """The census failure mode: untagged content naming a foreign path.

        The result carries no project tag at all (metadata == {}), so the
        filter cannot classify — let alone drop — it; it renders verbatim.
        The caveat must still be present so the agent is warned to verify
        the path before treating it as real.
        """
        envelope = _mcp_search_envelope([
            _result(
                '1',
                "A park on 'crates/reify-compiler/src' blocks acquire of "
                "'crates/reify-compiler'.",
                metadata={},
            ),
        ])

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'crates/reify-compiler' in context
        assert briefing.project_id in context
        assert 'verify' in context.lower()

    async def test_caveat_covers_recalled_sections_after_partial_failure(
        self, briefing: BriefingAssembler,
    ):
        """A later query raising must not blank the caveat for sections that
        were already successfully recalled.

        The caveat used to be gated on `memory_unavailable`, which — because
        it is set by a `try`/`except` wrapping every search — suppressed the
        caveat for the WHOLE block even when earlier queries had already
        returned real facts. This patches `_scoped_search` directly (rather
        than `mcp_call`, as the other tests in this module do) so exactly one
        of the two queries fails while the other returns real facts.
        """
        async def scoped_search_side_effect(spec, query, **_kwargs):
            if spec.slug == 'briefing-task-semantic':
                raise TimeoutError('memory service unreachable')
            return MemoryQueryOutcome(rendered=render_memory_results([
                _result('1', f'recalled for: {query}', source_store='mem0'),
            ]))

        with patch.object(
            briefing, '_scoped_search', new=AsyncMock(side_effect=scoped_search_side_effect),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert context.splitlines()[0] == '# Context'
        assert '## Conventions & Gotchas' in context
        assert '## Task Context' not in context
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context


@pytest.mark.asyncio
class TestFailureClassification:
    """A slow memory service reads differently from an unreachable one.

    Task 3659 (review fix 3). ``MemoryFailure`` promises that "a timeout
    says the service is alive and slow, a transport failure says it is
    unreachable" — but ``mcp_call`` re-raises a plain ``RuntimeError`` once
    its retries exhaust, so the ``except TimeoutError`` branch that was meant
    to honour that promise could never fire and every timeout was measured as
    ``transport``. These tests patch ``mcp_call`` to raise exactly what it
    really raises.
    """

    @staticmethod
    def _exhausted(cause: Exception) -> RuntimeError:
        """The wrap ``McpSession._raw_call`` raises on retry exhaustion.

        Mirrors the production f-string so the test fails if that wrap stops
        carrying its cause; the shape itself is pinned against the real
        retry loop in ``test_mcp_retry.py::TestTimeoutCausePredicate``.
        """
        err = RuntimeError(
            f'MCP tools/call failed after 3 attempts: {type(cause).__name__}: {cause}'
        )
        err.__cause__ = cause
        return err

    async def test_an_exhausted_timeout_is_classified_as_a_timeout(
        self, briefing: BriefingAssembler,
    ):
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._exhausted(httpx.ReadTimeout(''))),
        ):
            outcome = await briefing._mcp_search(
                TASK_SEMANTIC, 'anything',
                caller_agent_id='claude-task-3609-implementer', caller_task_id='3609',
            )

        assert outcome.failure is MemoryFailure.TIMEOUT

    async def test_an_exhausted_connect_error_is_classified_as_transport(
        self, briefing: BriefingAssembler,
    ):
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._exhausted(httpx.ConnectError('refused'))),
        ):
            outcome = await briefing._mcp_search(
                TASK_SEMANTIC, 'anything',
                caller_agent_id='claude-task-3609-implementer', caller_task_id='3609',
            )

        assert outcome.failure is MemoryFailure.TRANSPORT

    async def test_a_timeout_reaches_the_reader_as_a_timeout(
        self, briefing: BriefingAssembler,
    ):
        """The reason class is what an operator actually sees in the block."""
        with patch(
            'orchestrator.agents.briefing.mcp_call',
            new=AsyncMock(side_effect=self._exhausted(httpx.ReadTimeout(''))),
        ):
            context = await briefing._get_memory_context(_task_scope(), 'implementer')

        assert 'timeout' in context
        assert MemoryFailure.TRANSPORT.value not in context, (
            'an alive-but-slow service must not be reported as unreachable'
        )


@pytest.mark.asyncio
class TestScopedSearch:
    """Direct coverage of ``_scoped_search``, the seam between
    ``_mcp_search`` and ``_get_memory_context``.
    """

    async def test_an_empty_search_passes_straight_through(
        self, briefing: BriefingAssembler,
    ):
        """Nothing came back, so there is nothing to filter — and nothing
        broke either, which is what distinguishes this from an outage."""
        with patch.object(
            briefing, '_mcp_search', new=AsyncMock(return_value=MemoryQueryOutcome()),
        ):
            outcome = await briefing._scoped_search(
                TASK_SEMANTIC, 'anything',
                caller_agent_id='claude-task-3609-implementer', caller_task_id='3609',
            )

        assert outcome == MemoryQueryOutcome()
        assert outcome.failure is None

    async def test_a_failed_search_keeps_its_reason_class(
        self, briefing: BriefingAssembler,
    ):
        """The filter layer passes a reason class through untouched.

        NOT coverage of how a reason class is CHOSEN: ``_mcp_search`` is
        patched to RETURN a canned outcome here, so no exception is ever
        classified. What the classifier does with a real timeout is pinned by
        ``TestFailureClassification`` below and by
        ``test_mcp_retry.py::TestTimeoutCausePredicate``.
        """
        failed = MemoryQueryOutcome(failure=MemoryFailure.TIMEOUT)

        with patch.object(briefing, '_mcp_search', new=AsyncMock(return_value=failed)):
            outcome = await briefing._scoped_search(
                TASK_SEMANTIC, 'anything',
                caller_agent_id='claude-task-3609-implementer', caller_task_id='3609',
            )

        assert outcome.failure is MemoryFailure.TIMEOUT

    async def test_multi_text_block_response_fails_open_known_limitation(
        self, briefing: BriefingAssembler, caplog,
    ):
        """``_mcp_search`` joins every MCP text block with ``'\\n'`` before
        returning. If the search tool ever answers with more than one text
        block, the joined text is
        not a single valid JSON document, so the filter fails open — the
        cross-project filter silently stops working for that query, though
        the block still renders (nothing is lost from the prompt, just the
        filtering). This test PINS that known limitation so a change to the
        response shape — or a future per-block fix — is caught rather than
        drifting unnoticed.
        """
        envelope = {
            'result': {
                'content': [
                    {'type': 'text', 'text': json.dumps({'results': [
                        _result('1', 'Foreign fact.', metadata={'project_id': 'reify'}),
                    ]})},
                    {'type': 'text', 'text': json.dumps({'results': [
                        _result('2', 'Own fact.', metadata={'project_id': 'dark_factory'}),
                    ]})},
                ],
            },
        }

        with caplog.at_level(logging.WARNING), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            outcome = await briefing._scoped_search(
                TASK_SEMANTIC, 'anything',
                caller_agent_id='claude-task-3609-implementer', caller_task_id='3609',
            )

        # Known limitation: the joined multi-block text isn't valid JSON, so
        # the filter fails open rather than filtering each block — the
        # foreign fact is NOT removed.
        assert outcome.dropped == 0
        assert 'Foreign fact.' in outcome.rendered
        assert 'Own fact.' in outcome.rendered
        assert any(r.levelno == logging.WARNING for r in caplog.records)
