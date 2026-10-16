"""End-to-end tests for the dispatched-agent briefing ``# Context`` block.

Task 3609 (census R5): memory recalled for a dispatch can carry facts from
another project. This file covers the assembled block: cross-project
filtering, the query table, rendering, degradation, the outage streak and the
standing provenance caveat that covers the untagged leak channel the filter
cannot reach on its own. The pure parse, filter, render and compose functions
are tested directly in ``test_memory_recall.py``.

Every test drives a public prompt builder and answers the memory service at
its one transport seam, :func:`_briefing_helpers.memory_transport`, so the
real recall pipeline runs.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

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
    memory_transport,
    recorded_search_envelope,
    recorded_search_text,
)
from shared.briefing_queries import TASK_SEMANTIC, BriefingScope, queries_for

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.memory_recall import (
    ENTITY_CHANNEL_SUFFIX,
    MEMORY_CONTEXT_CAVEAT,
    MEMORY_DEGRADED_STORES_NOTICE,
    MEMORY_EMPTY_NOTICE,
    MEMORY_OUTAGE_NOTICE,
    MEMORY_OUTAGE_STREAK_THRESHOLD,
    MEMORY_SECTION_FAILURE_NOTICE,
    MemoryFailure,
)

_PLAN = {
    'task_id': '3609',
    'title': 'Project-scope the dispatched-agent briefing context block',
    'files': ['orchestrator/src/orchestrator/agents/briefing.py'],
    'steps': [],
}
"""A plan standing in for a real dispatch: an id, a title and declared files.

The query table asks what a dispatch is ABOUT, so a scope carrying only an id
would fire the single generic conventions query instead of the two-query
task-scoped set.
"""

_SCOPE = BriefingScope.from_plan(_PLAN)


async def _prompt(briefing: BriefingAssembler) -> str:
    return await briefing.build_implementer_prompt(_PLAN)


def _context_block(prompt: str) -> str:
    """The ``# Context`` block, which the implementer prompt puts ahead of its identity."""
    return prompt.split('\n\n## Agent Identity', 1)[0]


async def _recall(briefing: BriefingAssembler, transport: AsyncMock) -> str:
    """The ``# Context`` block of one implementer dispatch answered by *transport*."""
    with memory_transport(transport):
        return _context_block(await _prompt(briefing))


def _answering(reply: dict) -> AsyncMock:
    return AsyncMock(return_value=reply)


def _raising(exc: BaseException) -> AsyncMock:
    return AsyncMock(side_effect=exc)


@pytest.mark.asyncio
class TestRecallFiltersForeignFacts:
    """Foreign-tagged results are dropped end-to-end.

    One search is fired per spec the query table selects for the scope — for
    a task-scoped dispatch, the area-conventions and task-semantic pair. The
    stub answers every call identically, so a single foreign result per query
    yields a filtered count of 2 in the assembled block, not 1.
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

        with caplog.at_level(logging.INFO):
            context = await _recall(briefing, _answering(envelope))

        assert context.splitlines()[0] == '# Context'
        assert 'Own project fact about dark_factory.' in context
        assert 'crates/reify-compiler' not in context
        # The note names BOTH numbers (slots and queries): one foreign fact
        # matching every query must not read as N distinct leaked results.
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
        """Every recalled result foreign must not degrade silently: the drop
        count is visible in both the rendered message and the INFO log."""
        envelope = _mcp_search_envelope([
            _result('1', 'Foreign fact.', metadata={'project_id': 'reify'}),
        ])

        with caplog.at_level(logging.INFO):
            context = await _recall(briefing, _answering(envelope))

        assert context.splitlines()[0] == '# Context'
        assert 'Foreign fact.' not in context
        assert '_No memory context available' in context
        assert '2 memory result slot(s) across 2 queries' in context
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert 'filtered' in caplog.text.lower()

    async def test_a_nested_only_drop_is_reported_as_nested_records(
        self, briefing: BriefingAssembler, caplog,
    ):
        """A dropped CHILD is not a dropped result slot (task 4008 amendment).

        Every top-level result survives and only children inside them are
        removed, so the note must not tell an operator that recalled facts
        vanished.
        """
        envelope = _mcp_search_envelope([_grouped_parent()])
        # One foreign amendment + one foreign pinned body per query, and the
        # stub answers both queries identically.
        expected_nested = 2 * 2

        with caplog.at_level(logging.INFO):
            context = await _recall(briefing, _answering(envelope))

        assert 'FOREIGN AMENDMENT BODY' not in context
        assert 'FOREIGN SIGHTING BODY' not in context
        assert 'Native canonical.' in context
        assert (
            f'{expected_nested} nested memory record(s) across 2 queries were '
            'tagged to another project and filtered out'
        ) in context
        assert 'memory result slot(s)' not in context
        assert any(r.levelno == logging.INFO for r in caplog.records)

    async def test_mixed_drops_name_both_quantities_separately(
        self, briefing: BriefingAssembler,
    ):
        envelope = _mcp_search_envelope([
            _result('f1', 'Foreign fact.', metadata={'project_id': 'reify'}),
            _grouped_parent(),
        ])

        context = await _recall(briefing, _answering(envelope))

        assert 'Foreign fact.' not in context
        assert 'FOREIGN AMENDMENT BODY' not in context
        assert (
            '2 memory result slot(s) and 4 nested memory record(s) across 2 '
            'queries were tagged to another project and filtered out'
        ) in context


@pytest.mark.asyncio
class TestQueryTableComesFromTheSharedSpecs:
    """The queries fired are the ones ``shared.briefing_queries`` declares.

    Every assertion reads the expected values back OUT of the shared specs
    rather than re-spelling them, so a template reworded in ``shared`` cannot
    leave this file passing against a stale copy.
    """

    async def test_a_task_scoped_dispatch_fires_exactly_two_searches(
        self, briefing: BriefingAssembler,
    ):
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        await _recall(briefing, mcp)

        fired = [args['query'] for args in _search_arguments(mcp)]
        assert fired == [text for _spec, text in queries_for(_SCOPE)]

    async def test_every_search_carries_its_spec_s_scoping(
        self, briefing: BriefingAssembler,
    ):
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        await _recall(briefing, mcp)

        by_query = {args['query']: args for args in _search_arguments(mcp)}
        for spec, text in queries_for(_SCOPE):
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
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        context = await _recall(briefing, mcp)

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
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        with memory_transport(mcp):
            prompt = await _prompt(briefing)

        arguments = _search_arguments(mcp)
        assert arguments
        for args in arguments:
            assert args['caller_agent_id'] == 'claude-task-3609-implementer'
            assert args['caller_task_id'] == '3609'
        assert 'claude-task-3609-implementer' in prompt.split('## Agent Identity', 1)[1]

    # A task-less dispatch's query table, caller identity and absent
    # caller_task_id are pinned by
    # test_briefing.py::TestPerRoleMemoryTable::
    # test_the_reviewer_still_builds_without_a_task.


@pytest.mark.asyncio
class TestDistilledRendering:
    """Recalled memory renders as markdown bullets, never as raw JSON (D5).

    Each surviving result renders as one ``- [category · date · store]
    content`` bullet, with the content WHOLE.
    """

    async def _render(self, briefing: BriefingAssembler, results: list[dict]) -> str:
        return await _recall(briefing, _answering(_mcp_search_envelope(results)))

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
        """D5: entries are not capped, because a canonical memory record is
        long precisely because its reasoning is the payload."""
        long_content = 'The measured rationale. ' * 200
        entry = _result('1', long_content.strip(), source_store='mem0')

        context = await self._render(briefing, [entry])

        assert long_content.strip() in context
        assert '...' not in context
        assert 'truncated' not in context

    async def test_a_graphiti_result_falls_back_to_its_valid_at_date(
        self, briefing: BriefingAssembler,
    ):
        entry = _result('1', 'An edge fact about task 3659.', source_store='graphiti')
        entry['temporal'] = {'valid_at': '2026-09-14T07:58:01.179808+00:00', 'invalid_at': None}

        context = await self._render(briefing, [entry])

        assert (
            '- [uncategorized · 2026-09-14 · graphiti] An edge fact about task 3659.'
        ) in context

    async def test_the_category_falls_back_to_the_metadata_copy(
        self, briefing: BriefingAssembler,
    ):
        entry = _result('1', 'A convention.', metadata={'category': 'procedural_knowledge'},
                        source_store='mem0')
        entry['created_at'] = '2026-08-15T22:22:49+00:00'

        context = await self._render(briefing, [entry])

        assert '- [procedural_knowledge · 2026-08-15 · mem0] A convention.' in context

    async def test_an_untagged_undated_result_still_renders_its_content(
        self, briefing: BriefingAssembler,
    ):
        """The tag is best-effort; the content is not."""
        context = await self._render(
            briefing, [_result('1', 'A fact with no tags at all.', source_store='mem0')],
        )

        assert '- [uncategorized · undated · mem0] A fact with no tags at all.' in context
        assert 'None' not in context

    async def test_grouped_children_render_as_nested_bullets(
        self, briefing: BriefingAssembler,
    ):
        """A filtered amendment digest still renders, as a nested bullet, or
        the nested-drop note would announce blocking a leak nobody renders."""
        context = await self._render(briefing, [_grouped_parent()])

        assert '- [uncategorized · undated · mem0] Native canonical.' in context
        assert '  - [amendment · undated · mem0] NATIVE AMENDMENT BODY' in context
        assert 'FOREIGN AMENDMENT BODY' not in context

    async def test_a_native_matched_child_renders_as_a_nested_bullet(
        self, briefing: BriefingAssembler,
    ):
        """The other half of ``GROUPED_CHILD_KEYS``: ``matched_children``
        carries ``content`` where an amendment digest carries ``digest``."""
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

    async def test_the_briefing_shows_the_contesting_child_in_full(
        self, briefing: BriefingAssembler,
    ):
        """A correction recalled with the claim it contests reaches the agent
        whole and marked as contesting it, not as a cut digest under it."""
        results = json.loads(recorded_search_text('child-also-matched'))['results']
        parent = next(entry for entry in results if 'grouped' in entry)
        contesting_id = next(
            child['id'] for child in parent['grouped']['amendments'] if child.get('contested')
        )
        full_body = next(entry['content'] for entry in results if entry['id'] == contesting_id)

        context = await _recall(
            briefing, _answering(recorded_search_envelope('child-also-matched')),
        )

        assert any(
            line.startswith('  - [contests its parent · ') and line.endswith(f'] {full_body}')
            for line in context.splitlines()
        ), context

    async def test_the_section_headings_come_from_the_specs(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, [_result('1', 'A fact.', source_store='mem0')])

        assert '## Conventions & Gotchas' in context
        assert '## Task Context' in context
        for spec, _text in queries_for(_SCOPE):
            assert f'## {spec.section_title}' in context


@pytest.mark.asyncio
class TestTaskEntityChannel:
    """The second half of D3's dual-channel task context.

    ``get_entity`` is asked for ``Task <id>`` alongside the semantic search.
    Its fuzzy fallback answers a miss with a DIFFERENT entity, so the reply
    is admitted only on exact name equality, client-side.
    """

    async def _render(
        self, briefing: BriefingAssembler, entity_reply: dict,
    ) -> str:
        """Answer the searches with one fact and the entity call with *entity_reply*."""
        search_reply = _mcp_search_envelope([_result('1', 'A recalled fact.', source_store='mem0')])

        async def dispatch(_url, _method, params, **_kwargs):
            return entity_reply if params['name'] == 'get_entity' else search_reply

        return await _recall(briefing, AsyncMock(side_effect=dispatch))

    async def test_an_exactly_named_node_renders_its_summary_and_edges(
        self, briefing: BriefingAssembler,
    ):
        context = await self._render(briefing, _entity_envelope(
            nodes=[_node('Task 3609', 'Project-scopes the briefing context block.')],
            edges=[_edge('Task 3609 is related to task 3212.')],
        ))

        task_section = context.split('## Task Context')[1]
        assert 'Project-scopes the briefing context block.' in task_section
        assert 'Task 3609 is related to task 3212.' in task_section

    async def test_the_entity_is_asked_for_by_exact_task_name(
        self, briefing: BriefingAssembler,
    ):
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        await _recall(briefing, mcp)

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
        assert '- Task 3609 landed as commit abc123. (2026-09-14)' in task_section
        assert 'Task 3609 is related to task 3212.' in task_section

    async def test_a_graph_outage_is_named_not_rendered_as_an_empty_graph(
        self, briefing: BriefingAssembler,
    ):
        """``get_entity`` answers a Graphiti fault with the same fault-only
        ``degraded``/``failed_stores`` keys the search channel uses; without
        reading them a failed graph renders exactly like an empty one."""
        degraded = _entity_envelope(nodes=[], edges=[])
        payload = json.loads(degraded['result']['content'][0]['text'])
        payload.update({'degraded': True, 'failed_stores': ['graphiti']})
        degraded['result']['content'][0]['text'] = json.dumps(payload)

        context = await self._render(briefing, degraded)
        healthy = await self._render(briefing, _entity_envelope(nodes=[], edges=[]))

        assert MEMORY_DEGRADED_STORES_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX, stores='graphiti',
        ) in context
        assert context != healthy

    async def test_an_unreachable_graph_names_its_channel_and_reason(
        self, briefing: BriefingAssembler,
    ):
        search_reply = _mcp_search_envelope([_result('1', 'A recalled fact.', source_store='mem0')])

        async def dispatch(_url, _method, params, **_kwargs):
            if params['name'] == 'get_entity':
                raise ConnectionError('graph unreachable')
            return search_reply

        context = await _recall(briefing, AsyncMock(side_effect=dispatch))

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX,
            reason=MemoryFailure.TRANSPORT.value,
        ) in context
        assert 'A recalled fact.' in context, 'the semantic channel still renders'

    async def test_a_graph_tool_error_is_named_and_never_rendered(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The shared ``isError`` blind spot, on the channel that also has it."""
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

    @pytest.mark.parametrize(
        'text',
        ['not json', '["a", "list"]', '"a bare string"', 'null'],
        ids=['not-json', 'json-list', 'json-string', 'json-null'],
    )
    async def test_a_graph_reply_that_is_not_a_json_object_is_named_malformed(
        self, briefing: BriefingAssembler, caplog, text,
    ):
        """Unlike a search reply, a garbled graph reply cannot fail open: the
        exact-name admission needs the parsed node, so it is a failure, never
        an empty graph and never rendered."""
        with caplog.at_level(logging.WARNING):
            context = await self._render(briefing, {'result': {
                'content': [{'type': 'text', 'text': text}],
            }})

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context' + ENTITY_CHANNEL_SUFFIX,
            reason=MemoryFailure.MALFORMED.value,
        ) in context
        assert text not in context.split('## Task Context')[1]
        assert 'A recalled fact.' in context, 'the semantic channel still renders'
        assert any('Task 3609' in r.getMessage() for r in caplog.records)

    async def test_the_two_channels_of_one_section_name_themselves_apart(
        self, briefing: BriefingAssembler,
    ):
        """Both answer ``## Task Context``, so an unqualified notice would
        print the same sentence twice and name neither corpus."""
        context = await _recall(briefing, _raising(ConnectionError('everything is down')))

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
        """The loop asks ``wants_entity_block``, not "is this spec TASK_SEMANTIC"."""
        assert TASK_SEMANTIC.wants_entity_block
        fired = [spec.wants_entity_block for spec, _text in queries_for(_SCOPE)]
        assert fired.count(True) == 1

        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))
        await _recall(briefing, mcp)

        assert len([
            call for call in mcp.await_args_list
            if call.args[2].get('name') == 'get_entity'
        ]) == 1

    async def test_a_task_less_dispatch_asks_for_no_entity(
        self, briefing: BriefingAssembler,
    ):
        mcp = _answering(_mcp_search_envelope([_result('1', 'A fact.')]))

        with memory_transport(mcp):
            await briefing.build_reviewer_prompt('reviewer_comprehensive', 'DIFF')

        assert not [
            call for call in mcp.await_args_list
            if call.args[2].get('name') == 'get_entity'
        ]


@pytest.mark.asyncio
class TestDegradationIsLoud:
    """A memory outage is reported, not silently rendered as "nothing known" (D6/INV-2)."""

    def _dispatch(self, *, failing_slug: str | None = None, payload: dict | None = None):
        """Answer every call with *payload*, except the query for *failing_slug*."""
        payload = payload if payload is not None else {
            'results': [_result('1', 'A recalled fact.', source_store='mem0')],
        }
        reply = {'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]}}
        failing = {
            text for spec, text in queries_for(_SCOPE) if spec.slug == failing_slug
        }

        async def dispatch(_url, _method, params, **_kwargs):
            if params['arguments'].get('query') in failing:
                raise ConnectionError('memory service unreachable')
            return reply

        return AsyncMock(side_effect=dispatch)

    async def test_a_failed_query_names_its_missing_section_and_warns(
        self, briefing: BriefingAssembler, caplog,
    ):
        with caplog.at_level(logging.DEBUG):
            context = await _recall(
                briefing, self._dispatch(failing_slug='briefing-task-semantic'),
            )

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context', reason='transport',
        ) in context
        assert '## Conventions & Gotchas' in context, 'the healthy query still renders'
        assert 'A recalled fact.' in context

        failure_logs = [r for r in caplog.records if 'memory service unreachable' in r.getMessage()]
        assert failure_logs, 'the transport failure must reach the log'
        assert all(r.levelno >= logging.WARNING for r in failure_logs)

    async def test_a_degraded_reply_names_the_stores_that_failed(
        self, briefing: BriefingAssembler,
    ):
        """The signal was already on the wire; it must be read and rendered."""
        context = await _recall(briefing, self._dispatch(payload={
            'results': [_result('1', 'A recalled fact.', source_store='mem0')],
            'degraded': True,
            'failed_stores': ['graphiti'],
        }))

        assert MEMORY_DEGRADED_STORES_NOTICE.format(
            section='Conventions & Gotchas', stores='graphiti',
        ) in context
        assert 'A recalled fact.' in context
        assert '{' not in context and '}' not in context

    async def test_a_total_outage_reads_differently_from_an_empty_corpus(
        self, briefing: BriefingAssembler,
    ):
        assert MEMORY_EMPTY_NOTICE != MEMORY_OUTAGE_NOTICE

        outage = await _recall(briefing, _raising(ConnectionError('memory service unreachable')))
        empty = await _recall(briefing, _answering({'result': {'content': []}}))

        assert MEMORY_EMPTY_NOTICE in empty
        assert MEMORY_EMPTY_NOTICE not in outage
        assert MEMORY_OUTAGE_NOTICE.format(reasons='transport') in outage
        assert outage != empty

    async def test_a_drop_note_and_a_failure_notice_both_render(
        self, briefing: BriefingAssembler,
    ):
        """A blocked cross-project leak and a broken query are separate
        facts; neither may displace the other."""
        context = await _recall(briefing, self._dispatch(
            failing_slug='briefing-task-semantic',
            payload={'results': [
                _result('1', 'Own fact.', metadata={'project_id': 'dark_factory'}),
                _result('2', 'Foreign fact.', metadata={'project_id': 'reify'}),
            ]},
        ))

        assert 'tagged to another project and filtered out' in context
        assert '**Task Context**' in context
        assert 'Own fact.' in context
        assert 'Foreign fact.' not in context

    async def test_a_partial_failure_survives_having_nothing_left_to_render(
        self, briefing: BriefingAssembler,
    ):
        """One query fails AND the other recalls nothing: the notices must
        survive the no-sections branch rather than read as a healthy empty corpus."""
        context = await _recall(briefing, self._dispatch(
            failing_slug='briefing-task-semantic',
            payload={'results': [], 'degraded': True, 'failed_stores': ['graphiti']},
        ))

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Task Context', reason='transport',
        ) in context
        assert MEMORY_DEGRADED_STORES_NOTICE.format(
            section='Conventions & Gotchas', stores='graphiti',
        ) in context
        assert context != f'# Context\n\n{MEMORY_EMPTY_NOTICE}'

    async def test_a_total_outage_keeps_its_per_section_notices(
        self, briefing: BriefingAssembler,
    ):
        """The family line says "memory is unavailable", the notices say
        WHICH questions went unanswered; the family line leads."""
        context = await _recall(briefing, _raising(ConnectionError('memory service unreachable')))

        assert context.index(MEMORY_OUTAGE_NOTICE.split('{')[0]) < context.index(
            MEMORY_SECTION_FAILURE_NOTICE.split('{')[0]
        ), 'the family line leads the block, so the digest markers still match'
        for section in ('Conventions & Gotchas', 'Task Context'):
            assert MEMORY_SECTION_FAILURE_NOTICE.format(
                section=section, reason='transport',
            ) in context, f'{section} went unanswered and must say so'
        assert MEMORY_DEGRADED_STORES_NOTICE.split('{')[0] not in context

    async def test_a_reply_with_no_tool_result_is_named_malformed(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The JSON-RPC error envelope: no ``result`` key at all."""
        with caplog.at_level(logging.DEBUG):
            context = await _recall(
                briefing, _answering({'error': {'code': -32603, 'message': 'boom'}}),
            )

        for section in ('Conventions & Gotchas', 'Task Context'):
            assert MEMORY_SECTION_FAILURE_NOTICE.format(
                section=section, reason=MemoryFailure.MALFORMED.value,
            ) in context
        no_result_logs = [r for r in caplog.records if 'no tool result' in r.getMessage()]
        assert no_result_logs
        assert all(r.levelno >= logging.WARNING for r in no_result_logs)

    async def test_a_reply_with_null_content_is_an_honest_empty_answer(
        self, briefing: BriefingAssembler,
    ):
        """The service answered, oddly but without ``isError``: that is no
        text, read the same way the shared envelope reader reads it, never a
        broken recall loop that blames the transport."""
        context = await _recall(briefing, _answering({'result': {'content': None}}))

        assert context == f'# Context\n\n{MEMORY_EMPTY_NOTICE}'

    async def test_a_non_dict_tool_result_is_named_malformed(
        self, briefing: BriefingAssembler,
    ):
        context = await _recall(briefing, _answering({'result': ['not', 'a', 'dict']}))

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in context

    async def test_a_malformed_dispatch_counts_toward_the_outage_streak(
        self, briefing: BriefingAssembler, caplog,
    ):
        """All three failure classes are outages; only the reason differs."""
        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD):
                await _recall(
                    briefing, _answering({'error': {'code': -32603, 'message': 'boom'}}),
                )

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, errors

    async def test_a_tool_error_is_a_failure_not_a_recalled_fact(
        self, briefing: BriefingAssembler, caplog,
    ):
        """FastMCP reports a tool-level failure IN the body, not by raising.

        Its text block reads exactly like a result, so a shape-only reader
        would render ``Error: ...`` as remembered fact and count it as a
        recall that resets the outage streak.
        """
        tool_error = _answering({'result': {
            'isError': True,
            'content': [{'type': 'text', 'text': 'Error: embeddings backend refused'}],
        }})

        with caplog.at_level(logging.DEBUG):
            contexts = [
                await _recall(briefing, tool_error)
                for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD)
            ]

        assert 'embeddings backend refused' not in contexts[0]
        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in contexts[0]
        streak_errors = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and f' {MEMORY_OUTAGE_STREAK_THRESHOLD} consecutive ' in r.getMessage()
        ]
        assert len(streak_errors) == 1, 'a tool error recalled nothing, so it must not reset the streak'
        assert any(
            'embeddings backend refused' in r.getMessage() and r.levelno == logging.WARNING
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
        rejection = json.dumps({
            'error': "Invalid project_id 'foo-bar'", 'error_type': 'ValidationError',
        })
        rejecting = _answering({'result': {'content': [{'type': 'text', 'text': rejection}]}})

        with caplog.at_level(logging.DEBUG), memory_transport(rejecting):
            prompts = [
                await briefing.build_implementer_prompt({'steps': []}, task_id='3609')
                for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD)
            ]

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.MALFORMED.value,
        ) in prompts[0]
        assert 'error_type' not in prompts[0]
        assert '{' not in prompts[0]
        streak_errors = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and f' {MEMORY_OUTAGE_STREAK_THRESHOLD} consecutive ' in r.getMessage()
        ]
        assert len(streak_errors) == 1
        assert any(
            r.levelno >= logging.WARNING and 'results' in r.getMessage()
            for r in caplog.records
        )


@pytest.mark.asyncio
class TestOutageStreakEscape:
    """A sustained memory outage escalates above the per-dispatch noise (INV-4).

    A consecutive STREAK, not a time window: it resets on any dispatch that
    recalled something and re-alarms at every multiple of the threshold.
    """

    async def _failing_dispatch(self, briefing: BriefingAssembler) -> str:
        return await _recall(briefing, _raising(ConnectionError('memory service unreachable')))

    async def _healthy_dispatch(self, briefing: BriefingAssembler) -> str:
        return await _recall(briefing, _answering(_mcp_search_envelope([
            _result('1', 'A recalled fact.', source_store='mem0'),
        ])))

    @staticmethod
    def _errors(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]

    async def test_a_short_run_of_outages_does_not_escalate(
        self, briefing: BriefingAssembler, caplog,
    ):
        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
                await self._failing_dispatch(briefing)

        assert self._errors(caplog) == []

    async def test_crossing_the_threshold_escalates_exactly_once(
        self, briefing: BriefingAssembler, caplog,
    ):
        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD):
                await self._failing_dispatch(briefing)

        errors = self._errors(caplog)
        assert len(errors) == 1, errors
        assert f' {MEMORY_OUTAGE_STREAK_THRESHOLD} consecutive ' in errors[0]

    async def test_a_permanent_outage_re_alarms_on_each_further_crossing(
        self, briefing: BriefingAssembler, caplog,
    ):
        """Twice the threshold separates ``% THRESHOLD`` from ``== THRESHOLD``
        (silent after the first crossing) and ``>= THRESHOLD`` (one line per
        dispatch forever)."""
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
        for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
            await self._failing_dispatch(briefing)
        await self._healthy_dispatch(briefing)

        with caplog.at_level(logging.ERROR):
            for _ in range(MEMORY_OUTAGE_STREAK_THRESHOLD - 1):
                await self._failing_dispatch(briefing)

        assert self._errors(caplog) == []

    async def test_the_per_dispatch_layer_still_reports_every_failure(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The escalation is an added layer, not a replacement: every failing
        dispatch still warns and still says so in its own prompt."""
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

    Every Graphiti-sourced result is untagged, so it survives the filter
    unclassified; the caveat is what makes the agent verify a recalled path.
    """

    async def test_caveat_present_and_precedes_first_section(
        self, briefing: BriefingAssembler,
    ):
        envelope = _mcp_search_envelope([
            _result('1', 'Own project fact.', metadata={'project_id': 'dark_factory'}),
        ])

        context = await _recall(briefing, _answering(envelope))

        assert context.splitlines()[0] == '# Context'
        caveat = MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id)
        assert caveat in context
        assert context.index(caveat) < context.index('\n## ')

    async def test_caveat_absent_when_no_sections_survive(self, briefing: BriefingAssembler):
        context = await _recall(briefing, _answering({'result': {'content': []}}))

        assert context == f'# Context\n\n{MEMORY_EMPTY_NOTICE}'

    async def test_caveat_present_alongside_untagged_foreign_path_content(
        self, briefing: BriefingAssembler,
    ):
        """The census failure mode: untagged content naming a foreign path
        renders verbatim, so the caveat must be there to warn about it."""
        envelope = _mcp_search_envelope([
            _result(
                '1',
                "A park on 'crates/reify-compiler/src' blocks acquire of "
                "'crates/reify-compiler'.",
                metadata={},
            ),
        ])

        context = await _recall(briefing, _answering(envelope))

        assert 'crates/reify-compiler' in context
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context

    async def test_caveat_covers_recalled_sections_after_partial_failure(
        self, briefing: BriefingAssembler,
    ):
        """A query failing must not blank the caveat for sections that were
        already genuinely recalled."""
        failing = {text for spec, text in queries_for(_SCOPE) if spec.slug == TASK_SEMANTIC.slug}
        search_reply = _mcp_search_envelope([
            _result('1', 'A recalled convention.', source_store='mem0'),
        ])

        async def dispatch(_url, _method, params, **_kwargs):
            if params['name'] == 'get_entity':
                return _entity_envelope(nodes=[], edges=[])
            if params['arguments']['query'] in failing:
                raise TimeoutError('memory service unreachable')
            return search_reply

        context = await _recall(briefing, AsyncMock(side_effect=dispatch))

        assert context.splitlines()[0] == '# Context'
        assert '## Conventions & Gotchas' in context
        assert '## Task Context' not in context
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context


@pytest.mark.asyncio
class TestFailureClassification:
    """A slow memory service reads differently from an unreachable one.

    ``mcp_call`` re-raises a plain ``RuntimeError`` once its retries exhaust,
    keeping the cause only as ``__cause__``; these tests raise exactly that.
    """

    @staticmethod
    def _exhausted(cause: Exception) -> RuntimeError:
        """The wrap ``McpSession._raw_call`` raises on retry exhaustion.

        The shape itself is pinned against the real retry loop in
        ``test_mcp_retry.py::TestTimeoutCausePredicate``.
        """
        err = RuntimeError(
            f'MCP tools/call failed after 3 attempts: {type(cause).__name__}: {cause}'
        )
        err.__cause__ = cause
        return err

    async def test_an_exhausted_timeout_reaches_the_reader_as_a_timeout(
        self, briefing: BriefingAssembler,
    ):
        context = await _recall(briefing, _raising(self._exhausted(httpx.ReadTimeout(''))))

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.TIMEOUT.value,
        ) in context
        assert MemoryFailure.TRANSPORT.value not in context, (
            'an alive-but-slow service must not be reported as unreachable'
        )

    async def test_an_exhausted_connect_error_reaches_the_reader_as_transport(
        self, briefing: BriefingAssembler,
    ):
        context = await _recall(
            briefing, _raising(self._exhausted(httpx.ConnectError('refused'))),
        )

        assert MEMORY_SECTION_FAILURE_NOTICE.format(
            section='Conventions & Gotchas', reason=MemoryFailure.TRANSPORT.value,
        ) in context
        assert MemoryFailure.TIMEOUT.value not in context


@pytest.mark.asyncio
class TestSearchReplyShapes:
    """How the shape of a well-delivered search reply reaches the block."""

    async def test_a_reply_with_no_text_blocks_is_an_honest_empty_corpus(
        self, briefing: BriefingAssembler,
    ):
        """Nothing came back and nothing broke, which is what distinguishes
        this from an outage."""
        context = await _recall(briefing, _answering({'result': {'content': []}}))

        assert context == f'# Context\n\n{MEMORY_EMPTY_NOTICE}'

    async def test_multi_text_block_response_fails_open_known_limitation(
        self, briefing: BriefingAssembler, caplog,
    ):
        """A reply split across text blocks joins into invalid JSON, so the
        filter cannot run on it: it renders verbatim and unfiltered, with a
        WARNING. Pinned so a change to the response shape, or a per-block
        fix, is caught rather than drifting unnoticed."""
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

        with caplog.at_level(logging.WARNING):
            context = await _recall(briefing, _answering(envelope))

        conventions = context.split('## Conventions & Gotchas', 1)[1].split('\n\n---\n\n', 1)[0]
        assert 'Foreign fact.' in conventions
        assert 'Own fact.' in conventions
        assert 'filtered out' not in context
        assert any(r.levelno == logging.WARNING for r in caplog.records)
