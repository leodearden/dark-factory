"""Shared fixtures and wire-shape builders for the briefing test files.

One home (SPOT) for what ``test_briefing.py``, ``test_briefing_project_scope.py``
and ``test_memory_recall.py`` share: an assembler to drive, the ``search`` and
``get_entity`` reply shapes to answer it with, and the read-back of what it
asked. Hand-copies of these between files had already diverged before they
were gathered here.

A module rather than ``conftest.py`` for the same reason ``_orch_helpers.py``
is: a uniquely-named sibling can be imported by name from a test file
without colliding with a sibling subproject's conftest under
``sys.modules['conftest']``. The names keep their original leading
underscore so the importing files read unchanged at every call site; the
``_build_harness`` export of ``_workflow_helpers`` is the house precedent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


MEMORY_TRANSPORT = 'orchestrator.agents.memory_recall.mcp_call'
"""The one patch target for answering briefing memory recall in a test.

It is the network boundary, so patching it lets the real recall, parse,
filter, render and compose pipeline run.
"""


def memory_transport(mock):
    """Answer every memory call the briefing makes with *mock*."""
    return patch(MEMORY_TRANSPORT, new=mock)


def _result(
    id_: str,
    content: str,
    metadata: dict | None = None,
    source_store: str = 'graphiti',
) -> dict:
    """Build a dict matching the wire shape of ``fused_memory.models.memory.MemoryResult``.

    Mirrors the real result schema (id/content/category/source_store/
    relevance_score/provenance/temporal/entities/metadata/created_at) so the
    filter is exercised against the actual payload shape, not an invented one.
    """
    return {
        'id': id_,
        'content': content,
        'category': None,
        'source_store': source_store,
        'relevance_score': 0.9,
        'provenance': [],
        'temporal': None,
        'entities': [],
        'metadata': {} if metadata is None else metadata,
        'created_at': None,
    }


def _mcp_search_envelope(results: list[dict]) -> dict:
    """Build the real ``tools/call`` response envelope memory recall reads.

    FastMCP returns ``{'result': {'content': [{'type': 'text', 'text': ...}]}}``
    where ``text`` is the JSON-serialised ``search`` tool payload. Handed to
    :func:`memory_transport` so the real ``orchestrator.agents.memory_recall``
    pipeline runs end to end.
    """
    return {
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps({'results': results})},
            ],
        },
    }


CONTESTING_CHILD_FIXTURES = Path(__file__).parent / 'fixtures' / 'grouped_search_contesting_child'


def recorded_search_text(name: str) -> str:
    """The recorded ``search`` reply *name*, verbatim; see that directory's PROVENANCE.md."""
    return (CONTESTING_CHILD_FIXTURES / f'{name}.json').read_text(encoding='utf-8')


def recorded_search_envelope(name: str) -> dict:
    """The recorded reply *name* put on the wire byte for byte; see PROVENANCE.md."""
    return {'result': {'content': [{'type': 'text', 'text': recorded_search_text(name)}]}}


def _search_arguments(mcp_call_mock) -> list[dict]:
    """The ``arguments`` of every ``search`` tools/call the assembler made.

    Filters by tool name rather than counting awaits: a task-scoped dispatch
    also calls ``get_entity`` on the same transport, and the query-table
    assertions are about searches.
    """
    return [
        call.args[2]['arguments']
        for call in mcp_call_mock.await_args_list
        if call.args[2].get('name') == 'search'
    ]


def _grouped_parent(id_: str = 'p1', *, grouped: dict | None = None) -> dict:
    """A NATIVE canonical hit carrying a ``grouped`` block, as the server nests it.

    Mirrors what ``fused_memory.server.grouped_read.group_search_results``
    emits: the block is hung on a KEPT parent entry at ``entry['grouped']``,
    with bounded amendment digests under
    ``amendments`` and full swallowed bodies under ``matched_children``.
    """
    entry = _result('x', 'placeholder', metadata=None, source_store='mem0')
    entry['id'] = id_
    entry['content'] = 'Native canonical.'
    entry['metadata'] = {'project_id': 'dark_factory'}
    entry['grouped'] = _grouped_block() if grouped is None else grouped
    return entry


def _grouped_block() -> dict:
    return {
        'amendments': [
            {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment', 'metadata': {'src_project': 'reify'}},
            {'id': 'a2', 'digest': 'NATIVE AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment', 'metadata': {'project_id': 'dark_factory'}},
            {'id': 'a3', 'digest': 'UNTAGGED AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment'},
        ],
        'matched_children': [
            {'id': 's1', 'content': 'FOREIGN SIGHTING BODY', 'created_at': None,
             'kind': 'sighting', 'matched': True, 'metadata': {'src_project': 'reify'}},
        ],
        'amendment_count': 3,
        'sighting_count': 1,
    }


def _entity_envelope(nodes: list[dict], edges: list[dict]) -> dict:
    """The ``get_entity`` reply envelope, in the shape the server sends it.

    Mirrors ``fused_memory.services.memory_service``'s ``_node_to_dict`` /
    ``_edge_to_dict``: nodes carry ``{uuid, name, summary, labels}`` and edges
    ``{uuid, fact, temporal}``, where ``temporal`` is ``None`` on the
    exact-match path (EdgeDict has no valid_at) and a
    ``{valid_at, invalid_at}`` dict on the fuzzy path.
    """
    return {
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps({'nodes': nodes, 'edges': edges})},
            ],
        },
    }


def _node(name: str, summary: str = 'A task node.') -> dict:
    return {'uuid': 'n1', 'name': name, 'summary': summary, 'labels': ['Entity']}


def _edge(fact: str, valid_at: str | None = None) -> dict:
    return {
        'uuid': 'e1',
        'fact': fact,
        'temporal': None if valid_at is None else {'valid_at': valid_at, 'invalid_at': None},
    }
