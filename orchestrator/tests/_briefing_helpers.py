"""Shared fixtures and wire-shape builders for the briefing test files.

One home (SPOT) for the three things ``test_briefing.py`` and
``test_briefing_project_scope.py`` both need: an assembler to drive, the
``search`` reply envelope to answer it with, and the read-back of what it
asked. All three were hand-copied between the two files, and the copies had
already diverged — one carried the explanatory docstring the other lacked,
and a third hand-built envelope spelled the same FastMCP shape a fourth way.

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
    """Build the real ``tools/call`` response envelope ``_mcp_search`` reads.

    Mirrors ``BriefingAssembler._mcp_search``: FastMCP returns
    ``{'result': {'content': [{'type': 'text', 'text': ...}]}}`` where
    ``text`` is the JSON-serialised ``search`` tool payload. Used to patch
    ``orchestrator.agents.briefing.mcp_call`` directly (unlike most briefing
    tests, which patch ``_get_memory_context`` itself away to a stub) so the
    real ``_get_memory_context`` / ``_scoped_search`` /
    ``filter_foreign_project_results`` pipeline actually runs end-to-end.
    """
    return {
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps({'results': results})},
            ],
        },
    }


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
