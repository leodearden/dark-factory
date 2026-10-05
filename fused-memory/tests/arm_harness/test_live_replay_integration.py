"""Live end-to-end arm replay: a real FalkorDB, a mock OpenAI endpoint (boundary rows 6 and 7).

Runs the REAL path: open_arm_backend -> GraphitiBackend.initialize(skip_maintenance=True,
llm_client=...) -> run_llm_arm, onto two ``evalmem_test_`` scratch graphs, then reads
the journal, probes the indices, hashes the topology and tears both graphs down.
Deselected by the default addopts; run it with ``uv run pytest -m integration
tests/arm_harness``.
"""

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, falkor_skipif
from _mock_openai_server import MockOpenAIServer, mock_openai_server
from falkordb.asyncio import FalkorDB

from arm_harness._fakes import PROTECTED_GRAPHS, PreregRepo, llm_spec, make_prereg_repo
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.checks import check_index_configuration
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.replay import (
    ReplayItem,
    default_replay_settings,
    open_arm_backend,
)
from fused_memory.arm_harness.run import OUTCOMES_FILENAME, load_outcomes, run_llm_arm
from fused_memory.arm_harness.run_manifest import RunManifest
from fused_memory.arm_harness.teardown import teardown_arm
from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.config.schema import (
    EmbedderProvidersConfig,
    FalkorDBProviderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
    OpenAIProviderConfig,
)
from fused_memory.services.write_journal import OPERATOR_TELEMETRY_QUERY, WriteJournal

pytestmark = [falkor_skipif(), pytest.mark.integration, pytest.mark.timeout(300)]

JOURNAL_DIRNAME = 'journal'
CONCURRENT_TEST_GRAPH_PREFIXES = ('_test_', '_probe', 'evalmem_test_')
"""Throwaway graphs other test runs create and delete on a shared FalkorDB host."""


# --- the mock endpoint's answers ------------------------------------------------------


SCALAR_MINIMUMS: Mapping[str, Any] = {
    'array': [], 'string': '', 'integer': 0, 'number': 0, 'boolean': False,
}


def minimal_instance(schema: Mapping[str, Any], defs: Mapping[str, Any] | None = None) -> Any:
    """The smallest value valid against a JSON schema: empty lists, '', false/0, None."""
    known: Mapping[str, Any] = schema.get('$defs') or defs or {}
    if '$ref' in schema:
        return minimal_instance(known[schema['$ref'].rsplit('/', 1)[-1]], known)
    if 'const' in schema:
        return schema['const']
    if 'enum' in schema:
        return schema['enum'][0]
    options = schema.get('anyOf') or schema.get('oneOf') or schema.get('allOf')
    if options:
        if any(option.get('type') == 'null' for option in options):
            return None
        return minimal_instance(options[0], known)
    kind = schema.get('type')
    if kind == 'object':
        properties = schema.get('properties', {})
        return {name: minimal_instance(properties[name], known) for name in schema.get('required', [])}
    return SCALAR_MINIMUMS.get(kind) if isinstance(kind, str) else None


def schema_answer(body: Any) -> str:
    """A chat reply holding the minimal instance of the requested json_schema, else ``{}``."""
    response_format = (body or {}).get('response_format') or {}
    schema = (response_format.get('json_schema') or {}).get('schema')
    return json.dumps(minimal_instance(schema) if schema else {})


def test_minimal_instance_resolves_refs_and_fills_only_required_fields():
    schema = {
        '$defs': {'Inner': {'type': 'object', 'properties': {'n': {'type': 'string'}},
                            'required': ['n']}},
        'type': 'object',
        'properties': {
            'items': {'type': 'array', 'items': {'$ref': '#/$defs/Inner'}},
            'inner': {'$ref': '#/$defs/Inner'},
            'maybe': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
            'flag': {'type': 'boolean'},
            'optional': {'type': 'string'},
        },
        'required': ['items', 'inner', 'maybe', 'flag'],
    }

    assert minimal_instance(schema) == {
        'items': [], 'inner': {'n': ''}, 'maybe': None, 'flag': False,
    }


# --- live setup -----------------------------------------------------------------------


def _scratch_name() -> str:
    return f'evalmem_test_{uuid.uuid4().hex[:8]}'


def _base_config(mock_config: FusedMemoryConfig, server: MockOpenAIServer) -> FusedMemoryConfig:
    falkordb = FalkorDBProviderConfig(uri=f'redis://{FALKOR_HOST}:{FALKOR_PORT}')
    embedder = mock_config.embedder.model_copy(update={
        'providers': EmbedderProvidersConfig(
            openai=OpenAIProviderConfig(api_key='test-key', api_url=server.base_url)
        ),
    })
    return mock_config.model_copy(
        update={'graphiti': GraphitiBackendConfig(falkordb=falkordb), 'embedder': embedder},
        deep=True,
    )


def _spec(scratch: str, server: MockOpenAIServer, repo: PreregRepo) -> LlmArmSpec:
    return llm_spec(
        base_url=server.base_url,
        arm_id=scratch.replace('_', '-'),
        scratch_group_id=scratch,
        code_sha=repo.with_prereg,
        preregistration_sha=None,
        arm_role='control',
    )


def _items() -> list[ReplayItem]:
    return [
        ReplayItem(
            episode_id=f'live-ep-{i}',
            name=f'live-ep-{i}',
            content=f'Alice met Bob in Paris on day {i}.',
            source_description='lme live integration',
            reference_time=datetime(2026, 10, 5, 12, i, tzinfo=UTC),
        )
        for i in range(2)
    ]


async def _replay(
    spec: LlmArmSpec,
    base: FusedMemoryConfig,
    configuration: IndexConfiguration,
    run_dir: Path,
    repo: PreregRepo,
) -> RunManifest:
    settings = default_replay_settings(base, concurrency=2, index_configuration=configuration)
    journal = WriteJournal(run_dir / JOURNAL_DIRNAME)
    await journal.initialize()
    try:
        async with open_arm_backend(spec, base, settings) as (backend, ledger):
            return await run_llm_arm(
                spec,
                _items(),
                graph=backend,
                conformance=ledger,
                journal=journal,
                settings=settings,
                run_dir=run_dir,
                repo_root=repo.root,
                reference=None,
                base_config=base,
            )
    finally:
        await journal.close()


def _telemetry_rows(run_dir: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(run_dir / JOURNAL_DIRNAME / 'write_journal.db') as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(OPERATOR_TELEMETRY_QUERY, ('1970-01-01', 100)).fetchall()
    return [row for row in rows if row['operation'] == 'arm_replay_episode']


async def _protected_indexes(client: FalkorDB, graphs: set[str]) -> dict[str, list[Any]]:
    """``CALL db.indexes()`` over GRAPH.RO_QUERY on every protected graph present on the host."""
    present = sorted(graphs & set(PROTECTED_GRAPHS))
    return {
        name: (await client.select_graph(name).ro_query('CALL db.indexes()')).result_set
        for name in present
    }


# --- the live run ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_replay_lands_only_on_scratch_graphs(mock_config, tmp_path):
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with_indices, embedding_only = _scratch_name(), _scratch_name()
    scratch = {with_indices, embedding_only}
    graphs_before = set(await client.list_graphs())
    indexes_before = await _protected_indexes(client, graphs_before)
    repo = make_prereg_repo(tmp_path / 'repo')
    try:
        with mock_openai_server() as server:
            server.chat_responder = schema_answer
            base = _base_config(mock_config, server)
            spec_a = _spec(with_indices, server, repo)
            spec_b = _spec(embedding_only, server, repo)
            run_a = await _replay(
                spec_a, base, IndexConfiguration.WITH_INDICES, tmp_path / 'a', repo
            )
            run_b = await _replay(
                spec_b, base, IndexConfiguration.EMBEDDING_ONLY, tmp_path / 'b', repo
            )
            chat_requests = server.requests_to('/chat/completions')

        # (1) both runs complete, every episode ok
        assert chat_requests, 'the arm base_url received no traffic'
        for run, run_dir in ((run_a, tmp_path / 'a'), (run_b, tmp_path / 'b')):
            assert run.incomplete is False, run.abort
            outcomes = load_outcomes(run_dir / OUTCOMES_FILENAME)
            assert [o.ok for o in outcomes] == [True, True], outcomes

        # (2) boundary row 7, live: the journal row carries duration and tokens
        rows = _telemetry_rows(tmp_path / 'a')
        assert len(rows) == 2
        for row in rows:
            assert row['project_id'] == with_indices
            assert row['duration_ms'] > 0
            assert row['total_tokens'] > 0

        # (3) boundary row 6, live: each configuration differs in fact
        indexed = await check_index_configuration(
            client.select_graph(with_indices), with_indices, IndexConfiguration.WITH_INDICES
        )
        bare = await check_index_configuration(
            client.select_graph(embedding_only), embedding_only, IndexConfiguration.EMBEDDING_ONLY
        )
        assert indexed.passed, indexed.detail
        assert bare.passed, bare.detail

        # (4) the topology read sees both episodes, and its hash is stable
        first = await read_topology(client.select_graph(with_indices), with_indices)
        second = await read_topology(client.select_graph(with_indices), with_indices)
        episodic = [node for node in first.nodes if 'Episodic' in node.labels]
        assert len(episodic) == 2
        assert topology_hash(*first) == topology_hash(*second)

        # (5) teardown removes both scratch graphs
        await teardown_arm(client, None, spec_a)
        await teardown_arm(client, None, spec_b)
        assert not scratch & set(await client.list_graphs())
    finally:
        for name in scratch:
            with contextlib.suppress(Exception):
                await client.select_graph(name).delete()
        graphs_after = set(await client.list_graphs())
        indexes_after = await _protected_indexes(client, graphs_after)
        await client.aclose()

    # (6) hazard: only scratch graphs came and went; no protected graph's indices moved
    changed = graphs_before ^ graphs_after
    assert not changed & set(PROTECTED_GRAPHS)
    assert not scratch & graphs_after
    assert {name for name in changed if not name.startswith(CONCURRENT_TEST_GRAPH_PREFIXES)} == set()
    if not indexes_before:
        pytest.skip('no protected graph exists on this FalkorDB host to compare indices on')
    assert indexes_after == indexes_before
