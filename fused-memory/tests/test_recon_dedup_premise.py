"""Empirical probe (task 2221, W5-γ): does a recon infer=False system write
ever get silently dedup-dropped by Mem0 (§2.1's '~0.92 cosine' premise)?

FINDING — PREMISE FALSE. Mem0Backend.add pins infer=False, and mem0's
infer=False write path never runs a similarity search or an update-vs-add
choice — it unconditionally mints a fresh id and inserts a new point per
message. A message is skipped only if it is malformed or has role=='system',
which recon's plain-string add() calls never produce (mem0 always
normalizes a str payload to a single role='user' message). So a repeated,
byte-identical infer=False write is never dropped for being a near-duplicate.

Confirmed empirically below against a real Qdrant, each run isolated by its
own run_id in a reused collection (the harness contract is _mem0_live_probe's),
twice: once with a stubbed, identical-vector embedder (cosine=1.0 — a
strictly stronger worst case than the survey's ~0.92), and again with real
OpenAI embeddings when OPENAI_API_KEY is available. In both, every one of N
byte-identical infer=False writes lands as a distinct point (N distinct ids,
and exactly N points carry the run's run_id). See task 2221's memory notes
for the full code-trace citations backing this finding.

Consequence for downstream tasks: δ's add_system_record is a no-op
hardening (the premise it "fixes" is already false), and λ's deletion
logic should stay ledger-authority-anchored rather than assuming Mem0
dedup as a safety net either way.
"""
from __future__ import annotations

import os

import pytest
from _fm_qdrant_live import assert_reapable, qdrant_skipif
from _mem0_live_probe import (
    build_live_backend,
    build_stub_embedder_backend,
    isolated_run,
    probe_collection,
    reapable_config,
)

from fused_memory.models.scope import Scope

FIXED_RECON_SUMMARY = (
    'Stage 2 cycle summary for run task-2221-premise-probe — '
    'byte-identical fixture content issued N times to test whether '
    'Mem0.add(infer=False) ever dedup-drops a repeated recon system write.'
)

# Each live probe below is a real-Qdrant test (never a merge-lane gate), so it
# carries qdrant_skipif(), timeout(60) and integration itself: it belongs in the
# integration/offline lane per task 3019 / the integration-test-lane PRD, and
# fused-memory's `-m 'not integration'` addopts deselects it from the merge
# lane's post-merge sweep. The marks sit on the tests, not on the module, so the
# pure reapability guard below still runs in the merge gate.


@pytest.fixture
def probe_config(mock_config):
    return reapable_config(mock_config)


# One collection per probe: the two can run at once on different xdist workers,
# and a write racing another worker's cold create of a shared collection can
# answer 500 (task 11013).
STUB_EMBEDDER_PROJECT = '_test_recon_dedup_stub'
OPENAI_EMBEDDER_PROJECT = '_test_recon_dedup_openai'
RUN_LABEL = 'task-2221-premise-probe'


def _recon_scope(project_id: str) -> Scope:
    return Scope(project_id=project_id, agent_id='recon-stage-task_knowledge_sync')


@pytest.fixture
def stub_embedder_scope() -> Scope:
    return _recon_scope(STUB_EMBEDDER_PROJECT)


@pytest.fixture
def stub_embedder_run_id(stub_embedder_scope, probe_config):
    yield from isolated_run(stub_embedder_scope, probe_config, RUN_LABEL)


@pytest.fixture
def openai_embedder_scope() -> Scope:
    return _recon_scope(OPENAI_EMBEDDER_PROJECT)


@pytest.fixture
def openai_embedder_run_id(openai_embedder_scope, probe_config):
    yield from isolated_run(openai_embedder_scope, probe_config, RUN_LABEL)


def test_the_recon_dedup_probe_collections_are_ones_the_reaper_can_reclaim(probe_config):
    """The probe collections outlive a run by design, so they must be ones the reaper deletes."""
    for project_id in (STUB_EMBEDDER_PROJECT, OPENAI_EMBEDDER_PROJECT):
        assert_reapable(probe_collection(_recon_scope(project_id), probe_config))


async def _assert_all_writes_land_distinct(backend, scope, *, n: int, run_id: str) -> None:
    """Issue *n* byte-identical infer=False writes and assert none dedup-drop.

    Shared by both tests below: asserts each write returns exactly one
    result (the infer=False shape), that all *n* returned ids are distinct
    (no in-place update in lieu of a fresh point), and that exactly *n*
    points carry *run_id* (no silent drop).
    """
    metadata = {
        'kind': 'cycle_summary',
        'stage': 'task_knowledge_sync',
        'run_id': run_id,
    }
    ids = []
    for _ in range(n):
        response = await backend.add(
            content=FIXED_RECON_SUMMARY,
            scope=scope,
            metadata=metadata,
        )
        results = response.get('results') or []
        assert len(results) == 1, (
            f'expected exactly one result under infer=False, got {results!r}'
        )
        assert 'id' in results[0]
        ids.append(results[0]['id'])

    assert len(set(ids)) == n, (
        f'expected {n} distinct ids (no dedup drop), got {len(set(ids))} distinct: {ids!r}'
    )
    assert await backend.count_by_metadata(scope, {'run_id': run_id}) == n


@qdrant_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_identical_infer_false_writes_all_land_distinct(
    probe_config, stub_embedder_scope, stub_embedder_run_id, monkeypatch,
):
    """N byte-identical recon-stage infer=False writes must all land distinct.

    Issues N=8 identical Mem0Backend.add(...) calls (content + metadata both
    fixed — no nonce) through the real production write path (infer=False
    pinned in Mem0Backend.add) against a real Qdrant, isolated by run_id.
    If the §2.1 premise ('~0.92 cosine dedup can drop a repeat recon write')
    were true, at least one of these 8 writes would return zero results
    and/or fewer than 8 points would carry this run's run_id.
    """
    backend = await build_stub_embedder_backend(stub_embedder_scope, probe_config, monkeypatch)
    try:
        await _assert_all_writes_land_distinct(
            backend, stub_embedder_scope, n=8, run_id=stub_embedder_run_id,
        )
    finally:
        await backend.close()


@qdrant_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason='real embedder needs OPENAI_API_KEY')
@pytest.mark.asyncio
async def test_identical_writes_land_with_real_openai_embeddings(
    probe_config, openai_embedder_scope, openai_embedder_run_id, monkeypatch,
):
    """Real-OpenAI-embedder confirmation: identical text still never dedups.

    Same byte-identical-write probe as
    test_identical_infer_false_writes_all_land_distinct, but with NO
    embedder stub — genuine OpenAI embeddings for identical input text
    land at real cosine≈1.0. This directly, empirically rebuts the
    survey's '~0.92 cosine similarity dedup' observation: even the
    strongest realistic near-duplicate signal never triggers a drop,
    because the infer=False path never consults embeddings for dedup.

    Opt-in via `-m integration` (task 2736): this test makes a genuine
    OpenAI network call, so it is deselected from the default suite by
    fused-memory/pyproject.toml's `-m 'not integration'` addopts. The
    skipif above is retained as a belt-and-braces guard for the
    addopts-cleared recovery/flaky-rerun path (verify_cmd.serial_pytest).
    """
    config = probe_config.model_copy(deep=True)
    # mock_config's api_key is a fake 'test-key'; None makes mem0 read OPENAI_API_KEY.
    config.embedder.providers.openai.api_key = None
    backend = await build_live_backend(openai_embedder_scope, config, monkeypatch)
    try:
        await _assert_all_writes_land_distinct(
            backend, openai_embedder_scope, n=5, run_id=openai_embedder_run_id,
        )
    finally:
        await backend.close()
