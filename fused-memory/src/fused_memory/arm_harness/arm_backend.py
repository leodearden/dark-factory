"""An arm's backend: built on the arm's own client, with its scratch indices built explicitly.

``open_arm_backend`` (LLM axis) and ``open_embedding_arm_backend`` (embedding axis) are
the only places an arm's backend is opened. The scratch guard runs before anything
is built. An LLM arm's client is β's ``build_llm_client`` output with the conformance
audit installed; an embedding arm's is the query embedder its caller built. Either is
handed to the backend's own construction path.
"""

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient

from fused_memory.arm_harness.arm_config import embedding_arm_config, llm_arm_config
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.conformance import (
    ConformanceLedger,
    ResponseValidator,
    install_conformance_audit,
    validate_response,
)
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.replay import ArmGraph
from fused_memory.arm_harness.replay_types import ReplaySettings
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name
from fused_memory.backends.falkor_indices import IndexProvisionResult, IndexSpec
from fused_memory.backends.graphiti_client import GraphitiBackend, build_llm_client
from fused_memory.config.schema import FusedMemoryConfig


class ArmBackend(ArmGraph, Protocol):
    """An ArmGraph plus the lifecycle ``open_arm_backend`` drives; GraphitiBackend is the real one."""

    async def initialize(
        self,
        *,
        skip_maintenance: bool = ...,
        llm_client: LLMClient | None = ...,
        embedder: EmbedderClient | None = ...,
    ) -> None: ...

    async def ensure_indices(self, *, group_id: str) -> IndexProvisionResult: ...

    async def close(self) -> None: ...


BackendFactory = Callable[[FusedMemoryConfig], ArmBackend]


def scratch_graphiti_backend(arm_config: FusedMemoryConfig) -> GraphitiBackend:
    """A GraphitiBackend with no registered graphs, so first-write provisioning never runs.

    Production first-write provisioning absorbs failures, so it must never be the
    index path here. The with-indices configuration builds indices explicitly instead.
    """
    return GraphitiBackend(arm_config, registered_graph_ids=frozenset())


class IndexBuildError(RuntimeError):
    """The explicit scratch-graph index build left specs unbuilt."""

    def __init__(self, group_id: str, failed: tuple[tuple[IndexSpec, str], ...]) -> None:
        self.group_id = group_id
        self.failed = failed
        reasons = '; '.join(f'{spec}: {reason}' for spec, reason in failed)
        super().__init__(f'index build on {group_id!r} failed for {len(failed)} spec(s): {reasons}')


@contextlib.asynccontextmanager
async def open_arm_backend(
    spec: LlmArmSpec,
    base_config: FusedMemoryConfig,
    settings: ReplaySettings,
    *,
    backend_factory: BackendFactory = scratch_graphiti_backend,
) -> AsyncIterator[tuple[ArmBackend, ConformanceLedger]]:
    """The arm's backend, initialized with its audited client, and that client's ledger."""
    require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.REPLAY)
    cfg = llm_arm_config(spec, base_config)
    client, ledger = audited_arm_client(spec, cfg)
    backend = backend_factory(cfg)
    try:
        await backend.initialize(skip_maintenance=True, llm_client=client)
        if settings.index_configuration is IndexConfiguration.WITH_INDICES:
            await build_scratch_indices(backend, spec.scratch_group_id)
        yield backend, ledger
    finally:
        await backend.close()


@contextlib.asynccontextmanager
async def open_embedding_arm_backend(
    spec: EmbeddingArmSpec,
    base_config: FusedMemoryConfig,
    query_embedder: EmbedderClient,
    *,
    backend_factory: BackendFactory = scratch_graphiti_backend,
) -> AsyncIterator[ArmBackend]:
    """The arm's search backend over its scratch graph, embedding queries through ``query_embedder``.

    No index is built here: the caller moves the graph between index configurations.
    """
    require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.SEARCH)
    backend = backend_factory(embedding_arm_config(spec, base_config))
    try:
        await backend.initialize(skip_maintenance=True, embedder=query_embedder)
        yield backend
    finally:
        await backend.close()


def audited_arm_client(
    spec: LlmArmSpec,
    arm_config: FusedMemoryConfig,
    *,
    validator: ResponseValidator = validate_response,
) -> tuple[LLMClient, ConformanceLedger]:
    """The arm's client, built by β's seam from its arm config, with the conformance audit on."""
    client = build_llm_client(arm_config)
    if client is None:
        raise RuntimeError(f'arm {spec.arm_id!r}: build_llm_client built no client (no api_key)')
    ledger = ConformanceLedger()
    install_conformance_audit(client, ledger, validator=validator)
    return client, ledger


async def build_scratch_indices(backend: ArmBackend, group_id: str) -> None:
    require_scratch_name(group_id, checkpoint=GuardCheckpoint.INDEX_BUILD)
    result = await backend.ensure_indices(group_id=group_id)
    if result.failed:
        raise IndexBuildError(group_id, result.failed)
