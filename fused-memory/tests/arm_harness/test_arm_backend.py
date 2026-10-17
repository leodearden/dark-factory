"""The arm backends: guard first, the arm's client into the backend, scratch indices built loudly."""

from collections.abc import Iterable
from typing import Any

import pytest
from graphiti_core.embedder.client import EmbedderClient

from arm_harness._fakes import FakeArmGraph, embedding_spec, llm_spec
from fused_memory.arm_harness.arm_backend import (
    IndexBuildError,
    build_scratch_indices,
    open_arm_backend,
    open_embedding_arm_backend,
)
from fused_memory.arm_harness.arm_config import embedding_arm_config
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.conformance import ConformanceLedger
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.arm_harness.replay_types import ReplaySettings
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError
from fused_memory.backends.falkor_indices import IndexProvisionResult, IndexSpec
from fused_memory.config.schema import FusedMemoryConfig

UNBUILT: tuple[IndexSpec, str] = (('Entity', 'node', 'name', 'FULLTEXT'), 'boom')


def _provisioned(failed: tuple[tuple[IndexSpec, str], ...] = ()) -> IndexProvisionResult:
    return IndexProvisionResult(
        created=(), already_present=0, failed=failed, expected_total=len(failed), statements=()
    )


class SpyBackend(FakeArmGraph):
    """An ArmBackend double recording the lifecycle calls ``open_arm_backend`` makes."""

    def __init__(
        self, provisioned: IndexProvisionResult, *, initialize_error: Exception | None = None
    ) -> None:
        super().__init__()
        self._provisioned = provisioned
        self._initialize_error = initialize_error
        self.lifecycle: list[tuple[str, dict[str, Any]]] = []

    async def initialize(
        self, *, skip_maintenance: bool = False, llm_client: Any = None, embedder: Any = None
    ) -> None:
        self.lifecycle.append((
            'initialize',
            {'skip_maintenance': skip_maintenance, 'llm_client': llm_client, 'embedder': embedder},
        ))
        if self._initialize_error is not None:
            raise self._initialize_error

    async def ensure_indices(self, *, group_id: str) -> IndexProvisionResult:
        self.lifecycle.append(('ensure_indices', {'group_id': group_id}))
        return self._provisioned

    async def close(self) -> None:
        self.lifecycle.append(('close', {}))


class SpyFactory:
    def __init__(
        self,
        provisioned: IndexProvisionResult | None = None,
        *,
        initialize_error: Exception | None = None,
    ) -> None:
        self.configs: list[FusedMemoryConfig] = []
        self.built: list[SpyBackend] = []
        self._provisioned = provisioned or _provisioned()
        self._initialize_error = initialize_error

    def __call__(self, arm_config: FusedMemoryConfig) -> SpyBackend:
        self.configs.append(arm_config)
        backend = SpyBackend(self._provisioned, initialize_error=self._initialize_error)
        self.built.append(backend)
        return backend


def _settings(configuration: IndexConfiguration) -> ReplaySettings:
    return ReplaySettings(
        concurrency=1, episode_timeout_s=120.0, index_configuration=configuration
    )


def _calls(backend: SpyBackend) -> list[str]:
    return [name for name, _ in backend.lifecycle]


@pytest.mark.asyncio
async def test_a_validation_bypassed_spec_is_refused_before_any_backend_is_built(mock_config):
    bypassed = LlmArmSpec.model_construct(
        **(dict(llm_spec()) | {'scratch_group_id': 'dark_factory'})
    )
    factory = SpyFactory()

    with pytest.raises(ScratchGuardError) as caught:
        async with open_arm_backend(
            bypassed,
            mock_config,
            _settings(IndexConfiguration.WITH_INDICES),
            backend_factory=factory,
        ):
            pytest.fail('a guarded backend must never be yielded')

    assert caught.value.checkpoint is GuardCheckpoint.REPLAY
    assert factory.configs == []


@pytest.mark.asyncio
async def test_with_indices_builds_the_scratch_indices_on_the_audited_client(mock_config):
    spec = llm_spec()
    factory = SpyFactory()

    async with open_arm_backend(
        spec, mock_config, _settings(IndexConfiguration.WITH_INDICES), backend_factory=factory
    ) as (backend, ledger):
        (built,) = factory.built
        assert backend is built
        assert isinstance(ledger, ConformanceLedger)
        assert _calls(built) == ['initialize', 'ensure_indices']

    (arm_config,) = factory.configs
    assert arm_config.llm.model == spec.model_id
    (_, initialized), (_, ensured), _ = built.lifecycle
    assert initialized['skip_maintenance'] is True
    assert initialized['llm_client'].model == spec.model_id
    assert ensured == {'group_id': spec.scratch_group_id}
    assert _calls(built)[-1] == 'close'


@pytest.mark.asyncio
async def test_embedding_only_builds_no_index(mock_config):
    factory = SpyFactory()

    async with open_arm_backend(
        llm_spec(),
        mock_config,
        _settings(IndexConfiguration.EMBEDDING_ONLY),
        backend_factory=factory,
    ):
        pass

    (built,) = factory.built
    assert _calls(built) == ['initialize', 'close']


@pytest.mark.asyncio
async def test_a_failed_index_build_raises_naming_the_graph_and_still_closes(mock_config):
    spec = llm_spec()
    factory = SpyFactory(_provisioned(failed=(UNBUILT,)))

    with pytest.raises(IndexBuildError) as caught:
        async with open_arm_backend(
            spec, mock_config, _settings(IndexConfiguration.WITH_INDICES), backend_factory=factory
        ):
            pytest.fail('a backend whose index build failed must never be yielded')

    assert caught.value.group_id == spec.scratch_group_id
    assert caught.value.failed == (UNBUILT,)
    assert 'boom' in str(caught.value)
    (built,) = factory.built
    assert _calls(built) == ['initialize', 'ensure_indices', 'close']


class StubQueryEmbedder(EmbedderClient):
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_an_embedding_backend_is_refused_a_bypassed_name_before_any_backend_is_built(
    mock_config,
):
    bypassed = EmbeddingArmSpec.model_construct(
        **(dict(embedding_spec()) | {'scratch_group_id': 'dark_factory'})
    )
    factory = SpyFactory()

    with pytest.raises(ScratchGuardError) as caught:
        async with open_embedding_arm_backend(
            bypassed, mock_config, StubQueryEmbedder(), backend_factory=factory
        ):
            pytest.fail('a guarded backend must never be yielded')

    assert caught.value.checkpoint is GuardCheckpoint.SEARCH
    assert factory.configs == []


@pytest.mark.asyncio
async def test_the_embedding_backend_is_built_from_the_arms_embedding_config(mock_config):
    spec = embedding_spec(embedding_dim=768)
    factory = SpyFactory()

    async with open_embedding_arm_backend(
        spec, mock_config, StubQueryEmbedder(), backend_factory=factory
    ) as backend:
        (built,) = factory.built
        assert backend is built

    (arm_config,) = factory.configs
    assert arm_config.model_dump() == embedding_arm_config(spec, mock_config).model_dump()
    assert arm_config.embedder.model == spec.model_id
    assert arm_config.embedder.dimensions == 768
    assert arm_config.llm.model_dump() == mock_config.llm.model_dump()


@pytest.mark.asyncio
async def test_the_embedding_backend_searches_through_the_query_embedder_and_builds_no_index(
    mock_config,
):
    query_embedder = StubQueryEmbedder()
    factory = SpyFactory()

    async with open_embedding_arm_backend(
        embedding_spec(), mock_config, query_embedder, backend_factory=factory
    ):
        pass

    (built,) = factory.built
    assert _calls(built) == ['initialize', 'close']
    (_, initialized), _ = built.lifecycle
    assert initialized['skip_maintenance'] is True
    assert initialized['embedder'] is query_embedder
    assert initialized['llm_client'] is None


@pytest.mark.asyncio
async def test_the_embedding_backend_closes_when_its_body_raises(mock_config):
    factory = SpyFactory()

    with pytest.raises(RuntimeError, match='probe failed'):
        async with open_embedding_arm_backend(
            embedding_spec(), mock_config, StubQueryEmbedder(), backend_factory=factory
        ):
            raise RuntimeError('probe failed')

    (built,) = factory.built
    assert _calls(built) == ['initialize', 'close']


@pytest.mark.asyncio
async def test_the_embedding_backend_closes_when_initialize_raises(mock_config):
    factory = SpyFactory(initialize_error=RuntimeError('falkordb unreachable'))

    with pytest.raises(RuntimeError, match='falkordb unreachable'):
        async with open_embedding_arm_backend(
            embedding_spec(), mock_config, StubQueryEmbedder(), backend_factory=factory
        ):
            pytest.fail('a backend whose initialize failed must never be yielded')

    (built,) = factory.built
    assert _calls(built) == ['initialize', 'close']


@pytest.mark.asyncio
async def test_build_scratch_indices_ensures_the_indices_exactly_once():
    backend = SpyBackend(_provisioned())

    await build_scratch_indices(backend, 'evalmem_bge_m3')

    assert backend.lifecycle == [('ensure_indices', {'group_id': 'evalmem_bge_m3'})]


@pytest.mark.asyncio
async def test_build_scratch_indices_raises_on_a_failed_spec():
    backend = SpyBackend(_provisioned(failed=(UNBUILT,)))

    with pytest.raises(IndexBuildError) as caught:
        await build_scratch_indices(backend, 'evalmem_bge_m3')

    assert caught.value.group_id == 'evalmem_bge_m3'
    assert caught.value.failed == (UNBUILT,)
    assert _calls(backend) == ['ensure_indices']


@pytest.mark.asyncio
async def test_build_scratch_indices_refuses_a_live_graph_before_ensuring_anything():
    backend = SpyBackend(_provisioned())

    with pytest.raises(ScratchGuardError) as caught:
        await build_scratch_indices(backend, 'dark_factory')

    assert caught.value.checkpoint is GuardCheckpoint.INDEX_BUILD
    assert backend.lifecycle == []
