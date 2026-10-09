"""Turning an arm's texts into unit vectors through its served model, and timing it."""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from graphiti_core.embedder import EmbedderClient
from pydantic import Field
from shared.memory_eval_metrics import Metric

from fused_memory.arm_harness.arm_config import embedding_arm_config
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.llm_metrics import nearest_rank
from fused_memory.arm_harness.metrics_record import EmbeddingMetricId
from fused_memory.arm_harness.normalization import VectorInvariantError, raw_norm, unit_vector
from fused_memory.backends.graphiti_client import build_embedder
from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger(__name__)

QUERY_LATENCY_PERCENTILE = 0.95
QUERY_LATENCY_CONCURRENCY = 3

Clock = Callable[[], float]
Vector = tuple[float, ...]


def _one_line(text: str) -> str:
    return text.replace('\n', ' ')


def document_text(text: str) -> str:
    return _one_line(text)


def query_text(prefix: str | None, query: str) -> str:
    return (prefix or '') + _one_line(query)


class EmbedSettings(FrozenModel):
    batch_size: int = Field(gt=0)
    concurrency: int = Field(gt=0)


@dataclass(frozen=True)
class DocumentEmbeddings:
    vectors: Mapping[str, Vector]
    raw_norms: Mapping[str, float]
    embed_seconds: float
    failures: tuple[tuple[str, str], ...]
    """(key, error class name) for each text the arm could not embed."""


class DocumentEmbedder(Protocol):
    """What a store re-embed needs of an arm: its documents in, keyed unit vectors out."""

    async def embed_documents(self, items: Sequence[tuple[str, str]], /) -> DocumentEmbeddings: ...


@dataclass(frozen=True)
class _Embedded:
    key: str
    vector: Vector
    norm: float


@dataclass(frozen=True)
class _Failed:
    key: str
    error_class: str


class ArmEmbedder:
    """One arm's endpoint, whose every returned vector is unit and of the arm's dimension."""

    def __init__(
        self,
        inner: EmbedderClient,
        spec: EmbeddingArmSpec,
        settings: EmbedSettings,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.inner = inner
        self.spec = spec
        self.settings = settings
        self._clock = clock

    async def embed_documents(self, items: Sequence[tuple[str, str]]) -> DocumentEmbeddings:
        size = self.settings.batch_size
        batches = [items[start : start + size] for start in range(0, len(items), size)]
        gate = asyncio.Semaphore(self.settings.concurrency)
        started = self._clock()
        per_batch = await asyncio.gather(*(self._embed_batch(batch, gate) for batch in batches))
        embed_seconds = self._clock() - started
        outcomes = [outcome for batch in per_batch for outcome in batch]
        embedded = [outcome for outcome in outcomes if isinstance(outcome, _Embedded)]
        return DocumentEmbeddings(
            vectors=MappingProxyType({outcome.key: outcome.vector for outcome in embedded}),
            raw_norms=MappingProxyType({outcome.key: outcome.norm for outcome in embedded}),
            embed_seconds=embed_seconds,
            failures=tuple(
                (outcome.key, outcome.error_class)
                for outcome in outcomes
                if isinstance(outcome, _Failed)
            ),
        )

    async def embed_query(self, query: str) -> tuple[Vector, float]:
        text = query_text(self.spec.query_prefix, query)
        started = self._clock()
        raw = await self.inner.create(input_data=[text])
        latency_ms = (self._clock() - started) * 1000
        return unit_vector(raw, self.spec.embedding_dim), latency_ms

    async def _embed_batch(
        self, batch: Sequence[tuple[str, str]], gate: asyncio.Semaphore
    ) -> list[_Embedded | _Failed]:
        async with gate:
            raws = await self._batch_or_none([document_text(text) for _, text in batch])
            if raws is None:
                return [await self._embed_one(key, text) for key, text in batch]
        return [self._accept(key, raw) for (key, _), raw in zip(batch, raws, strict=True)]

    async def _batch_or_none(self, texts: list[str]) -> list[list[float]] | None:
        try:
            raws = await self.inner.create_batch(texts)
        except Exception as error:
            logger.warning(
                'arm %r: a batch of %d failed (%s); retrying it item by item',
                self.spec.arm_id, len(texts), type(error).__name__,
            )
            return None
        if len(raws) != len(texts):
            logger.warning(
                'arm %r: a batch of %d returned %d vectors; retrying it item by item',
                self.spec.arm_id, len(texts), len(raws),
            )
            return None
        return raws

    async def _embed_one(self, key: str, text: str) -> _Embedded | _Failed:
        try:
            raw = await self.inner.create(input_data=[document_text(text)])
        except Exception as error:
            return _Failed(key, type(error).__name__)
        return self._accept(key, raw)

    def _accept(self, key: str, raw: Sequence[float]) -> _Embedded | _Failed:
        try:
            vector = unit_vector(raw, self.spec.embedding_dim)
        except VectorInvariantError as error:
            return _Failed(key, type(error).__name__)
        return _Embedded(key, vector, raw_norm(raw))


class QueryEmbedder(EmbedderClient):
    """graphiti's EmbedderClient over an arm, for GraphitiBackend's search path: queries only."""

    def __init__(self, arm: ArmEmbedder) -> None:
        self.arm = arm

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        vector, _ = await self.arm.embed_query(_single_query(input_data))
        return list(vector)


def _single_query(input_data: object) -> str:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list) and len(input_data) == 1 and isinstance(input_data[0], str):
        return input_data[0]
    raise TypeError(f'QueryEmbedder embeds one query string, got {input_data!r}')


class QueryLatency(FrozenModel):
    metric: Metric | None
    failures: int


async def measure_query_latency(
    embedder: ArmEmbedder,
    queries: Sequence[str],
    concurrency: int = QUERY_LATENCY_CONCURRENCY,
) -> QueryLatency:
    if not queries:
        raise ValueError('measure_query_latency: no queries to time')
    await _warm_up(embedder, queries[0])
    gate = asyncio.Semaphore(concurrency)
    timed = await asyncio.gather(*(_latency_or_none(embedder, query, gate) for query in queries))
    latencies = sorted(ms for ms in timed if ms is not None)
    metric = (
        Metric(
            metric_id=EmbeddingMetricId.QUERY_EMBED_LATENCY_P95,
            kind='scalar',
            value=nearest_rank(latencies, QUERY_LATENCY_PERCENTILE),
            n=len(queries),
        )
        if latencies
        else None
    )
    return QueryLatency(metric=metric, failures=len(timed) - len(latencies))


async def _warm_up(embedder: ArmEmbedder, query: str) -> None:
    try:
        await embedder.embed_query(query)
    except Exception as error:
        logger.warning(
            'arm %r: the discarded warm-up query failed (%s)',
            embedder.spec.arm_id, type(error).__name__,
        )


async def _latency_or_none(
    embedder: ArmEmbedder, query: str, gate: asyncio.Semaphore
) -> float | None:
    async with gate:
        try:
            _, latency_ms = await embedder.embed_query(query)
        except Exception:
            return None
    return latency_ms


def build_arm_embedder(
    spec: EmbeddingArmSpec,
    base_config: FusedMemoryConfig,
    *,
    settings: EmbedSettings,
    clock: Clock = time.perf_counter,
) -> ArmEmbedder:
    inner = build_embedder(embedding_arm_config(spec, base_config))
    if inner is None:
        raise ValueError(
            f'arm {spec.arm_id!r}: no embedder can be built, because its openai provider '
            'block carries no api_key'
        )
    return ArmEmbedder(inner, spec, settings, clock=clock)
