"""An arm's texts turned into unit vectors, and timed (arm_harness/arm_embedder.py)."""

import asyncio
import math
from collections.abc import Mapping

import pytest
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from pydantic import ValidationError

from arm_harness._fakes import QWEN_QUERY_PREFIX, embedding_spec
from fused_memory.arm_harness.arm_config import LOCAL_ARM_API_KEY
from fused_memory.arm_harness.arm_embedder import (
    ArmEmbedder,
    EmbedSettings,
    QueryEmbedder,
    build_arm_embedder,
    document_text,
    measure_query_latency,
    query_text,
)
from fused_memory.arm_harness.metrics_record import EmbeddingMetricId

DIM = 4
SETTINGS = EmbedSettings(batch_size=3, concurrency=2)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class FakeInner(EmbedderClient):
    """graphiti's EmbedderClient surface. A text embeds to (3k, 4k, 0, …), k = 1 + len(text)."""

    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        seconds_per_call: float = 0.0,
        seconds_by_text: Mapping[str, float] | None = None,
        cold_start_s: float = 0.0,
        poison: frozenset[str] = frozenset(),
        short: frozenset[str] = frozenset(),
        pause: bool = False,
    ) -> None:
        self.clock = clock or FakeClock()
        self.seconds_per_call = seconds_per_call
        self.seconds_by_text = dict(seconds_by_text or {})
        self.cold_start_s = cold_start_s
        self.poison = poison
        self.short = short
        self.pause = pause
        self.batches: list[list[str]] = []
        self.singles: list[str] = []
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def _vector(self, text: str) -> list[float]:
        if text in self.poison:
            raise RuntimeError(f'cannot embed {text!r}')
        scale = 1 + len(text)
        length = DIM - 1 if text in self.short else DIM
        return [3.0 * scale, 4.0 * scale] + [0.0] * (length - 2)

    async def _call(self, texts: list[str]) -> None:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.pause:
                await asyncio.sleep(0)
            elapsed = self.seconds_per_call + sum(self.seconds_by_text.get(t, 0.0) for t in texts)
            if self.calls == 1:
                elapsed += self.cold_start_s
            self.clock.now += elapsed
        finally:
            self.in_flight -= 1

    async def create(self, input_data):
        (text,) = input_data
        self.singles.append(text)
        await self._call([text])
        return self._vector(text)

    async def create_batch(self, input_data_list):
        texts = list(input_data_list)
        self.batches.append(texts)
        await self._call(texts)
        return [self._vector(text) for text in texts]


def _arm(inner: FakeInner, *, query_prefix: str | None = None, settings=SETTINGS) -> ArmEmbedder:
    spec = embedding_spec(embedding_dim=DIM, query_prefix=query_prefix)
    return ArmEmbedder(inner, spec, settings, clock=inner.clock)


def _items(count: int) -> list[tuple[str, str]]:
    return [(f'k{i}', f'text {i}\nline') for i in range(count)]


def _l2(vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


# --- texts ----------------------------------------------------------------------------


def test_document_text_flattens_newlines_as_graphiti_and_mem0_do():
    assert document_text('a\nb\nc') == 'a b c'
    assert document_text('no newline') == 'no newline'


def test_query_text_flattens_the_query_and_keeps_the_prefix_newline():
    assert query_text(None, 'which\nmodel') == 'which model'
    assert query_text(QWEN_QUERY_PREFIX, 'which\nmodel') == QWEN_QUERY_PREFIX + 'which model'
    assert '\nQuery: ' in query_text(QWEN_QUERY_PREFIX, 'q')


@pytest.mark.parametrize('field', ['batch_size', 'concurrency'])
def test_embed_settings_refuse_a_non_positive_size(field):
    with pytest.raises(ValidationError, match=field):
        EmbedSettings.model_validate({'batch_size': 3, 'concurrency': 2} | {field: 0})


def test_embed_settings_are_frozen():
    with pytest.raises(ValidationError):
        SETTINGS.batch_size = 9  # type: ignore[misc]


# --- documents --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_documents_go_in_batches_of_exactly_batch_size_at_bounded_concurrency():
    inner = FakeInner(pause=True)

    result = await _arm(inner).embed_documents(_items(7))

    assert [len(batch) for batch in inner.batches] == [3, 3, 1]
    assert inner.max_in_flight == SETTINGS.concurrency
    assert inner.singles == []
    assert inner.batches[0][0] == 'text 0 line'
    assert set(result.vectors) == {f'k{i}' for i in range(7)}
    assert result.failures == ()


@pytest.mark.asyncio
async def test_document_vectors_are_unit_and_the_raw_norms_are_recorded():
    inner = FakeInner()

    result = await _arm(inner).embed_documents([('a', 'abc'), ('b', 'a\nb')])

    for key in ('a', 'b'):
        assert abs(_l2(result.vectors[key]) - 1.0) < 1e-9
        assert result.vectors[key] == pytest.approx((0.6, 0.8, 0.0, 0.0))
    assert result.raw_norms == {'a': pytest.approx(20.0), 'b': pytest.approx(20.0)}


@pytest.mark.asyncio
async def test_embed_seconds_is_the_wall_clock_of_the_whole_call():
    inner = FakeInner(seconds_per_call=0.5)

    result = await _arm(inner).embed_documents(_items(7))

    assert result.embed_seconds == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_a_failing_batch_is_retried_item_by_item_so_one_bad_text_fails_alone():
    poisoned = 'text 4\nline'
    inner = FakeInner(poison=frozenset({document_text(poisoned)}))

    result = await _arm(inner).embed_documents(_items(7))

    assert result.failures == (('k4', 'RuntimeError'),)
    assert inner.singles == ['text 3 line', 'text 4 line', 'text 5 line']
    assert set(result.vectors) == {f'k{i}' for i in range(7)} - {'k4'}
    assert set(result.raw_norms) == set(result.vectors)


@pytest.mark.asyncio
async def test_a_wrong_length_vector_is_a_recorded_failure_never_padded():
    inner = FakeInner(short=frozenset({'text 1 line'}))

    result = await _arm(inner).embed_documents(_items(3))

    assert result.failures == (('k1', 'VectorInvariantError'),)
    assert 'k1' not in result.vectors
    assert all(len(vector) == DIM for vector in result.vectors.values())


# --- queries ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_query_applies_the_prefix_once_and_times_the_call():
    inner = FakeInner(seconds_per_call=0.25)

    vector, latency_ms = await _arm(inner, query_prefix=QWEN_QUERY_PREFIX).embed_query('what\nnow')

    assert inner.singles == [QWEN_QUERY_PREFIX + 'what now']
    assert vector == pytest.approx((0.6, 0.8, 0.0, 0.0))
    assert latency_ms == pytest.approx(250.0)


@pytest.mark.asyncio
async def test_embed_query_without_a_prefix_sends_the_flattened_query():
    inner = FakeInner()

    await _arm(inner).embed_query('what\nnow')

    assert inner.singles == ['what now']


@pytest.mark.asyncio
async def test_query_embedder_is_graphitis_embedder_surface_over_the_arm():
    inner = FakeInner()
    adapter = QueryEmbedder(_arm(inner, query_prefix=QWEN_QUERY_PREFIX))

    vector = await adapter.create(input_data=['what now'])

    assert isinstance(adapter, EmbedderClient)
    assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert inner.singles == [QWEN_QUERY_PREFIX + 'what now']
    assert inner.singles[0].count(QWEN_QUERY_PREFIX) == 1


# --- query latency ----------------------------------------------------------------------


def _queries(count: int = 20) -> list[str]:
    return [f'q{i:02d}' for i in range(1, count + 1)]


def _latency_inner(**overrides) -> FakeInner:
    by_text = {query: int(query[1:]) / 1000 for query in _queries()}
    return FakeInner(seconds_by_text=by_text, cold_start_s=5.0, **overrides)


@pytest.mark.asyncio
async def test_query_latency_is_the_nearest_rank_p95_after_one_discarded_warm_up():
    inner = _latency_inner()

    measured = await measure_query_latency(_arm(inner), _queries())

    assert len(inner.singles) == 21
    metric = measured.metric
    assert metric is not None
    assert metric.metric_id == EmbeddingMetricId.QUERY_EMBED_LATENCY_P95
    assert metric.kind == 'scalar'
    assert metric.n == 20
    assert metric.value == pytest.approx(19.0)
    assert measured.failures == 0


@pytest.mark.asyncio
async def test_a_failed_query_contributes_no_latency_and_is_counted():
    inner = _latency_inner(poison=frozenset({'q07', 'q13'}))

    measured = await measure_query_latency(_arm(inner), _queries())

    assert measured.failures == 2
    assert measured.metric is not None
    assert measured.metric.n == 20
    assert measured.metric.value == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_query_latency_runs_at_most_concurrency_queries_in_flight():
    inner = FakeInner(pause=True)

    await measure_query_latency(_arm(inner), _queries(10), concurrency=3)

    assert inner.max_in_flight == 3


@pytest.mark.asyncio
async def test_query_latency_of_all_failed_queries_has_no_metric():
    inner = FakeInner(poison=frozenset(_queries(4)))

    measured = await measure_query_latency(_arm(inner), _queries(4))

    assert measured.metric is None
    assert measured.failures == 4


@pytest.mark.asyncio
async def test_query_latency_refuses_an_empty_query_set():
    with pytest.raises(ValueError, match='no queries'):
        await measure_query_latency(_arm(FakeInner()), [])


# --- construction -----------------------------------------------------------------------


def test_build_arm_embedder_builds_a_local_arm_through_the_production_path(mock_config):
    spec = embedding_spec(
        model_id='qwen3-embedding-0.6b',
        serving={'stack': 'vllm', 'base_url': 'http://127.0.0.1:8414/v1'},
        embedding_dim=1024,
        query_prefix=QWEN_QUERY_PREFIX,
    )

    arm = build_arm_embedder(spec, mock_config, settings=SETTINGS)

    assert arm.spec == spec
    assert arm.settings == SETTINGS
    assert isinstance(arm.inner, OpenAIEmbedder)
    assert arm.inner.config.api_key == LOCAL_ARM_API_KEY
    assert arm.inner.config.base_url == 'http://127.0.0.1:8414/v1'
    assert arm.inner.config.embedding_model == 'qwen3-embedding-0.6b'
    assert arm.inner.config.embedding_dim == 1024


def test_build_arm_embedder_refuses_a_metered_arm_without_a_key(mock_config):
    base = mock_config.model_copy(deep=True)
    base.embedder.providers.openai = None
    spec = embedding_spec(
        arm_id='incumbent-embed-a',
        model_id='text-embedding-3-small',
        serving={'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        embedding_dim=1536,
        preregistration_sha=None,
        arm_role='control',
    )

    with pytest.raises(ValueError, match='incumbent-embed-a'):
        build_arm_embedder(spec, base, settings=SETTINGS)
