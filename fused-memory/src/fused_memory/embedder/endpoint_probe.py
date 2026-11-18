"""Startup probe of the embedder endpoint (plans/local-embedder-cutover-prd.md §6 C3).

Calls ``/v1/models`` and ``/v1/embeddings`` directly through the openai SDK.
graphiti's OpenAIEmbedder truncates to ``embedding_dim``, so only the raw
response length measures the endpoint's native width. The probe reports what
it finds as typed faults: it changes no config, never falls back to an
ambient endpoint, and never raises for a network or HTTP fault.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import openai

from fused_memory.config.schema import EmbedderConfig

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0
"""Per-request bound; the SDK's own retries are off, so this bounds each call."""

_PROBE_INPUT = 'fused-memory embedder probe'


class ProbeFaultKind(StrEnum):
    NOT_CONFIGURED = 'not_configured'
    UNREACHABLE = 'unreachable'
    ENDPOINT_ERROR = 'endpoint_error'
    MODEL_NOT_SERVED = 'model_not_served'
    WIDTH_MISMATCH = 'width_mismatch'
    PROBE_FAILED = 'probe_failed'


@dataclass(frozen=True)
class EmbedderProbeFault:
    kind: ProbeFaultKind
    expected: Any
    observed: Any
    detail: str

    def as_status(self) -> dict[str, Any]:
        return {
            'kind': self.kind.value,
            'expected': _as_json(self.expected),
            'observed': _as_json(self.observed),
            'detail': self.detail,
        }


@dataclass(frozen=True)
class EmbedderProbeResult:
    model: str
    endpoint: str | None
    dimensions: int
    reachable: bool | None
    """None when the endpoint was never contacted."""
    served_models: tuple[str, ...]
    native_width: int | None
    faults: tuple[EmbedderProbeFault, ...]
    probed_at: str

    @property
    def ok(self) -> bool:
        return not self.faults

    def as_status(self) -> dict[str, Any]:
        return {
            'status': 'ok' if self.ok else 'degraded',
            'model': self.model,
            'endpoint': self.endpoint,
            'dimensions': self.dimensions,
            'reachable': self.reachable,
            'native_width': self.native_width,
            'served_models': list(self.served_models),
            'errors': [fault.as_status() for fault in self.faults],
            'probed_at': self.probed_at,
        }


def _as_json(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _endpoint(embedder: EmbedderConfig) -> str | None:
    provider = embedder.providers.openai
    return provider.api_url if provider is not None else None


def _result(
    embedder: EmbedderConfig,
    *,
    reachable: bool | None,
    faults: list[EmbedderProbeFault],
    served_models: tuple[str, ...] = (),
    native_width: int | None = None,
) -> EmbedderProbeResult:
    return EmbedderProbeResult(
        model=embedder.model,
        endpoint=_endpoint(embedder),
        dimensions=embedder.dimensions,
        reachable=reachable,
        served_models=served_models,
        native_width=native_width,
        faults=tuple(faults),
        probed_at=datetime.now(UTC).isoformat(),
    )


def probe_embedder_endpoint(
    embedder: EmbedderConfig, *, timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> EmbedderProbeResult:
    """Measure the configured endpoint against ``embedder``. Blocking; run it off the loop.

    An incomplete provider block constructs no client at all, so the SDK's
    own OPENAI_API_KEY / OPENAI_BASE_URL fallback can never fire (D11).
    """
    provider = embedder.providers.openai
    if embedder.provider != 'openai' or provider is None or not provider.api_key:
        return _result(embedder, reachable=None, faults=[EmbedderProbeFault(
            ProbeFaultKind.NOT_CONFIGURED, expected='an openai provider block with an api_key',
            observed=None, detail='embedder.providers.openai is absent or has no api_key',
        )])
    with openai.OpenAI(
        api_key=provider.api_key,
        base_url=provider.api_url,
        timeout=timeout_seconds,
        max_retries=0,
    ) as client:
        return _measure(client, embedder)


def _measure(client: openai.OpenAI, embedder: EmbedderConfig) -> EmbedderProbeResult:
    faults: list[EmbedderProbeFault] = []
    try:
        served = tuple(model.id for model in client.models.list())
    except openai.APIConnectionError as exc:
        return _result(embedder, reachable=False, faults=[_unreachable(exc)])
    except openai.APIError as exc:
        served = ()
        faults.append(_endpoint_error('GET /models', exc))
    else:
        if embedder.model not in served:
            faults.append(EmbedderProbeFault(
                ProbeFaultKind.MODEL_NOT_SERVED, expected=embedder.model, observed=served,
                detail='GET /models does not list the configured embedder.model',
            ))

    native_width = None
    try:
        response = client.embeddings.create(
            model=embedder.model, input=[_PROBE_INPUT], encoding_format='float',
        )
    except openai.APIConnectionError as exc:
        return _result(
            embedder, reachable=False, faults=[*faults, _unreachable(exc)], served_models=served,
        )
    except openai.APIError as exc:
        faults.append(_endpoint_error('POST /embeddings', exc))
    else:
        native_width = len(response.data[0].embedding) if response.data else None
        if native_width != embedder.dimensions:
            faults.append(EmbedderProbeFault(
                ProbeFaultKind.WIDTH_MISMATCH, expected=embedder.dimensions, observed=native_width,
                detail='POST /embeddings returned a vector of a width other than embedder.dimensions',
            ))
    return _result(
        embedder, reachable=True, faults=faults, served_models=served, native_width=native_width,
    )


def _unreachable(exc: openai.APIConnectionError) -> EmbedderProbeFault:
    return EmbedderProbeFault(
        ProbeFaultKind.UNREACHABLE, expected=None, observed=None,
        detail=f'{type(exc).__name__}: {exc}',
    )


def _endpoint_error(request: str, exc: openai.APIError) -> EmbedderProbeFault:
    status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
    return EmbedderProbeFault(
        ProbeFaultKind.ENDPOINT_ERROR, expected=None, observed=status,
        detail=f'{request}: {type(exc).__name__}: {exc.message}',
    )


class EmbedderEndpointProbe:
    """Runs the probe once in a worker thread and caches its result for get_status."""

    def __init__(
        self, embedder: EmbedderConfig, *, timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._embedder = embedder
        self._timeout_seconds = timeout_seconds
        self._task: asyncio.Task[EmbedderProbeResult] | None = None
        self._result: EmbedderProbeResult | None = None

    def start(self) -> None:
        """Schedule the probe on the running loop without awaiting it; idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            asyncio.to_thread(
                probe_embedder_endpoint, self._embedder, timeout_seconds=self._timeout_seconds,
            ),
            name='embedder-endpoint-probe',
        )
        self._task.add_done_callback(self._record)

    async def wait(self) -> EmbedderProbeResult | None:
        """The finished result; None only if the probe task was cancelled."""
        if self._task is None:
            raise RuntimeError('EmbedderEndpointProbe.wait() called before start()')
        await asyncio.wait({self._task})
        return self._result

    def status_block(self) -> dict[str, Any]:
        if self._result is not None:
            return self._result.as_status()
        return {
            'status': 'pending',
            'model': self._embedder.model,
            'endpoint': _endpoint(self._embedder),
            'dimensions': self._embedder.dimensions,
            'reachable': None,
            'native_width': None,
            'served_models': [],
            'errors': [],
            'probed_at': None,
        }

    def _record(self, task: asyncio.Task[EmbedderProbeResult]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error('embedder endpoint probe failed', exc_info=exc)
            self._result = _result(self._embedder, reachable=None, faults=[EmbedderProbeFault(
                ProbeFaultKind.PROBE_FAILED, expected=None, observed=None,
                detail=f'{type(exc).__name__}: {exc}',
            )])
            return
        self._result = task.result()
        if self._result.ok:
            logger.info('embedder endpoint probe ok: %s', json.dumps(self._result.as_status()))
        else:
            logger.warning(
                'embedder endpoint probe degraded: %s',
                json.dumps([fault.as_status() for fault in self._result.faults]),
            )
