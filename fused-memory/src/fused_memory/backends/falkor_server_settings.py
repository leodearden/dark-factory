"""The FalkorDB server settings fused-memory depends on, read live and never set (tasks 6330, 6848).

The query TIMEOUT and AOF persistence are durable only in
fused-memory/docker/docker-compose.yml. A container restart that loses them is
otherwise noticed only through "Query timed out" dead letters, so fused-memory
reads both, warns at startup, and reports them in get_status.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_NO_TIMEOUT = 0


class _RedisConfigReads(Protocol):
    async def config_get(self, name: str) -> Any: ...


class FalkorConfigReads(Protocol):
    """The two reads this module issues: GRAPH.CONFIG GET on the client, CONFIG GET on its connection."""

    async def config_get(self, name: str) -> Any: ...

    @property
    def connection(self) -> _RedisConfigReads: ...


@dataclass(frozen=True)
class FalkorServerSettings:
    expected_graph_timeout_ms: int
    graph_timeout_ms: int | None
    appendonly: str | None
    read_errors: tuple[str, ...] = ()

    @property
    def shortfalls(self) -> tuple[str, ...]:
        found: list[str] = []
        timeout = self.graph_timeout_ms
        if timeout is not None and timeout != _NO_TIMEOUT and timeout < self.expected_graph_timeout_ms:
            found.append(
                f'GRAPH.CONFIG TIMEOUT is {timeout} ms, below the expected '
                f'{self.expected_graph_timeout_ms} ms; heavy Graphiti queries will time out'
            )
        if self.appendonly is not None and self.appendonly != 'yes':
            found.append(
                f'appendonly is {self.appendonly!r}, not yes; writes since the last RDB '
                'snapshot are lost on a crash'
            )
        return tuple(found)

    @property
    def ok(self) -> bool:
        return not self.shortfalls and not self.read_errors

    def to_status(self) -> dict[str, Any]:
        return {
            'graph_timeout_ms': self.graph_timeout_ms,
            'expected_graph_timeout_ms': self.expected_graph_timeout_ms,
            'appendonly': self.appendonly,
            'ok': self.ok,
            'shortfalls': list(self.shortfalls),
            'read_errors': list(self.read_errors),
        }


async def read_falkor_server_settings(
    client: FalkorConfigReads | None, *, expected_graph_timeout_ms: int,
) -> FalkorServerSettings:
    if client is None:
        return FalkorServerSettings(
            expected_graph_timeout_ms, None, None, ('FalkorDB client not initialized',),
        )
    errors: list[str] = []
    timeout: int | None = None
    appendonly: str | None = None
    try:
        timeout = int(_text(await client.config_get('TIMEOUT')))
    except Exception as exc:
        logger.warning('Could not read FalkorDB GRAPH.CONFIG TIMEOUT: %r', exc)
        errors.append(f'GRAPH.CONFIG GET TIMEOUT: {exc!r}')
    try:
        appendonly = _text((await client.connection.config_get('appendonly'))['appendonly'])
    except Exception as exc:
        logger.warning('Could not read FalkorDB CONFIG appendonly: %r', exc)
        errors.append(f'CONFIG GET appendonly: {exc!r}')
    return FalkorServerSettings(expected_graph_timeout_ms, timeout, appendonly, tuple(errors))


def warn_on_shortfalls(settings: FalkorServerSettings) -> None:
    for shortfall in settings.shortfalls:
        logger.warning(
            'FalkorDB server setting is not what fused-memory expects: %s. The durable '
            'settings live in fused-memory/docker/docker-compose.yml; a mismatch means the '
            'running container does not carry them (see OPERATIONS.md "FalkorDB settings").',
            shortfall,
        )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
