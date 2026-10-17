"""Tell the offline lane whether the services fused-memory's integration tests use are usable.

Exit 0, printing nothing to stdout, when every checked dependency does the work the tests need. Otherwise
exit ``OUTAGE_EXIT_STATUS`` after printing ONE stdout line ``<dependency>: <step>: <cause>`` for the first
dependency that is not usable; the line is built from a fixed vocabulary so that it is identical on every
run of the same outage. Diagnostics go to stderr, never a traceback.

Usage: ``python scripts/lane_dependency_probe.py [dependency ...]`` from ``fused-memory/``; naming
dependencies checks only those, always in canonical order.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from cleanup_test_collections import PREFIX, hold_lease
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.lane_probe import OUTAGE_EXIT_STATUS

from fused_memory.arm_harness.checks import SMOKE_MESSAGES, SmokeEntities
from fused_memory.backends.graphiti_client import build_embedder, build_llm_client
from fused_memory.config.schema import FusedMemoryConfig

DEPENDENCIES = ('qdrant', 'falkordb', 'embedder', 'chat-model')
CHECK_TIMEOUT_SECS = 30
CLEANUP_TIMEOUT_SECS = 5
LEASE_OWNER = 'lane-dependency-probe'
PROBE_VALUE = 'lane dependency probe'
SCRATCH_PAYLOAD = {'probe': 'lane-dependency'}
BROKEN_CONFIG_EXIT_STATUS = 1


class NotConfigured(Exception):
    """The dependency cannot be used because the configuration names no way to reach it."""


class WrongAnswer(Exception):
    """The dependency answered, but not with what was just given to it."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


@dataclass
class Progress:
    step: str = 'connect'


@dataclass(frozen=True)
class Failure:
    line: str
    detail: str


Check = Callable[[FusedMemoryConfig, Progress], Awaitable[None]]


def cause_of(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return f'no answer within {CHECK_TIMEOUT_SECS} s'
    if isinstance(error, UnexpectedResponse):
        return f'HTTP {error.status_code}'
    if isinstance(error, WrongAnswer):
        return error.cause
    if isinstance(error, ResponseHandlingException):
        return type(error.source).__name__
    return type(error).__name__


def failure_of(dependency: str, progress: Progress, error: BaseException) -> Failure:
    detail = ' '.join(repr(error).split())
    if isinstance(error, NotConfigured):
        return Failure(f'{dependency}: not configured ({error})', detail)
    return Failure(f'{dependency}: {progress.step}: {cause_of(error)}', detail)


@contextlib.asynccontextmanager
async def scratch(progress: Progress, remove: Callable[[], Awaitable[object]]) -> AsyncIterator[None]:
    """Run the body, then remove the scratch object; on failure remove it best-effort and re-raise."""
    try:
        yield
    except BaseException:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(remove(), CLEANUP_TIMEOUT_SECS)
        raise
    progress.step = 'delete'
    await remove()


async def check_qdrant(cfg: FusedMemoryConfig, progress: Progress) -> None:
    name = f'{PREFIX}lane_probe_{os.getpid()}'
    client = AsyncQdrantClient(
        url=cfg.mem0.qdrant_url, timeout=CHECK_TIMEOUT_SECS, check_compatibility=False
    )
    try:
        with hold_lease(LEASE_OWNER):
            progress.step = 'get collections'
            await client.get_collections()
            async with scratch(progress, lambda: client.delete_collection(name)):
                progress.step = 'create collection'
                await client.create_collection(
                    name, vectors_config=VectorParams(size=4, distance=Distance.COSINE)
                )
                progress.step = 'upsert'
                await client.upsert(
                    name,
                    points=[PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload=SCRATCH_PAYLOAD)],
                    wait=True,
                )
                progress.step = 'retrieve'
                points = await client.retrieve(name, ids=[1])
                if [point.payload for point in points] != [SCRATCH_PAYLOAD]:
                    raise WrongAnswer('read-back mismatch')
    finally:
        await client.close()


async def check_falkordb(cfg: FusedMemoryConfig, progress: Progress) -> None:
    endpoint = cfg.graphiti.falkordb
    address = urlparse(endpoint.uri)
    client = FalkorDB(
        host=address.hostname or 'localhost',
        port=address.port or 6379,
        password=endpoint.password,
        socket_connect_timeout=CHECK_TIMEOUT_SECS,
        socket_timeout=CHECK_TIMEOUT_SECS,
    )
    graph = client.select_graph(f'_lane_dependency_probe_{os.getpid()}')
    try:
        async with scratch(progress, graph.delete):
            progress.step = 'write graph'
            await graph.query('CREATE (:LaneProbe {value: $value})', {'value': PROBE_VALUE})
            progress.step = 'read graph'
            result = await graph.query('MATCH (probe:LaneProbe) RETURN probe.value')
            if result.result_set != [[PROBE_VALUE]]:
                raise WrongAnswer('read-back mismatch')
    finally:
        await client.aclose()


async def check_embedder(cfg: FusedMemoryConfig, progress: Progress) -> None:
    embedder = build_embedder(cfg)
    if embedder is None:
        raise NotConfigured('no provider or api key')
    progress.step = 'embed'
    embedding = await embedder.create(input_data=[PROBE_VALUE])
    if not embedding:
        raise WrongAnswer('empty embedding')


async def check_chat_model(cfg: FusedMemoryConfig, progress: Progress) -> None:
    client = build_llm_client(cfg)
    if client is None:
        raise NotConfigured('no provider or api key')
    progress.step = 'complete'
    await client.generate_response(list(SMOKE_MESSAGES), response_model=SmokeEntities)


CHECKS: dict[str, Check] = {
    'qdrant': check_qdrant,
    'falkordb': check_falkordb,
    'embedder': check_embedder,
    'chat-model': check_chat_model,
}


async def run_check(dependency: str, cfg: FusedMemoryConfig) -> Failure | None:
    progress = Progress()
    try:
        await asyncio.wait_for(CHECKS[dependency](cfg, progress), CHECK_TIMEOUT_SECS)
    except Exception as error:
        return failure_of(dependency, progress, error)
    return None


async def first_failure(dependencies: Sequence[str], cfg: FusedMemoryConfig) -> Failure | None:
    for dependency in dependencies:
        failure = await run_check(dependency, cfg)
        if failure is not None:
            return failure
    return None


def build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        'dependencies',
        nargs='*',
        choices=DEPENDENCIES,
        help='check only these dependencies (default: all, in the order listed)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    requested = build_parser().parse_args(argv).dependencies
    selected = [name for name in DEPENDENCIES if not requested or name in requested]
    try:
        cfg = FusedMemoryConfig()
    except Exception as error:
        print(f'error: config did not load: {type(error).__name__}', file=sys.stderr)
        return BROKEN_CONFIG_EXIT_STATUS
    failure = asyncio.run(first_failure(selected, cfg))
    if failure is None:
        return 0
    print(failure.line)
    print(failure.detail, file=sys.stderr)
    return OUTAGE_EXIT_STATUS


if __name__ == '__main__':
    sys.exit(run_cli(main))
