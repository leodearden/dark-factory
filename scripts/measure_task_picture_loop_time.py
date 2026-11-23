"""How long one task picture holds the event loop, against the ordinary pass's bare task fetch.

Task 10476 item 4 (INV-8 in docs/legibility/design-invariants.md). A Scheduler
is built for the project from its dark-factory-orchestrator.yaml with no event
store, an OverrideStore on a COPY of the live overrides db, and the live
fused-memory behind one persistent session that refuses every tool except
get_tasks, get_statuses, get_task and get_external_statuses. It never calls
acquire_next, finish_startup or any write.

A monitor coroutine sleeps 1 ms at a time and records how late each wake-up
is. For (a) the ordinary pass's fetch alone and (b) Scheduler.build_task_picture,
each run reports wall time, loop occupancy (the summed lateness of wake-ups
late by at least one interval, so timer jitter is not counted) and the longest
single block; the table gives the median and the max over the runs.

Usage: python scripts/measure_task_picture_loop_time.py [--config PATH] [--runs N]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import _checkout as _checkout
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.mcp_lifecycle import fm_session
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import Scheduler
from orchestrator.task_status import ACTIVE_TASK_STATUSES
from shared.mcp_session import McpSession

READ_TOOLS = frozenset({'get_tasks', 'get_statuses', 'get_task', 'get_external_statuses'})
MONITOR_INTERVAL_SECS = 0.001
DEFAULT_CONFIG = _checkout.ROOT / 'dark-factory-orchestrator.yaml'


class ReadOnlySession:
    """The live fused-memory, read tools only: any other tool raises before it is sent."""

    def __init__(self, session: McpSession) -> None:
        self._session = session

    async def call_tool(self, name: str, arguments: dict, *, timeout: float = 30) -> dict:
        if name not in READ_TOOLS:
            raise PermissionError(f'measurement session refuses {name!r}: read tools only')
        return await self._session.call_tool(name, arguments, timeout=timeout)


@dataclass(frozen=True)
class Sample:
    wall_secs: float
    occupancy_secs: float
    longest_block_secs: float
    task_count: int


async def _monitor(lateness: list[float], stop: asyncio.Event) -> None:
    while not stop.is_set():
        started = time.perf_counter()
        await asyncio.sleep(MONITOR_INTERVAL_SECS)
        lateness.append(time.perf_counter() - started - MONITOR_INTERVAL_SECS)


async def _sample(operation: Callable[[], Awaitable[int]]) -> Sample:
    lateness: list[float] = []
    stop = asyncio.Event()
    monitor = asyncio.create_task(_monitor(lateness, stop))
    await asyncio.sleep(0.05)
    lateness.clear()
    started = time.perf_counter()
    task_count = await operation()
    wall = time.perf_counter() - started
    stop.set()
    await monitor
    blocks = [late for late in lateness if late >= MONITOR_INTERVAL_SECS]
    return Sample(wall, sum(blocks), max(blocks, default=0.0), task_count)


def _overrides_copy(config: OrchestratorConfig, workdir: Path) -> OverrideStore:
    live = config.overrides_db_path
    for sidecar in (live, live.with_name(f'{live.name}-wal')):
        if sidecar.exists():
            shutil.copyfile(sidecar, workdir / sidecar.name)
    return OverrideStore(workdir / live.name)


def _row(label: str, samples: Sequence[Sample]) -> str:
    def stat(values: list[float]) -> str:
        return f'{statistics.median(values) * 1000:9.1f} {max(values) * 1000:9.1f}'

    return (
        f'{label:<26} {stat([s.wall_secs for s in samples])} '
        f'{stat([s.occupancy_secs for s in samples])} '
        f'{stat([s.longest_block_secs for s in samples])} {samples[0].task_count:7d}'
    )


def _table(samples: Mapping[str, Sequence[Sample]]) -> list[str]:
    header = f'{"":<26} {"wall ms":>19} {"occupancy ms":>19} {"longest block ms":>19} {"tasks":>7}'
    sub = f'{"":<26} {"median":>9} {"max":>9} {"median":>9} {"max":>9} {"median":>9} {"max":>9}'
    return [header, sub, *(_row(label, rows) for label, rows in samples.items())]


async def measure(config: OrchestratorConfig, runs: int, workdir: Path) -> list[str]:
    session = fm_session(config.fused_memory.url)
    await session.initialize()
    scheduler = Scheduler(
        config,
        None,
        mcp_session=ReadOnlySession(session),
        override_store=_overrides_copy(config, workdir),
        branch_work_reader=dict,
    )

    async def fetch() -> int:
        tasks = await scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES, distinguish_failure=True)
        if tasks is None:
            raise RuntimeError('task fetch failed')
        return len(tasks)

    async def picture() -> int:
        built = await scheduler.build_task_picture()
        if built is None:
            raise RuntimeError('task fetch failed')
        return len(built.tasks)

    samples: dict[str, list[Sample]] = {'(a) ordinary fetch': [], '(b) build_task_picture': []}
    try:
        for _ in range(runs):
            samples['(a) ordinary fetch'].append(await _sample(fetch))
            samples['(b) build_task_picture'].append(await _sample(picture))
    finally:
        await scheduler.aclose()
    return _table(samples)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(f'project_root={config.project_root} runs={args.runs}')
    with tempfile.TemporaryDirectory(prefix='measure-task-picture-') as workdir:
        for line in asyncio.run(measure(config, args.runs, Path(workdir))):
            print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
