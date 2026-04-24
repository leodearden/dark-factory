"""Maintenance utility: seed autopilot_video triage guardrails into Mem0.

Task 361 authored ATTRIBUTION_STUB_GUARDRAIL and NAV_HINTS_GUARDRAIL constants
in /home/leo/src/autopilot-video/autopilot/guardrails.py but never performed
the live Mem0 write. Task 1040 completes that write. This module is the
reusable, testable encapsulation; the live MCP add_memory calls are done by
the implementer (see plan step 3).

Usage::

    uv run python -m fused_memory.maintenance.seed_autopilot_video_triage_guardrails \\
        [--config /path/to/config.yaml]
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from fused_memory.maintenance._utils import maintenance_service
from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

# Resolve the default source path from AUTOPILOT_VIDEO_ROOT if set, so the
# CLI default is not anchored to a single developer's filesystem layout.
# On any machine (CI, fresh clone, other devs) set:
#   export AUTOPILOT_VIDEO_ROOT=/path/to/autopilot-video
# When unset, the path falls back to the original author's workstation layout.
_DEFAULT_AUTOPILOT_VIDEO_GUARDRAILS_PATH = (
    Path(os.environ['AUTOPILOT_VIDEO_ROOT']) / 'autopilot' / 'guardrails.py'
    if 'AUTOPILOT_VIDEO_ROOT' in os.environ
    else Path('/home/leo/src/autopilot-video/autopilot/guardrails.py')
)


def load_guardrail_payloads(
    agent_id: str,
    *,
    source_path: Path | None = None,
) -> list[dict]:
    """Load autopilot_video's two triage guardrail payloads by file path.

    Uses importlib.util to exec the source module without sys.path pollution
    or requiring autopilot-video to be installed as a package. The source
    module has zero cross-module imports, so this is safe.

    Args:
        agent_id: Injected into each returned payload's `agent_id` field.
        source_path: Override for the guardrails.py location. Defaults to
            /home/leo/src/autopilot-video/autopilot/guardrails.py.

    Returns:
        List of two add_memory-ready payload dicts (one per guardrail),
        each independent of the source module's internal state.

    Raises:
        FileNotFoundError: If source_path does not exist.
    """
    path = source_path or _DEFAULT_AUTOPILOT_VIDEO_GUARDRAILS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f'autopilot_video guardrails source not found at {path!s}. '
            f'Set the AUTOPILOT_VIDEO_ROOT env var to point at your '
            f'autopilot-video checkout, or pass source_path= to override.'
        )
    spec = importlib.util.spec_from_file_location(
        'autopilot_video_triage_guardrails_source', path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not build ModuleSpec for {path!s}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Delegate to the source module's canonical payload builder, then
    # deep-copy the result so the caller's payloads are fully independent
    # of the source module's internal state.  This makes non-aliasing the
    # loader's own guarantee rather than relying on whatever the source
    # module's get_guardrail_payloads happens to do internally.
    return copy.deepcopy(list(module.get_guardrail_payloads(agent_id)))


@dataclass
class SeedReport:
    """Aggregate outcome of a guardrail-seed run."""

    project_id: str
    memory_ids_by_name: dict[str, list[str]] = field(default_factory=dict)
    stores_written_by_name: dict[str, list[str]] = field(default_factory=dict)


class PartialSeedError(RuntimeError):
    """Raised when a guardrail seed write returns empty memory_ids.

    The ``partial_report`` attribute records which guardrails were
    successfully written before the failure occurred, so an operator
    can identify which Graphiti episodes may need cleanup before
    retrying the seed.  Check ``partial_report.memory_ids_by_name``
    for completed writes.
    """

    def __init__(self, message: str, partial_report: SeedReport) -> None:
        super().__init__(message)
        self.partial_report = partial_report


class SeedManager:
    """Perform the Mem0 seed writes for the triage guardrails."""

    def __init__(self, service: MemoryService) -> None:
        self.service = service

    async def seed(
        self,
        agent_id: str,
        *,
        source_path: Path | None = None,
    ) -> SeedReport:
        payloads = load_guardrail_payloads(agent_id, source_path=source_path)
        project_ids = {p['project_id'] for p in payloads}
        if len(project_ids) != 1:
            raise RuntimeError(
                f'Expected exactly one project_id across payloads, got {project_ids!r}'
            )
        report = SeedReport(project_id=project_ids.pop())
        for payload in payloads:
            name = payload['metadata']['name']
            response = await self.service.add_memory(**payload)
            if not response.memory_ids:
                raise PartialSeedError(
                    f'Got empty memory_ids for guardrail {name!r}. '
                    f'This indicates the Task 360 Mem0 memory_ids=[] bug has '
                    f'recurred — do NOT retry blindly; investigate the '
                    f'Mem0 synchronous write path before re-running. '
                    f'Completed writes before this failure: '
                    f'{list(report.memory_ids_by_name.keys())!r}',
                    partial_report=report,
                )
            report.memory_ids_by_name[name] = list(response.memory_ids)
            report.stores_written_by_name[name] = [
                s.value if hasattr(s, 'value') else str(s)
                for s in response.stores_written
            ]
        return report


async def run_seed(
    config_path: str | None = None,
    agent_id: str = 'claude-task-1040-implementer',
    source_path: Path | None = None,
) -> SeedReport:
    """CLI-callable entrypoint. Loads config, constructs MemoryService, seeds."""
    async with maintenance_service(config_path) as (_config, service):
        manager = SeedManager(service=service)
        return await manager.seed(agent_id=agent_id, source_path=source_path)


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Seed autopilot_video triage guardrails into Mem0.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    parser.add_argument(
        '--agent-id',
        default='claude-task-1040-implementer',
        help='Agent ID to record as the writer (default: claude-task-1040-implementer).',
    )
    parser.add_argument(
        '--source-path',
        default=None,
        help='Override for the autopilot-video guardrails.py location.',
    )
    args = parser.parse_args()

    report = asyncio.run(run_seed(
        config_path=args.config,
        agent_id=args.agent_id,
        source_path=Path(args.source_path) if args.source_path else None,
    ))
    print(
        f'Seed complete: project_id={report.project_id} '
        f'memory_ids_by_name={report.memory_ids_by_name}'
    )
