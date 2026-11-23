"""Give this orchestrator's project slice its CPUWeight at startup, when the
daemon runs in a service directly under that slice, or in that service's daemon
subgroup.

The placement check establishes the unit's placement, not the caller's identity;
the caller opts in through ``Harness.run``'s ``apply_project_slice_weight``.
Contract: docs/cgroup-slice-topology.md (§Slice weight for the opt-in).
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import shutil
from pathlib import PurePosixPath

from shared.cgroup_slice import DAEMON_SUBGROUP, project_slice_name

logger = logging.getLogger(__name__)

__all__ = ['PROJECT_SLICE_CPU_WEIGHT', 'SliceWeightOutcome', 'ensure_project_slice_weight']

PROJECT_SLICE_CPU_WEIGHT = 100


class SliceWeightOutcome(enum.Enum):
    APPLIED = 'applied'
    UNIT_NOT_IN_SLICE = 'unit_not_in_slice'
    FAILED = 'failed'


async def ensure_project_slice_weight(
    project_id: str, own_cgroup_path: str, *, set_property_timeout_secs: float = 30.0,
) -> SliceWeightOutcome:
    """Set ``CPUWeight`` on the project slice when *own_cgroup_path* (the
    process's raw leaf unified cgroup) is a service directly under it, or that
    service's daemon subgroup.

    A placement or systemctl failure never raises: every outcome is logged and
    returned. Only an empty *project_id* raises, from the shared name rule."""
    slice_name = project_slice_name(project_id)
    if not _runs_in_a_service_directly_under(slice_name, own_cgroup_path):
        logger.warning(
            'orchestrator own cgroup %r is neither a service unit directly under its '
            'project slice %s nor that service\'s %s/ subgroup; if it runs as a systemd '
            'unit, reinstall the unit so Slice= applies (docs/cgroup-slice-topology.md)',
            own_cgroup_path, slice_name, DAEMON_SUBGROUP,
        )
        return SliceWeightOutcome.UNIT_NOT_IN_SLICE
    failure = await _set_slice_cpu_weight(slice_name, set_property_timeout_secs)
    if failure is not None:
        logger.warning(
            'could not set CPUWeight=%d on project slice %s: %s',
            PROJECT_SLICE_CPU_WEIGHT, slice_name, failure,
        )
        return SliceWeightOutcome.FAILED
    logger.info('project slice %s has CPUWeight=%d', slice_name, PROJECT_SLICE_CPU_WEIGHT)
    return SliceWeightOutcome.APPLIED


def _runs_in_a_service_directly_under(slice_name: str, own_cgroup_path: str) -> bool:
    own = PurePosixPath(own_cgroup_path)
    service = own.parent if own.name == DAEMON_SUBGROUP else own
    return service.name.endswith('.service') and service.parent.name == slice_name


async def _set_slice_cpu_weight(slice_name: str, timeout_secs: float) -> str | None:
    """Run ``systemctl --user set-property --runtime``; the failure cause, or None."""
    if shutil.which('systemctl') is None:
        return 'systemctl not found on PATH'
    try:
        proc = await asyncio.create_subprocess_exec(
            'systemctl', '--user', 'set-property', '--runtime', slice_name,
            f'CPUWeight={PROJECT_SLICE_CPU_WEIGHT}',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return f'spawn failed: {exc}'
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout_secs)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return f'timed out after {timeout_secs:g}s'
    if proc.returncode != 0:
        return f'rc={proc.returncode}: {stderr.decode(errors="replace").strip()}'
    return None
