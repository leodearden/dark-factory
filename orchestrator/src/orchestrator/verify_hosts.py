"""Remote verify-host construction and the task-path host pool.

PRD ``plans/remote-task-leg-verify-prd.md`` §4.4-4.5, §5.4.  This module sits
below both the merge lane and the task workflow: it must never import
``orchestrator.merge_lane`` or ``orchestrator.workflow``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from orchestrator.deployed_sha import DeployedSha, resolve_code_checkout
from orchestrator.verify_runner import (
    HostAllocator,
    HostLease,
    RemoteRunner,
    RunnerUnavailable,
    RunnerUnavailableSink,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

MergeDemand = Callable[[], bool]
RefusalReason = Literal['no_host', 'host_busy', 'host_quarantined', 'host_parked', 'merge_demand']
_HOST_STATE_PRECEDENCE: tuple[RefusalReason, ...] = ('host_busy', 'host_quarantined', 'host_parked')
Unwind = Callable[[], Awaitable[object]]


def build_remote_runners(
    config: OrchestratorConfig,
    cwd: str | Path,
    *,
    deployed_sha: DeployedSha,
    quarantine: set[str] | None = None,
) -> list[RemoteRunner]:
    """Build the list of RemoteRunner instances from operator config (Lever C).

    Returns REMOTES ONLY — the LocalRunner trust anchor is prepended by callers
    since it needs call-specific arguments (worktree path, module configs, etc.).

    Filters out disabled runners (enabled=False) and any runner whose name is in
    the quarantine set (in-memory worker-level quarantine from DriftDetector).
    quarantine=None is treated as an empty set (no quarantine).

    _build_verify_runners passes main_branch=config.git.main_branch so the
    remote host receives a freshness push before the merge-sha transport.

    Each runner gates its verifies on *deployed_sha* (task 6565, the commit
    this process's wire path was imported from) through the runner's
    :meth:`~orchestrator.config.VerifyRunnerConfig.gate_paths`, publishing the
    sha from this process's own code checkout, resolved once here.
    """
    code_checkout = resolve_code_checkout()
    runners = []
    for r in config.enabled_verify_runners:
        if quarantine is not None and r.name in quarantine:
            continue
        gate = r.gate_paths()
        runners.append(RemoteRunner(
            name=r.name,
            ssh_host=r.ssh_host,
            git_remote=r.git_remote,
            cwd=cwd,
            config_path=r.config_path,
            main_branch=config.git.main_branch,
            df_remote_checkout=gate.source,
            gate_checkouts_path=gate.root,
            deployed_sha=deployed_sha,
            code_checkout=code_checkout,
        ))
    return runners


@dataclass(frozen=True)
class Refusal:
    """Why :meth:`HostPool.acquire_for_task` handed out no remote host."""

    reason: RefusalReason

    def __post_init__(self) -> None:
        if self.reason not in get_args(RefusalReason):
            raise ValueError(
                f'Refusal reason {self.reason!r} is not one of {get_args(RefusalReason)}'
            )


@dataclass(frozen=True)
class _CancelThenUnwind:
    """*runner* as the allocator sees it during one cancel: *unwind* follows the host-side cancel."""

    runner: RemoteRunner
    unwind: Unwind

    async def cancel_verify(self) -> int:
        try:
            return await self.runner.cancel_verify()
        except Exception:
            logger.warning(
                'cancel-verify on %s raised; treating it as a failed cancel',
                self.runner.name, exc_info=True,
            )
            return 1
        finally:
            await self.unwind()

    async def probe_clean(self) -> bool:
        return await self.runner.probe_clean()


def _blocker(allocator: HostAllocator, name: str) -> RefusalReason | None:
    """Why remote *name* cannot be leased now, or None when it is free.

    Parked is checked first because ``is_busy`` is also true for a PARKED slot.
    """
    if allocator.is_parked(name):
        return 'host_parked'
    if allocator.is_quarantined(name):
        return 'host_quarantined'
    if allocator.is_busy(name):
        return 'host_busy'
    return None


def _run_once(unwind: Unwind | None) -> Callable[[], Awaitable[None]]:
    remaining = [] if unwind is None else [unwind]

    async def once() -> None:
        if remaining:
            await remaining.pop()()

    return once


class HostPool:
    """The task path's leases on the worker-lifetime :class:`HostAllocator`.

    Invariants: a hold is a lease this pool handed out and whose release or
    cancel has not yet returned, so its slot is BUSY (or PARKED while a failed
    cancel probes the host); the pool never writes the allocator's quarantine set (an
    unavailable host is reported through *on_unavailable* and readmission stays
    with the merge lane's tracker, PRD §4.5); :meth:`acquire_for_task` is
    synchronous, so its check-then-acquire is atomic on the event loop, and it
    refuses on host state before it reads merge demand.
    """

    def __init__(
        self,
        allocator: HostAllocator,
        *,
        merge_demand: MergeDemand,
        on_unavailable: RunnerUnavailableSink,
        on_recovered: Callable[[str], None],
        probe_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._allocator = allocator
        self._merge_demand = merge_demand
        self._on_unavailable = on_unavailable
        self._on_recovered = on_recovered
        self._probe_sleep = probe_sleep
        self._holds: dict[str, tuple[str, HostLease]] = {}

    @property
    def allocator(self) -> HostAllocator:
        return self._allocator

    def acquire_for_task(self, task_id: str) -> HostLease | Refusal:
        """Lease a free remote for *task_id*, or say why none was handed out.

        No remote configured is ``no_host``; when no remote is free the refusal
        names the most transient blocker present (busy, then quarantined, then
        parked); only then is merge demand read.
        """
        allocator = self._allocator
        remotes = [n for n in allocator.host_names if n != allocator.local_name]
        if not remotes:
            return Refusal('no_host')
        blockers = {_blocker(allocator, name) for name in remotes}
        if None not in blockers:
            return Refusal(next(r for r in _HOST_STATE_PRECEDENCE if r in blockers))
        if self._merge_demand():
            return Refusal('merge_demand')
        lease = allocator.acquire_remote()
        if lease is None:
            return Refusal('host_busy')
        self._holds[lease.name] = (task_id, lease)
        return lease

    async def release(self, lease: HostLease) -> None:
        """Free *lease*'s slot; a lease this pool does not hold is a no-op."""
        if not self._holds_lease(lease):
            return
        try:
            await self._allocator.release(lease)
        finally:
            self._holds.pop(lease.name, None)

    async def cancel_and_release(self, lease: HostLease, *, unwind: Unwind | None = None) -> bool:
        """Cancel the verify on *lease*'s host and release (or PARK) its slot.

        *unwind*, the caller's teardown of the dispatch that was using the runner,
        is awaited exactly once: after the host-side cancel and before the slot is
        freed or PARKED, so no other lease can reach a runner whose dispatch is
        still unwinding (PRD §4.3 exclusive use).
        """
        once = _run_once(unwind)
        try:
            if not self._holds_lease(lease):
                return True
            cancelling = dataclasses.replace(lease, runner=_CancelThenUnwind(lease.runner, once))
            try:
                return await self._allocator.cancel_and_release(cancelling, sleep=self._probe_sleep)
            finally:
                self._holds.pop(lease.name, None)
        finally:
            await once()

    def mark_unavailable(self, lease: HostLease, exc: RunnerUnavailable) -> None:
        self._on_unavailable(lease.name, exc)

    def mark_recovered(self, lease: HostLease) -> None:
        self._on_recovered(lease.name)

    def holders(self) -> Mapping[str, str]:
        """A fresh ``{host: 'task:<id>'}`` map of the holds this pool owns."""
        return {host: f'task:{task_id}' for host, (task_id, _) in self._holds.items()}

    def _holds_lease(self, lease: HostLease) -> bool:
        hold = self._holds.get(lease.name)
        return hold is not None and hold[1] is lease
