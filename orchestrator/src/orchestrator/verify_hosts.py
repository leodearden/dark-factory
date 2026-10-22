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

from orchestrator.verify_runner import (
    HostAllocator,
    HostLease,
    RemoteRunner,
    resolve_local_df_checkout,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

MergeDemand = Callable[[], bool]
RefusalReason = Literal['no_host', 'host_parked', 'merge_demand', 'pressure_hold']
Unwind = Callable[[], Awaitable[object]]


def build_remote_runners(
    config: OrchestratorConfig,
    cwd: str | Path,
    *,
    quarantine: set[str] | None = None,
) -> list[RemoteRunner]:
    """One :class:`RemoteRunner` per enabled ``config.verify_runners`` entry not in *quarantine*.

    Remotes only: no local runner is included.  Every runner pushes
    ``config.git.main_branch`` to its host before a verify, and carries the
    INV-2 contract-currency paths (task 2884) that ``sync_if_stale`` compares:
    its own ``df_checkout_path`` (``None`` leaves auto-sync off) and this
    process's DF code root, resolved once and shared by every runner.
    """
    _local_df = resolve_local_df_checkout()
    return [
        RemoteRunner(
            name=r.name,
            ssh_host=r.ssh_host,
            git_remote=r.git_remote,
            cwd=cwd,
            config_path=r.config_path,
            main_branch=config.git.main_branch,
            df_remote_checkout=r.df_checkout_path,
            df_local_checkout=_local_df,
        )
        for r in config.enabled_verify_runners
        if quarantine is None or r.name not in quarantine
    ]


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
    synchronous, so its check-then-acquire is atomic on the event loop.
    """

    def __init__(
        self,
        allocator: HostAllocator,
        *,
        merge_demand: MergeDemand,
        on_unavailable: Callable[[str, str], None],
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
        """Lease a free remote for *task_id*, or say why none was handed out."""
        allocator = self._allocator
        remotes = [n for n in allocator.host_names if n != allocator.local_name]
        if not remotes:
            return Refusal('no_host')
        if all(allocator.is_parked(name) for name in remotes):
            return Refusal('host_parked')
        if self._merge_demand():
            return Refusal('merge_demand')
        lease = allocator.acquire_remote()
        if lease is None:
            return Refusal('no_host')
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

    def mark_unavailable(self, lease: HostLease, reason: str) -> None:
        self._on_unavailable(lease.name, reason)

    def mark_recovered(self, lease: HostLease) -> None:
        self._on_recovered(lease.name)

    def holders(self) -> Mapping[str, str]:
        """A fresh ``{host: 'task:<id>'}`` map of the holds this pool owns."""
        return {host: f'task:{task_id}' for host, (task_id, _) in self._holds.items()}

    def _holds_lease(self, lease: HostLease) -> bool:
        hold = self._holds.get(lease.name)
        return hold is not None and hold[1] is lease
