"""The merge lane's collaborator ports, and the adapters that satisfy them in production.

PRD ``plans/merge-lane-quality-prd.md`` tasks ζ1 and β. A port is the
Protocol the worker calls; a production adapter is what satisfies it in the
running orchestrator; a test injects its own implementation
(``orchestrator/tests/_merge_lane_fakes.py``) in the adapter's place.

The verify and clock adapters are built from zero-argument RESOLVERS rather
than from the collaborator functions themselves. A resolver is evaluated on
every call, so the function used is whatever ``orchestrator.merge_queue``
binds at that moment: the test suite still patches those module names (PRD
phase γ migrates it onto injected fakes), and a function captured at
construction would go inert under such a patch. γ retires the resolvers.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable, Collection, Coroutine
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from escalation.models import Escalation

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_lane.types import DiskGuardOutcome, EscalationRecord
from orchestrator.merge_types import MergeRequest
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import VerifyResult

_T = TypeVar('_T')

Resolver = Callable[[], _T]


class VerifyPort(Protocol):
    """Everything the worker asks of verification.

    Each method takes what the module-level function behind it takes today
    (``run_scoped_verification``, ``_run_unscoped_typechecks``,
    ``_check_post_merge_pyright``, ``_check_post_merge_equivalence``,
    ``_ensure_verify_disk_space``, ``_run_cold_shadow_verify``,
    ``run_dry_run_unblock``), so the wiring is a rename, not a translation.
    """

    async def run_scoped(
        self,
        worktree: Path,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        task_files: list[str] | None = None,
        *,
        max_retries: int | None = None,
        is_merge_verify: bool = False,
        attempt_id: int | None = None,
        task_id: str | None = None,
        archive_root: Path | None = None,
        force_workspace: bool = False,
        role: Literal['merge', 'task'] = 'task',
        event_store: EventStore | None = None,
    ) -> VerifyResult: ...

    async def run_unscoped_typechecks(
        self,
        worktree: Path,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        *,
        block_on_timeout: bool,
        task_id: str | None = None,
    ) -> PostMergePyrightResult: ...

    async def check_post_merge_pyright(
        self,
        advanced_sha: str,
        git_ops: GitOps,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        *,
        task_id: str | None = None,
    ) -> PostMergePyrightResult: ...

    async def check_post_merge_equivalence(
        self,
        task_worktree: Path,
        advanced_sha: str,
        git_ops: GitOps,
        main_sha: str,
        *,
        task_id: str | None = None,
        merged_tip: str | None = None,
        allow_worktree_head_fallback: bool = True,
    ) -> list[str]: ...

    async def ensure_disk_space(
        self,
        git_ops: GitOps,
        merge_wt: Path,
        min_free_bytes: int,
        task_id: str,
        keep_worktrees: Collection[Path] | None = None,
    ) -> DiskGuardOutcome: ...

    async def cold_shadow(
        self,
        git_ops: GitOps,
        req: MergeRequest,
        merge_commit: str,
        event_store: EventStore | None,
    ) -> dict[str, str]: ...

    def dry_run_unblock(
        self,
        *,
        task_id: str,
        worktree: str,
        reason: str,
        detail: str,
        scheduler: Any,
        mcp: Any,
        config: Any,
        event_store: Any = None,
        usage_gate: Any = None,
        cost_store: Any = None,
        block_class: BlockClass | None = None,
    ) -> Coroutine[Any, Any, None]:
        """Build the investigation coroutine; the worker schedules it fire-and-forget."""
        ...


class ClockPort(Protocol):
    """The worker's two clocks, its sleep, and its merge-worktree progress probe."""

    def now(self) -> float: ...

    def monotonic(self) -> float: ...

    def newest_content_mtime(self, root: Path) -> float | None: ...

    async def sleep(self, secs: float) -> None: ...


class EscalationPort(Protocol):
    """Where the worker files an escalation of its own."""

    def file(self, record: EscalationRecord) -> str | None:
        """File *record*; returns the escalation id, or ``None`` when nothing was filed."""
        ...


@dataclasses.dataclass(frozen=True)
class ProductionVerifier:
    """``VerifyPort`` as the running orchestrator wires it.

    Every method resolves its function and forwards the call exactly as
    received -- positional arguments positionally, keywords by keyword -- so
    the function sees the same call it saw before the port existed.
    """

    scoped: Resolver[Callable[..., Awaitable[VerifyResult]]]
    unscoped: Resolver[Callable[..., Awaitable[PostMergePyrightResult]]]
    post_merge_pyright: Resolver[Callable[..., Awaitable[PostMergePyrightResult]]]
    post_merge_equivalence: Resolver[Callable[..., Awaitable[list[str]]]]
    disk_guard: Resolver[Callable[..., Awaitable[str | None]]]
    cold_shadow_verify: Resolver[Callable[..., Awaitable[dict[str, str]]]]
    dry_run: Resolver[Callable[..., Coroutine[Any, Any, None]]]

    async def run_scoped(self, *args: Any, **options: Any) -> VerifyResult:
        return await self.scoped()(*args, **options)

    async def run_unscoped_typechecks(self, *args: Any, **options: Any) -> PostMergePyrightResult:
        return await self.unscoped()(*args, **options)

    async def check_post_merge_pyright(self, *args: Any, **options: Any) -> PostMergePyrightResult:
        return await self.post_merge_pyright()(*args, **options)

    async def check_post_merge_equivalence(self, *args: Any, **options: Any) -> list[str]:
        return await self.post_merge_equivalence()(*args, **options)

    async def ensure_disk_space(self, *args: Any, **options: Any) -> DiskGuardOutcome:
        return DiskGuardOutcome(reason=await self.disk_guard()(*args, **options))

    async def cold_shadow(self, *args: Any, **options: Any) -> dict[str, str]:
        return await self.cold_shadow_verify()(*args, **options)

    def dry_run_unblock(self, **investigation: Any) -> Coroutine[Any, Any, None]:
        return self.dry_run()(**investigation)


@dataclasses.dataclass(frozen=True)
class ProductionClock:
    """``ClockPort`` on the real clocks.

    ``content_mtime`` resolves ``merge_liveness.newest_content_mtime`` the
    way the verify resolvers do, for the same reason.
    """

    content_mtime: Resolver[Callable[[Path], float | None]]

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def newest_content_mtime(self, root: Path) -> float | None:
        return self.content_mtime()(root)

    async def sleep(self, secs: float) -> None:
        await asyncio.sleep(secs)


@dataclasses.dataclass(frozen=True)
class ProductionEscalations:
    """``EscalationPort`` on the orchestrator's escalation queue.

    The queue mints the id and receives the ``Escalation``.
    """

    queue: Any

    def file(self, record: EscalationRecord) -> str:
        escalation = Escalation(
            id=self.queue.make_id(record.task_id),
            task_id=record.task_id,
            agent_role=record.agent_role,
            severity=record.severity,
            level=record.level,
            category=record.category,
            summary=record.summary,
            detail=record.detail,
            suggested_action=record.suggested_action,
        )
        self.queue.submit(escalation)
        return escalation.id


class DiscardingEscalations:
    """``EscalationPort`` for a worker wired without an escalation queue.

    Nothing is filed, which is what such a worker did before the port
    existed.
    """

    def file(self, record: EscalationRecord) -> None:
        return None


def escalation_port(queue: Any) -> EscalationPort:
    """The sink for a worker constructed with *queue*, which may be ``None``."""
    return DiscardingEscalations() if queue is None else ProductionEscalations(queue)
