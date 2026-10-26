"""The single routing point for a task-role verify: the local run or a remote host.

PRD ``plans/remote-task-leg-verify-prd.md`` §4.6-4.9, §5.5.  Imports point
downward only: this module must never import ``orchestrator.merge_lane`` or
``orchestrator.workflow``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, assert_never, get_args

from shared.git_async import run_git
from shared.safe_io import atomic_write_text

from orchestrator.task_verify_spec import (
    CollapsedOverride,
    build_task_verify_spec,
    collapsed_module_overrides,
)
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_hosts import HostPool, Refusal
from orchestrator.verify_plan import derive_verify_plan, executed_module_prefixes
from orchestrator.verify_runner import RunnerUnavailable

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig
    from orchestrator.event_store import EventStore
    from orchestrator.git_ops import GitOps
    from orchestrator.verify import VerifyResult
    from orchestrator.verify_runner import HostLease

logger = logging.getLogger(__name__)

_DECISIONS_KEPT = 1024
_SIDECAR_STAMP_FMT = '%Y%m%dT%H%M%S_%fZ'


class TaskVerifyPort(Protocol):
    """The workflow's task-verify call shape, which ``run_scoped_verification`` satisfies."""

    async def __call__(
        self,
        worktree: Path,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        task_files: list[str] | None = None,
        *,
        attempt_id: int,
        task_id: str,
        archive_root: Path | None,
        force_workspace: bool = False,
        role: Literal['task'] = 'task',
        event_store: EventStore | None = None,
    ) -> VerifyResult: ...


HostPolicy = Literal['local', 'remote_modules', 'remote_all']


@dataclass(frozen=True)
class TaskVerifyPolicy:
    """Which task verifies may run remotely, and how their verdicts are checked."""

    host_policy: HostPolicy = 'local'
    remote_modules: tuple[str, ...] = ()
    cross_check_every_n: int = 0
    red_confirms_locally: bool = False

    def __post_init__(self) -> None:
        if self.host_policy not in get_args(HostPolicy):
            raise ValueError(
                f'host_policy {self.host_policy!r} is not one of {get_args(HostPolicy)}'
            )
        if not isinstance(self.remote_modules, tuple) or not all(
            isinstance(m, str) for m in self.remote_modules
        ):
            raise ValueError(
                f'remote_modules must be a tuple of module prefixes, got {self.remote_modules!r}'
            )
        if self.host_policy == 'remote_modules' and not self.remote_modules:
            raise ValueError("host_policy 'remote_modules' needs a non-empty remote_modules")
        if isinstance(self.cross_check_every_n, bool) or not (
            isinstance(self.cross_check_every_n, int) and self.cross_check_every_n >= 0
        ):
            raise ValueError(
                f'cross_check_every_n must be an int >= 0, got {self.cross_check_every_n!r}'
            )
        if not isinstance(self.red_confirms_locally, bool):
            raise ValueError(
                f'red_confirms_locally must be a bool, got {self.red_confirms_locally!r}'
            )


@dataclass(frozen=True)
class CrossCheck:
    """A remote verdict re-run locally on the cross-check cadence."""

    remote_passed: bool
    local_passed: bool

    @property
    def agree(self) -> bool:
        return self.remote_passed == self.local_passed

    def to_dict(self) -> dict[str, bool]:
        return {
            'remote_passed': self.remote_passed,
            'local_passed': self.local_passed,
            'agree': self.agree,
        }


@dataclass(frozen=True)
class RedConfirmation:
    """A remote red re-run locally before it is believed."""

    local_passed: bool

    def to_dict(self) -> dict[str, bool]:
        return {'remote_passed': False, 'local_passed': self.local_passed}


DispatchReason = Literal[
    'remote', 'not_eligible', 'no_host', 'host_parked', 'merge_demand', 'pressure_hold',
    'dirty_tree', 'runner_unavailable', 'infra_transient', 'cross_check',
    'red_confirm', 'dispatcher_error',
]


@dataclass(frozen=True)
class DispatchDecision:
    """Where one task verify ran and why; *runner* names the host whose verdict was returned."""

    runner: str
    reason: DispatchReason
    tip_sha: str | None
    base_sha: str | None
    remote_wall_secs: float | None
    lease_held_secs: float | None
    cross_check: CrossCheck | None
    red_confirmed_locally: RedConfirmation | None

    def __post_init__(self) -> None:
        if self.reason not in get_args(DispatchReason):
            raise ValueError(
                f'dispatch reason {self.reason!r} is not one of {get_args(DispatchReason)}'
            )

    @property
    def decision(self) -> Literal['remote', 'local']:
        return 'remote' if self.reason == 'remote' else 'local'

    def to_dict(self) -> dict[str, Any]:
        """The dispatch sidecar's JSON object."""
        return {
            'decision': self.decision,
            'runner': self.runner,
            'reason': self.reason,
            'tip_sha': self.tip_sha,
            'base_sha': self.base_sha,
            'remote_wall_secs': self.remote_wall_secs,
            'lease_held_secs': self.lease_held_secs,
            'cross_check': self.cross_check.to_dict() if self.cross_check else None,
            'red_confirmed_locally': (
                self.red_confirmed_locally.to_dict() if self.red_confirmed_locally else None
            ),
        }


def eligible(
    policy: TaskVerifyPolicy,
    executed_prefixes: Iterable[str],
    *,
    force_workspace: bool,
    per_module_overrides_present: bool,
) -> bool:
    """Whether *policy* lets a verify executing *executed_prefixes* run remotely."""
    if force_workspace or per_module_overrides_present:
        return False
    match policy.host_policy:
        case 'local':
            return False
        case 'remote_all':
            return True
        case 'remote_modules':
            return not set(policy.remote_modules).isdisjoint(executed_prefixes)
        case _:
            assert_never(policy.host_policy)


@dataclass(frozen=True)
class _VerifyCall:
    """One task-verify call's arguments, forwarded unchanged to whichever leg runs."""

    worktree: Path
    config: OrchestratorConfig
    module_configs: list[ModuleConfig]
    task_files: list[str] | None
    attempt_id: int
    task_id: str
    archive_root: Path | None
    force_workspace: bool
    role: Literal['task']
    event_store: EventStore | None

    async def run(self, port: TaskVerifyPort) -> VerifyResult:
        return await port(
            self.worktree, self.config, self.module_configs,
            task_files=self.task_files,
            attempt_id=self.attempt_id,
            task_id=self.task_id,
            archive_root=self.archive_root,
            force_workspace=self.force_workspace,
            role=self.role,
            event_store=self.event_store,
        )


@dataclass(frozen=True)
class _TaskScope:
    tip_sha: str
    base_sha: str | None
    task_files: tuple[str, ...]
    executed_prefixes: frozenset[str]


@dataclass
class _DecisionDraft:
    """The decision as one call builds it; lives for that call only."""

    runner: str
    reason: DispatchReason | None = None
    tip_sha: str | None = None
    base_sha: str | None = None
    remote_wall_secs: float | None = None
    lease_held_secs: float | None = None
    cross_check: CrossCheck | None = None
    red_confirmed_locally: RedConfirmation | None = None

    def freeze(self) -> DispatchDecision:
        return DispatchDecision(
            runner=self.runner,
            reason=self.reason if self.reason is not None else 'dispatcher_error',
            tip_sha=self.tip_sha,
            base_sha=self.base_sha,
            remote_wall_secs=self.remote_wall_secs,
            lease_held_secs=self.lease_held_secs,
            cross_check=self.cross_check,
            red_confirmed_locally=self.red_confirmed_locally,
        )


def _worktree_reader(worktree: Path) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        try:
            return (worktree / path).read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return None
    return read


async def _reap(task: asyncio.Future[Any]) -> None:
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.debug('A cancelled remote task verify ended with %r instead', exc)


class TaskVerifyDispatcher:
    """Routes each task verify to a leased remote host or to *run_local*.

    Shared by every workflow of one orchestrator, so decisions are keyed by
    ``(task_id, attempt_id)`` and the policy is read on every call.  Each
    routed call leaves a dispatch sidecar beside its verify logs.
    """

    def __init__(
        self,
        pool: HostPool,
        git_ops: GitOps,
        *,
        policy_of: Callable[[OrchestratorConfig], TaskVerifyPolicy],
        run_local: TaskVerifyPort = run_scoped_verification,
    ) -> None:
        self._pool = pool
        self._git_ops = git_ops
        self._policy_of = policy_of
        self._run_local = run_local
        self._decisions: dict[tuple[str, int], DispatchDecision] = {}
        self._warned_overrides: set[CollapsedOverride] = set()
        self._remote_verdicts = 0

    async def __call__(
        self,
        worktree: Path,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        task_files: list[str] | None = None,
        *,
        attempt_id: int,
        task_id: str,
        archive_root: Path | None,
        force_workspace: bool = False,
        role: Literal['task'] = 'task',
        event_store: EventStore | None = None,
    ) -> VerifyResult:
        call = _VerifyCall(
            worktree, config, module_configs, task_files, attempt_id, task_id,
            archive_root, force_workspace, role, event_store,
        )
        policy = self._policy_of(config)
        if policy.host_policy == 'local':
            return await call.run(self._run_local)
        draft = _DecisionDraft(runner=self._pool.allocator.local_name)
        try:
            result = await self._dispatch(call, policy, draft)
        except Exception:
            self._record(call, draft.freeze())
            raise
        self._record(call, draft.freeze())
        return result

    def decision_for(self, task_id: str, attempt_id: int) -> DispatchDecision | None:
        """The latest decision recorded for this task attempt, if any."""
        return self._decisions.get((task_id, attempt_id))

    async def _dispatch(
        self, call: _VerifyCall, policy: TaskVerifyPolicy, draft: _DecisionDraft,
    ) -> VerifyResult:
        scope = await self._scope_or_none(call)
        if scope is None:
            return await self._run_local_because(call, draft, 'dispatcher_error')
        draft.tip_sha, draft.base_sha = scope.tip_sha, scope.base_sha
        if not self._eligible(call, policy, scope):
            return await self._run_local_because(call, draft, 'not_eligible')
        lease = self._pool.acquire_for_task(call.task_id)
        if isinstance(lease, Refusal):
            return await self._run_local_because(call, draft, lease.reason)
        outcome = await self._run_remote(call, scope, lease, draft)
        if isinstance(outcome, str):
            return await self._run_local_because(call, draft, outcome)
        draft.reason = 'remote'
        return await self._settle_remote(call, policy, outcome, draft)

    async def _settle_remote(
        self, call: _VerifyCall, policy: TaskVerifyPolicy, remote: VerifyResult,
        draft: _DecisionDraft,
    ) -> VerifyResult:
        """Return *remote*, or the local verdict when a parity knob re-runs it locally."""
        self._remote_verdicts += 1
        every_n = policy.cross_check_every_n
        cross = every_n > 0 and self._remote_verdicts % every_n == 0
        confirm = policy.red_confirms_locally and not remote.passed
        if not (cross or confirm):
            return remote
        draft.runner = self._pool.allocator.local_name
        local = await self._run_local_because(call, draft, 'cross_check' if cross else 'red_confirm')
        if cross:
            draft.cross_check = CrossCheck(remote_passed=remote.passed, local_passed=local.passed)
        if confirm:
            draft.red_confirmed_locally = RedConfirmation(local_passed=local.passed)
        return local

    async def _run_local_because(
        self, call: _VerifyCall, draft: _DecisionDraft, reason: DispatchReason,
    ) -> VerifyResult:
        draft.reason = reason
        return await call.run(self._run_local)

    async def _run_remote(
        self, call: _VerifyCall, scope: _TaskScope, lease: HostLease, draft: _DecisionDraft,
    ) -> VerifyResult | DispatchReason:
        """Run the verify on *lease*'s host; the lease ends here on every path."""
        acquired = time.monotonic()
        remote: asyncio.Future[VerifyResult] | None = None
        try:
            if await self._git_ops.has_uncommitted_work(call.worktree):
                return 'dirty_tree'
            spec = build_task_verify_spec(
                call.config, call.module_configs, scope.task_files,
                force_workspace=False, attempt_id=call.attempt_id,
                task_id=call.task_id, tip_sha=scope.tip_sha,
            )
            remote = verifying = asyncio.ensure_future(lease.runner.run_task_verify(
                spec, task_id=call.task_id, archive_root=call.archive_root,
                event_store=call.event_store,
            ))
            return await self._remote_verdict(verifying, lease, draft)
        finally:
            await self._end_lease(lease, remote)
            draft.lease_held_secs = time.monotonic() - acquired

    async def _remote_verdict(
        self, remote: asyncio.Future[VerifyResult], lease: HostLease, draft: _DecisionDraft,
    ) -> VerifyResult | DispatchReason:
        started = time.monotonic()
        try:
            result = await asyncio.shield(remote)
        except RunnerUnavailable as exc:
            self._pool.mark_unavailable(lease, exc)
            return 'runner_unavailable'
        finally:
            draft.remote_wall_secs = time.monotonic() - started
        self._pool.mark_recovered(lease)
        if not result.passed and result.category in INFRA_TRANSIENT_CATEGORIES:
            return 'infra_transient'
        draft.runner = lease.name
        return result

    async def _end_lease(
        self, lease: HostLease, remote: asyncio.Future[VerifyResult] | None,
    ) -> None:
        """Release *lease*; a remote verify still in flight (only on cancellation) is
        cancelled on its host while its dispatch is live, then reaped, and only then is
        its slot freed or PARKED (PRD §4.3 exclusive use, §4.8)."""
        if remote is None or remote.done():
            await self._pool.release(lease)
            return
        await asyncio.shield(
            self._pool.cancel_and_release(lease, unwind=functools.partial(_reap, remote)),
        )

    async def _scope_or_none(self, call: _VerifyCall) -> _TaskScope | None:
        """*call*'s scope, or None (logged) when it cannot be derived; nothing is leased yet."""
        try:
            return await self._scope_of(call)
        except Exception:
            logger.warning(
                'Task %s verify runs locally: its scope could not be derived in %s',
                call.task_id, call.worktree, exc_info=True,
            )
            return None

    async def _scope_of(self, call: _VerifyCall) -> _TaskScope:
        tip = await run_git(['git', 'rev-parse', 'HEAD'], cwd=call.worktree)
        if not tip.ok:
            raise RuntimeError(
                f'cannot resolve the task tip in {call.worktree}: {tip.stderr.strip()}'
            )
        tip_sha = tip.stdout.strip()
        base_sha = await self._git_ops.merge_base_with_main(tip_sha)
        files = await self._task_files(call, tip_sha, base_sha)
        existing = [f for f in files if (call.worktree / f).is_file()]
        plan = derive_verify_plan(
            existing, list(call.module_configs), call.config,
            _worktree_reader(call.worktree), role='task',
        )
        return _TaskScope(tip_sha, base_sha, files, executed_module_prefixes(plan))

    async def _task_files(
        self, call: _VerifyCall, tip_sha: str, base_sha: str | None,
    ) -> tuple[str, ...]:
        if call.task_files is not None:
            return tuple(call.task_files)
        if base_sha is None:
            return ()
        return tuple(await self._git_ops.get_changed_files(base_sha, tip_sha))

    def _eligible(self, call: _VerifyCall, policy: TaskVerifyPolicy, scope: _TaskScope) -> bool:
        collapsed = collapsed_module_overrides(call.module_configs)
        for override in collapsed:
            self._warn_once(override)
        files_derivable = call.task_files is not None or scope.base_sha is not None
        return files_derivable and eligible(
            policy, scope.executed_prefixes,
            force_workspace=call.force_workspace,
            per_module_overrides_present=bool(collapsed),
        )

    def _warn_once(self, override: CollapsedOverride) -> None:
        if override in self._warned_overrides:
            return
        self._warned_overrides.add(override)
        logger.warning(
            'Task verifies stay local: module %r carries a per-module %s override '
            'that a remote task-verify spec cannot carry',
            override.prefix, override.field,
        )

    def _record(self, call: _VerifyCall, decision: DispatchDecision) -> None:
        key = (call.task_id, call.attempt_id)
        self._decisions.pop(key, None)
        self._decisions[key] = decision
        while len(self._decisions) > _DECISIONS_KEPT:
            del self._decisions[next(iter(self._decisions))]
        if call.archive_root is not None:
            _write_sidecar(call.archive_root, call.task_id, call.attempt_id, decision)


def _write_sidecar(
    archive_root: Path, task_id: str, attempt_id: int, decision: DispatchDecision,
) -> None:
    stamp = datetime.now(UTC).strftime(_SIDECAR_STAMP_FMT)
    path = archive_root / task_id / f'attempt-{attempt_id}.dispatch-{stamp}.json'
    try:
        atomic_write_text(
            path, json.dumps(decision.to_dict(), indent=2, sort_keys=True), mkdir=True,
        )
    except OSError:
        logger.warning('Dispatch sidecar %s could not be written', path, exc_info=True)
