"""The single routing point for a task-role verify: the local run or a remote host.

PRD ``plans/remote-task-leg-verify-prd.md`` §4.6-4.9, §5.5; decision 14 (an
eligible verify queues on every host at once, and the first turn wins) and
decision 16 (a task slot is a gate turn on every host).  Imports point
downward only: this module must never import ``orchestrator.merge_lane`` or
``orchestrator.workflow``.
"""

# pyright: reportIncompatibleMethodOverride=true

from __future__ import annotations

import asyncio
import enum
import functools
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    assert_never,
    get_args,
    runtime_checkable,
)

from shared.git_async import run_git
from shared.safe_io import atomic_write_text

from orchestrator.config import VerifyTaskHostPolicy
from orchestrator.task_verify_spec import (
    CollapsedOverride,
    build_task_verify_spec,
    collapsed_module_overrides,
)
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_admission_gate import AdmissionPlan
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_plan import derive_verify_plan, executed_module_prefixes
from orchestrator.verify_runner import RunnerUnavailable

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig
    from orchestrator.event_store import EventStore
    from orchestrator.git_ops import GitOps

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


class LocalVerifyPort(TaskVerifyPort, Protocol):
    """``run_scoped_verification``'s shape with the plan a verify moved between hosts keeps."""

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
        admission_plan: AdmissionPlan | None = None,
    ) -> VerifyResult: ...


HostPolicy = VerifyTaskHostPolicy


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


def policy_from_config(config: OrchestratorConfig) -> TaskVerifyPolicy:
    """The policy *config*'s four ``verify_task_*`` keys declare, read afresh per call."""
    return TaskVerifyPolicy(
        host_policy=config.verify_task_host_policy,
        remote_modules=tuple(config.verify_task_remote_modules),
        cross_check_every_n=config.verify_task_remote_cross_check_every_n,
        red_confirms_locally=config.verify_task_remote_red_confirms_locally,
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


FallbackReason = Literal[
    'not_eligible', 'dirty_tree', 'runner_unavailable', 'infra_transient', 'dispatcher_error',
]
"""Why a verify that held a remote turn ran local after all."""

DispatchReason = Literal[
    FallbackReason, 'no_host', 'local_granted_first', 'remote', 'cross_check', 'red_confirm',
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
    remote_wait_secs: float | None
    """Arrival to the remote turn, when the verify ran remote; else None."""

    def __post_init__(self) -> None:
        if self.reason not in get_args(DispatchReason):
            raise ValueError(
                f'dispatch reason {self.reason!r} is not one of {get_args(DispatchReason)}'
            )
        if self.remote_wait_secs is not None and self.remote_wait_secs < 0:
            raise ValueError(f'remote_wait_secs must be >= 0, got {self.remote_wait_secs!r}')

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
            'remote_wait_secs': self.remote_wait_secs,
        }


LOCAL_RUNNER: Final = HostPool.local_name


@runtime_checkable
class DispatchRecords(Protocol):
    """A task-verify port that records where each task attempt's verdict came from."""

    def decision_for(self, task_id: str, attempt_id: int) -> DispatchDecision | None: ...


def verdict_runner(port: object | None, task_id: str, attempt_id: int) -> str:
    """The host whose verdict *port* returned for this attempt; LOCAL_RUNNER when unrecorded."""
    if not isinstance(port, DispatchRecords):
        return LOCAL_RUNNER
    decision = port.decision_for(task_id, attempt_id)
    return LOCAL_RUNNER if decision is None else decision.runner


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

    async def run(
        self, port: LocalVerifyPort, *, admission_plan: AdmissionPlan | None = None,
    ) -> VerifyResult:
        """Call *port*; the plan keyword is passed only when a plan is carried."""
        plan = {} if admission_plan is None else {'admission_plan': admission_plan}
        return await port(
            self.worktree, self.config, self.module_configs,
            task_files=self.task_files,
            attempt_id=self.attempt_id,
            task_id=self.task_id,
            archive_root=self.archive_root,
            force_workspace=self.force_workspace,
            role=self.role,
            event_store=self.event_store,
            **plan,
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
    remote_wait_secs: float | None = None

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
            remote_wait_secs=self.remote_wait_secs,
        )


def _worktree_reader(worktree: Path) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        try:
            return (worktree / path).read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return None
    return read


def _infra_transient_unavailability(host: str, result: VerifyResult) -> RunnerUnavailable:
    return RunnerUnavailable(
        f'{host!r} returned an infra-transient verify (category {result.category!r}): '
        f'{result.summary}',
        kind='no_verdict',
    )


async def _reap_remote(remote: asyncio.Future[VerifyResult]) -> None:
    """Cancel a remote verify, which normally ends in RunnerUnavailable, and await its end."""
    remote.cancel()
    await asyncio.wait({remote})
    if not remote.cancelled() and (exc := remote.exception()) is not None:
        logger.debug('A cancelled remote task verify ended with %r instead', exc)


async def _withdrawn(local: asyncio.Task[VerifyResult]) -> None:
    """Await a local run already cancelled; any end but the cancel is a failed unwind."""
    await asyncio.wait({local})
    if not local.cancelled() and (exc := local.exception()) is not None:
        logger.warning('A withdrawn local task verify failed to unwind', exc_info=exc)


class _Left(enum.Enum):
    """An attempt that left its host's turn without running the verify."""

    LEFT = 'left'


@dataclass(frozen=True)
class _RemoteVerdict:
    """The claiming attempt's verdict, with the policy read at its grant, which settles it."""

    result: VerifyResult
    policy: TaskVerifyPolicy


_AttemptOutcome = _RemoteVerdict | FallbackReason | Literal[_Left.LEFT]


def _warn_if_failed(host: str, attempt: asyncio.Task[_AttemptOutcome]) -> None:
    if not attempt.cancelled() and (exc := attempt.exception()) is not None:
        logger.warning('Task verify attempt on %s failed', host, exc_info=exc)


async def _reap_attempt(host: str, attempt: asyncio.Task[_AttemptOutcome]) -> None:
    attempt.cancel()
    await asyncio.wait({attempt})
    _warn_if_failed(host, attempt)


@dataclass(frozen=True)
class _LocalWon:
    """The local call finished while no remote attempt had claimed the verify."""

    local: asyncio.Task[VerifyResult]


_RaceOutcome = _LocalWon | _RemoteVerdict | FallbackReason


@dataclass(eq=False)
class _Race:
    """One eligible call's race between the local queue and every remote host's gate."""

    plan: AdmissionPlan
    local: asyncio.Task[VerifyResult] | None = None
    attempts: dict[asyncio.Task[_AttemptOutcome], str] = field(default_factory=dict)
    claimed_by: str | None = None
    reaping: asyncio.Future[None] | None = None
    remote_wait_secs: float | None = None

    def try_claim(self, host: str, policy: TaskVerifyPolicy) -> bool:
        """Claim the verify for *host*'s turn, unless it is no longer the remote's to run."""
        local_done = self.local is not None and self.local.done()
        if self.claimed_by is not None or self.plan.has_been_granted() or local_done:
            return False
        if policy.host_policy == 'local':
            return False
        self.claimed_by = host
        return True

    def withdraw_local(self) -> asyncio.Future[None] | None:
        """Cancel the local call once, in the caller's step, so no leg already
        scheduled starts after a claim; every caller awaits the same reap."""
        if self.reaping is None and self.local is not None and not self.local.done():
            self.local.cancel()
            self.reaping = asyncio.ensure_future(_withdrawn(self.local))
        return self.reaping


async def _reap_race(race: _Race) -> None:
    reaps: list[Awaitable[None]] = [
        _reap_attempt(host, attempt) for attempt, host in race.attempts.items()
    ]
    reaping = race.withdraw_local()
    if reaping is not None:
        reaps.append(reaping)
    await asyncio.gather(*reaps)


class TaskVerifyDispatcher(DispatchRecords):
    """Routes each task verify to whichever comes first: a remote host's turn or *run_local*.

    Shared by every workflow of one orchestrator, so decisions are keyed by
    ``(task_id, attempt_id)`` and the policy is read on every call and at every
    grant.  Each routed call leaves a dispatch sidecar beside its verify logs.
    """

    def __init__(
        self,
        pool: HostPool,
        git_ops: GitOps,
        *,
        policy_of: Callable[[OrchestratorConfig], TaskVerifyPolicy],
        run_local: LocalVerifyPort = run_scoped_verification,
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
            self._decisions.pop((task_id, attempt_id), None)
            return await call.run(self._run_local)
        draft = _DecisionDraft(runner=self._pool.local_name)
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
        if not self._pool.hosts:
            return await self._run_local_because(call, draft, 'no_host')
        race = _Race(plan=AdmissionPlan.begin(role='task', task_id=call.task_id))
        outcome = await self._race(race, call, draft)
        return await self._after_race(call, race, outcome, draft)

    async def _after_race(
        self, call: _VerifyCall, race: _Race, outcome: _RaceOutcome, draft: _DecisionDraft,
    ) -> VerifyResult:
        """Settle *outcome* once the race has unwound, so no local run holds a remote turn."""
        if isinstance(outcome, _LocalWon):
            draft.reason = 'local_granted_first'
            return outcome.local.result()
        if isinstance(outcome, _RemoteVerdict):
            draft.reason = 'remote'
            draft.remote_wait_secs = race.remote_wait_secs
            return await self._settle_remote(call, outcome.policy, outcome.result, draft)
        draft.reason = outcome
        return await call.run(self._run_local, admission_plan=race.plan)

    async def _race(self, race: _Race, call: _VerifyCall, draft: _DecisionDraft) -> _RaceOutcome:
        """Queue *call* on every remote host's gate, then locally unless a turn came at once."""
        try:
            for host in self._pool.hosts:
                attempt = asyncio.ensure_future(self._attempt(race, host, call, draft))
                race.attempts[attempt] = host
            await asyncio.sleep(0)
            if race.claimed_by is None:
                race.local = asyncio.ensure_future(
                    call.run(self._run_local, admission_plan=race.plan),
                )
            return await self._await_race(race)
        finally:
            await asyncio.shield(_reap_race(race))

    async def _await_race(self, race: _Race) -> _RaceOutcome:
        while True:
            local = {race.local} if race.local is not None and race.claimed_by is None else set()
            done, _ = await asyncio.wait(
                {*race.attempts, *local}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in race.attempts.keys() & done:
                outcome = self._attempt_outcome(race, race.attempts.pop(task), task)
                if outcome is not _Left.LEFT:
                    return outcome
            if race.claimed_by is None and race.local is not None and race.local.done():
                return _LocalWon(race.local)

    def _attempt_outcome(
        self, race: _Race, host: str, attempt: asyncio.Task[_AttemptOutcome],
    ) -> _AttemptOutcome:
        """The claiming attempt's outcome (its exception re-raised); any other attempt left."""
        if race.claimed_by == host:
            return attempt.result()
        _warn_if_failed(host, attempt)
        return _Left.LEFT

    async def _attempt(
        self, race: _Race, host: str, call: _VerifyCall, draft: _DecisionDraft,
    ) -> _AttemptOutcome:
        """Wait for *host*'s turn; the first eligible grant claims the verify and runs it there.

        The claim is checked at the grant with no await in between, and every
        return leaves the hold and the turn before the parent sees its value.
        """
        request = race.plan.request(module=None, scope_kind=None)
        gate_policy = functools.partial(self._pool.policy, host, call.config)
        async with self._pool.gate(host).admit(request, gate_policy) as grant:
            policy = self._policy_of(call.config)
            if not race.try_claim(host, policy):
                return _Left.LEFT
            race.remote_wait_secs = grant.granted - race.plan.arrival
            with self._pool.holding(host, call.task_id):
                return await self._claimed_turn(race, host, policy, call, draft)

    async def _claimed_turn(
        self, race: _Race, host: str, policy: TaskVerifyPolicy, call: _VerifyCall,
        draft: _DecisionDraft,
    ) -> _RemoteVerdict | FallbackReason:
        """Withdraw the local call, re-derive the scope at the turn, then run on *host*
        under *policy*, the one read at the claiming grant."""
        reaping = race.withdraw_local()
        if reaping is not None:
            await asyncio.shield(reaping)
        scope = await self._scope_or_none(call)
        if scope is None:
            return 'dispatcher_error'
        draft.tip_sha, draft.base_sha = scope.tip_sha, scope.base_sha
        if not self._eligible(call, policy, scope):
            return 'not_eligible'
        outcome = await self._run_remote(call, scope, host, draft)
        if isinstance(outcome, VerifyResult):
            return _RemoteVerdict(outcome, policy)
        return outcome

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
        draft.runner = self._pool.local_name
        local = await self._run_local_because(call, draft, 'cross_check' if cross else 'red_confirm')
        if cross:
            draft.cross_check = CrossCheck(remote_passed=remote.passed, local_passed=local.passed)
        if confirm:
            draft.red_confirmed_locally = RedConfirmation(local_passed=local.passed)
        return local

    async def _run_local_because(
        self, call: _VerifyCall, draft: _DecisionDraft, reason: DispatchReason,
    ) -> VerifyResult:
        """Run local under a fresh plan: before any remote queueing, and for parity re-runs (row 39)."""
        draft.reason = reason
        return await call.run(self._run_local)

    async def _run_remote(
        self, call: _VerifyCall, scope: _TaskScope, host: str, draft: _DecisionDraft,
    ) -> VerifyResult | FallbackReason:
        """Run the verify on a fresh runner for *host*'s turn; a verify still in flight
        (only on cancellation) is cancelled on its host while the turn is held."""
        started = time.monotonic()
        runner = self._pool.runner_for_turn(host)
        remote: asyncio.Future[VerifyResult] | None = None
        try:
            if await self._git_ops.has_uncommitted_work(call.worktree):
                return 'dirty_tree'
            spec = build_task_verify_spec(
                call.config, call.module_configs, scope.task_files,
                force_workspace=False, attempt_id=call.attempt_id,
                task_id=call.task_id, tip_sha=scope.tip_sha,
            )
            remote = verifying = asyncio.ensure_future(runner.run_task_verify(
                spec, task_id=call.task_id, archive_root=call.archive_root,
                event_store=call.event_store,
            ))
            return await self._remote_verdict(verifying, host, draft)
        finally:
            if remote is not None and not remote.done():
                await asyncio.shield(self._pool.cancel_in_flight(
                    host, runner, unwind=functools.partial(_reap_remote, remote),
                ))
            draft.lease_held_secs = time.monotonic() - started

    async def _remote_verdict(
        self, remote: asyncio.Future[VerifyResult], host: str, draft: _DecisionDraft,
    ) -> VerifyResult | FallbackReason:
        started = time.monotonic()
        try:
            result = await asyncio.shield(remote)
        except RunnerUnavailable as exc:
            self._pool.mark_unavailable(host, exc)
            return 'runner_unavailable'
        finally:
            draft.remote_wall_secs = time.monotonic() - started
        if not result.passed and result.category in INFRA_TRANSIENT_CATEGORIES:
            self._pool.mark_unavailable(host, _infra_transient_unavailability(host, result))
            return 'infra_transient'
        draft.runner = host
        return result

    async def _scope_or_none(self, call: _VerifyCall) -> _TaskScope | None:
        """*call*'s scope, or None (logged) when it cannot be derived."""
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
