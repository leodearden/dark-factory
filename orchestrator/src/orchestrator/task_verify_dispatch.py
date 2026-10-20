"""The single routing point for a task-role verify: the local run or a remote host.

PRD ``plans/remote-task-leg-verify-prd.md`` §4.6-4.9, §5.5.  Imports point
downward only: this module must never import ``orchestrator.merge_lane`` or
``orchestrator.workflow``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, assert_never, get_args

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig
    from orchestrator.event_store import EventStore
    from orchestrator.verify import VerifyResult


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
    'dirty_tree', 'sync_stale', 'runner_unavailable', 'infra_transient', 'cross_check',
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
