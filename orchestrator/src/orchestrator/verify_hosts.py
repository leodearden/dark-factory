"""Remote verify-host construction and the task path's per-host admission gates.

PRD ``plans/remote-task-leg-verify-prd.md`` decisions 14 and 16, §5.4.  This
module sits below both the merge lane and the task workflow: it must never
import ``orchestrator.merge_lane`` or ``orchestrator.workflow``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence, Set
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from orchestrator.deployed_sha import DeployedSha, resolve_code_checkout
from orchestrator.verify import admission_policy
from orchestrator.verify_admission_gate import AdmissionGate, AdmissionPolicy
from orchestrator.verify_runner import RemoteRunner, RunnerUnavailable, RunnerUnavailableSink

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

Unwind = Callable[[], Awaitable[object]]

_CANCEL_PROBE_ATTEMPTS = 10
_CANCEL_PROBE_INTERVAL_SECS = 1.0


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


class HostPool:
    """One admission gate per remote host, the same class the local verify queue uses.

    Invariants: the pool holds no HostAllocator and reads no merge state; a hold
    record exists iff a remote attempt holds that gate's turn; the pool never
    writes the quarantine set, which it reads by reference.
    """

    local_name: ClassVar[str] = 'local'

    def __init__(
        self,
        hosts: Sequence[str],
        *,
        runner_for: Callable[[str], RemoteRunner],
        quarantine: Set[str],
        on_unavailable: RunnerUnavailableSink,
        probe_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        probe_attempts: int = _CANCEL_PROBE_ATTEMPTS,
    ) -> None:
        self._hosts = _validated_hosts(hosts, local_name=self.local_name)
        self._gates = {host: AdmissionGate() for host in self._hosts}
        self._runner_for = runner_for
        self._quarantine = quarantine
        self._on_unavailable = on_unavailable
        self._probe_sleep = probe_sleep
        self._probe_attempts = probe_attempts
        self._holds: dict[str, list[str]] = {}

    @property
    def hosts(self) -> tuple[str, ...]:
        """The remote hosts; empty means every eligible verify runs local."""
        return self._hosts

    def gate(self, host: str) -> AdmissionGate:
        self._known(host)
        return self._gates[host]

    def policy(self, host: str, config: OrchestratorConfig) -> AdmissionPolicy:
        """The local admission policy, with no slots while *host* is benched."""
        self._known(host)
        policy = admission_policy(config)
        return dataclasses.replace(policy, slots=0) if host in self._quarantine else policy

    def runner_for_turn(self, host: str) -> RemoteRunner:
        """A fresh runner for one turn, so a runner is in exclusive use at any slot count."""
        self._known(host)
        runner = self._runner_for(host)
        if runner.name != host:
            raise ValueError(f'the runner built for host {host!r} is named {runner.name!r}')
        return runner

    def note_readmitted(self, host: str, config: OrchestratorConfig) -> None:
        self.gate(host).redispatch(self.policy(host, config))

    def mark_unavailable(self, host: str, exc: RunnerUnavailable) -> None:
        self._on_unavailable(host, exc)

    @contextlib.contextmanager
    def holding(self, host: str, task_id: str) -> Iterator[None]:
        """The hold record of *task_id* on *host* for the life of the body."""
        holds = self._holds.setdefault(host, [])
        holds.append(task_id)
        try:
            yield
        finally:
            holds.remove(task_id)
            if not holds:
                del self._holds[host]

    def holders(self) -> Mapping[str, str]:
        """A fresh ``{host: 'task:<id>[,task:<id>]'}`` map of the holds now recorded."""
        return {
            host: ','.join(f'task:{task_id}' for task_id in holds)
            for host, holds in self._holds.items()
        }

    async def cancel_in_flight(self, host: str, runner: RemoteRunner, *, unwind: Unwind) -> bool:
        """Cancel *runner*'s verify on *host*, then *unwind* the dispatch using it.

        A failed cancel probes the cancelled request; a request never probed
        clean benches the host (kind ``transport``) and returns False.  The
        caller still holds its turn throughout, so a bench precedes the release.
        """
        try:
            rc = await self._cancel(host, runner)
        finally:
            await unwind()
        if rc == 0 or await self._probed_clean(runner):
            return True
        self.mark_unavailable(host, RunnerUnavailable(
            f'cancel-verify on {host!r} exited rc={rc} and the cancelled request '
            f'still runs or cannot be probed',
            kind='transport',
        ))
        return False

    async def _cancel(self, host: str, runner: RemoteRunner) -> int:
        try:
            return await runner.cancel_verify()
        except Exception:
            logger.warning(
                'cancel-verify on %s raised; treating it as a failed cancel', host, exc_info=True,
            )
            return 1

    async def _probed_clean(self, runner: RemoteRunner) -> bool:
        for attempt in range(self._probe_attempts):
            if await runner.probe_clean():
                return True
            if attempt < self._probe_attempts - 1:
                await self._probe_sleep(_CANCEL_PROBE_INTERVAL_SECS)
        return False

    def _known(self, host: str) -> None:
        if host not in self._gates:
            raise ValueError(f'unknown verify host {host!r}; known hosts: {list(self._hosts)!r}')


def _validated_hosts(hosts: Sequence[str], *, local_name: str) -> tuple[str, ...]:
    hosts = tuple(hosts)
    if local_name in hosts:
        raise ValueError(f'{local_name!r} names the local host, not a remote verify host')
    duplicates = sorted({host for host in hosts if hosts.count(host) > 1})
    if duplicates:
        raise ValueError(f'duplicate remote verify hosts: {duplicates!r}')
    return hosts
