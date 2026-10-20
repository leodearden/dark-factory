"""A deployed sha, a gate layout and a gate-answering transport for RemoteRunner tests (task 6565).

Every RemoteRunner dispatch first publishes the deployed sha and runs
ensure-gate over ssh (``RemoteRunner.ensure_gate``).  Tests about something
else build their runner with :func:`gated_remote_runner`, whose injected
``run`` answers those two calls itself and hands every other call to the
test's own fake, so that fake sees exactly the calls it saw before gating.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Awaitable, Callable
from typing import Any

from orchestrator.deployed_sha import DeployedSha
from orchestrator.remote_gate import GateIdentity, local_gate_identity
from orchestrator.verify_runner import RemoteRunner

TEST_DEPLOYED_SHA = DeployedSha(sha='d' * 40, indeterminate=False, dirty_wire_path=False)
TEST_DF_REMOTE = '/remote/dark-factory'
TEST_GATE_ROOT = '/remote/dark-factory-gate'
TEST_CODE_CHECKOUT = '/local/dark-factory'
TEST_GATE_BIN = f'{TEST_GATE_ROOT}/{TEST_DEPLOYED_SHA.sha}/.venv/bin/orchestrator'

Run = Callable[..., Awaitable[tuple[int, str, str]]]


def is_gate_publish(argv: list[str]) -> bool:
    """The ``git push <host>:<df> <sha>:refs/gate/<sha>`` of ``RemoteRunner.publish_sha``."""
    return argv[:2] == ['git', 'push'] and len(argv) > 3 and ':refs/gate/' in argv[3]


def is_ensure(argv: list[str]) -> bool:
    """The ssh running ensure-gate.sh."""
    return bool(argv) and argv[0] == 'ssh' and argv[-1].endswith(' ensure-gate')


def ensure_reply(*, built: bool = False, identity: GateIdentity | None = None) -> str:
    """ensure-gate.sh's stdout for a gate answering *identity* (default: this process's)."""
    answered = identity or local_gate_identity(TEST_DEPLOYED_SHA.sha)
    return json.dumps({'built': built, 'identity': answered.to_wire()})


def gated_run(inner: Run, *, built: bool = False) -> Run:
    """Wrap *inner* so the gate's publish push and ensure ssh succeed without reaching it."""
    reply = ensure_reply(built=built)

    async def run(argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        if is_gate_publish(argv):
            return (0, '', '')
        if is_ensure(argv):
            return (0, reply, '')
        return await inner(argv, cwd=cwd)

    return run


def gated_remote_runner(
    name: str, ssh_host: str, git_remote: str, cwd: Any, **kwargs: Any,
) -> RemoteRunner:
    """A RemoteRunner on the test gate layout whose ``run`` answers the gate calls."""
    if kwargs.get('run') is not None:
        kwargs['run'] = gated_run(kwargs['run'])
    kwargs.setdefault('df_remote_checkout', TEST_DF_REMOTE)
    kwargs.setdefault('gate_checkouts_path', TEST_GATE_ROOT)
    kwargs.setdefault('deployed_sha', TEST_DEPLOYED_SHA)
    kwargs.setdefault('code_checkout', TEST_CODE_CHECKOUT)
    return RemoteRunner(name, ssh_host, git_remote, cwd, **kwargs)


def gate_argv(remote_cmd: str) -> list[str]:
    """The ``orchestrator`` arguments of a remote command sent through the test gate binary."""
    words = shlex.split(remote_cmd)
    assert words[:3] == ['env', f'ORCH_GATE_BIN={TEST_GATE_BIN}', 'orchestrator'], words[:3]
    return words[3:]
