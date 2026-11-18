"""Test doubles and helpers shared by the orchestrator.proc_supervision test files.

Imported by bare module name (``from _proc_supervision_doubles import ...``),
like ``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.

One copy each of the subprocess-runner and unit-inspector doubles, the
canonical detached RP-4 restart plan, the ``find_spec`` stub, the
``tmp_queue_dir`` escalation queue_dir fixture, the ``read_escalations``
EscalationQueue read helper, the environment-recording probe and the model of
a fired ``systemd-run`` unit, so every proc_supervision test file pins the
same shapes and degrades import roots the same way. A module using the
fixture re-exports it by redundant alias
(``from _proc_supervision_doubles import tmp_queue_dir as tmp_queue_dir``):
the import binding is what registers it with pytest, and the alias form
needs no lint suppression.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.proc_supervision import EscalationSpec, RestartPlan


class FakeRunner:
    """Async callable matching ``asyncio.create_subprocess_exec``'s signature.

    Records every ``(argv, kwargs)`` call in ``self.calls`` so tests can
    assert on exact positional argv and keyword args (cwd=, stdout=, ...).
    Returns a ``MagicMock`` proc whose ``communicate()`` is an
    ``AsyncMock(return_value=(stdout, None))`` and whose ``returncode`` is
    configurable (default 0) — mirroring the ``fake_proc`` idiom already used
    throughout test_service_restart.py and test_deterministic_runner.py.
    """

    def __init__(self, returncode: int = 0, stdout: bytes = b'') -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[tuple[tuple, dict]] = []
        # Every fake proc this runner has handed back, in call order — lets a
        # test assert on a spawned proc's post-return state (e.g. that
        # .communicate() was never awaited on a fire-and-forget leaf spawn)
        # without execute() needing to hand the proc back to the caller.
        self.procs: list[MagicMock] = []

    async def __call__(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(self.stdout, None))
        proc.returncode = self.returncode
        self.procs.append(proc)
        return proc


def make_fake_inspector(state: dict):
    """Build an async inspector callable: ``(unit, *, timeout_secs, reap_grace_secs=5.0) -> dict``.

    Returns a shallow copy of *state* on every call regardless of args, and
    records each call's kwargs on the returned callable's ``.calls`` list
    (each entry: ``{'unit', 'timeout_secs', 'reap_grace_secs'}``) so tests can
    assert the exact unit/timeout the RP-5 verify leg was invoked with.
    Mirrors the injected-inspector pattern in test_deterministic_runner.py
    (AsyncMock ``unit_inspector``).
    """
    calls: list[dict] = []

    async def _inspector(unit: str, *, timeout_secs: float, reap_grace_secs: float = 5.0) -> dict:
        calls.append({
            'unit': unit,
            'timeout_secs': timeout_secs,
            'reap_grace_secs': reap_grace_secs,
        })
        return dict(state)

    _inspector.calls = calls  # type: ignore[attr-defined]
    return _inspector


@pytest.fixture
def tmp_queue_dir(tmp_path: Path) -> Path:
    """A ``tmp_path`` subdirectory usable as an ``EscalationQueue`` queue_dir.

    Not pre-created — ``EscalationQueue.__init__`` mkdir(parents=True,
    exist_ok=True)'s it, matching production (the queue dir may not yet exist
    when the first escalation is filed).
    """
    return tmp_path / 'escalations'


def read_escalations(
    queue_dir: str | Path,
    task_id: str,
    *,
    status: str | None = None,
    level: int | None = None,
    agent_role: str | None = None,
) -> list:
    """Construct ``EscalationQueue(queue_dir)`` and return ``get_by_task(task_id, ...)``.

    Product read path (PRD Sec 7): tests assert filed escalations through
    this helper rather than inspecting on-disk JSON directly.
    """
    return EscalationQueue(Path(queue_dir)).get_by_task(
        task_id, status=status, level=level, agent_role=agent_role,
    )


def detached_rp4_plan(queue_dir: Path, *, with_spec: bool) -> RestartPlan:
    """The canonical detached (RP-4) self-restart plan.

    *with_spec* attaches the on-failure escalation filed into *queue_dir* as
    ``task-99``; without it the plan has no submit child at all.
    """
    spec = EscalationSpec(
        queue_dir=str(queue_dir),
        task_id='task-99',
        summary='Self-restart fire-time failure',
    )
    return RestartPlan(
        script=Path('/proj/scripts/restart-orchestrator.sh'),
        args=['--foo'],
        cwd=Path('/proj'),
        target_unit='orch.service',
        own_unit='orch.service',
        on_failure_escalation=spec if with_spec else None,
        verify=None,
        transient_unit='orch-redeploy-restart-99.service',
        on_active_secs=10,
    )


def stub_find_spec(monkeypatch: pytest.MonkeyPatch, unresolvable: set[str]) -> None:
    """Make ``importlib.util.find_spec`` return None for *unresolvable* names.

    Every other name delegates to the real resolver, so this isolates the
    package under test without disturbing unrelated imports.
    """
    real = importlib.util.find_spec

    def _fake(name: str, package: str | None = None):
        if name in unresolvable:
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, 'find_spec', _fake)


def write_env_probe(path: Path, *, record_to: Path, exit_code: int) -> Path:
    """Write an executable ``#!/bin/sh`` stub that ignores argv, records its
    whole environment into *record_to* (read it back with
    :func:`read_env_probe`) and exits *exit_code*."""
    path.write_text(
        '#!/bin/sh\n'
        f'env -0 > {shlex.quote(str(record_to))}\n'
        f'exit {exit_code}\n'
    )
    path.chmod(0o755)
    return path


def read_env_probe(record: Path) -> dict[str, str]:
    """The environment a :func:`write_env_probe` stub ran with."""
    env = {}
    for entry in filter(None, record.read_text().split('\0')):
        name, _, value = entry.partition('=')
        env[name] = value
    return env


# MEASURED in task 3453: ``systemd-run --user`` propagates none of the caller's
# environment; the unit starts from the user manager's own. This is a strict
# subset of any real user manager's environment, so a unit that works here
# cannot be leaning on something only the caller's environment supplies.
USER_MANAGER_ENV: Mapping[str, str] = MappingProxyType(
    {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home())},
)

# Inside pytest-timeout's 60s per-test budget, so a wedged child is named by
# the AssertionError below rather than by an os._exit of the xdist worker
# (same reasoning as test_deterministic_runner.py::_CHILD_TIMEOUT_SECS).
_FIRED_UNIT_TIMEOUT_SECS = 30

_EXECSTART_EXPANSION = re.compile(r'\$(?:\$|\{([A-Za-z_][A-Za-z0-9_]*)\})')


def _systemd_run_unit_env_and_cwd(
    options: Sequence[str], unit_env: Mapping[str, str],
) -> tuple[dict[str, str], str]:
    env = dict(unit_env)
    cwd = None
    for option in options:
        if option.startswith('--setenv='):
            name, _, value = option.removeprefix('--setenv=').partition('=')
            env[name] = value
        elif option.startswith('--working-directory='):
            cwd = option.removeprefix('--working-directory=')
    assert cwd is not None, f'no --working-directory option in {options!r}'
    return env, cwd


def _as_systemd_execstart_runs(word: str, env: Mapping[str, str]) -> str:
    """systemd's ``$$`` -> ``$`` and ``${NAME}`` expansion inside one ExecStart argv word."""
    return _EXECSTART_EXPANSION.sub(
        lambda m: '$' if m.group(0) == '$$' else env.get(m.group(1), ''),
        word,
    )


async def fire_like_systemd_run(
    argv: Sequence[str], *, unit_env: Mapping[str, str] = USER_MANAGER_ENV,
) -> tuple[int, str]:
    """Run a registered ``systemd-run ... /bin/sh -c <wrapped>`` unit now, the
    way the user manager would fire it; return ``(rc, combined_output)``.

    The command starts from *unit_env* plus any ``--setenv=`` options, never
    this process's environment; runs in ``--working-directory``; and has each
    ExecStart word put through systemd's ``$`` expansion first.
    """
    sh_at = list(argv).index('/bin/sh')
    env, cwd = _systemd_run_unit_env_and_cwd(argv[:sh_at], unit_env)
    proc = await asyncio.create_subprocess_exec(
        *(_as_systemd_execstart_runs(word, env) for word in argv[sh_at:]),
        env=env,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_FIRED_UNIT_TIMEOUT_SECS)
    except TimeoutError:
        proc.kill()
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
            out = b''
        raise AssertionError(
            f'the fired unit did not exit within {_FIRED_UNIT_TIMEOUT_SECS}s and '
            f'was killed. argv: {argv!r}. Output before the kill: '
            f'{out.decode(errors="replace")!r}'
        ) from None
    output = out.decode(errors='replace')
    rc = proc.returncode
    assert rc is not None, f'communicate() returned but the child has no exit status: {output!r}'
    return rc, output
