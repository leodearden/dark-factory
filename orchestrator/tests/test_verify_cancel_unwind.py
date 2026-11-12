"""A cancelled merge verify returns only once every leg it spawned is dead.

``run_scoped_verification`` and ``run_full_verification`` (role ``merge``) run on
a scratch repo whose first module's lint leg ignores SIGTERM, so only the kill
escalation ends it.  The call is cancelled once every leg has spawned and
``_SPAWN_RETURN_SECS`` more have passed, so the cancel lands after the spawns
returned (asyncio kills a child itself when the cancel lands inside its spawn).
The lint's liveness is sampled the moment the call finishes.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from _task_verify_fixtures import VERIFY_MODULE, seed_task_branch_repo

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult, run_full_verification, run_scoped_verification

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]

_MODULES = (VERIFY_MODULE, 'mod2')
_STUBBORN = f'lint-{VERIFY_MODULE}'
_SPAWN_RETURN_SECS = 0.5


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _Legs:
    """Leg commands that append their pid to ``<label>.pids``; the long ones then spin."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()

    def spinning(self, label: str, *, ignore_term: bool = False) -> str:
        trap = "trap '' TERM; " if ignore_term else ''
        return f'{trap}{self.logging_pid(label)}; while :; do sleep 0.02; done'

    def logging_pid(self, label: str) -> str:
        return f'echo $$ >> {shlex.quote(str(self.root / f"{label}.pids"))}'

    def pids(self, label: str) -> list[int]:
        path = self.root / f'{label}.pids'
        return [int(pid) for pid in path.read_text().split()] if path.exists() else []


def _module(legs: _Legs, prefix: str) -> ModuleConfig:
    return ModuleConfig(
        prefix,
        test_command=legs.spinning(f'test-{prefix}'),
        lint_command=legs.spinning(f'lint-{prefix}', ignore_term=f'lint-{prefix}' == _STUBBORN),
        type_check_command=legs.logging_pid(f'type-{prefix}'),
    )


def _config(project_root: Path, modules: list[ModuleConfig]) -> OrchestratorConfig:
    config = OrchestratorConfig(
        project_root=project_root,
        git=GitConfig(main_branch='main'),
        verify_env_probes=[],
        verify_clock_stop_enabled=False,
        concurrent_verify=True,
    )
    config._module_configs = {mc.prefix: mc for mc in modules}
    return config


Call = Callable[[Path, OrchestratorConfig, list[ModuleConfig]], Coroutine[Any, Any, VerifyResult]]


async def _scoped(
    repo: Path, config: OrchestratorConfig, modules: list[ModuleConfig],
) -> VerifyResult:
    return await run_scoped_verification(
        repo, config, modules, task_files=[f'{mc.prefix}/test_x.py' for mc in modules],
        is_merge_verify=True, role='merge', max_retries=0,
    )


async def _full(
    repo: Path, config: OrchestratorConfig, _modules: list[ModuleConfig],
) -> VerifyResult:
    return await run_full_verification(repo, config, role='merge', max_retries=0)


async def _until(predicate: Callable[[], bool], *, watching: asyncio.Task[Any]) -> None:
    async with asyncio.timeout(30.0):
        while not predicate():
            if watching.done():
                watching.result()
                raise AssertionError('the verify finished before every leg had spawned')
            await asyncio.sleep(0.01)


@pytest.mark.parametrize('call', [_scoped, _full], ids=['scoped', 'full'])
@pytest.mark.parametrize('modules', [_MODULES[:1], _MODULES], ids=['one', 'two'])
async def test_a_cancelled_merge_verify_returns_only_after_a_leg_only_sigkill_ends(
    tmp_path: Path, call: Call, modules: tuple[str, ...],
) -> None:
    repo, _base, _tip = seed_task_branch_repo(tmp_path, modules=modules)
    legs = _Legs(tmp_path / 'legs')
    module_configs = [_module(legs, prefix) for prefix in modules]
    verify = asyncio.create_task(call(repo, _config(repo.resolve(), module_configs), module_configs))
    spinning = [f'{kind}-{prefix}' for prefix in modules for kind in ('test', 'lint')]
    types = [f'type-{prefix}' for prefix in modules]
    await _until(
        lambda: all(legs.pids(label) for label in (*spinning, *types))
        and not any(_alive(pid) for label in types for pid in legs.pids(label)),
        watching=verify,
    )
    await asyncio.sleep(_SPAWN_RETURN_SECS)
    [stubborn] = legs.pids(_STUBBORN)
    alive_at_return: list[bool] = []
    verify.add_done_callback(lambda _verify: alive_at_return.append(_alive(stubborn)))

    verify.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(30.0):
            await verify

    assert alive_at_return == [False]
    assert [pid for label in spinning for pid in legs.pids(label) if _alive(pid)] == []
