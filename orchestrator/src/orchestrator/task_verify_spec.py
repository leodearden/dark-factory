"""The task-role verify spec a workstation ships to a remote host, and its codec.

Built on the dispatching workstation, applied onto the remote host's own
config there.  Field classification (the task-5295 audit pattern):

- CARRIED, verdict-deciding: ``verify_commands`` (the spec-authoritative
  module set, task 4536), ``global_verify_command`` (zero-module projects,
  INV-1 / task 2883), ``task_files`` (derived on the dispatcher; never None,
  so the host never diffs against a main ref it does not maintain),
  ``force_workspace``, ``verify_env`` (task 5496: the spec wins, host-only
  keys survive), ``scope_cargo``, ``verify_cold_preprovision_command``
  (non-empty replaces the host's; '' keeps the host's own).
- IDENTITY: ``attempt_id``, ``task_id``, ``tip_sha`` (the pushed ref's target).
- HOST-SHAPED, deliberately absent: role (the type is the role), pytest_n /
  verify_admission_pytest_n, base_sha, every timeout, verify_timeout_retries,
  concurrent_verify, the nice/cgroup knobs and clock-stop.
- COLLAPSED: a per-module ``verify_env`` (``verify.py::_resolve_verify_env``
  merges it per module) or per-module ``scope_cargo`` cannot ride one
  spec-level value, so ``collapsed_module_overrides`` names them and the
  build refuses rather than ship a different verdict-deciding run.

PRD: plans/remote-task-leg-verify-prd.md §4.1, §5.1.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_spec_shared import (
    VerifyCommand,
    apply_spec_to_host_config,
    canonical_json,
    global_verify_command_for,
    module_config_from_command,
    require_exact_keys,
)

__all__ = [
    "TaskVerifySpec",
    "CollapsedOverride",
    "collapsed_module_overrides",
    "build_task_verify_spec",
    "apply_task_verify_spec",
    "spec_to_dict",
    "spec_from_dict",
    "task_spec_to_json",
    "task_spec_from_json",
]

_FULL_SHA = re.compile(r'[0-9a-f]{40}')


@dataclass(frozen=True)
class TaskVerifySpec:
    verify_commands: tuple[VerifyCommand, ...]
    global_verify_command: VerifyCommand | None
    task_files: tuple[str, ...]
    force_workspace: bool
    verify_env: Mapping[str, str]
    scope_cargo: bool
    verify_cold_preprovision_command: str
    attempt_id: int
    task_id: str
    tip_sha: str

    def __post_init__(self) -> None:
        if not (
            isinstance(self.verify_commands, tuple)
            and all(isinstance(vc, VerifyCommand) for vc in self.verify_commands)
        ):
            raise TypeError(
                f'TaskVerifySpec.verify_commands must be a tuple of VerifyCommand, '
                f'got {self.verify_commands!r}'
            )
        if not (
            self.global_verify_command is None
            or isinstance(self.global_verify_command, VerifyCommand)
        ):
            raise TypeError(
                f'TaskVerifySpec.global_verify_command must be a VerifyCommand or None, '
                f'got {self.global_verify_command!r}'
            )
        if not (
            isinstance(self.task_files, tuple)
            and all(isinstance(path, str) for path in self.task_files)
        ):
            raise TypeError(
                f'TaskVerifySpec.task_files must be a tuple of str, got {self.task_files!r}'
            )
        if isinstance(self.attempt_id, bool) or not isinstance(self.attempt_id, int):
            raise TypeError(
                f'TaskVerifySpec.attempt_id must be an int, got {self.attempt_id!r}'
            )
        if not (isinstance(self.tip_sha, str) and _FULL_SHA.fullmatch(self.tip_sha)):
            raise ValueError(
                f'TaskVerifySpec.tip_sha must be a full 40-hex lowercase SHA, '
                f'got {self.tip_sha!r}'
            )
        object.__setattr__(self, 'verify_env', MappingProxyType(dict(self.verify_env)))


@dataclass(frozen=True)
class CollapsedOverride:
    prefix: str
    field: Literal['verify_env', 'scope_cargo']


def collapsed_module_overrides(
    module_configs: Iterable[ModuleConfig],
) -> tuple[CollapsedOverride, ...]:
    """The per-module overrides a TaskVerifySpec cannot carry, in module order."""
    collapsed: list[CollapsedOverride] = []
    for mc in module_configs:
        if mc.verify_env:
            collapsed.append(CollapsedOverride(mc.prefix, 'verify_env'))
        if mc.scope_cargo is not None:
            collapsed.append(CollapsedOverride(mc.prefix, 'scope_cargo'))
    return tuple(collapsed)


def build_task_verify_spec(
    config: OrchestratorConfig,
    module_configs: Sequence[ModuleConfig],
    task_files: Iterable[str],
    *,
    force_workspace: bool,
    attempt_id: int,
    task_id: str,
    tip_sha: str,
) -> TaskVerifySpec:
    """Project the dispatching workstation's config and module set into a spec.

    Raises ValueError when a module carries an override the spec would collapse.
    """
    collapsed = collapsed_module_overrides(module_configs)
    if collapsed:
        named = ', '.join(f'{c.prefix} ({c.field})' for c in collapsed)
        raise ValueError(
            f'a TaskVerifySpec cannot carry per-module overrides; collapsed: {named}'
        )
    verify_commands = tuple(VerifyCommand.from_module_config(mc) for mc in module_configs)
    return TaskVerifySpec(
        verify_commands=verify_commands,
        global_verify_command=global_verify_command_for(config, verify_commands),
        task_files=tuple(task_files),
        force_workspace=force_workspace,
        verify_env=dict(config.effective_verify_env),
        scope_cargo=config.scope_cargo,
        verify_cold_preprovision_command=config.verify_cold_preprovision_command,
        attempt_id=attempt_id,
        task_id=task_id,
        tip_sha=tip_sha,
    )


def apply_task_verify_spec(
    config: OrchestratorConfig,
    spec: TaskVerifySpec,
) -> tuple[OrchestratorConfig, list[ModuleConfig]]:
    """The remote host's half: its own *config* with *spec* applied, plus the rebuilt modules.

    The modules are host-shaped (no per-module env, no cold timeout): the spec
    env is applied once, at config level.
    """
    module_configs = [module_config_from_command(vc) for vc in spec.verify_commands]
    field_overrides: dict[str, object] = {'scope_cargo': spec.scope_cargo}
    if spec.verify_cold_preprovision_command:
        field_overrides['verify_cold_preprovision_command'] = (
            spec.verify_cold_preprovision_command
        )
    applied = apply_spec_to_host_config(
        config,
        verify_env=spec.verify_env,
        global_verify_command=spec.global_verify_command,
        module_configs=module_configs,
        field_overrides=field_overrides,
    )
    return applied, module_configs


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(TaskVerifySpec))
_COMMAND_KEYS = frozenset(f.name for f in dataclasses.fields(VerifyCommand))


def spec_to_dict(spec: TaskVerifySpec) -> dict:
    return {
        'verify_commands': [vc.to_dict() for vc in spec.verify_commands],
        'global_verify_command': (
            spec.global_verify_command.to_dict()
            if spec.global_verify_command is not None
            else None
        ),
        'task_files': list(spec.task_files),
        'force_workspace': spec.force_workspace,
        'verify_env': dict(spec.verify_env),
        'scope_cargo': spec.scope_cargo,
        'verify_cold_preprovision_command': spec.verify_cold_preprovision_command,
        'attempt_id': spec.attempt_id,
        'task_id': spec.task_id,
        'tip_sha': spec.tip_sha,
    }


def _command_from_dict(d: dict) -> VerifyCommand:
    require_exact_keys(d, _COMMAND_KEYS, 'VerifyCommand')
    return VerifyCommand.from_dict(d)


def spec_from_dict(d: dict) -> TaskVerifySpec:
    """Decode strictly: unknown or missing keys, at top level or in any command, are a TypeError.

    Arrays become tuples only when they ARE arrays, so any other wire value
    reaches ``__post_init__`` unchanged and is rejected there by field name.
    """
    require_exact_keys(d, _FIELD_NAMES, 'TaskVerifySpec')
    commands = d['verify_commands']
    if not isinstance(commands, list):
        raise TypeError(
            f'TaskVerifySpec.verify_commands must be a JSON array, got {commands!r}'
        )
    raw_global = d['global_verify_command']
    raw_files = d['task_files']
    return TaskVerifySpec(
        verify_commands=tuple(_command_from_dict(c) for c in commands),
        global_verify_command=(
            _command_from_dict(raw_global) if raw_global is not None else None
        ),
        task_files=tuple(raw_files) if isinstance(raw_files, list) else raw_files,
        force_workspace=d['force_workspace'],
        verify_env=d['verify_env'],
        scope_cargo=d['scope_cargo'],
        verify_cold_preprovision_command=d['verify_cold_preprovision_command'],
        attempt_id=d['attempt_id'],
        task_id=d['task_id'],
        tip_sha=d['tip_sha'],
    )


def task_spec_to_json(spec: TaskVerifySpec) -> str:
    return canonical_json(spec_to_dict(spec))


def task_spec_from_json(s: str) -> TaskVerifySpec:
    return spec_from_dict(json.loads(s))
