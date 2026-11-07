#!/usr/bin/env python3
"""List the merge gate's checks for a staged file set, for hooks/project-checks to run.

Reads newline-separated staged paths on stdin and prints one ``<unit>\\t<command>``
line per check: ``module:<prefix>:lint`` and ``module:<prefix>:type`` carry the
registered module's own commands, verbatim, for every module the gate would
execute for those paths. Exit 0 is a listing, possibly empty; exit 2 means no
listing could be derived, with the named cause on stderr. Contract:
plans/merge-lane-policy-prd.md D8 / §6 'Hook contract'.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence

import _checkout as _checkout
from orchestrator.config import ConfigRequiredError, ModuleConfig, OrchestratorConfig, load_config
from orchestrator.verify_evidence import module_leg_name
from orchestrator.verify_plan import (
    derive_verify_plan,
    effective_merge_module_configs,
    executed_module_prefixes,
)

TOP_CONFIG = 'dark-factory-orchestrator.yaml'
# A staged .py here can break any importer, so it type-checks every module (task 2551).
DEPENDENCY_MODULES = frozenset({'shared', 'escalation'})
Unit = tuple[str, str]


class HookChecksError(Exception):
    """No listing can be derived; the message names why."""


def read_staged(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def load_registry() -> tuple[OrchestratorConfig, list[ModuleConfig]]:
    os.environ['ORCH_PROJECT_ROOT'] = str(_checkout.ROOT)
    try:
        config = load_config(_checkout.ROOT / TOP_CONFIG)
    except ConfigRequiredError as exc:
        raise HookChecksError(str(exc)) from exc
    registry = effective_merge_module_configs(config, list(config.module_configs_or_empty.values()))
    return config, registry


def _read_checkout_file(path: str) -> str | None:
    try:
        return (_checkout.ROOT / path).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def touched_modules(
    staged: list[str], registry: list[ModuleConfig], config: OrchestratorConfig,
) -> frozenset[str]:
    plan = derive_verify_plan(staged, registry, config, _read_checkout_file, role='task')
    return executed_module_prefixes(plan)


def _require_dependency_modules(registry: list[ModuleConfig]) -> None:
    unregistered = sorted(DEPENDENCY_MODULES - {mc.prefix for mc in registry})
    if unregistered:
        raise HookChecksError(
            f'dependency module(s) {unregistered} are not registered modules; '
            'rename DEPENDENCY_MODULES with the module'
        )


def module_units(
    staged: list[str], registry: list[ModuleConfig], config: OrchestratorConfig,
) -> list[Unit]:
    _require_dependency_modules(registry)
    touched = touched_modules(staged, registry, config)
    typed = registry if touched & DEPENDENCY_MODULES else [mc for mc in registry if mc.prefix in touched]
    lint = [(mc.prefix, 'lint', mc.lint_command) for mc in registry if mc.prefix in touched]
    type_ = [(mc.prefix, 'type', mc.type_check_command) for mc in typed]
    return [
        (module_leg_name(prefix, leg), command)
        for prefix, leg, command in (*lint, *type_)
        if command is not None
    ]


def _render(units: Sequence[Unit]) -> Iterator[str]:
    for unit, command in units:
        if '\t' in command or '\n' in command:
            raise HookChecksError(f'{unit}: command holds a tab or newline: {command!r}')
        yield f'{unit}\t{command}\n'


def main() -> int:
    staged = read_staged(sys.stdin.read())
    try:
        config, registry = load_registry()
        lines = list(_render(module_units(staged, registry, config)))
    except HookChecksError as exc:
        print(f'hook_module_checks: {exc}', file=sys.stderr)
        return 2
    sys.stdout.write(''.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
