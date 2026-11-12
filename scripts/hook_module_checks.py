#!/usr/bin/env python3
"""List the merge gate's checks for a staged file set, for hooks/project-checks to run.

Reads newline-separated staged paths on stdin and prints one ``<unit>\\t<command>``
line per check: ``module:<prefix>:lint`` and ``module:<prefix>:type`` carry the
registered module's own commands, verbatim, for every module the gate would
execute for those paths; ``screen:<id>`` runs one scripts/merge_screen.yaml check
when a staged path is one the whole-tree screen reads. Exit 0 is a listing,
possibly empty; exit 2 means no listing could be derived, with the named cause on
stderr. Contract: plans/merge-lane-policy-prd.md D8 / §6 'Hook contract'.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import _checkout as _checkout
import yaml
from orchestrator.config import ConfigRequiredError, ModuleConfig, OrchestratorConfig, load_config
from orchestrator.verify_cmd import (
    ToolKind,
    VerifyCmd,
    apply_pytest_numprocesses,
    parse_config_command,
    render,
    scope_to,
)
from orchestrator.verify_evidence import module_leg_name
from orchestrator.verify_plan import (
    derive_verify_plan,
    effective_merge_module_configs,
    executed_module_prefixes,
)

TOP_CONFIG = 'dark-factory-orchestrator.yaml'
# A staged .py here can break any importer, so it type-checks every module (task 2551).
DEPENDENCY_MODULES = frozenset({'shared', 'escalation'})
SCREEN_MANIFEST = Path('scripts/merge_screen.yaml')
SCREEN_TREES = ('plans/', 'docs/', 'review/')
SCREEN_SUFFIXES = frozenset({'.yaml', '.json', '.md'})
SCREEN_TOP_KEYS = frozenset({'checks'})
SCREEN_ENTRY_KEYS = frozenset({'id', 'module', 'node_ids'})
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


@dataclass(frozen=True)
class ScreenCheck:
    id: str
    command: VerifyCmd


def member_package_dirs() -> frozenset[str]:
    return frozenset(src.parent.relative_to(_checkout.ROOT).as_posix() for src in _checkout.MEMBER_SRC)


def _screen_reads(path: str, package_dirs: frozenset[str]) -> bool:
    if path.startswith(SCREEN_TREES):
        return True
    in_package = any(path.startswith(f'{pkg}/') for pkg in package_dirs)
    return PurePosixPath(path).suffix in SCREEN_SUFFIXES and not in_package


def screen_triggered(staged: list[str], package_dirs: frozenset[str]) -> bool:
    return any(_screen_reads(path, package_dirs) for path in staged)


def _screen_base(label: str, mc: ModuleConfig) -> VerifyCmd:
    parsed = parse_config_command(mc.test_command) if mc.test_command else None
    if parsed is None or parsed.raw is not None or parsed.tool is not ToolKind.PYTEST:
        raise HookChecksError(
            f'{label}: module {mc.prefix!r} test_command {mc.test_command!r} is not one '
            'structured pytest command, so it cannot be scoped to node ids'
        )
    return parsed


def _read_manifest(path: Path) -> list[object]:
    try:
        doc: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HookChecksError(f'{SCREEN_MANIFEST}: unreadable screen manifest: {exc}') from exc
    if not isinstance(doc, dict) or set(doc) != SCREEN_TOP_KEYS:
        found = sorted(map(str, doc)) if isinstance(doc, dict) else type(doc).__name__
        raise HookChecksError(
            f'{SCREEN_MANIFEST}: top-level keys must be exactly {sorted(SCREEN_TOP_KEYS)}, found {found}'
        )
    checks = doc['checks']
    if not isinstance(checks, list) or not checks:
        raise HookChecksError(f'{SCREEN_MANIFEST}: checks must be a non-empty list, found {checks!r}')
    return checks


def _entry_fields(entry: object) -> tuple[str, Mapping[object, object]]:
    if not isinstance(entry, dict):
        raise HookChecksError(f'{SCREEN_MANIFEST}: a check must be a mapping, found {entry!r}')
    label = f'{SCREEN_MANIFEST}: check {entry.get("id")!r}'
    if set(entry) != SCREEN_ENTRY_KEYS:
        raise HookChecksError(
            f'{label}: keys must be exactly {sorted(SCREEN_ENTRY_KEYS)}, found {sorted(map(str, entry))}'
        )
    check_id = entry['id']
    if not isinstance(check_id, str) or not check_id or any(char.isspace() for char in check_id):
        raise HookChecksError(f'{label}: id must be a non-empty string with no whitespace')
    return label, entry


def _entry_node_ids(label: str, entry: Mapping[object, object], base: VerifyCmd) -> list[str]:
    node_ids = entry['node_ids']
    if not isinstance(node_ids, list) or not node_ids or not all(isinstance(n, str) and n for n in node_ids):
        raise HookChecksError(f'{label}: node_ids must be a non-empty list of strings, found {node_ids!r}')
    cwd = _checkout.ROOT / (base.cwd_rel or '')
    missing = [node for node in node_ids if not (cwd / node.split('::', 1)[0]).exists()]
    if missing:
        raise HookChecksError(f'{label}: node id file missing under {cwd}: {missing}')
    return node_ids


def _screen_check(entry: object, by_prefix: Mapping[str, ModuleConfig]) -> ScreenCheck:
    label, fields = _entry_fields(entry)
    module = fields['module']
    mc = by_prefix.get(module) if isinstance(module, str) else None
    if mc is None:
        raise HookChecksError(f'{label}: module {module!r} is not a registered module')
    base = _screen_base(label, mc)
    scoped = apply_pytest_numprocesses(scope_to(base, _entry_node_ids(label, fields, base)), '0')
    return ScreenCheck(str(fields['id']), scoped)


def load_screen_checks(path: Path, registry: list[ModuleConfig]) -> tuple[ScreenCheck, ...]:
    by_prefix = {mc.prefix: mc for mc in registry}
    checks = [_screen_check(entry, by_prefix) for entry in _read_manifest(path)]
    ids = [check.id for check in checks]
    duplicates = sorted({check_id for check_id in ids if ids.count(check_id) > 1})
    if duplicates:
        raise HookChecksError(f'{SCREEN_MANIFEST}: duplicate check id(s) {duplicates}')
    return tuple(checks)


def screen_units(checks: Sequence[ScreenCheck]) -> list[Unit]:
    return [(f'screen:{check.id}', render(check.command)) for check in checks]


def listed_units(staged: list[str]) -> list[Unit]:
    config, registry = load_registry()
    units = module_units(staged, registry, config)
    if not screen_triggered(staged, member_package_dirs()):
        return units
    checks = load_screen_checks(_checkout.ROOT / SCREEN_MANIFEST, registry)
    return [*units, *screen_units(checks)]


def _render(units: Sequence[Unit]) -> Iterator[str]:
    for unit, command in units:
        if '\t' in command or '\n' in command:
            raise HookChecksError(f'{unit}: command holds a tab or newline: {command!r}')
        yield f'{unit}\t{command}\n'


def main() -> int:
    staged = read_staged(sys.stdin.read())
    try:
        lines = list(_render(listed_units(staged)))
    except HookChecksError as exc:
        print(f'hook_module_checks: {exc}', file=sys.stderr)
        return 2
    sys.stdout.write(''.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
