"""Every registered module's task verify runs the raw envelope-literal guard over its own tree.

Task verify narrows pytest to the touched tests, so the whole-repo live check in
``test_no_raw_envelope_literal.py`` runs only at merge. The ``&&`` tail of the
lint slot is the one leg task verify runs verbatim and unscoped, so each
module's ``lint_command`` ends in ``python3 <GUARD> <its own prefix>``.

These tests read the lint command task verify EXECUTES, through the production
planner, and enumerate modules with the production discovery, so a module added
later must carry the leg on the day it lands.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import raw_envelope_literal_guard

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig, OrchestratorConfig

REPO_ROOT = Path(__file__).parents[2]
GUARD = Path(raw_envelope_literal_guard.__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
KNOWN_MODULES = frozenset({'orchestrator', 'tests/scripts'})


def _guard_leg(prefix: str) -> list[str]:
    return ['python3', GUARD, prefix]


def _registered_modules(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> dict[str, ModuleConfig]:
    configs = discover_module_configs()
    assert configs.keys() >= KNOWN_MODULES, (
        f'module discovery found {sorted(configs)}, missing {sorted(KNOWN_MODULES - configs.keys())}; '
        'a check over every registered module would pass vacuously'
    )
    return configs


def test_every_module_task_verify_runs_the_guard_over_its_own_tree(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
    executed_for_touched: Callable[[str, list[str], OrchestratorConfig], ModuleConfig],
    root_config: OrchestratorConfig,
) -> None:
    from orchestrator.verify_cmd import split_top_level_and

    failures: list[str] = []
    for prefix in sorted(_registered_modules(discover_module_configs)):
        leg = _guard_leg(prefix)
        executed = executed_for_touched(prefix, [f'{prefix}/guard_wiring_probe.py'], root_config)
        if not executed.lint_command:
            failures.append(f'{prefix}: has no lint_command; give it one ending in && {shlex.join(leg)}')
            continue
        legs = [shlex.split(segment) for segment in split_top_level_and(executed.lint_command)]
        if legs.count(leg) != 1:
            failures.append(
                f'{prefix}: task verify executes {executed.lint_command!r}; append '
                f'&& {shlex.join(leg)} to {prefix}/orchestrator.yaml lint_command'
            )
    assert not failures, (
        "These modules' task verify does not run the raw envelope-literal guard exactly "
        'once over their own tree:\n' + '\n'.join(failures)
    )


def test_every_module_lint_command_stays_a_ruff_command(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    from orchestrator.verify_cmd import ToolKind, parse_config_command

    not_ruff = {
        prefix: mc.lint_command
        for prefix, mc in sorted(_registered_modules(discover_module_configs).items())
        if not mc.lint_command or parse_config_command(mc.lint_command).tool is not ToolKind.RUFF
    }
    assert not not_ruff, (
        f'These lint_commands no longer parse as ruff: {not_ruff}. A pytest token anywhere '
        'in the chain makes the whole merge-time lint leg a pytest command '
        '(orchestrator/src/orchestrator/verify_cmd.py::_parse_chain), so the guard leg '
        'must stay a plain python3 script.'
    )
