"""Behavioural-artifact fixtures (task 5296), shared by the verify-plan,
scope-inversion-boundary and task-verify-dispatch suites.

The module shapes mirror the registered modules owning every consumer test in
``orchestrator/src/orchestrator/verify_artifact_consumers.py::BEHAVIOURAL_ARTIFACTS``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from orchestrator.config import ModuleConfig, OrchestratorConfig

ARTIFACT_ORCH_MC = ModuleConfig(
    prefix='orchestrator',
    test_command='uv run --directory orchestrator pytest tests/',
    lint_command='uv run --directory orchestrator ruff check src/ tests/',
    type_check_command='uv run --directory orchestrator pyright src/ tests/',
)
ARTIFACT_SCRIPTS_MC = ModuleConfig(
    prefix='scripts',
    test_command='uv run --project shared pytest tests/scripts/ scripts/tests/',
    lint_command='uv run --project shared ruff check scripts/',
    type_check_command='uv run --project shared pyright scripts/',
)
ARTIFACT_TESTS_SCRIPTS_MC = ModuleConfig(
    prefix='tests/scripts',
    test_command='uv run --project shared pytest tests/scripts/',
    lint_command='uv run --project shared ruff check tests/scripts/',
    type_check_command='uv run --project shared pyright tests/scripts/',
)
ARTIFACT_MODULES = [ARTIFACT_ORCH_MC, ARTIFACT_SCRIPTS_MC, ARTIFACT_TESTS_SCRIPTS_MC]

SPAWN_SH = 'skills/spawn/spawn-claude.sh'
HOOK_SH = 'skills/spawn/hooks/stop.sh'
UNBLOCK_SKILL = 'skills/unblock-auto/SKILL.md'
WATCHER_SKILL = 'skills/escalation-watcher-auto/SKILL.md'
ROOT_YAML = 'dark-factory-orchestrator.yaml'

SPAWN_CONSUMERS: dict[str, tuple[str, ...]] = {
    'orchestrator': (
        'orchestrator/tests/test_session_hooks.py',
        'orchestrator/tests/test_session_hooks_bash.py',
    ),
    'scripts': ('scripts/tests/test_legibility_inventory.py',),
    'tests/scripts': ('tests/scripts/test_spawn_claude.py',),
}
SKILL_PROMPT_CONSUMERS = (
    'orchestrator/tests/test_harness_watcher_supervisor.py',
    'orchestrator/tests/test_skill_prompt.py',
)


def registry_config(registry: Iterable[ModuleConfig], **fields: Any) -> OrchestratorConfig:
    """``OrchestratorConfig(**fields)`` whose module registry is *registry*.

    Passing none of the registry as the task's own module set gives the shape
    of a skills-only or root-yaml task.
    """
    config = OrchestratorConfig(**fields)
    config._module_configs = {mc.prefix: mc for mc in registry}
    return config
