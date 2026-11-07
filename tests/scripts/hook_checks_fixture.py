"""Throwaway checkout layouts for driving ``scripts/hook_module_checks.py``.

A layout is a directory shaped like this repo where it matters to the helper:
a root ``pyproject.toml`` naming its workspace members, a top-level
``dark-factory-orchestrator.yaml`` at merge breadth ``full``, one
``<prefix>/orchestrator.yaml`` per module, and copies of the REAL helper and
``scripts/_checkout.py``. Importable by bare name through
``tests/scripts/conftest.py``'s sys.path insert.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).parents[2]
HELPER_SOURCE = REPO_ROOT / 'scripts' / 'hook_module_checks.py'
CHECKOUT_SOURCE = REPO_ROOT / 'scripts' / '_checkout.py'
TOP_CONFIG_NAME = 'dark-factory-orchestrator.yaml'
SCREEN_MANIFEST_REL = 'scripts/merge_screen.yaml'
HELPER_TIMEOUT_SECS = 120


@dataclass(frozen=True)
class ModuleSpec:
    """One module of a layout; *package* makes it a workspace member with a src dir."""

    prefix: str
    lint_command: str | None
    type_check_command: str | None
    test_command: str | None
    package: bool


SCRIPTS = ModuleSpec(
    prefix='scripts',
    lint_command='uv run --project shared ruff check scripts/',
    type_check_command='uv run --project shared pyright scripts/',
    test_command='uv run --project shared pytest scripts/tests/ -q',
    package=False,
)
SHARED = ModuleSpec(
    prefix='shared',
    lint_command='uv run --directory shared ruff check src/ tests/',
    type_check_command='uv run --directory shared pyright src/ tests/',
    test_command='uv run --directory shared pytest tests/ -q',
    package=True,
)
ESCALATION = ModuleSpec(
    prefix='escalation',
    lint_command='uv run --directory escalation ruff check src/ tests/',
    type_check_command='uv run --directory escalation pyright src/ tests/',
    test_command='uv run --directory escalation pytest tests/ -q',
    package=True,
)
ORCHESTRATOR = ModuleSpec(
    prefix='orchestrator',
    lint_command='uv run --directory orchestrator ruff check src/ tests/',
    type_check_command='uv run --directory orchestrator pyright src/ tests/',
    test_command='uv run --directory orchestrator pytest tests/ -q',
    package=True,
)
DEFAULT_MODULES: tuple[ModuleSpec, ...] = (SCRIPTS, SHARED, ESCALATION, ORCHESTRATOR)


class HelperRun(NamedTuple):
    rc: int
    units: list[tuple[str, str]]
    stderr: str


def make_layout(
    root: Path,
    modules: Sequence[ModuleSpec] = DEFAULT_MODULES,
    *,
    screen_manifest: str | None = None,
    top_config: bool = True,
) -> Path:
    """Build a layout at *root* (created if absent) and return it."""
    root.mkdir(parents=True, exist_ok=True)
    members = [m.prefix for m in modules if m.package]
    (root / 'pyproject.toml').write_text(
        f'[project]\nname = "layout"\nversion = "0"\n\n'
        f'[tool.uv.workspace]\nmembers = {json.dumps(members)}\n',
        encoding='utf-8',
    )
    if top_config:
        (root / TOP_CONFIG_NAME).write_text('merge_verify_breadth: full\n', encoding='utf-8')
    for module in modules:
        _write_module(root, module)
    scripts_dir = root / 'scripts'
    scripts_dir.mkdir(exist_ok=True)
    shutil.copyfile(HELPER_SOURCE, scripts_dir / HELPER_SOURCE.name)
    shutil.copyfile(CHECKOUT_SOURCE, scripts_dir / CHECKOUT_SOURCE.name)
    if screen_manifest is not None:
        (root / SCREEN_MANIFEST_REL).write_text(screen_manifest, encoding='utf-8')
    return root


def _write_module(root: Path, module: ModuleSpec) -> None:
    module_dir = root / module.prefix
    module_dir.mkdir(parents=True, exist_ok=True)
    if module.package:
        (module_dir / 'src').mkdir(exist_ok=True)
    commands = {
        'lint_command': module.lint_command,
        'type_check_command': module.type_check_command,
        'test_command': module.test_command,
    }
    body = {key: value for key, value in commands.items() if value is not None}
    (module_dir / 'orchestrator.yaml').write_text(yaml.safe_dump(body), encoding='utf-8')


def write_file(root: Path, rel: str, content: str = '') -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    return target


def run_helper(
    layout: Path,
    staged: Sequence[str],
    env_overrides: Mapping[str, str] | None = None,
) -> HelperRun:
    """Run *layout*'s copy of the helper with *staged* on stdin, from *layout*."""
    env = {**os.environ, **(env_overrides or {})}
    proc = subprocess.run(
        [sys.executable, str(layout / 'scripts' / HELPER_SOURCE.name)],
        cwd=layout,
        env=env,
        input=''.join(f'{path}\n' for path in staged),
        capture_output=True,
        text=True,
        timeout=HELPER_TIMEOUT_SECS,
    )
    return HelperRun(proc.returncode, parse_listing(proc.stdout), proc.stderr)


def parse_listing(stdout: str) -> list[tuple[str, str]]:
    """The helper's ``unit<TAB>command`` lines as pairs; any other shape fails loudly."""
    units: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        unit, sep, command = line.partition('\t')
        assert sep and unit and command and '\t' not in command, (
            f'not a unit<TAB>command line: {line!r}'
        )
        units.append((unit, command))
    return units
