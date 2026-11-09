"""Throwaway checkout layouts for driving ``scripts/hook_module_checks.py``.

A layout is a directory shaped like this repo where it matters to the helper:
a root ``pyproject.toml`` naming its workspace members, a top-level
``dark-factory-orchestrator.yaml`` at merge breadth ``full``, one
``<prefix>/orchestrator.yaml`` per module, and copies of the REAL helper and
``scripts/_checkout.py``. ``make_hook_repo`` turns a layout into a git repo
committing through the REAL hooks/pre-commit and hooks/project-checks, with a
PATH-stub ``uv`` that runs python, pytest and pyright with this interpreter.
Importable by bare name through ``tests/scripts/conftest.py``'s sys.path insert.
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
HOOK_SOURCES = (REPO_ROOT / 'hooks' / 'pre-commit', REPO_ROOT / 'hooks' / 'project-checks')
TOP_CONFIG_NAME = 'dark-factory-orchestrator.yaml'
SCREEN_MANIFEST_REL = 'scripts/merge_screen.yaml'
HELPER_TIMEOUT_SECS = 120
COMMIT_TIMEOUT_SECS = 600
_HOSTILE_ENV_KEYS = ('PYTEST_ADDOPTS', 'PYTEST_CURRENT_TEST', 'VIRTUAL_ENV', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE')
_STUB_UV = r"""#!/usr/bin/env bash
[ "$1" = run ] || { echo "uv stub: only 'uv run' is stubbed: $*" >&2; exit 97; }
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --project) shift 2 ;;
        --directory) cd "$2" || exit 97; shift 2 ;;
        --quiet) shift ;;
        *) break ;;
    esac
done
tool="$1"
shift
printf '%s\t%s\t%s\n' "$PWD" "$tool" "$*" >> "$UV_STUB_LOG"
case "$tool" in
    python) exec "$REAL_PYTHON" "$@" ;;
    pytest) exec "$REAL_PYTHON" -m pytest -p no:cacheprovider "$@" ;;
    pyright) exec "$REAL_PYTHON" -m pyright --pythonpath "$REAL_PYTHON" "$@" ;;
    ruff) exit 0 ;;
    *) echo "uv stub: no delegate for '$tool'" >&2; exit 97 ;;
esac
"""


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


@dataclass(frozen=True)
class HookRepo:
    """A git repo over a layout whose commits run the real hooks."""

    root: Path
    env: Mapping[str, str]
    uv_log: Path

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ['git', *args],
            cwd=self.root,
            env=dict(self.env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=check,
            timeout=COMMIT_TIMEOUT_SECS,
        )

    def stage(self, rel: str, content: str, *, executable: bool = False) -> None:
        target = write_file(self.root, rel, content)
        if executable:
            target.chmod(0o755)
        self.git('add', '--', rel)

    def commit(self, *flags: str) -> subprocess.CompletedProcess[str]:
        return self.git('commit', '-m', 't', *flags, check=False)

    def head(self) -> str:
        return self.git('rev-parse', 'HEAD').stdout.strip()

    def uv_tools(self) -> list[str]:
        """The tool each stub ``uv run`` delegated to, in call order."""
        return [line.split('\t')[1] for line in self.uv_log.read_text(encoding='utf-8').splitlines()]



def make_hook_repo(
    tmp_path: Path,
    modules: Sequence[ModuleSpec] = DEFAULT_MODULES,
    *,
    screen_manifest: str | None = None,
    top_config: bool = True,
    copies: Mapping[str, Path] | None = None,
) -> HookRepo:
    """A layout at *tmp_path*/repo with the real hooks, committed once with --no-verify.

    *copies* maps a layout-relative path to a REAL file copied there.
    """
    root = make_layout(tmp_path / 'repo', modules, screen_manifest=screen_manifest, top_config=top_config)
    for source in HOOK_SOURCES:
        target = root / 'hooks' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755)
    for rel, source in (copies or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    uv_log = tmp_path / 'uv_stub.log'
    uv_log.write_text('', encoding='utf-8')
    repo = HookRepo(root, stub_uv_env(tmp_path, uv_log), uv_log)
    subprocess.run(['git', 'init', '--quiet', '-b', 'main'], cwd=root, env=dict(repo.env), check=True)
    for key, value in (
        ('user.name', 'hook parity test'),
        ('user.email', 'hook-parity@example.invalid'),
        ('core.hooksPath', 'hooks'),
        ('commit.gpgsign', 'false'),
    ):
        repo.git('config', key, value)
    repo.git('add', '-A')
    repo.git('commit', '--quiet', '--no-verify', '-m', 'base')
    return repo


def stub_uv_env(tmp_path: Path, uv_log: Path) -> dict[str, str]:
    """This environment with a PATH-stub ``uv`` in *tmp_path*/stub_bin logging each call to *uv_log*."""
    stub_bin = tmp_path / 'stub_bin'
    stub_bin.mkdir()
    stub = stub_bin / 'uv'
    stub.write_text(_STUB_UV, encoding='utf-8')
    stub.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _HOSTILE_ENV_KEYS and not key.startswith('PYTEST_XDIST_')
    }
    env['PATH'] = f'{stub_bin}{os.pathsep}{env.get("PATH", "")}'
    env['REAL_PYTHON'] = sys.executable
    env['UV_STUB_LOG'] = str(uv_log)
    return env
