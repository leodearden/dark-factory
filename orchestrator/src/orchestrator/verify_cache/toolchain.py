"""The environment a verify leg runs under, as cache-key inputs."""

from __future__ import annotations

import asyncio
import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.verify_cache import tree_objects
from orchestrator.verify_cache.modes import VERIFY_ROLES, VerifyRole

FINGERPRINTED_TOOLS = ('pyright', 'ruff')

_UV_LOCK = 'uv.lock'
_PACKAGE_LOCK = 'package-lock.json'
_PYVENV_CFG = Path('.venv') / 'pyvenv.cfg'


class ToolchainUnavailable(RuntimeError):
    """The toolchain a leg would run under cannot be known, so no key may name it."""


@dataclass(frozen=True)
class ToolchainFingerprint:
    """Everything outside the tree that decides a leg's verdict.

    ``tool_versions`` holds ``(tool, lockfile, version)`` and ``verify_env``
    ``(name, value)``; both are sorted at construction, so their order is not
    identity. ``orchestrator_code`` identifies the orchestrator code that
    rewrites, runs and judges the legs (its deployed code sha): that code is
    not in a target's tree, and for this repo the running copy is not the
    tree under verify.
    """

    role: VerifyRole
    interpreter: str | None
    uv_lock_blob: str | None
    tool_versions: tuple[tuple[str, str, str], ...]
    verify_env: tuple[tuple[str, str], ...]
    preprovision_command: str
    orchestrator_code: str

    def __post_init__(self) -> None:
        if self.role not in VERIFY_ROLES:
            raise ValueError(f'toolchain role {self.role!r} is not one of {VERIFY_ROLES}')
        if not self.orchestrator_code:
            raise ValueError('a toolchain fingerprint needs the orchestrator code identity')
        object.__setattr__(self, 'tool_versions', tuple(sorted(self.tool_versions)))
        object.__setattr__(self, 'verify_env', tuple(sorted(self.verify_env)))
        _require_unique('tool lockfile', [(tool, source) for tool, source, _v in self.tool_versions])
        _require_unique('verify_env name', [name for name, _value in self.verify_env])

    def as_inputs(self) -> tuple[tuple[str, str], ...]:
        """The fingerprint as sorted ``('toolchain.<field>', value)`` key inputs."""
        return tuple(sorted((
            ('toolchain.role', self.role),
            ('toolchain.interpreter', self.interpreter or 'none'),
            ('toolchain.uv_lock', self.uv_lock_blob or 'none'),
            *((f'toolchain.tool.{tool}@{source}', version) for tool, source, version in self.tool_versions),
            *((f'toolchain.env.{name}', value) for name, value in self.verify_env),
            ('toolchain.preprovision', self.preprovision_command),
            ('toolchain.orchestrator_code', self.orchestrator_code),
        )))


def interpreter_from_pyvenv_cfg(text: str) -> str:
    """``'<implementation> <version_info>'`` from a venv's pyvenv.cfg text."""
    settings = dict(
        (key.strip(), value.strip())
        for key, sep, value in (line.partition('=') for line in text.splitlines())
        if sep
    )
    missing = [key for key in ('implementation', 'version_info') if key not in settings]
    if missing:
        raise ToolchainUnavailable(f'pyvenv.cfg declares no {", ".join(missing)}')
    return f"{settings['implementation']} {settings['version_info']}"


def tool_versions_from_lockfiles(
    uv_lock_text: str | None, package_lock_text: str | None,
) -> tuple[tuple[str, str, str], ...]:
    """Each fingerprinted tool's locked version per lockfile; an absent lockfile adds nothing.
    A tool locked at several versions in one lockfile is one triple of its sorted versions."""
    triples: list[tuple[str, str, str]] = []
    if uv_lock_text is not None:
        triples += _locked_versions(_UV_LOCK, _uv_lock_packages(uv_lock_text))
    if package_lock_text is not None:
        triples += _locked_versions(_PACKAGE_LOCK, _package_lock_packages(package_lock_text))
    return tuple(sorted(triples))


async def read_toolchain(
    worktree: Path,
    repo: Path,
    tree: str,
    *,
    role: VerifyRole,
    verify_env: Mapping[str, str],
    preprovision_command: str,
    orchestrator_code: str,
) -> ToolchainFingerprint:
    """The toolchain *tree*'s legs run under in *worktree*.

    Lockfiles come from the dispatched tree (INV-3); the interpreter from the
    worktree's own ``.venv/pyvenv.cfg``, since worktree venvs differ. A tree
    with a uv.lock whose worktree has no venv yet is unavailable: the
    interpreter that will run its legs is unknown. Pass the effective env the
    legs see (``verify.py::_resolve_verify_env``); *role* is a field of its own
    as well, so the role is in every key even when a caller passes a raw
    ``config.verify_env``.
    """
    try:
        ids = await tree_objects.object_ids(repo, tree, [_UV_LOCK, _PACKAGE_LOCK])
        uv_lock_text, package_lock_text = await asyncio.gather(
            *(_present_blob_text(repo, tree, path, ids[path]) for path in (_UV_LOCK, _PACKAGE_LOCK)),
        )
    except tree_objects.TreeReadError as exc:
        raise ToolchainUnavailable(f'cannot read the lockfiles of {tree}: {exc}') from exc
    cfg_path = worktree / _PYVENV_CFG
    cfg_text = await asyncio.to_thread(_text_if_present, cfg_path)
    if cfg_text is None and ids[_UV_LOCK] is not None:
        raise ToolchainUnavailable(f'{tree} locks a Python environment but {cfg_path} does not exist')
    return ToolchainFingerprint(
        role=role,
        interpreter=None if cfg_text is None else interpreter_from_pyvenv_cfg(cfg_text),
        uv_lock_blob=ids[_UV_LOCK],
        tool_versions=tool_versions_from_lockfiles(uv_lock_text, package_lock_text),
        verify_env=tuple(verify_env.items()),
        preprovision_command=preprovision_command,
        orchestrator_code=orchestrator_code,
    )


async def _present_blob_text(repo: Path, tree: str, path: str, oid: str | None) -> str | None:
    return None if oid is None else await tree_objects.blob_text(repo, tree, path)


def _text_if_present(path: Path) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ToolchainUnavailable(f'cannot read {path}: {exc}') from exc


def _uv_lock_packages(text: str) -> list[tuple[Any, Any]]:
    try:
        packages = tomllib.loads(text).get('package', [])
    except tomllib.TOMLDecodeError as exc:
        raise ToolchainUnavailable(f'{_UV_LOCK} is not valid TOML: {exc}') from exc
    if not isinstance(packages, list):
        raise ToolchainUnavailable(f'{_UV_LOCK} [[package]] is not an array of tables')
    return [(package.get('name'), package.get('version')) for package in packages if isinstance(package, dict)]


def _package_lock_packages(text: str) -> list[tuple[Any, Any]]:
    try:
        packages = json.loads(text).get('packages', {})
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ToolchainUnavailable(f'{_PACKAGE_LOCK} is not a JSON object: {exc}') from exc
    if not isinstance(packages, dict):
        raise ToolchainUnavailable(f'{_PACKAGE_LOCK} packages is not an object')
    return [
        (path.removeprefix('node_modules/'), entry.get('version'))
        for path, entry in packages.items()
        if path.startswith('node_modules/') and isinstance(entry, dict)
    ]


def _locked_versions(source: str, packages: Iterable[tuple[Any, Any]]) -> list[tuple[str, str, str]]:
    locked = list(packages)
    triples = []
    for tool in FINGERPRINTED_TOOLS:
        versions = sorted({str(version) for name, version in locked if name == tool})
        if versions:
            triples.append((tool, source, ','.join(versions)))
    return triples


def _require_unique(what: str, names: list[Any]) -> None:
    duplicates = sorted({str(name) for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'duplicate {what} in a toolchain fingerprint: {duplicates}')
