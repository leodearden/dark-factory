"""Discover Cargo workspace crates and map task files to crate names.

Used by ``verify.py::scope_module_config`` to rewrite ``cargo --workspace``
commands into ``cargo -p <crate>`` for task-phase verify, so Rust tasks that
touch a single crate don't pay the cost of verifying the whole workspace.

Scoping is applied only when *all* task files live under known crates; any
file outside the workspace crates (e.g., top-level ``scripts/``, root
``build.rs``) causes the whole command to fall through to ``--workspace``.

**Known limitation**: ``cargo test -p X`` does not run integration tests in
crates that depend on X.  A task that breaks a cross-crate contract will pass
task-phase verify and be caught by the workspace-wide post-merge verify, which
is the designed failure mode rather than a bug.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache discovered crates per workspace root so we don't re-parse Cargo.toml
# on every task.  Keyed on the resolved project_root path.
_WORKSPACE_CACHE: dict[Path, dict[str, str]] = {}


def discover_workspace_crates(project_root: Path) -> dict[str, str]:
    """Return ``{crate_dir_rel_to_root: crate_name}`` for *project_root*.

    Parses ``<project_root>/Cargo.toml`` for ``[workspace].members``,
    resolves each glob against the filesystem, and reads ``[package].name``
    from each member crate's own ``Cargo.toml``.

    Returns an empty dict when the root has no Cargo.toml, no ``[workspace]``
    section, or when parsing fails.  Results are cached per *project_root*;
    call :func:`_clear_cache` from tests to force rediscovery.
    """
    project_root = project_root.resolve()
    cached = _WORKSPACE_CACHE.get(project_root)
    if cached is not None:
        return cached

    root_toml = project_root / 'Cargo.toml'
    if not root_toml.exists():
        _WORKSPACE_CACHE[project_root] = {}
        return {}

    try:
        with open(root_toml, 'rb') as f:
            root = tomllib.load(f)
    except Exception as e:
        logger.debug('cargo_scope: failed to parse %s: %s', root_toml, e)
        _WORKSPACE_CACHE[project_root] = {}
        return {}

    members = root.get('workspace', {}).get('members', [])
    if not isinstance(members, list) or not members:
        _WORKSPACE_CACHE[project_root] = {}
        return {}

    crates: dict[str, str] = {}
    for member in members:
        if not isinstance(member, str):
            continue
        # Resolve glob-style member paths (e.g. "crates/*")
        for crate_dir in sorted(project_root.glob(member)):
            if not crate_dir.is_dir():
                continue
            crate_toml = crate_dir / 'Cargo.toml'
            if not crate_toml.exists():
                continue
            try:
                with open(crate_toml, 'rb') as f:
                    crate_meta = tomllib.load(f)
            except Exception as e:
                logger.debug('cargo_scope: failed to parse %s: %s', crate_toml, e)
                continue
            name = crate_meta.get('package', {}).get('name')
            if not isinstance(name, str) or not name:
                continue
            try:
                rel = crate_dir.relative_to(project_root)
            except ValueError:
                continue
            crates[str(rel)] = name

    _WORKSPACE_CACHE[project_root] = crates
    return crates


def files_to_crates(
    task_files: list[str], crates: dict[str, str],
) -> list[str] | None:
    """Map *task_files* to unique crate names via parent-path matching.

    Walks each file path up toward the workspace root until a parent matches
    a key in *crates*.  Collects unique crate names in deterministic order
    (first-seen).

    Returns ``None`` if ANY file is not under a known crate — signalling the
    caller to fall through to ``--workspace`` rather than risk under-scoping
    a task that also touches the workspace root.
    """
    if not crates:
        return None
    matched: list[str] = []
    seen: set[str] = set()
    for raw in task_files:
        path = Path(raw)
        # Walk parents toward the root; first match wins.
        found: str | None = None
        # Include the path itself if it is literally a crate dir.
        candidates = [path] + list(path.parents)
        for candidate in candidates:
            key = str(candidate)
            if key in crates:
                found = crates[key]
                break
        if found is None:
            return None
        if found not in seen:
            seen.add(found)
            matched.append(found)
    return matched


def _clear_cache() -> None:
    """Reset the workspace-crate cache.  For tests only."""
    _WORKSPACE_CACHE.clear()
