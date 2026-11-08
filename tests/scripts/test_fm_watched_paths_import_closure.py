"""FM_WATCHED_PATHS covers exactly the workspace members the fused-memory process loads.

scripts/orchestrator-watchdog.py::FM_WATCHED_PATHS decides what counts toward
fused-memory staleness. A change to any workspace member fm loads, directly or
through another member, can alter fm's behaviour, so that member's src/ must be
watched. This guard derives the set from the source instead of trusting a comment.

The derivation over-approximates on purpose: it works at member granularity and
counts every absolute import in a file, including function-local, lazy and
TYPE_CHECKING-only ones. That errs toward an extra restart, which the watchdog's
8h min-interval caps, and never toward a missed one.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import tomllib
from collections import deque
from collections.abc import Iterator
from types import ModuleType
from typing import NamedTuple

REPO_ROOT = pathlib.Path(__file__).parents[2]
ROOT_MEMBER = "fused-memory"


class _Hop(NamedTuple):
    importer: str
    imported: str
    example: pathlib.Path


def _load_watchdog() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "orchestrator_watchdog", REPO_ROOT / "scripts" / "orchestrator-watchdog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _member_by_package() -> dict[str, str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    members = pyproject["tool"]["uv"]["workspace"]["members"]
    return {
        init.parent.name: member
        for member in members
        for init in (REPO_ROOT / member / "src").glob("*/__init__.py")
    }


def _absolute_import_roots(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.partition(".")[0]


def _members_imported_by(
    member: str, member_by_package: dict[str, str]
) -> dict[str, pathlib.Path]:
    """Map each other member that `member`'s src/ imports to one example importing file."""
    imported: dict[str, pathlib.Path] = {}
    for path in sorted((REPO_ROOT / member / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for package in _absolute_import_roots(tree):
            target = member_by_package.get(package)
            if target is not None and target != member:
                imported.setdefault(target, path.relative_to(REPO_ROOT))
    return imported


def _import_closure(root: str, member_by_package: dict[str, str]) -> dict[str, list[_Hop]]:
    """Every member reachable from `root` by imports, with the chain that first reaches it."""
    chains: dict[str, list[_Hop]] = {root: []}
    queue = deque([root])
    while queue:
        member = queue.popleft()
        for target, example in _members_imported_by(member, member_by_package).items():
            if target not in chains:
                chains[target] = [*chains[member], _Hop(member, target, example)]
                queue.append(target)
    return chains


def _describe_chain(hops: list[_Hop]) -> str:
    return " -> ".join(
        [ROOT_MEMBER, *(f"{hop.imported} (imported by {hop.example})" for hop in hops)]
    )


def test_fm_watched_paths_cover_exactly_the_members_fused_memory_loads() -> None:
    chains = _import_closure(ROOT_MEMBER, _member_by_package())
    loaded = {f"{member}/src/": hops for member, hops in chains.items()}
    watched = {p for p in _load_watchdog().FM_WATCHED_PATHS if p.endswith("/src/")}

    problems = [
        f"missing {prefix}, loaded via: {_describe_chain(loaded[prefix])}"
        for prefix in sorted(loaded.keys() - watched)
    ] + [
        f"extra {prefix}: no import chain from {ROOT_MEMBER} reaches it"
        for prefix in sorted(watched - loaded.keys())
    ]
    assert not problems, (
        "FM_WATCHED_PATHS must list the src/ of exactly the workspace members "
        f"the {ROOT_MEMBER} process loads:\n" + "\n".join(problems)
    )
