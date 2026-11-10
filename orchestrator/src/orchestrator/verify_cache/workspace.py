"""The uv workspace's member dependency graph, as a dispatched tree declares it."""

from __future__ import annotations

import asyncio
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from orchestrator.verify_cache import tree_objects

_PEP_508_NAME = re.compile(r'^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)')
_NAME_SEPARATORS = re.compile(r'[-_.]+')
_GLOB_CHARS = frozenset('*?[')
_DEV_GROUP = 'dev'


@dataclass(frozen=True)
class WorkspacePatterns:
    """The root pyproject's ``[tool.uv.workspace]`` member and exclude patterns."""

    members: tuple[str, ...]
    exclude: tuple[str, ...]

    @classmethod
    def from_root(cls, root_text: str) -> WorkspacePatterns:
        table = tomllib.loads(root_text).get('tool', {}).get('uv', {}).get('workspace')
        if table is None:
            return cls((), ())
        return cls(_string_list(table, 'members'), _string_list(table, 'exclude'))

    @property
    def needs_directories(self) -> bool:
        return any(_is_glob(pattern) for pattern in (*self.members, *self.exclude))

    def expand(self, directories: Iterable[str]) -> tuple[str, ...]:
        """The member dirs, with glob patterns matched segment-wise against *directories*."""
        listed = tuple(directories)
        excluded = set(_expand(self.exclude, listed))
        return tuple(sorted(set(_expand(self.members, listed)) - excluded))


@dataclass(frozen=True)
class WorkspaceGraph:
    """Each member dir's normalized project name, and the member dirs it depends on."""

    names: Mapping[str, str]
    dependencies: Mapping[str, frozenset[str]]

    @classmethod
    def from_manifests(
        cls,
        root_text: str,
        member_texts: Mapping[str, str],
        directories: Iterable[str] = (),
    ) -> WorkspaceGraph:
        """The graph *root_text* declares over *member_texts* (member dir -> its
        pyproject.toml text). A listed member with no text is a ValueError naming it."""
        manifests: dict[str, dict[str, Any]] = {}
        for member in WorkspacePatterns.from_root(root_text).expand(directories):
            if member not in member_texts:
                raise ValueError(f'workspace member {member!r} has no pyproject.toml')
            try:
                manifests[member] = tomllib.loads(member_texts[member])
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f'workspace member {member!r} has an unparseable pyproject.toml: {exc}') from exc
        names = {member: _project_name(member, manifest) for member, manifest in manifests.items()}
        member_by_name = {name: member for member, name in names.items()}
        dependencies = {
            member: frozenset(
                member_by_name[name]
                for name in _requirement_names(member, manifest)
                if name in member_by_name
            )
            for member, manifest in manifests.items()
        }
        return cls(MappingProxyType(names), MappingProxyType(dependencies))

    @property
    def members(self) -> frozenset[str]:
        return frozenset(self.names)

    def closure(self, dirs: Iterable[str]) -> frozenset[str]:
        """The members among *dirs* plus every member they depend on, transitively."""
        pending = [member for member in dirs if member in self.names]
        reached = set(pending)
        while pending:
            for dependency in self.dependencies[pending.pop()]:
                if dependency not in reached:
                    reached.add(dependency)
                    pending.append(dependency)
        return frozenset(reached)

    def members_containing(self, paths: Iterable[str]) -> frozenset[str]:
        """The members holding any of *paths* (a path equal to a member dir counts)."""
        wanted = tuple(paths)
        return frozenset(
            member for member in self.names
            if any(path == member or path.startswith(f'{member}/') for path in wanted)
        )


async def read_workspace(repo: Path, tree: str) -> WorkspaceGraph:
    """The workspace graph *tree* declares; empty when it has no root pyproject.toml.

    Manifests that do not form a workspace (unparseable TOML, a listed member
    without a pyproject.toml, a member without a project name) raise
    TreeReadError here, so a key derivation reports the tree as unavailable.
    """
    root_text = await tree_objects.blob_text(repo, tree, 'pyproject.toml')
    if root_text is None:
        return WorkspaceGraph(MappingProxyType({}), MappingProxyType({}))
    try:
        patterns = WorkspacePatterns.from_root(root_text)
        directories = await tree_objects.directories(repo, tree) if patterns.needs_directories else ()
        members = patterns.expand(directories)
        texts = await asyncio.gather(
            *(tree_objects.blob_text(repo, tree, f'{member}/pyproject.toml') for member in members),
        )
        member_texts = {member: text for member, text in zip(members, texts, strict=True) if text is not None}
        return WorkspaceGraph.from_manifests(root_text, member_texts, directories)
    except ValueError as exc:
        raise tree_objects.TreeReadError(f'{tree} declares no readable uv workspace: {exc}') from exc


def _string_list(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = table.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f'[tool.uv.workspace].{key} is not a list of strings: {values!r}')
    return tuple(values)


def _is_glob(pattern: str) -> bool:
    return not _GLOB_CHARS.isdisjoint(pattern)


def _expand(patterns: Iterable[str], directories: tuple[str, ...]) -> Iterable[str]:
    for pattern in patterns:
        normalized = PurePosixPath(pattern).as_posix()
        if not _is_glob(normalized):
            yield normalized
            continue
        pattern_parts = PurePosixPath(normalized).parts
        for directory in directories:
            parts = PurePosixPath(directory).parts
            if len(parts) == len(pattern_parts) and all(
                fnmatchcase(part, wanted) for part, wanted in zip(parts, pattern_parts, strict=True)
            ):
                yield directory


def _project_name(member: str, manifest: Mapping[str, Any]) -> str:
    name = manifest.get('project', {}).get('name')
    if not isinstance(name, str):
        raise ValueError(f'workspace member {member!r} declares no [project].name')
    return _normalize(name)


def _requirement_names(member: str, manifest: Mapping[str, Any]) -> frozenset[str]:
    requirements = [
        *manifest.get('project', {}).get('dependencies', []),
        *_group_requirements(manifest.get('dependency-groups', {}), _DEV_GROUP, frozenset()),
    ]
    return frozenset(_requirement_name(member, requirement) for requirement in requirements)


def _group_requirements(
    groups: Mapping[str, Any], group: str, visiting: frozenset[str],
) -> list[str]:
    """*group*'s requirement strings, PEP 735 include-group tables resolved within the member."""
    if group in visiting:
        return []
    requirements: list[str] = []
    for entry in groups.get(group, []):
        if isinstance(entry, dict):
            included = entry.get('include-group')
            if not isinstance(included, str):
                raise ValueError(f'dependency group {group!r} has an entry that is no include-group: {entry!r}')
            requirements += _group_requirements(groups, included, visiting | {group})
        else:
            requirements.append(entry)
    return requirements


def _requirement_name(member: str, requirement: Any) -> str:
    match = _PEP_508_NAME.match(requirement) if isinstance(requirement, str) else None
    if match is None:
        raise ValueError(f'workspace member {member!r} declares an unparseable requirement {requirement!r}')
    return _normalize(match.group(1))


def _normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return _NAME_SEPARATORS.sub('-', name).lower()
