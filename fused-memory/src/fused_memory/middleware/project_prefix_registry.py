"""Multi-project path-prefix registry for the path-scope guard.

The dark-factory-only guard introduced in task 1088 caught dark-factory paths
landing in non-dark-factory projects but ignored the reverse direction (e.g.
reify-scoped work landing in dark-factory).  This registry generalises the
prefix list to a per-project map, built once at process start from the
configured ``known_project_roots``.

Each prefix appears in at most one project: prefixes that show up under
multiple known roots are dropped on both sides so the guard stays
deterministic and false-positive-free at the cost of a few false negatives
on shared directory names (which would have produced ambiguous routing
suggestions anyway).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from fused_memory.models.scope import resolve_project_id

logger = logging.getLogger(__name__)


# Top-level directory names that appear in many projects and therefore do
# not uniquely identify any project.  Mirrors the intentional exclusions in
# the original dark_factory_path_guard (``shared/``, ``hooks/`` etc.) and
# extends with the obvious build/runtime/test dirs so the registry stays
# focused on source-tree subprojects.
_GENERIC_DIRS: frozenset[str] = frozenset({
    'src', 'tests', 'test', 'docs', 'doc', 'scripts', 'bin', 'examples',
    'node_modules', 'target', '.venv', 'venv', '__pycache__',
    'data', 'logs', 'plans', 'review', 'hooks', 'skills', 'shared',
    'escalation', 'dist', 'build', 'config', 'tmp', 'cache', '.cache',
    '.git', '.github', '.pytest_cache', '.ruff_cache', '.mypy_cache',
    '.claude', '.taskmaster', '.worktrees',
    # repo-management / ops dirs commonly seen at top level:
    'fixtures', 'assets', 'public', 'static', 'vendor', 'third_party',
    'proto', 'protos', 'sdks', 'sdk',
})


def _candidate_prefixes_for_root(root: Path) -> list[str]:
    """Return prefix-form names of top-level source dirs under *root*.

    Filters out generic directories (`_GENERIC_DIRS`), hidden directories,
    and non-directories.  The returned strings already have the trailing
    slash so they can be concatenated directly into the guard's regex
    alternation.

    Hyphenated names get an underscore alias (mirrors the original guard
    listing both ``fused-memory/`` and ``fused_memory/``).
    """
    if not root.is_dir():
        logger.warning(
            'project_prefix_registry: project_root %s is not a directory; skipping',
            root,
        )
        return []

    out: list[str] = []
    seen: set[str] = set()
    try:
        entries = sorted(os.listdir(root))
    except OSError as exc:
        logger.warning(
            'project_prefix_registry: cannot list %s: %s', root, exc,
        )
        return []

    for entry in entries:
        if entry.startswith('.'):
            continue
        if entry in _GENERIC_DIRS:
            continue
        if not (root / entry).is_dir():
            continue
        prefix = f'{entry}/'
        if prefix not in seen:
            seen.add(prefix)
            out.append(prefix)
        if '-' in entry:
            alias = f'{entry.replace("-", "_")}/'
            if alias not in seen:
                seen.add(alias)
                out.append(alias)
    return out


@dataclass(frozen=True)
class ProjectPrefixRegistry:
    """Maps top-level path prefixes to project_ids.

    Build once at process start with :meth:`from_roots`; consumers (the
    path-scope guard) treat the registry as immutable.
    """

    project_to_root: dict[str, str] = field(default_factory=dict)
    project_to_prefixes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    prefix_to_project: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_roots(
        cls, roots: list[str] | list[Path] | tuple[str | Path, ...],
    ) -> ProjectPrefixRegistry:
        """Build a registry from a list of project_root paths.

        Drops any prefix that appears under more than one root — the path
        guard would not be able to suggest a single target project for a
        candidate citing such a prefix, so silence beats ambiguity.
        Also drops projects whose only prefixes were collisions.
        """
        project_to_root: dict[str, str] = {}
        per_project: dict[str, list[str]] = {}
        # Track (prefix → set of project_ids) so we can detect collisions.
        prefix_to_projects: dict[str, set[str]] = {}

        for raw in roots:
            root_path = Path(str(raw)).resolve()
            project_id = resolve_project_id(str(root_path))
            if project_id in project_to_root:
                logger.warning(
                    'project_prefix_registry: duplicate project_id %r '
                    '(roots %s and %s); keeping first',
                    project_id, project_to_root[project_id], root_path,
                )
                continue
            project_to_root[project_id] = str(root_path)
            prefixes = _candidate_prefixes_for_root(root_path)
            per_project[project_id] = prefixes
            for p in prefixes:
                prefix_to_projects.setdefault(p, set()).add(project_id)

        # Resolve collisions: drop any prefix that >1 project claims.
        colliding = {p for p, owners in prefix_to_projects.items() if len(owners) > 1}
        if colliding:
            logger.info(
                'project_prefix_registry: dropping %d colliding prefix(es): %s',
                len(colliding), sorted(colliding),
            )

        prefix_to_project: dict[str, str] = {}
        project_to_prefixes: dict[str, tuple[str, ...]] = {}
        for project_id, prefixes in per_project.items():
            kept = tuple(p for p in prefixes if p not in colliding)
            project_to_prefixes[project_id] = kept
            for p in kept:
                prefix_to_project[p] = project_id

        registry = cls(
            project_to_root=project_to_root,
            project_to_prefixes=project_to_prefixes,
            prefix_to_project=prefix_to_project,
        )
        logger.info(
            'project_prefix_registry: built with %d project(s); %d prefix(es) total',
            len(project_to_root), len(prefix_to_project),
        )
        for pid, prefixes in project_to_prefixes.items():
            logger.info(
                'project_prefix_registry:   %s → %s', pid, list(prefixes) or '<empty>',
            )
        return registry

    def all_prefixes(self) -> tuple[str, ...]:
        """Return all registered prefixes (sorted, deduplicated).

        The guard's regex builder caches by the tuple, so a stable order
        keeps the cache hot across calls.
        """
        return tuple(sorted(self.prefix_to_project))

    def project_for_prefix(self, prefix: str) -> str | None:
        """Return the owning project_id for *prefix*, or None if unknown."""
        return self.prefix_to_project.get(prefix)

    def root_for_project(self, project_id: str) -> str | None:
        """Return the configured project_root for *project_id*, or None."""
        return self.project_to_root.get(project_id)

    def is_known(self, project_id: str) -> bool:
        """Return True iff *project_id* was registered (regardless of prefixes)."""
        return project_id in self.project_to_root

    def __bool__(self) -> bool:
        """Empty registries (no roots configured) are falsy.

        Lets callers treat a zero-project registry the same as ``None``
        (back-compat: guard does not run when no registry is configured).
        """
        return bool(self.project_to_root)
