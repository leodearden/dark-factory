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
from functools import cached_property
from pathlib import Path

from fused_memory.models.scope import resolve_project_id_for_root

logger = logging.getLogger(__name__)


# Top-level directory names that appear in many projects and therefore do
# not uniquely identify any project.  Mirrors the intentional exclusions of
# the original task-1088 dark-factory-only guard (``shared/``, ``hooks/``
# etc.) and extends with the obvious build/runtime/test dirs so the registry
# stays focused on source-tree subprojects.
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
    # task 2434: 'memory' and 'deploy' recur as generic top-level dir names
    # across orchestrator-target projects (e.g. reify's memory/ is a 2-file
    # scratch/notes dir, deploy/ is 6 systemd unit files) and were over-
    # matching unrelated DF prose ("memory/edges", "deterministic deploy").
    # Dropping them here is symmetric across every project, so they can
    # never become a prefix via either the prose heuristic (find_paths) or
    # the certain files-check (project_for_path).
    # NOTE: 'gui/' is deliberately NOT added here — reify's gui/ is a real
    # 13.6k-file frontend, so denylisting it would lose genuine coverage;
    # the cost/benefit differs from these two scratch/config dirs.
    'memory', 'deploy',
})


# ---------------------------------------------------------------------------
# Built-in dark_factory constants — folded in from the retired dark-factory-
# only back-compat shim (task 2208 / PRD D2).
# ---------------------------------------------------------------------------

DARK_FACTORY_PROJECT_ID: str = 'dark_factory'

# Assumes the canonical dark_factory checkout path. Two consumers:
#   - `suggested_root` on scope_violation escalations (surfaced to an
#     operator) — informational.
#   - LOAD-BEARING since task 3109: under the `default()` registry this is
#     the only known project root, so it decides ownership for ABSOLUTE
#     `metadata.files` paths (`project_for_path`) and therefore whether the
#     path-scope guard rejects them.
# A deployment checked out elsewhere degrades fail-OPEN: no root matches, the
# path classifies as unowned, and the verdict is exactly what it was before
# task 3109 — a missed rejection, never a false one (same direction the
# `suggested_root` staleness already accepted). If it needs to be exact,
# override by constructing a custom `ProjectPrefixRegistry` (e.g. via
# `from_roots`, which resolves real roots) instead of `default()`, or update
# this constant.
DARK_FACTORY_ROOT: str = '/home/leo/src/dark-factory'

DARK_FACTORY_PATH_PREFIXES: tuple[str, ...] = (
    'orchestrator/',
    'fused-memory/',
    'fused_memory/',
    'mem0/',
    'graphiti/',
    'dashboard/',
)


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
            project_id = resolve_project_id_for_root(str(root_path))
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

    @classmethod
    def default(cls) -> ProjectPrefixRegistry:
        """Return the built-in, filesystem-independent dark_factory registry.

        Folds in the retired back-compat shim's constants (task 2208 / PRD
        D2) so the guard always classifies dark-factory paths even when no
        ``known_project_roots`` are configured — this is the registry the
        task interceptor falls back to when constructed without an explicit
        ``prefix_registry``, making the multi-project guard path
        (:func:`check_text_for_scope` / ``check_files_for_scope``) the only
        path, single-project deployments included.
        """
        return cls(
            project_to_root={DARK_FACTORY_PROJECT_ID: DARK_FACTORY_ROOT},
            project_to_prefixes={DARK_FACTORY_PROJECT_ID: DARK_FACTORY_PATH_PREFIXES},
            prefix_to_project=dict.fromkeys(DARK_FACTORY_PATH_PREFIXES, DARK_FACTORY_PROJECT_ID),
        )

    def all_prefixes(self) -> tuple[str, ...]:
        """Return all registered prefixes (sorted, deduplicated).

        The guard's regex builder caches by the tuple, so a stable order
        keeps the cache hot across calls.
        """
        return tuple(sorted(self.prefix_to_project))

    def project_for_prefix(self, prefix: str) -> str | None:
        """Return the owning project_id for *prefix*, or None if unknown."""
        return self.prefix_to_project.get(prefix)

    def project_for_path(self, file_path: str | None) -> str | None:
        """Return the owning project_id for *file_path*, or None if unowned.

        Unlike :meth:`project_for_prefix` (exact prefix-string lookup) or the
        guard's regex-over-prose ``find_paths``, this resolves an arbitrary
        file path to an owner. There are TWO disjoint regimes, selected by
        whether the (normalised) path is absolute:

        ABSOLUTE (task 3109) — resolved against the registry's known project
        ROOTS (:attr:`project_to_root`), not against the prefix table.  The
        candidate is first normalised LEXICALLY (``os.path.normpath``: ``..``
        segments resolved, duplicate separators collapsed — still no
        filesystem access), then a root ``R`` owns *file_path* when
        ``file_path == R`` or ``file_path.startswith(R + '/')`` (component
        boundary, so ``'/home/leo/src/dark-factory-old/x'`` does NOT match the
        root ``'/home/leo/src/dark-factory'``); the LONGEST matching root wins
        when roots nest. This is ROOT-AUTHORITATIVE: an absolute path under a known
        root is owned by that project even when its remainder matches NO
        registered prefix (e.g. ``<root>/docs/x.md``, whose ``docs/`` is in
        ``_GENERIC_DIRS``, or a bare ``<root>/README.md``) — an absolute path
        under a project root is unambiguous evidence of ownership, whereas the
        denylist and prefix table exist only to disambiguate BARE RELATIVE
        prefixes across projects (tasks 1494/2434). Returns ``None`` when the
        path lies under no known root.

        A leading ``'~/'`` (or a bare ``'~'``) is expanded via ``$HOME`` and,
        when that yields an absolute path, handled by the ABSOLUTE regime —
        agents that have been shelling around in a foreign repo emit that
        spelling too.  ``'~user/...'`` is deliberately NOT expanded (it would
        need an NSS/passwd lookup, breaking the no-I/O property) and stays
        fail-open.

        RELATIVE — exact leading-path-*component* matching against registered
        prefixes: a prefix ``P`` (always trailing-slash-terminated) owns
        *file_path* when ``file_path == P.rstrip('/')`` (the bare directory
        name, no trailing content) or ``file_path.startswith(P)`` — the
        trailing ``/`` on ``P`` enforces the component boundary, so e.g.
        ``'cratesfoo/x'`` does NOT match the prefix ``'crates/'``, and
        ``'vendor/crates/x'`` does not either (``crates/`` is not a *leading*
        component there). When more than one registered prefix matches, the
        LONGEST one wins.

        A leading ``'./'`` is stripped, then — task 4156 — the candidate is
        normalised LEXICALLY with ``os.path.normpath``, by the SAME call and
        for the SAME reason the ABSOLUTE regime uses (see
        :meth:`_owner_for_absolute_path`): the scan below compares just as
        lexically, so an unnormalised spelling defeats it in both directions.
        ``'orchestrator/../crates/foo.rs'`` would match the prefix
        ``'orchestrator/'`` by bare ``startswith`` and produce a FALSE
        ownership of a file that is not under ``orchestrator/`` at all, while
        ``'foo/../orchestrator/x.py'`` would match nothing and silently disarm
        the guard.  Still NO filesystem access — ``normpath`` is a string
        operation, so ownership neither resolves symlinks nor depends on the
        file existing yet, and deliberately NOT ``abspath``/``realpath``/
        ``Path.resolve()``, which would splice in the process CWD and make the
        verdict depend on where the interceptor happens to be running.

        A candidate that normalises to a leading ``'..'`` (or to ``'.'``) is
        UNOWNED by design: this regime is given no base directory to resolve it
        against, so ``'../other-repo/x.py'`` is genuinely unclassifiable. That
        fails OPEN — a missed rejection, never a false one — matching the
        residue documented at :meth:`_owner_for_absolute_path`. Returns
        ``None`` for an empty/falsy *file_path*.

        This is the CERTAIN/structured counterpart to the heuristic prose
        scanners in :mod:`fused_memory.middleware.path_scope_guard` — used by
        ``check_files_for_scope`` to classify concrete ``metadata.files``
        entries (task 2206) and by ``all_files_foreign_owner`` for the
        cross-repo tagger (task 3004). Both spellings of the same logical file
        now classify identically (task 3109).
        """
        path = (file_path or '').strip()
        if path.startswith('./'):
            path = path[2:]
        if not path:
            return None

        if path == '~' or path.startswith('~/'):
            # Env-only expansion (no NSS/passwd lookup, so no I/O); falls
            # through unchanged when $HOME is unset or not absolute.
            expanded = os.path.expanduser(path)
            if expanded.startswith('/'):
                path = expanded

        if path.startswith('/'):
            # ABSOLUTE regime (task 3109): resolved against known project
            # ROOTS, never against the relative prefix table below. The two
            # regimes are disjoint by construction — a registered prefix is
            # always a top-level dir name, so it can never lead an absolute
            # path, and roots are always absolute, so they can never match a
            # relative candidate. That disjointness is what made ADDING this
            # early return in 3109 a pure short-circuit, leaving the relative
            # verdicts of the day untouched — a fact about that change, not a
            # standing claim about the scan below, which has since grown its
            # own normalisation (task 4156). This is a regime selector.
            return self._owner_for_absolute_path(path)

        path = os.path.normpath(path)

        best_prefix = ''
        best_owner: str | None = None
        for prefix, owner in self.prefix_to_project.items():
            matches = path == prefix.rstrip('/') or path.startswith(prefix)
            if matches and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_owner = owner
        return best_owner

    @cached_property
    def _roots_longest_first(self) -> tuple[tuple[str, str], ...]:
        """``(root_norm, project_id)`` pairs, longest root first.

        Precomputed once per registry rather than per lookup: the dataclass is
        frozen and consumers treat :attr:`project_to_root` as fixed at build
        time, so the trailing-slash strip, the degenerate-root filter and the
        longest-wins ordering are all loop-invariant on the hot submit path.
        Sorting descending by length lets :meth:`_owner_for_absolute_path`
        return the FIRST match — the longest-root-wins tiebreak, now expressed
        in the ordering instead of re-derived on every call.  The sort is
        stable, so equal-length roots keep insertion order.

        Roots are normalised with ``os.path.normpath`` for the same reason the
        candidate is (see :meth:`_owner_for_absolute_path`); roots built by
        ``from_roots`` are already clean, but a hand-built or config-sourced
        registry need not be.

        A surviving root must be a non-degenerate ABSOLUTE path.  ``''`` would
        turn ``startswith('' + '/')`` into a universal match; ``'/'`` (and
        anything normalising to it, e.g. ``'/a/..'``) rstrips back to ``''``;
        and a RELATIVE root can never match an absolute candidate anyway, so
        dropping it here keeps the scan honest rather than merely inert.  That
        last check also catches ``os.path.normpath('') == '.'``, which is
        truthy and would otherwise survive the emptiness filter.
        """
        pairs: list[tuple[str, str]] = []
        for project_id, root in self.project_to_root.items():
            root_norm = os.path.normpath(root).rstrip('/') if root else ''
            if not root_norm or not root_norm.startswith('/'):
                continue
            pairs.append((root_norm, project_id))
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return tuple(pairs)

    def _owner_for_absolute_path(self, path: str) -> str | None:
        """Return the project whose known root contains absolute *path*.

        Root-authoritative (task 3109): an absolute path under a registered
        ``project_root`` is owned by that project DIRECTLY, without requiring
        its remainder to also match a registered prefix. The ``_GENERIC_DIRS``
        denylist and the leading-component prefix match exist to disambiguate
        BARE RELATIVE prefixes between projects (tasks 1494/2434); an absolute
        path under a project root carries no such ambiguity.

        The candidate is normalised with ``os.path.normpath`` first, which
        collapses duplicate separators and resolves ``..`` segments PURELY
        LEXICALLY.  Both spellings otherwise defeat the comparison in opposite
        and equally wrong directions: ``'<root>/../sibling/x'`` would match
        *root* by ``startswith`` and produce a FALSE rejection of a file that
        is not in *root* at all, while ``'/home/leo/src//dark-factory/x'``
        would match nothing and silently disarm the guard.

        Still NO filesystem access: ``normpath`` is a string operation — it
        neither resolves symlinks nor requires the file to exist, so the
        pure-lookup property this hot submit path depends on is preserved.
        Deliberately NOT ``realpath``/``Path.resolve()`` on the candidate:
        that would make ownership depend on symlink state and on whether the
        file exists yet (a task may legitimately declare files it is about to
        create).  ``from_roots`` already resolved the ROOTS at build time,
        which is the side that needs canonicalising.

        The ``root_norm + '/'`` boundary is the analogue of the relative
        scan's trailing-slash component boundary, so a sibling root like
        ``/home/leo/src/dark-factory-old`` does NOT match the root
        ``/home/leo/src/dark-factory``.

        When roots NEST (a project root and a worktree root beneath it can
        both be configured), the LONGEST matching root wins, so the most
        specific root is authoritative deterministically, independent of dict
        insertion order — mirroring :meth:`project_for_path`'s
        longest-prefix-wins tiebreak at the other granularity.  Here the
        tiebreak lives in :attr:`_roots_longest_first`'s ordering.

        Returns ``None`` when *path* lies under no known root — genuinely
        unclassifiable, which fails OPEN (unowned = the pre-3109 verdict,
        never a false rejection).

        KNOWN FAIL-OPEN residue, deliberately not closed (each is pinned by a
        test so the gap stays visible rather than implied covered):
        ``'../foo/x.py'`` and other parent-relative spellings, which cannot be
        resolved without a base directory this function is not given;
        ``'~user/...'``, which would need an NSS lookup; and a leading ``'//'``
        which POSIX reserves and ``normpath`` therefore preserves.  All three
        classify as unowned — a missed rejection, never a false one.  The one
        residual FALSE-rejection shape is a path that crosses a symlink out of
        a known root (``'<root>/link-to-elsewhere/x.py'``); detecting it would
        require the I/O this function deliberately forgoes.
        """
        path = os.path.normpath(path)
        for root_norm, project_id in self._roots_longest_first:
            if path == root_norm or path.startswith(root_norm + '/'):
                return project_id
        return None

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
