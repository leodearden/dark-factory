"""Scope model — maps a request to backend-specific identifiers."""

import logging
import os
import subprocess
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Env var holding the comma-separated list of project_roots the system
# should treat as "known".  Reuses the dashboard's existing variable name
# so a single export propagates to both consumers — see
# ``dashboard/src/dashboard/config.py:88`` for the dashboard side.  A
# tracked followup is to promote this to a neutrally-named env var with a
# deprecation period.
KNOWN_PROJECT_ROOTS_ENV: str = 'DASHBOARD_KNOWN_PROJECT_ROOTS'

# Cache of input-path -> main-working-tree path, keyed by the absolute,
# resolved input path. Populated lazily by resolve_main_checkout so repeated
# calls with the same argument avoid spawning `git`.
_MAIN_CHECKOUT_CACHE: dict[str, str] = {}


def resolve_main_checkout(path: str | Path) -> str:
    """Return absolute string path of the main working tree that contains ``path``.

    A project's main git checkout is the single source of truth for
    ``tasks.json`` — worktrees must never read or write their own copy.
    This helper maps any path inside a git working tree (main or worktree,
    any subdirectory) to the absolute path of the *main* working tree.

    Uses ``git -C <path> worktree list --porcelain``. Git guarantees the
    first ``worktree <path>`` entry in the porcelain output is the main
    working tree, regardless of which working tree the command was run
    from. The input path is sanity-checked to confirm it is a descendant
    of some listed worktree.

    Results are cached by resolved absolute input path; clear
    ``_MAIN_CHECKOUT_CACHE`` directly in tests that mutate working trees.

    Raises:
        ValueError: if the input isn't inside any git working tree, or if
            ``git`` is unavailable on the host.
    """
    abs_input = str(Path(path).resolve())

    cached = _MAIN_CHECKOUT_CACHE.get(abs_input)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            ['git', '-C', abs_input, 'worktree', 'list', '--porcelain'],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ValueError(
            f'git executable not found; cannot resolve main checkout for {abs_input!r}'
        ) from e

    if result.returncode != 0:
        raise ValueError(
            f'{abs_input!r} is not inside a git working tree '
            f'(git returned {result.returncode}: {result.stderr.strip()})'
        )

    worktree_paths: list[str] = []
    main_path: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith('worktree '):
            wt = line[len('worktree '):].strip()
            if main_path is None:
                main_path = wt
            worktree_paths.append(wt)

    if main_path is None:
        raise ValueError(
            f'git worktree list returned no entries for {abs_input!r}'
        )

    abs_input_path = Path(abs_input)
    descendant_of_any = any(
        _is_descendant(abs_input_path, Path(wt)) for wt in worktree_paths
    )
    if not descendant_of_any:
        raise ValueError(
            f'{abs_input!r} is not a descendant of any listed worktree '
            f'for repo at {main_path!r}'
        )

    resolved_main = str(Path(main_path).resolve())
    _MAIN_CHECKOUT_CACHE[abs_input] = resolved_main
    return resolved_main


def _is_descendant(child: Path, parent: Path) -> bool:
    """True if ``child`` is ``parent`` or a path inside ``parent`` (resolved)."""
    try:
        child_r = child.resolve()
        parent_r = parent.resolve()
    except OSError:
        return False
    if child_r == parent_r:
        return True
    try:
        child_r.relative_to(parent_r)
        return True
    except ValueError:
        return False


def resolve_project_id(
    project_root: str, mapping: dict[str, str] | None = None
) -> str:
    """Derive a logical project_id from a filesystem project_root path.

    Priority:
      1. Explicit mapping dict (if provided and contains project_root).
      2. Derive from basename: lowercase, hyphens -> underscores.

    Examples:
        >>> resolve_project_id('/home/leo/src/dark-factory')
        'dark_factory'
        >>> resolve_project_id('/project/')
        'project'
    """
    if mapping and project_root in mapping:
        return mapping[project_root]
    name = PurePosixPath(project_root.rstrip('/')).name
    return name.lower().replace('-', '_')


def known_project_roots_from_env(
    env_var: str = KNOWN_PROJECT_ROOTS_ENV,
) -> list[str]:
    """Parse the configured comma-separated env var into a list of project_roots.

    Returns ``[]`` when the env var is unset or empty.  Whitespace around
    each entry is stripped; empty entries (e.g. trailing comma) are
    skipped.  Does *not* resolve symlinks or check existence — call sites
    decide whether stat-ing each root is appropriate.
    """
    raw = os.environ.get(env_var, '')
    if not raw:
        return []
    return [p.strip() for p in raw.split(',') if p.strip()]


def build_known_projects_map(
    primary_root: str,
    extra_roots: list[str] | None = None,
) -> dict[str, str]:
    """Build ``{project_id → project_root}`` from a primary root + extras.

    *primary_root* is included first (so its ``project_id`` always wins on
    duplicate-id collisions).  *extra_roots* defaults to the parsed
    :func:`known_project_roots_from_env` output so callers can rely on the
    env var by default but override in tests.

    Empty/missing primary roots are silently skipped.  Roots that can't be
    resolved (file not found, permission denied) are logged and skipped —
    this helper must not raise during process startup.
    """
    if extra_roots is None:
        extra_roots = known_project_roots_from_env()

    out: dict[str, str] = {}
    candidates: list[str] = []
    if primary_root:
        candidates.append(primary_root)
    candidates.extend(extra_roots)

    for raw in candidates:
        try:
            resolved = str(Path(raw).resolve())
        except OSError as exc:
            logger.warning(
                'build_known_projects_map: cannot resolve %r: %s', raw, exc,
            )
            continue
        pid = resolve_project_id(resolved)
        if pid in out:
            # Duplicate project_id — primary wins (since it was added
            # first).  Extras with a clashing name are dropped silently
            # except for an INFO log so operators can see why they didn't
            # take effect.
            logger.info(
                'build_known_projects_map: dropping duplicate project_id %r '
                '(root %s); first-wins is %s',
                pid, resolved, out[pid],
            )
            continue
        out[pid] = resolved
    return out


class Scope(BaseModel):
    """Per-request scope mapping project/agent/session to backend IDs."""

    project_id: str
    agent_id: str | None = None
    session_id: str | None = None

    @property
    def graphiti_group_id(self) -> str:
        """Graphiti group_id = project_id."""
        return self.project_id

    def mem0_collection_name(self, prefix: str) -> str:
        """Qdrant collection name: {prefix}_{project_id}."""
        return f'{prefix}_{self.project_id}'

    @property
    def mem0_user_id(self) -> str:
        """Mem0 requires at least one of user_id/agent_id/run_id.

        We use project_id as user_id to satisfy this requirement.
        """
        return self.project_id
