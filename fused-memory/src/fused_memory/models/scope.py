"""Scope model — maps a request to backend-specific identifiers.

``ProjectId``/``ProjectRoot`` (below) are distinct ``str`` :class:`~typing.NewType`\\ s
and ``ProjectScope`` is a frozen dataclass pairing them, so a transposed
``project_id``/``project_root`` argument pair is a pyright error rather than a
runtime bug.

This module (task α of the recon-project-scope batch; see
``plans/recon-project-scope-prd.md``) defines the type only — as of this
change, ``ProjectScope`` is not yet constructed by any production call site.
It is *intended* to have exactly two sanctioned construction sites, to be
wired in by follow-up tasks in the same batch —
``ReconciliationHarness._known_project_root_for`` (``reconciliation/harness.py``,
the registry-backed cycle entry; task β) and
``TargetedReconciler.reconcile_task`` (``reconciliation/targeted.py``, after
its ``require_project_root`` call; task δ). No other call site should
construct a ``ProjectScope`` directly once those tasks land.
"""

import dataclasses
import logging
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import NewType

import yaml
from pydantic import BaseModel, field_validator

from fused_memory.utils.validation import (
    InputValidationError,
    PathShapedProjectIdError,
    _to_underscore_canonical,
    canonicalize_project_id,
    require_project_root,
)

logger = logging.getLogger(__name__)


# Env var holding the comma-separated list of project_roots the system
# should treat as "known".  Reuses the dashboard's existing variable name
# so a single export propagates to both consumers — see
# ``dashboard/src/dashboard/config.py:88`` for the dashboard side.  A
# tracked followup is to promote this to a neutrally-named env var with a
# deprecation period.
KNOWN_PROJECT_ROOTS_ENV: str = 'DASHBOARD_KNOWN_PROJECT_ROOTS'

# Canonical project-manifest filename every DF-targeted project exposes at
# its root (see CLAUDE.md "Repo Map").  ``read_declared_project_id`` reads
# only the ``project_id`` key from this one file — no legacy config-name
# fallbacks, no orchestrator-package import.
ORCHESTRATOR_CONFIG_NAME: str = 'dark-factory-orchestrator.yaml'

# Cache of input-path -> main-working-tree path, keyed by the absolute,
# resolved input path. Populated lazily by resolve_main_checkout so repeated
# calls with the same argument avoid spawning `git`.
_MAIN_CHECKOUT_CACHE: dict[str, str] = {}


def resolve_main_checkout(path: str | Path) -> str:
    """Return absolute string path of the main working tree that contains ``path``.

    Each project's tasks.db lives under the main git checkout; worktrees share
    that database via path normalisation at the MCP boundary. This helper maps
    any path inside a git working tree (main or worktree, any subdirectory) to
    the absolute path of the *main* working tree.

    Uses ``git -C <path> worktree list --porcelain``. Git guarantees the
    first ``worktree <path>`` entry in the porcelain output is the main
    working tree, regardless of which working tree the command was run from.
    The input path is sanity-checked to confirm it is a descendant of some
    listed worktree.

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

    The underscore-canonical mapping (lowercase + hyphen->underscore) is the
    same rule applied by :func:`fused_memory.utils.validation.canonicalize_project_id`
    -- both delegate to the shared ``_to_underscore_canonical`` helper so the
    rule has one source of truth. Unlike ``canonicalize_project_id``, this
    function never raises on path-shaped input: it operates on an
    already-extracted path basename (see below), not a raw caller-supplied
    project_id, so the LOUD-reject guard does not apply here.

    Examples:
        >>> resolve_project_id('/home/leo/src/dark-factory')
        'dark_factory'
        >>> resolve_project_id('/project/')
        'project'
    """
    if mapping and project_root in mapping:
        return mapping[project_root]
    name = PurePosixPath(project_root.rstrip('/')).name
    return _to_underscore_canonical(name)


def read_declared_project_id(project_root: str | Path) -> str | None:
    """Return the project_id declared in ``<project_root>`` 's manifest, or None.

    Reads only the top-level ``project_id`` key from
    ``<project_root>/dark-factory-orchestrator.yaml`` (the canonical
    project-manifest filename — no legacy fallbacks, no orchestrator-package
    import), normalizing a non-empty string value via the same
    ``_to_underscore_canonical`` rule :func:`resolve_project_id` uses so a
    declared id and a basename-derived id share one canonicalization source
    of truth.

    Strictly **fail-open**: this NEVER raises.  A missing manifest,
    unparseable YAML (including invalid non-UTF-8 bytes, whose
    ``UnicodeDecodeError`` is a ``ValueError`` subclass rather than an
    ``OSError``/``yaml.YAMLError``), a non-mapping document, a
    missing/non-string/empty ``project_id``, or a path-shaped declared value
    (one containing a path separator or a leading ``-``/``/`` — something
    :func:`resolve_project_id` could never produce from a basename) all
    return ``None`` so callers degrade to today's basename derivation —
    mirroring :func:`build_known_projects_map`'s
    must-not-raise-during-startup contract (both registries are built once at
    process start).  A parse error is debug-logged, not surfaced.

    This is what makes the id rename-stable: a project whose directory was
    renamed but whose manifest still declares the original id resolves to the
    declared id, not the new basename.
    """
    manifest = Path(project_root) / ORCHESTRATOR_CONFIG_NAME
    try:
        with open(manifest, encoding='utf-8') as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError (a ValueError subclass, not an OSError/
        # yaml.YAMLError) is raised by yaml.safe_load when the manifest holds
        # invalid UTF-8 bytes; catch it too so the never-raise contract holds.
        logger.debug(
            'read_declared_project_id: cannot read %s: %s', manifest, exc,
        )
        return None

    if not isinstance(data, dict):
        return None
    declared = data.get('project_id')
    if not isinstance(declared, str) or not declared:
        return None
    # The declared value is raw operator-supplied input, so canonicalize it
    # through canonicalize_project_id's path-shape guard (NOT the bare
    # _to_underscore_canonical resolve_project_id applies to an already-
    # extracted basename): a value with a path separator or leading '-'/'/'
    # is something resolve_project_id could never produce, so reject it and
    # fail open to basename derivation instead of registering a nonsensical
    # id.  canonicalize_project_id delegates to the same _to_underscore_canonical
    # rule for every non-path-shaped value, so declared and derived ids still
    # share one canonicalization source of truth.
    try:
        return canonicalize_project_id(declared)
    except PathShapedProjectIdError as exc:
        logger.debug(
            'read_declared_project_id: declared project_id %r in %s is '
            'path-shaped; ignoring (falling back to basename): %s',
            declared, manifest, exc,
        )
        return None


def resolve_project_id_for_root(project_root: str | Path) -> str:
    """Rename-stable project_id resolver: manifest-declared id first.

    Returns the id declared in ``<project_root>/dark-factory-orchestrator.yaml``
    (via :func:`read_declared_project_id`) when present, else falls back to
    the pure basename derivation of :func:`resolve_project_id`.  This is the
    resolver the two registry BUILDERS
    (:func:`build_known_projects_map` and
    :class:`~fused_memory.middleware.project_prefix_registry.ProjectPrefixRegistry.from_roots`)
    use so a directory rename that preserves the manifest's ``project_id``
    re-points them to the new path under the SAME id instead of churning.

    :func:`resolve_project_id` itself is intentionally left unchanged (still a
    pure, no-I/O basename derivation) so its many hot-path callers are
    unaffected.
    """
    return read_declared_project_id(project_root) or resolve_project_id(
        str(project_root)
    )


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
        pid = resolve_project_id_for_root(resolved)
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


ProjectId = NewType('ProjectId', str)
ProjectRoot = NewType('ProjectRoot', str)


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectScope:
    """Type-level pairing of a project's canonical id and filesystem root.

    ``project_id`` and ``project_root`` are distinct ``NewType``\\ s (both
    backed by ``str``), so passing one where the other is expected is a
    pyright ``reportArgumentType`` error, not a silent runtime mix-up.
    Instances are frozen — reassigning either field after construction raises
    ``dataclasses.FrozenInstanceError`` at runtime and is a pyright
    ``reportAttributeAccessIssue`` at type-check time.

    Not yet constructed by any production call site. See the module
    docstring for the two *intended* construction sites, to be wired in by
    this batch's follow-up tasks.
    """

    project_id: ProjectId
    project_root: ProjectRoot

    def __post_init__(self) -> None:
        # Deliberately a plain non-empty check, NOT require_project_id (which
        # additionally enforces a safe-identifier charset allowlist). Both
        # intended construction sites (see class docstring) build project_id
        # from resolve_project_id/canonicalize_project_id output or an
        # already registry-resolved value -- always underscore-canonical and
        # allowlist-safe by construction -- so no caller reaching this
        # __post_init__ needs the stronger charset check, and applying it
        # here would over-reject otherwise-legitimate ids. If a future
        # construction site can pass an uncanonicalized/raw id, prefer
        # canonicalizing at that call site over widening this check.
        if not self.project_id:
            raise InputValidationError(
                f'project_id must be a non-empty string, got: {self.project_id!r}'
            )
        require_project_root(self.project_root)


class Scope(BaseModel):
    """Per-request scope mapping project/agent/session to backend IDs."""

    project_id: str
    agent_id: str | None = None
    session_id: str | None = None

    @field_validator('project_id')
    @classmethod
    def _canonicalize_project_id(cls, v: str) -> str:
        """Canonicalize project_id at construction (mode='after' — v is a
        guaranteed str post type-coercion) so every field derived from it
        (graphiti_group_id, mem0_collection_name, mem0_user_id) is always
        canonical too. Path-shaped input raises PathShapedProjectIdError
        (a ValueError subclass), which pydantic surfaces as a
        ValidationError at Scope() construction."""
        return canonicalize_project_id(v)

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
