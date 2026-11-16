"""The environment a script's git subprocess runs with, minus ambient redirection."""

from __future__ import annotations

from collections.abc import Mapping

# The git env vars that retarget git away from the path it is given (incident
# 2026-08-31). Deliberately NARROW: only names that change WHICH repository (or
# which config) a command acts on.
#
# WHY CALLERS ARE EXPOSED. A caller targets its repository via ``cwd=`` or
# ``git -C``, and both only change DIRECTORY, while ``GIT_DIR`` and its siblings
# SKIP repository discovery outright. So a single ambient ``GIT_DIR`` silently
# redirects EVERY such call into whatever repository it names, with the
# caller's own path inert. Measured on the pre-guard reclaim script
# (scripts/reclaim_orphaned_worktrees.py): under ``GIT_DIR=<decoy>/.git`` with
# ``--repo <sandbox>``, ``git worktree list`` enumerated the DECOY's worktrees,
# the parking-root band guard passed them, and ``worktree remove --force``
# DESTROYED a decoy-owned worktree — while the sandbox named by ``--repo`` was
# never swept at all.
#
# ``GIT_CEILING_DIRECTORIES`` cannot catch this and is deliberately PRESERVED: a
# ceiling bounds the upward WALK of repository discovery, and an explicit
# ``GIT_DIR`` never walks, so the ceiling is bypassed rather than weakened —
# and it is the mechanism of a DIFFERENT defence
# (``df_pytest_isolation._df_git_ceiling_at_basetemp``, incident esc-3072-3),
# which scrubbing it here would disarm whenever a caller runs under the suite.
# The identity vars (``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*``) are preserved for
# the same narrowness: they change what a commit SAYS, never where it lands.
#
# DUPLICATED, NOT IMPORTED, from ``df_pytest_isolation._GIT_REDIRECT_ENV``:
# this module is stdlib-only because its callers' systemd wrappers run plain
# ``python3`` with no ``uv``/service env, and that module imports pytest.
# ``scripts/memory-metadata-coverage-census.sh`` carries a bash copy for the
# same reason. This copy is pinned against drift by
# ``scripts/tests/test_git_env.py::test_scrubbed_git_env_matches_shared_redirect_classifier``.
GIT_REDIRECT_ENV = (
    'GIT_DIR',
    'GIT_WORK_TREE',
    'GIT_INDEX_FILE',
    'GIT_COMMON_DIR',
    'GIT_OBJECT_DIRECTORY',
    'GIT_ALTERNATE_OBJECT_DIRECTORIES',
    'GIT_NAMESPACE',
    'GIT_CONFIG_GLOBAL',
    'GIT_CONFIG_SYSTEM',
    'GIT_CONFIG_COUNT',
    'GIT_CONFIG_PARAMETERS',
)

# ``GIT_CONFIG_COUNT``'s pairs are an INDEXED family (GIT_CONFIG_KEY_0/VALUE_0,
# ...), so they cannot be enumerated by name. (``git -c`` itself propagates
# through ``GIT_CONFIG_PARAMETERS`` above, not through this family.)
GIT_REDIRECT_ENV_PREFIXES = ('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')


def scrubbed_git_env(environ: Mapping[str, str]) -> dict[str, str]:
    """A copy of *environ* with git-redirecting and config-injecting names
    removed, ``LC_ALL=C``.

    Pure: *environ* (in production ``os.environ``) is never mutated. Drops every
    name in :data:`GIT_REDIRECT_ENV` and every name carrying a
    :data:`GIT_REDIRECT_ENV_PREFIXES` prefix; keeps everything else verbatim
    (the unit's ``PATH``/``HOME``, the ceiling, the identity vars) and forces
    ``LC_ALL=C`` so porcelain output stays locale-stable.

    A targeted removal, NOT a hermetic env rebuild — see the block comment above
    for which names retarget git and why the two preserved families do not.
    """
    scrubbed = {
        key: value
        for key, value in environ.items()
        if key not in GIT_REDIRECT_ENV
        and not key.startswith(GIT_REDIRECT_ENV_PREFIXES)
    }
    scrubbed['LC_ALL'] = 'C'
    return scrubbed
