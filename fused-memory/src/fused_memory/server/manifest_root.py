"""Which worktree ``commit_planning`` accepts as ``manifest_root``.

A PRD decomposed before it lands on main has its PRD family — the PRD and its
capability-manifest sidecar — only in the linked worktree that authored it.
``commit_planning(manifest_root=...)`` names that worktree so the sidecar is
stamped there. It is accepted only when it is the top-level directory of an
existing worktree of *project_root*'s repository, other than *project_root*'s
own checkout. Anything else is refused before ``commit_planning`` reads,
flips or writes anything, with one of four ``reason_code`` values:

* ``not_a_worktree`` — not an absolute path, not an existing directory, not
  inside a git worktree, or a subdirectory of one (the enclosing worktree's
  root is named, because sidecar paths are relative to the root).
* ``other_repository`` — a worktree whose git common directory is not
  *project_root*'s.
* ``main_checkout`` — *project_root*'s own checkout, under any spelling;
  omitting ``manifest_root`` already means "search the main checkout".
* ``probe_failed`` — git could not be run, or did not answer in time.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.git_async import run_git

_GIT_TIMEOUT_SECONDS = 10.0


class ManifestRootRefusal(StrEnum):
    NOT_A_WORKTREE = 'not_a_worktree'
    OTHER_REPOSITORY = 'other_repository'
    MAIN_CHECKOUT = 'main_checkout'
    PROBE_FAILED = 'probe_failed'


async def resolve_manifest_root(project_root: str, manifest_root: str) -> Path | dict[str, Any]:
    """The accepted worktree root, realpath'd, or the refusal ``commit_planning`` returns."""
    if not os.path.isabs(manifest_root):
        return _refusal(ManifestRootRefusal.NOT_A_WORKTREE, manifest_root, 'must be an absolute path')
    if not Path(manifest_root).is_dir():
        return _refusal(ManifestRootRefusal.NOT_A_WORKTREE, manifest_root, 'no such directory')

    worktree = await _rev_parse(
        manifest_root, manifest_root, '--show-toplevel', '--git-common-dir',
        not_ok=ManifestRootRefusal.NOT_A_WORKTREE,
    )
    if isinstance(worktree, dict):
        return worktree
    toplevel, common_dir = worktree
    real_root = os.path.realpath(manifest_root)
    if os.path.realpath(toplevel) != real_root:
        return _refusal(
            ManifestRootRefusal.NOT_A_WORKTREE, manifest_root,
            f'is a subdirectory of the worktree {toplevel!r}; pass that root instead',
        )

    project = await _rev_parse(
        manifest_root, project_root, '--git-common-dir',
        not_ok=ManifestRootRefusal.OTHER_REPOSITORY,
    )
    if isinstance(project, dict):
        return project
    (project_common_dir,) = project
    if os.path.realpath(project_common_dir) != os.path.realpath(common_dir):
        return _refusal(
            ManifestRootRefusal.OTHER_REPOSITORY, manifest_root,
            f'its git common dir {common_dir!r} is not project_root {project_root!r}\'s '
            f'{project_common_dir!r}',
        )
    if real_root == os.path.realpath(project_root):
        return _refusal(
            ManifestRootRefusal.MAIN_CHECKOUT, manifest_root,
            f"is project_root {project_root!r}'s own checkout; omit manifest_root",
        )
    return Path(real_root)


async def _rev_parse(
    manifest_root: str, path: str, *flags: str, not_ok: ManifestRootRefusal,
) -> list[str] | dict[str, Any]:
    """One stdout line per *flag* from ``git rev-parse`` in *path*, or a refusal.

    A non-zero exit is refused with *not_ok*; a git that cannot be spawned,
    times out or answers with the wrong number of lines with ``probe_failed``.
    """
    cmd = ['git', '-C', path, 'rev-parse', '--path-format=absolute', *flags]
    try:
        result = await run_git(cmd, timeout=_GIT_TIMEOUT_SECONDS)
    except OSError as exc:
        return _refusal(
            ManifestRootRefusal.PROBE_FAILED, manifest_root, f'git could not run in {path!r}: {exc}',
        )
    if result.timed_out:
        return _refusal(
            ManifestRootRefusal.PROBE_FAILED, manifest_root,
            f'git rev-parse in {path!r} did not answer within {_GIT_TIMEOUT_SECONDS:g}s',
        )
    if result.returncode != 0:
        return _refusal(
            not_ok, manifest_root,
            f'{path!r} is not inside a git worktree: {result.stderr.strip()}',
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(flags):
        return _refusal(
            ManifestRootRefusal.PROBE_FAILED, manifest_root,
            f'git rev-parse {" ".join(flags)} in {path!r} answered {result.stdout!r}',
        )
    return lines


def _refusal(code: ManifestRootRefusal, manifest_root: str, message: str) -> dict[str, Any]:
    return {
        'error': f'commit_planning: manifest_root {manifest_root!r} refused ({code}): {message}',
        'error_type': 'ManifestRootRefused',
        'reason_code': code.value,
        'manifest_root': manifest_root,
    }
