"""scripts/legibility/unlanded.py — put back a machine-written legibility
write whose commit did not land.

The trickle and the census write into a target repo's machine-operated main
checkout and then commit. When that commit does not land — refused by the
target repo's pre-commit hook, or failed for a plain git reason — the written
paths stay dirty (or, after the scoped ``git add`` fallback, staged). Left
there they block the target's redeploy, trip the orchestrator's dirty-tree
escalation and are re-submitted the next night. The remedy, stated here as
this repo's copy of ruling R4 in the reify repo's landing contract
(``reify:docs/legibility/landing-contract.md``, a file that is NOT in
dark-factory): never commit with ``--no-verify``; restore the written paths
to HEAD, quarantine the refused content outside the tracked tree, and do not
advance census-state.

:func:`roll_back` does that in the one order that cannot lose mined signal:
quarantine first, and restore only once the copy is safe. Quarantine layout:
``<quarantine_root(project_id)>/<label>-<random>/<repo-relative path>``.
Nothing expires a quarantine; the operator removes one after recovering it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from legibility import trickle_state

QUARANTINE_DIRNAME = 'quarantine'

GIT_TIMEOUT_SECONDS = 60.0
"""Bound on each git call :func:`roll_back` makes: a hung git becomes a
rollback-INCOMPLETE result, so the caller's escalation is still sent."""


def quarantine_root(project_id: str) -> Path:
    """Directory holding every quarantined write for *project_id*: host-local,
    outside every checkout, beside the trickle run-state file."""
    return trickle_state.project_state_dir(project_id) / QUARANTINE_DIRNAME


@dataclass(frozen=True)
class Rollback:
    """What :func:`roll_back` did with one refused write."""

    quarantine_dir: Path | None
    """Holds a complete copy of every written path that existed, laid out at
    its repo-relative path; ``None`` when no complete copy could be made."""

    paths: tuple[str, ...]
    """The written paths, repo-relative POSIX."""

    failure: str | None = None
    """Why the checkout is NOT back at HEAD for these paths, or ``None``."""

    @property
    def restored(self) -> bool:
        return self.failure is None

    def describe(self) -> str:
        """One sentence for an escalation detail or a journal line."""
        if self.restored:
            return (
                f'refused content quarantined at {self.quarantine_dir}; '
                f'restored to HEAD: {", ".join(self.paths)}'
            )
        text = f'rollback INCOMPLETE: {self.failure}'
        if self.quarantine_dir is not None:
            text += f' (refused content quarantined at {self.quarantine_dir})'
        return text


def roll_back(
    repo: str | os.PathLike[str],
    paths: Iterable[str | os.PathLike[str]],
    *,
    project_id: str,
    label: str,
) -> Rollback:
    """Quarantine *paths* (relative to *repo*, or absolute inside it), then
    put them back to HEAD in both the index and the worktree.

    A path HEAD tracks is restored; a path HEAD does not track is removed,
    whether it was staged or never added. Every other path in the checkout is
    left alone. Each refusal gets its own quarantine dir, named after *label*.

    Never raises: both callers are already inside a failure branch whose
    escalation must not be lost. If the quarantine cannot be written the
    checkout is left untouched, because a dirty tree is recoverable and a
    destroyed night of mining is not.
    """
    try:
        root, rels = _repo_relative(repo, paths)
    except (OSError, RuntimeError, ValueError) as exc:
        return Rollback(None, (), failure=f'{exc}; checkout left untouched')
    try:
        quarantine_dir = _quarantine(root, rels, project_id=project_id, label=label)
    except OSError as exc:
        return Rollback(
            None, rels,
            failure=f'could not quarantine the refused content: {exc}; checkout left untouched',
        )
    return Rollback(quarantine_dir, rels, failure=_restore_to_head(root, rels))


def _repo_relative(
    repo: str | os.PathLike[str], paths: Iterable[str | os.PathLike[str]],
) -> tuple[Path, tuple[str, ...]]:
    root = Path(repo).resolve()
    rels = []
    for path in paths:
        resolved = (root / path).resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError(f'refusing to roll back {path}: not a file inside {root}')
        rels.append(resolved.relative_to(root).as_posix())
    return root, tuple(dict.fromkeys(rels))


def _quarantine(root: Path, rels: tuple[str, ...], *, project_id: str, label: str) -> Path:
    parent = quarantine_root(project_id)
    parent.mkdir(parents=True, exist_ok=True)
    quarantine_dir = Path(tempfile.mkdtemp(prefix=f'{label}-', dir=parent))
    try:
        for rel in rels:
            source = root / rel
            if source.exists():
                destination = quarantine_dir / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    except OSError as exc:
        raise OSError(f'partial copy left in {quarantine_dir}: {exc}') from exc
    return quarantine_dir


class _GitFailed(Exception):
    pass


def _restore_to_head(root: Path, rels: tuple[str, ...]) -> str | None:
    if not rels:
        return None
    try:
        listed = _git(root, 'ls-files', '-z', '--with-tree=HEAD', '--', *rels)
        known = {os.fsdecode(entry) for entry in listed.split(b'\0') if entry}
        if known:
            _git(root, 'restore', '--source=HEAD', '--staged', '--worktree', '--', *sorted(known))
        for rel in rels:
            if rel not in known:
                (root / rel).unlink(missing_ok=True)
    except (_GitFailed, OSError) as exc:
        return f'{exc}; the written paths are still dirty in {root}'
    return None


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ['git', '--literal-pathspecs', '-C', str(root), *args],
            capture_output=True, timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise _GitFailed(f'git {args[0]} timed out after {exc.timeout}s') from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode('utf-8', 'replace').strip()
        raise _GitFailed(f'git {args[0]} failed (rc={completed.returncode}): {stderr}')
    return completed.stdout
