"""scripts/legibility/unlanded.py — land a machine-written legibility write
with one bounded, lock-tolerant scoped commit, or put it back when that commit
does not land.

The trickle and the census write into a target repo's machine-operated main
checkout and then commit with :func:`commit_only`. When that commit does not
land — refused by the target repo's pre-commit hook, still contended after
its bounded lock retries, or failed for a plain git reason — the written
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
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from legibility import trickle_state

_LOCK_CONTENTION = re.compile(
    r"^(?:fatal|error): (?:cannot lock ref\b|unable to create '[^'\n]*\.lock')",
    re.IGNORECASE | re.MULTILINE,
)
"""git's own report of a transient ref/index-lock contention (the merge worker
or a hook holding the lock in the shared main checkout): retried. Anything
else, such as a hook refusal, is final: a retry re-runs the whole hook."""

COMMIT_ATTEMPTS = 5
COMMIT_BACKOFF_SECONDS = 0.2

QUARANTINE_DIRNAME = 'quarantine'

GIT_TIMEOUT_SECONDS = 60.0
"""Bound on each git call :func:`roll_back` makes: a hung git becomes a
rollback-INCOMPLETE result, so the caller's escalation is still sent."""


@dataclass(frozen=True)
class CommitResult:
    """What :func:`commit_only` did: landed as ``sha``, or not, with git's
    diagnostic from the last of its ``attempts``."""

    ok: bool
    sha: str | None
    stderr: str
    attempts: int


_Runner = Callable[..., subprocess.CompletedProcess[str]]


def commit_only(
    repo: str | os.PathLike[str],
    paths: Iterable[str | os.PathLike[str]],
    message: str,
    *,
    runner: _Runner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = COMMIT_ATTEMPTS,
    backoff: float = COMMIT_BACKOFF_SECONDS,
) -> CommitResult:
    """Commit only *paths* in *repo* (``git commit --only``), leaving every
    other staged or dirty path alone. Never stashes, never skips the hooks, and
    never raises for a git or launch failure.

    A path git does not know yet is made known by a scoped ``git add`` within
    the same attempt. A lock contention, in the commit or in that add, is
    retried after ``sleep(backoff)`` for up to *max_attempts* attempts; any
    other failure is final. *runner* is ``subprocess.run``, looked up at call
    time.
    """
    run = runner if runner is not None else subprocess.run
    git = ['git', '-C', str(repo)]
    names = [str(path) for path in paths]
    attempt = 1
    while True:
        result = _attempt_commit(run, git, names, message, attempt)
        if result.ok or attempt >= max_attempts or not _is_lock_contention(result.stderr):
            return result
        sleep(backoff)
        attempt += 1


def _attempt_commit(
    run: _Runner, git: list[str], paths: list[str], message: str, attempt: int,
) -> CommitResult:
    try:
        failure = _commit_once(run, git, paths, message)
        if failure is None:
            head = run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True)
            sha = (head.stdout or '').strip() or None
            return CommitResult(ok=True, sha=sha, stderr='', attempts=attempt)
    except OSError as exc:
        failure = f'git could not be run: {exc}'
    return CommitResult(ok=False, sha=None, stderr=failure, attempts=attempt)


def _commit_once(run: _Runner, git: list[str], paths: list[str], message: str) -> str | None:
    """``None`` once the commit lands, else git's diagnostic for why not."""
    commit = [*git, 'commit', '--only', *paths, '-m', message]
    committed = run(commit, capture_output=True, text=True)
    if committed.returncode == 0:
        return None
    failure = _diagnostic(committed)
    if 'did not match any file' not in failure.lower():
        return failure
    added = run([*git, 'add', '--', *paths], capture_output=True, text=True)
    if added.returncode != 0:
        return _diagnostic(added)
    committed = run(commit, capture_output=True, text=True)
    return None if committed.returncode == 0 else _diagnostic(committed)


def _diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
    """Both of git's streams: hook output arrives on stderr, while git's own
    verdict on a no-op commit arrives on stdout."""
    streams = (completed.stderr, completed.stdout)
    return '\n'.join(text.strip() for text in streams if text and text.strip())


def _is_lock_contention(diagnostic: str) -> bool:
    return _LOCK_CONTENTION.search(diagnostic) is not None


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
