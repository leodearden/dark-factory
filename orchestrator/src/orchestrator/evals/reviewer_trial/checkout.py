"""Materialise a corpus fixture's reviewed tree in a hermetic scratch clone.

The clone is ``git clone --shared`` of the fixture's source repository, detached
at the fixture's base commit with the stored diff applied and committed, so HEAD
is the reviewed tree and HEAD^ is its base: the view a production reviewer has of
a task branch. Each checkout is private to one panel and removed on exit, so
parallel panels share no mutable git state.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from shared.git_async import GitResult, run_git

from orchestrator.evals.reviewer_trial.corpus import DEFAULT_FIXTURE_CWD, CorpusDiff

_SCRATCH_COMMITTER = (
    '-c', 'user.name=reviewer-trial',
    '-c', 'user.email=reviewer-trial@localhost',
    '-c', 'commit.gpgsign=false',
)

SCRATCH_GIT_TIMEOUT_SECS = 300.0
"""How long any one git step building a scratch clone may run before the fixture is skipped."""


@dataclass(frozen=True)
class FixtureCheckout:
    """A scratch clone at a fixture's tree.

    Exactly one of ``change_base`` (the commit the fixture's change sits on, HEAD^)
    and ``unapplied_reason`` (why the change is not committed in this clone) is set.
    """

    root: Path
    change_base: str | None
    unapplied_reason: str | None

    def __post_init__(self) -> None:
        if (self.change_base is None) == (self.unapplied_reason is None):
            raise ValueError(
                'FixtureCheckout needs exactly one of change_base and unapplied_reason',
            )


class FixtureNotMaterialisable(Exception):
    """A fixture's reviewed tree cannot be built, so a review of it cannot run."""

    def __init__(self, diff_id: str, reason: str) -> None:
        super().__init__(f'fixture {diff_id} cannot be materialised: {reason}')
        self.diff_id = diff_id
        self.reason = reason


class FixtureEnrichmentMissing(Exception):
    """Fixtures lack the ``context`` block a context-complete review is briefed from."""

    def __init__(self, diff_ids: Sequence[str]) -> None:
        super().__init__(
            'a context-complete (d2) review is briefed from a fixture\'s `context` block, '
            f'which these fixtures lack: {", ".join(diff_ids)}',
        )
        self.diff_ids = tuple(diff_ids)


@asynccontextmanager
async def fixture_checkout(
    diff: CorpusDiff,
    scratch_parent: Path | None = None,
    *,
    git_timeout_secs: float = SCRATCH_GIT_TIMEOUT_SECS,
) -> AsyncIterator[FixtureCheckout]:
    """Yield *diff*'s reviewed tree in a scratch clone, removed when the block exits.

    A diff that does not apply at its recorded base leaves HEAD at that base, and a
    fixture with no recorded base leaves HEAD at the source's HEAD; either way the
    checkout's ``unapplied_reason`` says why. Raises FixtureNotMaterialisable when
    the source directory is missing, or when git cannot clone the source or check
    out the base at all, including a step killed after *git_timeout_secs*.
    """
    scratch = Path(tempfile.mkdtemp(prefix='reviewer-trial-', dir=scratch_parent))
    try:
        yield await _materialise(diff, scratch / 'clone', git_timeout_secs)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def _materialise(diff: CorpusDiff, clone: Path, timeout: float) -> FixtureCheckout:
    source = await _required_git(
        diff, ['rev-parse', '--show-toplevel'], cwd=diff.cwd or DEFAULT_FIXTURE_CWD,
        timeout=timeout,
    )
    await _required_git(
        diff, ['clone', '--shared', '--no-checkout', '--quiet', source, str(clone)],
        timeout=timeout,
    )
    base = diff.context.base_sha if diff.context is not None else None
    await _required_git(
        diff,
        ['-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', base or 'HEAD'],
        cwd=clone, timeout=timeout,
    )
    if base is None:
        return FixtureCheckout(clone, None, _no_base_reason(diff))

    applied = await run_git(
        ['git', 'apply', '--index', '--recount'], clone,
        input_text=diff.diff_text, timeout=timeout,
    )
    if not applied.ok:
        return FixtureCheckout(clone, None, _failure_text(applied))
    await _required_git(
        diff,
        [*_SCRATCH_COMMITTER, 'commit', '--quiet', '--no-verify',
         '-m', f'reviewer-trial fixture {diff.diff_id}'],
        cwd=clone, timeout=timeout,
    )
    return FixtureCheckout(clone, base, None)


async def _required_git(
    diff: CorpusDiff, args: Sequence[str], *, cwd: Path | None = None, timeout: float,
) -> str:
    command = f'`git {" ".join(args)}`'
    try:
        result = await run_git(['git', *args], cwd, timeout=timeout)
    except FileNotFoundError as exc:
        raise FixtureNotMaterialisable(
            diff.diff_id, f'{command} could not start in {cwd}: {exc}',
        ) from exc
    if not result.ok:
        raise FixtureNotMaterialisable(
            diff.diff_id, f'{command} failed: {_failure_text(result)}',
        )
    return result.stdout.strip()


def _no_base_reason(diff: CorpusDiff) -> str:
    if diff.context is None:
        return 'the fixture has no context block, so no base commit is recorded'
    return diff.context.base_sha_reason or 'the fixture records no base commit'


def _failure_text(result: GitResult) -> str:
    return result.stderr.strip() or f'exit code {result.returncode}'
