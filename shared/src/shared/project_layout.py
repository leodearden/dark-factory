"""Where a project's stores live under its root, and which roots are known.

Contract: plans/review-all-s14-shared-leaves-prd.md (decisions 1, 2 and 5).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    'ESCALATION_QUEUE_RELPATH',
    'KNOWN_PROJECT_ROOTS_ENV',
    'CheckoutRootError',
    'StrPath',
    'agent_transcripts_dir',
    'default_project_root',
    'escalation_queue_dir',
    'known_project_roots',
    'main_checkout_of',
    'offline_lane_dir',
    'orchestrator_data_dir',
    'park_eviction_requests_db',
    'reconciliation_tickets_db',
    'runs_db',
    'scheduler_eligibility_db',
    'scheduler_overrides_db',
    'scheduler_state_json',
    'startup_failures_json',
    'tasks_db',
    'verify_logs_dir',
]

ESCALATION_QUEUE_RELPATH = 'data/escalations'
KNOWN_PROJECT_ROOTS_ENV = 'DASHBOARD_KNOWN_PROJECT_ROOTS'

StrPath = str | os.PathLike[str]


def escalation_queue_dir(root: StrPath) -> Path:
    return Path(root) / ESCALATION_QUEUE_RELPATH


def orchestrator_data_dir(root: StrPath) -> Path:
    return Path(root) / 'data' / 'orchestrator'


def agent_transcripts_dir(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'agent-transcripts'


def offline_lane_dir(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'offline-lane'


def verify_logs_dir(root: StrPath) -> Path:
    return Path(root) / 'data' / 'verify-logs'


def runs_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'runs.db'


def scheduler_state_json(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_state.json'


def startup_failures_json(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'startup_failures.json'


def scheduler_overrides_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_overrides.db'


def park_eviction_requests_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'park_eviction_requests.db'


def scheduler_eligibility_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_eligibility.db'


def tasks_db(root: StrPath) -> Path:
    return Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'


def reconciliation_tickets_db(root: StrPath) -> Path:
    """fused-memory's tickets.db at its DEFAULT reconciliation.data_dir; a configured data_dir moves it."""
    return Path(root) / 'data' / 'reconciliation' / 'tickets.db'


def known_project_roots(environ: Mapping[str, str] = os.environ) -> tuple[Path, ...]:
    entries = environ.get(KNOWN_PROJECT_ROOTS_ENV, '').split(',')
    return tuple(Path(entry.strip()) for entry in entries if entry.strip())


class CheckoutRootError(RuntimeError):
    """No main checkout can be derived; ``checkout`` is the enclosing checkout when one was found."""

    def __init__(self, message: str, *, checkout: Path | None = None) -> None:
        super().__init__(message)
        self.checkout = checkout


_PROBE_ERRORS = (OSError, ValueError)


def default_project_root() -> Path:
    """The main checkout enclosing this module, even when imported from a linked worktree."""
    return main_checkout_of(Path(__file__))


def main_checkout_of(path: StrPath) -> Path:
    """The main checkout of the git checkout enclosing *path*, following a linked worktree.

    Raises CheckoutRootError when none can be derived (no enclosing checkout, an
    unprobeable path, an unreadable or broken gitfile, or a bare-backed worktree);
    its ``checkout`` is the enclosing checkout when one was found.
    """
    checkout = _enclosing_checkout(path)
    dot_git = checkout / '.git'
    with _probing(dot_git, checkout=checkout):
        if dot_git.is_dir():
            return checkout
    gitdir = _gitdir_of(checkout)
    commondir = gitdir / 'commondir'
    with _probing(commondir, checkout=checkout):
        if not commondir.is_file():
            return checkout
    common = _realpath(gitdir / _read_pointer_file(commondir, checkout=checkout).strip(), checkout=checkout)
    if common.name != '.git':
        raise CheckoutRootError(
            f'{checkout} is a linked worktree whose common git dir {common} is not '
            f'the .git of a main checkout (bare repo or separate git dir; named by '
            f'{commondir}), so no main checkout can be derived',
            checkout=checkout,
        )
    return common.parent


@contextmanager
def _probing(target: StrPath, *, checkout: Path | None) -> Iterator[None]:
    try:
        yield
    except _PROBE_ERRORS as exc:
        raise CheckoutRootError(
            f'probing {target} failed ({type(exc).__name__}: {exc}), '
            f'so no main checkout can be derived',
            checkout=checkout,
        ) from exc


def _realpath(path: StrPath, *, checkout: Path | None) -> Path:
    """``Path.resolve()`` as Python 3.13 does it: before 3.13 it raised RuntimeError on a symlink loop."""
    with _probing(path, checkout=checkout):
        return Path(os.path.realpath(path))


def _enclosing_checkout(path: StrPath) -> Path:
    start = _realpath(path, checkout=None)
    unprobeable: Exception | None = None
    for candidate in (start, *start.parents):
        try:
            if (candidate / '.git').exists():
                return candidate
        except _PROBE_ERRORS as exc:
            unprobeable = exc
    raise CheckoutRootError(
        f'no git checkout encloses {start}: no ancestor has a .git this process can see'
    ) from unprobeable


def _read_pointer_file(pointer_file: Path, *, checkout: Path) -> str:
    with _probing(pointer_file, checkout=checkout):
        return pointer_file.read_text(encoding='utf-8')


def _gitdir_of(checkout: Path) -> Path:
    dot_git = checkout / '.git'
    text = _read_pointer_file(dot_git, checkout=checkout)
    values = [
        line.removeprefix('gitdir:').strip()
        for line in text.splitlines()
        if line.startswith('gitdir:')
    ]
    if not values or not values[0]:
        raise CheckoutRootError(
            f'{dot_git} is not a worktree .git file: expected a "gitdir: <path>" line, '
            f'got {text!r}',
            checkout=checkout,
        )
    gitdir = _realpath(checkout / values[0], checkout=checkout)
    with _probing(gitdir, checkout=checkout):
        is_gitdir = gitdir.is_dir()
    if not is_gitdir:
        raise CheckoutRootError(
            f'{dot_git} names gitdir {values[0]!r}, which is not an existing directory '
            f'(resolved: {gitdir}); the main checkout\'s .git must be visible here',
            checkout=checkout,
        )
    return gitdir
