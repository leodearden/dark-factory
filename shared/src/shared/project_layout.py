"""Where a project's stores live under its root, and which roots are known.

Contract: plans/review-all-s14-shared-leaves-prd.md (decisions 1, 2 and 5).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    'orchestrator_data_dir',
    'park_eviction_requests_db',
    'runs_db',
    'scheduler_eligibility_db',
    'scheduler_overrides_db',
    'scheduler_state_json',
    'tasks_db',
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


def runs_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'runs.db'


def scheduler_state_json(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_state.json'


def scheduler_overrides_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_overrides.db'


def park_eviction_requests_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'park_eviction_requests.db'


def scheduler_eligibility_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_eligibility.db'


def tasks_db(root: StrPath) -> Path:
    return Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'


def known_project_roots(environ: Mapping[str, str] = os.environ) -> tuple[Path, ...]:
    entries = environ.get(KNOWN_PROJECT_ROOTS_ENV, '').split(',')
    return tuple(Path(entry.strip()) for entry in entries if entry.strip())


class CheckoutRootError(RuntimeError):
    """The main checkout cannot be derived from where this module lives."""


def default_project_root() -> Path:
    """The main checkout enclosing this module, even when imported from a linked worktree."""
    return main_checkout_of(Path(__file__))


def main_checkout_of(path: StrPath) -> Path:
    """The main checkout of the git checkout enclosing *path*, following a linked worktree.

    Raises CheckoutRootError when none can be derived (no enclosing checkout, an
    unreadable or broken gitfile, or a bare-backed worktree).
    """
    checkout = _enclosing_checkout(Path(path).resolve())
    dot_git = checkout / '.git'
    if dot_git.is_dir():
        return checkout
    gitdir = _gitdir_of(dot_git)
    commondir = gitdir / 'commondir'
    if not commondir.is_file():
        return checkout
    common = (gitdir / _read_pointer_file(commondir).strip()).resolve()
    if common.name != '.git':
        raise CheckoutRootError(
            f'{checkout} is a linked worktree whose common git dir {common} is not '
            f'the .git of a main checkout (bare repo or separate git dir; named by '
            f'{commondir}), so no main checkout can be derived'
        )
    return common.parent


def _enclosing_checkout(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / '.git').exists():
            return candidate
    raise CheckoutRootError(f'no git checkout encloses {start}: no ancestor has .git')


def _read_pointer_file(pointer_file: Path) -> str:
    try:
        return pointer_file.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise CheckoutRootError(
            f'git pointer file {pointer_file} is unreadable ({type(exc).__name__}: {exc}), '
            f'so no main checkout can be derived'
        ) from exc


def _gitdir_of(dot_git: Path) -> Path:
    text = _read_pointer_file(dot_git)
    values = [
        line.removeprefix('gitdir:').strip()
        for line in text.splitlines()
        if line.startswith('gitdir:')
    ]
    if not values or not values[0]:
        raise CheckoutRootError(
            f'{dot_git} is not a worktree .git file: expected a "gitdir: <path>" line, '
            f'got {text!r}'
        )
    gitdir = (dot_git.parent / values[0]).resolve()
    if not gitdir.is_dir():
        raise CheckoutRootError(
            f'{dot_git} names gitdir {values[0]!r}, which is not an existing directory '
            f'(resolved: {gitdir}); the main checkout\'s .git must be visible here'
        )
    return gitdir
