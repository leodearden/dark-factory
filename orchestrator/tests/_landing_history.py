"""This test root's landing history: a contained ``LandingRepo`` and an EventStore runs.db.

Imported by bare module name, like ``_fake_systemctl``. The repo and the event
writers live in ``shared.testing_landing_history``; what this root adds is its
containment, both esc-3072-3 guards from ``_orch_helpers`` (the ceiling
environment on every git call, and the repo-root pre-flight once the repo
exists), and ``create_runs_db``, which needs ``orchestrator.event_store``.
"""

from __future__ import annotations

from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling
from shared.testing_landing_history import LandingRepo

from orchestrator.event_store import EventStore


def landing_repo(root: Path, *, branch: str = 'main') -> LandingRepo:
    """A LandingRepo at *root* on *branch*, contained by both esc-3072-3 guards."""
    repo = LandingRepo(root, env=git_env_with_ceiling(root), branch=branch)
    assert_isolated_git_repo(root)
    return repo


def create_runs_db(path: Path) -> Path:
    """An empty runs.db at *path* carrying EventStore's schema."""
    EventStore(path, run_id='landing-history')
    return path
