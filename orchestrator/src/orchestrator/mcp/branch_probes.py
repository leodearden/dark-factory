"""Synchronous, never-raising git reads a plan-tools call runs against its worktree.

Every probe runs ``git`` with ``cwd=worktree`` and ``timeout=10``; a git
failure is an answer (an empty sha, ``False``, :attr:`Reachability.UNKNOWN`),
never an exception.
"""
from __future__ import annotations

import enum
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Reachability(enum.Enum):
    """Whether a sha is on the worktree's branch, or that the check could not run."""

    REACHABLE = 'reachable'
    UNREACHABLE = 'unreachable'
    UNKNOWN = 'unknown'


def resolve_main_sha(worktree: Path) -> str:
    """Return the current ``main`` HEAD SHA visible from *worktree*.

    Falls back to an empty string on git failure — the workflow side
    treats an empty ``main_sha_at_report`` as "advance check skipped".
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'main'],
            cwd=str(worktree),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Failed to resolve main SHA for blocking_dependency report: %s', exc
        )
        return ''


def sha_exists_on_branch(worktree: Path, sha: str) -> bool:
    """Return True iff *sha* is reachable from HEAD in the repo at *worktree*.

    The machine-checked corroborate-before-acting guard for
    ``orchestrator/src/orchestrator/mcp/plan_tools.py::_mark_step_committed``
    (INV-1/INV-3): the architect must be UNABLE to pre-satisfy a plan step
    against a commit that does not resolve on the current branch.

    Reachability (``git merge-base --is-ancestor``), NOT mere object existence
    (``git cat-file -e``): in this repo's shared/multi-worktree object store a
    commit from another task's branch — or a dangling/unreachable object —
    exists in the object DB yet is NOT on this branch. ``--is-ancestor`` returns
    non-zero (exit 1 for a real-but-unreachable commit, 128 for a bad object)
    for both, so the guard matches its advertised on-branch contract. It
    deliberately does NOT prove the step's semantics — VERIFY remains the gate.

    The fail-CLOSED projection of :func:`sha_branch_reachability`:
    :attr:`Reachability.REACHABLE` is the only True, so both UNREACHABLE and
    UNKNOWN are ``False``. A pre-satisfaction guard that cannot check must
    refuse — the opposite of
    ``orchestrator/src/orchestrator/mcp/plan_tools.py::_mark_step_done``'s
    fail-OPEN treatment of UNKNOWN.
    """
    return sha_branch_reachability(worktree, sha) is Reachability.REACHABLE


def sha_branch_reachability(worktree: Path, sha: str) -> Reachability:
    """Tri-state reachability probe for *sha* (task 3651).

    THE ONE PROBE: :func:`sha_exists_on_branch` projects it, so the argv, the
    cwd and the ``timeout=10`` are stated once. What it adds over that boolean
    is UNKNOWN, "the check could not run at all", which is what lets
    ``orchestrator/src/orchestrator/mcp/plan_tools.py::_mark_step_done`` fail
    OPEN on an infra fault while still rejecting a sha the branch genuinely
    does not carry.

    Two probes, in order:

    1. ``git rev-parse --git-dir`` — is there a usable repo at *worktree*? A
       non-zero rc (or an ``OSError``/``SubprocessError``) means git or the
       worktree is unavailable, which is an INFRA fault and says nothing about
       the sha: UNKNOWN.
    2. ``git merge-base --is-ancestor <sha> HEAD`` — rc 0 is REACHABLE; ANY
       other rc is UNREACHABLE. Folding rc 1 (a real commit that is not an
       ancestor) and rc 128 (a bad/fabricated object) together is correct here
       precisely because probe 1 already proved the repo healthy. An
       ``OSError``/``SubprocessError`` on this call is UNKNOWN.
    """
    try:
        health = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if health.returncode != 0:
            return Reachability.UNKNOWN
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Reachability probe: no usable git repo at %s (%s)', worktree, exc
        )
        return Reachability.UNKNOWN
    try:
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', sha, 'HEAD'],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Reachability probe failed for sha %r: %s', sha, exc
        )
        return Reachability.UNKNOWN
    if result.returncode == 0:
        return Reachability.REACHABLE
    return Reachability.UNREACHABLE
