"""Primitives that carry a teardown to a confirmed end, despite a cancelled
caller and without mistaking an unknown for done.

First consumer: git_ops.py::GitOps.ephemeral_worktree.
"""
from __future__ import annotations

import os
from pathlib import Path


def path_confirmed_absent(path: Path) -> bool:
    """True only when *path* is known not to exist; an lstat that fails for any other reason is not absence."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False
