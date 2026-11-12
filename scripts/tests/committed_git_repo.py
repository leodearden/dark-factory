"""A throwaway git repo with its files committed, for tests of code that reads a tree.

A plain module rather than a conftest fixture, the same convention as
``git_checkout_root``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def committed_repo(root: Path, files: Mapping[str, str]) -> str:
    """Init a repo at *root* whose single commit holds *files* (relpath -> text).

    Returns *root* as a string, the form the tree readers under test take.
    """
    def git(*args: str) -> None:
        subprocess.run(["git", *args], check=True, capture_output=True, timeout=30)

    git("init", "-b", "main", str(root))
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    git("-C", str(root), "add", "-A")
    git("-C", str(root), "-c", "user.email=t@e", "-c", "user.name=t",
        "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "seed")
    return str(root)
