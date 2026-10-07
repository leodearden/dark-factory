"""The git checkout this test directory lives in, for live-corpus pins over tracked files.

A plain module rather than a conftest fixture, the same convention as
``cli_subprocess_timeout``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def checkout_root_or_skip() -> str:
    """The checkout's top level, or skip the calling test when there is none."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("not a git checkout")
    if completed.returncode != 0:
        pytest.skip("not a git checkout")
    return completed.stdout.strip()
