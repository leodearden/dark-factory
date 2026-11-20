"""Observe, and race, the per-cgroup file reads ``sampler.consumers`` makes.

On a live host a short-lived cgroup (a verify scope) can be removed after its
parent's listing named it but before its own files are read. These helpers
wrap the real ``read_cgroup`` that ``sampler.consumers`` calls: one records
which directories had their files read, the other removes a directory just
before its files are read.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def record_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The directories whose files are read from now on, in read order."""
    from sampler import consumers

    real = consumers.read_cgroup
    read: list[Path] = []

    def recording(directory, *, present):
        read.append(Path(directory))
        return real(directory, present=present)

    monkeypatch.setattr(consumers, 'read_cgroup', recording)
    return read


def exit_after_listing(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Remove *directory*, with its subtree, just before its own files are read."""
    from sampler import consumers

    real = consumers.read_cgroup

    def exiting(path, *, present):
        if Path(path) == directory:
            shutil.rmtree(directory)
        return real(path, present=present)

    monkeypatch.setattr(consumers, 'read_cgroup', exiting)
