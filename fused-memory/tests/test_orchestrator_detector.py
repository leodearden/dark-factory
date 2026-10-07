"""Tests for the flock-based orchestrator liveness probe.

``is_orchestrator_lock_held`` reads whether some process holds a flock on
``<project_root>/data/orchestrator/orchestrator.lock``. flock is per open
file description, so a lock held on a separate ``open()`` in this process
conflicts with the probe exactly as the orchestrator's own lock does.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

from fused_memory.services.orchestrator_detector import is_orchestrator_lock_held


def _lock_path(root: Path) -> Path:
    lock_dir = root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / 'orchestrator.lock'


def test_missing_lock_file_reads_not_held(tmp_path):
    assert is_orchestrator_lock_held(str(tmp_path)) is False


def test_unheld_lock_file_reads_not_held(tmp_path):
    _lock_path(tmp_path).write_text('')

    assert is_orchestrator_lock_held(tmp_path) is False


def test_exclusively_held_lock_reads_held(tmp_path):
    lock_path = _lock_path(tmp_path)
    lock_path.write_text('')
    holder = lock_path.open('r+b')
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert is_orchestrator_lock_held(str(tmp_path)) is True
    finally:
        holder.close()


def test_unopenable_lock_path_reads_not_held(tmp_path):
    _lock_path(tmp_path).mkdir()

    assert is_orchestrator_lock_held(tmp_path) is False
