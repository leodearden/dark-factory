"""Tests for the watchdog's module-removal waiver tier (task 7198).

A watched commit that deletes or renames a Python module arms a
ModuleNotFoundError in every running orchestrator that imports it lazily, so
scripts/orchestrator-watchdog.py lets such a removal bypass the fleet-deploy
clock gates once a short grace has passed. OPERATIONS.md section 8 ("Fleet
redeploy & watchdog") describes the behaviour.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]


def _load_watchdog() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "orchestrator_watchdog", REPO_ROOT / "scripts" / "orchestrator-watchdog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_removal_grace_defaults_to_600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCH_MODULE_REMOVAL_GRACE_SECS", raising=False)
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 600


def test_module_removal_grace_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCH_MODULE_REMOVAL_GRACE_SECS", "42")
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 42


def test_malformed_module_removal_grace_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCH_MODULE_REMOVAL_GRACE_SECS", "ten-minutes")
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 600


def test_module_removal_watched_paths_extend_watched_paths_with_shared() -> None:
    wdog = _load_watchdog()
    watched = wdog.MODULE_REMOVAL_WATCHED_PATHS
    assert set(wdog.WATCHED_PATHS) <= set(watched)
    assert "shared/src/" in watched
    missing = [path for path in watched if not (REPO_ROOT / path).exists()]
    assert not missing, f"MODULE_REMOVAL_WATCHED_PATHS entries naming no path: {missing}"
