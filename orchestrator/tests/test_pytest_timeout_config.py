"""Watchdog tests that pin the pytest-timeout config in orchestrator/pyproject.toml.

These tests exist because:

* ~10 test files across orchestrator use bare ``MagicMock`` for UsageGate instead
  of ``shared.testing.make_gate_mock``.  Any one of them can trigger an
  asyncio.sleep hang if the mock's ``slot.detect_cap_hit`` returns a truthy
  MagicMock, causing the process to spin indefinitely.  A 60-second global cap
  converts a "silent 45-minute wedge" into a loud ``Failed: Timeout >60.0s``
  failure with a stack dump — far easier to triage.

  See ``shared/src/shared/testing.py:5-21`` for the canonical ``make_gate_mock``
  fixture and commit b9d39def62 where the shared fix landed.

* Consistency with ``fused-memory/pyproject.toml`` (which uses the same
  timeout = 60 / timeout_method = "thread" pair for the same reasons, and adds
  xdist compatibility as a secondary motivation).

``timeout_method = "thread"`` is chosen (rather than ``"signal"``) for
cross-subproject consistency and to future-proof orchestrator if xdist is ever
enabled.  The thread-based watchdog uses ``ctypes.PyThreadState_SetAsyncExc``
and works in any thread.

If this test fails it means the timeout config was removed or changed.  Restore
``timeout = 60`` and ``timeout_method = "thread"`` under
``[tool.pytest.ini_options]`` in orchestrator/pyproject.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Locate package root (orchestrator/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent


def _load_pytest_ini_options() -> dict:
    """Load and return ``[tool.pytest.ini_options]`` from orchestrator/pyproject.toml."""
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    return config.get("tool", {}).get("pytest", {}).get("ini_options", {})


def test_orchestrator_pytest_timeout_is_60_seconds() -> None:
    """orchestrator/pyproject.toml must set ``timeout = 60`` under [tool.pytest.ini_options].

    Without this cap, a misconfigured UsageGate mock (bare MagicMock instead of
    ``shared.testing.make_gate_mock``) causes the test process to spin in an
    asyncio.sleep loop indefinitely.  ~10 test files across orchestrator use bare
    MagicMock for UsageGate and are at risk of triggering this hang.

    See shared/src/shared/testing.py:5-21 and commit b9d39def62 for the canonical
    fix and rationale.
    """
    ini_options = _load_pytest_ini_options()
    timeout = ini_options.get("timeout")
    assert timeout == 60, (
        f"Expected [tool.pytest.ini_options] timeout = 60 in "
        f"{_PACKAGE_ROOT / 'pyproject.toml'}, got {timeout!r}.\n"
        "This setting caps per-test runtime so UsageGate-mock hangs produce "
        "loud failures instead of 45-minute silent wedges.  ~10 orchestrator "
        "test files use bare MagicMock for UsageGate and are at risk.\n"
        "Restore: add  timeout = 60  under [tool.pytest.ini_options].\n"
        "Reference: shared/src/shared/testing.py:5-21 and commit b9d39def62."
    )


def test_orchestrator_pytest_timeout_method_is_thread() -> None:
    """orchestrator/pyproject.toml must set ``timeout_method = \"thread\"`` under [tool.pytest.ini_options].

    ``thread`` method chosen for cross-subproject consistency with fused-memory
    (which requires thread for xdist compat) and to future-proof orchestrator if
    xdist is ever enabled.  The thread-based watchdog uses
    ``ctypes.PyThreadState_SetAsyncExc`` and works in any thread.
    """
    ini_options = _load_pytest_ini_options()
    method = ini_options.get("timeout_method")
    assert method == "thread", (
        f"Expected [tool.pytest.ini_options] timeout_method = \"thread\" in "
        f"{_PACKAGE_ROOT / 'pyproject.toml'}, got {method!r}.\n"
        "\"thread\" method chosen for cross-subproject consistency with "
        "fused-memory and to future-proof if xdist is enabled.\n"
        "Restore: add  timeout_method = \"thread\"  under [tool.pytest.ini_options].\n"
        "Reference: shared/src/shared/testing.py:5-21 and commit b9d39def62."
    )
