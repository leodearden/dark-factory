"""Watchdog tests that pin the pytest-timeout config in fused-memory/pyproject.toml.

These tests exist because:

* ``fused-memory`` tests run under pytest-xdist (``-n auto --dist loadgroup``).
  If any test hangs (e.g. an asyncio.sleep loop driven by a misconfigured
  UsageGate mock returning a truthy MagicMock from ``slot.detect_cap_hit``), the
  entire xdist worker wedges silently for the full run duration — no output, no
  progress.  See ``shared/src/shared/testing.py:5-21`` for the canonical mock
  fixture (``make_gate_mock``) and commit b9d39def62 where the shared fix landed.

* ``TaskInterceptor._curator_worker`` (task_interceptor.py:1455-1511) blocks on
  ``queue.get()`` indefinitely.  Fixture teardown at test_ticket_worker.py:52-59
  cancels these workers, but if teardown itself is interrupted the worker keeps
  running and the process never exits.

The 60-second global cap converts a "silent 45-minute wedge" into a loud
``Failed: Timeout >60.0s`` failure with a stack dump — far easier to triage.

``timeout_method = "thread"`` is required instead of ``"signal"`` because
signal-based timeout uses ``signal.alarm`` / SIGALRM, which only fires in the
main thread.  pytest-xdist spawns worker *processes* whose main-thread invariant
is not guaranteed.  The thread-based watchdog uses ``ctypes.PyThreadState_SetAsyncExc``
and works in any thread, including xdist workers.

If this test fails it means the timeout config was removed or changed.  Restore
``timeout = 60`` and ``timeout_method = "thread"`` under
``[tool.pytest.ini_options]`` in fused-memory/pyproject.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Locate package root (fused-memory/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent


def _load_pytest_ini_options() -> dict:
    """Load and return ``[tool.pytest.ini_options]`` from fused-memory/pyproject.toml."""
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    return config.get("tool", {}).get("pytest", {}).get("ini_options", {})


def test_fused_memory_pytest_timeout_is_60_seconds() -> None:
    """fused-memory/pyproject.toml must set ``timeout = 60`` under [tool.pytest.ini_options].

    Without this cap, a misconfigured UsageGate mock (bare MagicMock instead of
    ``shared.testing.make_gate_mock``) causes the test process to spin in a
    ``asyncio.sleep`` loop for the entire allowed run time, wedging the whole
    xdist worker pool.  See shared/src/shared/testing.py:5-21 and commit
    b9d39def62 for the canonical fix and rationale.
    """
    ini_options = _load_pytest_ini_options()
    timeout = ini_options.get("timeout")
    assert timeout == 60, (
        f"Expected [tool.pytest.ini_options] timeout = 60 in "
        f"{_PACKAGE_ROOT / 'pyproject.toml'}, got {timeout!r}.\n"
        "This setting caps per-test runtime so UsageGate-mock hangs and "
        "_curator_worker queue.get() blocks (task_interceptor.py:1455-1511) "
        "produce loud failures instead of 45-minute silent wedges.\n"
        "Restore: add  timeout = 60  under [tool.pytest.ini_options].\n"
        "Reference: shared/src/shared/testing.py:5-21 and commit b9d39def62."
    )


def test_fused_memory_pytest_timeout_method_is_thread() -> None:
    """fused-memory/pyproject.toml must set ``timeout_method = \"thread\"`` under [tool.pytest.ini_options].

    ``signal`` method (SIGALRM) only fires in the main thread and is incompatible
    with pytest-xdist, which is enabled via ``addopts = "-n auto --dist loadgroup"``.
    The ``thread`` method uses ``ctypes.PyThreadState_SetAsyncExc`` and works
    across all threads including xdist worker processes.
    """
    ini_options = _load_pytest_ini_options()
    method = ini_options.get("timeout_method")
    assert method == "thread", (
        f"Expected [tool.pytest.ini_options] timeout_method = \"thread\" in "
        f"{_PACKAGE_ROOT / 'pyproject.toml'}, got {method!r}.\n"
        "\"signal\" method (SIGALRM) is incompatible with pytest-xdist workers "
        "(addopts = \"-n auto --dist loadgroup\").  Must use \"thread\" method.\n"
        "Restore: add  timeout_method = \"thread\"  under [tool.pytest.ini_options].\n"
        "Reference: shared/src/shared/testing.py:5-21 and commit b9d39def62."
    )
