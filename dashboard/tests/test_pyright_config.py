"""Regression test that guards dashboard/pyproject.toml's [tool.pyright].extraPaths.

Dashboard imports ``shared.*`` (e.g. ``shared.async_sqlite_base`` in
``src/dashboard/app.py``) and requires ``"../shared/src"`` in pyright's
extraPaths so that pyright can resolve those types without depending solely on
the uv workspace venv install.  If the venv install breaks or pyright runs
without venv resolution, the missing extraPaths entry would silently cause
pyright to miss type errors against ``shared``.

Sibling configs orchestrator/pyproject.toml and fused-memory/pyproject.toml
both carry ``"../shared/src"`` in extraPaths; this test enforces the same
invariant for dashboard.

See task 1472 for full context.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

# Locate package root (dashboard/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent


def _load_pyright_config() -> dict[str, Any]:
    """Load and return the ``[tool.pyright]`` section of dashboard/pyproject.toml.

    Returns an empty dict when the section is absent so callers can assert
    truthiness or membership with clean messages.  The pyproject.toml path is
    resolved via ``_PACKAGE_ROOT``, mirroring the pattern in
    orchestrator/tests/test_pyright_gate_for_workflow_e2e.py.
    """
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    return config.get("tool", {}).get("pyright", {})


def test_dashboard_pyright_extrapaths_contains_shared_src() -> None:
    """[tool.pyright] extraPaths must include ``"../shared/src"``.

    Dashboard imports ``shared.*`` (e.g. ``shared.async_sqlite_base`` in
    ``src/dashboard/app.py`` and ``shared.cost_store`` referenced in tests).
    The ``"../shared/src"`` entry in extraPaths lets pyright resolve these
    imports directly from source, without depending solely on the uv workspace
    venv install.  Sibling configs orchestrator/pyproject.toml and
    fused-memory/pyproject.toml both carry this entry; dashboard must too.

    If this test fails, re-add ``"../shared/src"`` to
    ``[tool.pyright] extraPaths`` in dashboard/pyproject.toml.  See task 1472.
    """
    pyright_config = _load_pyright_config()
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    extra_paths = pyright_config.get("extraPaths", [])
    assert "../shared/src" in extra_paths, (
        f'"../shared/src" is not in [tool.pyright] extraPaths = {extra_paths!r} '
        f"in {toml_path}. "
        "Dashboard imports shared.* (e.g. shared.async_sqlite_base in "
        "src/dashboard/app.py) and relies on this entry so pyright resolves "
        "shared types without depending solely on the uv workspace venv install. "
        'Re-add "../shared/src" to [tool.pyright] extraPaths, between "src" and '
        '"../escalation/src", to match orchestrator/pyproject.toml and '
        "fused-memory/pyproject.toml. See task 1472."
    )
