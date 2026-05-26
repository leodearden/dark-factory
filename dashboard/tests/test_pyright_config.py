"""Regression test: ``"../shared/src"`` must appear in dashboard's ``[tool.pyright] extraPaths``.

Dashboard imports ``shared.*`` (e.g. ``shared.async_sqlite_base`` in
``src/dashboard/app.py``); this entry lets pyright resolve those types without
depending solely on the uv workspace venv install.  Mirrors the invariant
already enforced in orchestrator/pyproject.toml and fused-memory/pyproject.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

# Locate package root (dashboard/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent


def _load_pyright_config() -> dict[str, Any]:
    """Return the ``[tool.pyright]`` section of dashboard/pyproject.toml, or {} if absent."""
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    return config.get("tool", {}).get("pyright", {})


def test_dashboard_pyright_extrapaths_contains_shared_src() -> None:
    """``[tool.pyright] extraPaths`` must include ``"../shared/src"``."""
    pyright_config = _load_pyright_config()
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    extra_paths = pyright_config.get("extraPaths", [])
    assert "../shared/src" in extra_paths, (
        f'"../shared/src" missing from [tool.pyright] extraPaths = {extra_paths!r} '
        f"in {toml_path}. "
        "Add it so pyright can resolve shared.* imports (e.g. shared.async_sqlite_base "
        "in src/dashboard/app.py) without relying solely on the uv workspace venv install."
    )
