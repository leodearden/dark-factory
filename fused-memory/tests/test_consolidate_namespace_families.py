"""Tests for scripts/consolidate_namespace_families.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_purge_knowlive_namespace.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'consolidate_namespace_families.py'


def _load_module() -> types.ModuleType:
    """Load consolidate_namespace_families.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'consolidate_namespace_families'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Helpers
# ===========================================================================

def _make_graph_mock(rows: list[list] | None = None) -> MagicMock:
    """Minimal stand-in for conftest.make_graph_mock, scoped to this test
    module so it is usable without the fixture (kept consistent with its
    shape)."""
    result = MagicMock()
    result.result_set = rows if rows is not None else []
    graph = MagicMock()
    graph.ro_query = AsyncMock(return_value=result)
    graph.query = AsyncMock(return_value=result)
    graph.delete = AsyncMock(return_value=None)
    return graph


def _make_qdrant_mock(points: list | None = None) -> AsyncMock:
    """AsyncMock stand-in for an AsyncQdrantClient exposing scroll/upsert/
    delete_collection -- the raw transport consolidate_namespace_families
    reaches via memory.mem0._get_async_qdrant()."""
    client = AsyncMock()
    client.scroll = AsyncMock(return_value=(points if points is not None else [], None))
    client.upsert = AsyncMock(return_value=None)
    client.delete_collection = AsyncMock(return_value=None)
    return client


def _make_point(
    point_id: str,
    payload: dict | None = None,
    vector: list[float] | None = None,
) -> MagicMock:
    """Build a Qdrant-scroll-shaped point stand-in (id/payload/vector)."""
    point = MagicMock()
    point.id = point_id
    point.payload = payload if payload is not None else {}
    point.vector = vector if vector is not None else [0.1, 0.2, 0.3]
    return point
