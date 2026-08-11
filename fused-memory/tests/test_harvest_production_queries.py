"""Tests for harvest_production_queries.py — the production query set (task 4004).

The harvester reads the live reconciliation write journal READ-ONLY and
samples the query shapes that actually reach `search` in production, so the
read transforms can be scored on real traffic rather than only on the
blind-authored E2 query set.

Every test here builds a SYNTHETIC SQLite DB in ``tmp_path`` with the real
``write_ops`` column shape.  No test in this file may open the live journal:
it is a ~10 GB file the running fused-memory server is writing to, and a test
that opened it would be measuring a moving target under xdist.

The script is loaded via importlib so it can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file must be free of network, Qdrant and OPENAI_API_KEY.
If a live test is ever added it carries its markers PER-TEST
(``@pytest.mark.integration`` + ``@pytest.mark.timeout(N)`` +
``qdrant_skipif()`` + an OPENAI_API_KEY skipif), never via a module-level
``pytestmark``: ``fused-memory/pyproject.toml`` sets
``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_bake_off_storage_shape.py:9-24``.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'harvest_production_queries.py'
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_module() -> types.ModuleType:
    """Load harvest_production_queries.py from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'harvest_production_queries'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    return _load_module()
