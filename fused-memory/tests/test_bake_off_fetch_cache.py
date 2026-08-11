"""Tests for the fetch-replay cache in bake_off_storage_shape.py (task 4004).

`fetch_arm` is the only part of the E2 bake-off that costs an embedder call
and a live Qdrant collection; everything downstream of it (`read_path`,
`measure_arm`, `rescore`, `build_report`, `render_markdown`) is already pure.
Dumping its return value and replaying it makes every read-side variant free
and — the point of this module — makes the metric code unit-testable against
REAL rankings rather than only hand-built ones.

The script is loaded via importlib so it can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily (via ``_mod()``)
rather than bound at import time.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file must be free of network, Qdrant and OPENAI_API_KEY
**except a live end-to-end test**, which carries its markers PER-TEST::

    @pytest.mark.integration
    @pytest.mark.timeout(600)
    @qdrant_skipif()
    @pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), ...)

Never via a module-level ``pytestmark``.  ``fused-memory/pyproject.toml``
sets ``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_bake_off_storage_shape.py:9-24``.

This file does NOT extend ``test_bake_off_storage_shape.py``: task 3560 is
in-progress and claims that module.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_module() -> types.ModuleType:
    """Load bake_off_storage_shape.py from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'bake_off_storage_shape'
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
