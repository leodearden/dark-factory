"""Tests for read_transform_selection.py — the read-transform arms (task 4004).

Three candidate read transforms over the ratified C write shape, scored
against the flat-read baseline: a PROMOTING topic pin, a TOPIC-KEYED grouped
read, and a topic-diversity cap.  All three are pure post-ranking transforms
over an already-fetched hit list, so every test here injects hand-built
ranked lists with exactly-known answers — which is what permits exact
assertions with no tolerances.

Both scripts are loaded via importlib so they can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily.

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

SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
SCRIPT_PATH = SCRIPTS_DIR / 'read_transform_selection.py'
BAKE_OFF_PATH = SCRIPTS_DIR / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a standalone script from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
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
    return _load_script(SCRIPT_PATH, 'read_transform_selection')


@functools.cache
def _bake_off() -> types.ModuleType:
    return _load_script(BAKE_OFF_PATH, 'bake_off_storage_shape')
