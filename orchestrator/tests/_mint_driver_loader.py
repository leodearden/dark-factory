"""One by-path loader for ``scripts/mint_hard_v2_fixtures.py``.

``scripts/`` is not an importable package, so every test that needs the
fable-trial-v2 driver loads it by path through :func:`load_mint_driver`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts'
DRIVER_PATH = SCRIPTS_DIR / 'mint_hard_v2_fixtures.py'


def load_mint_driver(module_name: str = 'mint_hard_v2_fixtures') -> ModuleType:
    """Execute a fresh copy of the driver and register it as ``module_name``.

    The driver binds this checkout with ``import _checkout``, a sibling it
    resolves by bare name, so ``scripts/`` is on sys.path while it executes.
    Both that entry and the binder's own sys.path edits are undone afterwards,
    so the load changes import resolution for no other test in this process.
    """
    spec = importlib.util.spec_from_file_location(module_name, DRIVER_PATH)
    assert spec is not None and spec.loader is not None, f'no import spec for {DRIVER_PATH}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    saved_path = list(sys.path)
    binder_was_loaded = '_checkout' in sys.modules
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        if not binder_was_loaded:
            sys.modules.pop('_checkout', None)
    return module
