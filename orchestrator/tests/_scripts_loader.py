"""Load a ``scripts/`` module by path, containing the ``_checkout`` binder's side effects."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts'


def load_script(script_name: str, module_name: str) -> ModuleType:
    """Execute a fresh copy of ``scripts/<script_name>`` and register it as ``module_name``.

    Scripts bind this checkout with ``import _checkout``, a sibling resolved by
    bare name, so ``scripts/`` is on sys.path while the script executes. Both
    that entry and the binder's own sys.path edits are undone afterwards, and a
    script that fails to execute is not left registered.
    """
    script_path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None, f'no import spec for {script_path}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    saved_path = list(sys.path)
    binder_was_loaded = '_checkout' in sys.modules
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.path[:] = saved_path
        if not binder_was_loaded:
            sys.modules.pop('_checkout', None)
    return module
