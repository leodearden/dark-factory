"""Load a sibling script that is not part of a package, by file path.

The file at *path* is registered in ``sys.modules`` under *mod_name* before its
body runs. An existing entry under that NAME is returned as-is, so every loader
of one key shares one module object. A body that raises leaves no entry behind.

Tests do not use this: they load scripts through
``fused-memory/tests/_fm_helpers.py::load_script_module``, which reuses by file
identity and refuses to shadow an entry it did not install.
``tests/test_script_loader_routing_guard.py`` enforces that.
"""

import importlib.util
import sys
import types
from pathlib import Path


def load_script(path: Path, mod_name: str) -> types.ModuleType:
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module
