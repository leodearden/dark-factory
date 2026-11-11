"""One by-path loader for ``scripts/mint_hard_v2_fixtures.py``.

``scripts/`` is not an importable package, so every test that needs the
fable-trial-v2 driver loads it by path through :func:`load_mint_driver`.
"""

from __future__ import annotations

from types import ModuleType

from _scripts_loader import SCRIPTS_DIR, load_script

DRIVER_PATH = SCRIPTS_DIR / 'mint_hard_v2_fixtures.py'


def load_mint_driver(module_name: str = 'mint_hard_v2_fixtures') -> ModuleType:
    """Load the driver as ``module_name``; :func:`_scripts_loader.load_script` explains the binding."""
    return load_script(DRIVER_PATH.name, module_name)
