"""Where a relative fused-memory data path is anchored: the checkout this package was imported from, never the process CWD (task 5462)."""

import os
from pathlib import Path

#: config -> fused_memory -> src -> fused-memory -> checkout; holds only for an editable install.
CHECKOUT_ROOT: Path = Path(__file__).resolve().parents[4]


def anchor_at_checkout(path: str) -> str:
    if os.path.isabs(path):
        return path
    if not (CHECKOUT_ROOT / 'fused-memory' / 'pyproject.toml').is_file():
        raise RuntimeError(
            f'cannot anchor relative data path {path!r}: {str(CHECKOUT_ROOT)!r} holds no '
            'fused-memory/ checkout (a non-editable install?); configure it as an absolute path'
        )
    return str(CHECKOUT_ROOT / path)
